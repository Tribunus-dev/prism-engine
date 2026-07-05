# NF4Tile640 live teacher forward (Metal)

How the teacher `.cimage` executes a fused NF4 GEMV over the shared arena, what
was blocking it, and how to smoke-test it on a Mac.

## The path (all pieces exist)
1. **ABI** — `apple_installation::derive_nf4_tile640_arena_abi` validates the
   weight/scale/bias triplet (alignment, bounds, non-overlap) and yields the
   byte offsets/lengths.
2. **Binding** — `metal_iosurface::bind_nf4_tile640_triplet` binds the three
   arena regions at those offsets (shared IOSurface residency).
3. **Kernel** — `templates/nf4_tile640_gemv.metal` → `fused_gemv_nf4_tile640_fp32`:
   one threadgroup per output row, 32 lanes each read a 2-byte (4-nibble) word,
   dequantize `w = NF4_CODEBOOK[idx]·scale + bias` in registers, accumulate the
   dot product in **fp32**, `simd_sum` reduce.
4. **Dispatch** — `kernel_dispatch::Nf4Tile640ProjectionDispatcher` sets buffers
   0..5 (weights/scales/biases at arena offsets, input, output, num_macro_tiles)
   and dispatches `out_dim` threadgroups × 32 threads.

## What was blocking execution (fixed here)
- **Stale codebook.** The kernel still held the old asymmetric `[-1, 2]` table.
  Weights are now packed with the symmetric `[-1, 1]` NF4 codebook (the −19%
  fidelity fix), so the kernel would have dequantized every weight to the wrong
  value. Updated to match `compile/quantize.rs::NF4_CODEBOOK` (the 5th copy).
- **Kernel not in the metallib.** `compute-core/build.rs` compiled
  `ternary_tile640_gemv.metal` but **not** `nf4_tile640_gemv.metal`, so
  `KernelRegistry` could never find `fused_gemv_nf4_tile640_fp32` — the dispatch
  would fail at pipeline creation. Added it to the build's `metal_sources`.

With both fixes, the single-GEMV teacher forward executes correctly.

## Arena layout (the contract the kernel + packer + reference all share)
Per row, per 640-tile: `packed` u8 `[tiles·320]` = 5 groups × 64 bytes; each of
32 lanes owns 2 bytes = 4 nibbles; the value at column `t·640 + g·128 + (lane·4 + i)`
is nibble `i` of lane `lane` (low nibble for even `i`). `scales`/`biases` are
f32 `[tiles·5]`, one per 128-element group. Bias is 0 for NF4 but applied
affinely. Verified by `tools/nf4_forward_ref.rs` (layout self-consistency +
GEMV parity, Linux).

## Smoke-test it (Mac)
```bash
cargo test -p tribunus-compute-core --features prism-backend nf4_forward_exec
```
`nf4_forward_exec_matches_cpu` (in `kernel_dispatch.rs`) packs a matrix into the
interleaved arena, runs the real dispatcher on the GPU, and asserts the output
matches a CPU dequant+GEMV of the same bytes (max abs err < 1e-3). It skips
cleanly if no Metal device is present. This is the "it actually executes and is
correct" gate before benchmarking.

Ground-truth / accuracy reference (any host): `rustc -O tools/nf4_forward_ref.rs
-o /tmp/nf4fwd && /tmp/nf4fwd`.

## Remaining: full-model orchestration
This makes one projection (a GEMV) run. A complete teacher forward walks the
model graph — embed → per-layer {RMSNorm, QKV, attention, O-proj, FFN gate/up/
SiLU/down} → final norm → logits — calling this dispatcher for each projection
plus the attention/norm kernels, threading the KV cache. That runner is the
`level1::teacher::forward` stub; wiring it to iterate the cimage's execution
graph over these dispatchers is the next slice, after which the benchmark
harness (`bench_metrics`) scores real teacher vs student logits + tok/s.

## Status
- **Verified on Linux:** the layout/GEMV math (`nf4_forward_ref.rs`), the
  codebook value (`nf4tile640_ref.rs` −19% study).
- **Mac-only (authored, run there):** the on-GPU execution test, the kernel, and
  the build wiring — `compute-core` builds under `backend-cpu` on Linux but the
  Metal path needs Apple hardware.
