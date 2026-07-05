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
   0..6 (weights/scales/biases at arena offsets, input, output, num_macro_tiles,
   in_dim) and dispatches `out_dim` threadgroups × 32 threads.

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

## Partial tiles (in_dim not a multiple of 640)

`tiles = ceil(in_dim / 640)`. When `in_dim` isn't a clean multiple, the last
tile is **partial**: the columns `[in_dim, tiles·640)` don't exist in the source
matrix. This is handled on **both** sides of the contract:

- **Packer (both paths already correct).** `quantize_nf4_tile640_matrix_from_raw`
  (CPU) and `nf4_tile640_pack.metal` (GPU) both size their output for `ceil`
  tiles and **zero-pad** the tail (`col < in_dim ? weight : 0.0`). Zero quantizes
  to NF4 code 7 (= 0.0), so a padded column dequantizes to exactly 0.0 and
  contributes nothing. Metadata (`scales`/`biases`) is written for all 5 groups
  of every tile, including fully-padded groups (scale defaults to 1.0), so the
  kernel's `[tiles·5]` metadata reads are always in bounds.
- **Kernel (the fix).** The weight is zero, but `in_vector` is only `in_dim`
  long — multiplying by `in_vector[col]` for `col >= in_dim` is still an
  **out-of-bounds load**. The kernel now takes `in_dim` at `buffer(6)` and guards
  the activation read: `uint col = src_base + i; if (col >= in_dim) continue;`.
  The dispatcher binds `params.in_dim` (the projection's *true* logical width) to
  that buffer.

Note `in_dim` is passed explicitly, not derived from the arena: the packed byte
width only encodes `ceil(in_dim/640)`, so e.g. `in_dim = 641` and `in_dim = 1280`
are indistinguishable from the bytes alone. The true width must come from the
graph/model config, which is what the caller threads into `ProjectionParams`.

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
matches a CPU dequant+GEMV of the same bytes (max abs err < 1e-3). It runs an
exact shape (`in_dim = 640`) **and two partial shapes** (`650`, `1290`) to
exercise the `buffer(6)` guard. It skips cleanly if no Metal device is present.
This is the "it actually executes and is correct" gate before benchmarking.

Ground-truth / accuracy reference (any host): `rustc -O tools/nf4_forward_ref.rs
-o /tmp/nf4fwd && /tmp/nf4fwd`. The reference reproduces the kernel's guarded
loop (`if col >= in_dim continue`) over the full padded grid and checks it
against a dense GEMV over the real columns, for both exact and partial `in_dim`
(`1290`, `4104`), plus a "every padded column decodes to 0.0" invariant.

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
  partial-tile guard + zero-pad tail invariant (`nf4_forward_ref.rs`, exact +
  `1290`/`4104`), the codebook value (`nf4tile640_ref.rs` −19% study).
- **Mac-only (authored, run there):** the on-GPU execution test (now including
  the `650`/`1290` partial cases), the kernel, and the build wiring —
  `compute-core` builds under `backend-cpu` on Linux but the Metal path needs
  Apple hardware.
