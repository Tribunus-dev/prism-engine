# bf16 anchor runner — architecture decision (no MLX, no nightly, no new GEMM engine)

Decision record for how the activation-anchored PTQ pipeline
(`JOINT_MTP_COMPILE.md`) gets its golden `H_bf16_out` activations. Verdict on
the "drop MLX and hand-roll a BLIS GEMM" proposal: **right conclusion for the
anchor runner, wrong reasons, and an implementation spec that fights both the
toolchain and the model's own dimensions.** What we do instead is smaller.

## The decision

1. **The anchor runner does not use MLX.** Correct call — a PTQ reference pass
   needs one deterministic layer forward, not a framework. But be precise
   about what this buys:
2. **The numerical contract is a std-only reference** —
   `tools/bf16_anchor_ref.rs`, landed and Linux-verified (7/7 self-tests,
   plain `rustc`, zero dependencies, CI-ready). It defines every op: bf16⇄f32,
   the weights-transposed GEMM, RMSNorm (both γ conventions), partial RoPE
   (both pair conventions), SwiGLU, causal GQA SDPA (scale flag), and the full
   block assembly with residuals.
3. **The Mac production pass binds that contract to the Accelerate lane that
   already exists in-tree.** `backend/accelerate/ops.rs` already wraps
   `cblas_sgemm` (lines ~631/702) and vDSP primitives (`vDSP_vmul`/`vsmul`/
   `sve` — an RMSNorm pipeline, ~753+). The "write a custom blocked GEMM"
   proposal overlooked that the repo links CBLAS on macOS today. Per-layer
   flow: expand the layer's bf16 weights to f32 once (~450 MB transient for a
   torso layer), `cblas_sgemm` the projections, vDSP the norms, reference-Rust
   the RoPE/SwiGLU/softmax glue. AMX-backed sgemm runs ~1–2 TFLOP/s — the
   full 48-layer × 32×1024-token calibration pass is **~10–20 minutes**, vs
   hours for any pure-CPU Rust GEMM (the calibration pass is ~705 TFLOP:
   2 · 224 M params · 32 768 tokens · 48 layers).
4. **No nightly Rust.** The proposal's `core::simd` microkernel requires the
   nightly toolchain for the whole workspace; CI pins stable. The reference
   doesn't need SIMD (it's a correctness oracle running small shapes on
   Linux), and the production path gets its FLOPs from Accelerate. If a
   portable-CPU fast path is ever wanted, stable options exist
   (`matrixmultiply` is already in the dependency universe; `wide`; arch
   intrinsics behind cfg) — decide then, not now.

## Corrections to the proposal (each checked against the tree)

### "Dropping mlx-rs instantly resolves the kd_gate blocker" — false
The `-lamdhip64` / `-lze_loader` link failures come from
`device/probes/rocm.rs:41` and `device/probes/level_zero.rs:90` —
`#[link]` attributes in the **heterogeneous-GPU probes**, nothing to do with
MLX. Ripping out MLX would leave the link failure byte-for-byte identical.
The actual fix is feature-gating those two probes (parked in the working
tree). Separately: `mlx-rs`/`mlx-sys` are **unconditional dependencies**
imported by 8+ modules (`ane/*`, `attention.rs`, `audio/*`, `autopsy`,
`backend/*`) — excising MLX from the crate is a real project with its own
justification (the fork-checkout build fragility is genuine), but it is not
this task, and it is not a CI fix.

### "Lock N_c to the Tile640 boundary" — contradicts the model's own shapes
The proposal's `assert!(n % 640 == 0)` kills three of the model's seven
projection shapes: **Q out = 4096 (6.4×640), K/V out = 2048 (3.2×640),
vocab = 262144 (409.6×640)**. Only hidden (3840 = 6×640) and FFN
(15360 = 24×640) are clean multiples — the same partial-tile reality this
branch just finished guarding in the NF4 kernel. Cache blocking is a CPU-cache
concern, not a GPU-constant concern; "mirroring the engine" at the GEMM-
blocking level buys nothing numerically. What Tile640 alignment *actually*
means for the anchor: slice the **output** columns in 640-wide windows when
computing per-tile `act_err` for the auto-tuner — any GEMM provides that.

### Hardcoding megakernel quirks into the anchor — inverts its purpose
The proposal says to implement "plain-gamma without the 1+γ fold." The anchor
must implement the **true checkpoint semantics** — it is the instrument that
adjudicates the three megakernel flags from `PER_OP_FORWARD_PLAN.md`
(plain-γ vs (1+γ); shared pre-attn/pre-FFN norm weights; missing 1/√d), plus
a fourth surfaced here: **RoPE pair convention** (the megakernel rotates
adjacent pairs; HF-lineage checkpoints use split halves). The reference
therefore exposes all four as flags (`gamma_delta`, `share_norm_weights`,
`attn_scale`, `rope_conv`) and machine-checks the fold identity
`plain(1+γ) ≡ gemma(γ)` — if ingest folds correctly, both paths agree; if
not, the anchor-vs-megakernel diff *is* the bug report.

### Threading — the embedded question, answered
Keep the anchor GEMM **isolated and deterministic**; never share the ECS
schedule or any runtime pool with an AOT compile step. The reference ships a
`std::thread::scope` parallel GEMM over disjoint output-column strips that is
**bitwise-equal to the serial path** (asserted in its tests) — determinism is
a gate requirement (`act_err` thresholds must not flap run-to-run). `rayon` is
already in Cargo.toml if the production driver wants work-stealing across
*layers/tensors*, but within one GEMM, deterministic partitioning wins.

## What's landed vs. next

- **Landed (Linux-verified)**: `tools/bf16_anchor_ref.rs` — the contract +
  battery: GEMM vs f64 oracle on non-aligned shapes, bitwise parallel parity,
  bf16 RNE bounds, the γ-fold identity, RoPE invariants (pos-0 identity, norm
  preservation, untouched tail, convention divergence), SwiGLU hand value,
  GQA routing/causal/scale checks, zero-weight ⇒ identity, determinism, and a
  convention-flag sweep proving the flags produce measurably different
  anchors. Add to the CI `verified-tools` job:
  `rustc -O tools/bf16_anchor_ref.rs -o /tmp/bf16ref && /tmp/bf16ref`.
- **Next (Mac)**: `Bf16AnchorRunner` in `compilation/` binding the contract to
  `accelerate_ffi` per §The decision(3), implementing the `LayerForward`
  trait from `JOINT_MTP_COMPILE.md` §2.5 — parity-gated against this
  reference on small shapes, then against the real checkpoint (where the
  convention flags get their verdicts).
- **Then**: the streaming compile driver consumes it as the `H_bf16_out`
  source with the `(H_student_in → H_bf16_out)` pairing.
