# Per-op teacher forward — kernel-by-kernel plan

Goal: a **per-op Gemma 4 forward** whose layer boundaries are real buffer
boundaries, so the teacher exposes **per-layer activations for layer-wise
distillation** (activation capture + student block-swap + per-layer KD), with
every kernel gated by a Mac device-parity test.

This supersedes `KERNEL_AUDIT.md`'s "stay on the megakernel" recommendation —
that advice was conditional on *not* needing per-layer activations. Two facts
found while planning make the per-op path more valuable than the audit assumed:

1. **`decode_per_layer.metal` is 100% stubs.** Every per-layer entry point
   (`decode_layer_swa`, `decode_layer_full`, the fused pairs/triples/quads,
   `fused_mtp_roundtrip`, …) has an identity body (`hidden_out[tid] =
   hidden_in[tid]`). The buffer ABIs sketched there are a usable starting
   vocabulary, but **nothing** of the per-layer path exists.
2. **Neither megakernel variant executes NF4Tile640 weights.** `gemma4_full.metal`
   runs ternary tile-GEMV; `gemma4_full_int4.metal` runs `compute_fused_ternary_gemv`
   over a fused-ternary layout. The fused templates (`fused_rmsnorm_qkv`,
   `fused_o_proj_residual`, `fused_gate_up_activation`, `fused_down_proj_residual`)
   all take `PackedTernaryPage640` — **student-format** weights. The only kernel
   that executes NF4Tile640 natively is `fused_gemv_nf4_tile640_fp32` (ours,
   parity-tested). So the per-op path is not just instrumentation: **it is the
   only native execution path for the NF4 teacher.**

## Ground truth: what the megakernel actually computes

Read out of `gemma4_full.metal` (line refs from the current branch). The per-op
path must replicate THIS, not textbook Gemma:

| Stage | What it does | Where |
|---|---|---|
| Embed | ternary-packed embed table, **cluster-reordered** (`embed_clust` + `cluster_map`), fp16 block scales | ~409–441 |
| Input RMSNorm | `x·rsqrt(mean(x²)+1e-6)·w[i]` — **plain γ multiply** (no `1+γ`), fp16 weights, fp32 accum | 191–211, 452–455 |
| KV decompress | per layer: ternary KV nibbles (base-3, `KV_BLOCK=256`, 13 u32/block) + fp16 scales → fp16 scratch `[MAX_CTX, 8, h_dim]` | 473–540 |
| K/V proj + RoPE + scatter | per KV-group: K,V GEMV; **partial RoPE** (`rope_dim=64` of `h_dim`, θ=1e6, adjacent-pair rotation); write pos `kv_cache_pos` | 543–579 |
| Q proj + RoPE + SDPA | 2 Q heads per KV group (GQA 16:8); raw QK dots (**no visible 1/√d**), two-pass softmax (fp32 max/sum), PV accumulate ×`sigmoid(head_gates[qh])` | 582–657 |
| KV re-pack | current position's K/V → ternary nibbles + absmax scales per 256-block | 662–760 |
| O-proj + residual | GEMV, `h_buf += ` | 764–775 |
| Post-attn RMSNorm | **reuses `in_norm_w` — same weights as input norm** | 777–780 |
| FFN | gate & up GEMVs (15360), **SwiGLU** (`silu(g)·u`), down GEMV, `h_buf += ` | 782–837 |
| Attention entropy | per-position `−q·log₂q` accumulated → `entropy_map` (KV-eviction instrumentation) | 651–655, 839–850 |
| Logits | **approximate**: h·centroids (256), pick best cluster, unpack + GEMV **only that cluster's vocab range** | 853–889+ |

Constants: `HIDDEN=3840, LAYERS=48, Q_HEADS=16, KV_HEADS=8, HEAD_DIM=256`,
**`GLOBAL_HEAD_DIM=512` on every 6th layer** (`(layer+1)%6==0` → "shared"),
`FFN=15360, VOCAB=262144, MAX_CTX=2048, NUM_SINKS=4` (StreamingLLM sinks +
cyclic FIFO eviction for `pos ≥ MAX_CTX`).

### ⚠ Three behaviors to verify-or-flag (they transfer into the teacher)

1. **Norm-weight reuse**: pre-FFN norm uses the *same* `norms + layer·HIDDEN`
   weights as the pre-attention norm. Stock Gemma has distinct
   pre/post norms. Either the ingest folds them or this is a fidelity bug.
   *Plan stance: replicate exactly for parity, file the question separately.*
2. **Plain-γ RMSNorm**: stock Gemma applies `(1+γ)`. If ingest stores `1+γ`
   into the norms aux buffer, this is correct; if it stores raw γ, every norm
   is wrong. *Verify at ingest during Stage 2; CPU refs use the stored-weight
   convention (plain multiply) either way.*
3. **Approximate logits**: centroid-scout computes real logits only inside one
   cluster of the vocab. Fine for greedy decode; **not fine as a KD teacher
   signal** (soft targets over the full vocab are the point of distillation).
   The per-op logits stage therefore does a full-vocab GEMV by default and
   keeps scout as an opt-in fast mode.

## Two lanes, one set of non-projection kernels

- **Teacher lane (NF4Tile640)** — all 7 projections/layer + logits run through
  the existing, parity-tested `Nf4Tile640ProjectionDispatcher`
  (`fused_gemv_nf4_tile640_fp32`, fp32 accumulate, partial-in_dim guarded).
- **Student lane (ternary v7)** — projections run through
  `ternary_tile640_gemv` (in metallib) and, later, the fused
  `PackedTernaryPage640` templates. Needed for block-swap distillation
  (student layer k inside a teacher forward).
- **Shared** — everything that is not a weight GEMV: norm, RoPE, SDPA, KV
  pack/unpack, SwiGLU, residual, embed, logits assembly. Authored once below.

## Test harness conventions (applies to every kernel)

- Pattern: `#[cfg(all(test, feature = "metal-dispatch"))]` module in
  `kernel_dispatch.rs` (or a new `per_op_tests.rs`), exactly like
  `nf4_forward_exec_matches_cpu`: skip cleanly without a Metal device,
  synthesize inputs, run the real dispatcher, compare against an in-test CPU
  reference. Naming: `<kernel>_exec_matches_cpu`.
- **Tolerance ladder** (fp16 activations are the precision floor):
  - exact-index ops (embed gather, residual, RoPE): max-abs ≤ 2·fp16-ulp of range
  - reductions (rmsnorm, softmax/SDPA, GEMV): rel-L2 ≤ 1e-3 vs f64 CPU ref
  - megakernel-tap comparisons: rel-L2 ≤ 5e-3 initial, tightened after first
    measurement (megakernel's threadgroup-half intermediates are the limiter)
- **Real-weight tests** are `#[ignore]`-style env-gated: they run only when
  `TRIBUNUS_TEST_CIMAGE=/path/to/model.cimage` is set (your Mac), skip in CI.
- Every kernel lands in `compute-core/build.rs::metal_sources` in the same PR
  as its dispatcher — plus the Stage-1 registry test that would have caught
  today's name mismatches.

---

## Stage 0 — Megakernel activation taps (the oracle) — *before any kernel*

Add per-layer activation capture to the **megakernel itself**. This is ~150 LoC
and it is what makes every later parity test possible on real weights.

- `gemma4_full.metal` (+ `_int4` variant): new optional buffer
  `device half* layer_taps [[buffer(N)]]` + function-constant `TAPS_ENABLED`.
  When enabled, after the attention residual (post step 7) write `h_buf` to
  `taps[(2·layer)·HIDDEN …]`, and after the FFN residual (post step 10) to
  `taps[(2·layer+1)·HIDDEN …]`; plus slot 0 = post-embed, slot `2L+1` = final
  pre-logits hidden. Total `(2·48+2)·3840` halfs ≈ **753 KB** — negligible.
  Runtime-compiled via xcrun, so no build.rs change.
- Rust: `Orchestrator::decode_token_logits_with_taps(token) ->
  (logits, Taps { embed, post_attn[48], post_layer[48], final_hidden })`.
- **Tests**: (a) self-consistency — `taps.post_layer[47]` → final norm → logits
  must reproduce the returned logits (validates tap placement); (b) determinism
  — two identical runs produce bitwise-equal taps; (c) taps-off run is
  bitwise-identical to today's decode (function constant truly gates it).
- **Bonus**: this alone already gives you *megakernel-teacher* activation
  capture for distillation while the per-op build-out proceeds — mid-layer
  (post-attn) taps mean both the attention block and the FFN block are
  independently testable on real weights for every kernel below.

## Stage 1 — Registry hygiene (cheap, de-risks everything)

From `KERNEL_AUDIT.md`'s "cheap wiring", now with a gate:

- Add to `build.rs metal_sources`: `fused_rmsnorm_qkv.metal`,
  `fused_o_proj_residual.metal`, `fused_gate_up_activation.metal`,
  `fused_down_proj_residual.metal`, `kv_mixed.metal` (compile errors surface
  now, not later; these become the student lane).
- Fix the probe dispatcher name mismatches (`attention_probe`, `error_partial`,
  `candidate_score`, `pack_verify`).
- **Test** `metallib_contains_all_dispatcher_entry_points`: enumerate every
  dispatcher's `kernel_name` and assert `get_or_create` succeeds for each —
  the class of failure the audit found becomes a test failure, permanently.

## Stage 2 — Small exact-math kernels (norm, RoPE, elementwise)

| # | Kernel | ABI (buffers) | Grid | Parity test |
|---|---|---|---|---|
| K1 | `rmsnorm_f32` | in f32/f16[d], γ f16[d], out[d], `{d, eps}` | 1 threadgroup, fp32 tree-reduce | f64 CPU ref; dims {3840, 640, 100}; **must reproduce `fast_rmsnorm` semantics exactly** (plain γ, eps 1e-6, half output rounding). Real-weight variant: apply to a tapped layer input, compare vs f64 CPU on the same vector |
| K2 | `swiglu_mul` | g[n], u[n], out[n], n | 1D grid | `silu(g)·u` vs f64 ref, 1e-3 rel; n = 15360 and a non-multiple-of-tg size |
| K3 | `residual_add` | a[n], b[n], out[n], n | 1D grid | exact vs CPU (fp16 rounding only) |
| K4 | `rope_apply` | vec f16[h_dim], `{h_dim, pos, rope_dim=64, theta=1e6}` | 1 threadgroup | f64 ref of the megakernel's adjacent-pair rotation (lines 215–230); positions {0, 1, 2047, 5000 (post-FIFO)}, h_dim {256, 512}; asserts dims ≥ rope_dim are untouched |

Dispatchers: one small `ElementwiseDispatcher` + `RmsNormDispatcher` +
`RopeDispatcher` (`set_bytes` for params — these don't need the ProjectionParams
struct). ~400 LoC total incl. tests. **Verify the `(1+γ)` question at ingest
while writing K1's test** (read what the cimage compiler stores into `norms`).

## Stage 3 — Both ends: embed + logits

| # | Kernel | Notes | Parity test |
|---|---|---|---|
| K5 | `embed_gather` | Two variants driven by the cimage manifest: (a) dense f16/bf16 row gather ×scale; (b) ternary-clustered decode — extract megakernel Stage 0 (cluster_map lookup + tile decode + `embed_scales`) | CPU ref decodes the same bytes; env-gated real-cimage test: K5 output == megakernel `taps.embed` for 32 random tokens, max-abs ≤ fp16 ulp |
| K6 | logits | **No new kernel** — final `rmsnorm_f32` (K1) + full-vocab projection GEMV via the existing NF4/ternary dispatcher (`out_dim=262144`, 410 partial-guarded tiles… the in_dim guard from the partial-tile fix is load-bearing here). Optional later: `centroid_scout` extraction as a fast mode | Env-gated: argmax must equal megakernel's decode for N=64 tokens; full-vocab logits vs megakernel's **within the scout cluster only** (elsewhere megakernel is approximate — document, don't "fix" the test to match) |

This stage lands early because both are cheap, and K6 immediately gives the
**distillation-grade full-vocab teacher logits** the megakernel cannot produce.

## Stage 4 — FFN block assembly (first full block on real weights)

No new GEMV kernels — the block is `K1 → NF4-GEMV(gate) → NF4-GEMV(up) → K2 →
NF4-GEMV(down) → K3`. Work is Rust-side: a `FfnBlockRunner` that binds the
layer's gate/up/down arena offsets (from the ABI derivation) and encodes the 6
dispatches in one command buffer.

- **Parity test (the first big one)**: `ffn_block_matches_taps` — env-gated;
  for layers {0, 5 (pre-shared), 24, 47}: run the block on `taps.post_attn[k]`,
  assert output ≈ `taps.post_layer[k]`, rel-L2 ≤ 5e-3. This validates K1/K2/K3,
  the NF4 dispatcher offsets, and the norm-weight-reuse assumption in one shot.
- Synthetic CPU test: full block vs f64 reference on random weights packed with
  the CPU packer (in_dim 3840 exact + one partial case).

Student lane (deferrable): wire `FfnGateUpDispatcher` over `fused_gate_up_activation`
+ `fused_down_proj_residual` with the same block-level test.

## Stage 5 — KV cache trio (mirrors megakernel structure 1:1)

The megakernel's attention *internally* decompresses ternary KV to an fp16
scratch, attends over fp16, then re-packs. Splitting along the same seams keeps
each kernel small and each parity test tight:

| # | Kernel | Extract from | Parity test |
|---|---|---|---|
| K7 | `kv_ternary_decompress` | lines 473–540 (base-3 nibbles + scales → fp16 scratch `[ctx, 8, h_dim]`) | CPU ref over synthetic nibbles; h_dim {256, 512}; partial last block (`h_dim % 256`-style clamps, lines 497–510) |
| K8 | `kv_ternary_pack` | lines 662–760 (absmax per 256-block → round-to-nearest ternary, clamp ±1) | pack→K7 roundtrip == CPU quantize simulation, bit-exact on digits + scales |
| — | `kv_cache_pos` map | lines 467–471 (sinks + cyclic FIFO) | pure Rust fn + unit test (positions 0, 3, 4, 2047, 2048, 4095 → sink/FIFO slots) |

**Clean-teacher mode**: a flag that skips K7/K8 and keeps the cache fp16.
For distillation you likely *want* the un-noised teacher; for megakernel parity
you want ternary KV. Same SDPA kernel (K9) reads the fp16 scratch either way —
the mode only changes what fills it. Both modes ship; parity gates run in
ternary mode.

## Stage 6 — SDPA decode (the hard one, done last-with-most-oracle)

| # | Kernel | Spec |
|---|---|---|
| K9 | `sdpa_decode_gqa` | q f16[16·h_dim] (post-RoPE), K/V fp16 scratch, gates f16[16], out f16[3840], `{num_cached, h_dim, kv_heads=8, q_per_kv=2}`. Raw QK dots (**replicate the no-1/√d behavior exactly** — if it's folded into packed Q at quantize time, matching the megakernel is still the correct parity target), two-pass softmax with fp32 max/sum reductions, PV accumulate, ×`sigmoid(gate[qh])`. Grid: one threadgroup per Q head (16 tgs — parallel across heads where the megakernel is serial; math identical). |

- **Synthetic parity**: f64 CPU ref; sweep num_cached {1, 4 (sinks only), 37,
  640, 2048}, h_dim {256, 512}, random gates; rel-L2 ≤ 1e-3.
- **Real-weight parity (attention block)**: `attn_block_matches_taps` —
  env-gated; for layers {0, 5, 24, 47}: K1(taps.post_layer[k−1]) → K/V GEMVs +
  K4 → scatter → Q GEMVs + K4 → K9 → O-proj GEMV → K3, assert ≈
  `taps.post_attn[k]`, rel-L2 ≤ 5e-3. Run in ternary-KV mode with the cache
  pre-populated by megakernel-driven decode so `num_cached > 1`.
- Deferred: `attn_entropy` (KV-eviction instrumentation) — only needed if the
  per-op path must also drive eviction research.

## Stage 7 — Assembly: `teacher::forward_per_op` + capture API

- `PerOpLayerRunner` (Rust): encodes one layer = K1 → per-KV-group {K,V GEMV,
  K4, scatter} → {Q GEMV, K4} → K9 → O GEMV → K3 → K1(same weights) → FFN block
  (Stage 4). All projections via `Nf4Tile640ProjectionDispatcher` with
  per-layer arena offsets; `h_dim` switches 256/512 on shared layers.
- `Gemma4TeacherPerOp::forward_with_activations(token) -> (Vec<f32> logits,
  Vec<LayerActivation>)` — activations are the inter-layer buffers you already
  have; **capture is free** (shared-mode buffer reads, no extra kernels).
- **Parity gates (the payoff, all env-gated on the real cimage)**:
  1. `per_op_layer0_matches_tap` — layer 0 output vs `taps.post_layer[0]`.
  2. `per_op_drift_bounded` — per-layer rel-L2 vs taps across all 48; assert
     the curve stays ≤ tol (expect slow fp16-ordering drift, not blowup).
  3. `per_op_logits_match` — argmax parity with megakernel over 64 greedy
     steps + top-8 overlap ≥ 7/8 (full-vocab mode is *allowed* to beat scout).
  4. Determinism: two runs bitwise-equal.
- Then the distillation hooks are one small module: capture teacher acts →
  run student block k on `teacher_act[k−1]` → per-layer KD / `block_accept`
  (ties directly into `distill_core`). Block-swap = swap one layer's runner
  from NF4 dispatchers to ternary dispatchers.

---

## Order & effort summary

| Stage | Deliverable | New/edited LoC (est.) | Risk | Depends on |
|---|---|---|---|---|
| 0 | Megakernel taps + taps API + 3 tests | ~150 Metal+Rust | Low | — |
| 1 | build.rs wiring + registry test | ~60 | Low | — |
| 2 | K1–K4 + dispatchers + CPU-parity tests | ~400 | Low | 0 (real-weight variants) |
| 3 | K5 embed, K6 logits assembly + tests | ~250 | Low-Med | 0, 2 |
| 4 | FFN block runner + block parity | ~200 | Med (offsets) | 0, 2 |
| 5 | K7, K8, pos-map + roundtrip tests | ~350 | Med | 0 |
| 6 | K9 SDPA + block parity | ~400 | **High** | 2, 4, 5 |
| 7 | Per-op runner + capture API + 4 gates | ~450 | Med | all |

Rationale for the order: the oracle first (0), the failure-class killer second
(1), then strictly smallest-risk-first — every stage's parity test leans on the
taps, and SDPA (the only genuinely hard kernel) arrives when norm/RoPE/GEMV/KV
are all independently proven, so an SDPA parity failure can only *be* SDPA.

## Honest constraints

- **Per-op will be slower than the megakernel** — ~48 layers × ~12 dispatches
  vs 1 persistent kernel; expect several ms/token of encoder overhead. It is
  the instrumentation/distillation path, not the serving path. (It is also the
  batched-verification substrate the spec-decode discussion needs — per-op
  stages generalize to multi-token far more easily than the persistent kernel.)
- **All Metal code here is authored-for-Mac**: this sandbox verifies the CPU
  references and Rust-side plumbing (Linux, `backend-cpu`); every kernel's
  device test needs your machine. Suggested cadence: land per stage, run that
  stage's tests on the Mac, then start the next — stages are sized so a parity
  failure implicates one kernel.
- The three ⚠ findings above (norm reuse, plain-γ, approximate logits) should
  each get a yes/no verdict during Stages 2–3; if any is a real bug, it's a
  *separate* fix applied to megakernel + per-op together, so parity stays
  meaningful throughout.
