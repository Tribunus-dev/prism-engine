# Teacher-forward kernel audit

Which kernels a full Gemma 4 forward needs, and their real status. There are
**two forward paths** and they must not be confused:

- **Megakernel (works, complete):** `megakernel/shaders/gemma4_full.metal` →
  `gemma4_full_decode_persistent`. Compiled at **runtime** from `include_str!`
  (`megakernel/kernels.rs::compile_kernel` → xcrun), it implements the ENTIRE
  forward internally — embed → 48 layers {RMSNorm, QKV, attention, O-proj, FFN
  gate/up/SiLU/down} → final norm → logits, with the KV cache threaded. This is
  what `Orchestrator::decode_token` / `Gemma4Teacher` run today.
- **Per-op dispatchers (partial):** `kernel_dispatch.rs` dispatchers load from
  the **build-time** `palettized_kernels.metallib` (`compute-core/build.rs`),
  which contains only **6** kernels. Most per-op kernels are NOT in it.

## The build-time metallib contains only these 6
`palettized_gemv`, `palettized_gemv_swiglu`, `palettized_gemm`, `fused_gate_up`,
`ternary_tile640_gemv`, `fused_gemv_nf4_tile640_fp32`.

Everything a dispatcher requests by name must be in this metallib or
`get_or_create` fails at runtime.

## Per-stage status

| Forward stage | Standalone kernel source | In metallib? | Dispatcher | Status |
|---|---|---|---|---|
| Embedding lookup | `embedding_assembly` (vision_projection.metal) | ❌ | ❌ | **author/wire** (also inside megakernel) |
| RMSNorm (pure) | none — only `fused_rmsnorm_qkv` + `rmsnorm_residual_probe` | ❌ | probe only | **author** if a standalone norm is needed |
| QKV projection | `fused_rmsnorm_qkv` (fused w/ norm) ✅ src | ❌ | `FusedRmsnormQkvDispatcher` | **wire**: add to build.rs (name already matches) |
| Attention / SDPA | `kv_mixed_attn` (kv_mixed.metal) ✅ src; `attention_score_probe` (probe) | ❌ | probe only (`attention_probe`, name mismatch) | **author/verify**: real SDPA only exists inside the megakernel/`decode_per_layer` |
| O-projection | `fused_o_proj_residual` ✅ src | ❌ | `FusedOProjResidualDispatcher` | **wire**: add to build.rs (name matches) |
| FFN gate+up (+SiLU) | `fused_gate_up` ✅ (in metallib), `fused_gate_up_activation` ✅ src, `palettized_gemv_swiglu` ✅ (in metallib) | ✅ (2 of 3) | ❌ none | **wire a dispatcher** (kernels present) |
| FFN down-proj | `fused_down_proj_residual` ✅ src | ❌ | ❌ | **wire**: add to build.rs + author dispatcher |
| Final RMSNorm | same as RMSNorm | ❌ | ❌ | **author** if standalone |
| Logits / lm_head | projection kernels (nf4/ternary/palettized GEMV) reusable ✅ | ✅ | projection dispatchers ✅ | **reuse** a projection dispatcher |
| Weight GEMV (proj) | `ternary_tile640_gemv`, `fused_gemv_nf4_tile640_fp32`, `palettized_gemv` | ✅ | ✅ (3 projection dispatchers) | **done** |

## Honest summary

- **A complete teacher forward already exists** — the runtime-compiled megakernel. Use it (that's what `Gemma4Teacher` does).
- **The per-op dispatch path is not currently functional beyond the 3 projection kernels.** Two fused block kernels (`fused_rmsnorm_qkv`, `fused_o_proj_residual`) have matching-name source + dispatchers but are **not compiled into the metallib** — they would fail `get_or_create` today. FFN kernels are partly present (`fused_gate_up`, `palettized_gemv_swiglu` in the metallib) but have **no dispatcher**. `fused_down_proj_residual`, `fused_gate_up_activation`, `kv_mixed_attn` are source-only.
- **Genuinely missing as standalone loadable kernels:** a pure RMSNorm, a clean loadable attention/SDPA (the working attention is fused inside the megakernel/`decode_per_layer`), embedding lookup, and the final logit projection as a discrete step. The probe dispatchers (`attention_probe`, `error_partial`, `candidate_score`, `pack_verify`) also carry **names that don't match** their `.metal` entry points, so they'd fail even if compiled.

## To make a per-op forward real (in order)

1. **Wire what already exists** (cheap): add `fused_rmsnorm_qkv.metal`, `fused_o_proj_residual.metal`, `fused_gate_up_activation.metal`, `fused_down_proj_residual.metal`, `kv_mixed.metal` to `compute-core/build.rs` `metal_sources`, and add dispatchers for the FFN kernels. Fix the probe dispatcher name mismatches.
2. **Author the genuinely missing kernels**: standalone attention/SDPA (with RoPE + GQA + causal mask + KV read), embedding gather, final logits projection, and a pure RMSNorm — unless you extract them from the megakernel source.
3. **Parity-test each kernel against the megakernel** on a real Mac (per-stage activation diff), since the megakernel is the ground truth.
4. Only then assemble `teacher::forward` as a per-op walk.

**Recommendation:** unless you need per-layer activation capture for layer-wise
distillation, stay on the megakernel (`Gemma4Teacher`) — it's complete and
proven. The per-op path is a multi-kernel authoring project (~4–6 new/verified
kernels + dispatchers), each needing Mac compile + parity, not a quick wire-up.
