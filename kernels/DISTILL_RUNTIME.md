# NF4 teacher → ternary student distillation runtime

How the pieces wire into the block-by-block loop in `server::distill_worker`
(`Level1/2/3` schedulers already exist). This slice adds the shared-arena
contract and the numeric core; the live forward/backward passes run on the Mac.

## Data flow (per block)

```
            ┌─────────────────────── NF4Tile640 shared arena ───────────────────────┐
            │  packed_nf4_weights (u8)   scales (f32)   biases (f32)   [one residency] │
            └───────────────▲───────────────────────────────────▲────────────────────┘
                            │ derive_nf4_tile640_arena_abi()      │  (aligned, in-bounds,
                            │  proves offsets/lengths BEFORE bind  │   non-overlapping)
              ┌─────────────┴───────────┐             ┌───────────┴─────────────┐
              │ Metal lane (today)      │             │ stateless ANE lane      │
              │ fused NF4Tile640 GEMV   │             │ .mlmodelc (CoreAI: fall)│
              └─────────────┬───────────┘             └───────────┬─────────────┘
                            │  teacher logits / hidden activations │
                            └──────────────┬───────────────────────┘
                                           ▼
                       distill_core: kd_divergence · top1_agreement
                                           ▼
                 student forward (ternary) via ternary_pipeline::quantize_tensor
                                           ▼
                       distill_core::block_accept(teacher_act, student_act)
                                  ├── accepted → write ternary block to student .cimage
                                  └── rejected → re-distill w/ richer QuantConfig
                                                 (↑ outlier_frac, ↓ τ, enable AWQ)
```

## What this slice adds
- **`apple_installation::derive_nf4_tile640_arena_abi`** now proves the teacher's
  three slots are naturally aligned, in-bounds, and non-overlapping — so the
  Metal and (future) ANE lanes bind the *same* resident teacher bytes rather
  than shapes that only happened to line up. Both lanes read the arena; the
  derivation is the single gate before either binds.
- **`compilation::distill_core`** (std-only, unit-tested): `kd_divergence`
  (T²·KL, the teacher→student loss), `top1_agreement`, and `block_accept` (the
  activation-parity gate). These are the platform-independent numerics the loop
  logs and gates on.
- **`compute_image::compile::ternary_pipeline`** (already merged) produces the
  student ternary block (absmean + per-lane micro-scale + outlier + two-level
  int8), driven by a `QuantConfig` the loop can tighten per rejected block.

## Loop integration (in `run_distillation_loop`)
1. Bind the NF4 teacher arena once (`derive_…_abi` → `bind_nf4_tile640_triplet`
   on the Metal lane now; the mlmodelc/ANE lane when Golden Gate ships).
2. For each block: run the teacher forward → capture logits + hidden acts.
3. Quantize the student block with the current `QuantConfig`
   (`ternary_pipeline::quantize_tensor`) and run its forward.
4. `kd_divergence` / `block_accept` decide accept vs. re-distill; feed the
   existing Level 1 numerical gate and Level 2 joint-acceptance.
5. On accept, serialize the ternary block into the student `.cimage`.

## Sequencing / honesty
- **Drive the Metal lane and the stateless mlmodelc ANE path first.** The
  `coreai` executable path is scaffolding until macOS Golden Gate (fall); the
  ABI is lane-agnostic so it's ready for it, but don't gate the loop on CoreAI.
- **Wire the MTLSharedEvent handoff last**, on top of this proven ABI (not
  inferred shapes) — teacher-produces → student-consumes over the shared arena.
- **Verification boundary:** `distill_core` + the ABI derivation are verified on
  Linux (unit tests). The live teacher/student forward passes, the QAT/STE
  backward, and the actual perplexity/activation-parity numbers require the Mac
  (MLX + Metal, real Gemma 4 NF4 weights) — the loop is authored/gated here.
```
