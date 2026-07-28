# Goal: Delete `compute-core/src/ecs/lut/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved; migration complete.

## Source

`compute-core/src/ecs/lut/` — 7 files, 2,682 LOC.

## Constitutional target

`crates/prism-ecs-codec/` (the engine's `lut/` is the
lookup-table codec path; the canonical home is the
`prism-ecs-codec` crate, which also absorbs the evaluator
subsystem).

## Migration pattern

Followed the assistant-graph / evaluator migration
pattern: a single constitutional-side commit at the front
of the chain (E-1), one caller-migration commit (E-3 + E-4
combined because the two changes are tightly coupled), a
safety-net commit (E-5), and a final legacy-deletion commit
(E-6). The engine had only one external caller
(`compute-core/src/bin/prism.rs`), which collapsed the
E-2..E-{N-2} caller-migration steps to zero separate
commits.

The migration is recorded in 7 commits on
`migrate/lut`:

| Commit  | Description                                              |
|---------|----------------------------------------------------------|
| E-0     | `chore(engine): add prism-ecs-codec dep`                 |
| E-1     | `feat(constitutional): add prism-ecs-codec::lut surface` |
| E-2     | `feat(constitutional): add prism-ecs-codec::lut::compile`|
| E-3+E-4 | `chore(engine): migrate engine callers to prism-ecs-codec::lut` |
| E-5     | `feat(architecture): add lut legacy-import safety net`   |
| E-6     | `chore(engine): delete the legacy engine's lut subsystem`|
| E-7     | `docs: mark lut engine-subsystem deletion goal achieved` |

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

The migration was performed in the isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-lut` on branch
`migrate/lut`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** The migration target is
  `prism-ecs-codec` — that name appears in every commit.
- **Engine dep audit at E-0.** `prism-ecs-codec` was added
  to the engine's `Cargo.toml` because `bin/prism.rs`
  imports `prism_ecs_codec::lut::graph::*` after E-3.

## Constitutional surface layout

`crates/prism-ecs-codec/src/lut/` (6 files, one authority
per file):

| File                  | Authority                                                      |
|-----------------------|----------------------------------------------------------------|
| `mod.rs`              | Module index, re-exports                                       |
| `graph.rs`            | Backend-neutral model graph descriptor (ModelGraph, ComputeNode, TensorRole, TensorBlueprint, ArchitectureFamily, ActivationFunction, UnifiedConfig, HfConfigBlock, RawHfConfig) |
| `evaluator.rs`        | FP16 CPU math kernels (lut_gemv_cpu, rms_norm_inplace, vec_add_inplace, silu_inplace, gelu_inplace, attention_cpu, rope_inplace, lut_embed, evaluate_activations) |
| `quantization.rs`     | INT8 KV cache and ternary weight packing (quantize_token, dequant_inline, pack_ternary_weights, extract_scale) |
| `table_builder.rs`    | Palettized LUT matrix format (LutRow, LutMatrix, pack_indices) |
| `compile.rs`          | `CompiledTensor` data type (the immutable AOT-compiled payload for a single palettized LUT tensor) |

31 unit tests cover the surface (5 quantization, 6
table_builder, 8 graph, 9 evaluator, 2 compile, plus
existing evaluator surface tests).

## Engine-side code that stayed in the engine

Two files were relocated but not lifted to codec (they
depend on engine-specific I/O and hardware backends):

- `compute-core/src/ecs/lut_compile.rs` (renamed from
  `compute-core/src/ecs/lut/compiler.rs`): the AOT
  compile orchestration (`compile_to_cimage`,
  `compile_gguf_to_cimage`) and the local re-export of
  `CompiledTensor` from `prism_ecs_codec::lut::compile`.
- `compute-core/src/ecs/lut_runtime.rs` (renamed from
  `compute-core/src/ecs/lut/engine.rs`): the legacy
  `PrismEngine` (KV cache, FP16 math, the inference loop)
  and the Metal/ANE backend wiring.

## Success criteria

- All 7 files of `compute-core/src/ecs/lut/` removed
  (`git rm -r` of 5 files: `evaluator.rs`, `graph.rs`,
  `mod.rs`, `quantization.rs`, `table_builder.rs`; the
  other 2 — `compiler.rs` and `engine.rs` — were renamed
  to `lut_compile.rs` and `lut_runtime.rs`).
- Constitutional surface in
  `crates/prism-ecs-codec/src/lut/` (6 files, 31 unit
  tests, all green).
- All engine callers migrated
  (`compute-core/src/bin/prism.rs` lines 441, 446, 531,
  593, 598, 603, 705 use the new paths).
- `workspace_contains_no_legacy_lut_imports` architecture
  test passes.
- `rg "use crate::ecs::lut::" compute-core/src/` returns
  no results.
- `rg "tribunus_compute_core::lut::" compute-core/src/`
  returns no results.
- Engine pre-existing build error count is **221**
  (unchanged from baseline).
- Constitutional-side tests green: `cargo test -p
  prism-architecture --lib` (5/5 pass), `cargo test -p
  prism-ecs-codec --lib lut` (31/31 pass).
