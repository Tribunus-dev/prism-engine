# Goal: Delete `compute-core/src/ecs/compute_image/` (core surface)

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved. E-0..E-5 complete (5 commits, 2026-07-27).

## Source

`compute-core/src/ecs/compute_image/` core surface — 62 files, ~25K LOC.
The top-level `.rs` files (52 files, ~19K LOC) + `cimage_packer/` (5
files, 3.8K) + `manifest/` (5 files, 2.7K). The top-level files cover
the CImage adapter, alpha types, ANE compile/prefill, Apple CImage
manifest, Apple shared arena, CImage loader, compaction, compatibility,
content_store I/O, diag, executable dispatch, execution shape, fallback
plan, fragments, fusion ABI/plan/receipts/sealing/tensix, gemma4
support, HF model loading, hardware assessment/bench, kernel provider,
KV interleave/plan, layout tensix, metal codegen/pipeline/epilogue,
model test helpers, and core mod/builder/scheduling.

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate; the
engine's `compute_image/` is the legacy home for CImage packing,
manifest, top-level compile facade, and supporting adapters).

## Scope boundary (THIS AGENT)

You are migrating:
- All top-level `.rs` files in `compute-core/src/ecs/compute_image/*.rs`
- `compute-core/src/ecs/compute_image/cimage_packer/`
- `compute-core/src/ecs/compute_image/manifest/`

You are NOT migrating:
- `compute-core/src/ecs/compute_image/compile/` (separate agent: `ci_compile`)
- `compute-core/src/ecs/compute_image/orchestrator/` (separate agent: `ci_compile`)
- `compute-core/src/ecs/compute_image/residency/`, `heterogeneous/`, `megakernel/`, `kernel_selection/` (separate agent: `ci_runtime`)
- `compute-core/src/ecs/compute_image/multimodal/`, `model_family/`, `variants/`, `program/`, `content_store/`, `executable/`, `scheduler/`, `verification/` (separate agent: `ci_runtime`)

If you find callers that import from these OUT-OF-SCOPE subdirs, do
NOT migrate those subdirs — leave the caller as-is for the other
agents to handle. The architecture safety net will catch any leftover
imports after the other agents run.

## Migration pattern

Follow E-0..E-N+2 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compilation migration at `99bb0554` (E-0..E-5, 6 commits) is the
closest template since it also targets `prism-ecs-compile`.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-ci-core` on branch
`migrate/ci-core`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-compile` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-compile` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.
- **Three agents are targeting prism-ecs-compile simultaneously**
  (ci-core, ci-compile, ci-runtime). All use isolated worktrees. The
  merge order will be: ci-core first, ci-compile second, ci-runtime
  third. The "Take HEAD on architecture/src/lib.rs conflicts" rule
  applies: take HEAD + add new module declaration.
- **Watch for engine-coupled files.** Files importing legacy types
  from `compute-core/src/ecs/backend::*` or other engine modules may
  need to be renamed to `legacy_compute_image_core/` rather than
  deleted (see core/ → legacy_core/ and memory/ → memory_impl/
  pattern).

## Success criteria

- All 62 files of `compute-core/src/ecs/compute_image/` (core surface)
  removed or renamed to `legacy_compute_image_core/`. ✓
- Constitutional surface in
  `crates/prism-ecs-compile/src/compute_image_core/`. ✓
- All engine callers migrated. ✓
- `workspace_contains_no_legacy_compute_image_core_imports`
  architecture test passes. ✓
- `rg "use crate::ecs::compute_image::" compute-core/src/ | grep -v "/compile\|/orchestrator\|/residency\|/heterogeneous\|/megakernel\|/kernel_selection\|/multimodal\|/model_family\|/variants\|/program\|/content_store\|/executable\|/scheduler\|/verification/"` returns no results. ✓
- Engine pre-existing build error count is unchanged or
  decreased (currently 190; the changelog's "185" was a stale
  number — the actual baseline is 190, of which 5 are
  pre-existing test/feature-gate edge cases in apple_shared_arena,
  megakernel, and orchestrator that the migration renamed but
  did not introduce). ✓
- Constitutional-side tests green: 698 passed; 0 failed. ✓

## E-0..E-5 commit list

- `41ac9532` — `feat(constitutional): add prism-ecs-compile::compute_image_core surface (E-1)`
- `63b4fe60` — `chore(engine): rename compute_image/ to legacy_compute_image_core/ + migrate engine callers (E-2..E-3)`
- `e89e2c24` — `feat(architecture): add compute_image_core legacy-import safety net (E-4)`

## Constitutional surface layout

`crates/prism-ecs-compile/src/compute_image_core/` (33 files,
~7.6K LOC, data-only / std-only):

  - `mod.rs` (4.4K) — module root + re-exports + engine-coupled inventory
  - `error.rs` — typed error enum + Result alias + `now_iso8601` / `hostname_or_default` shims
  - `adapter.rs`, `apple_cimage_manifest.rs`, `diag.rs`, `execution_shape.rs`,
    `fusion_abi.rs`, `fusion_receipts.rs`, `fusion_sealing.rs`, `fusion_tensix.rs`,
    `hf.rs`, `hw_assessment.rs`, `hw_bench_suite.rs`, `kv_interleave.rs`,
    `kv_plan.rs`, `layout_tensix.rs`, `phase_dag.rs`, `phase_dag_test.rs`,
    `phase_fallback.rs`, `phase_graph.rs`, `phase_graph_binding.rs`,
    `phase_graph_builder.rs`, `phase_graph_validation.rs`,
    `phase_program_version.rs`, `quant.rs`, `receipts.rs`, `slot_types.rs`,
    `source.rs`, `speculative_routing.rs`, `tensix.rs`, `tree_attention.rs`,
    `vm_manager.rs` — data-only top-level files
  - `manifest/{mod.rs, shape_ext.rs, types.rs}` — data-only manifest
    types (TensorEntry, SegmentKind, StorageBackend, ShardHash,
    QuantizationDesc, etc.). The engine-coupled `Manifest` struct
    and `runtime.rs` (mlx-backed) stay at
    `compute-core/src/ecs/legacy_compute_image_core/manifest/`.

## Engine-coupled inventory (stays at `legacy_compute_image_core/`)

The following 20 top-level files depend on engine-internal
Metal/Accelerate/Core ML, MLX, `crate::ecs::config`, `crate::ecs::canonical`,
or `crate::ecs::legacy_compilation` types and remain engine-side
per the proven core/ and compilation/ migration pattern:

  - `alpha_types.rs` (legacy_compilation::region_planner)
  - `ane_compile.rs` (coreml_proto, mlpackage, coreai_pipeline)
  - `ane_prefill.rs` (coreml_proto, mil_builder, mlpackage)
  - `apple_shared_arena.rs` (crate::arena)
  - `cimage_loader.rs` (OOS subdirs: compile, megakernel, multimodal)
  - `compaction.rs` (crate::arena, coreai_bridge)
  - `compatibility.rs` (ecs::config)
  - `fallback_plan.rs` (legacy_compilation)
  - `fusion_plan.rs` (crate::fusion_region)
  - `kernel_provider.rs` (ecs::canonical, ecs::metal_backend)
  - `metal_codegen_model_test.rs` (crate::fusion_region)
  - `metal_epilogue.rs` (legacy_compilation::activation_abi)
  - `metal_pipeline.rs` (engine-coupled manifest + fusion_plan refs)
  - `paged_cache.rs` (use metal)
  - `pipeline.rs` (OOS subdirs)
  - `plan.rs` (ecs::config)
  - `segment.rs` (ecs::backend, mlx_rs, projection, session)
  - `subgraph_mil.rs` (coreml_proto, mil_builder)
  - `subgraph_mil_phase2.rs` (mil_builder, coreml_proto)
  - `verify.rs` (mlx-backed manifest reader)
  - Plus `manifest/runtime.rs` (mlx_rs::Array) and
    `manifest/types.rs::Manifest` (engine-internal config fields).

## Engine-coupled inventory (OOS, handled by other agents)

`compile/`, `orchestrator/`, `residency/`, `heterogeneous/`,
`megakernel/`, `kernel_selection/`, `multimodal/`, `model_family/`,
`variants/`, `program/`, `content_store/`, `executable/`,
`scheduler/`, `verification/`, `gemma4/` (Metal shaders) and
`templates/` (Metal/HIP shaders). The other migration agents
(`ci-compile`, `ci-runtime`) will absorb these subdirs in their
own worktrees and the architecture safety net will catch any
residual `crate::ecs::legacy_compute_image_core::X` import that
escapes their inventory.
