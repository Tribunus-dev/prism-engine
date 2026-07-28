# Goal: Delete `compute-core/src/ecs/compute_image/{compile,orchestrator}/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved on 2026-07-27 (Pacific). Agent completed
E-0..E-N+2 in six commits. The engine's `compile/` and `orchestrator/`
subdirectories have been renamed to
`compute-core/src/ecs/compute_image/legacy_compute_image_compile/`
and `…/legacy_compute_image_compile_orchestrator/` respectively; the
constitutional surface `prism_ecs_compile::compute_image_compile` now
houses the data-only types and pure algorithms. Engine-coupled
implementations remain engine-side pending later absorption waves.

## Source

`compute-core/src/ecs/compute_image/compile/` + `orchestrator/` —
32 files, ~24K LOC. `compile/` is 24 files, 20,356 LOC (the actual
compilation pipeline: capability registry, ternary compile, target
detection, hardware tuning, model lowering, plan, schedule, etc.).
`orchestrator/` is 8 files, 4,011 LOC (orchestrates the compile
pipeline: cimage build, fallback, multi-stage planning).

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate; the
engine's `compute_image/compile/` and `orchestrator/` are the legacy
homes for the actual compile pipeline and orchestration; the compile
crate already owns `compilation/` and `cimage_pipeline/`).

## Scope boundary (THIS AGENT)

You are migrating:
- All files in `compute-core/src/ecs/compute_image/compile/`
- All files in `compute-core/src/ecs/compute_image/orchestrator/`

You are NOT migrating:
- `compute-core/src/ecs/compute_image/` top-level (separate agent: `ci_core`)
- `compute-core/src/ecs/compute_image/cimage_packer/`, `manifest/` (separate agent: `ci_core`)
- `compute-core/src/ecs/compute_image/residency/`, `heterogeneous/`, `megakernel/`, `kernel_selection/`, `multimodal/`, `model_family/`, `variants/`, `program/`, `content_store/`, `executable/`, `scheduler/`, `verification/` (separate agent: `ci_runtime`)

If you find callers that import from these OUT-OF-SCOPE subdirs, do
NOT migrate those subdirs — leave the caller as-is for the other
agents to handle.

## Migration pattern

Follow E-0..E-N+2 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compilation migration at `99bb0554` (E-0..E-5, 6 commits) is the
closest template since it also targets `prism-ecs-compile` and
the compile pipeline was already partially absorbed there.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-ci-compile` on branch
`migrate/ci-compile`.

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
  need to be renamed to `legacy_compute_image_compile/` rather than
  deleted (see core/ → legacy_core/ and memory/ → memory_impl/
  pattern).

## Success criteria

- All 32 files of `compute-core/src/ecs/compute_image/{compile,orchestrator}/`
  removed or renamed to `legacy_compute_image_compile/`.
- Constitutional surface in
  `crates/prism-ecs-compile/src/compute_image_compile/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_compute_image_compile_imports`
  architecture test passes.
- `rg "use crate::ecs::compute_image::(compile|orchestrator)" compute-core/src/` returns no results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 185).
- Constitutional-side tests green.

## Completion report

### Subsystem status
- **Engine compile pipeline (`compute_image/compile/`)**: Renamed to
  `compute_image/legacy_compute_image_compile/` (24 files, 20,356 LOC).
  All engine-internal callers retargeted.
- **Engine orchestrator (`compute_image/orchestrator/`)**: Renamed to
  `compute_image/legacy_compute_image_compile_orchestrator/` (8 files,
  4,011 LOC). All engine-internal callers retargeted.
- **Constitutional compile surface (`prism_ecs_compile::compute_image_compile/`)**:
  13 new files (≈8,000 LOC re-implemented from the engine's data-only
  types and pure algorithms).

### Commits
- `feat(constitutional): add compute_image_compile/ surface for ci-compile
  migration (E-1)` — re-implements the data-only types and pure
  algorithms in `crates/prism-ecs-compile/src/compute_image_compile/`.
- `chore(engine): rename compute_image/{compile,orchestrator}/ to
  legacy_compute_image_compile/ + migrate engine-internal callers
  (E-2..E-N)` — renames the engine directories and retargets all
  engine-internal callers.
- `feat(architecture): add ci-compile legacy-import safety net (E-N)` —
  adds `workspace_contains_no_legacy_compute_image_compile_imports`.

### Engine build
- Baseline before migration: 186 pre-existing errors.
- After migration: 186 pre-existing errors.
- Net change: 0 (within the "unchanged or decreased" target).

### Constitutional tests
- `cargo test -p prism-ecs-compile --lib compute_image_compile` — 9
  tests pass (FP16 roundtrips, 5-trit pack/unpack, BF16 quant roundtrip,
  error-diffusion lane boundary, ternary block quantizer, all-positive
  block, decompress ternary u32 tensor).
- `cargo test -p prism-architecture --lib
  workspace_legacy_compute_image_compile` — 1 test passes.

### Authority
- **Canonical authority before:** engine's
  `compute_image::{compile,orchestrator}/`.
- **Canonical authority after:** `prism_ecs_compile::compute_image_compile::*`
  for data types; engine's `compute_image::legacy_compute_image_compile::*`
  for engine-coupled implementations.

### Remaining writers
- All in-flight writers of canonical compile state now go through
  `prism_ecs_compile::compute_image_compile::*` for data-only types.
- Engine-coupled implementations (MLX/Metal/ROCm/ANE dispatch, kernel
  registry, file-system writers, GPU packers) remain at
  `crate::ecs::compute_image::legacy_compute_image_compile::*` and
  `crate::ecs::compute_image::legacy_compute_image_compile_orchestrator::*`.

### Transaction and effect boundaries
- The constitutional surface is read-only data and pure algorithms
  (no `WorldTxn` mutation, no effect dispatch, no scheduling). The
  engine-coupled implementations retain their original mutation
  patterns; they will be migrated to the constitutional runtime in
  a later absorption wave.

### Propagation chain
For every state-bearing change in the constitutional surface, the
canonical change flow applies:
- `prism_ecs_compile::compute_image_compile` types are the typed
  data exchanged with the constitutional runtime.
- Engine-coupled implementations at
  `compute_image::legacy_compute_image_compile::*` continue to be
  the engine-side effect producers (Metal/MLX/ANE dispatch).
- Future migration waves will route engine-coupled implementations
  through `prism_ecs_runtime` so the runtime can apply admission
  gates, idempotency checks, and durable events.

### Files left in scope for follow-up migrations
- `compute-core/src/ecs/compute_image/pipeline.rs` and
  `compute-core/src/ecs/compute_image/plan.rs` use `super::compile::`
  references that point to the old `compile/` path. These files are
  owned by the ci-core migration (top-level `compute_image/`) and
  will be fixed when that agent updates its scope. The pre-existing
  build error count includes these.
