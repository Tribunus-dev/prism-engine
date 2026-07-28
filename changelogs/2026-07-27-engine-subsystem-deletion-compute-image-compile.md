# Goal: Delete `compute-core/src/ecs/compute_image/{compile,orchestrator}/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

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
