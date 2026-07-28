# Goal: Delete `compute-core/src/ecs/compute_image/{residency,...,verification}/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/compute_image/` runtime + ancillary surface —
68 files, ~14K LOC. Subsystems:
- `residency/` (7 files, 2,506 LOC): tensor residency, allocation, I/O
- `heterogeneous/` (3 files, 2,116 LOC): heterogeneous compute targeting
- `megakernel/` (5 files, 1,935 LOC): megakernel fusion
- `kernel_selection/` (4 files, 195 LOC): runtime kernel selection
- `multimodal/` (5 files, 1,741 LOC): multimodal model adapters
- `model_family/` (6 files, 1,345 LOC): per-model-family bindings
- `variants/` (5 files, 1,323 LOC): variant configurations
- `program/` (8 files, 702 LOC): program/launch IR
- `content_store/` (8 files, 718 LOC): content-addressed store
- `executable/` (8 files, 289 LOC): executable descriptors
- `scheduler/` (3 files, 267 LOC): scheduling helpers
- `verification/` (6 files, 72 LOC): verification helpers

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate; the
engine's `compute_image/{residency,...,verification}/` are the legacy
homes for runtime surfaces and ancillary adapters; the compile crate
already owns `compilation/`, `cimage_pipeline/`, and `compute_image/`
core).

## Scope boundary (THIS AGENT)

You are migrating:
- All 12 subdirs listed in the Source section above.

You are NOT migrating:
- `compute-core/src/ecs/compute_image/` top-level (separate agent: `ci_core`)
- `compute-core/src/ecs/compute_image/cimage_packer/`, `manifest/` (separate agent: `ci_core`)
- `compute-core/src/ecs/compute_image/compile/`, `orchestrator/` (separate agent: `ci_compile`)

If you find callers that import from these OUT-OF-SCOPE subdirs, do
NOT migrate those subdirs — leave the caller as-is for the other
agents to handle.

## Migration pattern

Follow E-0..E-N+2 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compilation migration at `99bb0554` (E-0..E-5, 6 commits) is the
closest template since it also targets `prism-ecs-compile`.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-ci-runtime` on branch
`migrate/ci-runtime`.

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
  need to be renamed to `legacy_compute_image_runtime/` rather than
  deleted.

## Success criteria

- All 68 files of `compute-core/src/ecs/compute_image/{residency,...,verification}/`
  removed or renamed to `legacy_compute_image_runtime/`.
- Constitutional surface in
  `crates/prism-ecs-compile/src/compute_image_runtime/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_compute_image_runtime_imports`
  architecture test passes.
- `rg "use crate::ecs::compute_image::(residency|heterogeneous|megakernel|kernel_selection|multimodal|model_family|variants|program|content_store|executable|scheduler|verification)" compute-core/src/` returns no results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 185).
- Constitutional-side tests green.
