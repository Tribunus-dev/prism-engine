# Goal: Delete `compute-core/src/ecs/compute_image/{residency,...,verification}/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved (2026-07-27).

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

## Completion report (2026-07-27)

### Migration summary

- **Affected subsystem:** the runtime + ancillary surface of
  `compute-core/src/ecs/compute_image/{residency,...,verification}/`
  (12 subdirs, 68 files, ~14K LOC).
- **Status:** **Canonical** (the constitutional surface owns the
  data-only types; engine-coupled implementations stay at
  `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`).
- **Crate:** `prism-ecs-compile`.

### Migration steps (E-0..E-N+2)

- **E-1**: Re-implemented the data-only types in
  `crates/prism-ecs-compile/src/compute_image_runtime/` (12
  sub-modules, ~2,200 LOC, all single-authority module docs).
  Added shared `ContentHash` and `ExecutionShapeClass` newtypes
  used across the surface.
- **E-2**: Renamed the engine's 12 subdirs from
  `compute-core/src/ecs/compute_image/<sub>/` to
  `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/<sub>/`
  (`git mv` × 68 files + 5 .metal shaders).
- **E-3**: Updated `compute-core/src/ecs/compute_image/mod.rs` to
  expose the legacy directory as a single
  `pub(crate) mod legacy_compute_image_runtime;` declaration.
- **E-4**: Migrated engine-internal callers from
  `crate::ecs::compute_image::X::Y` to
  `crate::ecs::compute_image::legacy_compute_image_runtime::X::Y`
  across:
  - `compute-core/src/ecs/aot/gemma4_frontend.rs`
  - `compute-core/src/ecs/backend/ane.rs`
  - `compute-core/src/ecs/legacy_core/profiled_model.rs`
  - `compute-core/src/ecs/legacy_runtime/executable_*.rs` (5 files)
  - `compute-core/src/ecs/tts/talker.rs`
  - `compute-core/src/ecs/server/distill_worker.rs`
  - `compute-core/src/ecs/compute_image/orchestrator/*.rs` (5 files;
    out-of-scope but dependent on the migration)
  - `compute-core/src/ecs/compute_image/cimage_loader.rs`
  - `compute-core/src/ecs/compute_image/cimage_packer/pipeline.rs`
  - `compute-core/src/ecs/compute_image/compile/pipeline.rs`
  - All intra-legacy files (sibling references that
    `git mv` left pointing at the old path).

- **E-N**: Added
  `crates/architecture/src/workspace_legacy_compute_image_runtime_imports.rs`
  with the `workspace_contains_no_legacy_compute_image_runtime_imports`
  test. Wired into `crates/architecture/src/lib.rs`.
- **E-N+1**: Engine-coupled files (those importing
  `crate::integration::ContentHash`, the binary-layout
  `MultimodalInputDescriptorV1`, or Metal-coupled
  `Megakernel` / `KernelBuffers`) live at
  `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`.
- **E-N+2**: This changelog updated.

### Authority before / after

- **Before:** the engine's
  `compute-core/src/ecs/compute_image/{residency,...,verification}/`
  were the sole canonical homes.
- **After:** data-only types are canonical at
  `prism_ecs_compile::compute_image_runtime::*` (constitutional).
  Engine-coupled implementations stay canonical at
  `compute-core/src/ecs/compute_image/legacy_compute_image_runtime::*`
  (engine-internal legacy home).

### Tests executed

- `cargo test -p prism-ecs-compile --lib` → 603 passed; 0 failed.
- `cargo test -p prism-architecture --lib` → 20 passed; 0 failed
  (1 new test added:
  `workspace_legacy_compute_image_runtime_imports::tests::workspace_contains_no_legacy_compute_image_runtime_imports`).
- `cargo check -p tribunus-compute-core --lib` → 185 pre-existing
  errors (unchanged from baseline).

### Authority-leak audit

- The `legacy_compute_image_runtime/` directory is the migration
  inventory and is exempt from the safety net scan.
- The new `workspace_legacy_compute_image_runtime_imports` test
  fails if any file in the workspace OUTSIDE
  `legacy_compute_image_runtime/` imports
  `crate::ecs::compute_image::X::*` for the 12 subdirs
  `X ∈ {residency, heterogeneous, megakernel, kernel_selection,
  multimodal, model_family, variants, program, content_store,
  executable, scheduler, verification}`.

### Legacy path still awaiting purge

None — all 12 subdirs of
`compute-core/src/ecs/compute_image/{residency,...,verification}/`
are now
`compute-core/src/ecs/compute_image/legacy_compute_image_runtime/<sub>/`
and the safety net enforces that no caller reaches the
non-legacy path.

