# Goal: Delete `compute-core/src/ecs/inference/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/inference/` — 5 files, 470 LOC.

## Constitutional target

`crates/prism-ecs-server/` (already exists; this inference state
moves to the server crate, since inference state is closer to
session/server lifecycle than to runtime scheduling).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.

Note: this migration is partially done — the E-6, E-7, E-8
commits already updated 3 of the inference/* files to import
from `prism_ecs_runtime::scheduling::*`. The remaining work is:
1. Survey what `prism-ecs-server` needs from `crate::ecs::inference::*`.
2. Move the types (e.g. ComputeImageState, InferenceSessionState,
   InferenceStepState) to `prism-ecs-server`.
3. Update the import paths.
4. `git rm -r compute-core/src/ecs/inference/`.

## Safety

- Work on branch `migrate/inference` (not main).
- Checkpoint commits every 30 minutes.

## Success criteria

- All callers migrated.
- `git rm -r compute-core/src/ecs/inference/` succeeds.
- Engine pre-existing build error count is unchanged.
- `cargo test -p prism-architecture --lib` passes.
- Constitutional surface tests pass.
