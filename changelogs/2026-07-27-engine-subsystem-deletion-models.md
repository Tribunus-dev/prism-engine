# Goal: Delete `compute-core/src/ecs/models/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/models/` — 2 files, 120 LOC.

## Constitutional target

`crates/prism-ecs-compile/` (already exists; add a `models` submodule
or fold the small set of types into an existing `prism-ecs-compile` module).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.

## Safety

- Work on branch `migrate/models` (not main).
- Checkpoint commits every 30 minutes.
- No destructive ops; file-scoped recovery only.

## Success criteria

- `rg "use crate::ecs::models::" compute-core/src/` returns no results.
- `git rm -r compute-core/src/ecs/models/` succeeds.
- Engine pre-existing build error count is unchanged.
- Constitutional surface tests pass.
