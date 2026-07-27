# Goal: Delete `compute-core/src/ecs/audio/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/audio/` — 3 files, 829 LOC.

## Constitutional target

`crates/prism-audio/` (already exists in workspace).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.

## Safety

- Work on branch `migrate/audio` (not main).
- Checkpoint commits every 30 minutes.
- No destructive ops; file-scoped recovery only.

## Success criteria

- `rg "use crate::ecs::audio::" compute-core/src/` returns no results.
- `git rm -r compute-core/src/ecs/audio/` succeeds.
- Engine pre-existing build error count is unchanged.
- Constitutional surface tests pass.
