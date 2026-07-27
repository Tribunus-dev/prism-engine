# Goal: Delete `compute-core/src/ecs/evaluator/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.
**Follow-up to:** `f2cfee80` (scheduling engine-deletion goal achieved).

## Source

`compute-core/src/ecs/evaluator/` — 10 files, 487 LOC.

## Constitutional target

`crates/prism-ecs-codec/` (already exists; add an `evaluator` submodule
if no other target fits) or create `crates/prism-ecs-eval/`.

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.

## Safety

- Work on branch `migrate/evaluator` (not main).
- Checkpoint commits every 30 minutes.
- No destructive ops; file-scoped recovery only.

## Success criteria

- `rg "use crate::ecs::evaluator::" compute-core/src/` returns no results.
- `git rm -r compute-core/src/ecs/evaluator/` succeeds.
- Engine pre-existing build error count is unchanged.
- Constitutional surface tests pass.
