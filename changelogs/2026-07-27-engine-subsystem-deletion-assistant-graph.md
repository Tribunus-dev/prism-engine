# Goal: Delete `compute-core/src/ecs/assistant_graph/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.
**Follow-up to:** `f2cfee80` (scheduling engine-deletion goal achieved).

## Source

`compute-core/src/ecs/assistant_graph/` — 9 files, 1,568 LOC.

## Constitutional target

`crates/prism-ecs-agent/` (already exists in workspace).

## Migration pattern

Follow the E-0..E-16 pattern from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`:

1. Survey the engine callers (any `use crate::ecs::assistant_graph::*`).
2. Add `prism-ecs-agent` dep to engine Cargo.toml.
3. For each caller, update the import path.
4. Verify pre-existing error count is unchanged.
5. `git rm -r compute-core/src/ecs/assistant_graph/`.
6. Clean up engine Cargo.toml.

## Safety

- Work on branch `migrate/assistant-graph` (not main).
- Checkpoint commits every 30 minutes.
- No `git reset`, `git stash`, `git checkout -- <file>`, or `mavis-trash`.
- File-scoped recovery only: `git checkout <own-hash> -- <own-file>`.

## Token budget

If the agent's token budget is exhausted mid-task, the work is
recoverable from the agent's session snapshot at
`/Users/user/.minimax/v2/sessions/<task_id>/snapshot.json`. A new
agent can be dispatched to resume from the last checkpoint.

## Success criteria

- `rg "use crate::ecs::assistant_graph::" compute-core/src/` returns no results.
- `git rm -r compute-core/src/ecs/assistant_graph/` succeeds.
- Engine pre-existing build error count is unchanged.
- Architecture test (or its successor) still passes.
- Constitutional surface tests pass.
