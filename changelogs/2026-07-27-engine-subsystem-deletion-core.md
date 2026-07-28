# Goal: Delete `compute-core/src/ecs/core/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/core/` — 121 files, 53,532 LOC. The single
largest engine subsystem still on disk; spans engine, gguf, model,
projection, session, supervisor, validator, worker, and many more
in-tree modules.

## Constitutional target

`crates/prism-ecs-core/` (the constitutional core crate; the
engine's `core/` is the legacy home for everything that doesn't
belong to a more specific subsystem — types move to either
`prism-ecs-core` itself or to a more specific existing
constitutional crate as appropriate per file's authority).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md` and
the assistant-graph migration recipe. The system migration
(49 files → `prism-ecs-runtime`, E-0..E-8) is the closest
template for a large multi-submodule migration.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-core` on branch
`migrate/core`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.** Land a commit on `migrate/core`
  even if incomplete.
- **Correct crate name.** You are migrating to `prism-ecs-core` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-core` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 121 files of `compute-core/src/ecs/core/` removed (or
  re-homed to a more specific constitutional crate; document any
  re-homing in the goal doc).
- Constitutional surface in `crates/prism-ecs-core/src/core/`
  (or wherever appropriate per file's authority).
- All engine callers migrated.
- `workspace_contains_no_legacy_core_imports` architecture test
  passes.
- `rg "use crate::ecs::core::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.

## Note on size

This is the largest engine subsystem. Expect many commits (E-0
through E-15+), long runtime, and many caller migrations. The
caller migration step is likely the longest single phase — audit
all callers carefully and migrate them in atomic per-caller
commits.
