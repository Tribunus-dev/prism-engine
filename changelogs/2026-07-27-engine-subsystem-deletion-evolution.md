# Goal: Delete `compute-core/src/ecs/evolution/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/evolution/` — 10 files, 5,581 LOC.

## Constitutional target

`crates/prism-ecs-ir/` (the engine's `evolution/` is the IR-level
evolution / rewriting layer; the canonical home is the
`prism-ecs-ir` crate, which already exists for IR kernels and
dialects).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.
The models migration (M-0..M-2, commits
`fe02e8a8`..`6b287348` on `migrate/models`) is the simplest
3-step template: constitutional surface → engine deletion →
goal achieved.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-evolution` on branch
`migrate/evolution` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-evolution -b
migrate/evolution main`).

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-ir` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-ir` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 10 files of `compute-core/src/ecs/evolution/` removed.
- Constitutional surface in `crates/prism-ecs-ir/src/evolution/`
  (or appropriate submodule).
- All engine callers migrated.
- `workspace_contains_no_legacy_evolution_imports` architecture
  test passes.
- `rg "use crate::ecs::evolution::" compute-core/src/` returns
  no results.
- Engine pre-existing build error count is unchanged (currently
  221).
- Constitutional-side tests green.
