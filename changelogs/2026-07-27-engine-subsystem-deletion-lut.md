# Goal: Delete `compute-core/src/ecs/lut/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/lut/` — 7 files, 2,682 LOC.

## Constitutional target

`crates/prism-ecs-codec/` (the engine's `lut/` is the
lookup-table codec path; the canonical home is the
`prism-ecs-codec` crate, which already absorbs the evaluator
subsystem).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.
The evaluator migration (E-0..E-6, commits
`9acc8b11`..`3c8ebaa8` on `migrate/evaluator`) is the closest
template — both absorb small codec-style modules into
`prism-ecs-codec`.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-lut` on branch
`migrate/lut` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-lut -b migrate/lut
main`).

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to
  `prism-ecs-codec` — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-codec` to
  the engine's `Cargo.toml` if there are engine callers of the
  new constitutional surface.

## Success criteria

- All 7 files of `compute-core/src/ecs/lut/` removed.
- Constitutional surface in `crates/prism-ecs-codec/src/lut/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_lut_imports` architecture test
  passes.
- `rg "use crate::ecs::lut::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged (currently
  221).
- Constitutional-side tests green.
