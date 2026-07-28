# Goal: Delete `compute-core/src/ecs/system/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/system/` — 49 files, 5,915 LOC.

## Constitutional target

`crates/prism-ecs-runtime/` (the engine's `system/` is the
runtime's dispatch + system-orchestration layer; the constitutional
home is `prism-ecs-runtime::scheduling::systems` which already
exists and absorbs scheduling dispatchers; engine-side system
types move to `prism-ecs-runtime::systems`).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.
The audio migration's caller-migration-then-engine-deletion
pattern is the simplest template.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-system` on branch
`migrate/system` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-system -b
migrate/system main`).

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.** Land a commit on
  `migrate/system` even if incomplete.
- **Bisectable commits.**
- **Correct crate name.** You are migrating to
  `prism-ecs-runtime` — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-runtime` to
  the engine's `Cargo.toml` if there are engine callers of the
  new constitutional surface.

## Success criteria

- All 49 files of `compute-core/src/ecs/system/` removed.
- Constitutional surface in `crates/prism-ecs-runtime/src/systems/`
  (extend the existing `scheduling::systems` module or add new
  modules as needed).
- All engine callers migrated to the constitutional home.
- `workspace_contains_no_legacy_system_imports` architecture
  test passes.
- `rg "use crate::ecs::system::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged (currently
  221).
- Constitutional-side tests green.
