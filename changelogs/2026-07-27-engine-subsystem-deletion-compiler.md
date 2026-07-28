# Goal: Delete `compute-core/src/ecs/compiler/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/compiler/` — 25 files, 11,020 LOC.

## Constitutional target

`crates/prism-ecs-compile/` (the engine's `compiler/` is the
largest remaining legacy block in the ecs tree; the canonical
home for everything in it is the constitutional `prism-ecs-compile`
crate, which already absorbs `models/embedding/` and the
`compute_image/` work-in-progress).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`,
with the scheduling goal as the canonical recipe. The
assistant-graph migration
(`changelogs/2026-07-27-engine-subsystem-deletion-assistant-graph.md`,
commits `f0a4fe89`..`db4bb6c6`) is the more recent and cleaner
template — the E-0..E-8 sequence there (dep → surface → caller
migrations → safety net → pre-deletion → git rm → goal achieved)
is the minimum the compiler migration must follow.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared with other agents and is currently on the
`migrate/inference` branch. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-compiler` on branch
`migrate/compiler` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-compiler -b
migrate/compiler main` from any clean repo checkout). All edits,
commits, and tests for this migration must happen in that
isolated worktree.

## Safety

- **No destructive ops.** Never run `git reset`, `git stash`,
  `git checkout -- <file>`, or `mavis-trash` in a way that affects
  another agent's work. File-scoped recovery (`git checkout
  <own-checkpoint-sha> -- <own-file>`) is the only allowed
  recovery.
- **Checkpoint every 30 min.** Land a commit on
  `migrate/compiler` even if incomplete.
- **Bisectable commits.** Each commit is independently
  buildable (or, for the engine, has the same pre-existing
  error count as the parent commit).
- **Correct crate name in commit messages.** You are migrating
  to `prism-ecs-compile` — write that name in your commits, not
  `prism-ecs-agent` or `prism-ecs-codec` or anything else.
- **Engine dep audit at E-0.** Only add `prism-ecs-compile` to
  the engine's `Cargo.toml` if there are engine callers of the
  new constitutional surface. If the engine's `compiler/` has no
  external callers, E-0 can be a no-op (the E-1 surface goes
  straight in).

## Success criteria

- All 25 files of `compute-core/src/ecs/compiler/` removed.
- Constitutional surface in `crates/prism-ecs-compile/src/compiler/`
  (or appropriate submodule), one authority per file.
- All engine callers of the engine's compiler surface migrated
  to the constitutional home.
- `workspace_contains_no_legacy_compiler_imports` architecture
  test in `crates/architecture/src/workspace_legacy_compiler_imports.rs`
  passes.
- `rg "use crate::ecs::compiler::" compute-core/src/` returns
  no results.
- Engine pre-existing build error count is unchanged (currently
  221).
- Constitutional-side tests green
  (`cargo test -p prism-ecs-compile --lib compiler` or
  equivalent).
