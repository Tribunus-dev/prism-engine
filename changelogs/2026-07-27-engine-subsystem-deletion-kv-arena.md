# Goal: Delete `compute-core/src/ecs/kv_arena/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/kv_arena/` — 5 files, 933 LOC. KV-cache
arena: backend, block, prefix, refcount, mod.

## Constitutional target

`crates/prism-kv-cache/` (a dedicated constitutional crate for
KV-cache). Smallest of this batch — expect fast turnaround.

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
models migration (M-0..M-2, 3 commits) is the simplest template.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-kv-arena` on branch
`migrate/kv-arena`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-kv-cache`
  — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-kv-cache` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 5 files of `compute-core/src/ecs/kv_arena/` removed.
- Constitutional surface in `crates/prism-kv-cache/src/arena/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_kv_arena_imports` architecture
  test passes.
- `rg "use crate::ecs::kv_arena::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.
