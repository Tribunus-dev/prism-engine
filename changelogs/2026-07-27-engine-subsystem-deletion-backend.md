# Goal: Delete `compute-core/src/ecs/backend/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/backend/` — 37 files, 15,169 LOC. Hardware
backend abstractions: accelerate, ane, cpu, coreai, metal,
heterogeneous_executor, flex_dispatch, routing, placement, etc.

## Constitutional target

`crates/prism-ecs-kernel/` (the constitutional kernel crate; the
engine's `backend/` is the legacy home for backend hardware
abstractions. The kernel backends already absorbed from
scheduling migration (metal/, ane/, accelerate/, cpu/, legacy/,
dispatcher/, lane_executor_registry.rs) provide the template for
this absorption).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
scheduling kernel-backend migration is the most relevant
template — it absorbed the same kind of multi-backend-executor
code.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-backend` on branch
`migrate/backend`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.** Land a commit even if incomplete.
- **Correct crate name.** You are migrating to `prism-ecs-kernel`
  — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-kernel` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 37 files of `compute-core/src/ecs/backend/` removed (or
  re-homed; document any re-homing in the goal doc).
- Constitutional surface in `crates/prism-ecs-kernel/src/backend/`
  (extending the existing kernel backend modules).
- All engine callers migrated.
- `workspace_contains_no_legacy_backend_imports` architecture
  test passes.
- `rg "use crate::ecs::backend::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.
