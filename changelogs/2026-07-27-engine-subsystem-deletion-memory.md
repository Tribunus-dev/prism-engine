# Goal: Delete `compute-core/src/ecs/memory/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/memory/` — 11 files, 2,584 LOC. Memory
subsystem: allocator, ane_warmup.mil, candle_bridge,
compute_image_bridge, coreai_warmup, enforcer, iosurface_storage,
monitor, plan, pool, telemetry.

## Constitutional target

`crates/prism-ecs-data/` (the constitutional data crate; the
engine's `memory/` is the legacy home for memory lifecycle and
telemetry; the data crate is the canonical home for
allocator/pool/telemetry abstractions).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
evaluator migration (E-0..E-6, 6 commits) is the closest
template for a small codec-style migration.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-memory` on branch
`migrate/memory`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-data` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-data` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 11 files of `compute-core/src/ecs/memory/` removed.
- Constitutional surface in `crates/prism-ecs-data/src/memory/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_memory_imports` architecture
  test passes.
- `rg "use crate::ecs::memory::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.
