# Goal: Delete `compute-core/src/ecs/bitnet/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/bitnet/` — 8 files, 3,661 LOC.

## Constitutional target

`crates/prism-ecs-quantization/` (the engine's `bitnet/` is the
1-bit / 1.58-bit quantization path; the canonical home is the
`prism-ecs-quantization` crate, which already exists for
quantization contracts).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.
The models migration (M-0..M-2) is the simplest 3-step template.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-bitnet` on branch
`migrate/bitnet` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-bitnet -b
migrate/bitnet main`).

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to
  `prism-ecs-quantization` — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-quantization`
  to the engine's `Cargo.toml` if there are engine callers of
  the new constitutional surface.

## Success criteria

- All 8 files of `compute-core/src/ecs/bitnet/` removed.
- Constitutional surface in `crates/prism-ecs-quantization/src/bitnet/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_bitnet_imports` architecture
  test passes.
- `rg "use crate::ecs::bitnet::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged (currently
  221).
- Constitutional-side tests green.
