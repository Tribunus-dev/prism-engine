# Goal: Delete `compute-core/src/ecs/compilation/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/compilation/` — 37 files, 17,165 LOC.
Compilation subsystem: graph compilation, layer planning,
quantization integration, compute-image compilation, CImage
lifecycle, kernel lowering, MIL construction, schema handling.

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate;
the engine's `compilation/` is the legacy home for graph compile;
the compile crate is the canonical home for compiler + CImage
lifecycle).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compiler migration (E-1..E-9, 9 commits) is the closest template
since it also targets `prism-ecs-compile`.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-compilation` on
branch `migrate/compilation`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-compile`
  — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-compile` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.
- **Three agents are targeting prism-ecs-compile simultaneously**
  (compilation, decode_attribution, cimage). All use isolated
  worktrees. The merge order will be: compilation first,
  decode_attribution second, cimage third. The "Take HEAD on
  architecture/src/lib.rs conflicts" rule applies: take HEAD +
  add new module declaration.

## Success criteria

- All 37 files of `compute-core/src/ecs/compilation/` removed.
- Constitutional surface in `crates/prism-ecs-compile/src/compilation/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_compilation_imports` architecture
  test passes.
- `rg "use crate::ecs::compilation::" compute-core/src/` returns
  no results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 221).
- Constitutional-side tests green.
