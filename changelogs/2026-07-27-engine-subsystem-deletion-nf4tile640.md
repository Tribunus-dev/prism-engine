# Goal: Delete `compute-core/src/ecs/nf4tile640/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/nf4tile640/` — 15 files, 8,586 LOC. NF4
tiled-640 quantization path: accelerate, awls, calibration,
fused, hw_proof, learn, metal_tests, outliers, plan, profile,
protection, roles, squat, verify.

## Constitutional target

`crates/prism-ecs-quantization/` (the constitutional quantization
crate; the engine's `nf4tile640/` is the legacy home for the
NF4 tiled-640 path; the bitnet migration (E-1..E-8 → already
merged) absorbed bitnet/ into this same crate and provides the
template).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
bitnet migration is the most relevant template — both absorb
quantization modules into `prism-ecs-quantization`.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-nf4tile640` on branch
`migrate/nf4tile640`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to
  `prism-ecs-quantization` — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-quantization`
  to the engine's `Cargo.toml` if there are engine callers of
  the new constitutional surface.

## Success criteria

- All 15 files of `compute-core/src/ecs/nf4tile640/` removed.
- Constitutional surface in
  `crates/prism-ecs-quantization/src/nf4tile640/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_nf4tile640_imports`
  architecture test passes.
- `rg "use crate::ecs::nf4tile640::" compute-core/src/` returns
  no results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.
