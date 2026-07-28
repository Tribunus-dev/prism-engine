# Goal: Delete `compute-core/src/ecs/nf4tile640/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved. Migration E-0..E-6 complete; engine
subsystem physically deleted; constitutional surface in
`crates/prism-ecs-quantization/src/nf4tile640/`; architecture
safety net test green.

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

## Migration sequence (E-0..E-6)

The engine already had a `prism-ecs-quantization` dependency in
its Cargo.toml (left over from the bitnet migration), so E-0
was a no-op. The actual sequence was E-1..E-6.

  - E-1 `f2357de5` — `feat(constitutional): add
    prism-ecs-quantization::nf4tile640 surface` — re-implement
    the engine's ecs::nf4tile640 module as the canonical
    constitutional surface. 15 source files, one authority per
    file, 8,450 LOC after dropping one dead function.
  - E-2 `c740dadd` — `chore(engine): migrate nf4tile640
    engine-internal callers to constitutional surface` — 7
    engine-internal files: cimage/shard_builder.rs,
    cimage/mlp_reference.rs, backend/metal.rs,
    compilation/matrix_distill.rs, tts/code_predictor.rs,
    tts/pipeline.rs, tts/talker.rs.
  - E-3 `66b2586c` — `chore(engine): migrate nf4tile640
    cross-crate callers to constitutional surface` — 4
    cross-crate files: bin/diagnose_nf4_roundtrip.rs,
    bin/tribunus-compute-image.rs,
    tests/metal_nf4_int8_conformance.rs,
    tests/residency_tests.rs.
  - E-4 `098e6bd4` — `chore(engine): drop nf4tile640 re-export
    and module declaration` — drop `pub use crate::ecs::nf4tile640`
    in lib.rs:465 and `pub mod nf4tile640;` in ecs/mod.rs:143.
  - E-5 `3d56cc1e` — `feat(architecture): add nf4tile640
    legacy-import safety net` — new
    `workspace_contains_no_legacy_nf4tile640_imports` test in
    crates/architecture/src/workspace_legacy_nf4tile640_imports.rs.
  - E-6 `0003755c` — `chore(engine): delete the legacy engine's
    nf4tile640 subsystem` — `git rm -r` the 15 files, 8,586 LOC
    directory.

## Verification

- `cargo test -p prism-architecture --lib` → 9 passed
  (incl. the new `workspace_contains_no_legacy_nf4tile640_imports`).
- `cargo test -p prism-ecs-quantization --lib nf4tile640`
  → 109 passed.
- `cargo check -p tribunus-compute-core --lib` → 192 errors
  (down from 193; the legacy module's own compile errors no
  longer count).
