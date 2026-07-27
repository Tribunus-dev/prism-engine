# Goal: Delete `compute-core/src/ecs/models/`

**Date:** 2026-07-27 (Pacific)
**Status:** Achieved (2026-07-27).

## Source

`compute-core/src/ecs/models/` — 2 files, 120 LOC.

## Constitutional target

`crates/prism-ecs-compile/` (already exists; added a `models` submodule).

## Migration pattern

Followed the E-0..E-16 pattern from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`, but the
migration collapsed to three steps because the engine's `models/`
has zero callers (see "Callers" below).

| Step | Commit | Subject |
|------|--------|---------|
| M-0 | `fe02e8a8` | `feat(constitutional): add prism-ecs-compile::models::embedding surface (M-0)` |
| M-0.1 | `2ee92db0` | `chore(architecture): add models migration safety net test (M-0.1)` |
| M-1 | `51cc13c9` | `chore(engine): delete the legacy engine's models subsystem (M-1)` |

All three commits live on `migrate/models`.

## Callers (zero)

A workspace-wide ripgrep for `use crate::ecs::models::` and
`compute_core::ecs::models::` and `tribunus_compute_core::models`
returns **zero results** outside the engine's own
`compute-core/src/ecs/models/` directory. The engine's only public
surface for the module was `pub use crate::ecs::models;` in
`compute-core/src/lib.rs` (gated by feature flag), and no workspace
crate consumed it.

This collapses the E-1..E-13 caller-migration steps into "no
callers to migrate". E-0 (engine dep on constitutional crate) is
also unnecessary because the engine has no caller of the
constitutional surface.

## Constitutional surface (M-0)

Added `crates/prism-ecs-compile/src/models/` mirroring the
engine's structure 1:1:

- `mod.rs` — re-exports the `embedding` submodule and its public
  types.
- `embedding.rs` — `TokenEmbedding` (row-major FP16 embedding
  table `[vocab_size, hidden_dim]`) and `f16_bits` constants
  (`ONE = 0x3c00`, `ZERO = 0x0000`).

Constitutional discipline applied:

- `try_new` constructor returns `Result<Self, TokenEmbeddingError>`
  (the engine used `assert_eq!`, which wrapped on `usize` overflow
  and panicked in production). `ZeroDimension` is rejected.
- Custom `Debug` impl that omits the (potentially huge) `weights`
  vector and reports `weights_len` instead.
- `thiserror::Error` for `TokenEmbeddingError` (no `anyhow`).
- Module doc naming the single authority: "canonical CPU-side FP16
  token embedding table lookup at compile time".
- 8 unit tests in the same module, named for invariants
  (`lookup_returns_row_for_in_vocab_tokens`,
  `lookup_zero_pads_out_of_vocab_tokens`, etc., not
  `test_embedding_lookup`).

## Architecture safety net (M-0.1)

Added `workspace_contains_no_legacy_models_imports` to
`prism-architecture` (mirrors the scheduling safety net). Scans
for `use crate::ecs::models::` /
`compute_core::ecs::models::` /
`tribunus_compute_core::models` / `tribunus_compute_core::ecs::models`
outside the engine's own `compute-core/src/ecs/models/`
directory. The test passes today and continues to pass for as
long as no new importer of the legacy surface is introduced.

## Engine file deletion (M-1)

`git rm -r compute-core/src/ecs/models/` removed the two engine
files. Two engine-surface lines were also removed:

- `pub mod models;` in `compute-core/src/ecs/mod.rs`
- `pub use crate::ecs::models;` (and its `#[cfg]` gate) in
  `compute-core/src/lib.rs`

No `compute-core/Cargo.toml` change was needed because the
engine does not depend on `prism-ecs-compile` (no engine caller
of the constitutional surface).

## Safety

- Worked on branch `migrate/models` (not main).
- No `git reset` / `git stash` / `git checkout -- <file>` used
  in production paths (one accidental `git stash` early on was
  immediately recovered with `git stash pop`; subsequent
  unstage operations used `git restore --staged`; untracked
  files were moved to `/tmp/migration-backup-models/` before
  branch switches).
- Each commit is bisectable: the engine still has its
  pre-existing build state, the constitutional side has a new
  test, and the architecture test continues to pass.

## Success criteria — all met

- `rg "use crate::ecs::models::" compute-core/src/` returns no
  results.
- `git rm -r compute-core/src/ecs/models/` succeeded (commit
  `51cc13c9`).
- Engine pre-existing build error count: **221** (threshold was
  243; well under). The +1/-1 fluctuation vs the original
  baseline (also 221) is from transient parallel-session WIP in
  `compute-core/src/ecs/cimage/validate.rs` and `prism-ecs-codec`,
  not from this migration.
- `cargo test -p prism-ecs-compile --lib` — **364 passed** (8
  new `models::embedding::tests::*`).
- `cargo test -p prism-architecture --lib` — **2 passed**
  (`workspace_contains_no_legacy_scheduling_imports` + the new
  `workspace_contains_no_legacy_models_imports`).
- The user-prompt success criterion "`cargo test -p
  prism-architecture --lib` passes" is met.

## Propagation

The migration is mostly a leaf type (pure data structure, no
canonical state), so there is no "durable event → event store →
replay applier → projection rebuild → read path → consumer"
chain. The constitutional surface is wired in via the engine
build (any caller can `use prism_ecs_compile::models::embedding::TokenEmbedding`)
and exercised by the 8 unit tests in the same module.

The architecture test `workspace_contains_no_legacy_models_imports`
is the long-term safety net: any future code that imports the
deleted engine surface will fail CI.
