# Goal: Delete `compute-core/src/ecs/evolution/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Achieved (commits `b93cb832`..`8e32c87d` on
`migrate/evolution`).

## Source

`compute-core/src/ecs/evolution/` — 10 files, 5,581 LOC.

## Constitutional target

`crates/prism-ecs-ir/` (the engine's `evolution/` is the IR-level
evolution / rewriting layer; the canonical home is the
`prism-ecs-ir` crate, which already exists for IR kernels and
dialects).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.
The models migration (M-0..M-2, commits
`fe02e8a8`..`6b287348` on `migrate/models`) is the simplest
3-step template: constitutional surface → engine deletion →
goal achieved.

## Commit list (E-0..E-8)

The migration completed in **8 commits** (E-0 was a no-op since
`prism-ecs-ir` was already an engine dep):

| Step    | SHA        | Commit                                                           |
| ------- | ---------- | ---------------------------------------------------------------- |
| E-0     | (no-op)    | `prism-ecs-ir` was already in `compute-core/Cargo.toml`          |
| E-1     | `b93cb832` | `feat(constitutional): add prism-ecs-ir::evolution::receipts`    |
| E-2     | `62bd4064` | `chore(engine): migrate canonical/provenance.rs evolution import`|
| E-3     | `41ef907b` | `chore(engine): migrate cimage/generation_api.rs evolution import`|
| E-4     | `9cb1b9a9` | `chore(engine): migrate runtime/compilation_systems.rs evolution import`|
| E-5     | `7c07051c` | `chore(engine): migrate compiler/lifecycle_coordinator.rs evolution import`|
| E-6     | `0d05fbb6` | `chore(engine): drop evolution module declaration`               |
| E-7     | `b69f7901` | `feat(architecture): add evolution legacy-import safety net`     |
| E-8     | `8e32c87d` | `chore(engine): delete the legacy engine's evolution subsystem`  |

## Constitutional surface layout

The engine's evolution/ was a 5,581-LOC, 10-file subsystem that
exposed many internal types (`EvolveCandidate`, `CostMetrics`,
`EvolutionState`, `MilProgramFragment`, `JointEvolution`, etc.).
Audit of the engine's 4 external callers of
`crate::ecs::evolution::foundation::{NumericalReceipt, PerformanceReceipt}`
showed that only those two types are referenced outside the
engine evolution/ directory itself.

The constitutional home is two-layered:

1. **Existing** `crates/prism-ecs-ir/src/evolution/` (20 files,
   6,565 LOC) — the canonical re-implementation of the broader
   search pipeline (already constitutional before this
   migration). It carries its own `CandidateGenome`,
   `FitnessScore`, `JointEvolutionSystem`, `ParetoFrontier`, etc.
   that do **not** 1:1 correspond to engine types.
2. **New** `crates/prism-ecs-ir/src/evolution/receipts.rs` (added
   in E-1) — the canonical re-home of the two receipt types the
   engine callers actually consumed: `NumericalReceipt` and
   `PerformanceReceipt`. This file is a single-authority module
   (it owns exactly one authority: the receipt shapes for
   evolution-stage validation and measurement).

The single authority owned by `receipts.rs` is recorded in its
module doc, satisfying the "one authority per file" rule from
`AGENTS.md`.

## Engine callers migrated

Four engine files imported `NumericalReceipt` / `PerformanceReceipt`
from the engine evolution surface. All four were migrated to the
constitutional `prism_ecs_ir::evolution::receipts` module:

| File                                                           | Before                                                                  | After                                                                       |
| -------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `compute-core/src/ecs/canonical/provenance.rs:15`              | `use crate::ecs::evolution::foundation::{...}`                          | `use prism_ecs_ir::evolution::receipts::{...}`                              |
| `compute-core/src/ecs/cimage/generation_api.rs:15`             | `use crate::ecs::evolution::foundation::{...}`                          | `use prism_ecs_ir::evolution::receipts::{...}`                              |
| `compute-core/src/ecs/runtime/compilation_systems.rs:17`       | `use crate::ecs::evolution::foundation::{...}`                          | `use prism_ecs_ir::evolution::receipts::{...}`                              |
| `compute-core/src/ecs/compiler/lifecycle_coordinator.rs:27`    | `use crate::ecs::evolution::foundation::{...}`                          | `use prism_ecs_ir::evolution::receipts::{...}`                              |

The `lifecycle_coordinator.rs` was the only caller that
constructed a receipt with a non-empty `provenance` chain. The
mapping was `provenance_map.into_values().map(|p|
p.compiled_byte_digest).collect()` — engine-side
`ArtifactProvenance` reduced to a list of content-addressed
digests for the constitutional `Vec<String>` provenance.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-evolution` on branch
`migrate/evolution` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-evolution -b
migrate/evolution main`).

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-ir` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-ir` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria — final state

- ✅ All 10 files of `compute-core/src/ecs/evolution/` removed.
- ✅ Constitutional surface in
  `crates/prism-ecs-ir/src/evolution/` (with the receipts
  module added in E-1, single-authority module doc).
- ✅ All 4 engine callers migrated.
- ✅ `workspace_contains_no_legacy_evolution_imports` architecture
  test passes.
- ✅ `rg "use crate::ecs::evolution::" compute-core/src/` returns
  no results.
- ✅ Engine pre-existing build error count **decreased** from
  221 to 218 (3 errors removed with the deleted files; no new
  errors introduced). Still within the 243-error budget.
- ✅ Constitutional-side tests green (4 new receipts tests
  added; 82 pre-existing evolution tests still pass; 5
  architecture safety nets all pass).
