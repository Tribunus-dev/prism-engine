# Goal: Delete `compute-core/src/ecs/evaluator/`

**Date:** 2026-07-27 (Pacific)
**Status:** **Goal achieved.**
**Follow-up to:** `f2cfee80` (scheduling engine-deletion goal achieved).

## Source

`compute-core/src/ecs/evaluator/` — 10 files, 487 LOC.

## Constitutional target

`crates/prism-ecs-codec/src/evaluator/` (submodule of the existing
`prism-ecs-codec` crate). The evaluation surface is about
backend-neutral evaluation of codec-correct candidates (NF4, ternary,
INT8, FP16 fixtures; backend evaluator trait; immutable evidence;
admission decisions), which is a natural extension of the codec crate's
existing role around codec-correct serialization.

## Migration pattern

Followed a simplified E-1..E-5 pattern (E-0 and E-1..E-13 collapsed
because the engine's evaluator/ has **no external callers** —
verified by `rg "use crate::ecs::evaluator::" compute-core/src/`
returning zero results before any change). The only references to
the engine's evaluator/ were `pub mod evaluator;` and the
re-export block in `compute-core/src/ecs/mod.rs`, plus the engine's
own self-references inside `compute-core/src/ecs/evaluator/`.

| Step | Commit | Description |
|------|--------|-------------|
| E-1 | `a4193731` | Re-implement the engine's evaluator/ as `prism-ecs-codec::evaluator` (11 files, 30 unit tests) |
| E-2 | `8ce43315` | Add `workspace_contains_no_legacy_evaluator_imports` architecture test |
| E-3 | (folded into E-4+E-5) | Remove `pub mod evaluator;` and the re-export block from `compute-core/src/ecs/mod.rs` |
| E-4 + E-5 | `eba73bc3` | Pre-deletion verification + `git rm -r compute-core/src/ecs/evaluator/` |

## Safety

- Worked on branch `migrate/evaluator` (not main).
- Checkpoint commits per step.
- No destructive ops; file-scoped recovery only.

## Success criteria — all met

- `rg "use crate::ecs::evaluator::" compute-core/src/` returns no results. ✓
- `git rm -r compute-core/src/ecs/evaluator/` committed. ✓
- Engine pre-existing build error count unchanged: **221** (baseline). ✓
- Constitutional surface tests pass: **34/34** (4 pre-existing + 30 new). ✓
- Architecture tests pass: **2/2** (scheduling + evaluator safety nets). ✓
- No `legacy_mutations`-style escape hatch; no compatibility facade. ✓

## Result

The engine's `compute-core/src/ecs/evaluator/` is deleted. The
constitutional surface at `prism-ecs-codec::evaluator` is the only
home for the backend-neutral evaluation surface. The engine has
its 221 pre-existing build errors, none of which are
evaluator-related. The `compute-core.legacy/src/ecs/evaluator/`
archaeology snapshot is preserved (not part of the workspace
build).

## Files

### Constitutional surface (new, `crates/prism-ecs-codec/src/evaluator/`)

- `mod.rs` — module index (re-exports + single-authority doc)
- `kernel_abi.rs` — `KernelAbi` placeholder (full `KernelAbi` is
  engine-local at `compute-core/src/ecs/canonical/kernel_abi.rs`;
  this is the minimal subset the evaluator needs)
- `generated_executable.rs` — `GeneratedExecutable` (backend-neutral
  identity of one executable)
- `fixture.rs` — `EvaluationFixture` (Nf4, Ternary, Int8, Fp16)
- `binding_plan.rs` — `BindingPlan`, `BindingSlot`, `ConstantSlot`
- `backend_trait.rs` — `BackendEvaluator` trait, `EvaluationConfig`,
  `EvaluationError`, `TemperaturePolicy`
- `artifact.rs` — `BackendArtifact` (Metal, Ane, Accelerate, FutureNpu)
- `receipts.rs` — `EvaluationReceiptBundle` + 8 receipt types
- `role.rs` — `EvaluationRole` (Oracle, Candidate, PlanarTransform,
  CrossCheck, Replay)
- `admission.rs` — `AdmissionDecision` (Admitted, Rejected, Deferred)
- `system.rs` — `HeterogeneousEvaluatorSystem`, `AdmissionPolicy`

### Architecture test (new, `crates/architecture/src/`)

- `workspace_legacy_evaluator_imports.rs` — safety net test
  (mirrors the scheduling safety net)

### Engine (deleted)

- `compute-core/src/ecs/evaluator/` — 10 files, 487 LOC
- `compute-core/src/ecs/mod.rs` — `pub mod evaluator;` and
  re-export block removed

## Notes for the next migration

- The `KernelAbi` placeholder in `prism-ecs-codec::evaluator` is a
  minimal subset. The full `KernelAbi` lives at
  `compute-core/src/ecs/canonical/kernel_abi.rs` (305 LOC) and is
  engine-local. A future migration could move the full `KernelAbi`
  to the constitutional side (likely `prism-ecs-kernel`).
- The `prism-ecs-codec` crate now has two top-level surfaces:
  tensor serialization (existing) and backend-neutral evaluation
  (new). The `prism-ecs-codec::evaluator` re-exports are at the
  submodule level, not the crate level, to keep the tensor API
  clean.

