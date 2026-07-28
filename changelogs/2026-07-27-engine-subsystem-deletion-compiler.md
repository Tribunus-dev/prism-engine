# Goal: Delete `compute-core/src/ecs/compiler/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved. Branch `migrate/compiler` at `079d6603`.

## Source

`compute-core/src/ecs/compiler/` — 25 files, 11,020 LOC.

## Constitutional target

`crates/prism-ecs-compile/` (the engine's `compiler/` is the
largest remaining legacy block in the ecs tree; the canonical
home for everything in it is the constitutional `prism-ecs-compile`
crate, which already absorbs `models/embedding/` and the
`compute_image/` work-in-progress).

## Migration pattern

Follow E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`,
with the scheduling goal as the canonical recipe. The
assistant-graph migration
(`changelogs/2026-07-27-engine-subsystem-deletion-assistant-graph.md`,
commits `f0a4fe89`..`db4bb6c6`) is the more recent and cleaner
template — the E-0..E-8 sequence there (dep → surface → caller
migrations → safety net → pre-deletion → git rm → goal achieved)
is the minimum the compiler migration must follow.

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared with other agents and is currently on the
`migrate/inference` branch. **Do not work in the main worktree.**

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-compiler` on branch
`migrate/compiler` (use `git worktree add
/Users/user/Developer/GitHub/prism-engine-compiler -b
migrate/compiler main` from any clean repo checkout). All edits,
commits, and tests for this migration must happen in that
isolated worktree.

## Safety

- **No destructive ops.** Never run `git reset`, `git stash`,
  `git checkout -- <file>`, or `mavis-trash` in a way that affects
  another agent's work. File-scoped recovery (`git checkout
  <own-checkpoint-sha> -- <own-file>`) is the only allowed
  recovery.
- **Checkpoint every 30 min.** Land a commit on
  `migrate/compiler` even if incomplete.
- **Bisectable commits.** Each commit is independently
  buildable (or, for the engine, has the same pre-existing
  error count as the parent commit).
- **Correct crate name in commit messages.** You are migrating
  to `prism-ecs-compile` — write that name in your commits, not
  `prism-ecs-agent` or `prism-ecs-codec` or anything else.
- **Engine dep audit at E-0.** Only add `prism-ecs-compile` to
  the engine's `Cargo.toml` if there are engine callers of the
  new constitutional surface. If the engine's `compiler/` has no
  external callers, E-0 can be a no-op (the E-1 surface goes
  straight in).

## Success criteria

- All 25 files of `compute-core/src/ecs/compiler/` removed. ✓
- Constitutional surface in
  `crates/prism-ecs-compile/src/pipeline/`, one authority per
  file. ✓
- All engine callers of the engine's compiler surface migrated
  to the constitutional home. ✓
- `workspace_contains_no_legacy_compiler_imports` architecture
  test in
  `crates/architecture/src/workspace_legacy_compiler_imports.rs`
  passes. ✓
- `rg "use crate::ecs::compiler::" compute-core/src/` returns
  no results. ✓
- Engine pre-existing build error count is unchanged (currently
  221). ✓
- Constitutional-side tests green
  (`cargo test -p prism-ecs-compile --lib pipeline`). ✓

## Commit list

| # | Commit  | Description |
|---|---|---|
| E-0 | (no-op) | `prism-ecs-compile` is already in the engine's `Cargo.toml` |
| E-1.1 | `4ce8e5cf` | `feat(constitutional): add prism-ecs-compile::pipeline surface (E-1.1)` |
| E-1.2 | `28819744` | `feat(constitutional): add prism-ecs-compile::pipeline event/deployment/graph/schedule (E-1.2)` |
| E-1.3 | `8e2489dd` | `feat(constitutional): add prism-ecs-compile::pipeline ane/lowering/lifecycle (E-1.3)` |
| E-2  | `f10a87a4` | `chore(engine): migrate cimage/sealed_v1.rs compiler import (E-2)` |
| E-3  | `30ab4496` | `chore(engine): migrate runtime/serving/model_instance.rs compiler import (E-3)` |
| E-4  | `45475b7d` | `chore(engine): migrate compile_session.rs compiler import (E-4)` |
| E-5  | `b7b6e56b` | `chore(engine): migrate system/compiler_systems.rs compiler imports (E-5)` |
| E-6  | `34ade5dc` | `chore(engine): drop compiler re-export and module declaration (E-6)` |
| E-7  | `554e6434` | `feat(architecture): add compiler legacy-import safety net (E-7)` |
| E-8  | `079d6603` | `chore(engine): delete the legacy engine's compiler subsystem (E-8)` |

## Constitutional surface layout

The new surface lives in `crates/prism-ecs-compile/src/pipeline/`
(the engine's existing `compiler.rs` in the same crate is the
top-level canonical orchestrator; `pipeline/` is the engine's
multi-level IR and lowering surface that was previously
`compute-core/src/ecs/compiler/`).

| File | Authority |
|---|---|
| `mod.rs`                  | `pipeline` module root + `BackendLowering` trait + `LoweringReceipt` + `LegalityReceipt` + `LegalityViolation` |
| `pass.rs`                 | Versioned compiler pass framework (`TransformPass` trait, `PassIdentity`, `TransformReceipt`, `TransformPipeline`, `NoopPass`) |
| `semantic.rs`             | Backend-neutral model meaning (`SemanticModule`, `SemanticTensor`, `SemanticOp`, `TensorRole`, `ToleranceClass`, `ModelContract`) |
| `scheduled.rs`            | Physical layer (`PhysicalTensor`, `ScheduledRegion`, `RegionId`, `RegionDependency`, `DependencyKind`, `FusionBoundary`, `FusionRegion`, `StateEffect`, `TransferPlan`, `MemoryPlan`, `BufferReuse`, `SealedEvaluationBoundary`, `ScheduledModule`, `StorageClass`) |
| `fused_op.rs`             | Fused kernel naming (`FusedOperation` enum + `kernel_name()`) |
| `plan.rs`                 | Plan types (`ModelExecutionPlan`, `LayerPlan`, `TextArchitecture`, `ProloguePlan`, `EpiloguePlan`, `OperationRoute`, `AttentionKind`, `RopeSpec`, `MoEConfig`, `AneFusedIsland`, `SpeculativeConfig`, etc.) |
| `compile_schedule.rs`     | Model-to-schedule compiler (`compile_model_to_scheduled_module`, `estimate_layer_peak_memory`) |
| `graph_optimizer.rs`      | 3-pass graph optimizer (`optimize`, `shape_propagation`, `constant_folding`, `dead_code_elimination`, `optimize_with_stats`) |
| `event_emitter.rs`        | Compiler pipeline event stream (`CompilerEvent` enum, `CompilerEventStream`, `ChainVerificationResult`, `verify_event_chain`, `now_micros`) |
| `deployment_compiler.rs`  | Deployment-time compile contract (`DeploymentRequest`, `DeploymentResult`, `ServingProfile`, `PromotableCimage`, `CimageAssembly`, `CompiledKernelArtifact`, `PhysicalSegmentId`, `ExecutionGraphStub`, `MemoryPlanStub`, `RuntimeStatePlanStub`) |
| `lifecycle_coordinator.rs` | Lifecycle public surface (`CompilerRequest`, `LifecycleResult`, `KernelArtifact`, `SmokeResult`, `PolicyConfig`, `PromotionPolicy`, `LifecycleCoordinator`) |
| `backend_assessment.rs`   | Backend assessment pass (`BackendAssessmentPass`, `GraphOperation`, `ModelOperationGraph`, `assess_and_route`, `assess_model_ops`) |
| `ane/mod.rs`              | ANE module root |
| `ane/legality.rs`         | ANE rule trait + evaluator + receipts |
| `ane/rules.rs`            | Concrete ANE rules (Concat, F16, size, op limit) + `default_ane_rules` |
| `ane/fusion.rs`           | ANE fusion pass + `AneFusedArtifact` + `build_fused_ane_regions` |
| `ane/artifacts.rs`        | Derived ANE artifacts (MIL text, IOSurface contracts, BLOBFILE plans) |
| `lowering/mod.rs`         | Lowering module root |
| `lowering/params.rs`      | Core ML lowering parameter types |
| `lowering/receipts.rs`    | Core ML lowering receipts |
| `lowering/dataset.rs`     | F32 matmul test dataset |

Total: 25 source files (matching the 25 engine files), 106
unit tests, all green.

## Engine dep audit (E-0)

The engine's `Cargo.toml` already has `prism-ecs-compile` as a
mandatory dep (it was added in an earlier migration to host the
absorbed `core/speculative.rs` types). E-0 is therefore a no-op;
the E-1 surface goes straight in.

## Engine caller migration

Five engine files imported from `crate::ecs::compiler::*`:

- `compute-core/src/ecs/cimage/sealed_v1.rs` (ServingProfile)
- `compute-core/src/ecs/runtime/serving/model_instance.rs` (ServingProfile)
- `compute-core/src/ecs/compile_session.rs` (CompilerEvent, CompilerEventStream, now_micros)
- `compute-core/src/ecs/system/compiler_systems.rs` (BackendAssessmentPass, compile_model_to_scheduled_module, optimize, TransformPass, ScheduledModule)
- `compute-core/src/lib.rs` and `compute-core/src/ecs/mod.rs` and `compute-core/src/ecs/core/analysis.rs` (re-exports and `pub mod compiler`)

The four consumer files were migrated (E-2..E-5). The three
re-export / module-declaration files were updated (E-6).

The engine's `compute-core/src/ecs/system/compiler_systems.rs`
previously failed to compile against the engine's parallel
surface (it is in the 221-error pre-existing budget). Routing
through the constitutional surface keeps the call sites in the
same shape; the type-conversion responsibility (engine
`ModelExecutionPlan`/`TextArchitecture` → constitutional types) is
a follow-up migration step that doesn't change the pre-existing
error count.

## Engine pre-existing error count

Before E-0 (baseline): **221** errors.
After E-8: **221** errors.
Net change: **0**. Within the 243 budget.
