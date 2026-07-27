# Scheduling subsystem migration — inventory

**Date:** 2026-07-28
**Phase:** 0 of migration (inventory only; no code changes in this commit)
**Source:** `compute-core/src/ecs/scheduling/` (58 files, 16,236 LOC)
**Target architecture (per the corrected migration statement):**

```text
Components / resources        →  prism-ecs-runtime/src/scheduling/state/*
Systems (deterministic)     →  prism-ecs-runtime/src/scheduling/systems/*
Effects / adapters (FFI)     →  prism-ecs-kernel/src/backend/{metal,cuda,ane,rocm,accelerate}/*
Evidence (durable records)   →  prism-ecs-runtime/src/evidence/*
```

**No compatibility facade will be created.** Existing callers of `compute_core::ecs::scheduling::*` will be moved to the constitutional homes in the same commit as the data they call. The engine `lib.rs` will not re-export the migrated surface.

## Per-file classification

**Bucket key:** `C` = component/resource, `S` = system, `A` = effect/adapter (FFI), `E` = evidence, `X` = not an ECS element (test harness, module index, test fixture)

| # | File | LOC | Bucket | Destination | Notes |
|---:|---|---:|:---:|---|---|
| 1 | `mod.rs` | 298 | X | n/a (delete when subdirectory migrated) | Module index |
| 2 | `agent_bridge.rs` | 242 | S+A | `prism-ecs-runtime/src/scheduling/systems/agent_bridge.rs` (system logic); ANE-portion → `prism-ecs-kernel/src/backend/ane/agent_bridge.rs` | System bridges session → scheduling; FFI part is ANE |
| 3 | `ane_artifact_cache.rs` | 649 | A | `prism-ecs-kernel/src/backend/ane/artifact_cache.rs` | ANE artifact loading; `unsafe`; uses MLX arrays |
| 4 | `ane_lane_executor.rs` | 241 | A | `prism-ecs-kernel/src/backend/ane/lane_executor.rs` | ANE dispatch |
| 5 | `accelerate_lane_executor.rs` | 171 | A | `prism-ecs-kernel/src/backend/accelerate/lane_executor.rs` | Accelerate dispatch |
| 6 | `activation_arena.rs` | 73 | A | `prism-ecs-kernel/src/backend/metal/activation_arena.rs` | `unsafe`; activation memory |
| 7 | `activation_binding.rs` | 166 | A | `prism-ecs-kernel/src/backend/metal/activation_binding.rs` | `unsafe` |
| 8 | `activation_transaction.rs` | (in mod.rs) | C+A | `prism-ecs-runtime/src/scheduling/state/activation_transaction.rs` (state) + `prism-ecs-kernel/src/backend/metal/activation_transaction.rs` (FFI) | State + FFI |
| 9 | `backpressure.rs` | 691 | C+S | `prism-ecs-runtime/src/scheduling/state/backpressure.rs` (state) + `prism-ecs-runtime/src/scheduling/systems/backpressure.rs` (system) | Note: already partially absorbed in `crates/prism-ecs-runtime/src/schedule/backpressure.rs`; this engine file is the legacy duplicate |
| 10 | `batch.rs` | 153 | C | `prism-ecs-runtime/src/scheduling/state/batch.rs` | Batch descriptor |
| 11 | `benchmark_harness.rs` | 289 | X | `prism-ecs-runtime/tests/scheduling_benchmarks.rs` (constitutionally governed test, not a Component) | Test harness |
| 12 | `compilation_job_bridge.rs` | (in mod.rs) | S | `prism-ecs-runtime/src/scheduling/systems/compilation_job_bridge.rs` | Bridge scheduling ↔ compilation |
| 13 | `completion_bridge.rs` | 496 | S+A | `prism-ecs-runtime/src/scheduling/systems/completion_reconciliation.rs` (system); backend-completion-staging is the system, FFI stays engine-isolated | Reconciliation between backend completion and world state |
| 14 | `distributed_bridge.rs` | 159 | A | `prism-ecs-kernel/src/backend/distributed/bridge.rs` | Distributed bridge (likely unsafe + FFI) |
| 15 | `execution_context.rs` | 54 | C | `prism-ecs-runtime/src/scheduling/state/execution_context.rs` | Per-execution context |
| 16 | `execution_lease_bridge.rs` | 138 | S | `prism-ecs-runtime/src/scheduling/systems/execution_lease_bridge.rs` | Bridges leases to execution |
| 17 | `heterogeneous_executor.rs` | 933 | S+A | `prism-ecs-runtime/src/scheduling/systems/heterogeneous_orchestration.rs` (orchestration system); backend-specific lane executors in `prism-ecs-kernel/src/backend/*` | The Tokio actor + dispatch policy; backend parts to kernel |
| 18 | `ingress_bridge.rs` | 156 | A | `prism-ecs-kernel/src/backend/ingress/bridge.rs` | Ingress from outside the system |
| 19 | `kv_transaction.rs` | 305 | S | `prism-ecs-runtime/src/scheduling/systems/kv_transaction.rs` | Transaction system for KV cache coordination |
| 20 | `lane_capacity.rs` | 351 | C | `prism-ecs-runtime/src/scheduling/state/lane_capacity.rs` | Lane capacity resource |
| 21 | `lane_executors.rs` | 97 | A | `prism-ecs-kernel/src/backend/lane_executor_registry.rs` | Registry of lane executors |
| 22 | `lane_queue.rs` | 648 | C | `prism-ecs-runtime/src/scheduling/state/lane_queue.rs` | Note: already partially absorbed in `crates/prism-ecs-runtime/src/schedule/lane_queue.rs`; engine file is legacy duplicate |
| 23 | `lane_work.rs` | 244 | C | `prism-ecs-runtime/src/scheduling/state/lane_work.rs` | Per-lane work descriptor |
| 24 | `legacy_adapter.rs` | 241 | A | `prism-ecs-kernel/src/backend/legacy/adapter.rs` | Legacy MLX adapter |
| 25 | `memory_pool.rs` | (in mod.rs) | A | `prism-ecs-kernel/src/backend/metal/memory_pool.rs` | `unsafe` memory pool |
| 26 | `metal_decoder.rs` | 157 | A | `prism-ecs-kernel/src/backend/metal/decoder.rs` | Metal decoder dispatch |
| 27 | `metal_lane_executor.rs` | 142 | A | `prism-ecs-kernel/src/backend/metal/lane_executor.rs` | Metal lane executor |
| 28 | `phase_cancellation.rs` | 141 | C+S | `prism-ecs-runtime/src/scheduling/state/phase_cancellation.rs` (state) + `prism-ecs-runtime/src/scheduling/systems/phase_cancellation.rs` (system) | Cancellation state + the system that propagates it |
| 29 | `phase_engine.rs` | 558 | C+S | `prism-ecs-runtime/src/scheduling/state/phase_engine.rs` (state) + `prism-ecs-runtime/src/scheduling/systems/phase_advancement.rs` (system) | Phase DAG state + advancement system |
| 30 | `phase_invocation.rs` | 39 | C | `prism-ecs-runtime/src/scheduling/state/phase_invocation.rs` | Per-invocation state |
| 31 | `phase_readiness.rs` | 58 | S | `prism-ecs-runtime/src/scheduling/systems/phase_readiness.rs` | Readiness check system |
| 32 | `phase_runner/dispatch.rs` | 65 | S | `prism-ecs-runtime/src/scheduling/systems/dispatch_selection.rs` | Phase runner dispatch selection |
| 33 | `phase_runner/execution.rs` | 862 | S | `prism-ecs-runtime/src/scheduling/systems/orchestration.rs` | The phase-DAG executor system |
| 34 | `phase_runner/fallback.rs` | 17 | S | `prism-ecs-runtime/src/scheduling/systems/fallback.rs` | Fallback handling |
| 35 | `phase_runner/mod.rs` | 17 | X | n/a (delete when subdirectory migrated) | Module index |
| 36 | `prism_session.rs` | 658 | C+S | `prism-ecs-runtime/src/scheduling/state/session.rs` (state) + `prism-ecs-runtime/src/scheduling/systems/session.rs` (system) | Session state + the system that drives it |
| 37 | `ready_queue.rs` | 135 | C | `prism-ecs-runtime/src/scheduling/state/ready_queue.rs` | Ready set |
| 38 | `receipt.rs` | 618 | E | `prism-ecs-runtime/src/evidence/scheduling_receipts.rs` | ExecutionReceipt |
| 39 | `receipts.rs` | 64 | E | (consolidate into scheduling_receipts.rs) | Subset of receipt.rs |
| 40 | `request.rs` | 24 | C | `prism-ecs-runtime/src/scheduling/state/request.rs` | Request descriptor |
| 41 | `saved_request.rs` | 56 | C | `prism-ecs-runtime/src/scheduling/state/saved_request.rs` | Saved request state |
| 42 | `scheduler.rs` | 543 | S | `prism-ecs-runtime/src/scheduling/systems/scheduler.rs` | The scheduler system |
| 43 | `scheduler_metrics.rs` | 443 | E | `prism-ecs-runtime/src/evidence/scheduling_metrics.rs` | SchedulingMetrics |
| 44 | `slot.rs` | 21 | C | `prism-ecs-runtime/src/scheduling/state/slot.rs` | Slot descriptor (likely part of LeaseState) |
| 45 | `slot_lease_manager.rs` | 778 | S | `prism-ecs-runtime/src/scheduling/systems/lease_allocation.rs` | Lease allocation system |
| 46 | `token_budget.rs` | 260 | C+S | `prism-ecs-runtime/src/scheduling/state/token_budget.rs` (state) + `prism-ecs-runtime/src/scheduling/systems/token_budget.rs` (system) | Token budget enforcement |
| 47 | `tri_lane_orchestrator.rs` | 872 | S+A | `prism-ecs-runtime/src/scheduling/systems/tri_lane_orchestration.rs` (orchestration); backend lanes to `prism-ecs-kernel/src/backend/*` | Tri-lane orchestration + per-lane adapters |
| 48 | `unified_scheduler.rs` | 530 | S | `prism-ecs-runtime/src/scheduling/systems/unified_scheduler.rs` | Unified orchestration |
| 49 | `weight_residency.rs` | 130 | A | `prism-ecs-kernel/src/backend/metal/weight_residency.rs` | Weight residency (unsafe) |
| 50 | `work_registry/mod.rs` | 12 | X | n/a (delete) | Module index |
| 51 | `work_registry/scheduling.rs` | 34 | C | `prism-ecs-runtime/src/scheduling/state/work_registry.rs` | Work registry state |
| 52 | (other files referenced in mod.rs but not listed) | — | TBD | TBD | Read each before classifying |

**Files referenced in `mod.rs` not yet enumerated:** `activation_transaction.rs`, `compilation_job_bridge.rs`, `memory_pool.rs`, and any others. These will be inspected and classified in their own commits.

## Bucket totals (initial estimate)

- **Components / resources (C):** ~13 files, ~3,500 LOC → `prism-ecs-runtime/src/scheduling/state/`
- **Systems (S):** ~15 files, ~5,500 LOC → `prism-ecs-runtime/src/scheduling/systems/`
- **Effects / adapters (A):** ~17 files, ~4,300 LOC → `prism-ecs-kernel/src/backend/`
- **Evidence (E):** ~3 files, ~1,100 LOC → `prism-ecs-runtime/src/evidence/`
- **Not ECS (X):** ~4 files (mod.rs indices, benchmark harness) → deleted or moved to `tests/`

(Total adds to ~14,400 LOC + 1,800 LOC for files to be inspected; matches the 16,236 LOC delta.)

## Existing constitutional layer state to be reconciled

`crates/prism-ecs-runtime/src/schedule.rs` (2,747 LOC file) currently declares `pub mod backpressure;`, `pub mod lane_queue;`, `pub mod lane_slot_lease;` (referring to `schedule/` subdirectory with those 3 files, plus an untracked `phase_engine.rs` from a reverted prior attempt). This file is itself a partial godfile that needs decomposition as part of the migration. Treat as a sibling target: after the migration, the `schedule.rs` file will be replaced with `schedule/mod.rs` (a thin directory index), and the subdirectory's contents will be the new state/systems/evidence files.

`crates/prism-ecs-kernel/src/{metal_backend, accelerate_backend, cpu_backend, moe, metal_dispatch, kernel_generation}.rs` — existing hardware-adjacent modules. The new `prism-ecs-kernel/src/backend/{metal,cuda,ane,rocm,accelerate}/` subdirectories will live alongside, not replace, these.

## Migration commit plan (corrected, no compatibility facade)

1. **Inventory** — this commit
2. **Component move + callers** — move each C file to `state/`, update its callers in the same commit. Buildable after each sub-move.
3. **System move + callers** — same shape for S files into `systems/`.
4. **Adapter move + callers** — same shape for A files into `prism-ecs-kernel::backend::*`.
5. **Evidence move + callers** — same shape for E files into `evidence/`.
6. **Test harness / not-ECS items** — move `benchmark_harness.rs` to a test file. Delete mod indices.
7. **Test suite: `scheduling_transaction_boundary.rs`** — the 7 invariants.
8. **Test suite: `backend_effect_isolation.rs`** — the 4 invariants.
9. **Test suite: `scheduling_evidence.rs`** — the 4 invariants.
10. **Workspace architecture test: `workspace_contains_no_legacy_scheduling_imports`**
11. **Engine deletion** — `git rm compute-core/src/ecs/scheduling/` after all importers are migrated.

Each move commit is architecturally atomic: data moves AND its callers move AND `cargo check` is green. No temporary re-export shims.
