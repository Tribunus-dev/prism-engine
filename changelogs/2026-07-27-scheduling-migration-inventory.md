# Scheduling subsystem migration — complete authority inventory

**Date:** 2026-07-27 (Pacific) / 2026-07-27 (UTC, per uploaded file metadata)
**Phase:** 0 of migration (inventory only; no code changes in this commit)
**Follow-up to:** `fa62f99c` (which is now corrected by this document)
**Source:** `compute-core/src/ecs/scheduling/` — 58 files, 16,236 LOC
**Callers (current engine, to be migrated alongside):** `compute-core/src/ecs/compiler/{lifecycle_coordinator,deployment_compiler}.rs`
**`compute-core/compute-core.legacy/`** is a frozen archaeology snapshot; its imports of `compute_core::ecs::scheduling::*` are NOT migrated.

**No compatibility facade.** All callers move with the data they call. The engine `lib.rs` will not re-export the migrated surface.

## Refinements vs. `fa62f99c`

1. **Date corrected** (file renamed; July 28 → July 27).
2. **All 58 files enumerated** with exact LOC, callers, destination, and survival disposition. No catch-all row.
3. **`phase_engine.rs` vocabulary corrected** — `state/phase_engine.rs` was carrying the old "engine" ownership model into the destination. New destinations: `state/phase.rs`, `systems/phase_advancement.rs`, dispatch-selection pieces go with the `dispatch_selection` system. The word "engine" is removed from destination paths.
4. **`prism_session.rs` decomposed by authority**, not moved as an aggregate object:
   - Scheduling-local session state (lane assignments, active phase, pending dispatches, scheduling-local lifecycle) → `prism-ecs-runtime`
   - Connection / request / client / authentication / server-lifecycle → `prism-ecs-server` (existing)
   - Stable session identifiers and shared wire-level session metadata → `prism-ecs-protocol`
5. **`heterogeneous_executor.rs` split during absorption**:
   - Runtime side: dispatch policy, placement, ordering, conversion of committed scheduling state → backend-neutral dispatch request
   - Kernel side: submission, device sync, executor handles, completion production
   - Runtime emits a dispatch command; kernel consumes it; kernel returns a completion value; runtime reconciliation stages the accepted transition through the world transaction
6. **Adapter bucket narrowed** — "every adapter with FFI" was too broad. The adapter bucket now means **hardware backends and execution-plane adapters**:
   - Hardware submission, streams, buffers, device synchronization, hardware FFI → `prism-ecs-kernel::backend::*`
   - Network ingress → its actual protocol/server boundary (not kernel)
   - Distributed coordination → runtime port or protocol crate (not kernel by default)
   - Compilation bridges → their compile/runtime boundary
   - Product integration → its server/compile boundary
7. **`ingress_bridge.rs` reclassified** — moved out of kernel; "ingress from outside the system" is not a hardware backend. Placed in `prism-ecs-runtime::ports::ingress` (a port the constitutional runtime exposes).
8. **`distributed_bridge.rs` reclassified** — needs evidence before kernel placement. If the file wraps RDMA / NCCL / raw sockets, it goes in `prism-ecs-kernel::backend::distributed`. Otherwise it is a runtime port. TBD on inspection.
9. **`completion_bridge.rs` contradiction removed** — "FFI stays engine-isolated" was inconsistent with the migration promise to delete the engine subsystem. Reworded: kernel produces a typed, non-authoritative completion value; runtime reconciliation validates it and stages the accepted transition through the world transaction.
10. **Evidence bucket refined** — "durable records" overstated. Split into:
    - **Admitted evidence** (`ExecutionReceipt`) — durable, projection-rebuildable, replay-respects
    - **Advisory metrics** (`SchedulingMetrics`) — not automatically durable, not automatically evidence; treat as a rebuildable projection until a measurement is admitted into an immutable receipt
11. **Single post-migration scheduling authority map** — each of 9 scheduling decisions has exactly one canonical owner.
12. **Reconciliation matrix for every existing target-side duplicate** — each of the 4 existing `schedule/*` files and the 6 existing `prism-ecs-kernel::*` modules has an explicit `survive / merge-into-X / replace / delete` disposition.
13. **Commit order is dependency-driven**, not bucket-driven. Each slice runs the focused crate tests AND the affected integration tests, not just `cargo check`.
14. **Tests move with each slice**, not postponed. The new invariant suites accumulate incrementally.

## The four-bucket classification (final)

- **C** = component / resource (authoritative data on entities or process-wide). Staged through `ConstitutionalWorldTxn`.
- **S** = system (deterministic behavior, transforms state). Reads staged + committed state, stages its own mutations through `ConstitutionalWorldTxn`.
- **A** = adapter (hardware FFI, execution-plane side-effects, unsafe implementation). Lives in `prism-ecs-kernel::backend::*` (for hardware) or in the appropriate protocol/server/compile/runtime boundary (for non-hardware external effects). **Does NOT stage mutations of authoritative world state** — it performs the side effect; the completion is data until runtime reconciliation stages it.
- **E** = evidence (admitted, durable, replay-respects). Lives in `prism-ecs-runtime::evidence::*`. A receipt is emitted only for a committed transaction.
- **M** = advisory metric (rebuildable projection, not evidence, not durable). Lives with the system that produces it; projected into a `prism-ecs-runtime::metrics::*` snapshot.

## The single post-migration scheduling authority map

Each of the 9 scheduling decisions has **exactly one** canonical owner. Overlapping schedulers (`scheduler.rs`, `unified_scheduler.rs`, `tri_lane_orchestrator.rs`, `heterogeneous_executor.rs`, `phase_runner/execution.rs`) are reconciled during absorption so the post-migration code has one owner per decision.

| Decision | Owner (post-migration) | Source |
|---|---|---|
| Admission | `prism-ecs-runtime::scheduling::systems::admission` | new; from `lane_queue.rs` + `ready_queue.rs` + parts of `prism_session.rs` |
| Readiness | `prism-ecs-runtime::scheduling::systems::phase_readiness` | new; from `phase_readiness.rs` + `phase_invocation.rs` |
| Batching | `prism-ecs-runtime::scheduling::systems::batching` | new; from `batch.rs` + parts of `lane_queue.rs` |
| Placement | `prism-ecs-runtime::scheduling::systems::placement` | new; from `lane_capacity.rs` + parts of `lane_work.rs` |
| Lease assignment | `prism-ecs-runtime::scheduling::systems::lease_allocation` | new; from `slot_lease_manager.rs` |
| Phase advancement | `prism-ecs-runtime::scheduling::systems::phase_advancement` | new; from `phase_engine.rs` system half + `phase_runner/execution.rs` |
| Dispatch selection | `prism-ecs-runtime::scheduling::systems::dispatch_selection` | new; from `phase_runner/dispatch.rs` + `tri_lane_orchestrator.rs` dispatch logic |
| Submission | `prism-ecs-kernel::backend::*::dispatch` (per backend) | new; from `heterogeneous_executor.rs` submission half + `metal_lane_executor.rs` + `ane_lane_executor.rs` + `accelerate_lane_executor.rs` |
| Completion reconciliation | `prism-ecs-runtime::scheduling::systems::completion_reconciliation` | new; from `completion_bridge.rs` + `work_lifecycle_bridge.rs` |

## Per-file classification (all 58)

Legend: bucket `C` / `S` / `A` / `E` / `M` / `X` (X = not-ECS, constitutionally governed). `t` = touched by current engine callers. `+` = current canonical layer has a partial equivalent.

| # | File | LOC | Bucket | Destination | Current callers (engine) | Target-side equivalent (today) | Disposition |
|---:|---|---:|---|---|---|---|---|
| 1 | `mod.rs` | 298 | X | (deleted) | n/a (self) | n/a | **delete** when subdirectory migrated |
| 2 | `accelerate_lane_executor.rs` | 171 | A | `prism-ecs-kernel::backend::accelerate::lane_executor.rs` | n/a | `prism-ecs-kernel::accelerate_backend.rs` (210) | **merge** into `backend::accelerate::lane_executor`; delete `accelerate_backend.rs` |
| 3 | `activation_arena.rs` | 73 | A | `prism-ecs-kernel::backend::metal::activation_arena.rs` | n/a | n/a | **move** (no equivalent) |
| 4 | `activation_binding.rs` | 166 | A | `prism-ecs-kernel::backend::metal::activation_binding.rs` | n/a | n/a | **move** (no equivalent) |
| 5 | `activation_transaction.rs` | 173 | C+A | `prism-ecs-runtime::scheduling::state::activation_transaction.rs` (state) + `prism-ecs-kernel::backend::metal::activation_transaction.rs` (FFI half) | n/a | n/a | **split** |
| 6 | `agent_bridge.rs` | 242 | S+A | `prism-ecs-runtime::scheduling::systems::agent_bridge.rs` (logic) + `prism-ecs-kernel::backend::ane::agent_bridge.rs` (ANE FFI half) | n/a | n/a | **split** |
| 7 | `ane_artifact_cache.rs` | 649 | A | `prism-ecs-kernel::backend::ane::artifact_cache.rs` | `phase_telemetry.rs` | n/a | **move** (no equivalent — too ANE-specific for runtime) |
| 8 | `ane_lane_executor.rs` | 241 | A | `prism-ecs-kernel::backend::ane::lane_executor.rs` | n/a | n/a | **move**; create new `backend::ane` directory |
| 9 | `backpressure.rs` | 691 | C+S | `prism-ecs-runtime::scheduling::state::backpressure.rs` (state) + `prism-ecs-runtime::scheduling::systems::backpressure.rs` (system) | n/a | `prism-ecs-runtime::schedule::backpressure.rs` (1,077) | **survive with re-implementation**: target-side `schedule::backpressure` is the canonical home; engine file is the legacy duplicate. Engine file **delete**, target-side survives. |
| 10 | `batch.rs` | 153 | C | `prism-ecs-runtime::scheduling::state::batch.rs` | n/a | n/a | **move** |
| 11 | `benchmark_harness.rs` | 289 | X | `prism-ecs-runtime/tests/scheduling_benchmarks.rs` | n/a | n/a | **move** (test file, not a Component) |
| 12 | `compilation_job_bridge.rs` | 215 | S | `prism-ecs-runtime::scheduling::systems::compilation_job_bridge.rs` | n/a | n/a | **move** |
| 13 | `completion_bridge.rs` | 496 | S | `prism-ecs-runtime::scheduling::systems::completion_reconciliation.rs` | n/a | n/a | **move**; FFI part goes to `prism-ecs-kernel::backend::*::completion_consumer` (a typed non-authoritative completion value) |
| 14 | `distributed_bridge.rs` | 159 | S or A | `prism-ecs-runtime::ports::distributed` (TBD on inspection) | n/a | n/a | **inspect for raw RDMA/NCCL/socket FFI**; if present → `prism-ecs-kernel::backend::distributed::bridge`; else → runtime port |
| 15 | `execution_context.rs` | 54 | C | `prism-ecs-runtime::scheduling::state::execution_context.rs` | n/a | n/a | **move** |
| 16 | `execution_lease_bridge.rs` | 138 | S | `prism-ecs-runtime::scheduling::systems::execution_lease_bridge.rs` | n/a | n/a | **move** |
| 17 | `heterogeneous_executor.rs` | 933 | S+A | runtime half: `prism-ecs-runtime::scheduling::systems::heterogeneous_orchestration.rs`; kernel half: `prism-ecs-kernel::backend::dispatcher::heterogeneous.rs` | `phase_engine.rs` | n/a | **split during absorption** (Tokio actor lives in kernel, dispatch-policy logic lives in runtime) |
| 18 | `ingress_bridge.rs` | 156 | A (port) | `prism-ecs-runtime::ports::ingress.rs` | n/a | n/a | **move** (NOT kernel; this is a runtime port, not a hardware backend) |
| 19 | `kv_transaction.rs` | 305 | S | `prism-ecs-runtime::scheduling::systems::kv_transaction.rs` | n/a | n/a | **move** |
| 20 | `lane_capacity.rs` | 351 | C | `prism-ecs-runtime::scheduling::state::lane_capacity.rs` | `lane_queue.rs`, `unified_scheduler.rs` | n/a | **move** (first implementation slice) |
| 21 | `lane_executors.rs` | 97 | A | `prism-ecs-kernel::backend::lane_executor_registry.rs` | n/a | n/a | **move** (registry of backend executors) |
| 22 | `lane_queue.rs` | 648 | C | `prism-ecs-runtime::scheduling::state::lane_queue.rs` | `phase_engine.rs` | `prism-ecs-runtime::schedule::lane_queue.rs` (883) | **survive with re-implementation**: target-side is canonical; engine file **delete** as legacy duplicate |
| 23 | `lane_work.rs` | 244 | C | `prism-ecs-runtime::scheduling::state::lane_work.rs` | n/a | n/a | **move** |
| 24 | `legacy_adapter.rs` | 241 | A | `prism-ecs-kernel::backend::legacy::adapter.rs` | n/a | n/a | **move** |
| 25 | `memory_pool.rs` | 195 | A | `prism-ecs-kernel::backend::metal::memory_pool.rs` | n/a | n/a | **move** |
| 26 | `metal_decoder.rs` | 157 | A | `prism-ecs-kernel::backend::metal::decoder.rs` | n/a | n/a | **move** |
| 27 | `metal_lane_executor.rs` | 142 | A | `prism-ecs-kernel::backend::metal::lane_executor.rs` | n/a | n/a | **move** |
| 28 | `outlier_detector.rs` | 234 | M | `prism-ecs-runtime::scheduling::metrics::outlier_detector.rs` | n/a | n/a | **move** (advisory metric, not evidence) |
| 29 | `phase_cancellation.rs` | 141 | C+S | `prism-ecs-runtime::scheduling::state::phase_cancellation.rs` (state) + `prism-ecs-runtime::scheduling::systems::phase_cancellation.rs` (system) | n/a | n/a | **move** |
| 30 | `phase_engine.rs` | 558 | C+S | `prism-ecs-runtime::scheduling::state::phase.rs` (state) + `prism-ecs-runtime::scheduling::systems::phase_advancement.rs` (system) | `lane_queue.rs`, `phase_runner/execution.rs` | `prism-ecs-runtime::schedule::phase_engine.rs` (1,063, untracked from prior attempt) | **split during absorption; rename to drop "engine" from destination paths**; the prior `schedule::phase_engine` is the surviving target-side |
| 31 | `phase_engine_state.rs` | 212 | C | `prism-ecs-runtime::scheduling::state::phase_engine_state.rs` (or merge into `state::phase.rs`) | n/a | n/a | **move** (likely merge into `state::phase.rs`) |
| 32 | `phase_invocation.rs` | 39 | C | `prism-ecs-runtime::scheduling::state::phase_invocation.rs` | n/a | n/a | **move** |
| 33 | `phase_readiness.rs` | 58 | S | `prism-ecs-runtime::scheduling::systems::phase_readiness.rs` | n/a | n/a | **move** |
| 34 | `phase_runner/dispatch.rs` | 65 | S | `prism-ecs-runtime::scheduling::systems::dispatch_selection.rs` | n/a | n/a | **move** (renamed for consistency) |
| 35 | `phase_runner/execution.rs` | 862 | S | `prism-ecs-runtime::scheduling::systems::orchestration.rs` (or `phase_advancement.rs`) | `phase_engine.rs` | n/a | **move**; parts may merge with `phase_advancement.rs` |
| 36 | `phase_runner/fallback.rs` | 17 | S | `prism-ecs-runtime::scheduling::systems::fallback.rs` | n/a | n/a | **move** |
| 37 | `phase_runner/mod.rs` | 17 | X | (deleted) | n/a | n/a | **delete** when subdirectory migrated |
| 38 | `phase_telemetry.rs` | 202 | M | `prism-ecs-runtime::scheduling::metrics::phase_telemetry.rs` | `ane_artifact_cache.rs` (read-only) | n/a | **move** (advisory metric) |
| 39 | `pipeline_bridge.rs` | 180 | S | `prism-ecs-runtime::scheduling::systems::pipeline_bridge.rs` | n/a | n/a | **move** |
| 40 | `prefill_orchestrator.rs` | 182 | S | `prism-ecs-runtime::scheduling::systems::prefill_orchestration.rs` | n/a | n/a | **move** |
| 41 | `prism_session.rs` | 658 | C+S (decomposed) | decomposed by authority: scheduling-local state → `prism-ecs-runtime::scheduling::state::session`; scheduling session system → `prism-ecs-runtime::scheduling::systems::session`; connection / request / client / auth / server lifecycle → `prism-ecs-server` (existing); stable session ID + wire metadata → `prism-ecs-protocol` | n/a | `prism-ecs-server::runtime::server::session_lifecycle` (already partially absorbed) | **split during absorption; do NOT move the aggregate** |
| 42 | `ready_queue.rs` | 135 | C | `prism-ecs-runtime::scheduling::state::ready_queue.rs` | n/a | n/a | **move** |
| 43 | `receipt.rs` | 618 | E | `prism-ecs-runtime::evidence::scheduling_receipts.rs` | `phase_telemetry.rs` | `prism-ecs-runtime::engine_receipts.rs` (canonical receipt shape lives here) | **merge**: scheduling receipts are a specialization of `engine_receipts`. `engine_receipts.rs` survives; the scheduling-receipt types become subtypes / variants. |
| 44 | `receipts.rs` | 64 | E | (consolidate into `scheduling_receipts.rs`) | n/a | n/a | **merge** into `scheduling_receipts.rs`; delete |
| 45 | `request.rs` | 24 | C | `prism-ecs-runtime::scheduling::state::request.rs` | n/a | n/a | **move** |
| 46 | `saved_request.rs` | 56 | C | `prism-ecs-runtime::scheduling::state::saved_request.rs` | n/a | n/a | **move** |
| 47 | `scheduler.rs` | 543 | S | `prism-ecs-runtime::scheduling::systems::scheduler.rs` (or merge into `unified_scheduler.rs`) | n/a | n/a | **merge** with `unified_scheduler.rs`; one canonical scheduler survives |
| 48 | `scheduler_metrics.rs` | 443 | M | `prism-ecs-runtime::scheduling::metrics::scheduler_metrics.rs` | n/a | n/a | **move** (advisory metric, rebuildable projection) |
| 49 | `slot.rs` | 21 | C | `prism-ecs-runtime::scheduling::state::slot.rs` (or merge into `state::lease.rs`) | n/a | n/a | **merge** into `state::lease.rs`; delete |
| 50 | `slot_lease_manager.rs` | 778 | S | `prism-ecs-runtime::scheduling::systems::lease_allocation.rs` | n/a | n/a | **move** |
| 51 | `token_budget.rs` | 260 | C+S | `prism-ecs-runtime::scheduling::state::token_budget.rs` (state) + `prism-ecs-runtime::scheduling::systems::token_budget.rs` (system) | n/a | n/a | **move** |
| 52 | `tri_lane_orchestrator.rs` | 872 | S (orchestration logic) + A (per-lane parts to kernel) | runtime orchestration → `prism-ecs-runtime::scheduling::systems::tri_lane_orchestration.rs`; per-lane parts → `prism-ecs-kernel::backend::{metal,cuda,ane,accelerate}::lane_executor.rs` | n/a | n/a | **split during absorption** |
| 53 | `unified_scheduler.rs` | 530 | S | `prism-ecs-runtime::scheduling::systems::unified_scheduler.rs` (the surviving canonical scheduler) | n/a | n/a | **survive**; absorbs `scheduler.rs` (merge) |
| 54 | `weight_residency.rs` | 130 | A | `prism-ecs-kernel::backend::metal::weight_residency.rs` | n/a | n/a | **move** |
| 55 | `work_lifecycle_bridge.rs` | 401 | S | `prism-ecs-runtime::scheduling::systems::work_lifecycle_bridge.rs` | n/a | n/a | **move** |
| 56 | `work_registry/mod.rs` | 12 | X | (deleted) | n/a | n/a | **delete** when subdirectory migrated |
| 57 | `work_registry/registry.rs` | 213 | C | `prism-ecs-runtime::scheduling::state::work_registry.rs` | n/a | n/a | **move** |
| 58 | `work_registry/scheduling.rs` | 34 | C | (consolidate into `state::work_registry.rs`) | n/a | n/a | **merge** into `state::work_registry.rs`; delete |

## Bucket totals (refined)

- **Components (C):** 19 files, ~3,800 LOC → `prism-ecs-runtime/src/scheduling/state/`
- **Systems (S):** 21 files, ~6,200 LOC → `prism-ecs-runtime/src/scheduling/systems/`
- **Adapters (A, hardware backends only):** 12 files, ~3,400 LOC → `prism-ecs-kernel/src/backend/`
- **Adapters (A, runtime ports, non-hardware):** 2 files (`ingress_bridge`, possibly `distributed_bridge`) → `prism-ecs-runtime/src/ports/`
- **Evidence (E, admitted durable):** 1 file (`receipt.rs`, partial merge into `engine_receipts`) → `prism-ecs-runtime::evidence::scheduling_receipts`
- **Advisory metrics (M):** 2 files (`outlier_detector`, `scheduler_metrics`, `phase_telemetry`) → `prism-ecs-runtime::scheduling::metrics/`
- **Not-ECS (X):** 4 files (mod.rs indices, benchmark harness) → deleted or moved to `tests/`

(Total ~16,000 LOC + reconciliations.)

## Reconciliation matrix — existing target-side code

Each of the existing target-side implementations has an explicit `survive / merge-into / replace / delete` disposition. The migration will not leave overlapping authority between engine-side and target-side.

### `crates/prism-ecs-runtime/src/schedule.rs` and `schedule/` directory

| Existing file | LOC | Disposition | Notes |
|---|---:|---|---|
| `schedule.rs` (file) | 2,747 | **delete** | After the migration, the file is replaced with `schedule/mod.rs` (a thin directory index). The 2,747 LOC become the new `scheduling/{state,systems}/` modules. |
| `schedule/backpressure.rs` | 1,077 | **survive** | Canonical home for backpressure state + system. The engine `backpressure.rs` is the legacy duplicate. |
| `schedule/lane_queue.rs` | 883 | **survive** | Canonical home for lane queue. Engine file is the legacy duplicate. |
| `schedule/lane_slot_lease.rs` | 943 | **survive** | Canonical home for lease state. Engine file is the legacy duplicate. |
| `schedule/phase_engine.rs` (untracked from prior attempt) | 1,063 | **re-implement + rename** | The prior attempt was on the wrong vocabulary (used "engine" in the destination). Re-implement as `state/phase.rs` + `systems/phase_advancement.rs` and delete this untracked file. |

### `crates/prism-ecs-kernel/src/*` modules

| Existing file | LOC | Disposition | Notes |
|---|---:|---|---|
| `metal_backend.rs` | 210 | **delete** after the per-backend modules under `backend::metal::*` are created and absorb the behavior. |
| `metal_dispatch.rs` | 1,821 | **delete** after `backend::metal::dispatch` is created. |
| `accelerate_backend.rs` | 190 | **delete** after `backend::accelerate::dispatch` is created. |
| `cpu_backend.rs` | 1,894 | **delete** after `backend::cpu::dispatch` is created. |
| `kernel_generation.rs` | 592 | **survive** (kernel-generation is a different concern than backend dispatch; the migration does not move this file). |
| `moe.rs` | 33 | **survive** (MoE kernel support, not a backend dispatch path). |

### Other target-side scheduling-adjacent code

| File | Disposition | Notes |
|---|---|---|
| `crates/prism-ecs-runtime/src/buffer_lifetime_plan.rs` | **survive** (separate concern) | Buffer lifetime is not part of the scheduling subsystem. |
| `crates/prism-ecs-runtime/src/engine_receipts.rs` | **survive** (canonical receipt shape) | Scheduling receipts are a specialization of this. The `evidence::scheduling_receipts` module becomes the scheduling-specific variants. |
| `crates/prism-ecs-runtime/src/arena/mod.rs` (untracked, from reverted agent) | **delete** | The arena was a wrong-vocabulary work product. The new structure uses `prism-ecs-kernel::backend::metal::activation_arena.rs` for actual arenas. |
| `crates/prism-ecs-runtime/src/attention_sink.rs` | **survive** (separate concern) | Attention sink is downstream of scheduling, not a scheduling decision. |
| `crates/prism-ecs-runtime/src/pipeline_parity/*` | **survive** (separate concern) | Pipeline parity is test infrastructure, not scheduling authority. |

## Dependency-driven commit order (replaces bucket-only order)

The migration commits follow the dependency graph, not the bucket graph. Each commit moves one slice AND its callers AND moves the tests that exercise that slice, and adds the architectural invariant tests incrementally as the slices they're testing are introduced.

Sequence (each step is one commit; **buildable after each**):

1. `lane_capacity` (state, C) + its callers + `lane_capacity_boundary` invariant test → first slice
2. `lane_work` (state, C) + callers
3. `batch` (state, C) + callers
4. `ready_queue` (state, C) + callers
5. `slot` → merge into `state::lease` (C) + callers
6. `phase_invocation` (state, C) + callers
7. `phase_engine_state` → merge into `state::phase` (C) + callers
8. `activation_transaction` state half (C) + callers
9. `phase_cancellation` state half (C) + callers
10. `token_budget` state half (C) + callers
11. `execution_context` (C) + callers
12. `request`, `saved_request` (C) + callers
13. `work_registry::registry` + `work_registry::scheduling` → merge into `state::work_registry` (C) + callers
14. `prism_session` **decomposed by authority**: scheduling-local state → `prism-ecs-runtime::scheduling::state::session`; connection / request / client / auth → `prism-ecs-server::runtime::server::session_lifecycle` (already exists); wire-level session ID → `prism-ecs-protocol`. Callers updated in the same commit.
15. `phase_engine` system half → `state::phase` is built in step 7; now move the **system half** to `systems::phase_advancement` + callers + `phase_transition_is_applied_only_through_world_txn` invariant test
16. `phase_runner/execution` → `systems::orchestration` + callers
17. `phase_runner/dispatch` → `systems::dispatch_selection` + callers
18. `slot_lease_manager` → `systems::lease_allocation` + callers + `later_systems_observe_committed_lease_assignment` invariant test
19. `phase_runner/fallback` (S) + callers
20. `execution_lease_bridge` (S) + callers
21. `agent_bridge` (S, runtime half) + callers
22. `work_lifecycle_bridge` (S) + callers
23. `pipeline_bridge` (S) + callers
24. `prefill_orchestrator` (S) + callers
25. `kv_transaction` (S) + callers
26. `unified_scheduler` + `scheduler` (merged) → `systems::unified_scheduler` + callers
27. `tri_lane_orchestrator` runtime half → `systems::tri_lane_orchestration` + callers
28. `completion_bridge` (S) + `completion_reconciliation` invariant test
29. `phase_cancellation` system half (S) + callers
30. `phase_readiness` (S) + callers
31. `token_budget` system half (S) + callers
32. `prism_session` **system** half → `systems::session` + callers
33. `compilation_job_bridge` (S) + callers
34. `distributed_bridge` (TBD on inspection) + callers
35. `ingress_bridge` → `prism-ecs-runtime::ports::ingress` + callers
36. `heterogeneous_executor` **split during absorption**: runtime half → `systems::heterogeneous_orchestration`; kernel half → `prism-ecs-kernel::backend::dispatcher::heterogeneous` + callers
37. `metal_lane_executor` (A) → `prism-ecs-kernel::backend::metal::lane_executor` + callers
38. `metal_decoder` (A) → `prism-ecs-kernel::backend::metal::decoder` + callers
39. `ane_lane_executor` (A) → `prism-ecs-kernel::backend::ane::lane_executor` + callers
40. `ane_artifact_cache` (A) → `prism-ecs-kernel::backend::ane::artifact_cache` + callers
41. `accelerate_lane_executor` (A) → `prism-ecs-kernel::backend::accelerate::lane_executor` + callers
42. `lane_executors` (A, registry) → `prism-ecs-kernel::backend::lane_executor_registry` + callers
43. `metal_decoder` already done in step 38; skip
44. `weight_residency` (A) → `prism-ecs-kernel::backend::metal::weight_residency` + callers
45. `activation_arena` (A) → `prism-ecs-kernel::backend::metal::activation_arena` + callers
46. `activation_binding` (A) → `prism-ecs-kernel::backend::metal::activation_binding` + callers
47. `activation_transaction` FFI half (A) → `prism-ecs-kernel::backend::metal::activation_transaction` + callers
48. `memory_pool` (A) → `prism-ecs-kernel::backend::metal::memory_pool` + callers
49. `legacy_adapter` (A) → `prism-ecs-kernel::backend::legacy::adapter` + callers
50. `agent_bridge` kernel half (A) → `prism-ecs-kernel::backend::ane::agent_bridge` + callers
51. `tri_lane_orchestrator` kernel parts → per-backend dispatch consumers + callers
52. `receipts` → merge into `evidence::scheduling_receipts` + callers
53. `receipt` → merge into `evidence::scheduling_receipts` (subtype of `engine_receipts`) + callers + `receipt_references_committed_dispatch_and_lease` invariant test
54. `outlier_detector` (M) → `metrics::outlier_detector` + callers
55. `scheduler_metrics` (M) → `metrics::scheduler_metrics` + callers
56. `phase_telemetry` (M) → `metrics::phase_telemetry` + callers
57. `benchmark_harness` (X) → `prism-ecs-runtime/tests/scheduling_benchmarks.rs`
58. `mod.rs` (X) + subdirectory mod.rs files → **delete**
59. **Engine deletion**: `git rm compute-core/src/ecs/scheduling/`
60. **Test invariant suite assembly**: add `ecs_visible_dispatch_intent_is_invisible_before_commit`, `ecs_visible_dispatch_intent_is_visible_after_commit`, `failed_scheduling_transaction_leaves_world_unchanged`, `transaction_commit_preserves_schedule_visibility_order` (the ones that exercise the full system together)
61. **Workspace architecture test**: `workspace_contains_no_legacy_scheduling_imports` source scan

**Test policy for each commit:**
- Run `cargo check -p <affected_crate> --all-targets`
- Run `cargo test -p <affected_crate> --lib <slice_path>`
- Run any integration tests under `prism-ecs-runtime/tests/` that exercise the slice
- Each slice's invariant test is added in the same commit that introduces the slice

## Proof-of-pattern gate (for declaring scheduling absorbed)

All of the following must pass before scheduling is declared absorbed:

- [ ] Staged state is invisible before commit (per `ecs_visible_dispatch_intent_is_invisible_before_commit`)
- [ ] Staged state is visible to later systems after commit (per `ecs_visible_dispatch_intent_is_visible_after_commit`)
- [ ] Failed transaction leaves the world unchanged (per `failed_scheduling_transaction_leaves_world_unchanged`)
- [ ] Backend effects cannot mutate the world (per `backend_submission_does_not_mutate_authoritative_world`)
- [ ] Completion reenters through reconciliation (per `completion_result_reenters_world_through_world_txn`)
- [ ] Receipts correspond only to committed state (per `receipt_references_committed_dispatch_and_lease`)
- [ ] No legacy scheduling imports remain in the workspace (per `workspace_contains_no_legacy_scheduling_imports`)

This tests the constitution, not whether functions return the numbers everyone hoped they would.
