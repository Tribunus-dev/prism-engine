# 2026-07-25 — `scheduling/` subsystem design and Phase 4 absorption plan

This document is the design phase for the engine's `scheduling/`
subsystem absorption into the constitutional ECS. It is required reading
before any re-implementation under Phase 4 new-absorption. The plan
follows the project-absorption pattern established by the
`tinygrad_core.rs` → `phase_graph/` work and the `core/` /
`compute_image/` Phase 4C / 4B work.

Authority: this is the design proposal for the new-absorption of
`compute-core/src/ecs/scheduling/` (16K LOC, ~50 files). It does not
authorize code removal; it is the design.

---

## 1. Engine's `scheduling/` responsibilities

`compute-core/src/ecs/scheduling/mod.rs` declares the engine's
continuous-batching scheduler. The 50-file subsystem is responsible for
five conceptually distinct concerns that have grown together over the
MLX / omlx port. Reading the `mod.rs` and the largest files (mod.rs,
heterogeneous_executor.rs, lane_queue.rs, slot_lease_manager.rs,
phase_engine.rs, phase_runner/execution.rs, backpressure.rs,
lane_capacity.rs, lane_work.rs, scheduler.rs) reveals the
responsibility surface:

### 1.1 Heterogeneous backend dispatch

`heterogeneous_executor.rs` (933 LOC) is the engine's central Tokio actor
for dispatching work to one of three execution lanes (Metal GPU, ANE,
Accelerate/CPU). It owns:

- Per-lane `LaneExecutor` references (Metal / ANE / Accelerate).
- A `LaneCapacityManager` instance and `LaneQueueSet` for backpressure.
- A `SlotLeaseManager` for output-slot access tracking.
- A `BackpressureEventController` for typed backpressure.
- A `WorkLifecycleBridge` and `ExecutionLeaseBridge` for the
  constitutional handshake.
- An `mpsc::UnboundedReceiver<WorkCompletion>` for completion processing.
- A receipt collector and metrics handle.
- The `SessionSubmitRequest` / `EpochExecutionResult` envelope types.
- The variant-selection scoring function that picks the best admissible
  variant for each phase.

### 1.2 Per-lane work queueing

`lane_queue.rs` (648 LOC) owns:

- A bounded per-lane `LaneQueue` with priority ordering (Compilation →
  Warmup → Low → Normal → High → Interactive) and FIFO within priority.
- A `LaneQueueSet` aggregating queues for the three primary lanes.
- A `BackpressureReason` enum mapping lane → capacity reason.
- The `WorkPriority` enum and `QueueEntry` record.

### 1.3 Slot-based resource leasing

`slot_lease_manager.rs` (778 LOC) owns:

- A `SlotLease` record (lease_id, slot_id, owner, access mode, state,
  consumer count).
- An `Acquire-write` / `acquire-read` access protocol.
- A state machine `LeaseState` (Free → WriteReserved → WriteActive →
  OutputReady → ReadActive → Consumed → Poisoned).
- Per-slot reader-count tracking and last-reader-cleans-up semantics.
- Bulk operations: `release_session`, `poison_slot`.

### 1.4 Phase-graph execution

`phase_engine.rs` (558 LOC) and `phase_runner/execution.rs` (862 LOC)
own:

- A `PhaseEngine` struct that executes an `EmittedPhaseGraph` through a
  `PhaseRunnerRegistry`.
- A `PhaseLifecycleTracker` (212 LOC) for ready/completed phase state.
- A `ReadyQueue` (135 LOC) for the ready-set computation.
- A `PhaseRunner` trait and concrete runners (MLX, Metal, Core ML,
  Accelerate).
- A `PhaseReceipt` collection and `PhaseGraphResult` envelope.

### 1.5 Backpressure, batching, receipt, and metrics

- `backpressure.rs` (691 LOC) — `BackpressureEventController` (resource
  events), `BackpressureController` (latency window), and
  `SchedulingMetrics` (token budget).
- `batch.rs` (~200 LOC) — the `Batch` envelope for prefill+decode
  batching.
- `receipt.rs` (618 LOC) and `receipts.rs` (64 LOC) — execution
  receipts with full timing provenance and the `ReceiptCollector`.
- `scheduler_metrics.rs` (443 LOC) — atomic counters and EXO
  autoscaler-facing metrics.

### 1.6 Orchestration glue (out of scope for this design)

The remaining ~12 files (lane_executors.rs, lane_work.rs,
lane_capacity.rs, work_lifecycle_bridge.rs, execution_lease_bridge.rs,
ingress_bridge.rs, agent_bridge.rs, completion_bridge.rs, prism_session.rs,
saved_request.rs, scheduler.rs, unified_scheduler.rs, etc.) are
plumbing between the five concerns above and the engine's other
subsystems. They are not candidate first-batch re-implementations
because each one in isolation is too small and too tied to its caller
to absorb cleanly. They will be absorbed as part of the
follow-up waves listed in §6.

---

## 2. Prism-domain decomposition

The engine's `scheduling/` is one logical concern (continuous batching)
implemented across 50 files. The constitutional Prism splits that one
concern into **five** domain authorities, each with a single
authoritative file (or small directory). The split is a Prism-domain
split — it is not "map the engine file 1:1". The pattern is:
**identify the design idea, rewrite using Prism's types and crate
boundaries**.

### 2.1 Lane admission (constitutional `lane_admission.rs`)

Owns: the canonical admission gate for dispatching one work item onto
one execution lane. This is the Prism-domain authority for the
engine's `LaneCapacityManager` + `LaneQueue` + per-lane queue
aggregation.

Why a new file: the engine's three types collapse into one Prism
authority — "admitting a work item to a lane." The constitutional
authority is the typed admission decision (priority, deadline, permit),
not the implementation choice of "queue + counter" vs "counter-only."

Files re-implemented from: `lane_queue.rs`, `lane_capacity.rs`,
parts of `lane_work.rs`.

Crate: `prism-ecs-runtime::schedule::lane_admission` (peer of
`AdmitSystem`, `LeaseSystem`, `DispatchSystem`).

### 2.2 Lane slot leasing (constitutional `lane_slot_lease.rs`)

Owns: the canonical lease lifecycle for output-slot access (write vs
read, reader tracking, consumer-count accounting, force-release on
cancellation). This is the Prism-domain authority for the engine's
`SlotLeaseManager`.

Why a new file: the engine's lease state machine is one authority —
"who can read or write this slot right now." It does not own
`ExecutionLease` (which is in the constitutional `execution.rs`); it
owns the lane-level access pattern on top of `ExecutionLease`.

Files re-implemented from: `slot_lease_manager.rs`, parts of
`slot.rs`.

Crate: `prism-ecs-runtime::schedule::lane_slot_lease` (peer of
`lane_admission`).

### 2.3 Phase-graph execution (constitutional `phase_engine/` directory)

Owns: the canonical phase-DAG execution, ready-set computation, and
phase-runner dispatch. This is the Prism-domain authority for the
engine's `PhaseEngine` + `PhaseLifecycleTracker` + `ReadyQueue`.

Why a directory: the engine's phase execution is a layered authority
(ready-set → lifecycle → runner dispatch). A single file at 1300+ LOC
would violate the 900-LOC hard limit. The directory splits by:

- `phase_engine/` `mod.rs` — the directory index, re-exports
- `phase_engine/` `ready_set.rs` — `ReadyQueue` and ready-set
  computation (was `ready_queue.rs`)
- `phase_engine/` `lifecycle.rs` — `PhaseLifecycleTracker` and the
  state machine (was `phase_engine_state.rs` + parts of
  `phase_engine.rs`)
- `phase_engine/` `runner.rs` — `PhaseRunner` trait, `PhaseRunnerRegistry`,
  and the dispatch (was `phase_runner/` + parts of `phase_engine.rs`)
- `phase_engine/` `graph.rs` — `PhaseEngine` and `execute_graph` /
  `execute_until_terminal` (was `phase_engine.rs` orchestration)

Crate: `prism-ecs-runtime::schedule::phase_engine` (peer of the
schedule stages). The `PhaseEngine` is a kernel-internal executor, not
a constitutional command; it lives in `prism-ecs-runtime` not
`prism-ecs-constitutional` per the crate boundary rules.

### 2.4 Backpressure (constitutional `backpressure.rs`)

Owns: the typed backpressure events, severity levels, and the
latency-window controller. This is the Prism-domain authority for the
engine's `BackpressureEventController` + `BackpressureController` +
`SchedulingMetrics`.

Why a single file: the engine's two `Backpressure*` types and the
`SchedulingMetrics` struct are all one authority — "should the
scheduler throttle new admissions, and why?" Splitting them across
files would scatter one concern.

Files re-implemented from: `backpressure.rs` (full), parts of
`scheduler_metrics.rs`.

Crate: `prism-ecs-runtime::schedule::backpressure` (peer of
`AdmitSystem` — backpressure is the admission gate's input).

### 2.5 Receipt and metrics (constitutional `execution_receipts.rs` + `metrics.rs`)

Owns: the typed execution receipt (timing provenance, fallback
recording) and the per-process atomic metrics counters.

Why two files: the engine's `receipt.rs` is the per-work-item
evidence record (one file); the `scheduler_metrics.rs` is the global
counters (one file). They are distinct authorities.

Files re-implemented from: `receipt.rs`, `scheduler_metrics.rs`.

Crate: `prism-ecs-runtime::schedule::execution_receipts` and
`prism-ecs-runtime::schedule::metrics`.

### 2.6 Out of this design

- `heterogeneous_executor.rs` — the orchestrator actor. It composes
  the five authorities above. It is itself a Prism-domain authority
  ("the heterogeneous executor") but it is **not** re-implemented
  in this phase. The first-batch files (§4) are the leaf authorities;
  the orchestrator is the second-batch file. Absorbing the actor
  before the leaves it composes would be premature.
- `phase_runner/execution.rs` — the concrete runner implementations
  (MLX, Metal, Core ML, Accelerate). Each runner is a backend-specific
  executor; the runners live in the backend crate (or stay in the
  engine until the backend is absorbed). They are not constitutional
  types.
- `prism_session.rs`, `saved_request.rs`, `scheduler.rs`,
  `unified_scheduler.rs` — these are session/legacy-bound schedulers
  that own request lifecycles, not lane- or phase-graph authorities.
  They are out of scope for Phase 4; they will be handled when the
  engine's session subsystem is absorbed (Phase 5+).

---

## 3. Target crate decision

**Recommendation: place the new code in `prism-ecs-runtime` as
`schedule::*` peers, not in a new crate.**

Reasoning:

1. **Schedule is the right authority.** The engine's `scheduling/` is
   a runtime-schedule concern. The existing `prism-ecs-runtime` crate
   owns `RuntimeSchedule` (Observe / Plan / Admit / Lease / Dispatch /
   Collect / Publish / Cleanup). The new lane-admission, lane-slot-lease,
   phase-engine, backpressure, and execution-receipt authorities are
   runtime-schedule peers.

2. **No new crate boundary is justified.** A new `prism-ecs-scheduling`
   crate would create a peer of `prism-ecs-runtime` and
   `prism-ecs-constitutional`. But the new types depend on the
   constitutional `Command`, `WorkItemComponent`, `ResourceClaim`, and
   `LifecycleCommand`. They are not a foundation layer. They are
   runtime policies.

3. **The `prism-ecs-runtime::schedule` directory already exists** as
   the peer of `crates/prism-ecs-runtime/src/schedule.rs`. The new
   files will land at `crates/prism-ecs-runtime/src/schedule/<name>.rs`
   or under a subdirectory (e.g. `schedule/phase_engine/`).

4. **No high-level `prism-ecs-scheduling` is needed at this stage.**
   If the absorbed code grows past the point where the
   `prism-ecs-runtime` crate is dominated by schedule code, the
   follow-up change can split a `prism-ecs-scheduling` crate at that
   time. Premature crate creation is a migration cost.

### 3.1 Dependency direction

The new code depends on:

- `prism_ecs_constitutional::types::*` (Generation, Epoch, LeaseToken,
  CommandId, plus newtypes `LaneId`, `SlotId`, `PhaseId`,
  `BackpressureLevel`).
- `prism_ecs_constitutional::work::*` (WorkKind, ResourceClaim).
- `prism_ecs_constitutional::lifecycle_command::*` (for the typed
  command surface).
- `prism_ecs_core::Entity` (for the canonical entity handle).
- `crate::schedule::*` (RuntimeSchedule, SystemId, SystemStage).
- `crate::ports::*` (WorkDispatcher, ProviderSelectionRequest,
  ProviderDescriptor — the executor calls into these for effects).

It does not depend on `compute-core::*`. The engine's `scheduling/`
is archaeology; the re-implementation is constitutional.

---

## 4. Re-implementation order (first batch)

The first batch of 3 files to re-implement, by leverage:

### 4.1 `lane_queue.rs` (648 LOC original)

Why first:

- It is the smallest of the eight candidates (648 LOC).
- It is a pure data structure (no actor, no async, no backend
  dependency). Re-implementing it is a clean test of the project-
  absorption pattern with low risk.
- It is the leaf authority for lane admission (the engine's
  heterogeneous_executor and scheduler both consume it).
- Its test coverage (13 tests in the original) is complete and
  self-contained; porting them is a clean test of the new crate's
  build gate.

Target file: `crates/prism-ecs-runtime/src/schedule/lane_queue.rs`.

Target re-implementation:

- Type newtypes `LaneId(u32)`, `WorkId(u64)`, `BackpressureReason`
  (already a `Copy` enum in the engine).
- `BTreeMap<LaneId, LaneQueue>` for the `LaneQueueSet` aggregation
  (no `HashMap` per the AGENTS.md rule).
- `WorkPriority` enum as a typed newtype, `Ord` impl for the
  `VecDeque::max_by_key` ordering.
- `BackpressureReason` preserved; `BackpressureLevel` from the
  constitutional types module (re-exported as
  `prism_ecs_constitutional::types::BackpressureLevel` if it exists
  in the constitutional crate, or re-declared as a newtype if not —
  to be confirmed during re-implementation).
- `thiserror`-derived `BackpressureError` replacing the
  `Result<(), BackpressureReason>` (the engine's `BackpressureReason`
  is a `Copy` enum but the constitutional pattern prefers a typed
  error with `Rejected` / `Failed` / `Stale` classification).
- All 13 tests ported; `unwrap` removed (tests use `Result` returns).
- `LaneQueue::pop` returns `Option<QueueEntry>` (preserved).

Estimated re-implementation: 350-450 LOC, ~12 public items.

### 4.2 `slot_lease_manager.rs` (778 LOC original)

Why second:

- It is the next-smallest of the eight candidates (778 LOC).
- It is a pure data structure (no actor, no async, no backend
  dependency).
- It is the leaf authority for lane-slot access (the
  heterogeneous_executor and lane executors both depend on it).
- Its test coverage (28+ tests in the original) is complete and
  self-contained; the state machine has a clean test surface.

Target file: `crates/prism-ecs-runtime/src/schedule/lane_slot_lease.rs`.

Target re-implementation:

- Type newtypes `SlotId(u64)`, `LeaseId(u64)` (the engine uses
  `SlotLeaseId` from `activation_abi`; the constitutional side should
  use a Prism-domain newtype `LeaseId`).
- `BTreeMap<LeaseId, SlotLease>` for the lease store; `BTreeMap<SlotId,
  LeaseId>` for the writer map; `BTreeMap<SlotId, u32>` for the reader
  counter map.
- `SlotAccess` and `LeaseState` as typed enums (preserved names).
- `thiserror`-derived `LeaseError` replacing the `Result<_, String>`
  pattern.
- `SlotLeaseManager::acquire_write` / `acquire_read` / `mark_output_ready`
  / `release` / `release_session` / `poison_slot` / `current_writers`
  / `reader_count` methods (the engine's public surface).
- All 28+ tests ported; `unwrap` removed.

Estimated re-implementation: 600-700 LOC, ~20 public items.

### 4.3 `backpressure.rs` (691 LOC original)

Why third:

- It is the largest of the three first-batch files (691 LOC) but
  remains under the 900-LOC hard limit when re-implemented with
  constitutional types.
- It is the admission gate's input (it lives between the
  `LaneQueue` authority and the orchestrator).
- It owns the typed `BackpressureLevel` and the `SchedulingMetrics`
  struct; both are constitutional types.
- The test surface is small (~5 tests in the engine) but the type
  surface is large (3 controllers, 1 reason enum, 1 level enum, 1
  metrics struct, 1 summary struct).

Target file: `crates/prism-ecs-runtime/src/schedule/backpressure.rs`.

Target re-implementation:

- Type newtypes `BackpressureLevel` (replacing the engine's enum
  with a constitutional newtype that wraps a u8 severity ordinal —
  this aligns with the constitutional `Generation` / `Epoch` /
  `Priority` patterns).
- `BTreeMap<BackpressureReason, SeverityBucket>` for the event store
  (preserving the engine's `Vec<BackpressureEvent>` aggregation but
  moving to an ordered map for the canonical event log).
- `BatchCompletionRecord` and `BackpressureController` preserved.
- `SchedulingMetrics` re-implemented as a typed struct using the
  constitutional `Generation` for the `num_running_requests` /
  `max_num_scheduled_tokens` fields.
- `thiserror`-derived `BackpressureError`.
- All tests ported; `unwrap` removed.

Estimated re-implementation: 500-600 LOC, ~15 public items.

### 4.4 Why these three over the alternatives

| File | LOC | Why not first |
|---|---:|---|
| `heterogeneous_executor.rs` | 933 | Composes the five authorities; absorbing the leaves first is the safe order. |
| `phase_engine.rs` | 558 | Splits into 4 sub-files; too big for first batch. Belongs to second batch. |
| `phase_runner/execution.rs` | 862 | Backend-specific runners; the runners belong with the backends, not the runtime. |
| `lane_executors.rs` | 97 | Stub implementations; not a real authority. |
| `lane_work.rs` | 244 | Mostly transfer types; will be re-exported from the new `lane_queue.rs` and `lane_slot_lease.rs`. |

The first batch is intentionally leaf authorities. The second batch
(Phase 4d) will absorb the orchestrator (`heterogeneous_executor.rs`)
and the phase-engine directory.

---

## 5. Per-file re-implementation plan (5-step pattern)

For each of the three first-batch files, the absorption follows the
5-step pattern in `references/project-absorption.md`:

1. **One-sentence authority doc.** Every new file's module doc states
   the single Prism-domain authority in one sentence and explicitly
   says "It does not own X." No external project name appears in the
   doc.

2. **Crate ownership.** Place the new file in
   `crates/prism-ecs-runtime/src/schedule/<name>.rs` as a peer of
   `AdmitSystem` / `LeaseSystem` / `DispatchSystem`.

3. **Re-implement the pattern, not the code.** Study the engine file;
   identify the design idea (per-lane bounded queue, slot access state
   machine, admission gate); rewrite using Prism's types
   (`Generation`, `Epoch`, `LeaseToken`, new `LaneId` / `SlotId` /
   `PhaseId` newtypes, `BTreeMap` instead of `HashMap`, `thiserror`
   instead of `String` errors).

4. **Wire into the canonical change flow.** Each new file participates
   in the schedule:
   - `lane_queue.rs` → `AdmitSystem` calls `LaneQueue::try_push` /
     `pop` from the `Admit` stage.
   - `lane_slot_lease.rs` → `LeaseSystem` calls `acquire_write` /
     `mark_output_ready` / `release` from the `Lease` stage.
   - `backpressure.rs` → `AdmitSystem` reads `BackpressureLevel`
     before admitting; `DispatchSystem` records
     `BatchCompletionRecord` after dispatch.

5. **Delete the original** in the same commit. The engine file
   `compute-core/src/ecs/scheduling/lane_queue.rs` (and the others)
   are moved to `compute-core.legacy/scheduling/`. The legacy path
   is the marker.

---

## 6. Roadmap for absorbing the remaining `scheduling/` files

The remaining 47 files split into four follow-up waves, each
gated on a Completion report and review:

### Phase 4d (next)

- `heterogeneous_executor.rs` (933 LOC) — the orchestrator actor.
  Decomposed into `HeterogeneousExecutor` (the actor) + `VariantSelector`
  (the score function) + `EpochPlanner` (the phase-graph walk).
- `phase_engine.rs` (558 LOC) + `phase_engine_state.rs` (212 LOC) +
  `phase_runner/*` (944 LOC) + `ready_queue.rs` (135 LOC) — the
  `phase_engine/` directory. Target:
  `crates/prism-ecs-runtime/src/schedule/phase_engine/`.

### Phase 4e

- `lane_executors.rs` (97 LOC) — stub per-lane executors. Replace
  with the typed `LaneExecutor` trait already defined in
  `prism_ecs_runtime::backend`.
- `lane_work.rs` (244 LOC) — `LaneWorkRequest` / `WorkCompletion` /
  `CompletionClock` / `WorkSubmission`. Move to
  `prism-ecs-runtime::schedule::lane_work`.
- `lane_capacity.rs` (~300 LOC) — `LaneCapacityManager` /
  `LaneCapacityConfig` / `LaneCapacitySnapshot`. Move to
  `prism-ecs-runtime::schedule::lane_capacity`.

### Phase 4f

- `prism_session.rs` (658 LOC) + `saved_request.rs` (~400 LOC) +
  `scheduler.rs` (543 LOC) + `unified_scheduler.rs` (530 LOC) +
  `prefill_orchestrator.rs` (182 LOC) + `token_budget.rs` (260 LOC) +
  `ready_queue.rs` (135 LOC). These are the engine's request
  lifecycle and continuous-batching loop. They are bigger and more
  coupled to the engine's session model; they will be absorbed
  alongside the engine's session subsystem (which is the next
  non-scheduling subsystem on the Phase 4 list).

### Phase 4g (cleanup)

- `execution_context.rs` (config-only) — replace with the
  constitutional `ExecutionContext` (or remove if redundant).
- `receipt.rs` (618 LOC) + `receipts.rs` (64 LOC) — port the
  receipt types to `prism-ecs-runtime::schedule::execution_receipts`.
- `scheduler_metrics.rs` (443 LOC) — port to
  `prism-ecs-runtime::schedule::metrics`.
- The 12+ bridge files (`agent_bridge.rs`, `completion_bridge.rs`,
  `execution_lease_bridge.rs`, `ingress_bridge.rs`,
  `compilation_job_bridge.rs`, `distributed_bridge.rs`,
  `pipeline_bridge.rs`, `work_lifecycle_bridge.rs`,
  `phase_cancellation.rs`, `activation_arena.rs`,
  `activation_binding.rs`, `activation_transaction.rs`,
  `kv_transaction.rs`, `memory_pool.rs`, `metal_decoder.rs`,
  `weight_residency.rs`, `ane_artifact_cache.rs`,
  `benchmark_harness.rs`, `legacy_adapter.rs`,
  `outlier_detector.rs`, `phase_invocation.rs`,
  `phase_readiness.rs`, `phase_telemetry.rs`,
  `tri_lane_orchestrator.rs`, `work_lifecycle_bridge.rs`,
  `work_registry/*`) — these are absorbed as the engines they
  bridge to are absorbed. Most are pure-legacy and will be
  quarantined under `compute-core.legacy/scheduling/bridges/`
  with no constitutional re-implementation.

---

## 7. Risks and open questions

1. **The `BackpressureLevel` type may not exist in the constitutional
   crate.** The design assumes `prism_ecs_constitutional::types::BackpressureLevel`
   exists. If it does not, the new file will define it locally as a
   newtype and re-export it. The constitutional crate already has
   `Generation` / `Epoch` / `Priority` / `Sequence` / `CommandId`
   newtypes; `BackpressureLevel` follows the same pattern.

2. **`SlotLeaseId` is in the engine's `activation_abi`.** The
   constitutional `lane_slot_lease.rs` will define a new `LeaseId`
   newtype (Prism-domain name) and not depend on `activation_abi`.
   This is a crate-boundary fix: the engine's `SlotLeaseId` is
   upstream; the new `LeaseId` is constitutional.

3. **`PhaseId` and `WorkId` are not in the constitutional crate
   today.** They will be added to `prism_ecs_constitutional::types`
   as newtypes in the same change. The engine's `compilation::phase_ir::PhaseId`
   and `scheduling::lane_work::WorkId` are upstream types; the
   constitutional versions are the canonical types.

4. **The heterogeneous executor's variant-selection scoring function
   uses `f64::NEG_INFINITY` and a hand-rolled score function.** The
   re-implementation will use the constitutional `Priority` / `Generation`
   enums and replace the f64 score with a typed ranking. The exact
   ranking heuristic is preserved (lower cost + lower risk + lower
   queue depth wins), but the implementation uses a typed `BTreeMap`
   of ranked variants rather than a linear scan.

5. **The engine's `HashMap` usage in the `lane_queue.rs`,
   `slot_lease_manager.rs`, and `backpressure.rs` will be replaced
   with `BTreeMap`.** The engine uses `HashMap` for performance;
   the constitutional rule prefers `BTreeMap` for canonical
   collections. Performance impact: O(log n) vs O(1). At the
   expected sizes (per-lane queue depth ≤ 64, per-lease id space
   in the millions), `BTreeMap` is fast enough. The rule is
   explicit in AGENTS.md.

6. **The 33+ engine files under `scheduling/` and `phase_runner/`
   import `mlx_rs::Array` and other engine-internal types.** These
   are not absorbed; the runners are backend-specific. The
   constitutional re-implementation does not re-implement the
   runners; it re-implements the orchestrator and the leaves.

7. **The `prism-ecs-runtime` crate currently has a 1-warning lint
   baseline** (`unused import: World` in `buffer_lifetime_plan.rs`).
   The new files must not add new warnings.

---

## 8. Completion criteria for the first batch

Phase 4 scheduling absorption is "first batch done" when:

1. **Design doc** (this file) is committed.
2. **3 new files** exist at
   `crates/prism-ecs-runtime/src/schedule/{lane_queue,lane_slot_lease,backpressure}.rs`.
3. **3 engine files** are moved to `compute-core.legacy/scheduling/`.
4. **No new lint warnings** in `prism-ecs-runtime`.
5. **All tests pass** in `prism-ecs-runtime` (existing + ported).
6. **Each new file has a one-sentence authority doc** in its module
   comment, with no external project name.
7. **Each new file uses BTreeMap** for canonical collections.
8. **Each new file uses typed errors** (`thiserror`, not `String`).
9. **Each new file has a test module** with the original tests
   ported.
10. **A Completion report** is written at
    `changelogs/2026-07-25-compute-core-absorption-phase-4-scheduling.md`.

When all ten are true, the follow-up waves (4d, 4e, 4f, 4g) are
unblocked.
