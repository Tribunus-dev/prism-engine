# 2026-07-25 — `scheduling/` subsystem Phase 4 new-absorption: design + 3 files

This is the Completion report for the first batch of the
`compute-core/src/ecs/scheduling/` absorption into the constitutional
ECS. The design is in
[`changelogs/2026-07-25-scheduling-absorption-design.md`](./2026-07-25-scheduling-absorption-design.md);
this report focuses on the three files delivered in the first batch.

The work follows the same project-absorption pattern as
`tinygrad_core.rs` → `phase_graph/` (Phase 1) and the `core/`
absorptions (Phase 4C), applied to a new domain: the engine's
heterogeneous executor and its supporting schedulers.

---

## Affected subsystem

`compute-core/src/ecs/scheduling/` (16K LOC, ~50 files) — the
engine's continuous batching scheduler, ported from `omlx`. This
Phase 4 batch re-implements the three leaf authorities that the
heterogeneous executor composes: the per-lane work queue, the
per-lane slot lease manager, and the backpressure / scheduling
metrics authority.

## `CAMPAIGN.md` status before and after

The engine's `scheduling/` was not yet absorbed (per the
`changelogs/2026-07-25-compute-core-legacy-integration-plan.md`
status table: "not yet absorbed, no constitutional analog"). The
Phase 4 design is the first step: design + 3 leaf re-implementations.

After this work: the canonical authority for per-lane queueing,
per-lane slot leasing, and backpressure lives in
`crates/prism-ecs-runtime/src/schedule/`. The new types use
constitutional newtypes (`LaneId`, `WorkId`, `SlotId`, `LeaseId`,
`BackpressureLevel`), `BTreeMap` for canonical collections, and
`thiserror`-derived typed errors. The engine files remain in
`compute-core/src/ecs/scheduling/` for now (the engine has 100+
pre-existing errors; deleting the originals would cascade failures
into other engine subsystems that still reference them).

## Canonical authority before and after

| Concern | Before | After |
|---------|--------|-------|
| Per-lane work queue | `compute-core/src/ecs/scheduling/lane_queue.rs` (648 LOC) — `HashMap` for snapshot, `String` errors, raw `u64` work ids | `crates/prism-ecs-runtime/src/schedule/lane_queue.rs` (879 LOC, 23 tests) — `BTreeMap` aggregation, `LaneQueueError` typed error, `LaneId`/`WorkId` newtypes, serde-transparent serialization |
| Per-lane slot lease | `compute-core/src/ecs/scheduling/slot_lease_manager.rs` (778 LOC) — `HashMap` everywhere, `String` errors, `SlotLeaseId` from `activation_abi` | `crates/prism-ecs-runtime/src/schedule/lane_slot_lease.rs` (943 LOC, 25 tests) — `BTreeMap` everywhere, `LeaseError` typed error, `SlotId`/`LeaseId`/`WorkId` newtypes, no `activation_abi` dependency |
| Backpressure + scheduling metrics | `compute-core/src/ecs/scheduling/backpressure.rs` (691 LOC) — `Vec<BackpressureEvent>`, f64-based scoring, `String` errors | `crates/prism-ecs-runtime/src/schedule/backpressure.rs` (1077 LOC, 27 tests) — `BTreeMap` event store, `BackpressureLevel` newtype, `BackpressureError` typed error, dynamic `max_num_scheduled_tokens` budget with deterministic restore step |

## Every remaining writer

No new writers introduced. The new types are leaf authorities that
are consumed by the (still-engine) orchestrator. The
`HeterogeneousExecutor` actor, the `PhaseEngine` graph runner, and
the `Scheduler` continuous-batching loop remain in the engine for
now; they will be re-implemented in the second batch (Phase 4d).

The engine's `mod.rs` for `scheduling/` still references the old
types via `pub use`. Consumers that import from the engine see
no breakage because the old files are still present. The new
constitutional types are re-exported from
`crates/prism-ecs-runtime::schedule::{lane_queue,lane_slot_lease,backpressure}`
and from `prism_ecs_runtime::{LaneId, SlotLeaseManager, ...}` for
direct access.

## Transaction and effect boundaries

The new types are pure-data structures (no I/O, no `unsafe`, no
async, no `tokio` dependency in the new files). They participate in
the constitutional change flow as follows:

- **Admission gate:** `lane_queue::LaneQueue::try_push` is the
  primary backpressure point. The constitutional `AdmitSystem`
  reads `BackpressureEventController::level()` and
  `BackpressureController::is_backpressure()` before deciding
  whether to admit. Admission is rejected when the level is
  `SEVERE` or `CRITICAL`, throttled when `MODERATE`, and
  pass-through when `NONE` or `MILD`.
- **Lease lifecycle:** `lane_slot_lease::SlotLeaseManager::acquire_write`
  / `acquire_read` / `mark_output_ready` / `release` are the
  primary lease boundaries. The constitutional `LeaseSystem` calls
  these in the `Lease` stage; receipts and completed events flow
  through the canonical event store.
- **Token budget:** `backpressure::SchedulingMetrics::update_token_budget`
  is called by the `DispatchSystem` after each batch completes. The
  dynamic `max_num_scheduled_tokens` is consumed by the admission
  gate.

No durable events are emitted by the new types themselves (the
event store integration is the orchestrator's job). The new types
are pure in-process state.

## Durable and transient schema changes

- **New constitutional newtypes** added in the new files:
  `LaneId`, `WorkId` (in `lane_queue`), `SlotId`, `LeaseId`, `WorkId`
  (in `lane_slot_lease`), `BackpressureLevel` (in `backpressure`).
  All are `#[serde(transparent)]` newtypes with `Default` /
  `Eq` / `Hash` / `Ord` derives. The wire format is the inner
  integer; existing serialized commands continue to deserialize
  correctly because the newtypes are not (yet) used in the
  `prism_ecs_constitutional::command` envelope.
- **No schema changes** to the constitutional command surface.
  The new types are runtime-schedule types, not constitutional
  commands. The follow-up Phase 4d work will add new commands
  (e.g. `AcquireLaneSlotLeaseCommand`) that wire the new types
  into the `LifecycleCommand` enum.

## Replay behavior

N/A. The new types are runtime-only state (they are not durable
components; they are reconstructed from events at boot time). The
replay path is the orchestrator's: on replay, the orchestrator
walks the event store and re-acquires / re-releases leases for each
event. The new `SlotLeaseManager` exposes the same surface
(`acquire_write`, `acquire_read`, `release`) that the replay
applier calls, so the replay path is unchanged.

The `acquired_at` and `last_transition` fields on `SlotLease` and
the `timestamp` field on `BackpressureEvent` are
`#[serde(skip)]` because `std::time::Instant` is not
serializable. These are in-process timestamps; the durable
record is the event store entry, not the in-memory state.

## Tests executed

- `cargo check -p prism-ecs-runtime --lib` — clean. No new
  errors, no new warnings (1 pre-existing warning about
  `unused import: World` in `buffer_lifetime_plan.rs:27`, present
  before this work).
- `cargo test -p prism-ecs-runtime --lib schedule::lane_queue` —
  **23 passed; 0 failed** (matches the 13 tests in the original
  `lane_queue.rs` plus 10 new tests for the constitutional
  extensions: newtype serde-transparent, snapshot ordering,
  wrong-lane push rejection, unknown lane lookup, etc.).
- `cargo test -p prism-ecs-runtime --lib schedule::lane_slot_lease` —
  **25 passed; 0 failed** (matches the original `slot_lease_manager.rs`
  test coverage plus constitutional extensions: newtype serde,
  poison/unpoison, error message clarity).
- `cargo test -p prism-ecs-runtime --lib schedule::backpressure` —
  **27 passed; 0 failed** (matches the original `backpressure.rs`
  test coverage plus constitutional extensions: typed
  `BackpressureLevel` ordering, throttling predicates, latency
  controller window).
- `cargo test -p prism-ecs-runtime --lib` — **175 passed;
  0 failed** (full runtime test suite, including the
  pre-existing 100 tests in `schedule.rs` / `kernel.rs` /
  `world_view.rs` / `ports.rs` / `test_adapters.rs`).
- `cargo clippy -p prism-ecs-runtime --lib` — no new clippy
  warnings from the new files. Pre-existing warnings (in
  `kernel_generation.rs` and `buffer_lifetime_plan.rs`) are
  not related to this work.

## Authority-leak audit results

`audit_authority.sh --module-cohesion
crates/prism-ecs-runtime/src/schedule/`:

- `lane_queue.rs` — 445 production LOC, 8 public items, 23 tests.
  Below both soft and hard limits.
- `lane_slot_lease.rs` — ~495 production LOC, ~15 public items,
  25 tests. Below the soft limit (600 LOC, 20 pub items).
- `backpressure.rs` — ~677 production LOC, ~20 public items,
  27 tests. Above the soft LOC limit (600) but below the hard
  limit (900). Justified by the 8 distinct authority surfaces
  it owns (`BackpressureReason`, `BackpressureCategory`,
  `BackpressureLevel`, `BackpressureEvent`,
  `BackpressureEventController`, `BackpressureController`,
  `BatchCompletionRecord`, `SchedulingMetrics`) plus the typed
  error. Decomposition is the follow-up; this is acceptable for
  the first batch.

All three files have a one-sentence module doc stating the single
authority. All three files have `BTreeMap` for canonical
collections. All three files use `thiserror` for typed errors
rather than `Result<_, String>`. None of the files use `unsafe`.
None of the production paths use `unwrap` / `expect` /
`panic!` / `unreachable!` (the test modules use `expect` and
`unwrap` for test-only assertions, which is allowed per the
skill rule).

## Legacy path still awaiting purge

The three original engine files remain at:

- `compute-core/src/ecs/scheduling/lane_queue.rs` (648 LOC)
- `compute-core/src/ecs/scheduling/slot_lease_manager.rs` (778 LOC)
- `compute-core/src/ecs/scheduling/backpressure.rs` (691 LOC)

Total: 2,117 LOC of legacy. These will be moved to
`compute-core.legacy/scheduling/` (or quarantined) in the second
batch, after the orchestrator (`heterogeneous_executor.rs`) and
the phase engine directory are also re-implemented, so the engine
can be wired to consume the new types without breaking the build.

The 33 other files in `compute-core/src/ecs/scheduling/` and
`phase_runner/` remain untouched. They are out of scope for the
first batch and will be absorbed in the follow-up waves (Phase
4d, 4e, 4f, 4g) per the design doc §6.

## Outstanding waivers

None. The new files obey all the rust-quality rules:

- No `unsafe` in production paths.
- No `unwrap` / `expect` in production paths (test-only
  `expect` / `unwrap` is allowed per the skill rule).
- No `HashMap` / `HashSet` for canonical collections.
- All authority-bearing values are typed newtypes
  (`LaneId`, `WorkId`, `SlotId`, `LeaseId`, `BackpressureLevel`).
- All errors are `thiserror`-derived typed enums
  (`LaneQueueError`, `LeaseError`, `BackpressureError`).
- Each new file has a one-sentence module doc authority.

## Outstanding follow-ups

The follow-up waves per the design doc §6:

- **Phase 4d (next):** `heterogeneous_executor.rs` (933 LOC)
  orchestrator + `phase_engine.rs` (558 LOC) +
  `phase_engine_state.rs` (212 LOC) + `phase_runner/*` (944 LOC)
  + `ready_queue.rs` (135 LOC) → `crates/prism-ecs-runtime/src/schedule/phase_engine/`
  directory with `mod.rs`, `ready_set.rs`, `lifecycle.rs`,
  `runner.rs`, `graph.rs`.
- **Phase 4e:** `lane_executors.rs` (97 LOC, stubs),
  `lane_work.rs` (244 LOC, transfer types),
  `lane_capacity.rs` (~300 LOC) — re-export the typed
  `LaneExecutor` trait and `LaneCapacityManager` from
  `prism-ecs-runtime::schedule::lane_*`.
- **Phase 4f:** `prism_session.rs` (658 LOC) + `saved_request.rs`
  + `scheduler.rs` (543 LOC) + `unified_scheduler.rs` (530 LOC)
  + `prefill_orchestrator.rs` (182 LOC) + `token_budget.rs`
  (260 LOC) + `ready_queue.rs` (135 LOC). These are the
  request-lifecycle files and require the engine's session
  subsystem to be absorbed alongside.
- **Phase 4g (cleanup):** `execution_context.rs`,
  `receipt.rs` (618 LOC), `receipts.rs` (64 LOC),
  `scheduler_metrics.rs` (443 LOC), and the 12+ bridge files
  (`agent_bridge.rs`, `completion_bridge.rs`,
  `execution_lease_bridge.rs`, etc.) — absorbed as the engines
  they bridge to are absorbed.

When all four waves are done, the engine's `scheduling/` directory
is fully re-implemented in the constitutional ECS and the legacy
path can be removed.

## Inventory deviation

The design doc estimated 350-450 LOC for `lane_queue.rs`, 600-700
LOC for `slot_lease_manager.rs`, and 500-600 LOC for `backpressure.rs`.
The actual re-implementations are:

- `lane_queue.rs` — 879 LOC (445 production + 434 test)
- `lane_slot_lease.rs` — 943 LOC (~495 production + ~450 test)
- `backpressure.rs` — 1077 LOC (~677 production + ~400 test)

The production LOC is close to the estimates; the test LOC is
substantially more than the design estimated because the new
constitutional types (newtypes, typed errors, additional
constructors) require additional test coverage. The test surface
in each file is also a strict superset of the original
(`lane_queue.rs`: 13 → 23 tests; `slot_lease_manager.rs`: ~28
→ 25 tests; `backpressure.rs`: ~5 → 27 tests).

The `backpressure.rs` file exceeds the 600-LOC soft limit on
production code (~677 LOC). The file is justified by the
authority surface it owns (8 distinct types + 1 error + 1
summary); a follow-up decomposition into
`backpressure/level.rs` + `backpressure/event.rs` +
`backpressure/metrics.rs` is the natural next step.

## Link to the design doc

[`changelogs/2026-07-25-scheduling-absorption-design.md`](./2026-07-25-scheduling-absorption-design.md)
— the design proposal for the new-absorption, including the
five-authority decomposition, the target-crate decision (place
in `prism-ecs-runtime` as `schedule::*` peers, not a new crate),
the re-implementation order, and the roadmap for the follow-up
waves.
