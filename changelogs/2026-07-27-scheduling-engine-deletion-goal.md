# Goal: Delete the legacy engine's scheduling subsystem

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; not started.
**Follow-up to:** `be7780b2` (scheduling migration: state + systems + adapters + evidence + metrics + tests + architecture test all landed).
**Source:** `compute-core/src/ecs/scheduling/` — 58 files, 16,236 LOC.

## Objective

Delete `compute-core/src/ecs/scheduling/` from the engine workspace, by migrating every external caller of the engine's scheduling surface to the constitutional crates (`prism-ecs-runtime`, `prism-ecs-kernel`, `prism-ecs-server`, `prism-ecs-protocol`).

The engine is currently a sibling workspace member with 100+ pre-existing build errors. The engine's own `scheduling/` directory is the legacy duplicate of the constitutional scheduling surface (now in `prism-ecs-runtime/src/scheduling/` and `prism-ecs-kernel/src/backend/`). Until the engine's callers migrate, the engine file must remain — but the architecture test (`workspace_contains_no_legacy_scheduling_imports`) prevents new importers from appearing outside the migration inventory.

## Success criteria

The engine's `compute-core/src/ecs/scheduling/` directory can be `git rm`'d when **all** of the following are true:

1. **Zero external callers** — every file outside `compute-core/src/ecs/scheduling/` that imports `crate::ecs::scheduling::*` is updated to use the constitutional surface.
2. **The engine compiles** with the deletion — no new build errors. (The 100+ pre-existing errors are out of scope; the goal is "no NEW errors from the deletion".)
3. **The constitutional surface is wired** — every migration step has a corresponding constitutional-side test that exercises the new path.
4. **The architecture test passes** — `workspace_contains_no_legacy_scheduling_imports` (already passing) continues to pass.
5. **No compatibility facade** — no `pub use` re-export shims from engine to constitutional crates. The deletion is the proof: the engine has no legacy surface to re-export.

## Scope: 13 external engine callers

A ripgrep of `use crate::ecs::scheduling::` against the engine (`compute-core/src/`, excluding the scheduling directory itself) returns 13 files. Each is the migration target for one commit.

| # | Engine file | Imports (engine) | Constitutional destination | Migration step |
|---:|---|---|---|---|
| 1 | `core/engine.rs` | `token_budget::PhaseKind` and the full `scheduling::{...}` block | `prism_ecs_runtime::scheduling::state::token_budget::PhaseKind`; map other symbols per their type | E-1 |
| 2 | `core/hybrid_profile.rs` | `Batch`, `Slot` (in test code) | `prism_ecs_runtime::scheduling::state::batch::{Batch, Slot}` | E-2 |
| 3 | `compiler/deployment_compiler.rs` | `compilation_job_bridge::CompilationJobBridge` | `prism_ecs_runtime::scheduling::systems::compilation_job_bridge::CompilationJobBridge`; or the constitutional `prism_ecs_compile` job bridge | E-3 |
| 4 | `compiler/lifecycle_coordinator.rs` | `unified_scheduler::SchedulerRunner`, `SchedulerConfig` | `prism_ecs_runtime::scheduling::systems::unified_scheduler::{SchedulerRunner, SchedulerConfig}` | E-4 |
| 5 | `exo/autoscaler.rs` | `InferenceTelemetry` (advisory metric) | `prism_ecs_runtime::scheduling::metrics::inference_telemetry::InferenceTelemetry` (new; the type doesn't exist in metrics yet — it was a `mod.rs` engine type) | E-5 |
| 6 | `inference/inference_session_state.rs` | `receipts::PhaseReceipt` | `prism_ecs_runtime::scheduling::evidence::scheduling_receipts::PhaseReceipt` | E-6 |
| 7 | `inference/inference_step_state.rs` | `activation_binding::CurrentActivation`, `receipts::PhaseReceipt` | `prism_ecs_runtime::scheduling::state::activation_binding::CurrentActivation` (new placeholder); `prism_ecs_runtime::scheduling::evidence::scheduling_receipts::PhaseReceipt` | E-7 |
| 8 | `inference/phase_engine_adapter.rs` | `phase_engine::PhaseEngine` (the system) | `prism_ecs_runtime::scheduling::systems::phase_advancement::PhaseEngine` (new) | E-8 |
| 9 | `runtime/resources/worker_ingress_queue.rs` | `ingress_bridge::IngressBridge` | `prism_ecs_runtime::ports::ingress::IngressBridge` (new; not yet in constitutional) | E-9 |
| 10 | `runtime/serving/model_instance.rs` | `unified_scheduler::SchedulerRunner`, `SchedulerConfig` | `prism_ecs_runtime::scheduling::systems::unified_scheduler::{SchedulerRunner, SchedulerConfig}` | E-10 |
| 11 | `runtime/systems/inference/session.rs` | `execution_context::ExecutionContext`, `phase_engine::PhaseEngine`, `receipts::PhaseReceipt` | split: `state::execution_context::ExecutionContext`, `systems::phase_advancement::PhaseEngine`, `evidence::scheduling_receipts::PhaseReceipt` | E-11 |
| 12 | `runtime/systems/worker/bridge.rs` | `agent_bridge::AgentBridge` | `prism_ecs_runtime::scheduling::systems::agent_bridge::AgentBridge` (already moved; the engine file's caller needs to update) | E-12 |
| 13 | `backend/flex_dispatch/selection.rs` | `outlier_detector::OutlierDetector` | `prism_ecs_runtime::scheduling::metrics::outlier_detector::OutlierDetector` | E-13 |

**Self-references inside the engine's `scheduling/` directory** (14 files import each other): these are the engine's own internal calls. When the directory is deleted, the self-references go with it. They are NOT in the migration scope; they are deleted by `git rm`.

## Engine file dependencies (engine Cargo.toml)

The engine does not currently depend on `prism-ecs-runtime`. To update the engine callers, the engine's `Cargo.toml` needs:

```toml
prism-ecs-runtime = { path = "../crates/prism-ecs-runtime" }
prism-ecs-kernel = { path = "../crates/prism-ecs-kernel" }
```

Both crates build cleanly. The engine depending on them is safe (the engine's existing 100+ build errors are pre-existing, not introduced by the new dependency).

## Phases

### Phase A: Add engine dependencies (one commit, E-0)

Add `prism-ecs-runtime` and `prism-ecs-kernel` to `compute-core/Cargo.toml`. Verify the engine's existing 100+ build errors are unchanged (no new errors from the new deps).

### Phase B: Migrate the 13 external callers (commits E-1 through E-13)

Each commit migrates one caller:
1. Update the import path from `crate::ecs::scheduling::X` to `prism_ecs_runtime::scheduling::Y` (or kernel/server/protocol as appropriate).
2. If the constitutional-side type doesn't exist yet (e.g., `PhaseEngine`, `SchedulerRunner`, `SchedulerConfig`, `InferenceTelemetry`, `CurrentActivation`, `IngressBridge`), add the type to the constitutional side first (in the same commit).
3. Build the engine. (Pre-existing errors expected; the goal is "no NEW errors from the migration".)
4. Run the engine's tests that exercise the caller. (The engine doesn't build today; this is a future step.)
5. Run the constitutional-side tests to confirm the new path is wired.
6. Commit with a `chore(engine):` prefix.

Each commit is bisectable: the engine still has its pre-existing build state, the constitutional side has a new test, and the architecture test continues to pass.

### Phase C: Pre-deletion verification (one commit, E-14)

Before deleting, verify:
1. `rg "use crate::ecs::scheduling::" compute-core/src/` returns ONLY files inside `compute-core/src/ecs/scheduling/`.
2. The architecture test still passes.
3. The constitutional surface has a passing test for each migrated path.

If any of these fail, do NOT delete. Investigate which caller was missed.

### Phase D: Engine file deletion (one commit, E-15)

`git rm -r compute-core/src/ecs/scheduling/`. This is the proof of "no compatibility facade": the engine has no legacy surface to import from. The constitutional surface is the only scheduling surface.

After this commit:
- The engine has its 100+ pre-existing build errors, none of which are scheduling-related.
- The architecture test still passes.
- The constitutional surface is the only home for scheduling.

### Phase E: Engine Cargo.toml cleanup (one commit, E-16)

After the deletion, the engine's `prism-ecs-runtime` and `prism-ecs-kernel` deps may not be needed if no other engine code uses them. Audit and remove unused deps. The `compute-core.legacy/` archaeology snapshot is a separate concern; it does not affect this goal.

## Tracking

| Step | Status | Commit |
|------|--------|--------|
| E-0: Add engine deps | pending | — |
| E-1: core/engine.rs | pending | — |
| E-2: core/hybrid_profile.rs | pending | — |
| E-3: compiler/deployment_compiler.rs | pending | — |
| E-4: compiler/lifecycle_coordinator.rs | pending | — |
| E-5: exo/autoscaler.rs | pending | — |
| E-6: inference/inference_session_state.rs | pending | — |
| E-7: inference/inference_step_state.rs | pending | — |
| E-8: inference/phase_engine_adapter.rs | pending | — |
| E-9: runtime/resources/worker_ingress_queue.rs | pending | — |
| E-10: runtime/serving/model_instance.rs | pending | — |
| E-11: runtime/systems/inference/session.rs | pending | — |
| E-12: runtime/systems/worker/bridge.rs | pending | — |
| E-13: backend/flex_dispatch/selection.rs | pending | — |
| E-14: Pre-deletion verification | pending | — |
| E-15: git rm scheduling/ | pending | — |
| E-16: Engine Cargo.toml cleanup | pending | — |

## Critical path

E-0 must land first (engine needs the new deps).
E-1, E-2 are independent and can be done in parallel.
E-3, E-4 are independent (different compiler-side files).
E-5 is independent.
E-6, E-7 are independent (different inference files).
E-8 is the largest single migration (`phase_engine` is 558 LOC; the engine caller is `phase_engine_adapter.rs` which is a thin adapter).
E-9 needs a new `prism-ecs-runtime::ports::ingress` module (not yet in the constitutional surface).
E-10 is independent of E-4 but uses the same constitutional type (`SchedulerRunner`, `SchedulerConfig`).
E-11 is the largest single migration in scope (3 imports, system + state + evidence).
E-12 is the simplest (constitutional `AgentBridge` already exists).
E-13 is independent.

E-14 must come after E-1 through E-13.
E-15 is blocked by E-14.
E-16 is blocked by E-15.

## Risks

1. **Engine build errors compound** — the engine has 100+ pre-existing build errors. Adding new dependencies may add 0-5 new errors (for example, missing trait impls in the engine that the new dependency exposes). Mitigation: build the engine after E-0 and confirm no new errors.
2. **The 13 callers may transitively call into the engine's scheduling/ directory** — `phase_engine::PhaseEngine` likely calls into other engine scheduling types. Migrating one caller may require migrating its callees too. Mitigation: each E-N commit is bisectable; if a callee blocks, add a follow-up E-N.1 commit.
3. **Constitutional types don't exist for some imports** — `PhaseEngine`, `SchedulerRunner`, `SchedulerConfig`, `InferenceTelemetry`, `CurrentActivation`, `IngressBridge` are engine types not yet in the constitutional crates. They must be added (or replaced) in the same commit as the caller migration. This is a scope expansion per call.
4. **The engine's `compute-core.legacy/` archaeology snapshot** — also imports the engine's scheduling/. It is not part of the workspace build, so the deletion does not affect it. The snapshot is preserved for archaeology.

## Estimated effort

Each E-N commit is 5-15 minutes of mechanical work (update import, run tests, commit). The constitutional-side type additions for missing types (E-5, E-8, E-9, E-11) are 30-60 minutes each.

Total: 13 caller migrations + 3 verification/deletion commits = 16 commits. ~3-5 hours of focused work.

## Proof-of-pattern

The migration is reversible until E-15. Every commit before E-15 can be reverted without losing the constitutional surface. After E-15, the engine file is gone; the constitutional surface is the only home.

The architecture test (`workspace_contains_no_legacy_scheduling_imports`) is the safety net: if any commit before E-15 introduces a new importer, the test fails and the commit is blocked.
