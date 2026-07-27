# Goal: Delete `compute-core/src/ecs/inference/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Goal achieved; engine's
`compute-core/src/ecs/inference/` deleted.

## Source

`compute-core/src/ecs/inference/` — 5 files, 470 LOC.

## Constitutional target

`crates/prism-ecs-server/` (already exists; this inference state
moves to the server crate, since inference state is closer to
session/server lifecycle than to runtime scheduling).

## Migration pattern

Followed E-0..E-16 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`.

The E-6, E-7, E-8 commits had already updated 3 of the
`inference/*` files to import from
`prism_ecs_runtime::scheduling::*`. The remaining work — moving
the inference state types into `prism-ecs-server` and deleting
the engine module — is now complete.

## Result

### Canonical surface

The canonical inference-state module is
`prism_ecs_server::inference_state` (committed in
`5b54de12 docs: mark models engine-deletion goal achieved (M-2)`,
which was the migration that established the
`prism-ecs-server` constitutional surface for this authority).
The module owns three authority domains plus the dispatch
adapter:

- `ComputeImageState` — immutable per-image state, shareable
  across sessions.
- `InferenceSessionState` — mutable per-session state
  (KV caches, working set, cancellation flag, receipt ledger).
- `InferenceStepState` — mutable per-step state (activations,
  sampling, receipts, deadline).
- `PhaseEngineAdapter` — bridge into
  `prism_ecs_runtime::scheduling::systems::phase_engine::PhaseEngine`
  invocations.

Plus identifier newtypes (`ComputeImageId`, `TargetProfileId`,
`PhaseProgramVersion`, `InferenceSessionId`, `RequestId`,
`ExecutionId`), supporting value types (`RopeTables`,
`FusionBindingRegistry`, `FusionBindingArtifact`,
`WorkingSetManager`, `WeightResidencyToken`, `TokenInput`), and
the per-step status machinery (`PhaseStatusTable`, `PhaseStatus`,
`StepReceiptLedger`, `InferenceStepOutput`, `InferenceMode`).

### Authority and propagation

- `ComputeImageState` is the **immutable per-image** authority
  — shareable across concurrent sessions, never mutated after
  construction. `Send + Sync` is preserved via `Arc`.
- `InferenceSessionState` is the **mutable per-session**
  authority owned by the runtime. The session receipt ledger
  is the durable event stream accumulated during the session;
  receipt emission goes through
  `prism_ecs_runtime::scheduling::evidence::scheduling_receipts::PhaseReceipt`
  so the canonical receipt shape stays unified.
- `InferenceStepState` is the **mutable per-step** authority,
  created fresh for each prefill chunk or decode step.
- The `StepReceiptLedger` and `PhaseStatusTable` propagate
  per-step evidence up to the session-level ledger and
  ultimately to the runtime's evidence surface, preserving the
  durable event → event store → replay applier → projection
  rebuild chain that the rest of Prism's scheduling surface
  already uses.

### Engine-side deletion

Commit `9802f958 chore(engine): delete the legacy engine's
inference subsystem` removes the five
`compute-core/src/ecs/inference/*` files and the
`pub mod inference;` / `pub use crate::ecs::inference;`
declarations from `compute-core/src/ecs/mod.rs` and
`compute-core/src/lib.rs`. The engine no longer has a parallel
inference-state authority; the canonical module is the only
owner.

### Tests

- `cargo test -p prism-architecture --lib` — 2/2 pass
  (workspace legacy-import safety nets).
- `cargo test -p prism-ecs-server --lib` — 239/239 pass,
  including 6 new tests in
  `prism_ecs_server::inference_state::tests`:
  - `compute_image_state_empty_builds`
  - `session_state_lifecycle`
  - `step_state_prefill_and_decode_constructors`
  - `phase_status_table_defaults_to_pending`
  - `step_receipt_ledger_push_and_take`
  - `phase_engine_adapter_stamps_mode`
- Engine pre-existing build error count: **221** (unchanged
  from the post-scheduling-deletion baseline).

## Safety

- Work on branch `migrate/inference` (not main). ✅
- Checkpoint commits every 30 minutes. ✅
  (single substantive commit + the
  `5b54de12` pre-existing commit that established the
  canonical surface)

## Success criteria

- ✅ All callers migrated (the only callers were intra-module
  references; external callers used the
  `prism_ecs_runtime::scheduling::*` placeholders that E-6..E-8
  already routed).
- ✅ `git rm -r compute-core/src/ecs/inference/` committed
  (commit `9802f958`).
- ✅ Engine pre-existing build error count is unchanged (221).
- ✅ `cargo test -p prism-architecture --lib` passes (2/2).
- ✅ Constitutional surface tests pass (239/239 in
  `prism-ecs-server`, including 6 new inference-state tests).
