# One Canonical Reality — Constitutional ECS Migration

Objective: Every authority-bearing subsystem operates through the canonical ECS.
No legacy map, registry, manager, cache, or database table can independently contradict the ECS.

## Status Legend

- **Inventory** — Authority identified, boundaries drawn
- **Design** — Types, schemas, lifecycle, commands specified
- **Shadow** — Constitutional path running alongside legacy, results compared
- **Canonical** — Constitutional path is authoritative (legacy may still observe)
- **LegacyRemoved** — Legacy write path deleted
- **ReplayVerified** — Restart recovery, replay determinism, stale rejection proven

## Methodology Status

The Cutover Protocol status describes whether the canonical path is authoritative. The
methodology status describes whether the code implementing the canonical path is clean.
A subsystem can be `Canonical` and still carry methodology debt; that debt blocks the
next cutover step until it is paid or formally waived. The methodology is owned by the
`prism-constitutional-rust-ecs` skill (`references/module-discipline.md`,
`references/rust-quality.md`, `references/project-absorption.md`,
`SKILL.md` §Propagation verification).

- **Clean** — passes all four methodology gates (Module cohesion, Rust quality, Project
  absorption, Propagation). Eligible to advance to the next state. Subsystem's
  Cutover Protocol status is unblocked.
- **Migrate** — has known methodology debt in one or more areas. A migration entry
  exists in the Methodology Migration Backlog below. Cannot advance to the next state
  until the debt is paid (the migration is part of the change) or formally waived.
- **Waived** — has known methodology debt that is explicitly waived for the current
  cutover. The waiver is recorded in the change's `Completion report`, names the
  invariant or test that justifies the waiver, and expires at the next major version.
  A waiver on a public API fails the Rust quality gate at review.

## Subsystem Registry

| # | Subsystem | Status | Entity Kinds | Schemas | Owner |
|---|-----------|--------|-------------|---------|-------|
| 1 | **Artifact Ingestion** | `ReplayVerified` | Artifact | ArtifactPath, ArtifactDigest, ArtifactMetadata, ArtifactLifecycle | kernel |
| 2 | **Device Discovery** | `Canonical` | Device | DeviceStableId, DriverFactoryId, BackendFamily, DeviceCapabilities, DeviceMemoryLimits, DeviceTopology, DeviceHealth, DeviceLifecycle, DesiredDeviceState, ObservedDeviceState, LastObservation, RuntimeHandleKey | kernel |
| 3 | **Model Deployment & Residency** | `Shadow` | Model, Residency | ModelId, ModelArtifactRef, ModelLifecycle, ResidencyDeviceRef, ResidencyMemoryClaim, ResidencyFormat, ResidencyLifecycle, AllocationToken | kernel |
| 4 | **Session Lifecycle** | `Shadow` | Session | SessionConfig, SessionModels, SessionDevices, SessionLifecycle, ResidencyModelRef | kernel |
| 3 | **Model Deployment & Residency** | `Canonical` | Model, Residency | ModelId, ModelArtifactRef, ModelLifecycle, ResidencyDeviceRef, ResidencyMemoryClaim, ResidencyFormat, ResidencyLifecycle, AllocationToken | kernel |
| 5 | **Work Scheduling** | `Shadow` | WorkItem | WorkItemComponent, WorkState, WorkLeaseComponent, ResourceClaimComponent, WorkPrerequisites, WorkOutput | kernel |
| 3 | **Model Deployment & Residency** | `Shadow` | Model, Residency | ModelId, ModelArtifactRef, ModelLifecycle, ResidencyDeviceRef, ResidencyMemoryClaim, ResidencyFormat, ResidencyLifecycle, AllocationToken | kernel |
| 6 | **Execution Leases** | `Shadow` | — | ExecutionLease, LeaseOwner, LeaseTokenRange, KvSlot, KvOwnership, ExecutionOutput | kernel |
| 7 | **Compilation & Model Production** | `Shadow` | CompilationJob | CompilationJob, JobInput, JobConfig, JobOutput, JobLifecycle, ValidationReceipt, QuantizationPlan, CimagePromotion | kernel |
| 8 | **Agent & Tool Execution** | `Shadow` | Agent | AgentRun, AgentTask, AgentPhase, ToolInvocation, ToolOutcome, AgentMessage, AgentConfig, AgentLifecycle | kernel |
| 9 | **Multimodal Pipelines** | `Shadow` | Pipeline | Pipeline, PipelineStage, PipelineModality, InputArtifactRef, OutputArtifactRef, PipelineLifecycle, WorkLeaseRef | kernel |
| 10 | **Distributed Topology** | `Shadow` | Node | PeerIdentity, NodeMembership, PeerCapabilities, NodeTopology, TrustState, WorkerHealth, RemoteLease, RemoteCapabilityObservation | kernel |
| 11 | **Server & API Bridges** | `Shadow` | — | IngressRequest, ApiKey, RateLimiterState, RequestQueue, TransportSession, IngressLifecycle | kernel |
| 12 | **Persistence & Projections** | `Design` | — | ReplayRegistry, EventStore (InMemory), Snapshot, ReplayEngine, ProjectionCheckpoint | kernel |
| 12 | **Persistence & Projections** | `Shadow` | — | FsEventStore (file-backed, durable-before-ack), ReplayRegistry (16 appliers), EventStore trait, InMemoryEventStore, Snapshot, ReplayEngine, ProjectionCheckpoint | kernel |
| 12 | **Persistence & Projections** | `Shadow` | — | FsEventStore (file-backed, durable-before-ack, restart recovery proven), ReplayRegistry (16 appliers), ReplayEngine::replay_into, restart recovery integration test | kernel |
|| 13 | **Dashboard & Authority Purge** | `LegacyRemoved` | — | — | kernel |

## Cutover Protocol

Every subsystem follows these 8 steps. A subsystem cannot advance directly from
`Design` to `Canonical` — it must first demonstrate equivalent or improved results
in shadow mode.

1. **Inventory** — Find all mutable maps, registries, queues, state machines,
   caches, global resources, database writes, background workers, files, and native
   objects that determine the subsystem's behavior.

2. **Classify** — Separate canonical state, execution-plane state, immutable
   evidence, derived projections, and pure computation.

3. **Specify** — Define stable identities, component schemas, ownership,
   lifecycle states, commands, effects, events, receipts, idempotency, and replay.

4. **Implement** — Schema registration + constitutional transactions alongside
   the legacy authority.

5. **Shadow** — Run constitutional implementation in parallel. Compare state
   transitions, results, receipts, resource claims, and failure behavior.

6. **Cutover** — Switch ECS path to authority. Legacy may observe but no longer
   independently mutate domain truth.

7. **Purge** — Remove legacy write path and cutover flag.

8. **Verify** — Prove restart recovery, replay determinism, stale-outcome
   rejection, failure atomicity, and projection rebuilding.

### Methodology promotion gates

The 8 cutover steps are necessary but not sufficient. Each transition between states
also requires satisfying the methodology gates below. A subsystem that has not paid its
methodology debt cannot advance, even if the constitutional path is correct. The
methodology status of each subsystem is recorded above (`Methodology Status`) and the
work to clear each gate is tracked in the `Methodology Migration Backlog` below.

**Inventory → Design** requires:
- **Module cohesion inventory.** All files in the subsystem's crates are identified,
  with their single-authority statement (or decomposition plan if the file owns more
  than one authority). New files in this transition are subject to the
  `references/module-discipline.md` thresholds.
- **Project absorption inventory.** Any file named after an external project is
  classified as `format adapter`, `hardware backend`, `vendored dependency`, or
  `absorbed pattern`. The first three categories keep their external name; the last
  is a migration entry in the Methodology Migration Backlog with a re-implementation
  target.

**Design → Shadow** requires the above plus:
- **Rust quality.** The constitutional path uses typed errors (`thiserror`-derived
  enums, no `anyhow::Error` in `prism-ecs-constitutional`, `prism-ecs-runtime`, or
  `prism-ecs-kernel`), encodes authority-bearing values as newtypes
  (`IdempotencyKey`, `Generation`, `Epoch`, `LeaseToken`, `ArtifactDigest`,
  `SchemaKey`, `CommandId`), and has no `unwrap` / `expect` / `panic!` / `unreachable!`
  / `todo!` / `unimplemented!` in production paths.
- **Propagation chain documented.** Every state-bearing change in the constitutional
  path has a named propagation chain (durable event → event store → replay applier →
  projection rebuild → read path → downstream consumer) and at least one
  projection-rebuild test.

**Shadow → Canonical** requires the above plus:
- **Module cohesion.** No file added to the subsystem during the Shadow run exceeds
  the hard thresholds (900 LOC or 35 public items) without a decomposition plan in
  the change's `Completion report`. Existing files above the threshold have a
  backlog entry in the Methodology Migration Backlog with a target decomposition.
- **Project absorption.** No `absorbed pattern` file remains in the subsystem
  without a re-implementation target in the Methodology Migration Backlog. A
  subsystem that contains an absorbed-pattern file cannot advance to `Canonical`
  until the file is re-implemented natively, re-exported as a deprecation shim, and
  scheduled for removal at the next minor version.
- **Propagation tests.** Every state-bearing change has a replay test in addition
  to the projection-rebuild test. The replay test re-derives state from durable
  events without rerunning effects and verifies identical committed state.
- **No `unsafe` outside the allowed crates.** If the subsystem is in a crate where
  `unsafe` is forbidden (constitutional, runtime, server, protocol), the
  Shadow run must produce zero `unsafe` blocks.

**Canonical → LegacyRemoved** requires the above plus:
- No legacy write path can independently contradict the world. The change that
  moves the subsystem to `LegacyRemoved` includes the deletion of the legacy
  writer, not just the deprecation flag.
- No `legacy_mutations` feature or equivalent escape hatch is reachable from the
  default build. The default `cargo build` produces a build with no legacy
  mutation path.

**LegacyRemoved → ReplayVerified** requires the above plus:
- Restart recovery integration test passes (a process restart reconciles or
  expires process-local resources and orphaned leases).
- Replay determinism test passes (live execution and replay produce identical
  committed state from the same durable events).
- Stale-outcome rejection test passes (a result produced before a fencing
  generation change is rejected; no canonical mutation follows).
- Failure atomicity test passes (a failed preflight or effect leaves zero
  canonical residue; rollback is complete).
- Projection rebuild test passes (delete the projection, rebuild from durable
  events, verify observable equivalence).

## Methodology Migration Backlog

The methodology gates above require subsystem hygiene. The backlog below is the
workspace-level work that must be paid before subsystems can advance through the
Cutover Protocol. Subsystem-specific rows in the `Current Migration State` section
reference these entries.

### How to regenerate the numbers

```bash
# Module cohesion (godfile candidates by LOC and pub-item count)
bash $SKILL_DIR/scripts/audit_authority.sh . --module-cohesion

# Rust quality (unwraps, expects, denied methods, lint warnings)
cargo clippy --workspace --all-targets 2>&1 | tee /tmp/clippy-baseline.log

# Project absorption (files named after external projects)
rg -l 'tinygrad|burn|candle|jax|bonsai|uop|tinyrun|fastai' crates/ 2>/dev/null
```

The baseline below is the current snapshot under default features, all targets.
Re-run before any state transition.

### Workspace Baseline (snapshot)

- **Module cohesion.** 38 files over the hard threshold (900 LOC or 35 public items).
  Top entries by combined LOC + pub count: `tinygrad_core.rs` (6762 LOC, 103 pub),
  `uop.rs` (6407, 76), `runtime.rs` (4746, 143), `cimage.rs` (3285, 194),
  `search.rs` (2775, 92), `schedule.rs` (2646, 84), `ecs.rs` (2581, 61).
- **Rust quality.** 89 `unwrap` / `expect` calls in production paths (the new
  constitutional `disallowed_methods` rule). Top hot files: `prism-mcp-core/src/protocol.rs`
  (33), `crates/prism-ecs-core/src/world.rs` (11), `crates/prism-gguf/src/lib.rs` (9),
  `crates/prism-plugin/src/lib.rs` (8), `prism-mcp-core/src/scheduler.rs` (5).
  128 total clippy warnings, 3 errors (pre-existing `not_unsafe_ptr_arg_deref` in
  `prism-plugin`, not from the constitutional config).
- **Project absorption.** 5 absorbed-pattern files in the canonical paths:
  `tinygrad_core.rs`, `uop.rs`, `bonsai_ternary.rs`, `bonsai_cimage.rs`,
  `turboquant_kv.rs`. All in canonical paths; none under a vendored exception.
- **Propagation.** All currently `Shadow` and `Canonical` subsystems have replay
  appliers registered in the ReplayRegistry (16 appliers in total per
  `Persistence & Projections` row below). The propagation gate is satisfied at
  the design level; replay tests at the Shadow → Canonical transition are the
  per-change evidence.

### Module Cohesion Backlog

Files over the hard threshold (900 LOC or 35 public items) in canonical paths.
The full list regenerates from the audit script. The migration plan for each file
is in `references/module-discipline.md` §Concrete decomposition patterns for Prism.

| File | LOC | Pub | Migration target |
|---|---:|---:|---|
| `crates/prism-spatial-ir/src/tinygrad_core.rs` | 6762 | 103 | `phase_graph/` directory (also project absorption) |
| `crates/prism-ecs-compile/src/uop.rs` | 6407 | 76 | `ir_value.rs` + `ir_op.rs` (also project absorption) |
| `crates/prism-ecs-compile/src/runtime.rs` | 4746 | 143 | Decompose by entity kind |
| `crates/prism-ecs-compile/src/cimage.rs` | 3285 | 194 | Decompose by authority |
| `crates/prism-ecs-compile/src/search.rs` | 2775 | 92 | Decompose by authority |
| `crates/prism-ecs-runtime/src/schedule.rs` | 2646 | 84 | One file per schedule stage |
| `crates/prism-ecs-compile/src/ecs.rs` | 2581 | 61 | Split per `EntityKind` |
| `crates/prism-ecs-server/src/runtime/server.rs` | 2284 | 7 | Split by ingress / router / serve / observe |
| `crates/prism-ecs-server/src/engine/bpe_tokenizer.rs` | 2256 | 38 | Split by tokenizer responsibility |
| `crates/prism-ecs-quantization/src/bonsai_ternary.rs` | 1995 | 54 | `ternary_quantization/` (also project absorption) |
| `crates/prism-ecs-quantization/src/bonsai_cimage.rs` | 1958 | 59 | `cimage_quantization/` (also project absorption) |
| `crates/prism-ecs-runtime/src/kernel.rs` | 1939 | 63 | One file per schedule stage |
| `crates/prism-ecs-kernel/src/cpu_backend.rs` | 1894 | 1 | Decompose by target path |
| `crates/prism-ecs-kernel/src/metal_dispatch.rs` | 1821 | 10 | Decompose by dispatch shape |
| `crates/prism-ecs-compile/src/evaluator.rs` | 1784 | 32 | Decompose by evaluation phase |
| `crates/prism-ecs-compile/src/compiler.rs` | 1777 | 11 | `ir_build.rs` + `plan_apply.rs` (build vs apply) |
| `crates/prism-spatial-ir/src/execution_plan.rs` | 1583 | 72 | Decompose by plan element |
| `crates/prism-ecs-quantization/src/turboquant_kv.rs` | 1566 | 33 | `kv_quantization/` (also project absorption) |
| `crates/prism-amd-npu-runtime/src/codegen.rs` | 1553 | 10 | Split by codegen phase |
| `crates/prism-spatial-ir/src/evolution.rs` | 1535 | 44 | Decompose by evolution operator |

The remaining ~18 files in the full audit output are not listed here; they are
recoverable from a single `audit_authority.sh --module-cohesion` run. The table
above is the priority queue, ordered by LOC.

### Project Absorption Backlog

| File | Target name | Target authority |
|---|---|---|
| `crates/prism-spatial-ir/src/tinygrad_core.rs` | `phase_graph/` directory (`value.rs`, `op.rs`, `phase.rs`, `spatial.rs`, `abi.rs`) | Phase graph semantics in the spatial IR |
| `crates/prism-ecs-compile/src/uop.rs` | `ir_value.rs` + `ir_op.rs` | IR value and operation types in the compile path |
| `crates/prism-ecs-quantization/src/bonsai_ternary.rs` | `ternary_quantization/` directory | Ternary quantization in Prism |
| `crates/prism-ecs-quantization/src/bonsai_cimage.rs` | `cimage_quantization/` directory | CImage-targeted quantization in Prism |
| `crates/prism-ecs-quantization/src/turboquant_kv.rs` | `kv_quantization/` directory | KV-cache quantization in Prism |

The re-implementation pattern, the exception categories (format adapters, hardware
backends, vendored dependencies — all exempt), and the migration sequence are in
`references/project-absorption.md`.

### Rust Quality Backlog

89 `unwrap` / `expect` calls in production paths. The migration target is zero;
each violation either becomes `?` propagation, a typed error, or a `// WAIVER`
with a justification. Per-file priority queue, top entries first:

| File | Count | Plan |
|---|---:|---|
| `prism-mcp-core/src/protocol.rs` | 33 | Migrate to `?` and typed errors; this is the ingress layer, no `anyhow` constraint but no-panic still applies |
| `crates/prism-ecs-core/src/world.rs` | 11 | Public API surface; change `add_component` / `remove_component` / `get_component_mut` to return `Result` so the no-panic discipline holds at the foundation |
| `crates/prism-gguf/src/lib.rs` | 9 | Format adapter; typed errors via `thiserror`, no `unwrap` in parser hot path |
| `crates/prism-plugin/src/lib.rs` | 8 | FFI boundary; the `unsafe` constraint already forces typed error handling — extend to the rest of the file |
| `prism-mcp-core/src/scheduler.rs` | 5 | Same as protocol.rs |
| `crates/prism-multimodal/src/multimodal/vision_encoder.rs` | 4 | Subsystem-internal; migrate as part of `Multimodal Pipelines` cutover |
| `crates/prism-ecs-core/src/column.rs` | 4 | Internal storage primitive; `Result` on the column-mutation API |
| `crates/prism-video/src/lib.rs` | 3 | Subsystem-internal; migrate as part of `Multimodal Pipelines` cutover |
| `crates/prism-ecs-codec/src/lib.rs` | 3 | Serialization layer; typed errors per codec format |
| `crates/prism-multimodal/src/lib.rs` | 2 | Subsystem-internal |
| `prism-mcp-core/src/subprocess.rs` | 2 | Subprocess management; `Result` with `io::Error` propagation |
| `build.rs` | 1 | Build script; this category is acceptable per the rust-quality `#[allow(clippy::unwrap_used)]` test convention if scoped to the build path only |
| `crates/prism-audio/src/lib.rs` | 1 | Subsystem-internal |
| `crates/prism-gguf/src/writer.rs` | 1 | Format adapter |
| `crates/prism-ecs-core/src/query.rs` | 1 | Internal storage primitive |

The full per-violation list regenerates from the audit script. The table above is
the priority queue, ordered by file-level violation count. Subsystem ownership is
indicated in the right column where it is unambiguous; cross-cutting violations
(scheduler, world) are the highest leverage because they touch every caller.

### Propagation Backlog

No subsystem is currently blocked on a missing propagation chain — every
`Shadow` and `Canonical` subsystem has its replay applier registered in
`ReplayRegistry`. The propagation gate's per-change evidence (projection-rebuild
test + replay test) is collected at the change level, not the subsystem level.
A change that fails to provide the evidence fails the propagation gate at review,
independent of CAMPAIGN state.

## Current Migration State

### Inventory (not yet started)
- **Authority Purge (Wave 10)** — Adversarial audit complete. 12 high-severity legacy
  registries identified across 8 files. 7 REMOVED, 4 preserved as legitimate backends.
  Dominant patterns: `world.add_component()` bypassing
  WorldTxn (60+ violations in production code), legacy HashMap registries for model
  lifecycle, cancellation, distillation, weight residency, trust, and session state.
  Root enabler: legacy CompWorld mutation methods (add_component, remove_component,
  add_resource, get_component_mut) in mod.rs directly access component_store.data.
  
  ### REMOVED (6 total — files deleted from disk):
  1. ~~AdapterRegistry (model_adapter/mod.rs)~~ — DELETED. 158 lines. Single mod.rs, no remaining external references.
  2. ~~ModelRegistry (server/models.rs)~~ — DELETED. 149 lines. server/mod.rs module decl removed.
  3. ~~WeightCache (backend/residency.rs)~~ — DELETED. 432 lines. Trait methods normalized, check_transfer removed.
  4. ~~TrustStore (registry/trust_store.rs)~~ — DELETED. 57 lines. registry/mod.rs module decl removed.
  5. ~~GLOBAL_PREFIX_CACHE (cache/prefix_cache.rs)~~ — DELETED. 579 lines. cache/mod.rs module decl removed.
  6. ~~CancellationManager~~ — DELETED (Wave 9). 1058 lines. Orphan.
  7. ~~ServerEngine~~ — DELETED (Wave 9). 300 lines. Orphan.
  
  ### PRESERVED (4 files with multiple cfg gates — legitimate backends, not pure registries)
  1. DistillationEngine (server/distill_worker.rs) — behind `prism-backend` only. Legitimate production service.
  2. AppState (server/routes.rs) — behind `mlx-backend` gate. MLX route handler.
  3. HeterogeneousExecutor — behind `macos + mlx|prism` gate. Production executor.
  4. AneBackend — behind `mlx|prism|ane-executor` gate. Production ANE backend.
  
  `legacy_mutations` feature reduced from 9 to 0 file gates — completely removed.
  
  ### Constitutional mode is now the DEFAULT
  - `legacy_mutations` REMOVED from default features in compute-core/Cargo.toml
  - `cargo build` (no extra features) — constitutional-only mode: 0 errors
  - `cargo test -p tribunus-compute-core --lib --features prism-backend` — 2552 passed, 0 failures
  - `cargo test -p tribunus-compute-core --lib ecs::constitutional --no-default-features --features prism-backend` — 177 passed, 0 failures
  - Legacy source files removed from disk — `legacy_mutations` feature removed from all module declarations
  - Remaining 4 backends compile under their original feature gates (prism-backend/mlx-backend), not legacy_mutations
  
  Full audit reports:
  - local://audit-direct-store.md (60 production violations)
  - local://audit-legacy-registries.md (11 HIGH, 7 MEDIUM, 20 LOW)
  - local://audit-spawn-patterns.md (11 direct spawn violations)
  - local://audit-projection-authority.md (no violations — projections correctly tiered)

### Complete (ReplayVerified)
- **Artifact Ingestion** — LoadArtifactCommand, effect validation, transactional
  spawn, schema-bound components, replay through EventStore.

### Canonical (no legacy removal yet)
- **Device Discovery** — DiscoverDevicesCommand, idempotent create-or-update,
  stable identity types, ephemeral handle separation, desired/observed state
  split.

### Design (types exist, not wired into authority)
  - **Persistence & Projections** — ReplayRegistry with 12 event kind → replay applier
    mappings. ReplayEngine::replay_into for batch reconstruction. InMemoryEventStore,
    Snapshot, ProjectionCheckpoint. Full replay integration test proves
    artifact→device→model→session→work workflow survives replay.

### Shadow (constitutional path running, legacy comparison pending)
- **Model Deployment & Residency** — Schema-bound deployment, preflight validation,
     idempotent redeployment, replay safety. 18 tests covering entity/component
     attachment, effect failure/mismatch, stale outcome rejection, replay without
     fake allocations. Comparison target: `loader`, `residency`, `model_cache` modules.
- **Session Lifecycle** — CreateSessionCommand, TransitionSessionCommand, 10 tests.
    replay_session_admitted registered in ReplayRegistry.
- **Work Scheduling** — 5 commands (Create/Lease/Complete/Fail/Cancel), 7 tests.
    replay_work_created registered in ReplayRegistry.
- **Execution Leases** — AcquireExecutionLeaseCommand, CompleteExecutionLeaseCommand,
    4 tests. replay_lease_acquired/completed registered.
- **Compilation & Model Production** — 3 commands, 8 types, 11 tests.
  - **Compilation & Model Production** — 3 commands, 8 types, 11 tests. replay_compilation_job_created registered.
  - **Agent & Tool Execution** — 2 commands, 9 types, 9 tests. replay_agent_run_created registered.
  - **Multimodal Pipelines** — 2 commands, 7 types, 8 tests. replay_pipeline_created registered.
  - **Distributed Topology** — 2 commands, 8 types, 5 tests. replay_peer_registered registered.
  - **Server & API Bridges** — 3 commands, 6 types, 11 tests. replay_ingress_request_submitted registered.
  - **Persistence & Projections** — FsEventStore (file-backed, durable-before-ack, restart recovery proven),
    ReplayRegistry with 16 appliers, ReplayEngine::replay_into, InMemoryEventStore, Snapshot. 177 tests.
  - **Legacy Spawn Guard (feature-gated)** — `legacy_mutations` feature in Cargo.toml,
    enabled by default for backward compatibility. Without the feature, `CompWorld::new()`
    sets `direct_mutation_allowed=false`, causing panic on any `world.spawn()` or
    `world.add_component()` outside WorldTxn. Constitutional tests verified: 177/177 pass
    WITHOUT `legacy_mutations` feature. All direct mutations converted to WorldTxn in
    test helpers and inline setups. Proves the constitutional kernel needs no legacy
    mutation paths.

## Wave Plan
## Authority Purge Registry

### Wave 1: Placement Authority (current)
Model deployment + residency. Canonical model entities from artifacts,
device placement, weight residency lifecycle, allocation ownership.

### Wave 2: Session & Work Authority
Session admission/lifecycle + authoritative work scheduler.

### Wave 3: Runtime Execution Authority
Bounded execution leases, KV-cache identity, outcome validation.

### Wave 4: Compilation & Model-Production Authority
CImage compiler pipeline, quantization/distillation decisions as canonical plans.

### Wave 5: Multimodal & Generation Pipelines
Vision, audio, diffusion — all through same artifact/work/deployment primitives.

### Wave 6: Agent, Tool, & External-Service Authority
Agent runs, tool calls, effects — validated external outcomes.

### Wave 7: Distributed Topology & Worker Authority
Remote worker identity, capability advertisement, work routing.

### Wave 8: Product Ingress & Application Bridges
Server/API/FFI/Swift — command producers and projection consumers.

### Wave 9: Persistence, Projections, & Dashboard
Durable-before-ack, snapshots, replay, rebuildable projections.

### Wave 10: Authority Purge
Adversarial audit, remove legacy paths, enforce constitutional-only mutations.
