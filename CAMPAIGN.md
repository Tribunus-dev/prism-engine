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
| 13 | **Dashboard & Authority Purge** | `Inventory` | — | — | kernel |

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

## Current Migration State

### Inventory (not yet started)
- **Authority Purge (Wave 10)** — Adversarial audit complete. 11 high-severity legacy
  registries identified across 8 files. Dominant patterns: `world.add_component()` bypassing
  WorldTxn (60+ violations in production code), legacy HashMap registries for model
  lifecycle, cancellation, distillation, weight residency, trust, and session state.
  Root enabler: legacy CompWorld mutation methods (add_component, remove_component,
  add_resource, get_component_mut) in mod.rs directly access component_store.data.
  
  Top 11 HIGH-severity items requiring migration:
  1. AdapterRegistry (adapter/mod.rs) — model role assignment bypasses ECS
  2. ModelRegistry (server/models.rs) — model lifecycle outside ECS
  3. DistillationEngine (server/distill_worker.rs) — job lifecycle in HashMap
  4. CancellationManager (scheduling/cancellation.rs) — cancellation authority
  5. WeightCache (backend/residency.rs) — weight residency decisions
  6. TrustStore (registry/trust_store.rs) — provider trust decisions
  7. AppState (server/routes.rs) — composite anti-pattern with multiple HashMaps
  8. GLOBAL_PREFIX_CACHE (cache/prefix_cache.rs) — global mutable singleton
  9. ServerEngine (server/engine.rs) — session/request state
  10. HeterogeneousExecutor.routing_table (backend/heterogeneous_executor.rs) — routing
  11. AneBackend (backend/ane.rs) — ANE program/weight binding state
  
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
