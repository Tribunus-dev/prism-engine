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
| 4 | **Session Lifecycle** | `Design` | Session | SessionConfig, SessionModels, SessionDevices, SessionLifecycle, ResidencyModelRef | kernel |
| 5 | **Work Scheduling** | `Design` | WorkItem | WorkItemComponent, WorkState, WorkLeaseComponent, ResourceClaimComponent, WorkPrerequisites, WorkOutput | kernel |
| 6 | **Execution Leases** | `Design` | — | ExecutionLease, LeaseOwner, LeaseTokenRange, KvSlot, KvOwnership, ExecutionOutput | kernel |
| 7 | **Compilation & Model Production** | `Design` | CompilationJob | CompilationJob, JobInput, JobConfig, JobOutput, JobLifecycle, ValidationReceipt, QuantizationPlan, CimagePromotion | kernel |
| 8 | **Agent & Tool Execution** | `Design` | Agent | AgentRun, AgentTask, AgentPhase, ToolInvocation, ToolOutcome, AgentMessage, AgentConfig, AgentLifecycle | kernel |
| 9 | **Multimodal Pipelines** | `Design` | Pipeline | Pipeline, PipelineStage, PipelineModality, InputArtifactRef, OutputArtifactRef, PipelineLifecycle, WorkLeaseRef | kernel |
| 10 | **Distributed Topology** | `Design` | Node | PeerIdentity, NodeMembership, PeerCapabilities, NodeTopology, TrustState, WorkerHealth, RemoteLease, RemoteCapabilityObservation | kernel |
| 11 | **Server & API Bridges** | `Design` | — | IngressRequest, ApiKey, RateLimiterState, RequestQueue, TransportSession, IngressLifecycle | kernel |
| 12 | **Persistence & Projections** | `Inventory` | — | — | kernel |
| 13 | **Dashboard** | `Inventory` | — | — | dashboard |

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

### Complete (ReplayVerified)
- **Artifact Ingestion** — LoadArtifactCommand, effect validation, transactional
  spawn, schema-bound components, replay through EventStore.

### Canonical (no legacy removal yet)
- **Device Discovery** — DiscoverDevicesCommand, idempotent create-or-update,
  stable identity types, ephemeral handle separation, desired/observed state
  split.

### Design (types exist, not wired into authority)
- **Session Lifecycle** — CreateSessionCommand with preflight (model+device validation),
    TransitionSessionCommand, schema-bound SessionConfig/SessionModels/SessionDevices.
    10 tests.
- **Work Scheduling** — WorkItemComponent, WorkState, WorkLeaseComponent, ResourceClaimComponent,
    CreateWorkCommand, LeaseWorkCommand, CompleteWorkCommand, FailWorkCommand, CancelWorkCommand.
    7 tests.
- **Execution Leases** — AcquireExecutionLeaseCommand, CompleteExecutionLeaseCommand,
    ExecutionLease/LeaseOwner/LeaseTokenRange/KvSlot/KvOwnership. 4 tests.
- **Compilation & Model Production** — CreateCompilationJobCommand, PromoteCimageCommand,
    SubmitValidationReceiptCommand, 8 component types, JobLifecycle with 6 states. 11 tests.
- **Agent & Tool Execution** — CreateAgentRunCommand, SubmitToolOutcomeCommand,
    9 component types, AgentPhase with 7 states. 9 tests.
- **Multimodal Pipelines** — CreatePipelineCommand with stage/artifact preflight,
    SubmitStageOutputCommand, 7 component types, PipelineLifecycle with 7 states. 8 tests.
- **Distributed Topology** — RegisterPeerCommand, ObserveWorkerCapabilityCommand,
    8 component types (PeerIdentity, TrustState, WorkerHealth, RemoteLease), TrustState
    with 5 states. 5 tests.
- **Server & API Bridges** — SubmitIngressRequestCommand, ResolveIngressCommand,
    IngressLifecycleTransitionCommand, 6 component types, IngressLifecycle with 6 states.
    11 tests.

### Shadow (constitutional path running, legacy comparison pending)
- **Model Deployment & Residency** — Schema-bound deployment, preflight validation,
     idempotent redeployment, replay safety. 18 tests covering entity/component
     attachment, effect failure/mismatch, stale outcome rejection, replay without
     fake allocations. Comparison target: `loader`, `residency`, `model_cache` modules.

## Wave Plan

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
