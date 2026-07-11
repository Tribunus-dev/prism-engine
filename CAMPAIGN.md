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
| 3 | **Model Deployment & Residency** | `Design` | Model, Residency | — | kernel |
| 4 | **Session Lifecycle** | `Design` | Session | SessionLifecycle, InferencePhase | kernel |
| 5 | **Work Scheduling** | `Design` | WorkItem | WorkItem, WorkState, WorkLease | kernel |
| 6 | **Inference Execution** | `Inventory` | — | — | runtime |
| 7 | **Compilation & Model Production** | `Inventory` | — | — | compiler |
| 8 | **Multimodal Pipelines** | `Inventory` | — | — | multimodal |
| 9 | **Agent & Tool Execution** | `Inventory` | — | — | agent |
| 10 | **Distributed Topology** | `Inventory` | — | — | distributed |
| 11 | **Server & API Bridges** | `Inventory` | — | — | server |
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
- **Session Lifecycle** — SessionLifecycle + InferencePhase enums in lifecycle.rs
- **Work Scheduling** — WorkItem, WorkState, Scheduler types in scheduler.rs
- **Model Deployment & Residency** — Next to implement

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
