# ADR-005: ECS-Native MLIR/IREE/TableGen Absorption

**Status:** Accepted plan; implementation in progress. Wave 1 is complete and Waves 2–23 are pending.

**Supersedes:** The informal Waves 12–15 plan from session 2026-07-15. This document is the authoritative plan.

---

## Context

Prism Engine produces compiled inference artifacts (CImages) consumed by a production ECS runtime on Apple Silicon and, increasingly, CUDA, Vulkan, and CPU targets. The experimental MLIR path currently depends on an external LLVM 22 toolchain, `mlir-sys`, Melior, and TableGen bindings. IREE is an upstream compatibility target for the planned portable compiler/runtime path; it is not yet a production dependency or an absorbed subsystem in this repository.

This dependency carries four concrete costs:

1. **Compile-time and toolchain complexity.** LLVM 22, `mlir-tblgen`, `mlir-sys`, Melior, and the Rust `tblgen` crate must all build and stay compatible. Every Rust compiler upgrade risks breaking the FFI boundary.
2. **Data-model impedance.** The ECS world (component queries, entity stores, `WorldTxn`, receipt buses) and MLIR's arena-backed IR have different identity, mutation, and ownership models. Crossing that boundary requires explicit adaptation and makes transactional provenance harder.
3. **Parallelism is not governed by Prism.** MLIR can run nested operation pipelines concurrently when pass invariants permit it. Prism cannot currently schedule that work through its own resource, evidence, cancellation, and admission policies because the IR lives outside the World.
4. **No canonical Prism evidence chain through compilation.** MLIR provides pass instrumentation and failure reproducers, but it does not emit Prism's typed receipts. `CompileEvent` and `LifecycleReceiptBundle` therefore require an explicit adapter or an ECS-native execution path.

---

### Wave 1: Placement Authority

**Status:** Complete.

**Deliverable:** Model deployment and device residency through ECS. Canonical model entities from artifacts. Device placement, weight residency lifecycle, allocation ownership as components.

| Legacy | ECS replacement |
|---|---|
| ModelRegistry (~150L) | Model entity: ModelId, ModelArtifactRef, ModelLifecycle |
| WeightCache (~430L) | Residency entity: ResidencyDeviceRef, ResidencyMemoryClaim, ResidencyFormat |
| Device state in ad-hoc structs | Device entity: DeviceStableId, DriverFactoryId, BackendFamily, DeviceCapabilities, DeviceHealth |

**Gate:** `DiscoverDevicesCommand` produces idempotent create-or-update. Model deployment through `CreateResidency` transaction. Replay recovers identical entity state.

---

### Wave 2: Session & Work Authority

**Status:** Pending.

**Dependency:** Wave 1 placement authority. A session cannot be admitted against a model or device whose identity and residency are not canonical.

**Deliverable:** Session lifecycle and work scheduling become authoritative ECS state. Existing session and work types are scaffolding until every production admission, scheduling, cancellation, and completion path uses them and legacy state can no longer contradict the World.

| Command | Effect |
|---|---|
| `CreateSessionCommand` | Resolve canonical model/residency identities, enforce admission policy, and spawn `SessionConfig`, `SessionModels`, `SessionDevices`, and `SessionLifecycle`. |
| `TransitionSessionCommand` | Apply a validated lifecycle transition with expected epoch, terminal-state, and idempotency checks. |
| `CreateWorkItem` | Create work with session owner, prerequisites, resource claims, deadline, priority, and immutable input identity. |
| `LeaseWorkItem` | Atomically assign a bounded lease to one worker and reject duplicate or stale claims. |
| `CompleteWorkItem` / `FailWorkItem` / `CancelWorkItem` | Commit one terminal outcome, release claims, attach output or failure evidence, and reject late results. |

| Canonical component | Required semantics |
|---|---|
| `SessionConfig` | Immutable request policy, generation binding, tokenizer identity, limits, and serving profile. |
| `SessionModels`, `SessionDevices` | Generation-safe references to admitted models, residencies, and devices. |
| `SessionLifecycle` | Explicit state machine from requested through admitted, active, draining, and terminal. |
| `WorkItemComponent`, `WorkState` | Stable work identity and transition state. |
| `WorkLeaseComponent` | Lease owner, generation, deadline, heartbeat, and fencing token. |
| `ResourceClaimComponent` | Memory, compute, KV, backend, and concurrency claims. |
| `WorkPrerequisites`, `WorkOutput` | Dependency graph and content-addressed terminal result. |

**Authority cutover:** Inventory the server registry, scheduler queues, cancellation maps, session caches, and background workers that currently decide session or work truth. Run ECS commands in shadow mode, compare every transition and scheduling decision, then switch protocol handlers and runtime producers to command-only mutation. Legacy structures may remain read-only projections during one qualification window before deletion.

**Failure and replay requirements:** Duplicate admission is idempotent, stale leases cannot complete work, cancellation wins according to a documented race policy, restart reconstructs active sessions and pending work, and orphaned leases expire without accepting late output.

**Gate:** Production OpenAI/Ollama request paths create and drive sessions through these commands. Concurrent lease, cancellation, retry, and stale-result tests pass. Restart replay reconstructs identical session/work state and resumes or expires leases according to policy. Repository search finds no independent mutable session registry, work queue, or cancellation authority.

---

### Wave 3: Runtime Execution Authority

**Status:** Pending.

**Dependency:** Wave 2 session and work authority.

**Deliverable:** Runtime execution, scheduler slots, transactional model state, and page-backed KV ownership become canonical ECS state. A backend may execute work, but it cannot independently decide ownership, accepted tokens, committed KV pages, or terminal outcome.

| Component | Purpose |
|---|---|
| `ExecutionLease` | Bounded compute-capacity claim with fencing generation and deadline. |
| `LeaseOwner` | Session, work item, generation, and worker holding the claim. |
| `LeaseTokenRange` | Prompt/decode token range covered by the transaction. |
| `KvSlot`, `KvOwnership` | Page-backed KV identity, layer/range metadata, codec, generation, and exclusive/shared ownership. |
| `ExecutionInput` | Content-addressed token, tensor, executable, and state identities used for dispatch. |
| `ExecutionOutput` | Candidate tokens, numerical/timing evidence, backend outcome, and state delta before commit. |
| `CommitState` | Prepared, committed, rolled back, cancelled, or expired transaction state. |

| Runtime system | Responsibility |
|---|---|
| `ExecutionAdmissionSystem` | Validate session/work lease, generation, backend capability, memory, and KV claims. |
| `KvAllocationSystem` | Allocate or reuse pages according to the sealed KV plan and admitted codec. |
| `BackendDispatchSystem` | Materialize an immutable dispatch descriptor and execute outside the World write transaction. |
| `ExecutionValidationSystem` | Validate backend completion, timing, accepted token count, and numerical/ABI evidence. |
| `ExecutionCommitSystem` | Atomically publish accepted tokens, KV mutations, scheduler progress, and receipts. |
| `ExecutionRollbackSystem` | Release uncommitted pages and claims on cancellation, timeout, backend failure, or disconnect. |

**Authority cutover:** Route prefill, target decode, MTP draft/verify/accept, commit, rollback, unload, and replay through these systems. Legacy executors may temporarily perform physical dispatch but must consume immutable descriptors and return outcomes; they may not mutate session or KV truth directly.

**Failure and concurrency requirements:** Disconnect races roll back only uncommitted state. A stale generation cannot mutate current KV pages. MTP fallback commits exactly the verified target prefix. Backend crashes and daemon restarts leave leases recoverable or expirable. Concurrent sessions cannot alias writable KV ownership.

**Gate:** A real model session performs prefill, decode, MTP, commit, cancellation rollback, unload, and replay through canonical systems. KV page mutation and dispatch timing are positive and attributable. Restart recovery preserves committed state and rejects stale outcomes. No runtime scheduler, KV coordinator, or backend callback independently owns canonical execution state.

---

### Wave 4: Compilation & Model-Production Authority

**Status:** Pending.

**Dependency:** Waves 1–3 provide canonical artifacts, devices, sessions/work, and execution evidence.

**Deliverable:** Checkpoint inspection, sensitivity analysis, evolutionary mixed precision, kernel compilation, cimage assembly, validation, promotion, replay, and rollback become one canonical compilation lifecycle. A `CompilationJob` entity is the sole authority for requested policy, admitted representations, produced artifacts, and terminal promotion state.

| Command | Schema |
|---|---|
| `SubmitCompilationJob` | `CompilationJob`, source identity, `JobInput`, `JobConfig`, target machine profile, tokenizer identity, and requested policy. |
| `RecordSensitivityEvidence` | Per-tensor/per-projection sensitivity receipts and calibration corpus identity. |
| `AdmitRepresentationPlan` | Exact codec, packing, mixed-precision override, fallback, and evaluator-evidence identities. |
| `RecordBackendArtifact` | Kernel source/IR, ABI, binding plan, compiled bytes, toolchain identity, and target capability contract. |
| `SealCimage` | Deterministic cimage manifest, payload index, execution/MTP graphs, memory/KV plans, serving profile, and replay manifest. |
| `PromoteCompilationGeneration` | Atomically make the validated generation current with parent rollback identity. |
| `FailCompilationJob` | Commit a terminal typed failure without publishing partial artifacts. |

| Compilation system | Responsibility |
|---|---|
| `CheckpointInspectionSystem` | Stream and classify checkpoint tensors without unbounded reads. |
| `SensitivitySystem` | Produce measured sensitivity evidence from the declared calibration corpus. |
| `EvolutionSystem` | Search mixed precision and kernels under explicit budgets using the heterogeneous evaluator. |
| `PackingSystem` | Pack only admitted representations and store payloads content-addressably. |
| `BackendCompilationSystem` | Compile required backend artifacts and validate ABI/binding contracts. |
| `CimageAssemblySystem` | Assemble one deterministic sealed artifact with complete referential integrity. |
| `LifecyclePromotionSystem` | Validate receipts and atomically promote or reject the exact assembly. |

**Authority cutover:** `prism_compile`, deployment compiler services, lifecycle coordination, generation APIs, and runtime model loading must consume the same compilation entity and sealed output. No test-local coordinator, hardcoded precision target, source-checkpoint runtime fallback, synthetic receipt, or post-hoc artifact substitution counts as canonical.

**Failure and replay requirements:** Every intermediate artifact is either content-addressed and referenced or discarded. Cancellation cannot promote. Replaying a promoted generation resolves all payloads and executable identities without reopening source weights, rerunning evolution, or repacking tensors. Parent rollback is atomic.

**Gate:** A real checkpoint compiles into one validated sealed cimage, promotes atomically, loads in a fresh process without source weights, executes on admitted hardware, replays within numerical policy, and rolls back to its parent. Every receipt and payload digest resolves and corruption is detected. Production compiler and lifecycle entry points use this exact path.

---

### Wave 5: Multimodal & Generation Pipelines

**Status:** Pending.

**Dependency:** Wave 4 model-production authority and Waves 2–3 work/execution authority.

**Deliverable:** Vision, audio, speech, diffusion, and other generation pipelines use the same canonical artifact, deployment, work, execution, receipt, and promotion primitives as text inference. Modality-specific code remains computation; pipeline identity and lifecycle remain ECS authority.

| Component | Purpose |
|---|---|
| `Pipeline`, `PipelineStage`, `PipelineModality` | Versioned topology, stage semantics, ordering, and modality contract. |
| `StageExecutableRef`, `StageDevicePolicy` | Admitted executable and device capability requirements per stage. |
| `InputArtifactRef`, `OutputArtifactRef` | Content-addressed media, tensor, text, and metadata bindings. |
| `WorkLeaseRef`, `StageExecutionRef` | Canonical work and execution claims for each stage attempt. |
| `PipelineLifecycle` | Requested, admitted, active, blocked, completed, failed, or cancelled state. |
| `ModalityReceipt` | Shape, codec, timing, quality, safety, and provenance evidence. |

| System | Responsibility |
|---|---|
| `PipelineAdmissionSystem` | Validate topology, artifact types, executable compatibility, policy, and resource budgets. |
| `StageReadinessSystem` | Materialize runnable stage work when prerequisites and artifacts resolve. |
| `StageDispatchSystem` | Submit canonical work through Wave 3 execution authority. |
| `StageCommitSystem` | Publish outputs and unlock downstream stages transactionally. |
| `PipelineTerminalSystem` | Commit completion, failure, or cancellation and release all outstanding claims. |

**Authority cutover:** Existing vision preprocessors, audio/ASR/TTS paths, diffusion schedulers, and cross-model handoffs become stage implementations behind canonical descriptors. They cannot maintain independent pipeline truth or silently pass untracked temporary files between stages.

**Failure and replay requirements:** Stage retries are idempotent by input/executable identity. Cancellation propagates without deleting committed upstream outputs. Restart reconstructs the DAG and resumes only eligible stages. Cross-modal handoffs declare whether they transfer text, tokens, latents, audio frames, or tensors; no ambiguous bridge is accepted.

**Gate:** At least one real multimodal pipeline and one generation pipeline run end to end through canonical stage work, execution, and receipts. Restart recovery resumes a partially completed pipeline, cancellation releases outstanding claims, and every output resolves to immutable inputs and executable identities. No modality registry or scheduler remains an independent authority.

---

### Wave 6: Agent, Tool, & External-Service Authority

**Status:** Pending.

**Dependency:** Wave 2 work authority. Durable recovery is completed in Wave 9, but agent identity, claims, and effects must use canonical commands before then.

**Deliverable:** Agent sessions, delegated work, file/path claims, tool invocations, browser tabs, subprocess jobs, and external effects become canonical entities and receipts. Prism MCPD remains the shared physical coordinator, while the World is authoritative for who owns work and what effect was accepted.

| Component | Purpose |
|---|---|
| `AgentRun`, `AgentTask`, `AgentPhase` | Stable session identity, purpose, parent work, lifecycle, and terminal status. |
| `AgentCapability`, `AgentBudget` | Declared tools, concurrency, token/time/resource limits, and policy. |
| `WorkClaim`, `PathLock`, `Handoff` | Distributed ownership, fencing, expiry, and explicit delegation. |
| `ToolInvocation`, `ToolOutcome` | Typed request, idempotency key, daemon job identity, structured result, and evidence. |
| `AgentMessage` | Durable coordination message with sender, recipient, context, and acknowledged state. |
| `ExternalEffect` | Proposed, authorized, executed, verified, or compensated effect lifecycle. |

| System | Responsibility |
|---|---|
| `AgentAdmissionSystem` | Register an agent against capability and policy constraints. |
| `ClaimArbitrationSystem` | Grant one fenced owner for exclusive work or path scopes. |
| `ToolDispatchSystem` | Translate accepted invocations into MCPD-managed jobs or tool calls. |
| `ToolOutcomeSystem` | Validate daemon identity, structured result, artifacts, and terminal status before commit. |
| `ExternalEffectSystem` | Require explicit authority for pushes, messages, deployments, destructive actions, and compensations. |
| `AgentRecoverySystem` | Reconcile orphaned sessions, claims, browser tabs, and supervised jobs after restart. |

**Authority cutover:** OMP custom tools and MCP tools become thin clients over the same MCPD implementation and canonical coordination entities. Direct process spawning, private per-agent daemons, implicit CWD inference, and untracked browser sessions are rejected. One persistent daemon serves all agents while work ownership remains generation-fenced.

**Failure and security requirements:** Duplicate tool requests deduplicate by fingerprint only when inputs, workspace state, toolchain, and policy identities match. Leases expire safely. External effects require durable intent and terminal evidence. Secrets are redacted from events and receipts. A daemon restart adopts valid jobs without accepting stale owners.

**Gate:** Twelve concurrent agents can claim independent work, share cached Cargo results, stream structured progress, drive isolated browser tabs, hand off tasks, and recover through a daemon restart without duplicate writers or lost terminal outcomes. Conflicting path claims are rejected. Exactly one MCPD daemon runs. Every accepted external effect has authority and evidence.

---

### Wave 7: Distributed Topology & Worker Authority

**Status:** Pending.

**Dependency:** Waves 2, 3, and 6 establish work, execution, and agent authority.

**Deliverable:** Remote nodes and workers participate through stable identity, attested capabilities, fenced leases, and evidence-bearing outcomes. Discovery data is observational; only validated topology and capability transitions become canonical.

| Component | Purpose |
|---|---|
| `PeerIdentity`, `NodeMembership`, `NodeTopology` | Stable node identity, cluster membership epoch, locality, and connectivity. |
| `PeerCapabilities` | Hardware, backend, memory, compiler, artifact, and protocol capability digest. |
| `TrustState`, `AttestationEvidence` | Admission, revocation, key identity, and evidence authority. |
| `WorkerHealth`, `LastObservation` | Derived liveness, load, thermal, storage, and failure state. |
| `RemoteLease` | Fenced work/execution claim tied to node membership and deadline. |
| `RemoteCapabilityObservation` | Untrusted observation awaiting validation and canonical publication. |
| `ArtifactTransfer` | Content-addressed transfer plan, source, destination, integrity, and residency result. |

| System | Responsibility |
|---|---|
| `PeerAdmissionSystem` | Validate protocol, trust, identity, and cluster policy. |
| `CapabilityReconciliationSystem` | Convert observations into versioned canonical capabilities. |
| `RemoteRoutingSystem` | Select eligible workers using claims, locality, artifact residency, and policy. |
| `RemoteLeaseSystem` | Grant, renew, expire, and fence distributed work. |
| `ArtifactTransferSystem` | Move immutable artifacts and verify digests before execution eligibility. |
| `RemoteOutcomeSystem` | Reject outcomes from revoked members, stale leases, incompatible generations, or unresolved artifacts. |

**Authority cutover:** Replace ad-hoc peer maps, trust stores, worker registries, and scheduler-local capability snapshots. Network transports remain execution mechanisms and cannot mutate canonical topology directly.

**Failure and partition requirements:** Membership changes advance epochs. Network partitions cannot create two valid exclusive owners. Clock skew does not bypass fencing. Rejoining nodes reconcile leases and artifacts before receiving work. Revocation blocks new work immediately and rejects late outcomes.

**Gate:** A multi-node test proves capability discovery, artifact transfer, remote execution, lease expiry, partition handling, stale-outcome rejection, revocation, and restart replay. Routing decisions are reproducible from canonical inputs. No remote registry or transport maintains independent authority.

---

### Wave 8: Product Ingress & Application Bridges

**Status:** Pending.

**Dependency:** Waves 2–7 provide the canonical authorities that ingress may invoke.

**Deliverable:** HTTP, OpenAI, Ollama, management, FFI, Swift, and application bridges become thin translators into canonical commands and projections. They authenticate, parse, stream, and apply transport backpressure; they do not own model, session, work, execution, or generation truth.

| Component | Purpose |
|---|---|
| `IngressRequest` | Protocol-neutral request identity, route, body digest, model alias, and requested options. |
| `Principal`, `ApiKeyRef`, `AuthorizationDecision` | Authenticated identity and scoped authority without storing secret material in the World. |
| `RateLimiterState`, `AdmissionDecision` | Canonical quota window, cost, rejection reason, and policy identity. |
| `RequestQueue` | Bounded admission queue and scheduling relationship to canonical work. |
| `TransportSession` | Connection/stream identity, cancellation signal, flow-control state, and terminal record. |
| `IngressLifecycle` | Received, authenticated, admitted, active, completed, disconnected, rejected, or failed. |

| Adapter | Required behavior |
|---|---|
| OpenAI HTTP/SSE | Translate models, completions, and chat requests; stream committed tokens and one terminal event. |
| Ollama HTTP/NDJSON | Translate tags/show/ps/generate/chat; preserve protocol terminal records and cancellation. |
| Management API | Expose canonical generations, jobs, health, receipts, rollback, and unload commands. |
| Rust/Swift/FFI | Use stable command and projection DTOs; never expose internal entity pointers or runtime handles. |
| PrismAgent bridge | Consume the same server/tool contracts rather than private execution paths. |

**Authority cutover:** Remove legacy executor calls and server-local model/session registries from protocol handlers. Startup loads promoted cimages and tokenizer identities into canonical model/generation state. Every generation request reserves work, creates or resumes a session, drives Wave 3 execution, and records a terminal receipt.

**Failure and streaming requirements:** Client disconnect propagates cancellation and rolls back uncommitted state. Slow consumers trigger bounded backpressure rather than unbounded buffering. Duplicate idempotent requests reuse only policy-compatible results. FFI panics and malformed protocol input cannot corrupt the World.

**Gate:** Process-level OpenAI and Ollama tests discover a promoted model, generate coherent non-canned output, chat, stream tokens plus one terminal record, cancel by disconnect, and resume or replay according to policy. Metal/CPU dispatch and KV mutation evidence are positive. Server and application code contain no legacy executor or independent registry authority.

---

### Wave 9: Persistence, Projections, & Dashboard

**Status:** Pending.

**Dependency:** Waves 1–8 must define the authoritative commands, events, and identities that persistence records.

**Deliverable:** Every authority-bearing transition is durable before acknowledgement, snapshots and replay reconstruct the canonical World, and all dashboards/search indexes are rebuildable projections rather than competing truth. PostgreSQL is the durable concurrent coordination/event authority, DuckDB is analytical projection storage, and Valkey provides ephemeral cache/lease acceleration without becoming canonical.

| Component | Purpose |
|---|---|
| `EventStore` | Append-only durable transition log with sequence, epoch, checksum, schema, and causation identity. |
| `ReplayRegistry` | Versioned event-kind to replay-applier mapping with migration policy. |
| `ReplayEngine::replay_into` | Deterministic reconstruction with corruption, unknown-event, and partial-tail handling. |
| `Snapshot` | Content-addressed World checkpoint bound to event sequence and schema catalogue digest. |
| `ProjectionCheckpoint` | Rebuildable consumer offset, projection schema, and source event digest. |
| `ReceiptStore` | Content-addressed evidence payloads whose identifiers resolve and verify. |
| `CompactionPolicy` | Retention and snapshot rules that never discard unrecoverable authority. |

| Storage role | Authority boundary |
|---|---|
| PostgreSQL | Durable event, coordination, job, generation, and receipt metadata with concurrent writers. |
| DuckDB | Rebuildable analytical projections and historical query acceleration. |
| Valkey | Ephemeral deduplication, hot cache, pub/sub notification, and lease acceleration; loss must not lose truth. |
| Artifact store | Immutable cimages, payloads, executables, snapshots, and large receipt bodies by digest. |

**Persistence sequence:** Prepare the World transaction, append and fsync/commit its event envelope, apply the prepared epoch, acknowledge the caller, and asynchronously advance projections. Recovery loads the latest verified snapshot and replays subsequent events. Projection rebuild never writes domain authority back into the World.

**Migration sequence:** Version every event and projection schema. Backfill existing durable records through explicit migration jobs. Run old and new projections against the same event interval, compare outputs, cut readers over, then retire old projection schemas. SQLite and file stores may remain test fixtures only if clearly isolated from production configuration.

**Failure and operations requirements:** Detect torn writes, digest corruption, sequence gaps, incompatible schemas, stale projection checkpoints, unavailable Valkey, unavailable DuckDB, and PostgreSQL failover. Backpressure prevents the event log from outrunning storage budgets. Backup/restore and disaster recovery have measured recovery point and recovery time objectives.

**Gate:** Kill-and-restart tests during every lifecycle phase reconstruct identical canonical state and resume safely. Snapshot plus replay matches full replay. Corruption and unknown schemas fail closed with actionable evidence. DuckDB projections rebuild from zero and match source events; deleting Valkey loses no authority. Concurrent writers preserve ordering and fencing. Dashboards read projections only. Backup restoration passes a production-sized rehearsal.

---

### Wave 10: Authority Purge

**Status:** Pending.

**Dependency:** Waves 1–9 must have replaced each legacy authority with a canonical component, command, system, event, and recovery path. This wave deletes the superseded paths only after their replacements pass shadow and replay gates.

**Deliverable:** One production World and one transactional mutation protocol remain across compilation, deployment, execution, agents, topology, ingress, and persistence. Every other stateful structure is classified as an execution mechanism, immutable artifact, evidence record, or rebuildable projection. Legacy writers, compatibility worlds, hidden registries, process-local authorities, and direct component-store mutation are removed from production builds.

| Audit class | Required disposition |
|---|---|
| Registry, manager, coordinator, or singleton | Replace authority-bearing state with canonical entities/resources and commands; retain only stateless adapters. |
| Process-local map, queue, cache, or global | Prove it is rebuildable and non-authoritative, bind it to generation/epoch identities, or remove it. |
| Backend runtime handle | Preserve only as an opaque physical resource referenced by canonical identity and lifecycle components. |
| Database table or direct write path | Route domain writes through the canonical event/transaction boundary; projections remain read models. |
| Background worker or daemon state | Reconcile from canonical leases/jobs on restart and reject stale ownership with fencing. |
| Compatibility API or deprecated type | Migrate all callers, narrow any temporary allowance to a named module, then delete the alias and feature. |
| Test fixture authority | Replace test-local worlds, coordinators, synthetic receipts, and canned execution with production entry points where the gate claims production coverage. |

The purge runs as an evidence-producing sequence. First, a static and runtime inventory records every mutable store and writer, including feature-gated code. Second, each item is assigned an owner and one of five legal classifications: canonical authority, opaque execution resource, immutable artifact, durable evidence, or rebuildable projection. Third, replacements run in shadow mode and compare transitions, outputs, and restart behavior. Fourth, writers switch to canonical commands while legacy paths become read-only assertions. Fifth, an observation interval proves no hidden writer remains. Finally, the old path is deleted and the authority inventory is regenerated from the resulting tree.

**Mutation enforcement:** Production uses `MutationPolicy::TransactionalOnly`. Entity/component mutation outside an active `WorldTxn` is unavailable to domain code. Direct `ComponentStore` access is restricted to the World implementation. Generation-safe entity handles are mandatory at all boundaries. Broad `#![allow(deprecated)]`, `legacy_mutations`, and production aliases such as `CompWorld`, `LegacyCompWorld`, or `CompEntity` are removed. New authority-bearing globals, registries, and database writers fail a static policy check.

**Preserved execution mechanisms:** Metal, ANE, Accelerate, CUDA, Vulkan, ROCm, MLX, browser, subprocess, network, and storage clients may retain backend-native handles because the World must not pretend to be a device driver or connection pool. Those handles cannot decide lifecycle state, precision, placement, admission, ownership, promotion, or commit. Canonical entities identify their capability contract, owner, generation, lease, and terminal evidence.

| Candidate removal | Permitted replacement or preservation |
|---|---|
| `AdapterRegistry` | Stateless adapter lookup generated from canonical capability data. |
| `ModelRegistry` | Canonical model/generation entities plus rebuildable server projection. |
| `WeightCache` and `GLOBAL_PREFIX_CACHE` | Generation-keyed rebuildable caches with no admission or lifecycle authority. |
| `TrustStore` | Canonical peer/principal trust state backed by durable events and secret references. |
| `CancellationManager` | Canonical cancellation commands and lifecycle transitions propagated to execution resources. |
| `ServerEngine` and server-local session state | Thin ingress adapters over canonical model, work, session, and execution commands. |
| `HeterogeneousExecutor`, `AneBackend`, and backend engines | Opaque execution resources behind Wave 3 capability and receipt contracts, never alternate schedulers or authorities. |
| `AppState` and feature-specific roots | Composition-only handles containing clients and projections, with no domain mutation path. |

**Failure and rollback requirements:** Each deletion commit has a reversible migration boundary until the replacement survives restart, cancellation, replay, and load. Rollback restores a whole compatible authority version; it never enables two writers. Feature combinations that cannot honor the one-World contract fail at build or startup rather than silently selecting a legacy path.

**Gate:** The generated authority inventory contains exactly one production World and no unclassified mutable store. Repository-wide searches and policy tests find no production use of legacy entity/world aliases, direct domain `ComponentStore` mutation, lifecycle-authoritative globals, server-local model/session authority, or persistence bypass. The default and supported feature matrix build without warnings, the complete test suite passes, process-level compiler-to-server and distributed-agent gates pass, and kill/restart replay reconstructs the same state. A fault-injection run proves stale handles, stale leases, duplicate effects, partial persistence, cancellation, and backend failure cannot create a second authority or publish uncommitted state.

### Foundation dependency chain

The foundation is intentionally sequential at its authority boundaries even where file-disjoint implementation can proceed in parallel:

```text
Wave 1 Placement Authority (complete)
  -> Wave 2 Work and Scheduling Authority
  -> Wave 3 Execution, State, and KV Authority
  -> Wave 4 Model Production and Promotion Authority
  -> Wave 5 Multimodal and Generation Pipelines
  -> Wave 6 Agent, Tool, and External-Service Authority
  -> Wave 7 Distributed Topology and Worker Authority
  -> Wave 8 Product Ingress and Application Bridges
  -> Wave 9 Persistence, Projections, and Recovery
  -> Wave 10 Authority Purge
  -> Wave 11 Canonical World API Completion
```

Wave 1 is the only completed wave in this chain. Waves 2–10 are pending implementation campaigns and remain part of the normative plan. Wave 11 and the compiler-absorption work that follows may prototype isolated contracts, but they cannot claim architectural convergence or production readiness until every Wave 2–10 gate has passed.

## Decision

Port the required MLIR and IREE transformation semantics **into Prism's canonical ECS model** — not into a Rust class hierarchy that mirrors the C++ object graph. Each selected pass preserves upstream mathematical, structural, verification, and failure behavior with source-to-port traceability. Translation may restructure code to fit ECS transactions and the selected IR storage model; line-by-line similarity is neither required nor sufficient. The *container* changes:

| Upstream MLIR concept | Prism ECS-native form |
|---|---|
| `Operation*` | Entity with dialect-specific component(s) |
| `Value` | EntityId + `Value` component (SSA def) |
| `Block*`, `Region*` | Entity hierarchy with `Block` / `Region` components |
| `OpOperand` / `OpResult` | EntityId in component field |
| `PatternRewriter` | `WorldTxn` command builder |
| `Pass` | ECS system (one `fn run` per pass phase) |
| `PassManager` / pipeline | `Schedule` with ordered phases |
| `DialectRegistry` | Module-level component + trait registration |
| `MLIRContext` (allocator + uniquer) | World (entity store) + `Uniquer` resource |
| `Type`, `Attribute` | Entity with `Type` / `Attribute` components (uniqued via resource) |
| IREE Flow → Stream → HAL → VM | ECS phase graph: one phase per pipeline stage |
| HAL backends (Metal, CUDA, Vulkan, CPU) | Backend dispatch systems reading `HALExecutable` components |

This decision does **not** authorize a completeness-driven rewrite of every MLIR dialect, pass, IREE backend, or utility. Prism ports only the semantic surface required by admitted model pipelines and supported deployment targets. Upstream MLIR and IREE remain the conformance oracles until Wave 23 closes, even after production execution stops linking their C++ libraries.

The port preserves behavior, not incidental implementation shape. A translated system must match upstream verification, transformation, failure, and runtime semantics, but may use a different storage layout, deterministic printer, artifact container, or scheduling strategy where the ADR defines an explicit equivalence rule.

### Program invariants

| Invariant | Required behavior |
|---|---|
| One production World | Compiler, evaluator, lifecycle, runtime, and server use the same canonical World contract. |
| Transactional compiler mutation | A failed pass commits no partial rewrite. Successful passes publish one epoch and one evidence bundle. |
| Pinned upstream authority | Every ported unit records the exact LLVM/MLIR or IREE commit, source identity, build configuration, and fixture corpus digest used as its oracle. |
| Semantic differential, not formatting coincidence | Comparisons normalize locations, symbol ordering, generated identifiers, and other documented nondeterminism before structural and semantic comparison. |
| Independent execution oracle | Hardware candidates are checked against a backend-independent CPU or mathematical reference; a backend cannot admit itself. |
| Production dependency isolation | Upstream C++ tools may exist in an isolated conformance lane while production builds and runtime artifacts remain free of those dependencies. |
| Evidence before promotion | No generated dialect, pass, executable, or backend enters a promoted cimage without resolvable receipts and replay identities. |
| Bounded resource use | Every wave defines memory, latency, compile-time, artifact-size, and concurrency budgets before implementation fan-out. |

---

## Methodology: the Bun playbook

Every porting wave follows a spec-first, isolated-work, adversarial-review, differential-testing loop. Bun's published migration report is a useful case study for high-concurrency translation, but its throughput and cost figures are not Prism estimates or acceptance evidence. Prism's own measured trial waves determine concurrency and schedule.

### Core loop

Every non-trivial porting task is one iteration of this loop, orchestrated by a supervising agent:

```text
1. Write or update the spec document (porting guide mapping source patterns to target patterns)
2. Fan out parallel implementation agents, each working on a bounded set of independent files
3. Each implementation agent passes its output to 2+ adversarial review agents (different model families)
4. Review findings apply back to the implementation
5. Run the invariant test suite (language-independent: `.mlir` parse/print/verify, diff)
6. Treat failures as work items — spawn fix agents that consume the failure list
7. Repeat until the invariant suite passes at 100%
```

### Spec document (the porting guide)

Before implementation begins, the supervising agent commits a durable spec document under `docs/porting-guides/<wave>.md` that maps every source pattern to its target equivalent. A `local://` document or agent memory is not sufficient authority because it cannot be reviewed, versioned, or reproduced in CI.

For Prism's MLIR absorption, the spec document covers:

| Section | Content |
|---|---|
| `Dialect op → Component` | How each `Operation*` subclass becomes entity spawns + component attachments |
| `Pass → System` | How `runOnOperation()` becomes `System::run()` with `Query<&OpTy>` |
| `PatternRewriter → WorldTxn` | How `replaceOpWithNewOp` becomes `add_component + remove_component` |
| `Type uniquing → Uniquer resource` | How MLIRContext's type uniquing tables become a World resource |
| `SSA use-def → Component field` | How `Value` pointers become EntityId fields with generation checks |
| `No-go` | What must NOT be ported (diagnostics formatting, TableGen itself pre-Wave 20) |

### Adversarial review pairs

Every implementation agent's output is reviewed independently from the implementer. High-risk framework, rewrite, serialization, and backend work receives two independent reviews; bounded generated code may receive one review plus exhaustive differential fixtures. Model diversity is encouraged, but the gate is evidence quality rather than a named model family.

Review agents check:
- Does the output match the spec document's pattern mappings?
- Does the ported pass logic match the original C++ source's behavior on a set of canned inputs?
- Are there missing edge cases (null checks, error paths, type mismatches) that the original C++ handled but the port dropped?

### Work isolation (preventing agent stepping)

Parallel agents editing the same repo will step on each other. Jared's solution: each agent works on an independent set of files (no two agents touch the same file), and agents are forbidden from running `git stash`, `git reset`, or any destructive git command. For Prism, the isolation strategy is:

- **Files are the unit of implementation ownership.** The spec assigns each file to one agent. Shared contracts are frozen by a designated integration owner before fan-out; dialect agents do not independently evolve common IR APIs.
- **Semantic conflicts are detected by conformance gates.** `cargo check` detects type integration errors, while normalized IR differentials, verifier parity, replay, and performance gates detect behavioral incompatibility.

### Compiler errors as work queue

Cargo check output is the primary feedback signal. Jared treated every compilation error as a work item for a fix agent, not a pause in the pipeline. The workflow is:

```text
agent writes code → cargo check → failures → fix agent consumes each failure → cargo check again → repeat until green
```

For Prism, this is augmented with a second queue: **differential failures**. `cargo check` catches type errors but not semantic drift. A passing `cargo check` is followed by fixtures that run the ported pass and pinned upstream pass on the same input. Results are normalized and compared structurally, through verifier outcomes, and where applicable through execution semantics. Differential failures are work items for fix agents.

### Starting small: the 3-file trial

Before scaling to a full dialect, the first wave of any new port starts with exactly **3 files**. Jared started Bun's port with 3 Zig files before scaling to all 1448. For each of the 3 files:

1. One implementer writes the Rust file
2. Two reviewers check for behavioral match against the spec document
3. One fixer applies suggestions

Only when all 3 trial files pass both `cargo check` and the diff fixture does the wave scale to the full dialect.

### Cost and throughput characteristics

Jared published these numbers from Bun's port. They serve as the planning baseline for Prism:

| Metric | External Bun case study | Prism planning rule |
|---|---|---|
| Source volume | Large-scale language migration | Scope by required semantic surface, never line count. |
| Concurrency | High agent concurrency was reported | Begin with 3-file trials; increase only while merge and defect rates improve. |
| Wall time | Case-study-specific | Estimate after Prism's Wave 13 and Wave 15 measured trials. |
| Throughput | Translation throughput was high | Measure accepted semantic units per day, not generated lines. |
| Cost | Case-study-specific | Record compute, token, review, and human-attention cost per accepted unit. |
| Regressions | Regressions remained possible | Zero known semantic differential failures at merge; escaped defects remain an explicit metric. |
| Trial phase | Small trial before fan-out | Mandatory 3-file or 3-pass trial at every new architectural boundary. |
| Test suite | Existing language-independent suite | Pinned upstream parse, verify, rewrite, runtime, and numerical oracles. |

### Applying this to each wave

Every wave below that uses parallel agents follows this exact protocol. The wave's spec document is written first. Then a 3-file trial runs. Then the full fan-out executes with adversarial review, compiler-errors-as-work-queue, and diff fixtures as the merge gate.

No wave skips the trial phase. No wave merges without diff fixtures passing.

Bun's playbook validates the parallel porting methodology: spec documents per dialect → parallel agents per independent unit → adversarial review pairs → diff-differential fixtures as work queue.

But the *sequencing* follows a strict vertical-slice proof, not a horizontal wave.

### Wave governance

Each wave begins with a recorded baseline and ends with a signed-off gate report. A wave may not fan out merely because the previous wave compiles.

| Gate field | Required content |
|---|---|
| Authority | Pinned upstream commit and Prism commit. |
| Scope | Exact operations, passes, backends, fixtures, and unsupported surface. |
| Correctness | Verification parity, normalized structural differential, numerical oracle, and negative fixtures. |
| Performance | Baseline and maximum regression for wall time, peak RSS, allocations, artifact size, and startup. |
| Determinism | Repeated-run digest comparison under fixed inputs and toolchain. |
| Failure behavior | Cancellation, timeout, malformed IR, pass failure, compiler crash, and restart recovery. |
| Evidence | Content-addressed receipts, raw fixture digests, and replay manifest. |
| Rollback | Feature flag or generation rollback that restores the last qualified path. |
| Decision | Proceed, repeat, narrow scope, choose hybrid representation, or stop. |

The default performance budget is no more than 20% regression against the selected compact baseline unless a wave-specific ADR accepts a measured tradeoff. Wave 12 may replace this default with evidence-backed budgets for IR storage and rewrites.

## Revised wave plan

The entire plan is structured around one question: *"What is the smallest end-to-end path that validates the ECS-native IR model against upstream MLIR/IREE, and when does it produce a real compiled kernel on real hardware?"*

Every wave after the foundation produces a testable artifact. No wave produces "types that compile" without differential evidence.
Waves 1–10 establish the canonical ECS substrate in dependency order. At ADR adoption only Wave 1 is complete; Waves 2–10 remain required implementation work. Waves 11+ may begin only after Waves 2–10 pass their gates and the constitutional substrate is genuinely authoritative across the listed domains.

### Current implementation checkpoint

At ADR adoption, Wave 1 is complete. Waves 2–10 are pending. Some types and tests for later waves may already exist, but typed API presence is not accepted as wave completion until the production authority cutover and stated gate pass. `WorldError`, `MutationPolicy`, capacity, typed-resource, and evaluator contract modules exist on `main`, while `World::spawn` still returns a bare `Entity`, the runtime World remains, and mutable/multi-component query convergence is incomplete. Wave 11 must verify live code rather than infer completion from structural scaffolding.

---

### Wave 11: Campaign A completion (prerequisite)

**Status:** Pending; blocked by Waves 2–10.

**Deliverable:** One canonical World, full query API, no legacy mutations.

| Item | Detail |
|---|---|
| `World::spawn` returns `Result<SpawnedEntity, WorldError>` | Replace bare `Entity` return. Fix all callers. |
| `World::with_capacity(WorldCapacity)` | Accept the existing `WorldCapacity` enum. |
| `MutationPolicy` replaces `direct_mutation_allowed: bool` | Enforce policy in `WorldTxn` preflight. |
| `QueryMut` | Safe mutable component iteration. |
| `Query<(A, B)>` and `Query<(A, B, C)>` | Multi-component read queries. |
| `EntityRef` generation-safe lookup | Lock `Entity → (u32, u32)` with generation check on component access. |
| `WorldError` on remaining fallible methods | Replace panics with Result. |
| Retire the second runtime World | One World type. All subsystems use it. |

**Gate:** `cargo check --no-default-features`, the production feature matrix, canonical World tests, and the full `tribunus-compute-core` library suite pass with zero warnings. Repository search finds exactly one production `World` definition and no `CompWorld`, `CompEntity`, `legacy_mutations`, or runtime-world compatibility export. Stale handles, transactional rollback, mutable-query alias safety, and concurrent immutable snapshots have dedicated tests.

**Parallelism:** 4–6 agents, single wave.

---

### Wave 12: IR representation benchmark

**Deliverable:** A benchmark that tests three IR storage models against a real MLIR module with one million operations, and a published result that selects one model for the rest of the plan.

**Why before any MLIR port:** The central architectural risk is that entity-per-op overhead makes ECS-native IR too expensive for compiler-scale work (a million ops in a single module). This must be measured, not assumed.

#### Three candidates

| Representation | Entity count | Query cost | Locality | Transaction cost |
|---|---|---|---|---|
| **Entity per op / per value** | ~3M entities for a 1M-op module | Best — direct component query | Poorest — each op is a separate entity | Highest — each op mutated individually |
| **Entity per block, compact ops in SoA arenas** | ~100K entities for 1M ops | Good — query yields block entities, then scan arena | Better — ops in dense flat arrays | Medium — batch arena edits |
| **Entity per compilation, specialized IR arena as World resource** | 1 entity per compilation | Poorest — must go through arena API then translate back | Best — same layout as MLIR | Lowest — arena bulk operations |

The benchmark measures entity spawn throughput, component query latency (single + tuple), SSA use-def traversal, transaction throughput for batch replacement, peak RSS, allocation count, serialization size, clone/snapshot cost, and deterministic replay. It uses both a synthetic million-operation stress graph and representative upstream MLIR modules. “Full pipeline time” is deferred until Wave 15 because no ECS-native pipeline exists yet.

**Bonsai relevance:** Bonsai's 1-bit binary and 1.58-bit ternary weights introduce non-standard integer arithmetic that stresses IR representation. The benchmark should include a Bonsai-like matmul node with ternary weights and FP16 scales to measure whether entity-per-op overhead impacts the tight inner-loop patterns that dominate inference-bound models.

**If entity-per-op exceeds 3x the compact representation's memory or 5x its traversal time, the plan defaults to the hybrid model** (entity per block + compact operation arenas + ECS systems and transactions orchestrate the pass pipeline). The report may select the hybrid earlier when rewrite complexity, cache locality, or snapshot behavior makes entity-per-op operationally inferior. Any exception requires a written rationale and an explicit Wave 13 budget.

**Gate:** Published benchmark result in `docs/ir-representation-benchmark.md` with raw numbers and a clear recommendation. The recommendation is binding for Waves 13–22.

---

### Wave 13: ECS-native IR kernel

**Deliverable:** A working ECS-native IR kernel that can represent structured ops, regions, SSA values, types, and attributes; serialize and deserialize deterministically; and produce the same output as upstream MLIR on a single `arith.addf` + `func.return` test.

**This is the hardest wave.** Every downstream dialect and pass depends on the framework contracts being correct.

#### Required framework contracts (all before any dialect)

| Contract | Test |
|---|---|
| **Type and attribute uniquing** | Two equivalent `TensorType` entities produce the same uniqued id. Duplicates are reused. |
| **SSA use-def chains** | Insert a `Value` entity, add a use, verify the use chain. Remove the value, verify use is invalid. |
| **Block and region hierarchy** | Spawn Block child of Region. Spawn op child of Block. Verify walk. |
| **Dominance and isolation** | Structured op with region: ops inside the region must correctly reference region-arg values. |
| **Symbol tables** | `func @foo` registrable. `func.call @foo` resolves to the same entity. |
| **Trait and interface dispatch** | `OpTrait::has_side_effect` and `Interface::infer_shapes` dispatch correctly for a test op. |
| **Parser/printer** | Textual MLIR round-trips to structurally equivalent component state after canonical normalization. |
| **Rewrite driver** | Greedy pattern application: apply one canonicalization, verify result, verify convergence. |
| **Diagnostics and locations** | Error during verification produces attributable diagnostic with source location. |
| **Canonical serialization** | Prism's own versioned binary form is deterministic. Upstream MLIR bytecode compatibility is a separate optional capability, not implied. |

**Implementation strategy (3 sequential sub-waves):**

| Sub-wave | Scope | Agents |
|---|---|---|
| **13.1** | Type/attribute uniquing + SSA use-def chains + entity hierarchy (Block, Region) | 3 agents, must merge before 13.2 |
| **13.2** | Symbol tables + trait/interface dispatch + parser/printer | 3 agents, must merge before 13.3 |
| **13.3** | Rewrite driver + diagnostics + serialization | 3 agents |

After each sub-wave, the current state must pass:
- `cargo check -p tribunus-compute-core --features mlir-runtime`
- Differential test against pinned upstream MLIR on a `func` + `arith` test file, compared through a canonical structural representation with documented normalization
- Unit tests for each contract above

#### Evolutionary pipeline search

The CImage compiler pipeline spans six stages: ModelIr (ingestion) → RepresentationPlan (quantization planning) → ExecutionGraph (graph formation) → KernelPlan (kernel selection) → CompiledKernelArtifact (codegen) → CimageBuildInput (assembly, residency). AlphaEvolve currently searches only the KernelPlan stage (tile sizes, loop orders, threadgroup counts). This sub-wave expands the evolutionary search to **every stage**.

| Stage | Searchable parameter space |
|---|---|
| **ModelIr** | Graph rewrite pattern selection, operator decomposition strategy, fusion boundary placement |
| **RepresentationPlan** | Per-tensor format assignment (NF4 vs INT8 vs ternary vs FP16 vs BF16), mixed-precision boundary placement, block size selection |
| **ExecutionGraph** | Fusion granularity (fuse everything vs keep separable), tensor partitioning strategy, memory layout (SoA vs AoS), buffering depth |
| **KernelPlan** | Tiling strategy, pipeline depth, threadgroup shape, register pressure target, instruction selection — existing AlphaEvolve scope, now coordinated with other stages |
| **CompiledKernelArtifact** | Codegen flags (fast-math, denormal mode), barrier placement, spill threshold, LICM aggressiveness |
| **CimageBuildInput** | Assembly order, prefetch distance, residency policy, memory pool allocation, cacheline alignment |

Within CompiledKernelArtifact and CimageBuildInput, two dimensions require specific search operators:

**Packing layout.** How quantized weights are packed into bytes affects cache utilization and kernel arithmetic. The search swaps between packing schemes per tensor:

| Packing scheme | Ternary (1.58-bit) | Binary (1-bit) | NF4 | INT4 |
|---|---|---|---|---|
| Sequential (row-major) | 5 vals/2 bytes, stride = width | 8 vals/byte, stride = width / 8 | 2 vals/byte, stride = width / 2 | 2 vals/byte, stride = width / 2 |
| Tile-major (column tiles) | 5 vals/2 bytes per tile, tile stride padded | 8 vals/byte per tile | 2 vals/byte per tile | 2 vals/byte per tile |
| Block-interleaved | Group of 128 weights, FP16 scale, then next group | Group of 256 weights, FP16 scale | Group of 64 weights, FP32 scale | Group of 32 weights, FP16 scale |
| Interleaved-scale | Scale factor interleaved with weight group every N rows | Scale interleaved | Scale interleaved | Scale interleaved |

**CImage tile ordering.** The spatial layout of tiles in the compute image determines prefetch and locality. The search swaps between orderings:

| Ordering | Access pattern | Best for |
|---|---|---|
| Sequential (row-major) | Tile[0], Tile[1], ..., Tile[N] | Dense linear reads |
| Morton (Z-order) | Interleaved x/y bits | 2D spatial locality on GPU |
| Hilbert | Fractal curve | Maximum locality, complex addressing |
| Weight-swizzled | Weights tiles interleaved with scale-factor tiles | Mixed precision where scale is read alongside weight |

Each packing and ordering choice is per-tensor, constrained by the target backend's supported address modes. The mutation operator swaps a tensor's packing and ordering independently of its format and operation.

**The search operates on a `CompilePlan` entity that carries components for all six stages. Each generation:**

**Ternary weights change the operation graph.** Ternary weights are {-1, 0, +1} encoded in 1.58 bits. The dot product of a ternary weight and an FP16 activation is pure addition/subtraction — multiplication by 0 is skipped, multiplication by ±1 is just sign. This is not merely a narrow matmul; it is a different operation:

- **Standard matmul**: load weight × load activation → multiply → accumulate
- **Ternary operation**: load ternary {-1,0,+1} → skip or negate activation → accumulate
- **Binary operation (1-bit)**: load binary {0,1} → popcount(XNOR → accumulate)

The evolutionary search must include **operation-type swapping** as a mutation operator. A layer initially lowered as `linalg.matmul` may be mutated to `ternary.gemm` when its weights are quantized to ternary — or to `binary.popcount_gemm` for 1-bit weights. The searchable space for each layer includes not just tile sizes and formats but the fundamental operation used to compute it:

| Weight format | Operation candidate | Why different |
|---|---|---|
| FP16/BF16 | Standard matmul (tiled) | Multiplication required |
| INT8 | INT8 dot product (sdot) | Narrow multiply, accumulator wider |
| NF4 | NF4 dequant → FP16 matmul | Dequantize on load, then standard |
| INT4 | INT4 dot product (sdot) with sign extension | Narrower multiply than INT8 |
| NF8 | NF8 dequant → FP16 matmul | Between NF4 and INT8 density |
| Ternary (1.58-bit) | **TernaryGemm**: gather-scale-accumulate | No multiplication; {-1,0,+1} × activation = sign/zero |
| Binary (1-bit) | **BinaryGemm**: popcount(XNOR) | Pure bitwise; no FP arithmetic until scale |

**This is per-tensor, not per-layer or per-model.** A single attention layer may have separate weight tensors for Q, K, V, and output projection, each with a different format and a different optimal operation. The search space jointly optimizes (format, operation) per tensor, constrained by the hardware lane's supported operation set. A tensor assigned NF4 weights runs an NF4 dequant → FP16 matmul; a tensor assigned ternary weights runs a TernaryGemm. The evolution mutation operator can swap any tensor's (format, operation) independently.

This is registered in the evolution module's mutation table as an op-type mutation adjacent to the existing tile-size and format mutations.

| Stage | Searchable parameter space |

1. **Mutate** — Select one or more stage components on the plan entity and modify their parameters. Mutation operators per stage (e.g., swap tensor format, adjust tile size, toggle fusion boundary) are registered in the evolution module's mutation table.
2. **Evaluate** — Lower the mutated `CompilePlan` through the full ECS-native compiler pipeline to produce a CImage. Measure latency, memory, power, or a composite fitness function on the target hardware.
3. **Select** — Keep the top-N configurations by fitness. Crossover recombines stage components across surviving plans.
4. **Repeat** — Until convergence, budget exhaustion, or a qualifying CImage is found.

The `ecs/evolution/` systems already provide the selection, crossover, and mutation framework. This sub-wave wires that framework into every pipeline stage — not by writing a new search algorithm, but by registering each stage's parameter types as searchable components and implementing the per-stage mutation operators.

**Gate:** AlphaEvolve can produce a CImage that differs from the default pipeline in at least two stages simultaneously (e.g., a different quantization format AND a different tiling scheme), and the resulting artifact executes correctly on target hardware. The search budget and convergence criteria are published.

**Dependency:** This sub-wave runs after the full compiler pipeline (Waves 15–16) produces working CImages through the deterministic default path. It does not block Waves 14–16.

---

### Wave 14: One dialect from upstream TableGen

**Deliverable:** The existing upstream `llvm-tblgen` pipeline produces Prism ECS component definitions for the `arith` dialect, and those components compile and pass differential tests against upstream MLIR on the same test cases.

**TableGen is not yet ported.** The workspace has `mlir-tblgen`, `llvm-tblgen`, `mlir-sys`, Melior, and the Rust `tblgen` crate working. This wave uses them as a temporary oracle:

```
upstream .td → existing llvm/mlir-tblgen → normalized schema → generated Prism ECS components → cargo check + diff test
```

The normalized schema is a versioned, canonical intermediate format produced by a small pinned extractor or custom upstream TableGen backend. The extractor choice is made in the Wave 14 porting guide after a spike against LLVM 22; the ADR does not assume that arbitrary `mlir-tblgen` output can be losslessly translated. Generated Rust embeds the upstream commit, source `.td` digest, schema version, and generator digest.

**TableGen itself is not absorbed until Wave 20.** The question of whether absorbing TableGen's ~52K lines of C++ (lexer, parser, AST, record system, DAG semantics, multiclass expansion, bang operators, inheritance, backend framework) provides enough value over the upstream-oracle pipeline is deferred until the ECS-native IR model is proven and at least one dialect generates correct differential results.

**Gate:**

```text
arith.td → existing mlir-tblgen → normalized schema → Prism component types
→ test: parse upstream `arith` test file, spawn ops as ECS components, normalized structure and verifier outcomes match upstream
→ test: canonicalize `addf(x, 0)` → `x` through ECS-native rewrite driver, result matches upstream
→ cargo check
```

---

### Wave 15: One vertical MLIR path — func + arith + scf + linalg → vector

**Deliverable:** An ECS-native pass pipeline that takes a small `linalg.matmul` + `arith.addf` + `func.return` module, lowers it through scf loops to vector operations, and produces the same IR as upstream MLIR's `linalg-to-vector` pipeline.

**This is the first proof that the ECS-native model can run a real compiler pipeline.** Not just single ops — an actual multi-pass lowering.

Dialects in scope (in order of dependency):

| Dialect | Agent scope | Upstream source path |
|---|---|---|
| `func` | FuncOp, ReturnOp, CallOp. Symbol resolution. | `mlir/lib/Dialect/Func/` |
| `arith` | ~20 ops: AddFOp, AddIOp, SubFOp, MulFOp, DivFOp, CmpFOp, ConstantOp, SelectOp, etc. 3 canonicalization patterns. | `mlir/lib/Dialect/Arith/` |
| `scf` | ForOp, IfOp, YieldOp, WhileOp. Region semantics, dominance verification. | `mlir/lib/Dialect/SCF/` |
| `linalg` | MatmulOp (structured op semantics), Conv2DOp. Generic structured op interface. Indexing maps. | `mlir/lib/Dialect/Linalg/` |
| `vector` | TransferReadOp, TransferWriteOp, MultiDimReduceOp. Vector layouts. | `mlir/lib/Dialect/Vector/` |

Each dialect agent ports:
1. Operation definitions (from Wave 14's generated types)
2. Verification logic
3. Canonicalization patterns (at most 3 per dialect for the vertical slice)
4. Lowering pattern(s) to the next dialect

**Differential test for every lowering:**

```text
input.mlir (upstream canonical form)
→ Prism ECS-native pass pipeline
→ output.ecs.mlir (printer through the ECS-native IR kernel)
→ upstream MLIR pass pipeline on same input
→ output.upstream.mlir
→ normalize both outputs → structural and semantic differential must pass
```

**Gate:** `linalg.matmul` lowers through `scf.for` → `vector.transfer_read/write`; normalized operation structure, types, attributes, use-def relationships, verifier result, and execution semantics match the pinned upstream pipeline. Textual differences that survive normalization are failures. Cargo check and replay pass.

**Bonsai relevance:** Bonsai's ternary matmul (1.58-bit weights with FP16 scale per 128 elements) and 1-bit binary matmul are lowering targets in this pass. The vertical slice should include at minimum the ternary matmul lowering path — ternary weights are structurally similar to the Tile640 format Prism already supports, making the lowering extensions small.

---

### Wave 16: Vertical slice through existing evaluators

**Deliverable:** The vector IR from Wave 15 is lowered to a Metal kernel through Prism's existing backend compiler, dispatched on real hardware, and its output is compared to a CPU reference.

This uses no IREE yet. Prism already has `MlirExecutionContract → MetalBackendCompiler → MetalEvaluator` for the NF4 tile640 fixture. This wave extends that contract to accept the lowered vector IR from Wave 15 and execute it through the same Metal dispatch path.

**Pipeline:**

```text
linalg.matmul input
→ ECS-native lower to vector IR
→ MlirExecutionContract (existing)
→ MetalBackendCompiler (existing)
→ Metal dispatch on GPU
→ compare output to CPU oracle (Accelerate BLAS)
→ numerical receipt populated
```

**Gate:** The numerical receipt satisfies a declared operation- and dtype-specific policy between the GPU path and an independent Accelerate/scalar oracle. The gate records absolute, relative, and ULP error rather than choosing “bitwise or ULP” after execution. `LifecycleReceiptBundle` contains compiler provenance for every stage and replay resolves every artifact without regenerating source.

This is the first artifact anyone can point to and say: *"The ECS-native compiler path produced a correct result on real hardware."*

---

### Wave 17: IREE Flow → Stream → HAL vertical slice

**Deliverable:** A small linalg module passes through ported IREE Flow dispatch formation → Stream scheduling → HAL lowering to the CPU backend, producing a Prism executable whose normalized pipeline state, ABI behavior, and numerical result match pinned upstream IREE.

**Agents (3 sequential, not parallel):**

| Sub-wave | Scope | Notes |
|---|---|---|
| **17.1** | Flow → dispatch region formation | IREE's flow transform ported as ECS system. Partition the model entity into dispatch region entities. Test against upstream Flow output. |
| **17.2** | Stream → resource scheduling | Memory planning + task ordering as ECS systems. Requires Flow region entities + HAL target capability entities. Test against upstream Stream output. |
| **17.3** | HAL → CPU backend | Port IREE's CPU HAL driver: command buffer, buffer allocation, executable ABI. Wires into existing `cpu_runtime/lowering.rs`. Test against upstream IREE CPU output. |

**Gate:**

```text
linalg.matmul input
→ Prism ECS-native Flow → Stream → HAL (CPU)
runner output satisfies the same numerical policy as upstream IREE on the same input
→ normalized Flow, Stream, and HAL state matches
→ executable ABI and replay behavior are equivalent; container bytes need not be identical
```

---

### Wave 18: CPU differential gates + CUDA HAL backend

**Deliverable:** The CPU backend from Wave 17.3 is hardened with differential gates running on pinned upstream IREE commits. One additional portable accelerator target (CUDA) is ported independently and passes the same differential gate against upstream IREE's CUDA HAL backend.

| Agent scope | Detail | Dependency |
|---|---|---|
| CPU differential gates | CPU path runs against pinned upstream IREE on 100+ linalg inputs; normalized pipeline state and numerical results are compared under declared policies. Every mismatch is a work item. | CPU path from Wave 17.3 |
| CUDA HAL backend | `cuModuleLoad`, `cuLaunchKernel`, stream sync, device allocation via `cust` or `cudarc`. Port IREE's CUDA HAL driver. | Rust CUDA crate |
| CUDA differential gates | CUDA path runs against upstream IREE CUDA on the same 100+ inputs. | CUDA backend agents |

**Why CPU + CUDA:** CPU is the portable reference lane. CUDA is the first non-Apple accelerator lane, but its implementation and hardware qualification run only on CUDA-capable workers. Together they exercise the HAL abstraction before additional targets scale out.

**Gate:** Both CPU and CUDA paths produce identical numerical output to upstream IREE on 100+ `linalg` test inputs. Gate failures are zero. Both receipts show matching compiler provenance with pinned upstream commit digests.

---

### Wave 19: Independently qualified portable targets — Metal, CUDA, ANE, Vulkan, ROCm

**Deliverable:** Every remaining hardware target is an independently qualified lane with its own capability contract, artifact ABI, memory model, synchronization model, machine qualification suite, and hardware gate. Where upstream IREE provides an equivalent backend and operation path, Prism runs a differential gate. Otherwise, Prism uses the common normalized HAL contract plus an independent numerical oracle and clearly marks the evidence as Prism-specific rather than upstream-equivalent.

These are independent, not sequential. Each target can be ported in parallel by a separate agent team once the IREE HAL contract (command buffer, buffer allocation, semaphores, executable ABI) is stabilized by Waves 17–18.

| Target | Qualification scope | Dependency |
|---|---|---|
| **Metal** | Map the stabilized HAL contract to existing Prism Metal compilation and dispatch; use upstream differential fixtures only where an equivalent upstream path exists. | Prism Metal dispatch (exists), Metal GPU |
| **CUDA** | Full differential against upstream IREE CUDA (if not already gated in Wave 18). | CUDA-capable GPU |
| **ANE** | Prism-native MIL program generation. No upstream IREE ANE backend is assumed. HAL behavior is checked through the common contract, planar-transform checkpoints, and operation/dtype-specific CPU-oracle thresholds. | ANE hardware, IOSurface infrastructure |
| **Vulkan** | Port IREE HAL Vulkan backend via `ash`. Cross-vendor qualification (AMD, NVIDIA, Intel, Apple via MoltenVK). | `ash` crate |
| **ROCm** | Port IREE HAL ROCm backend via existing Rust bindings. AMD GPU qualification. | ROCm stack |

**Bonsai relevance:** Bonsai achieves its throughput (163 tok/s 1-bit on RTX 5090, 87 tok/s 1-bit on M5 Max) through custom low-bit kernels that operate at 1-1.58 bits per weight. The Metal and CUDA backends qualified here must support those same kernel patterns — ternary accumulate with FP16 scale broadcast, binary dot product with popcount, and group-wise dequantize-on-load. Without backend qualification that covers these patterns, Prism cannot match Bonsai's published throughput on its own backends.

**Gate:** Every target passes 100% of its applicable fixture suite. Receipts distinguish upstream differential, cross-backend differential, and independent-oracle evidence; none may be mislabeled. Each receipt includes compiler provenance, capability contract, hardware identity, dispatch count, device timing, and replay result.

---

### Wave 20: TableGen absorption decision

**Deliverable:** A data-backed decision to either (a) absorb TableGen as a Rust port, or (b) keep the upstream oracle pipeline.

By this point, 4+ dialects are generated from upstream `.td` files via the existing `mlir-tblgen` pipeline. The decision criteria:

| For absorption | Against |
|---|---|
| Upstream `.td` changes require no C++ toolchain to regenerate | 52K C++ surface is a large permanent maintenance burden |
| Rust TableGen can produce Prism-specific annotations (ECS component traits, ANE policy metadata, shader codegen hints) | The intermediate normalized schema already allows custom annotations |
| Eliminates `mlir-tblgen` as a build dependency | That dependency works and is well-maintained |

**Gate:** A published document (`docs/tablegen-absorption-decision.md`) with the recommendation. If absorb, it gets a proper wave plan following the same vertical-slice → parallel-agent pattern. If not, the decision is closed and the oracle pipeline is frozen as the permanent approach.

---

### Wave 21: DuckDB projections from immutable event streams

**Deliverable:** The immutable event stream from the ECS world (receipts, compile events, inference step records) flows into DuckDB via a projection system. A small set of latency-sensitive aggregates (current-session token throughput, last-hour admission pass rate, per-model decode latency P50/P90/P99) remain as ECS resources for hot-path queries.

**The DuckDB C++ engine is not ported.** It remains an external dependency for SQL tooling, historical analytics, and ad-hoc exploration. The ECS world does not become a database.

| Item | Detail |
|---|---|
| Event stream → DuckDB projection | System reads `PhaseReceipt` components, writes rows to DuckDB via its Rust API. One-time config, not per-event — events batch every N ticks. |
| ECS-resident hot-path aggregates | `AggregateTokenThroughput`, `AggregateLatencyQuantiles`, `AggregateAdmissionPassRate` — systems that write summary components. |
| DuckDB for everything else | Dashboards, time-range queries, joins across model/session/admission tables, percentiles, external SQL access. |

**Gate:** For the explicitly duplicated hot aggregates only, DuckDB SQL and ECS resources consume the same immutable event interval and produce identical count and quantile-policy results. Historical joins and arbitrary SQL have no required ECS equivalent.

---

### Wave 22: Coverage expansion

**Deliverable:** Dialect coverage and backend coverage expand according to real model requirements — not a completeness target.


**Bonsai integration (high priority):** Add Bonsai architecture support across four dimensions:

| Dimension | Scope | Priority |
|---|---|---|
| **1-bit binary codec** | New `BinaryCodec` in quantization sweep families (Tile640 format, 640 elements = 80 bytes, popcount-based dot product). Bonsai uses 1-bit binary in its 1-bit variants. | High |
| **1.58-bit ternary codec variant** | Bonsai ternary uses {-1, 0, +1} per weight with FP16 scale per 128 elements. Maps to existing ternary infrastructure but with different group size and scale precision than Prism's current ternary Tile640. New `BonsaiTernary` codec variant. | High |
| **Architecture config** | Bonsai 8B and 27B model descriptors (layer counts, hidden sizes, attention heads, vision tower for multimodal). Model ingestion path for the Bonsai safetensors/MLX format. | Medium |
| **262K context validation** | Bonsai supports 262K token context. KV cache and memory planning must validate at this length with the selected codec. | Medium |

This wave has no fixed end date. It adds:

| Item | Priority |
|---|---|
| `tosa` dialect (ONNX/PyTorch import path) | High |
| `bufferization` + `memref` (memory planning for targets without unified memory) | High for CUDA/Vulkan, Low for Metal |
| `gpu` dialect | Medium |
| `transform` dialect (schedule-as-data for evolutionary search) | Medium |
| `spirv` dialect | Low (Vulkan HAL backend can consume SPIR-V without dialect-level ops) |
| IREE HAL Vulkan backend | Medium |
| IREE HAL ROCm backend | Low |
| DuckDB analytics: model-level aggregate views | Medium |
| DuckDB analytics: cross-model comparison dashboards | Low |

Each addition follows the same pattern as Waves 15–16: port the dialect/backend, verify against upstream via differential tests, execute on real hardware, publish receipts.

---
### Wave 23: C++ dependency removal

**Deliverable:** Production builds and runtime execution no longer require or link MLIR, IREE, or TableGen C++ code. Pinned upstream tools may remain in an isolated conformance environment used to qualify changes; they are not fetched, built, or invoked by ordinary Prism builds.

| Dependency | Removal scope |
|---|---|
| LLVM 22 / MLIR | Removed from production Cargo features, linker inputs, release images, and default developer builds. A pinned oracle toolchain may remain in dedicated conformance CI. |
| IREE | No production link or runtime dependency. HAL backends are replaced by qualified ECS-native dispatch systems from Waves 17–19. A pinned oracle runner may remain in conformance CI. |
| `mlir-tblgen`, `llvm-tblgen` | If Wave 20 selects absorption, replace generation with the qualified Rust path. If it selects schema freezing, check generated schemas into source and invoke upstream tools only in explicit regeneration/conformance jobs. If neither is achieved, Wave 23 is blocked. |
| `mlir-sys`, `melior` | The C API bridge crate removed. All IR construction and transformation through ECS-native types. |

**Counter-indication.** Do not start this wave until:
- Every MLIR dialect pass that Prism uses has an ECS-native equivalent (verified by differential gate)
- Every IREE HAL backend that Prism targets has an ECS-native equivalent (verified by numerical gate)
- The TableGen decision from Wave 20 is closed and the chosen path produces correct output in CI
- production and default developer builds succeed with the conformance toolchain unavailable

**Gate:** Production and default developer builds pass in a clean environment with no LLVM/MLIR/IREE executables, headers, or libraries installed. Compiler lifecycle tests pass using only ECS-native paths. A separate pinned conformance job still proves compatibility against upstream and cannot contaminate production artifact identities.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Entity-per-op IR has unacceptable overhead** | Medium | High | Wave 12 benchmark decides. Hybrid model (entity per block, compact SoA arenas) is the fallback. |
| **ECS rewrite driver diverges from MLIR convergence or failure semantics** | High | High | Port the worklist, benefit ordering, bounded recursion, listener, rollback, and failure contracts explicitly; run negative and non-convergent fixtures before dialect fan-out. |
| **MLIR upstream evolves incompatible .td definitions** | High | Medium | Pin upstream authority per release. Version generated types with `.td` and generator digests; upgrades are explicit qualification campaigns. |
| **Rust driver or binding surface is insufficient for a backend** | Medium | High | Qualify the selected crate/API with allocation, launch, synchronization, error, and recovery spikes before assigning a backend wave. Permit a narrow maintained FFI boundary until an ECS-native replacement passes the same gate. |
| **Parallel agent port introduces semantic drift** | Medium | High | Differential tests catch every mismatch. No pass is accepted without matching upstream output. Compiler errors alone are insufficient — diff fixtures are the work queue. |
| **Campaign A completion reveals deeper World incompatibilities** | Low | High | Wave 11 includes the second-runtime-World retirement. If deeper issues surface, they block everything and must be resolved before Wave 12. |
| **Pinned oracle is mistaken for permanent semantic truth** | Medium | High | Record supported upstream versions and compatibility windows. Never claim compatibility beyond the pinned fixtures and commits. |
| **Generated code or upstream source creates licensing/provenance ambiguity** | Medium | High | Preserve source notices, commit and file provenance, generator identity, and a machine-readable source-to-port manifest; review licensing before distribution. |
| **Conformance infrastructure leaks into production artifacts** | Medium | High | Separate Cargo features, CI images, caches, and artifact namespaces; verify production binaries and cimages contain no upstream library dependencies. |

## Parallelism model per wave

| Wave | Agents | Degree of parallelism |
|---|---|---|
| 1 | 2–3 | Parallel within the wave |
| 2 | 2 | Session + work, run concurrently |
| 3 | 1–2 | Single subsystem |
| 4 | 2 | Compilation commands + schemas |
| 5 | 2 | Multimodal pipeline types |
| 6 | 2 | Agent + tool types |
| 7 | 1–2 | Distributed topology types |
| 8 | 1–2 | Ingress bridge types |
| 9 | 2 | Event store + replay |
| 10 | 1–2 | Adversarial audit + removal |
| 11 | 4–6 | Parallel within the wave, sequential merge gate |
| 12 | 1–2 | Single benchmark implementation |
| 13.1 | 3 | Parallel types/uniquing + SSA + hierarchy |
| 13.2 | 3 | Parallel symbols + traits + parser, depend on 13.1 |
| 13.3 | 3 | Parallel rewrites + diagnostics + serialization, depend on 13.2 |
| 14 | 1 | Single pipeline: upstream tblgen → schema → Prism components |
| 15 | 5 | One per dialect (func, arith, scf, linalg, vector) |
| 16 | 1–2 | Wiring: existing evaluator contract |
| 17 | 3 | Sequential (17.1 → 17.2 → 17.3) |
| 18 | 1–2 | CPU differential gates + CUDA backend port |
| 19 | 5 | Fully parallel — Metal, CUDA, ANE, Vulkan, ROCm independent lanes |
| 20 | 1 | Decision document |
| 21 | 2 | Event stream projection + hot-path aggregates |
| 22 | Variable | Add dialect/backend per real model requirement |
| 23 | 1 | Cleanup: remove C++ deps, verify compile |

Total agent count is irrelevant — it's the parallelism *within a wave* that determines wall-clock time. Waves 13 and 15 are the widest (6 agents each). Every wave after 17 is narrower because the framework is proven.

## Acceptance criteria for the entire program

The absorption is complete when:

1. A model ingested as GGUF or safetensors flows through the ECS-native compiler path (MLIR dialect ops → lowering passes → backend dispatch) and produces correct inference output on Metal, CUDA, and CPU backends **without touching any external C++ MLIR or IREE library at runtime**.

2. The compiler provenance for that run (every lowering step, every receipt, every backend artifact digest) lives in the same ECS world as the inference runtime — retrievable by entity query, not reconstructed from logs.

3. Upstream `.td` definitions can be updated (a new `arith` op, a changed `linalg` matmul interface) and the Prism pipeline regenerates its dialect components in a single `cargo build`, with differential tests catching any semantic drift from the upstream C++ path.

4. Historical analytics remain authoritative in DuckDB. The explicitly duplicated hot-path aggregates return identical results from DuckDB and ECS resources when evaluated over the same immutable event interval and quantile policy.

5. A fresh process loads a sealed cimage containing the selected ECS-native compiler and backend artifacts without reopening source weights, invoking conformance tools, repacking representations, or selecting a new precision policy.

6. Compilation, hardware execution, promotion, replay, cancellation, and rollback use persisted identities and remain coherent across daemon or process restart. Numerical and performance drift are classified separately.

7. CI contains distinct production, conformance, and hardware lanes. Production proves dependency absence; conformance proves compatibility with pinned upstream; hardware lanes prove real dispatch and timing on each admitted target.

---

## References

- [MLIR execution contract](../compute-core/src/ecs/mlir.rs) — existing MlirExecutionContract, MlirToMetalAdapter
- [Compiler systems](../compute-core/src/ecs/system/compiler_systems.rs) — existing CompileScheduleSystem, BackendAssessmentSystem, GraphOptimizerSystem, GraphEqualizationSystem
- [Compiler pipeline](../compute-core/src/ecs/core/compile_pipeline.rs) — existing parallel relocation pipeline
- [Provenance and receipts](../compute-core/src/ecs/canonical/provenance.rs) — existing LifecycleReceiptBundle, ReplayManifest
- [Bun Zig→Rust port post](https://bun.com/blog/bun-in-rust) — methodological reference for agent-based parallel port
- [MLIR pass infrastructure](https://mlir.llvm.org/docs/PassManagement/) — upstream authority for pass nesting, multithreading invariants, failure, instrumentation, and pipelines
- [IREE device replay](https://iree.dev/developers/performance/device-replay/) — upstream reference for replaying HAL resources and command streams
- [REMAINING_WORK.md](../compute-core/src/ecs/canonical/REMAINING_WORK.md) — current state of MLIR integration, engram training, ternarization, evaluator
- [CAMPAIGN.md](../CAMPAIGN.md) — source for waves 1–10 subsystem registry and cutover protocol
