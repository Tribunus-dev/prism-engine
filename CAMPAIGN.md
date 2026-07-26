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
| 3 | **Model Deployment & Residency** | `Canonical` | Model, Residency | ModelId, ModelArtifactRef, ModelLifecycle, ResidencyDeviceRef, ResidencyMemoryClaim, ResidencyFormat, ResidencyLifecycle, AllocationToken | kernel |
| 4 | **Session Lifecycle** | `Shadow` | Session | SessionConfig, SessionModels, SessionDevices, SessionLifecycle, ResidencyModelRef | kernel |
| 5 | **Work Scheduling** | `Shadow` | WorkItem | WorkItemComponent, WorkState, WorkLeaseComponent, ResourceClaimComponent, WorkPrerequisites, WorkOutput | kernel |
| 6 | **Execution Leases** | `Shadow` | — | ExecutionLease, LeaseOwner, LeaseTokenRange, KvSlot, KvOwnership, ExecutionOutput | kernel |
| 7 | **Compilation & Model Production** | `Shadow` | CompilationJob | CompilationJob, JobInput, JobConfig, JobOutput, JobLifecycle, ValidationReceipt, QuantizationPlan, CimagePromotion; plus `cimage_pipeline` (admission, authority, canonical, diagnostics, differential, publish, receipts), `cimage_packer` (V4 unified packer, pack-from-dir, segment writer, helpers, multimodal), `cimage_validation` (per-kernel validators + ValidationMatrix), and the re-implementations of `system/{compile_planning,hardware_tuning,fusion_analysis,fusion_scheduling,compile_pipeline}.rs` (commit `472d9754`) | kernel |
| 8 | **Agent & Tool Execution** | `Shadow` | Agent | AgentRun, AgentTask, AgentPhase, ToolInvocation, ToolOutcome, AgentMessage, AgentConfig, AgentLifecycle | kernel |
| 9 | **Multimodal Pipelines** | `Shadow` | Pipeline | Pipeline, PipelineStage, PipelineModality, InputArtifactRef, OutputArtifactRef, PipelineLifecycle, WorkLeaseRef | kernel |
| 10 | **Distributed Topology** | `Shadow` | Node | PeerIdentity, NodeMembership, PeerCapabilities, NodeTopology, TrustState, WorkerHealth, RemoteLease, RemoteCapabilityObservation | kernel |
| 11 | **Server & API Bridges** | `Shadow` | — | IngressRequest, ApiKey, RateLimiterState, RequestQueue, TransportSession, IngressLifecycle | kernel |
| 12 | **Persistence & Projections** | `Shadow` | — | FsEventStore (file-backed, durable-before-ack, restart recovery proven), ReplayRegistry (16 appliers), ReplayEngine::replay_into, restart recovery integration test | kernel |
| 13 | **Dashboard & Authority Purge** | `LegacyRemoved` | — | — | kernel |
| 14 | **Engine Receipts** | `Shadow` | Worker | ModelLoadReceipt, RequestAdmissionReceipt, PhaseReceipt, StepReceipt, TerminalRequestReceipt, WorkerExitReceipt, Timeline, ReceiptBuilder, ReceiptId. Re-implements `compute-core/src/ecs/core/engine_receipts.rs` (1,264 LOC) at `crates/prism-ecs-runtime/src/engine_receipts.rs` (~660 LOC, 20 tests) — commit `b7d92c40`. Original kept in place for shadow comparison. | runtime |
| 15 | **Attention Sinks** | `Shadow` | (no entity — pattern overlay) | SinkHandle, SinkStore (trait), SinkWindow, SinkWindowConfig, AttentionRange, SinkError. Re-implements the `SinkState` design from `compute-core/src/ecs/core/executor.rs` (1,308 LOC, of which ~172 LOC is the sink pattern) at `crates/prism-ecs-runtime/src/attention_sink.rs` (~430 LOC, 13 tests) — commit `b7d92c40`. Original kept in place; MLX-coupled `run_prologue` / `run_layer` / `moe_forward` parts stay engine-side. | runtime |
| 16 | **GGUF Manifest Extraction** | `Shadow` | (no entity — format adapter) | TextArchitecture, AttentionKind, RopeSpec, MoeConfig, ManifestError, plus canonical GGUF metadata keys. Re-implements the manifest-extraction portion of `compute-core/src/ecs/core/gguf.rs` (1,118 LOC) at `crates/prism-gguf/src/manifest.rs` (~440 LOC, 9 tests) — commit `b7d92c40`. Original kept in place (the duplicate format parser is a deferred-deletion candidate). | kernel |
| 17 | **ANE MIL Builder** | `Canonical` | (no entity — builder) | MilBuilder, MilProgram, MilSpec, high-level ANE program constructors. The engine's 2,226-LOC `compute-core/src/ecs/core/mil_builder.rs` (the *superset*) was merged into `crates/prism-ane/src/mil_builder.rs`; the engine file is now a 68-LOC re-export shim. Plus new `crates/prism-ane/src/mil_layer_programs.rs` (278 LOC) for high-level ANE program constructors. 28/28 prism-ane tests pass (+19 new) — commit `7cd96e16`. | ane |
| 18 | **Compile-Phase Admission Gates** | `Shadow` | CompilationJob | AneAdmissionGate, LaneAdmissionGate, AneArtifactQualificationRecord, AneQualificationKey, EvidenceProbeBuffer. Re-implements `compute-core/src/ecs/system/gates.rs` (1,044 LOC) at `crates/prism-ecs-constitutional/src/admission_gates.rs` (470 LOC, 17 tests) — commit `472d9754`. Original deleted. | constitutional |
| 19 | **Buffer Lifetime Planning** | `Shadow` | Dispatch, Value | BufferLifetimePlan, ValueLifetime, SlotState, BufferLifetimeError. Re-implements `compute-core/src/ecs/system/buffer_lifetime.rs` (350 LOC) at `crates/prism-ecs-runtime/src/buffer_lifetime_plan.rs` (376 LOC, 10 tests) — commit `472d9754`. Original deleted. | runtime |
| 20 | **Hardware Tuning & Kernel Generation** | `Shadow` | Dispatch, GpuProfile | GpuProfileId, KernelFamily, KernelTemplateId, CodecFamily, DType, TileShape. Re-implements `compute-core/src/ecs/system/{tuning.rs, kernel_gen.rs}` (843 LOC) at `crates/prism-ecs-compile/src/hardware_tuning.rs` (281 LOC, 11 tests) and `crates/prism-ecs-kernel/src/kernel_generation.rs` (364 LOC, 15 tests) — commit `472d9754`. Originals deleted. | kernel |
| 21 | **Engine Singleton Systems** | `Shadow` | Engine | EngineSingleton, ModelInstallRequest, ModelLoadRequest, GenerationRequest, InFlightDecode, Pressure. Re-implements `compute-core/src/ecs/system/engine_systems.rs` (1,036 LOC) at `crates/prism-ecs-runtime/src/engine_systems.rs` (281 LOC, 15 tests) — commit `472d9754`. Original deleted. | runtime |
| 22 | **Text Architecture Extraction** | `Shadow` | Model | TextArchitecture, AttentionKind, RopeSpec, MoeConfig. Re-implements `compute-core/src/ecs/system/model_load.rs` (350 LOC) at `crates/prism-ecs-artifact/src/text_architecture_extract.rs` (280 LOC, 12 tests) — commit `472d9754`. Original deleted. | artifact |
| 23 | **Engine Runtime WorldTxn** | `Canonical` (engine-local) | — | WorldTxn, PendingEntity, InsertTarget, WorldTxnError, WorldTxnErrorCategory, CommitReceipt. Engine-local `WorldTxn` mirroring the constitutional `prism_ecs_constitutional::WorldTxn` shape, scoped to the engine's runtime `World` (entity/component storage, not the constitutional `ComponentStore`). Lives at `compute-core/src/ecs/runtime/world_txn.rs` (459 LOC, 14 unit tests). Used to port the 10 remaining direct world mutations in `runtime/` and `core/` — commit `ebcaf2bc`. | runtime |

> Subsystems 14–23 are the new surfaces introduced by the
> `compute-core.legacy/` → constitutional ECS absorption (2026-07-25).
> See `changelogs/2026-07-25-compute-core-legacy-integration-plan.md`
> for the per-phase commit and changelog pointers.
>
> **Status convention for absorbed subsystems.** A re-implementation in
> a constitutional crate enters at `Shadow` and advances to
> `Canonical` only when (a) the original engine file is deleted (no
> parallel authority) and (b) the constitutional path has a propagation
> test. Subsystems 18, 19, 20, 21, 22 reached `Shadow` with the
> original engine files deleted in the same commit (single authority);
> subsystems 14, 15, 16 left the originals in place for shadow
> comparison and remain `Shadow` until coordinated deletion in a
> follow-up phase. Subsystem 17 (ANE MIL builder) is `Canonical`
> because the engine file is now a re-export shim delegating to the
> constitutional crate. Subsystem 23 is engine-local and is
> `Canonical` within the engine; the constitutional libraries do not
> see it directly.

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

# Rust quality — production-scope unwrap/expect count, scoped to canonical paths.
# Excludes #[cfg(test)] mod tests blocks (permitted by the rust-quality rule)
# and compute-core.legacy/ (archaeology).
# Use the project-local script (./scripts/unwrap_baseline.py) for the full
# production/test split.
python3 scripts/unwrap_baseline.py

# Rust quality — full clippy baseline including macro-expansion false positives,
# pedantic, nursery, and pre-existing lint warnings
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
- **Rust quality.** **482 production `unwrap` / `expect` calls across 88 canonical-path
  files** + **2,631 test-scope unwraps** (excluded by the rust-quality rule).
  The production count excludes both `#[cfg(test)] mod tests { ... }` blocks
  *and* integration-test files under `<crate>/tests/`, both of which are
  permitted by the rust-quality rule. Earlier versions of this baseline
  conflated the two; the authoritative count comes from
  `scripts/unwrap_baseline.py`, which parses each file to find test
  boundaries and excludes them from the production count.
- **Project absorption.** 5 absorbed-pattern files in the canonical paths:
  `tinygrad_core.rs`, `uop.rs`, `bonsai_ternary.rs`, `bonsai_cimage.rs`,
  `turboquant_kv.rs`. All in canonical paths; none under a vendored exception.
- **compute-core.legacy absorption (2026-07-25).** 10 `system/` files
  re-implemented in 6 constitutional crates (commit `472d9754`,
  changelog `changelogs/2026-07-25-compute-core-absorption-phase-2-system.md`).
  3 `compute_image/` files re-implemented in `prism-ecs-compile`
  (`cimage_pipeline/`, `cimage_packer/`, `cimage_validation/`, commit
  `14e8edb1`, changelog
  `changelogs/2026-07-25-compute-core-absorption-phase-4b-compute-image.md`).
  3 `core/` files re-implemented in `prism-ecs-runtime` and
  `prism-gguf` (`engine_receipts`, `attention_sink`, `manifest`, commit
  `b7d92c40`, changelog
  `changelogs/2026-07-25-compute-core-absorption-phase-4c-core.md`).
  `mil_builder` absorbed into `prism-ane` (commit `7cd96e16`).
  10 remaining direct world mutations ported to a new engine-local
  `WorldTxn` at `compute-core/src/ecs/runtime/world_txn.rs` (commits
  `ebcaf2bc` + `c5ad9070`, changelog
  `changelogs/2026-07-25-compute-core-absorption-phase-3-runtime.md`).
  All 16+ absorbed files live in constitutional crates under
  Prism-domain names; 4 shim directories (`constitutional/`,
  `quantization/`, `kv_cache/`, `inference_profile/`) were removed
  from `compute-core/src/ecs/mod.rs` in commit `ef826363`. The engine
  is renamed from `compute-core.legacy/` to `compute-core/` and is
  now a workspace member.
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
| `crates/prism-ecs-compile/src/compiler.rs` | 1655 | 11 | **PARTIAL DECOMPOSITION (2026-07-25).** `plan_apply.rs` extracted (169 lines) — owns the canonical world-mutating apply path (`compile_source_ecs`). `compiler.rs` keeps the orchestrator + pure-IR + compat wrappers. `ir_build.rs` split not done; deferred to a follow-up. World-mutating-during-codegen violation moved out of the orchestrator. |
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

### Project Absorption — `compute-core.legacy/` → constitutional ECS (2026-07-25)

The following engine files have been re-implemented in the constitutional
crates during the 2026-07-25 absorption wave. The new files live under
Prism-domain names and are the canonical home for the relevant authority;
the engine files are either deleted (single authority) or left in
place as shadow copies for follow-up deletion (dual authority, awaiting
coordination with the engine's own callers).

| Original (`compute-core/src/ecs/...`) | Re-implementation | Authority the new file owns | Status | Commit |
|---|---|---|---|---|
| `system/buffer_lifetime.rs` (350 LOC) | `crates/prism-ecs-runtime/src/buffer_lifetime_plan.rs` (376 LOC) | Buffer lifetime planning: per-buffer alloc/free epoch derivation from a dataflow graph's topological sort, plus the scratch buffer sizing heuristic for dispatch entities | `Shadow` (original deleted) | `472d9754` |
| `system/model_load.rs` (350 LOC) | `crates/prism-ecs-artifact/src/text_architecture_extract.rs` (280 LOC) | Translating a HuggingFace-style config JSON (and any `text_config` sub-section) into a `TextArchitecture` value for downstream compile-time systems | `Shadow` (original deleted) | `472d9754` |
| `system/planning_core.rs` (352 LOC) | `crates/prism-ecs-compile/src/compile_planning.rs` (285 LOC) | The four planning-time decisions between graph construction and kernel lowering: ANE eligibility, memory budget check, region catalogue / planner, and packaging receipt | `Shadow` (original deleted) | `472d9754` |
| `system/tuning.rs` (358 LOC) | `crates/prism-ecs-compile/src/hardware_tuning.rs` (281 LOC) | Hardware-targeted kernel tuning: tile shape selection by score and AMD GPU profile matching by compute-unit proximity | `Shadow` (original deleted) | `472d9754` |
| `system/kernel_gen.rs` (485 LOC) | `crates/prism-ecs-kernel/src/kernel_generation.rs` (364 LOC) | Post-dispatch kernel-generation: select a template by root op + codec, resolve `KernelParameters` from the dispatch's shape, expand the template source with strict `{{PLACEHOLDER}}` substitution | `Shadow` (original deleted) | `472d9754` |
| `system/fusion/analysis.rs` (539 LOC) | `crates/prism-ecs-compile/src/fusion_analysis.rs` (384 LOC) | Fusion analysis: build a `DataflowGraph` from layer `CanonicalRole`s, identify fusion groups, emit one dispatch per group | `Shadow` (original deleted) | `472d9754` |
| `system/fusion/scheduler.rs` (623 LOC) | `crates/prism-ecs-compile/src/fusion_scheduling.rs` (393 LOC) | Fusion scheduling: backend evaluation, group growth for singleton groups, and cost-based candidate selection | `Shadow` (original deleted) | `472d9754` |
| `system/gates.rs` (1,044 LOC) | `crates/prism-ecs-constitutional/src/admission_gates.rs` (470 LOC) | Compile-phase admission: ANE admission (determinism, perf, memory, bridge copy, numerical error), qualification gate, and evidence probe | `Shadow` (original deleted) | `472d9754` |
| `system/engine_systems.rs` (1,036 LOC) | `crates/prism-ecs-runtime/src/engine_systems.rs` (281 LOC) | Engine singleton systems: init, generation requests, model install / load / unload, cancel, metrics, and shutdown | `Shadow` (original deleted) | `472d9754` |
| `system/pipeline_core.rs` (1,242 LOC) | `crates/prism-ecs-compile/src/compile_pipeline.rs` (472 LOC) | Per-model compile pipeline state: distillation, epoch schedule, calibration frontier, phase IR, profitability, and tri-lane cost model | `Shadow` (original deleted) | `472d9754` |
| `compute_image/compile/pipeline.rs` (2,664 LOC) | `crates/prism-ecs-compile/src/cimage_pipeline/` (1,811 LOC across 9 files: `mod`, `admission`, `authority`, `canonical`, `diagnostics`, `differential`, `publish`, `receipts`, `tests`) | Authority-aware compile pipeline: preflight, profile check, compatibility detect, differential compile, `publish_image` step, `CompileReceipt`, `DiagnosticReport` | `Shadow` (original left in place) | `14e8edb1` |
| `compute_image/cimage_packer/pipeline.rs` (3,372 LOC) | `crates/prism-ecs-compile/src/cimage_packer/` (1,304 LOC across 6 files: `mod`, `pack_unified`, `pack_from_dir`, `segment_writer`, `helpers`, `multimodal`, `tests`) | V4 unified `.cimage` packer: 5-segment unified packer, directory-aware packer, page-alignment, multimodal segment synthesis types | `Shadow` (original left in place) | `14e8edb1` |
| `compute_image/compile/validation_matrix.rs` (3,118 LOC) | `crates/prism-ecs-compile/src/cimage_validation/` (575 LOC across 14 files: `mod`, `result`, `run`, per-kernel `validators/*`, `tests`) | Post-emission kernel validation matrix: `ValidationMatrix`, `ValidationResult`, per-kernel `validate_*` functions abstracted behind a `ValidationDevice` port | `Shadow` (original left in place) | `14e8edb1` |
| `core/engine_receipts.rs` (1,264 LOC) | `crates/prism-ecs-runtime/src/engine_receipts.rs` (~660 LOC, 20 tests) | Engine receipt types: `ModelLoadReceipt`, `RequestAdmissionReceipt`, `PhaseReceipt`, `StepReceipt`, `TerminalRequestReceipt`, `WorkerExitReceipt`, `Timeline`, `ReceiptBuilder`. Authority-bearing fields promoted to typed enums (`AdmissionDecision`, `RequestOutcome`, `CancellationMode`, `ExecutionPhase`); `ReceiptId` re-exported from `prism_ecs_constitutional` | `Shadow` (original left in place) | `b7d92c40` |
| `core/executor.rs` (SinkState pattern, ~172 LOC of 1,308) | `crates/prism-ecs-runtime/src/attention_sink.rs` (~430 LOC, 13 tests) | Attention-sink pattern: backend-neutral `SinkHandle`, `SinkStore` trait, `SinkWindow`, `SinkWindowConfig`, `AttentionRange`, plus the entropy-driven adaptive window heuristic | `Shadow` (original left in place; MLX-coupled parts stay engine-side) | `b7d92c40` |
| `core/gguf.rs` (manifest extraction, ~400 LOC of 1,118) | `crates/prism-gguf/src/manifest.rs` (~440 LOC, 9 tests) | Typed `TextArchitecture` extraction from parsed GGUF metadata. `keys` module with canonical GGUF metadata keys; `ManifestError` with `MissingKey` / `InvalidValue` variants; `RopeSpec`, `MoeConfig`, `AttentionKind` types | `Shadow` (original left in place; duplicate format parser is a deferred-deletion candidate) | `b7d92c40` |
| `core/mil_builder.rs` (2,226 LOC) | `crates/prism-ane/src/mil_builder.rs` (1,016 LOC) + `crates/prism-ane/src/mil_layer_programs.rs` (278 LOC) | ANE MIL builder and high-level ANE program constructors. Engine was the *superset* — unique methods (`topk`, `batch_size`, `silu`, `softmax`, `matmul_transpose_y`, `concat`, `conv`, `reshape`, `transpose`, `const_i32`, `reserve_names`, full 3-arg `gather`) absorbed; engine file replaced with 68-LOC re-export shim | `Canonical` (engine file is now a re-export shim) | `7cd96e16` |
| `runtime/` (10 direct world mutations) + `core/engine.rs:870` (1 direct world mutation) | `compute-core/src/ecs/runtime/world_txn.rs` (459 LOC, 14 unit tests) — engine-local `WorldTxn` mirroring the constitutional `prism_ecs_constitutional::WorldTxn` shape | Engine-local staged-mutation buffer scoped to the engine's runtime `World`. `WorldTxn::stage_spawn`, `stage_insert_on`, `stage_insert`, `stage_remove`, `commit`; `PendingToken`; `InsertTarget`; `WorldTxnError`; `CommitReceipt` | `Canonical` (engine-local) | `ebcaf2bc` |

The re-implementation pattern, the exception categories (format adapters, hardware
backends, vendored dependencies — all exempt), and the migration sequence are in
`references/project-absorption.md`. The per-file `Completion report` for each
phase lives in `changelogs/2026-07-25-compute-core-absorption-phase-*.md`.

### Rust Quality Backlog

**1,104 production-scope `unwrap` / `expect` calls across 94 canonical-path files.**
The migration target is zero; each violation either becomes `?` propagation, a
typed error, or a `// WAIVER` with a justification. Test-scope unwraps (2,627
across the same files) are permitted by the rust-quality rule and excluded
from this count. Per-file priority queue, ordered by production count, top
entries first:

| File | Prod | Test | Status / Plan |
|---|---:|---:|---|
| `crates/prism-spatial-ir/src/tinygrad_core.rs` | 246 | 136 | Migrate as part of project-absorption decomposition to `phase_graph/` |
| `crates/prism-ecs-backend.legacy/src/metal.rs` | 86 | 0 | Legacy path; migrate to canonical backend or delete |
| `crates/prism-ecs-server/src/engine/safetensors.rs` | 66 | 0 | Server engine; subsystem cutover target |
| `crates/prism-ecs-kernel/src/cpu_backend.rs` | 52 | 42 | Backend dispatch; surface errors via `Result`, not panic |
| `crates/prism-ecs-runtime/tests/recovery.rs` | 46 | 0 | Integration test; the production count here is the test setup helper, not the SUT — review whether the helper is itself production |
| `crates/prism-ecs-compile/src/uop.rs` | 40 | 122 | Migrate as part of project-absorption decomposition to `ir_value.rs` + `ir_op.rs` |
| `crates/prism-ecs-runtime/src/kernel.rs` | 30 | 10 | Runtime kernel; migrate as part of module-cohesion decomposition to per-stage files |
| `crates/prism-ecs-kernel/src/metal_dispatch.rs` | 28 | 0 | Backend dispatch; typed ABI errors |
| `crates/prism-ane/src/mil_builder.rs` | 26 | 6 | ANE builder; typed MIL errors |
| `crates/prism-ecs-server/src/runtime/receipt.rs` | 24 | 7 | Server receipt; subsystem cutover target |
| `crates/prism-ecs-ir/src/serde.rs` | 22 | 82 | IR serialization; typed errors via `thiserror` |
| `crates/prism-ecs-server/src/runtime/server.rs` | 22 | 0 | Server runtime; subsystem cutover target |
| `crates/prism-ecs-ir/src/traits.rs` | 20 | 0 | IR traits; typed errors |
| `crates/prism-ecs-compile/src/runtime.rs` | 18 | 22 | Compile runtime; migrate as part of module-cohesion decomposition |
| `crates/prism-rocm-runtime/src/ternary.rs` | 16 | 1 | ROCm ternary kernel; backend errors must be typed |
| `crates/prism-plugin/src/lib.rs` | 14 | 0 | FFI boundary; typed errors mandatory |
| `crates/prism-ecs-server/src/runtime/mod.rs` | 14 | 0 | Server runtime module index; decompose if it crosses 200 LOC |
| `crates/prism-gguf/src/writer.rs` | 10 | 5 | Format adapter; typed errors via `thiserror` |
| `crates/prism-ecs-runtime/src/test_adapters.rs` | 10 | 0 | Test adapters; the production count here is the test setup, not SUT — review |
| `crates/prism-ecs-compile/src/cimage.rs` | 8 | 130 | Migrate as part of module-cohesion decomposition by authority |
| `crates/prism-ecs-compile/src/compiler.rs` | 8 | 30 | Migrate as part of module-cohesion decomposition to `ir_build.rs` + `plan_apply.rs` |
| `crates/prism-ecs-quantization/src/sweep/families/nf4.rs` | 8 | 6 | Quantization sweep family; subsystem cutover target |
| `crates/prism-ecs-core/src/column.rs` | 8 | 0 | Internal storage primitive; `Result` on the column-mutation API |
| `crates/prism-ecs-compile/src/ecs.rs` | 6 | 39 | Migrate as part of module-cohesion decomposition |
| `crates/prism-ecs-codec/src/lib.rs` | 6 | 9 | Serialization layer; typed errors per codec format |
| `crates/prism-ecs-quantization/src/onnx_adapter.rs` | 6 | 5 | Format adapter; typed errors |
| `crates/prism-ecs-core/src/world.rs` | **0** | 0 | **CLEARED** — pilot refactor (2026-07-25). `spawn` constructs `Occupant` once; `stage_component` and `despawn` return `Result<_, WorldError>`. 6 canonical callers updated. Build clean, 19/19 core tests pass, 340/340 IR tests pass. |
| `prism-mcp-core/src/protocol.rs` | **0** | 2 | **CLEARED in production.** The 2 `.expect()` calls in this file are inside `#[cfg(test)] mod tests` and permitted by the rust-quality rule. |
| `crates/prism-ecs-constitutional/src/work.rs` | **0** | 41 | **CLEARED in production.** The 41 `.unwrap()` / `.expect()` calls in this file are all inside `#[cfg(test)] mod tests` and permitted by the rust-quality rule. The earlier "41 violations" entry in this table was a file-level count that included test scope. |

The full per-violation list regenerates from `scripts/unwrap_baseline.py`. The
table above is the priority queue, ordered by production count. Subsystem
ownership is indicated in the right column where it is unambiguous;
cross-cutting violations (compiler, world) are the highest leverage because
they touch every caller.

**Pilot (2026-07-25):** `crates/prism-ecs-core/src/world.rs` (11 → 0
*production* violations, all production unwraps cleared) demonstrated the
pattern: change a public API to return `Result<_, WorldError>`, propagate at
the constitutional call sites, update the test-scope callers to use
`.expect()`. The same pattern applies to the other entries in the table. The
`world.rs` refactor also fixed a structural issue in `spawn` (9 unwraps
collapsed into a single `Occupant` construction). Detailed plan:
`references/rust-quality.md` §The override mechanism (waivers).

The production/test split is enforced by `scripts/unwrap_baseline.py`, which
parses each file to find the `#[cfg(test)] mod tests {` brace block and
excludes the contained lines from the production count. The script is
authoritative; raw `rg '\.unwrap\(\)|\.expect\('` over-counts by including
test scope.

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
