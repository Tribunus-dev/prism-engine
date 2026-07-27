# Godfile decomposition + engine mapping — Phase 0

**Date:** 2026-07-27
**Status:** Phase 0 (mapping) → ready for Phase 1 dispatch
**Pattern:** Two-birds-one-stone decomposition. Each godfile is decomposed into
focused sub-modules by single authority. For each new sub-module, classify it as
**canonical** (lands in a constitutional crate) or **execution-boundary**
(stays engine-only with a typed port interface). Engine code that maps to
canonical sub-modules is absorbed in the same commit.

## The four criteria for canonical vs execution-boundary

This is the only rule that matters. Do not classify by "is it engine code" — the
compute-core absorption arc is precisely the act of moving engine code into
constitutional crates. Classify per sub-module using these four criteria
(extracted from `AGENTS.md`):

1. **Owns hardware handles, file descriptors, OS primitives?** → execution-boundary
   - Examples: `MetalComputeEncoder`, `ANEProgramHandle`, kqueue/epoll, mmap-backed buffers (when the buffer is the handle, not the byte format), GPU dispatch queues
2. **Uses `unsafe`?** → hardware crates and `prism-ecs-core` only; otherwise boundary
   - Per AGENTS.md: "No `unsafe` in constitutional, runtime, server, or protocol crates"
3. **Owns process-local state** (channels, locks, mpsc receivers, `OnceLock`)? → execution-boundary
   - Examples: `WorkerIngressSystem::run` receivers, heterogeneous executor's tokio actor, per-lane `LaneExecutor` references, the slot lease manager's reader-count tracking
4. **Raw FFI to a hardware/OS surface?** → execution-boundary
   - Examples: CoreML/Accelerate FFI, MLX bindings, ANE compiler shim, libxpc

Everything else is canonical. Schemas, types, validation, plans, IR, receipts,
projections, command shapes, replays — all canonical, regardless of origin.

## Per-godfile mapping

### 1. `crates/prism-ecs-constitutional/src/world_txn.rs` (1147 LOC, 84 pub)

**Authority surface:** canonical WorldTxn shape — `AccessKind`, `AccessDeclaration`,
`ComponentChange`, `ChangeType`, `ClassifiedComponent`, `DurableClass`,
`DurableComponent`, `CommittedEpoch`, `WorldTxn`, `WorldTxnError`.

**Engine counterparts (TWO already exist — this is the duplication problem):**
- `compute-core/src/ecs/runtime/world_txn.rs` (from `ebcaf2bc`) — engine-local copy
- `compute-core/src/ecs/runtime/constitutional_world_txn.rs` (from `e633567e`) — bridge copy

**Decomposition axis (per single authority):**
- `access.rs` — `AccessKind`, `AccessDeclaration`, dependency declarations
- `journal.rs` — `ComponentChange`, `ChangeType`, journal types
- `durable.rs` — `DurableClass`, `DurableComponent`, classification
- `txn.rs` — `WorldTxn` itself, `stage_*` methods, `commit`/`abort`
- `epoch.rs` — `CommittedEpoch`, epoch transitions
- `error.rs` — `WorldTxnError` with `Rejected`/`Failed`/`Stale` variants

**Engine mapping decision:**
- All sub-modules are **canonical** (no hardware, no unsafe, no process-local)
- Engine's two copies of `WorldTxn` get **consolidated** — pick the constitutional
  one as canonical, engine gets a re-export shim
- Engine's `constitutional_world_txn.rs` bridge becomes obsolete (delete it)

### 2. `crates/prism-ecs-constitutional/src/compilation.rs` (1190 LOC, 91 pub)

**Authority surface:** Compilation jobs, job lifecycle, validation receipts,
quantization plans, cimage promotion — schemas 31-39.

**Engine counterparts:**
- `compute-core/src/ecs/compile/{audio,mod,pipeline,vision}.rs` (4 files, partially absorbed)
- `compute-core/src/ecs/core/compile_pipeline.rs` (1,930 LOC, partially absorbed by `ddb2d261` → `compile_pipeline::pipeline_parity::*`)
- `compute-core/src/ecs/core/compile_state.rs` (~600 LOC)
- `compute-core/src/ecs/core/compile_progress.rs` (~400 LOC)

**Decomposition axis:**
- `job.rs` — `CompilationJob`, `JobConfig`, `JobInput`, `JobOutput`, `JobLifecycle`
- `validation.rs` — `ValidationReceipt`, validation logic
- `quantization.rs` — `QuantizationPlan`, `QuantizationResult`
- `cimage_promotion.rs` — cimage promotion schema, promotion flow
- `schema_ids.rs` — schema 31-39 constants

**Engine mapping decision:**
- All sub-modules are **canonical** (no hardware, no unsafe, no FFI)
- Engine `compile/pipeline.rs` partially absorbed already; remaining parts land in `job.rs` + `quantization.rs`
- Engine `core/compile_state.rs` and `compile_progress.rs` are observation surfaces — they could be either canonical (as projection state) or execution-boundary (as progress reporting). Default: canonical, but flag for review.

### 3. `crates/prism-ecs-runtime/src/kernel.rs` (1979 LOC, 63 pub)

**Authority surface:** Runtime kernel — the 8-stage schedule executor, command
dispatch, agent snapshots, kernel health, planner/admit/publish/observe marker
components.

**Engine counterparts:**
- `compute-core/src/ecs/core/executor.rs` (1,308 LOC) — direct counterpart
- `compute-core/src/ecs/core/executor_projection.rs` (1,074 LOC) — projection
- `compute-core/src/ecs/system/kernel_catalog.rs` (8 mutations, ported in e633567e)
- Already absorbed: `kernel_generation.rs` (472d9754), `buffer_lifetime_plan.rs` (472d9754), `engine_systems.rs` (472d9754)

**Decomposition axis:**
- `markers.rs` — `PlannedMarker`, `AdmittedMarker`, `PublishedMarker` (Component impls)
- `command_dispatch.rs` — command/effect dispatch logic
- `agent_snapshot.rs` — `AgentSnapshot` + related state
- `kernel_health.rs` — `KernelHealth`, health reporting
- `executor_loop.rs` — the 8-stage schedule executor (or this could be a port — see below)

**Engine mapping decision:**
- `markers.rs` is **canonical** (just `Component` impls, no behavior)
- `command_dispatch.rs` is **execution-boundary** (it walks the mpsc receiver per AGENTS.md criterion 3)
- `agent_snapshot.rs` is **canonical** (data type)
- `kernel_health.rs` is **canonical** (data type + computation)
- `executor_loop.rs` is **execution-boundary** (per criterion 3, owns runtime state)
- Engine `core/executor.rs` and `executor_projection.rs` — the projection part is canonical, the executor loop is execution-boundary; split them

### 4. `crates/prism-ecs-compile/src/ecs.rs` (2581 LOC, 61 pub)

**Authority surface:** ECS compilation components — session components,
compilation orchestrator, world resources, pipeline-stage state attachments.

**Engine counterparts:**
- `compute-core/src/ecs/core/compile_pipeline.rs` (1,930 LOC, partially absorbed)
- `compute-core/src/ecs/core/profiled_model.rs` (1,339 LOC)
- `compute-core/src/ecs/core/pipeline_parity.rs` (1,930 LOC, already absorbed in ddb2d261 → `pipeline_parity::*` 8 files)
- `compute-core/src/ecs/runtime/compilation_systems.rs` (still in engine)

**Decomposition axis:**
- `components.rs` — `CompilationSession`, `SourceModel`, `TensorCollection`, `SpatialGraphComponent`, `SearchStateComponent`, `LegalizedPlan`, `KernelCollection`, `CImageArtifact`
- `resources.rs` — `SessionHandle`, `CurrentSource`, `VecEventSink`, `TargetCaps`, `ExecutionMode`
- `orchestrator.rs` — `CompilationOrchestrator`, the world+session-entity pipeline driver
- `stage_systems.rs` — per-stage system functions (`sync_compilation_entity`, etc.)

**Engine mapping decision:**
- `components.rs` is **canonical** (data types)
- `resources.rs` is **canonical** (data types)
- `orchestrator.rs` is **canonical** (drives the canonical pipeline; doesn't talk to hardware directly)
- `stage_systems.rs` is **canonical** (functions on a `World`)
- Engine `core/compile_pipeline.rs` remaining parts land in `orchestrator.rs`
- Engine `core/profiled_model.rs` — partially canonical (the data), partially execution-boundary (the profiling — uses timers and possibly hardware counters); split it

### 5. `crates/prism-ecs-compile/src/evaluator.rs` (1784 LOC, 32 pub)

**Authority surface:** Evaluator integration for evolutionary search —
production-mode evaluation, canary window, KV cache candidate evaluation,
hardware-backed measurement with fail-closed semantics.

**Engine counterparts:**
- `compute-core/src/ecs/core/speculative.rs` (1,356 LOC, already partially absorbed in ddb2d261 → `speculative_decoding.rs` 862 LOC)
- `compute-core/src/ecs/core/profiled_model.rs` (1,339 LOC) — overlap with `ecs.rs` godfile

**Decomposition axis:**
- `canary_window.rs` — `CanaryWindow` (bounded working set for canary evaluation)
- `kv_evaluator.rs` — `KvCompressionEvaluator`, `KvCompressionEvidence`
- `strategy.rs` — `EvaluationStrategy`, `ProgressiveStageExecutor` wrappers
- `objective.rs` — `TernaryObjectiveEvidence`, objective composition
- `fail_closed.rs` — production-mode fail-closed semantics

**Engine mapping decision:**
- `canary_window.rs` is **canonical** (data type)
- `kv_evaluator.rs` is **canonical** (the evaluation logic, not the execution)
- `strategy.rs` is **canonical** (strategy types)
- `objective.rs` is **canonical** (evidence types)
- `fail_closed.rs` is **canonical** (semantic gate; doesn't touch hardware)
- Engine `core/speculative.rs` remaining parts land in `strategy.rs` + `objective.rs`

### 6. `crates/prism-ecs-server/src/runtime/server.rs` (2284 LOC, 7 pub)

**Authority surface:** Server runtime — request handling, session lifecycle,
resource claims, cancel/recovery flow, modality dispatch.

**Note:** the path is `prism-ecs-server/src/runtime/server.rs`, not `prism-ecs-server/src/server.rs`. The `runtime/` subdirectory was already decomposed in C-2.

**Engine counterparts:**
- `compute-core/src/ecs/core/session.rs` (988 LOC)
- `compute-core/src/ecs/core/engine.rs` (1,389 LOC) — the engine's main "server"
- `compute-core/src/ecs/core/mlx_inventory.rs` (953 LOC) — execution-boundary

**Decomposition axis:**
- `session_lifecycle.rs` — session create/destroy/extend
- `request_handling.rs` — request → command translation
- `resource_claims.rs` — server-side resource claim management
- `cancel_recovery.rs` — cancel propagation, recovery reports
- `modality_dispatch.rs` — modality routing

**Engine mapping decision:**
- `session_lifecycle.rs` is **canonical** (session schema)
- `request_handling.rs` is **canonical** (request shapes, no hardware)
- `resource_claims.rs` is **canonical** (claim data; the slot lease lock state stays engine)
- `cancel_recovery.rs` is **canonical** (recovery report types)
- `modality_dispatch.rs` is **canonical** (routing, not execution)
- Engine `core/session.rs` lands in `session_lifecycle.rs`
- Engine `core/engine.rs` — the request handling parts land in `request_handling.rs`; the MLX dispatch parts stay in engine (criterion 1)

### 7. `crates/prism-ecs-server/src/engine/bpe_tokenizer.rs` (2256 LOC, 38 pub)

**Authority surface:** Pure-Rust HuggingFace tokenizer — BPE, WordPiece, Unigram,
pre/post-processing, truncation, padding, decoding.

**Engine counterparts:**
- `compute-core/src/ecs/core/tokenizer.rs` (small)
- `compute-core/src/ecs/tokenizer.rs`
- `compute-core/src/ecs/parsing/tokenizer/` (directory)

**Decomposition axis:**
- `model.rs` — model types (BPE, WordPiece, Unigram) and their construction
- `pretokenizer.rs` — pre-tokenization (whitespace, metaspaces, punctuation splits)
- `normalizer.rs` — normalization (lowercase, NFC, BERT-style, etc.)
- `postprocessor.rs` — post-processing (template processors, special token insertion)
- `decoder.rs` — decoding back to text
- `truncation_padding.rs` — truncation and padding
- `encoding.rs` — `Encoding` struct, attention masks, type ids, word_ids
- `loader.rs` — loading from `tokenizer.json`

**Engine mapping decision:**
- **All sub-modules are canonical.** The tokenizer is pure data transformation with no hardware, no `unsafe`, no process-local state, no FFI. This is the cleanest case.
- Engine `core/tokenizer.rs` + `tokenizer.rs` + `parsing/tokenizer/` all land in respective sub-modules
- Note: the engine's tokenizer is a thin C++ wrapper; the constitutional one is the actual pure-Rust implementation, so this is an **absorption**, not a duplication

## Order of dispatch

1. **world_txn.rs** — highest leverage (already has 2 engine duplicates to consolidate)
2. **compilation.rs** — strong engine mapping, schema 31-39 boundary is clean
3. **ecs.rs and evaluator.rs together** — same crate, related compilation surface
4. **kernel.rs** — heavy engine counterpart (`core/executor.rs`)
5. **bpe_tokenizer.rs** — cleanest case, no execution-boundary
6. **server.rs** — last (mixed canonical/execution, requires care)

## Per-agent brief structure (for Phase 1 dispatch)

Each agent receives:
1. This mapping doc (just for their godfile)
2. The four canonical-vs-execution-boundary criteria (above)
3. The decomposition axis (above) as a starting point, but they may adjust
4. Hard rules: no `unsafe` in non-hardware crates, no `unwrap`/`expect`/`panic!`
   in production paths, no `anyhow::Error` in constitutional/runtime/kernel,
   `BTreeMap` for canonical collections, newtypes for authority-bearing values
5. Workflow: decompose → classify per sub-module → port engine counterpart
   for canonical sub-modules → document boundary for execution-boundary
   sub-modules → tests → commit → report back
6. Recovery: if you hit the token limit, commit what you have and write a
   partial changelog. The parent session will recover from the snapshot.

## Out of scope (separate effort)

- `prism-ecs-quantization/src/contract.rs` 203 pub (the worst pub count in the
  workspace) — separate problem
- Duplicate `CreateWorkCommand` between `lifecycle_command.rs` and `work.rs` —
  separate cleanup
- 18 mutations in `#[cfg(feature = "legacy_mutations")]` test blocks — separate
  cleanup
- Engine build cleanup (100+ pre-existing errors) — separate effort
- MCP product decision (constitutionalize vs separate boundary) — separate
