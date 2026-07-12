# Remaining migration work

> **Current state snapshot** — Commit ebeaecb. All claims annotated with:
> - **DECLARED**: type/API exists but may be empty
> - **CONNECTED**: wired into a call chain
> - **EXECUTED**: produces real output from real input
> - **VALIDATED**: tested with real data

---

## PR G — Metal Implementation Catalogue

**Status:** Catalogue and backend compiler exist structurally and are internally wired (`lower()` reads catalogue entries; `compile()` rejects empty source in production), but no production dispatch path consumes them. The legacy `include_str!` / build-time metallib paths remain the only active paths.

### What exists

**MetalImplementationCatalogue** (DECLARED, CONNECTED). Full catalogue with 12 registrations.

- **Megakernel** (`register_megakernel`): source path, entry point, 4 buffer bindings (weights, activations, kv_cache, constants), 5 function constants (hidden_size, num_heads, head_dim, num_layers, seq_len), dispatch policy `FromConstant`, threadgroup (256,1,1). — **DECLARED / CONNECTED**
- **Per-layer decoder** (`register_per_layer`): source path, entry point, 3 buffer bindings, 3 function constants, dispatch policy `FromOutputBuffer`. — **DECLARED / CONNECTED**
- **NF4 linear** (`register_linear_nf4`): source path, entry point, 4 buffer bindings (weights_packed, scales, input, output), 3 function constants (in_features, out_features, group_size). — **DECLARED / CONNECTED**
- **Ternary GEMV** (`register_ternary_gemv`): source path, entry point, 4 buffer bindings, no function constants. — **DECLARED / CONNECTED**
- **RMSNorm** (`register_rmsnorm`): source path, entry point, 3 buffer bindings, 2 function constants. — **DECLARED / CONNECTED**
- **Generic primitives** (`register_primitives`): 7 entries (linear_rawf32, silu, rope, attention_scores, softmax, apply, residual_add) with empty source/ABI — no semantic overlap with explicit NF4/ternary/rmsnorm registrations. — **DECLARED**

**MetalBackendCompiler** (DECLARED, CONNECTED). Implements `BackendCompiler`:

- `lower()`: looks up the first matching entry in the catalogue, reads source from `source_path` (relative to `CARGO_MANIFEST_DIR`), uses `source_entry_point`. Falls back to empty source for registrations with no path. — **CONNECTED internally** (not consumed by any production path)
- `compile()`: rejects empty source with `BackendCompileError::CompilationFailed` in production (`cfg!(test)` mode returns structural artifact for backward compat). Invokes `MetalToolchain::compile_source` for non-empty source. — **CONNECTED internally**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Catalogue consumed by production dispatch" | NOT CONNECTED | `kernel_registry.rs`, `kernel_dispatch.rs`, `region_runner.rs` still use `include_str!` on shader files and `mtl::new_library_with_source` directly. Neither touches the catalogue. |
| "ABI byte_sizes are real" | DECLARED only | All `BufferBinding.byte_size` fields are 0 — they name bindings correctly but assert no actual buffer size. |
| "ABI constant defaults are real" | DECLARED only | All `ConstantBinding.default_value` fields are `None` — they name constants but provide no defaults. |
| "Duplicate NF4/ternary implementations eliminated" | NOT EXECUTED | Three independent NF4 decode implementations still exist (`cimage_linear_nf4.metal`, `nf4_tile640_gemv.metal`, megakernel). Canonical fragments exist in `fragments/` but no shader consumer has been migrated. |
| "lower() used in production" | NOT CONNECTED | `MetalBackendCompiler::lower()` code path exists but no production caller invokes it. |
| "compile() used in production" | NOT CONNECTED | `MetalBackendCompiler::compile()` code path exists but no production caller invokes it. Only tested by unit tests. |

### Remaining

1. **Replace `include_str!` paths with catalogue lookups** in `kernel_registry.rs`, `kernel_dispatch.rs`, `region_runner.rs` — route Megakernel compilation, per-layer dispatch, and primitive invocations through the `MetalBackendCompiler`.
2. **Populate real ABI byte_sizes** — either from shader introspection or from metadata in the catalogue registrations.
3. **Consolidate NF4/ternary/RMSNorm implementations** — migrate `cimage_linear_nf4.metal` and `nf4_tile640_gemv.metal` to use canonical fragments from `fragments/`, then eliminate the independent implementations.
4. **Fallback to generic primitives** — for `linear_rawf32` and other primitives without explicit source, wire generated/dynamic kernel source or template expansion in `lower()` so the catalogue path works for all semantic IDs.

### Gate

Every production Metal dispatch resolves to exactly one catalogue registration. Source, entry point, ABI, and artifact digest are non-empty and validated.

---

## PR H — PrismCompiler Integration

**Status:** `PrismCompiler` is structurally complete and partially wired: `compile()` delegates GGUF sources to the real legacy pipeline through `compile_gguf_to_canonical()`, and a `GgufFrontend` is registered by default. The CLI binary and authority/speculative paths still bypass the canonical API entirely.

### What exists

**PrismCompiler** (DECLARED, PARTIALLY CONNECTED).

- `default()` — now registers `GgufFrontend` (behind `#[cfg(feature = "prism-backend")]`). — **DECLARED (frontend exists), CONNECTED (registered in default)**
- `inspect()` — iterates frontends, delegates to first matching `ModelFrontend::inspect`. — **DECLARED, CONNECTED**
- `plan()` — imports model through first matching frontend, builds structural (empty) `RepresentationPlan`, `ExecutionGraph`, `KernelPlan`. — **DECLARED, CONNECTED (empty sub-plans)**
- `compile()` — GGUF sources: delegates to `compile_gguf_to_canonical()` behind `#[cfg(feature = "mlx-backend")]`. Non-GGUF or missing feature: produces structural outcome from plan. — **DECLARED, CONNECTED (GGUF path delegates to real pipeline)**

**GgufFrontend** (`gguf_frontend.rs`, behind `#[cfg(feature = "prism-backend")]`). Implements `ModelFrontend`:
- `inspect()` — reads GGUF header, extracts metadata, returns `ModelInspection` with arch, tensor count, model type. — **DECLARED, CONNECTED, partially EXECUTED (reads real file metadata)**
- `import()` — reads GGUF header, builds `ModelIr` with configuration, identity, tensor catalogue. — **DECLARED, CONNECTED, partially EXECUTED (reads real file metadata)**

**compile_gguf_to_canonical()** (`pipeline.rs`, behind `#[cfg(feature = "prism-backend")]`). Adapter that runs the real legacy pipeline (`compile_gguf_unchecked`) then wraps the result in canonical types. — **DECLARED, CONNECTED, EXECUTED (runs real pipeline)**

**CompileEvent / CompileEventStream** (`compile_plan.rs`). Rich event types with stage, success, timestamp, duration, message, and optional digests. — **DECLARED, CONNECTED**

**CompileRequest** — carries `source_path`, `source_type`, `output_path`, `policy_path`, `quant_mode`. — **DECLARED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "CLI uses PrismCompiler" | NOT CONNECTED | `prism.rs` lines 760, 777, 793 call `compile_gguf_speculative()`, `compile_gguf_with_authority()`, `compile_gguf_unchecked()` directly. PrismCompiler is not imported. |
| "compile() handles authority/speculative" | DECLARED only | `CompileRequest` lacks fields for `CompilationAuthority`, `HardwareTarget`, `ane_models_dir`, `metallib_path`, `mlx_capture_dir`, `draft_gguf_path`. The `compile()` GGUF path passes `None` for all optional directories. |
| "Plan produces real execution graphs" | DECLARED only | `plan()` returns structural empty execution graph and kernel plan. No real region planning, kernel selection, or memory budgeting occurs. |
| "Event stream has real digests" | DECLARED only | `CompileEvent.source_digest`, `policy_digest`, `artifact_digest`, `toolchain_version` are all `None` — reconstructed post-hoc from pipeline receipts. |
| "compile() feature-gated correctly" | PARTIAL | GGUF delegation in `compile()` uses `#[cfg(feature = "mlx-backend")]` but `compile_gguf_to_canonical` is behind `#[cfg(feature = "prism-backend")]` — mismatch means only builds with both features work. |
| "GgufFrontend import reconstructs full tensor graph" | PARTIAL | `import()` reads GGUF header and builds tensor catalogue metadata, but does not map tensor names to a logical compute graph or wire them into an execution graph. |

### Remaining

1. **Route CLI binary through PrismCompiler** — Replace the three direct legacy calls in `prism.rs` with a single `PrismCompiler::compile()` call. Requires extending `CompileRequest` with authority, hardware target, optional directories.
2. **Unify feature gates** — Align `#[cfg(feature = "mlx-backend")]` in `compile()` with the `#[cfg(feature = "prism-backend")]` gate on `compile_gguf_to_canonical`.
3. **Populate execution graph in plan()** — `plan()` must produce real region plans, kernel selections, and memory budgets instead of structural empties.
4. **Wire event digests** — `CompileEvent` digests should be populated from artifacts produced during compilation, not left `None`.
5. **Join receipt graph** — Connect `CompilerReceipt` to execution receipts for full provenance (parent receipt IDs, kernel implementation identity, timestamps).

### Gate

One GGUF fixture produces a non-empty cimage through `PrismCompiler`. Direct binary calls to unchecked/speculative/authority compilation are absent. Legacy vs. canonical manifests match.

---

## Engram (training, lookup, scheduling)

**Status:** Types exist for engram artifacts, lookup parameters, policies, receipts, and training. The scheduler constructs `DataflowOp::EngramLookup` nodes (not `KvRead`). The trainer exists but produces zero-byte artifacts with empty digests. The resolver emits empty engram targets.

### What exists

**Training target types** (`training_target::spec`):
- `EngramArtifact` — engram_id, tensor_class, insertion_point, codec, payload_size, payload_digest — **DECLARED**
- `EngramLookupParams` — engram_id, lookup_policy (AlwaysApply / ThresholdGate / Scaled), retrieval_threshold — **DECLARED**
- `EngramLookupReceipt` — engram_id, tensor_class, looked_up, timestamp, latency — **DECLARED**

**Dataflow integration** (`plan/fusion.rs`, `plan/fusion/scheduler.rs`):
- `DataflowOp::EngramLookup` variant exists with engram_id, lookup_params, weights, output — **DECLARED**
- Scheduler constructs `EngramLookup` nodes from op kind string `"engram_lookup"` — **DECLARED, CONNECTED**
- Scheduler populates `EngramLookupParams` with `EngramLookupPolicy::AlwaysApply` — **DECLARED, CONNECTED**
- ANE planar lowering rejects `EngramLookup` as unsupported — **DECLARED, CONNECTED, EXECUTED**

**EngramTrainer** (`training_target/engram/trainer.rs`):
- `train()`: accepts `CalibrationEvidence`, returns `(EngramArtifact, EngramTrainingReceipt)` — **DECLARED**
- Produces artifact with `payload_size: 0`, `payload_digest: ""` — **DECLARED, EXECUTED (returns empty artifact)**
- Training loop is documented but not implemented — single step with no gradient or optimization — **DECLARED only**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Trainer uses calibration data" | NOT EXECUTED | `train()` argument `_calibration: &CalibrationEvidence` is prefixed with `_` — unused in function body. |
| "Resolver emits engram targets" | DECLARED only | Resolver produces empty `engram_targets` vectors. No engram target emerges from any real model compilation. |
| "EngramLookup is a real inference op" | DECLARED only | No runtime kernel can execute an engram lookup. The op is lowered to error in ANE path and is a no-op in Metal path. |
| "Engram artifact has real digest" | NOT EXECUTED | `payload_digest` is `String::new()` — not a hash of actual bytes since payload is empty. |

### Remaining

1. **Implement training loop** — Train must analyze calibration residuals, optimize engram weights, and produce a real payload with digest.
2. **Wire resolver** — The execution planner must emit concrete engram targets from model compilation.
3. **Runtime kernel for lookup** — Implement Metal/ANE kernel that applies an engram pattern at the configured insertion point.
4. **Evidence chain** — Link training receipt → artifact digest → inference lookup receipt.

### Gate

A training call with real calibration data produces a non-empty engram artifact with a valid payload digest. The scheduler emits engram targets for a real model graph.

---

## Evolutionary Search (M7 — M10)

**Status:** Foundation types, ECS systems, and config structs exist. The ECS lifecycle (seed, mutate, crossover, evaluate, select) is structurally wired. No real search executes against compiled programs.

### What exists

**Foundation types** (`evolution/foundation.rs`):
- `EvolveCandidate` — tensor_id, target_backend, format, program, measured_cost, generation, parents — **DECLARED**
- `CostMetrics` — wall_ns, energy_uj, alu_cycles, bandwidth_bytes — **DECLARED**
- `EvolutionState` — tensor_id, seed_program, population, generation, best_cost, converged, search_config — **DECLARED**
- `EvolveProgram` — `MetalShader(String)` variant holding shader source — **DECLARED**
- `SearchConfig` — population_size, mutation_rate, crossover_rate, convergence_threshold, cost_function — **DECLARED**

**ECS systems** (`evolution/systems.rs`):
- `evolve_seed()` — spawns `EvolutionState` entity + population from seed program using `mutate_program` — **DECLARED, CONNECTED, VALIDATED (unit tests)**
- `evolve_evaluate()` — records `CostMetrics` on a candidate — **DECLARED, CONNECTED, VALIDATED (unit tests)**
- `evolve_select()` — sorts by wall_ns, updates best, checks convergence, truncates — **DECLARED, CONNECTED, VALIDATED (unit tests)**
- `mutate_program()` — appends `// mutated` comment to shader source — **DECLARED, EXECUTED (syntactic mutation only)**
- `crossover()` — clones parent_a unchanged — **DECLARED, EXECUTED (no-op crossover)**

**MetalDecompositionSearch** (`evolution/decomposition.rs`):
- Config struct with for_nf4() / for_ternary() constructors — **DECLARED, CONNECTED, VALIDATED (unit tests)**
- `DecompositionResult` — winner, cost, converged — **DECLARED**
- No `search()` or `run()` method — **NOT CONNECTED**

**JointGenome / JointSearchConfig** (`evolution/joint.rs`):
- `JointGenome` — program, engram_codec, engram_capacity, insertion_point, retrieval_threshold, tensor_representation, kernel_variant — **DECLARED**
- `JointSearchConfig` — population_size, mutation_rate, engram_configs, kernel_variants — **DECLARED**
- No `search()` or `optimize()` method — **NOT CONNECTED**

**TernaryAssimilation** (`quantization/ternary_assimilation.rs`):
- `TernaryAssimilationConfig` — max_nrmse, min_weight_magnitude, residual_compensation, research_only — **DECLARED, CONNECTED (Default impl)**
- `TernaryAssimilationResult` — tensor_id, nrmse, weights_assimilated, passed — **DECLARED**
- `TernaryAssimilationGate` — evaluate() checks nrmse ≤ max_nrmse — **DECLARED, CONNECTED, VALIDATED (unit tests)**
- No `assimilate()` function — **NOT CONNECTED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Mutation produces real program variants" | DECLARED only | Appends `// mutated` comment — no structural AST changes, no tile geometry modification, no loop unrolling or fusion changes. |
| "Crossover combines programs" | DECLARED only | Returns parent_a unchanged with `#[allow(unused_variables)]` on parent_b. |
| "evolve_evaluate compiles and measures" | NOT EXECUTED | Records externally-supplied metrics only. Does not invoke compilation or run timing. |
| "MetalDecompositionSearch searches" | NOT CONNECTED | Config struct has no run/search method. Only constructs config. |
| "Joint search optimizes engram+tensor jointly" | NOT CONNECTED | Types are parameter containers only. No search loop exists. |
| "Ternary assimilation converts tensors" | NOT CONNECTED | Config/Result/Gate exist but `assimilate()` function is absent. |
| "Evolutionary training data is recorded" | NOT EXECUTED | No system records evolution provenance to execution profile. |

### Remaining

1. **Real program mutation** — `mutate_program` must modify tile sizes, loop orders, unroll factors, or reduction strategies — not just append comments.
2. **Real crossover** — `crossover` must combine programs from two parents meaningfully.
3. **Eval harness** — `evolve_evaluate` must compile the program, run it on the target backend, and measure wall/energy/bandwidth.
4. **Search loop** — `MetalDecompositionSearch` needs `start()` or `run()` that drives the ECS lifecycle (seed → evaluate → select → mutate → crossover → repeat).
5. **Assimilate function** — Implement `ternary_assimilation::assimilate()` that converts a tensor to ternary representation with residual compensation.
6. **Wire to execution_profile** — Evolution provenance (best cost, generations, winning program) must record to `ExecutionProfile`.

### Gate

One Metal program evolves through ≥3 generations with real compilation and measurement. One tensor is assimilated to ternary with valid residual.

---

## Test Coverage

**Status:** Extensive structural unit tests exist. Integration tests exist but all test empty artifact paths or structural properties. No test exercises real compilation, real program search, or real tensor assimilation.

### What exists

- **Catalogue tests** — verify registration counts, source path existence, ABI field presence — **DECLARED, EXECUTED, VALIDATED (field-level checks)**
- **Lower/compile tests** — verify lower returns source from catalogue, compile rejects empty source in production — **DECLARED, EXECUTED, VALIDATED (structural paths)**
- **Evolution tests** — verify seeding, selection sorting, evaluate recording, mutation text — **DECLARED, EXECUTED, VALIDATED (unit-level)**
- **Decomposition config tests** — verify config struct fields — **DECLARED, EXECUTED, VALIDATED**
- **Assimilation gate tests** — verify nrmse threshold evaluation — **DECLARED, EXECUTED, VALIDATED**
- **Compiler tests** — verify `compile()` produces structural outcome for non-GGUF sources — **DECLARED, EXECUTED, VALIDATED (structural)**
- **2690+ tests pass** — **EXECUTED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Real GGUF compilation tested" | NOT VALIDATED | No test fixture compiles a real GGUF to a real cimage through PrismCompiler. |
| "Real program search tested" | NOT VALIDATED | No test runs evolution with real compilation and measurement. |
| "Real tensor assimilation tested" | NOT VALIDATED | No test assimilates a real weight tensor. |
| "End-to-end event digests tested" | NOT VALIDATED | No test checks that CompileEvent digests are non-None after compilation. |

### Remaining

1. **GGUF integration test** — Add a test fixture compiling a known-small GGUF through `PrismCompiler::compile()` and verifying the output cimage is non-empty.
2. **Catalogue end-to-end test** — Add a test running `MetalBackendCompiler::lower()` + `compile()` for a populated registration and verify real `.metallib` bytes.
3. **Evolution integration test** — Add a test driving 3+ generations of evolution with a mock cost function returning varied metrics, then verify the best candidate is selected.
4. **Parallel structural tests** — All existing structural tests should remain passing after each integration test addition.
