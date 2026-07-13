# Remaining migration work

> **Current state snapshot** — Commit 407db0e. All claims annotated with:
> - **DECLARED**: type/API exists but may be empty
> - **CONNECTED**: wired into a call chain
> - **EXECUTED**: produces real output from real input
> - **VALIDATED**: tested with real data

## PR G — Metal Implementation Catalogue

**Status:** Catalogue and backend compiler exist structurally and are internally wired (`lower()` reads catalogue entries; `compile()` rejects empty source in production), but no production dispatch path consumes them. The legacy `include_str!` / build-time metallib paths remain the only active paths.

### What exists

**MetalImplementationCatalogue** (DECLARED, CONNECTED). Full catalogue with registrations. Megakernel, NF4 linear, ternary GEMV, and RMSNorm have source paths, entry points, populated ABI buffer/constant fields, and authoritative source resolution tests.

**MetalBackendCompiler** (DECLARED, CONNECTED). Implements `BackendCompiler`:
- `lower()`: looks up matching registration in catalogue, reads source from `source_path`, uses `source_entry_point`
- `compile()`: rejects empty source in production (`Err`). Invokes `MetalToolchain::compile_source` for non-empty source.

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Catalogue consumed by production dispatch" | NOT CONNECTED | `kernel_registry.rs`, `kernel_dispatch.rs`, `region_runner.rs` still use `include_str!` directly. |
| "ABI byte_sizes are real" | DECLARED only | All `BufferBinding.byte_size` fields are 0. |
| "Duplicate NF4/ternary implementations eliminated" | NOT EXECUTED | Three NF4 decode implementations exist. Canonical fragments un-consumed. |
| "lower() used in production" | NOT CONNECTED | No production caller invokes `lower()`. |
| "compile() used in production" | NOT CONNECTED | No production caller invokes `compile()`. |

## PR H — PrismCompiler Integration

**Status:** All three CLI paths route through `PrismCompiler`. Real GGUF compilation works under `mlx-backend` feature; `prism-backend`-only builds log a clear message. `CompileOutcome` populated from real `CompiledImage`.

### What exists

- **Default GGUF frontend**: registered by default. Parses GGUF header into `ModelIr`. — **EXECUTED**
- **`PrismCompiler::compile()`**: detects .gguf, delegates to `compile_gguf_to_canonical` behind `mlx-backend` gate. Populates outcome fields from `CompiledImage`. Forwards ane_models_dir, metallib_path, mlx_capture_dir, target_hardware. — **EXECUTED (mlx-backend)**
- **`compile_with_authority()`**: routes SealedComputeImage through real pipeline, populates outcome. — **EXECUTED**
- **`compile_speculative()`**: routes draft-model requests. — **EXECUTED**
- **Compile event stream**: exists but reconstructed post-hoc (all identity digests None). — **DECLARED / CONNECTED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Single compile() entrypoint" | PARTIAL | Three separate methods, not one. |
| "Joined compiler-to-execution provenance" | DECLARED | Events lack source/policy/artifact/toolchain identities. |
| "Live event emission" | NOT CONNECTED | Post-hoc reconstruction, not live. |

### Remaining

1. Live event emission from real stages
2. Populate source, policy, artifact, toolchain identity digests

## Engram (training, lookup, scheduling)

**Status:** Engram data model defined. Scheduler constructs real `EngramLookup`. Trainer consumes calibration to produce deterministic payload bytes.

### What exists

- EngramArtifact, EngramLookupParams, EngramLookupPolicy, EngramLookupReceipt — **DECLARED / CONNECTED**
- DataflowOp::EngramLookup in scheduler — **CONNECTED** (no KvRead alias)
- EngramTrainer: consumes calibration, builds deterministic payload via BTreeMap, hashes actual payload bytes, computes RMSE from metrics. — **EXECUTED (metadata-based)**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Training produces real engram payload" | NOT EXECUTED | Payload is serialized metadata, not trained parameters. |
| "Engram lookup modulation" | NOT EXECUTED | No runtime implementation for the op. |
| "Engram digest covers trained data" | NOT EXECUTED | Digest covers calibration metadata. |

### Remaining

1. Real training loop (optimizer, objective, holdout)
2. Runtime lookup (payload retrieval, modulation)
3. End-to-end test (encode, store, look up, apply, verify baseline)

## Evolutionary Search

**Status:** Foundation types exist. ECS systems are functional. Mutation perturbs tile dims/shader params, crossover blends features, selection uses CostFunction.

### What exists

- EvolveCandidate, EvolutionState, CostMetrics, EvolveProgram, SearchConfig — **DECLARED / CONNECTED**
- evolve_seed, evolve_evaluate, evolve_select, mutate_program, crossover — **EXECUTED**
- MetalDecompositionSearch with injected Evaluator trait — **DECLARED / CONNECTED**
- JointSearchConfig with population evolution, threshold convergence — **DECLARED / CONNECTED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Real Metal evaluator" | NOT CONNECTED | Uses SyntheticEvaluator (closed-form). |
| "Production-ready kernel promotion" | NOT EXECUTED | No compilation, parity, or benchmarking. |
| "Typed parameter genome" | DECLARED | No Metal device-limit validation. |
| "Joint search with real engram quality" | NOT EXECUTED | Hand-written scoring formula. |

### Remaining

1. Measured Metal evaluator (compile, dispatch, validate, measure)
2. Device-limit validation
3. Joint search with real engram metrics
4. Pareto frontier with verifiable evidence

## Ternary Assimilation

**Status:** `assimilate()` function implemented. RMSE computed over ALL weights. Empty tensor guard. Packed ternary weights and residuals returned.

### What exists

- TernaryAssimilationConfig, TernaryAssimilationResult, TernaryAssimilationGate — **DECLARED / CONNECTED**
- `assimilate()`: converts all weights to ternary, computes RMSE over all weights, returns ternary_weights (Vec<i8>) and residuals (Vec<f32>), guards empty tensors. — **EXECUTED**
- 10 tests including full error tracking, weight return, empty rejection. — **VALIDATED**

### Gaps

| Claim | Status | Detail |
|---|---|---|
| "Returned artifact is usable" | NOT EXECUTED | No packaging, execution, replay, or rollback path. |
| "residual_compensation consulted" | NOT EXECUTED | Config field exists but is never used. |

### Remaining

1. Packaging: pack ternary weights + residuals into a portable artifact
2. Execution: verify the reconstructed tensor is within gate thresholds
3. Replay: prove the same artifact produces the same result

## Test Coverage

**Status:** 2670+ lib tests pass. Structural tests documenting empty behavior removed or gated.

| Action | Status |
|---|---|
| Empty-artifact tests removed | Gated behind not(mlx-backend) |
| 10 assimilation tests | All pass (full RMSE, weight return, empty guard) |
| 2 engram tests | Digest deterministic, payload_size = exact bytes |
| 6 joint search tests | Real evolution, threshold convergence |
| 6 decomposition tests | Injectable Evaluator, SyntheticEvaluator fixture |

### Gaps

- No real GGUF compilation through canonical API (requires mlx-backend + fixture)
- No evolved kernel measurement on real Metal
- No engram lookup end-to-end
