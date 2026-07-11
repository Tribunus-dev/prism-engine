# Cross-Crate Duplicate Audit: `src/` vs `compute-core/src/`

## Overview

Audited the workspace root crate (`prism-engine`, `src/`) and the internal crate (`tribunus_compute_core`, `compute-core/src/`) for code duplication, module overlap, dead code, and scheduling architecture fragmentation. Findings below are grouped by category, with file paths, descriptions, and severity (HIGH/MEDIUM/LOW).

---

## DUPLICATE: Independent implementations in both crates

### 1. `src/tokenizer.rs` vs `compute-core/src/tokenizer.rs` — HIGH
**Paths:** `src/tokenizer.rs`, `compute-core/src/tokenizer.rs`
**Description:** Byte-for-byte identical 39-line file. Both define `TribunusTokenizer` wrapping `tokenizers::Tokenizer` with the exact same `from_dir`, `encode`, and `decode` methods. Neither references the other. Since root crate depends on compute-core, this should be `pub use tribunus_compute_core::tokenizer::TribunusTokenizer;` in `src/tokenizer.rs` — or just `pub use tribunus_compute_core::tokenizer;` in `src/lib.rs`. Every call to `TribunusTokenizer` compiles both definitions. At link time, the compiler deduplicates, but the source-level maintenance hazard (fix in one, miss the other) remains.

### 2. `src/lut/cpu_fallback.rs` vs `compute-core/src/lut/evaluator.rs` — MEDIUM
**Paths:** `src/lut/cpu_fallback.rs`, `compute-core/src/lut/evaluator.rs`
**Description:** `cpu_fallback.rs` has `lut_gemv_cpu` and supporting ops; `evaluator.rs` has `evaluate_lut_gemv` and supporting ops. Both implement CPU LUT GEMV as a fallback. The `cpu_fallback.rs` doc comment explicitly acknowledges the duplication: *"These are defined here so the workspace crate compiles without the prism-backend feature (which provides them via compute-core/src/lut/evaluator.rs). TODO: Remove this file once prism-backend becomes a required dependency."* Functions are structurally similar but not byte-for-byte identical. Acknowledged intentional duplication with a removal TODO.

### 3. `src/ane/` vs `compute-core/src/ane/` — MEDIUM
**Paths:** `src/ane/` (10+ files), `compute-core/src/ane/` (10+ files)
**Description:** Both have full module trees under `ane/` but with **completely different content**. `src/ane/` has `mil_builder`, `mil_gen_full`, `mlpackage`, `compile_full_model`, `diffusion_ane`, `coreml_bridge`, `coreml_state`, `coreml_audit`, `arena`, `arena_info` — focused on model compilation to CoreML/ANE MIL. `compute-core/src/ane/` has `draft_model`, `hot_row_predictor`, `kv_decompress_program`, `moe_scheduler`, `page_migration_policy`, `sink_detector`, `weight_row_cache` — focused on ANE runtime optimization. Neither delegates to the other. No re-exports. These are genuinely different concerns sharing a module name, but the naming collision is confusing.

### 4. `src/backend/` vs `compute-core/src/backend/` — LOW
**Paths:** `src/backend/mod.rs`, `compute-core/src/backend/` (40+ files)
**Description:** `src/backend/` is a stub (22 bytes: `pub mod text_encoder;`). `compute-core/src/backend/` is the massive backend implementation: ane, metal, accelerate_lane, heterogeneous_executor, megakernel_backend, cpu_attn, routing, flex_dispatch, NPU/ANE dispatch FFI, etc. Not a true duplicate — root's is a near-empty shell, compute-core's is the real implementation. Could be removed from root.

### 5. `src/compute_backend.rs` vs `compute-core/src/backend/` — LOW
**Paths:** `src/compute_backend.rs`, `compute-core/src/backend/mod.rs`
**Description:** `compute_backend.rs` defines a portable `ComputeBackend` trait with `BackendCaps`, `MemKind`, `TernaryWeights`, and `CpuBackend` — a high-level abstract interface. `compute-core/src/backend/mod.rs` defines `TensorBackend` and concrete implementations (Metal, Accelerate, etc.). Different abstraction levels: one is a portable trait referenced by the scheduler, the other is the actual compute implementation. Not a duplicate per se, but the naming invites confusion.

### 6. `src/video/` vs `compute-core/src/video/` — LOW (facade pattern)
**Paths:** `src/video/mod.rs`, `compute-core/src/video/mod.rs`
**Description:** Root `src/video/` is a Prism facade: `generate_video()` delegates to compute-core's `video_provider` when `generation-video` is enabled, returns `MissingFeature` otherwise. Its own submodules (conv3d, temporal_attention, frame_scheduler, vae_3d) are data types and stub operations — no actual compute implementation. `compute-core/src/video/` has real encoder/decoder implementations for multi-modal video. Correct delegation pattern.

### 7. `src/diffusion/` vs `compute-core/src/diffusion/` — LOW (facade pattern)
**Paths:** `src/diffusion/mod.rs`, `compute-core/src/diffusion/` (3 files)
**Description:** Root `src/diffusion/` is a facade with `generate_text()` delegating to compute-core's `diffusion_provider`. `compute-core/src/diffusion/` has the actual sampler, canvas, and scheduler modules. Correct delegation pattern.

### 8. `src/audio/` vs `compute-core/src/audio/` — LOW (facade pattern)
**Paths:** `src/audio/mod.rs`, `compute-core/src/audio/` (3 files)
**Description:** Root `src/audio/` is a facade with `generate_speech()` delegating to compute-core's `audio_provider`. `compute-core/src/audio/` has the actual encoder and preprocessor. Root also has its own `AudioOp`, `Resampler`, `AudioStreamState` types as stubs — these are independent of compute-core. Correct delegation pattern for the generation entry point.

---

## RE-EXPORT / Delegation (correct pattern)

These modules or symbols are correctly wired: root crate re-exports or calls compute-core symbols with proper feature gating.

| File | What | Target |
|------|------|--------|
| `src/image/mod.rs:58` | `pub use tribunus_compute_core::compute_image::adapter::ComputeImageGenerationAdapter` | compute-core |
| `src/llm/grammar.rs` | re-exports `GrammarTokenizer`, `GrammarNode` | compute-core |
| `src/llm/runtime/scheduler.rs` | wraps `tribunus_compute_core::scheduling::Scheduler` | compute-core |
| `src/llm/runtime/lanes.rs` | uses `tribunus_compute_core::backend::*`, `compute_lane::*` | compute-core |
| `src/llm/runtime/residency.rs` | uses `tribunus_compute_core::kv_cache`, `profiled_executor`, `residency` | compute-core |
| `src/llm/runtime/memory.rs` | uses `tribunus_compute_core::memory::monitor` | compute-core |
| `src/llm/tools.rs` | delegates `parse_and_repair`, `execute_tool_call` | compute-core |
| `src/audio/mod.rs` | `generate_via_compute_core()` → `audio_provider` | compute-core |
| `src/diffusion/mod.rs` | `generate_via_compute_core()` → `diffusion_provider` | compute-core |
| `src/video/mod.rs` | `generate_via_compute_core()` → `video_provider` | compute-core |
| `src/embedding/mod.rs` | `generate_via_compute_core()` → compute-core providers | compute-core |

---

## DEAD CODE: Files with `#[allow(dead_code)]` annotations

### Root `src/`

| File | Line(s) | Likely Cause |
|------|---------|-------------|
| `src/audio/rvq.rs` | 1 | Entire `RvqState` struct |
| `src/audio/temporal_attention.rs` | 1 | Entire `TemporalAttention` struct |
| `src/image/router.rs` | 48 | `find_qualified_provider` function |
| `src/llm/tools.rs` | 490, 500 | `format_node`, `format_node_ctx` |
| `src/llm/runtime/kv.rs` | 47, 620, 626 | `KvManagerInner`, `epoch_to_sequence`, `page_to_block` |
| `src/llm/runtime/lanes.rs` | 256, 258 | `ComputeLaneRouter` struct fields |
| `src/llm/runtime/residency.rs` | 22, 438, 459 | `LoadedModelHandle`, `ComputeWeightResidencyManager` | 
| `src/llm/runtime/scheduler.rs` | 19 | `DispatchRecord` |
| `src/llm/runtime/session.rs` | 216, 232, 260, 267, 275, 295, 313, 322 | `ComputeSessionManager` and state converters |
| `src/lut/cimage_engine.rs` | 277 | `load_one_f32_2d` |
| `src/lut/engine_impl.rs` | 588, 905 | `KVCacheMode`, `lm_head_projection` |
| `src/quantization/palette.rs` | 296 | `vdsp_squared_distances` |

### `compute-core/src/`

| File | Line(s) | Likely Cause |
|------|---------|-------------|
| `compute-core/src/arena.rs` | 29 | `tribunus_arena_alloc_f32` |
| `compute-core/src/backend/ane.rs` | 177 | `slot_mut` |
| `compute-core/src/backend/heterogeneous_executor.rs` | 160, 383, 420, 426 | `slot_assignments`, `allocate_tensor_for_op`, `register_op_outputs`, `op_outputs` |
| `compute-core/src/backend/megakernel_backend.rs` | 26, 28 | `batch_size`, `int4_mode` |
| `compute-core/src/backend/metal.rs` | 40 | `weight_free` |
| `compute-core/src/backend/unified_arena.rs` | 107 | `write_hazards` |
| `compute-core/src/ane/weight_row_cache.rs` | 150 | `dot_buffer` |
| `compute-core/src/bin/prism_server.rs` | 147–714 (40+ annotations) | Large server file with many stubbed config/endpoint structs |
| `compute-core/src/bin/gemma4_ingest.rs` | 808, 1003, 1379, 1491, 1515, 3728 | Ingesting utilities — ingestion tools, likely unused after initial use |
| `compute-core/src/bin/gpu_cluster_assign.rs` | 15, 17, 20, 62 | Cluster assignment tool |

---

## SCHEDULING PROBLEM: Three scheduling modules

There are two scheduling modules in `compute-core/src/` and none in root `src/`. No `src/scheduling/` directory exists.

### Scheduler A: `compute-core/src/scheduling/` — Continuous Batching Scheduler
**Path:** `compute-core/src/scheduling/`
**Modules (50+):** `scheduler.rs`, `batch.rs`, `request.rs`, `slot.rs`, `phase_engine.rs`, `tri_lane_orchestrator.rs`, `prism_session.rs`, `ready_queue.rs`, `token_budget.rs`, `memory_pool.rs`, `prefill_orchestrator.rs`, `lane_executors.rs`, `phase_runner/`, `phase_telemetry.rs`, `benchmark_harness.rs`, `cancellation.rs`, `backpressure.rs`, `completion_bridge.rs`, `activation_binding.rs`, `activation_arena.rs`, `ane_lane_executor.rs`, `ane_artifact_cache.rs`, `metal_decoder.rs`, `metal_lane_executor.rs`, `heterogeneous_executor.rs`, `kv_transaction.rs`, `legacy_adapter.rs`, `outlier_detector.rs`, `phase_cancellation.rs`, `phase_engine_state.rs`, `phase_invocation.rs`, `phase_readiness.rs`, `receipt.rs`, `receipts.rs`, `slot_lease_manager.rs`, `saved_request.rs`, `scheduler_metrics.rs`, `weight_residency.rs`, `work_registry/`, `workspace_receipt.rs`... and more (40+ files total).
**Role:** Request-level continuous batching. Manages request queuing, prefill/decode phase scheduling, batch construction, and token budget allocation. Ported from `ref/omlx/scheduler.py`.
**Feature-gating:** Mix of `cfg(feature = "mlx-backend")` (scheduler, batch, phase_engine, etc.) and unconditional (request, slot, token_budget, tri_lane_orchestrator).

### Scheduler B: `compute-core/src/runtime/scheduling/` — ECS System Schedule Compiler
**Path:** `compute-core/src/runtime/scheduling/`
**Modules (10):** `mod.rs`, `access.rs`, `command.rs`, `component_id.rs`, `error.rs`, `graph.rs`, `manifest.rs`, `metadata.rs`, `schedule.rs`, `tests.rs`
**Role:** System-level ECS schedule. Systems declare access at compile time; the schedule compiler validates causality and hazards, compiles a canonical manifest, and emits a fixed execution order. Used by the runtime's `Schedule` to order ECS system execution across accelerators.
**Feature-gating:** Unconditional (no `cfg` gates visible in `mod.rs`).
**Relationship to Scheduler A:** Complementary. Scheduler A handles *which requests run when* (request-level). Scheduler B handles *what order ECS systems execute in* (system-level). They are not duplicates — they operate at different abstraction levels in the same stack.

### Scheduler C: `src/scheduling/` — DOES NOT EXIST
**Path:** No `src/scheduling/` directory or `pub mod scheduling` in `src/lib.rs`.
**Description:** The root crate has no scheduling module. It relies entirely on `compute-core/src/scheduling/` (via the `llm/runtime/` subsystem). Root `src/lib.rs` has no `pub mod scheduling;` line.

### Assessment
No duplication between A and B — they serve different scheduling layers. The three-scheduling-problem concern is moot since the third (`src/scheduling/`) does not exist. However, the two existing modules have significant code mass (40+ files for A, 10 for B) with limited documentation about their relationship. A module-level doc in each pointing to the other would help maintainers understand the boundary.

---

## FEATURE GATING in root `src/lib.rs`

| Module | Always compiled? | Notes |
|--------|-----------------|-------|
| `ane` | **No** | `#[cfg(feature = "ane")]` |
| `audio` | Yes | Always compiles; internal code gated |
| `compute_backend` | Yes | Always compiles |
| `diffusion` | Yes | Always compiles; fallback to error on missing feature |
| `embedding` | Yes | Always compiles |
| `image` | Yes | Always compiles; internal submodules gated on `generation-image` |
| `llm` | Yes | Always compiles |
| `lut` | Yes | Always compiles |
| `quantization` | Yes | Always compiles |
| `tokenizer` | Yes | Always compiles |
| `video` | Yes | Always compiles |
| `multimodal` | Yes | Always compiles |

Only `ane` is feature-gated at the top level. The rest compile unconditionally but delegate to compute-core via `#[cfg(feature = "...")]` internal paths. This means types like `AudioGenerationReceipt`, `VideoGenerationReceipt`, `EmbeddingResult`, etc. are always available even when the corresponding generation feature is off — they only error at runtime/call-time.

---

## Summary of Action Items

1. **ELIMINATE:** `src/tokenizer.rs` should become `pub use tribunus_compute_core::tokenizer;` — pure duplicate (HIGH severity).
2. **TRACK:** `src/lut/cpu_fallback.rs` removal TODO (medium priority once `prism-backend` is required).
3. **RENAME/CLARIFY:** `src/ane/` vs `compute-core/src/ane/` have unrelated content — consider renaming one to avoid confusion (MEDIUM).
4. **CONSOLIDATE:** `src/backend/` stub can be removed if `text_encoder` lives elsewhere (LOW).
5. **DOCUMENT:** `compute-core/src/scheduling/` and `compute-core/src/runtime/scheduling/` should cross-reference each other (LOW).
6. **MONITOR:** Dead code in `src/llm/runtime/session.rs` (8 `#[allow(dead_code)]` annotations) suggests significant unused code in session management.
7. **MONITOR:** `compute-core/src/bin/prism_server.rs` has 40+ dead_code annotations — suggests large untested or unused surface in the server binary.
