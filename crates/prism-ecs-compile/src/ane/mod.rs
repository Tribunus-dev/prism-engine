//! `prism_ecs_compile::ane` — Apple Neural Engine (ANE) compile-time
//! primitives.
//!
//! This module is the constitutional home for the engine's
//! `compute-core/src/ecs/ane/` subsystem: hot-row prediction, weight-row
//! cache, MoE expert scheduling, draft model precompilation, MIL program
//! text generation, FP16 conversion, and entropy-based window growth
//! prediction. These are the ANE-specific compile-time primitives that
//! generate the ANE-compatible portion of a CImage.
//!
//! Higher-leverage, engine-coupled adapter code (Core ML harness, IOSurface
//! zero-copy paths, MLX projection executor integration) is engine-internal
//! at `compute-core/src/ecs/legacy_ane/` because it depends on engine FFI
//! bridges and the per-backend ANE/CoreAI/MLX executor stack. This surface
//! is the cross-platform, constitutional home for the data types, the pure
//! logic, and the backend-neutral contracts that all implementations share.
//!
//! # Authority
//!
//! The `ane` surface is the canonical authority for ANE compile-time
//! primitives. The legacy engine surface at
//! `compute-core/src/ecs/legacy_ane/` re-exports these types and hosts
//! the engine-coupled adapter code.
//!
//! # Submodules
//!
//! - [`fp16`] — IEEE 754 binary16 ↔ binary32 conversion (no engine deps).
//! - [`sampling`] — pure sampling helpers (greedy argmax, token probability,
//!   softmax). No engine deps.
//! - [`slot_allocator`] — pure LRU slot allocator for ANE SRAM row slots.
//!   No engine deps.
//! - [`token_routing`] — MoE token-routing data types (`TokenRouting`,
//!   `AneCoreExpertLayout`). No engine deps.
//! - [`moe_scheduler`] — `AneMoEScheduler` (pure scheduling) and
//!   `expert_sram_footprint` helper. The MLX-coupled `ExpertWeights`
//!   payload type and `forward_moe` effect live engine-side.
//! - [`mil_program`] — pure MIL program text generators (KV decompress,
//!   sliding-window attention, L2/L3 compress, L3 decompress). No engine deps.
//! - [`hot_row_predictor`] — `HotRowPredictor` config + statistics. The
//!   Core ML inference path is engine-coupled and lives in `legacy_ane/`.
//! - [`weight_row_cache`] — `WeightRowCache` config + slot re-export. The
//!   IOSurface-backed arena storage is engine-coupled and lives in
//!   `legacy_ane/`.
//! - [`draft_model`] — `AneDraftModel` and `AneMultiCoreDraft` config + a
//!   `DraftBackend` trait. The Core ML backend adapter is engine-coupled.
//! - [`sink_detector`] — `AneSinkDetector` config + CPU entropy heuristic.
//!   The Core ML backend is engine-coupled.
//! - [`page_migration_policy`] — `AnePageMigrationPolicy` config. The
//!   `PageMigrationPolicy` trait impl that calls `AneCompressor` lives
//!   engine-side.

#![forbid(unsafe_code)]

pub mod draft_model;
pub mod error;
pub mod fp16;
pub mod hot_row_predictor;
pub mod mil_program;
pub mod moe_scheduler;
pub mod page_migration_policy;
pub mod sampling;
pub mod sink_detector;
pub mod slot_allocator;
pub mod token_routing;
pub mod weight_row_cache;

pub use draft_model::{
    AneDraftModel, AneDraftModelConfig, AneMultiCoreDraft, DraftBackend, DraftForwardOutput,
};
pub use error::AneError;
pub use fp16::{f16_to_f32, f32_to_f16};
pub use hot_row_predictor::{
    HotRowPredictor, HotRowPredictorBackend, HotRowPredictorConfig, HotRowPredictorStats,
};
pub use mil_program::{
    generate_attention_mil, generate_kv_compress_mil, generate_kv_decompress_mil,
    generate_l3_compress_mil, generate_l3_decompress_mil,
};
pub use moe_scheduler::{
    expert_sram_footprint, select_top_k_for_token, AneMoEScheduler,
};
pub use page_migration_policy::{
    AnePageMigrationPolicy, AnePageMigrationPolicyConfig, MigrationTier, ANE_MIGRATION_POLICY_NAME,
};
pub use sampling::{greedy_argmax, softmax_probabilities, token_probability_from_logits};
pub use sink_detector::{
    cpu_entropy_should_grow, AneSinkDetector, AneSinkDetectorConfig, SinkDetectorBackend,
};
pub use slot_allocator::SlotAllocator;
pub use token_routing::{AneCoreExpertLayout, TokenRouting};
pub use weight_row_cache::{
    WeightRowCache, WeightRowCacheBackend, WeightRowCacheConfig,
};
