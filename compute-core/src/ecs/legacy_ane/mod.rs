//! `compute-core::ecs::legacy_ane` — engine-internal ANE compile-time
//! surface (legacy continuation).
//!
//! This module is the engine-internal continuation of the absorbed
//! `compute-core/src/ecs/ane/` subsystem. The cross-platform data
//! types, pure logic, MIL program text generators, FP16 helpers,
//! sampling helpers, and the backend-neutral config / statistics /
//! hit-rate surface have been migrated to the constitutional home at
//! `prism_ecs_compile::ane::*` and are re-exported here for
//! source-compatibility with the engine binaries and tests that
//! historically imported them as
//! `tribunus_compute_core::ecs::ane::*`.
//!
//! The engine-coupled adapter code (Core ML harness, IOSurface
//! zero-copy paths, MLX projection executor integration, Core ML
//! `predict_pixelbuffer` FFI calls, MLX `forward_moe`) remains
//! engine-internal here because it depends on engine FFI bridges
//! and per-backend executor stacks that are out of scope for the
//! constitutional crate.
//!
//! # Migration status
//!
//! This module was renamed from `compute-core/src/ecs/ane/` to
//! `compute-core/src/ecs/legacy_ane/` as part of the
//! ane → prism-ecs-compile migration (2026-07-28). The engine
//! re-exports the constitutional types from `prism_ecs_compile::ane::*`
//! and keeps the engine-coupled wrapper types (e.g. `AneMoEScheduler`
//! with `mlx_rs::Array` experts) engine-internal.
//!
//! The architecture safety net
//! (`workspace_legacy_ane_imports`) enforces that no NEW engine code
//! imports the legacy `crate::ecs::ane::` path; it must use either
//! the constitutional surface directly or the engine's
//! `legacy_ane` shim.

pub mod draft_model;
pub mod hot_row_predictor;
pub mod kv_decompress_program;
#[cfg(feature = "mlx-backend")]
pub mod moe_scheduler;
pub mod page_migration_policy;
pub mod sink_detector;
#[cfg(feature = "mlx-backend")]
pub mod weight_row_cache;

// ── Constitutional re-exports ────────────────────────────────────────────
//
// The constitutional surface at `prism_ecs_compile::ane::*` owns the
// cross-platform data types, pure logic, MIL program text generators,
// FP16 helpers, sampling helpers, and the backend-neutral config /
// statistics / hit-rate surface. The engine-internal legacy module
// re-exports them under the same name path so engine binaries that
// historically imported `tribunus_compute_core::ecs::ane::*` continue
// to compile. The re-exports are explicit (not glob) so the migration
// is auditable.

pub use prism_ecs_compile::ane::draft_model::{
    AneDraftModelConfig, DraftBackend, DraftForwardOutput,
};
pub use prism_ecs_compile::ane::error::AneError;
pub use prism_ecs_compile::ane::fp16::{f16_to_f32, f32_to_f16};
pub use prism_ecs_compile::ane::hot_row_predictor::{
    HotRowPredictorBackend, HotRowPredictorConfig, HotRowPredictorStats,
};
pub use prism_ecs_compile::ane::mil_program::{
    generate_attention_mil, generate_kv_compress_mil, generate_kv_decompress_mil,
    generate_l3_compress_mil, generate_l3_decompress_mil,
};
pub use prism_ecs_compile::ane::moe_scheduler::{
    expert_sram_footprint, select_top_k_for_token,
};
pub use prism_ecs_compile::ane::page_migration_policy::{
    AnePageMigrationPolicyConfig, MigrationTier, ANE_MIGRATION_POLICY_NAME,
};
pub use prism_ecs_compile::ane::sampling::{
    greedy_argmax, softmax_probabilities, token_probability_from_logits,
};
pub use prism_ecs_compile::ane::sink_detector::{
    cpu_entropy_should_grow, AneSinkDetectorConfig, SinkDetectorBackend,
};
pub use prism_ecs_compile::ane::slot_allocator::SlotAllocator;
pub use prism_ecs_compile::ane::token_routing::{AneCoreExpertLayout, TokenRouting};
pub use prism_ecs_compile::ane::weight_row_cache::{
    WeightRowCacheBackend, WeightRowCacheConfig,
};

// ── Engine-coupled shim types ────────────────────────────────────────────
//
// The engine-coupled implementations of `AneMoEScheduler` (with
// `mlx_rs::Array` experts and `forward_moe`), `AneDraftModel`
// (with `Arena` / `CoreAiModel` fields), `HotRowPredictor`
// (with `Arena` / `CoreAiModel` fields), `WeightRowCache`
// (with `Arena` / `MlxBackend` fields), `AneSinkDetector`
// (with `Arena` / `CoreAiModel` fields), and `AneCompressor`
// (with `Arena` / `CoreAiModel` / FFI) live in their respective
// submodules below. They are re-exported here under their original
// names for engine-internal use.
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::legacy_ane::draft_model::{AneDraftModel, AneMultiCoreDraft};
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::legacy_ane::hot_row_predictor::HotRowPredictor;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::legacy_ane::sink_detector::AneSinkDetector;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::legacy_ane::weight_row_cache::WeightRowCache;
