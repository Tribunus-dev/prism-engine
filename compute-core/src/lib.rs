#![allow(unexpected_cfgs)]

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[cfg(not(any(
    feature = "backend-cpu",
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "candle-cpu",
    feature = "intel",
    feature = "tensix",
    feature = "amd-rocm",
    feature = "stub-backend",
    feature = "storage-adapters",
)))]
compile_error!(
    "Tribunus Compute requires a supported backend: Apple Silicon (macOS arm64), Candle CPU (Linux x86), or a stub/storage backend feature."
);

extern crate self as tribunus_compute_core;

#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::ane;
/// ANE runtime — planar engine program descriptor and lowering.
#[cfg(target_os = "macos")]
pub use crate::ecs::ane_runtime;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::analysis;
#[cfg(any(
    target_os = "macos",
    all(feature = "prism-backend-ios", target_os = "ios")
))]
pub use crate::ecs::core::ane_bridge;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub use crate::ecs::core::ane_compile;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::arena;
#[cfg(target_os = "macos")]
pub use crate::ecs::core::arena_info;
#[cfg(any(feature = "prism-backend", feature = "prism-backend-ios"))]
pub use crate::ecs::tts;
// Pure Rust (atomics + uuid) — must stay unconditional: `errors.rs` (an
// unconditional module) imports `arena_lifecycle::LifecycleState`, so gating
// this to macOS broke every non-mac build.
#[cfg(any(target_os = "macos", feature = "amd-rocm"))]
pub mod aot_kernels;
pub use crate::ecs::assistant_graph;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::audio;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::autopsy;
pub use crate::ecs::backend;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::benchmark;
pub use crate::ecs::bitnet;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::cache;
#[cfg(any(feature = "prism-backend", feature = "prism-backend-ios"))]
pub use crate::ecs::calibration;
pub use crate::ecs::cimage;
pub use crate::ecs::cimage_runtime;
pub use crate::ecs::compilation;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::compile;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::compiler;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "ffi"
))]
pub use crate::ecs::compute_image;
pub use crate::ecs::compute_image_v0;
pub use crate::ecs::config;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::contracts;
pub use crate::ecs::core::arena_lifecycle;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::arena_pool;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::attention;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::audio_preprocess_accelerate;
#[cfg(feature = "generation-tts")]
pub use crate::ecs::core::audio_provider;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub use crate::ecs::core::bridge;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::capability;
pub use crate::ecs::core::cli;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::compile_pipeline;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::compile_progress;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::compile_state;
pub use crate::ecs::core::compute_ir;
pub use crate::ecs::core::compute_lane;
pub use crate::ecs::core::compute_service;
pub use crate::ecs::core::config_namespace;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::copy_ledger;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::coreai_audit;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::coreai_bridge;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::coreai_pipeline;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::core::coreai_state;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::cpu_benchmarks;
pub use crate::ecs::core::crash_breadcrumb;
#[cfg(feature = "generation-diffusion")]
pub use crate::ecs::core::diffusion_provider;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub use crate::ecs::coreai;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::decode_attribution;
/// Device registry — runtime hardware enumeration and capability discovery.
pub use crate::ecs::device;
pub use crate::ecs::diffusion;
/// ECS compiler pipeline — entity-component-system world, systems, and component types.
pub mod ecs;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::engine;
pub use crate::ecs::core::engine_error;
pub use crate::ecs::core::engine_policy;
pub use crate::ecs::core::engine_receipts;
pub use crate::ecs::core::errors;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::executor_projection;
pub use crate::ecs::core::experiment;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::external_array;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "ffi"
))]
pub use crate::ecs::core::ffi;
pub use crate::ecs::core::fusion_region;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::gemma;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::gguf;
pub use crate::ecs::core::gpu_memory;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::gpu_worker;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::heterogeneous;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::hybrid_profile;
#[cfg(feature = "generation-image")]
pub use crate::ecs::core::image_provider;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::integration;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::layout_compiler;
pub use crate::ecs::core::layout_transform;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::loader;
/// Ternary codec — 2-bit {-1, 0, +1} quantization.
#[cfg(feature = "generation-video")]
pub use crate::ecs::core::video_provider;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::evidence;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::generation;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::inference;
pub use crate::ecs::inference_profile;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::kv_arena;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::kv_cache;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::parsing;
pub use crate::ecs::ternary;
#[macro_use]
pub use crate::ecs::core::logging;
/// AMD ROCm backend — multi-die GPU compute module for AMD hardware.
#[cfg(feature = "amd-rocm")]
pub use crate::ecs::core::amd_rocm;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::editing;
/// CPU fusion backend — Accelerate + Rayon as a first-class fusion candidate.
pub use crate::ecs::cpu_runtime;
/// Execution plan — kernel specialization, region batching, and plan data types.
pub mod execution_plan;
/// Pure-data types for KV cache, usable without mlx dependency.
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::kv_cache_types;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::lora;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::mapped_image;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub use crate::ecs::core::metal_capture;
#[cfg(feature = "metal-dispatch")]
pub use crate::ecs::core::metal_launcher;
pub use crate::ecs::core::metrics;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::core::mil_builder;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::core::mlpackage;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::mlx_api_compat;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::mlx_executor;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::mlx_inventory;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::mlx_patch_register;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::mlx_runtime_probe;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::model;
/// Execution profiling — measures whether a codec policy is worth using.
pub use crate::ecs::execution_profile;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::exo;
pub use crate::ecs::lut;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::memory;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub use crate::ecs::metal_runtime;
pub mod model_adapter;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::agent;
#[cfg(feature = "candle-cpu")]
pub use crate::ecs::core::candle_cpu_backend;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::model_cache;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::model_runtime;
pub use crate::ecs::core::model_store;
pub use crate::ecs::core::native_kernel;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::operation_catalog;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::core::pipeline_parity;
pub use crate::ecs::core::placement_profile;
pub use crate::ecs::core::plugin;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::primitives;
pub use crate::ecs::core::profile_compiler;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::profiled_executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::profiled_model;
#[cfg(any(feature = "mlx-backend", feature = "candle-cpu"))]
pub use crate::ecs::core::projection_executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::projection_identity;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::projection_tests;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "candle-cpu",
    feature = "intel",
    feature = "tensix",
))]
/// Always-available projection data types (see module docs).
pub use crate::ecs::core::projection_types;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::quantized;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::readiness_gates;
pub use crate::ecs::core::receipt;
pub use crate::ecs::core::receipts;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::replay_projection;
pub use crate::ecs::core::requalification;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::core::research_contracts;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::core::research_metrics;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::core::research_trace;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::residency;
pub use crate::ecs::core::ring;
pub use crate::ecs::core::runtime_contract;
pub use crate::ecs::core::runtime_orchestration;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::runtime_trace;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::session;
#[cfg(feature = "mlx-backend")]
pub use crate::ecs::core::sidecar;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::speculative;
pub use crate::ecs::core::storage_kernel;
pub use crate::ecs::core::streaming;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::supervisor_crash;
pub use crate::ecs::core::tokenizer;
pub use crate::ecs::core::toolchain_attest;
pub use crate::ecs::core::transform_recipe;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub use crate::ecs::core::treatment;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::validator;
pub use crate::ecs::core::worker_crash_ledger;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::worker_dispatch;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::core::worker_memory;
pub use crate::ecs::core::worker_protocol;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::models;
pub use crate::ecs::nf4tile640;
pub use crate::ecs::quantization;
pub use crate::ecs::reasoning_evidence;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::ecs::registry;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::research;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::runtime;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::scheduling;
#[cfg(all(target_os = "macos", feature = "server"))]
pub use crate::ecs::server;
pub use crate::ecs::state_store;
#[cfg(feature = "storage-adapters")]
pub use crate::ecs::storage_adapters;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::tools;
/// Training-aware compilation — targets, gates, feedback, and receipts.
pub use crate::ecs::training_target;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::video;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub use crate::ecs::vision;
#[cfg(feature = "mlx-backend")]
pub use crate::session::{
    ControlSessionState, GenerationControlSession, InferenceSession, InferenceSessionState,
    SamplerConfig,
};
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use coreml_proto;

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub use crate::compilation::phase_ir::{
    LogicalTensorId, MaterializationPlan, PhaseEdge, PhaseRegion, RegionId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    InvalidArg,
    GenericFailure,
    InternalError,
    Cancelled,
    Timeout,
}

#[derive(Debug)]
pub struct Error {
    pub status: Status,
    pub reason: String,
}

impl Error {
    pub fn new(status: Status, reason: impl Into<String>) -> Self {
        Self {
            status,
            reason: reason.into(),
        }
    }
    pub fn from_reason(reason: impl Into<String>) -> Self {
        Self {
            status: Status::GenericFailure,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Current timestamp as ISO 8601 UTC string.
pub fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO 8601 (simple: YYYY-MM-DDTHH:MM:SSZ)
    let days = (secs / 86400) as i64;
    let time_secs = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    let hour = time_secs / 3600;
    let min = (time_secs % 3600) / 60;
    let sec = time_secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, min, sec
    )
}

/// Hostname or "unknown" if unavailable.
pub fn hostname_or_default() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Convert a days-from-epoch value to (year, month, day) in the Gregorian
/// civil calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Algorithm from Howard Hinnant
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as i64, d as i64)
}
