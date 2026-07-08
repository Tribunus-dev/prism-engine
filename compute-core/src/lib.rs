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
    feature = "stub-backend",
    feature = "storage-adapters",
)))]
compile_error!(
    "Tribunus Compute requires a supported backend: Apple Silicon (macOS arm64), Candle CPU (Linux x86), or a stub/storage backend feature."
);

extern crate self as tribunus_compute_core;

#[cfg(feature = "mlx-backend")]
pub mod analysis;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod ane;
#[cfg(any(
    target_os = "macos",
    all(feature = "prism-backend-ios", target_os = "ios")
))]
pub mod ane_bridge;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod ane_compile;
/// ANE runtime — planar engine program descriptor and lowering.
#[cfg(target_os = "macos")]
pub mod ane_runtime;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod arena;
#[cfg(target_os = "macos")]
pub mod arena_info;
#[cfg(any(feature = "prism-backend", feature = "prism-backend-ios"))]
pub mod tts;
// Pure Rust (atomics + uuid) — must stay unconditional: `errors.rs` (an
// unconditional module) imports `arena_lifecycle::LifecycleState`, so gating
// this to macOS broke every non-mac build.
pub mod arena_lifecycle;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod arena_pool;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod attention;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod audio;
#[cfg(any(
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "mlx-backend"
))]
pub mod audio_preprocess_accelerate;
#[cfg(feature = "generation-tts")]
pub mod audio_provider;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod autopsy;
pub mod backend;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod benchmark;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod bridge;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod cache;
#[cfg(any(feature = "prism-backend", feature = "prism-backend-ios"))]
pub mod calibration;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod capability;
pub mod cli;
pub mod compilation;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compile;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compile_pipeline;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compile_progress;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compile_state;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod compiler;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "ffi"
))]
pub mod compute_image;
pub mod compute_image_v0;
pub mod compute_ir;
pub mod compute_lane;
pub mod compute_service;
pub mod config;
pub mod config_namespace;
#[cfg(feature = "mlx-backend")]
pub mod contracts;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod copy_ledger;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod coreai;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod coreai_audit;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod coreai_bridge;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod coreai_pipeline;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend"),
))]
pub mod coreai_state;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod cpu_benchmarks;
pub mod crash_breadcrumb;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod decode_attribution;
/// Device registry — runtime hardware enumeration and capability discovery.
pub mod device;
pub mod diffusion;
#[cfg(feature = "generation-diffusion")]
pub mod diffusion_provider;
#[cfg(feature = "mlx-backend")]
pub mod engine;
pub mod engine_error;
pub mod engine_policy;
pub mod engine_receipts;
pub mod errors;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod evidence;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod executor_projection;
pub mod experiment;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod external_array;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "ffi"
))]
pub mod ffi;
pub mod fusion_region;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod gemma;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod generation;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod gguf;
pub mod gpu_memory;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod gpu_worker;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod heterogeneous;
#[cfg(feature = "mlx-backend")]
pub mod hybrid_profile;
#[cfg(feature = "generation-image")]
pub mod image_provider;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod inference;
pub mod inference_profile;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod integration;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod kv_arena;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod kv_cache;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod layout_compiler;
pub mod layout_transform;
#[cfg(feature = "mlx-backend")]
pub mod loader;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod parsing;
#[cfg(feature = "generation-video")]
pub mod video_provider;
#[macro_use]
pub mod logging;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod editing;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod exo;
/// Pure-data types for KV cache, usable without mlx dependency.
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod kv_cache_types;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod lora;
pub mod lut;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod mapped_image;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod memory;
#[cfg(all(target_os = "macos", feature = "mlx-backend"))]
pub mod metal_capture;
#[cfg(feature = "metal-dispatch")]
pub mod metal_launcher;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod metal_runtime;
pub mod metrics;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod mil_builder;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod mlpackage;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod mlx_api_compat;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod mlx_executor;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod mlx_inventory;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod mlx_patch_register;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod mlx_runtime_probe;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod model;
pub mod model_adapter;
#[cfg(feature = "mlx-backend")]
pub mod model_cache;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod model_runtime;
pub mod model_store;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod models;
pub mod native_kernel;
pub mod nf4tile640;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod operation_catalog;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod pipeline_parity;
pub mod placement_profile;
pub mod plugin;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod primitives;
pub mod profile_compiler;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod profiled_executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod profiled_model;
#[cfg(any(feature = "mlx-backend", feature = "candle-cpu"))]
pub mod projection_executor;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod projection_identity;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod projection_tests;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios",
    feature = "candle-cpu",
    feature = "intel",
    feature = "tensix",
))]
/// Always-available projection data types (see module docs).
pub mod projection_types;
pub mod quantization;
/// Execution profiling — measures whether a codec policy is worth using.
pub mod execution_profile;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod quantized;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod readiness_gates;
pub mod receipt;
pub mod receipts;
/// Execution plan — kernel specialization, region batching, and plan data types.
pub mod execution_plan;
/// CPU fusion backend — Accelerate + Rayon as a first-class fusion candidate.
pub mod cpu_runtime;
/// Training-aware compilation — targets, gates, feedback, and receipts.
pub mod training_target;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod registry;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod replay_projection;
pub mod requalification;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod research;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod research_contracts;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod research_metrics;
#[cfg(feature = "mlx-backend")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod research_trace;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod residency;
pub mod ring;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod runtime;
pub mod runtime_contract;
pub mod runtime_orchestration;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod runtime_trace;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod scheduling;
#[cfg(all(target_os = "macos", feature = "server"))]
pub mod server;
#[cfg(feature = "mlx-backend")]
pub mod session;

#[cfg(feature = "mlx-backend")]
pub mod sidecar;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod speculative;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod supervisor_crash;

#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod agent;
#[cfg(feature = "candle-cpu")]
pub mod candle_cpu_backend;
#[cfg(feature = "storage-adapters")]
pub mod storage_adapters;
pub mod storage_kernel;
pub mod streaming;
pub mod tokenizer;
pub mod toolchain_attest;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod tools;
pub mod transform_recipe;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod treatment;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod validator;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod video;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod vision;
pub mod worker_crash_ledger;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod worker_dispatch;
#[cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
pub mod worker_memory;
pub mod worker_protocol;
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
