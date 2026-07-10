//! Engine-level ECS component types — decomposed state from the old
//! `ComputeEngine` struct, now stored as components on a singleton entity.
//!
//! These components are created and consumed by the engine systems in
//! `ecs/system/engine_systems.rs`.

use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EngineState — singleton component tracking overall engine lifecycle
// ---------------------------------------------------------------------------

/// Global engine lifecycle state.
///
/// Attached to the engine singleton entity by `EngineInitSystem`.
/// Updated by lifecycle systems (shutdown, error transitions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    /// Monotonically increasing serial number for ordering or telemetry.
    pub serial_number: u64,
    /// Pending engine-level error message, if any.
    pub engine_error: Option<String>,
    /// If `true`, the engine has been asked to shut down.
    pub shutdown: bool,
    /// Human-readable summary of resource state (model loaded, memory, etc.).
    pub resource_summary: String,
}
impl Component for EngineState {}

// ---------------------------------------------------------------------------
// GenerationRequest — per-request generation parameters
// ---------------------------------------------------------------------------

/// A single generation request, spawned as a component on a request entity.
///
/// Created by `GenerationRequestSystem` (Phase: Execution) from parsed
/// stdin/args, or injected directly by the host.  Consumed by downstream
/// inference systems to produce tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Input text prompt or token description.
    pub prompt: String,
    /// Opaque session identifier for this generation run.
    pub session_id: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Temperature for softmax scaling (0.0 = greedy).
    pub temperature: f64,
    /// Top-k filter width.
    pub top_k: u32,
    /// Top-p nucleus threshold.
    pub top_p: f64,
    /// Optional PRNG seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Token ID sequences at which generation should stop.
    pub stop_sequences: Vec<String>,
    /// Response channel for streaming generation events.
    /// Skipped during serialization (not serializable).
    #[serde(skip)]
    pub response_tx: Option<std::sync::mpsc::Sender<crate::ecs::streaming::GenerationEvent>>,
}
impl Component for GenerationRequest {}

// ---------------------------------------------------------------------------
// InFlightDecode — tracks the progress of an active decode step
// ---------------------------------------------------------------------------

/// Per-request decode progress, updated each inference cycle.
///
/// Attached to an inference entity during the decode loop.
/// Cleared when the request completes or is cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InFlightDecode {
    /// Number of tokens generated so far (including the current decode step).
    pub token_count: u32,
    /// KV-cache block index assigned to this decode slot.
    pub kv_block_index: u64,
    /// Whether an end-of-sequence token has been emitted.
    pub eos: bool,
}
impl Component for InFlightDecode {}

// ---------------------------------------------------------------------------
// EngineMetrics — aggregated performance and resource usage
// ---------------------------------------------------------------------------

/// Performance and resource-usage counters for the engine.
///
/// Updated periodically by `EngineMetricsSystem` (Phase: Packaging) or
/// on demand.  Stored on the engine singleton entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Total number of generation requests handled since engine start.
    pub request_count: u64,
    /// Running average tokens generated per second across all requests.
    pub avg_tokens_per_second: f64,
    /// Peak memory usage observed (bytes).
    pub peak_memory_bytes: u64,
}
impl Component for EngineMetrics {}

// ---------------------------------------------------------------------------
// ModelInstallState — record of installed models in the persistent store
// ---------------------------------------------------------------------------

/// Records every model currently installed in the persistent store.
///
/// Refreshed by `ModelInstallSystem` (Phase: ModelLoading) after each
/// install operation.  Stored on the engine singleton entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstallState {
    /// List of installed model metadata records.
    pub installed_models: Vec<crate::ecs::core::model_store::InstalledModel>,
}
impl Component for ModelInstallState {}

// ---------------------------------------------------------------------------
// MemoryPressure — current system memory pressure level
// ---------------------------------------------------------------------------

/// Memory-pressure severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PressureLevel {
    /// Normal operation — no memory concerns.
    None,
    /// Elevated usage — consider proactive measures.
    Moderate,
    /// Significant pressure — reduce batch sizes or clear caches.
    High,
    /// Critical — halt generation to avoid OOM.
    Critical,
}

/// Current memory-pressure state of the engine.
///
/// Updated by `MemoryPressureSystem` (Phase: Execution) each cycle.
/// Read by scheduler and admission systems to throttle work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressure {
    /// Computed pressure level.
    pub level: PressureLevel,
    /// Currently active memory (bytes).
    pub active_bytes: u64,
    /// Configured memory limit (bytes); zero indicates no limit.
    pub limit_bytes: u64,
}
impl Component for MemoryPressure {}

// ---------------------------------------------------------------------------
// HostInferenceHandle — bookend handle for host-side inference
// ---------------------------------------------------------------------------

/// Handle tracking a registered host-side inference pipeline.
///
/// Created by `HostInferenceInitSystem` (Phase: ModelLoading) after
/// the scheduler and hybrid executor have been initialised.  Acts as a
/// bookend token: the pipeline runs while this handle exists and is
/// torn down when the handle is removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInferenceHandle {
    /// Unique identifier for this inference handle.
    pub handle_id: String,
}
impl Component for HostInferenceHandle {}
