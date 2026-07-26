//! Engine systems — engine singleton initialization, generation
//! requests, model install/load/unload, cancel, metrics, and shutdown.
//!
//! This module owns the canonical authority for the engine-level
//! lifecycle. The engine singleton is the canonical "host process"
//! entity; systems on it coordinate model install, model load,
//! request lifecycle, and process-level metrics.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The actual `ModelStore` disk I/O (owned by the engine; the
//!   runtime only triggers it via typed commands).
//! - The cimage parsing/loading (owned by the cimage subsystem).
//! - The kernel dispatch (owned by the kernel layer).
//!
//! All exposed types are pure value types. The module never mutates
//! the world directly; the schedule stages all state changes through
//! a `WorldTxn`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Engine singleton components
// ---------------------------------------------------------------------------

/// Engine serial number — increments on every engine initialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineState {
    pub serial_number: u64,
    pub engine_error: Option<String>,
    pub shutdown: bool,
    pub resource_summary: String,
}

impl prism_ecs_core::Component for EngineState {}

/// Engine-level metrics (request count, throughput, peak memory).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineMetrics {
    pub request_count: u64,
    pub avg_tokens_per_second: f64,
    pub peak_memory_bytes: u64,
}

impl prism_ecs_core::Component for EngineMetrics {}

/// Model install state — list of installed models in the local store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInstallState {
    pub installed_models: Vec<String>,
}

impl prism_ecs_core::Component for ModelInstallState {}

/// Memory pressure level for the host process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PressureLevel {
    None,
    Moderate,
    High,
    Critical,
}

/// Current memory pressure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPressure {
    pub level: PressureLevel,
    pub active_bytes: u64,
    pub limit_bytes: u64,
}

impl prism_ecs_core::Component for MemoryPressure {}

/// One in-flight decode tracking for a generation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InFlightDecode {
    pub token_count: u32,
    pub kv_block_index: u32,
    pub eos: bool,
}

impl prism_ecs_core::Component for InFlightDecode {}

/// Generation request — submitted by a user, processed by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub max_tokens: u32,
    #[serde(skip)]
    pub response_tx: Option<std::sync::mpsc::Sender<GenerationEvent>>,
}

impl PartialEq for GenerationRequest {
    fn eq(&self, other: &Self) -> bool {
        self.max_tokens == other.max_tokens
    }
}

impl prism_ecs_core::Component for GenerationRequest {}

/// Events emitted by the engine on a generation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationEvent {
    Started,
    Token(String),
    Finished,
    Failed(String),
}

// ---------------------------------------------------------------------------
// Model install / load
// ---------------------------------------------------------------------------

/// Result of a model install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModel {
    pub image_hash: String,
    pub source_identity: String,
    pub installed_at: String,
}

/// A request to install a model into the local store.
#[derive(Debug)]
pub struct ModelInstallRequest {
    pub source_dir: String,
    pub image_hash: String,
    pub source_identity: String,
    pub compiler_version: String,
    pub result_tx: Option<std::sync::mpsc::Sender<Result<InstalledModel, String>>>,
}

impl prism_ecs_core::Component for ModelInstallRequest {}

/// A request to load an installed model by its image hash.
#[derive(Debug)]
pub struct ModelLoadRequest {
    pub image_hash: String,
    pub result_tx: Option<std::sync::mpsc::Sender<Result<(), String>>>,
}

impl prism_ecs_core::Component for ModelLoadRequest {}

/// A request to load a cimage artifact.
#[derive(Debug)]
pub struct CimageLoadRequest {
    pub cimage_bytes: Vec<u8>,
    pub result_tx: Option<std::sync::mpsc::Sender<Result<(), String>>>,
}

impl prism_ecs_core::Component for CimageLoadRequest {}

// ---------------------------------------------------------------------------
// Cancel
// ---------------------------------------------------------------------------

/// A request to cancel an in-flight generation.
#[derive(Debug)]
pub struct CancelRequest {
    pub job_id: String,
    pub result_tx: Option<std::sync::mpsc::Sender<Result<(), String>>>,
}

impl prism_ecs_core::Component for CancelRequest {}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineSystemError {
    #[error("engine singleton entity not found")]
    EngineEntityMissing,
    #[error("model install request is missing a result channel")]
    MissingResultChannel,
    #[error("cimage artifact is too small: got {got} bytes, need at least {min}")]
    CimageTooSmall { got: usize, min: usize },
    #[error("cimage magic mismatch: expected `PRISMCIM`, got `{got:?}`")]
    CimageBadMagic { got: Vec<u8> },
}

// ---------------------------------------------------------------------------
// Helpers — pure functions used by the schedule
// ---------------------------------------------------------------------------

/// The canonical magic bytes for a cimage header.
pub const CIMAGE_MAGIC: [u8; 8] = *b"PRISMCIM";

/// Validate a cimage header without mutating state. Returns
/// `Ok(())` if the bytes look like a cimage, or a typed error if
/// not.
pub fn validate_cimage_header(bytes: &[u8]) -> Result<(), EngineSystemError> {
    if bytes.len() < 8 {
        return Err(EngineSystemError::CimageTooSmall { got: bytes.len(), min: 8 });
    }
    if bytes[..8] != CIMAGE_MAGIC {
        return Err(EngineSystemError::CimageBadMagic {
            got: bytes[..8].to_vec(),
        });
    }
    Ok(())
}

/// Build an initial `EngineState` for a freshly-spawned engine
/// singleton. The serial number starts at 1; shutdown is false; the
/// resource summary is "initialised".
pub fn initial_engine_state() -> EngineState {
    EngineState {
        serial_number: 1,
        engine_error: None,
        shutdown: false,
        resource_summary: "initialised".into(),
    }
}

/// Build a fresh `EngineMetrics` with all counters at zero.
pub fn initial_engine_metrics() -> EngineMetrics {
    EngineMetrics {
        request_count: 0,
        avg_tokens_per_second: 0.0,
        peak_memory_bytes: 0,
    }
}

/// Build an initial `ModelInstallState` from the list of installed
/// models. Empty by default; the engine populates this from the
/// `ModelStore::list` call at startup.
pub fn initial_model_install_state(installed: Vec<String>) -> ModelInstallState {
    ModelInstallState {
        installed_models: installed,
    }
}

/// Build a fresh `MemoryPressure` with no pressure observed.
pub fn initial_memory_pressure() -> MemoryPressure {
    MemoryPressure {
        level: PressureLevel::None,
        active_bytes: 0,
        limit_bytes: 0,
    }
}

/// Classify memory pressure from a usage/limit ratio.
pub fn classify_memory_pressure(ratio: f64) -> PressureLevel {
    if ratio > 0.95 {
        PressureLevel::Critical
    } else if ratio > 0.85 {
        PressureLevel::High
    } else if ratio > 0.70 {
        PressureLevel::Moderate
    } else {
        PressureLevel::None
    }
}

/// Apply a generation-request side-effect: increment the engine
/// metrics' request count.
pub fn increment_request_count(metrics: &mut EngineMetrics) {
    metrics.request_count = metrics.request_count.saturating_add(1);
}

/// Update `EngineMetrics.peak_memory_bytes` if the new active-bytes
/// count exceeds the previous peak.
pub fn update_peak_memory(metrics: &mut EngineMetrics, active_bytes: u64) {
    if active_bytes > metrics.peak_memory_bytes {
        metrics.peak_memory_bytes = active_bytes;
    }
}

/// Mark the engine as shutting down.
pub fn mark_shutting_down(state: &mut EngineState) {
    state.shutdown = true;
    state.resource_summary = "shutdown".into();
}

/// Update the engine's memory-pressure component from a fresh
/// `active` / `limit` reading.
pub fn update_memory_pressure(
    pressure: &mut MemoryPressure,
    active_bytes: u64,
    limit_bytes: u64,
) {
    pressure.active_bytes = active_bytes;
    pressure.limit_bytes = limit_bytes;
    let ratio = if limit_bytes > 0 {
        active_bytes as f64 / limit_bytes as f64
    } else {
        0.0
    };
    pressure.level = classify_memory_pressure(ratio);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_engine_state_has_serial_1() {
        let s = initial_engine_state();
        assert_eq!(s.serial_number, 1);
        assert!(!s.shutdown);
        assert_eq!(s.resource_summary, "initialised");
    }

    #[test]
    fn initial_engine_metrics_is_zero() {
        let m = initial_engine_metrics();
        assert_eq!(m.request_count, 0);
        assert_eq!(m.avg_tokens_per_second, 0.0);
        assert_eq!(m.peak_memory_bytes, 0);
    }

    #[test]
    fn initial_memory_pressure_is_none() {
        let p = initial_memory_pressure();
        assert_eq!(p.level, PressureLevel::None);
        assert_eq!(p.active_bytes, 0);
        assert_eq!(p.limit_bytes, 0);
    }

    #[test]
    fn classify_memory_pressure_thresholds() {
        assert_eq!(classify_memory_pressure(0.50), PressureLevel::None);
        assert_eq!(classify_memory_pressure(0.80), PressureLevel::Moderate);
        assert_eq!(classify_memory_pressure(0.90), PressureLevel::High);
        assert_eq!(classify_memory_pressure(0.99), PressureLevel::Critical);
    }

    #[test]
    fn classify_memory_pressure_handles_zero_limit() {
        assert_eq!(classify_memory_pressure(0.0), PressureLevel::None);
    }

    #[test]
    fn increment_request_count_saturates() {
        let mut m = EngineMetrics {
            request_count: u64::MAX,
            avg_tokens_per_second: 0.0,
            peak_memory_bytes: 0,
        };
        increment_request_count(&mut m);
        assert_eq!(m.request_count, u64::MAX);
    }

    #[test]
    fn update_peak_memory_only_increases() {
        let mut m = initial_engine_metrics();
        update_peak_memory(&mut m, 100);
        assert_eq!(m.peak_memory_bytes, 100);
        update_peak_memory(&mut m, 50);
        assert_eq!(m.peak_memory_bytes, 100);
        update_peak_memory(&mut m, 200);
        assert_eq!(m.peak_memory_bytes, 200);
    }

    #[test]
    fn mark_shutting_down_sets_flags() {
        let mut s = initial_engine_state();
        mark_shutting_down(&mut s);
        assert!(s.shutdown);
        assert_eq!(s.resource_summary, "shutdown");
    }

    #[test]
    fn update_memory_pressure_classifies_correctly() {
        let mut p = initial_memory_pressure();
        update_memory_pressure(&mut p, 9_000, 10_000);
        assert_eq!(p.level, PressureLevel::High);
        assert_eq!(p.active_bytes, 9_000);
        update_memory_pressure(&mut p, 9_900, 10_000);
        assert_eq!(p.level, PressureLevel::Critical);
        update_memory_pressure(&mut p, 7_500, 10_000);
        assert_eq!(p.level, PressureLevel::Moderate);
        update_memory_pressure(&mut p, 3_000, 10_000);
        assert_eq!(p.level, PressureLevel::None);
    }

    #[test]
    fn validate_cimage_header_accepts_magic() {
        let mut bytes = vec![0u8; 32];
        bytes[..8].copy_from_slice(&CIMAGE_MAGIC);
        assert!(validate_cimage_header(&bytes).is_ok());
    }

    #[test]
    fn validate_cimage_header_rejects_too_small() {
        let bytes = vec![0u8; 4];
        let err = validate_cimage_header(&bytes).unwrap_err();
        assert!(matches!(err, EngineSystemError::CimageTooSmall { .. }));
    }

    #[test]
    fn validate_cimage_header_rejects_bad_magic() {
        let mut bytes = vec![0u8; 16];
        bytes[..8].copy_from_slice(b"OTHERMGC");
        let err = validate_cimage_header(&bytes).unwrap_err();
        assert!(matches!(err, EngineSystemError::CimageBadMagic { .. }));
    }

    #[test]
    fn initial_model_install_state_carries_list() {
        let s = initial_model_install_state(vec!["m1".into(), "m2".into()]);
        assert_eq!(s.installed_models.len(), 2);
        assert_eq!(s.installed_models[0], "m1");
    }

    #[test]
    fn generation_request_carries_max_tokens() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let r = GenerationRequest {
            max_tokens: 256,
            response_tx: Some(tx),
        };
        assert_eq!(r.max_tokens, 256);
    }

    #[test]
    fn engine_state_serializes_round_trip() {
        let s = EngineState {
            serial_number: 42,
            engine_error: Some("oops".into()),
            shutdown: true,
            resource_summary: "shutdown".into(),
        };
        let json = serde_json::to_string(&s).expect("serialize");
        let back: EngineState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
    }
}
