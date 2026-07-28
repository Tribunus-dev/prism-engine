//! `prism_ecs_compile::decode_attribution::backend_adapters` —
//! backend-agnostic adapter contracts and conformance metrics.
//!
//! This module owns the canonical authority for backend
//! classification and conformance scoring that is not coupled to
//! any specific executor. The engine-internal adapter
//! implementations (Core ML, MLX, Accelerate, Reference) live at
//! `compute-core/src/ecs/legacy_decode_attribution/backend_adapters/`
//! because they depend on engine FFI bridges and per-backend
//! executor stacks.

pub mod conformance;

use std::fmt;

/// Backend identifiers participating in the decode-attribution gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Apple Core ML (via the engine's `coreai_bridge`).
    CoreAi,
    /// Apple Accelerate framework (BLAS / vDSP / vForce).
    Accelerate,
    /// Apple MLX.
    Mlx,
    /// Generic pure-Rust reference evaluator.
    Reference,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendKind::CoreAi => write!(f, "coreai"),
            BackendKind::Accelerate => write!(f, "accelerate"),
            BackendKind::Mlx => write!(f, "mlx"),
            BackendKind::Reference => write!(f, "reference"),
        }
    }
}

/// Static capability tier: what a backend can represent in principle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendSupportTier {
    /// Backend has a direct kernel for this graph family.
    SupportedNative,
    /// Built from supported primitives (e.g., MLX chain = matmul+add+silu).
    SupportedComposed,
    /// Cannot express this graph in the backend's execution model.
    UnsupportedGraph,
    /// Backend has the primitives but no adapter yet.
    NotImplemented,
}

impl fmt::Display for BackendSupportTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendSupportTier::SupportedNative => write!(f, "supported_native"),
            BackendSupportTier::SupportedComposed => write!(f, "supported_composed"),
            BackendSupportTier::UnsupportedGraph => write!(f, "unsupported_graph"),
            BackendSupportTier::NotImplemented => write!(f, "not_implemented"),
        }
    }
}

/// Phase-specific failure classification for observed runtime outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredictFailureClass {
    /// MIL program generation failed (package write, protobuf build).
    MaterializeLimited,
    /// Core ML compiler (coremlcompiler) failed.
    CompileLimited,
    /// Load of compiled model failed.
    LoadBlocked,
    /// Prediction bridge failed.
    PredictBlocked,
    /// Output diverged from reference beyond tolerance.
    NumericalDivergence,
    /// Execution exceeded timeout.
    Timeout,
    /// Out-of-memory during execution.
    MemoryOom,
}

impl fmt::Display for PredictFailureClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictFailureClass::MaterializeLimited => write!(f, "materialize_limited"),
            PredictFailureClass::CompileLimited => write!(f, "compile_limited"),
            PredictFailureClass::LoadBlocked => write!(f, "load_blocked"),
            PredictFailureClass::PredictBlocked => write!(f, "predict_blocked"),
            PredictFailureClass::NumericalDivergence => write!(f, "numerical_divergence"),
            PredictFailureClass::Timeout => write!(f, "timeout"),
            PredictFailureClass::MemoryOom => write!(f, "memory_oom"),
        }
    }
}

/// Backend support status for a given graph family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSupportStatus {
    /// Backend can run this graph family natively or composed.
    Supported,
    /// Backend cannot express this graph in its execution model.
    UnsupportedGraph,
    /// Backend has the primitives but no adapter yet.
    NotImplemented,
    /// Status is not applicable for this backend / graph combination.
    NotApplicable,
    /// Adapter error during classification.
    Error,
}

impl fmt::Display for BackendSupportStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendSupportStatus::Supported => write!(f, "supported"),
            BackendSupportStatus::UnsupportedGraph => write!(f, "unsupported_graph"),
            BackendSupportStatus::NotImplemented => write!(f, "not_implemented"),
            BackendSupportStatus::NotApplicable => write!(f, "not_applicable"),
            BackendSupportStatus::Error => write!(f, "error"),
        }
    }
}

/// A single timing result from a backend execution.
#[derive(Debug, Clone)]
pub struct BackendTiming {
    /// Elapsed wall-clock duration in nanoseconds.
    pub duration_ns: u64,
    /// Optional SHA-256 hash of the output for determinism verification.
    pub output_hash: Option<String>,
}
