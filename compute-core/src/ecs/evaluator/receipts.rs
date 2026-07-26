//! Evaluation receipt types — complete evidence from one evaluation run.

use serde::{Deserialize, Serialize};

/// Complete evidence bundle from one evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReceiptBundle {
    /// Static validation receipt.
    pub static_validation: Option<StaticValidationReceipt>,
    /// Compilation receipt.
    pub compilation: Option<CompilationReceipt>,
    /// Codec correctness receipt.
    pub codec: Option<CodecReceipt>,
    /// Numerical comparison receipt.
    pub numerical: Option<NumericalReceipt>,
    /// Performance measurement receipt.
    pub performance: Option<PerformanceReceipt>,
    /// Repeatability receipt.
    pub repeatability: Option<RepeatabilityReceipt>,
    /// Provenance chain.
    pub provenance: Option<ProvenanceReceipt>,
    /// Rejection evidence (if evaluation failed).
    pub rejection: Option<RejectionReceipt>,
}

/// Static validation receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticValidationReceipt {
    pub passed: bool,
    pub messages: Vec<String>,
}

/// Compilation receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationReceipt {
    pub compile_duration_ms: u64,
    pub artifact_digest: [u8; 32],
    pub success: bool,
}

/// Codec correctness receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecReceipt {
    pub packed_digest: [u8; 32],
    pub decode_correct: bool,
    pub dimensions_match: bool,
}

/// Numerical comparison receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalReceipt {
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub passed: bool,
}

/// Performance measurement receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReceipt {
    pub wall_time_ns: u64,
    pub sample_count: usize,
    pub median_ns: u64,
}

/// Repeatability receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatabilityReceipt {
    pub std_dev_pct: f64,
    pub passed: bool,
}

/// Provenance receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceReceipt {
    pub candidate_id: String,
    pub machine_id: String,
    pub timestamp: String,
}

/// Rejection receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionReceipt {
    pub stage: String,
    pub reason: String,
    pub detail: String,
}
