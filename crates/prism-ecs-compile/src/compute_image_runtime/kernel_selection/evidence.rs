//! Candidate benchmark evidence — measurement results for each
//! candidate kernel variant.

use serde::{Deserialize, Serialize};

/// Per-candidate benchmark evidence for a kernel selection decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBenchmarkEvidence {
    /// Stable identifier for this candidate within the evidence bundle.
    pub candidate_id: String,
    /// Name of the operation this candidate implements.
    pub operation_name: String,
    /// Execution shape class this candidate was measured against.
    pub shape_class: String,
    /// Median latency in nanoseconds.
    pub median_latency_ns: u64,
    /// 95th percentile latency in nanoseconds.
    pub p95_latency_ns: u64,
    /// Minimum observed latency in nanoseconds.
    pub min_latency_ns: u64,
    /// Maximum observed latency in nanoseconds.
    pub max_latency_ns: u64,
    /// Number of measurement samples.
    pub sample_count: u32,
    /// Whether the candidate fits the target's resources.
    pub resource_fit_pass: bool,
    /// Whether the candidate passed numerical verification.
    pub numerical_verification_pass: bool,
    /// Environment the measurements were taken in.
    pub measurement_environment: MeasurementEnvironment,
}

/// Environment metadata for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementEnvironment {
    /// Hardware model identifier.
    pub hardware_model: String,
    /// Operating system version.
    pub os_version: String,
    /// GPU name.
    pub gpu_name: String,
    /// Driver version string.
    pub driver_version: String,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Thermal state at measurement time.
    pub thermal_state: String,
}

/// Aggregated candidate evidence for one operation/shape-class pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledCandidateEvidence {
    /// Operation name.
    pub operation: String,
    /// Execution shape class.
    pub shape_class: String,
    /// Per-candidate benchmark evidence.
    pub candidates: Vec<CandidateBenchmarkEvidence>,
    /// Identifier of the candidate that was selected.
    pub selected_candidate_id: String,
    /// Confidence level of the selection.
    pub selection_confidence: SelectionConfidence,
}

/// How confident the selection policy is in its pick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionConfidence {
    /// Strong evidence supports the selection.
    High,
    /// Reasonable evidence supports the selection.
    Medium,
    /// Marginal evidence supports the selection.
    Low,
    /// Insufficient evidence to make a confident selection.
    Insufficient,
}

impl Default for CompiledCandidateEvidence {
    fn default() -> Self {
        Self {
            operation: String::new(),
            shape_class: String::new(),
            candidates: Vec::new(),
            selected_candidate_id: String::new(),
            selection_confidence: SelectionConfidence::Insufficient,
        }
    }
}
