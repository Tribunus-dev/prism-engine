//! CandidateEvaluator trait and supporting types.
//!
//! Defines the evaluation pipeline for evolutionary search candidates:
//! static validation → compilation → numerical validation → performance measurement.
//! Each stage produces a typed receipt.

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::identity::*;
use crate::ecs::evolution::foundation::EvolutionCandidate;

// ── Supporting types ─────────────────────────────────────────────────────────

/// Performance workload description.
#[derive(Debug, Clone)]
pub struct Workload {
    pub tensor_id: LogicalTensorId,
    pub shape: Vec<usize>,
    pub repetitions: usize,
}

/// A compiled candidate ready for numerical validation and measurement.
#[derive(Debug, Clone)]
pub struct CompiledCandidate {
    pub candidate_id: CandidateId,
    pub compiled_bytes: Vec<u8>,
    pub compile_duration_ms: u64,
}

/// Static validation receipt — validates ABI, device limits, constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticValidationReceipt {
    pub candidate_id: CandidateId,
    pub passed: bool,
    pub violations: Vec<String>,
    pub validated_at: String,
}

/// Numerical validation receipt — compares candidate output to reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalReceipt {
    pub candidate_id: CandidateId,
    pub passed: bool,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub threshold: f64,
}

/// Performance measurement receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReceipt {
    pub candidate_id: CandidateId,
    pub latency_p50_ns: u64,
    pub latency_p95_ns: u64,
    pub encode_time_ns: u64,
    pub sync_time_ns: u64,
    pub memory_traffic_bytes: u64,
    pub energy_uj: Option<u64>,
    pub repetitions: usize,
}

// ── Trait ────────────────────────────────────────────────────────────────────

/// Evaluator for search candidates — compiles, validates, measures.
#[allow(unused_variables)]
pub trait CandidateEvaluator {
    /// Validate a candidate against static constraints (ABI, device limits).
    fn validate_static(
        &self,
        candidate: &EvolutionCandidate,
    ) -> Result<StaticValidationReceipt, String>;

    /// Compile a validated candidate into runnable form.
    fn compile(&self, candidate: &EvolutionCandidate) -> Result<CompiledCandidate, String>;

    /// Validate numerical correctness against a CPU reference.
    fn validate_numerical(&self, candidate: &CompiledCandidate)
        -> Result<NumericalReceipt, String>;

    /// Measure performance on a target workload.
    fn measure(
        &self,
        candidate: &CompiledCandidate,
        workload: &Workload,
    ) -> Result<PerformanceReceipt, String>;
}
