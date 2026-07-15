//! BackendEvaluator trait — backend-neutral evaluation contract.

use super::fixture::EvaluationFixture;
use super::generated_executable::GeneratedExecutable;
use super::receipts::EvaluationReceiptBundle;
use super::role::EvaluationRole;
use serde::{Deserialize, Serialize};

/// Configuration for a single evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub timeout_ms: u64,
    pub temperature_policy: TemperaturePolicy,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measured_iterations: 10,
            timeout_ms: 30000,
            temperature_policy: TemperaturePolicy::ReportThermalState,
        }
    }
}

/// Thermal throttling awareness policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemperaturePolicy {
    AllowThrottling,
    CooldownOnly,
    ReportThermalState,
}

/// Evaluation error with structured detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationError {
    pub stage: String,
    pub message: String,
    pub detail: String,
    pub is_retryable: bool,
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} — {}", self.stage, self.message, self.detail)
    }
}

impl std::error::Error for EvaluationError {}

/// Backend-neutral evaluator trait.
///
/// Every backend (Metal, ANE, Accelerate, future NPU) implements this.
/// Validates, lowers, compiles, binds, dispatches, measures, and reports
/// without changing codec identity.
pub trait BackendEvaluator: Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Check if this backend can evaluate the given executable.
    fn can_evaluate(&self, executable: &GeneratedExecutable) -> bool;

    /// Lower, compile, bind, dispatch, measure, and report.
    fn evaluate(
        &self,
        executable: &GeneratedExecutable,
        fixture: &EvaluationFixture,
        role: EvaluationRole,
        config: &EvaluationConfig,
    ) -> Result<EvaluationReceiptBundle, EvaluationError>;
}
