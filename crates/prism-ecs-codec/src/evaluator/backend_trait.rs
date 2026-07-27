//! BackendEvaluator — the backend-neutral evaluation contract.
//!
//! This module owns the canonical authority for the trait each
//! backend (Metal, ANE, Accelerate, future NPU) implements to
//! participate in heterogeneous evaluation. Every backend lowers,
//! compiles, binds, dispatches, measures, and reports against the
//! same contract without changing codec identity.

use serde::{Deserialize, Serialize};

use super::fixture::EvaluationFixture;
use super::generated_executable::GeneratedExecutable;
use super::receipts::EvaluationReceiptBundle;
use super::role::EvaluationRole;

/// Configuration for a single evaluation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial backend that always refuses — proves the trait is
    /// object-safe and that a `Box<dyn BackendEvaluator>` can be
    /// constructed.
    struct RejectingBackend {
        name: String,
    }

    impl BackendEvaluator for RejectingBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn can_evaluate(&self, _executable: &GeneratedExecutable) -> bool {
            false
        }

        fn evaluate(
            &self,
            _executable: &GeneratedExecutable,
            _fixture: &EvaluationFixture,
            _role: EvaluationRole,
            _config: &EvaluationConfig,
        ) -> Result<EvaluationReceiptBundle, EvaluationError> {
            Err(EvaluationError {
                stage: "can_evaluate".to_string(),
                message: "rejected".to_string(),
                detail: "intentional".to_string(),
                is_retryable: false,
            })
        }
    }

    #[test]
    fn default_evaluation_config_is_sensible() {
        let cfg = EvaluationConfig::default();
        assert_eq!(cfg.warmup_iterations, 3);
        assert_eq!(cfg.measured_iterations, 10);
        assert_eq!(cfg.timeout_ms, 30000);
        assert_eq!(cfg.temperature_policy, TemperaturePolicy::ReportThermalState);
    }

    #[test]
    fn evaluation_error_displays_with_stage_message_detail() {
        let err = EvaluationError {
            stage: "compile".to_string(),
            message: "kernel build failed".to_string(),
            detail: "undefined symbol".to_string(),
            is_retryable: true,
        };
        let s = format!("{}", err);
        assert!(s.contains("compile"));
        assert!(s.contains("kernel build failed"));
        assert!(s.contains("undefined symbol"));
    }

    #[test]
    fn backend_evaluator_is_object_safe() {
        let backend: Box<dyn BackendEvaluator> = Box::new(RejectingBackend {
            name: "rejecting".to_string(),
        });
        assert_eq!(backend.name(), "rejecting");
    }
}
