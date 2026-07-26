//! HeterogeneousEvaluatorSystem skeleton — ECS system coordinating evaluation lanes.
//!
//! A later phase will implement full ECS system integration with transactional
//! commit of evaluation results.

use super::admission::AdmissionDecision;
use super::backend_trait::{BackendEvaluator, EvaluationConfig};
use super::fixture::EvaluationFixture;
use super::generated_executable::GeneratedExecutable;
use super::role::EvaluationRole;

/// Policy for admission decisions.
#[derive(Debug, Clone)]
pub struct AdmissionPolicy {
    pub require_independent_oracle: bool,
    pub min_oracle_samples: usize,
    pub max_numerical_error: f64,
    pub require_performance_evidence: bool,
}

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            require_independent_oracle: true,
            min_oracle_samples: 1,
            max_numerical_error: 0.02,
            require_performance_evidence: false,
        }
    }
}

/// ECS system coordinating evaluation lanes across heterogeneous backends.
///
/// Validates, lowers, compiles, binds, dispatches, measures, and commits
/// evaluation evidence transactionally.
pub struct HeterogeneousEvaluatorSystem {
    backends: Vec<Box<dyn BackendEvaluator>>,
    policy: AdmissionPolicy,
}

impl HeterogeneousEvaluatorSystem {
    pub fn new(backends: Vec<Box<dyn BackendEvaluator>>, policy: AdmissionPolicy) -> Self {
        Self { backends, policy }
    }

    /// Evaluate a candidate executable across all capable backends.
    ///
    /// Returns admission decisions per backend. Each decision carries its
    /// own evidence chain.
    pub fn evaluate_candidate(
        &self,
        executable: &GeneratedExecutable,
        fixture: &EvaluationFixture,
    ) -> Vec<AdmissionDecision> {
        let mut decisions = Vec::new();

        for backend in &self.backends {
            if !backend.can_evaluate(executable) {
                continue;
            }

            let config = EvaluationConfig::default();
            let result = backend.evaluate(executable, fixture, EvaluationRole::Candidate, &config);

            match result {
                Ok(bundle) => {
                    let passed = bundle.numerical.as_ref().map(|n| n.passed).unwrap_or(false);

                    if passed {
                        decisions.push(AdmissionDecision::Admitted {
                            executable: executable.clone(),
                            evidence: vec![bundle],
                            confidence: 1.0,
                        });
                    } else {
                        decisions.push(AdmissionDecision::Rejected {
                            reason: format!("numerical validation failed on {}", backend.name()),
                            evidence: vec![bundle],
                        });
                    }
                }
                Err(err) => {
                    decisions.push(AdmissionDecision::Rejected {
                        reason: format!("{} evaluation error: {}", backend.name(), err.message),
                        evidence: vec![],
                    });
                }
            }
        }

        decisions
    }

    pub fn policy(&self) -> &AdmissionPolicy {
        &self.policy
    }

    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }
}
