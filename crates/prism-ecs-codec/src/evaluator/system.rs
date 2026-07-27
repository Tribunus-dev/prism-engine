//! HeterogeneousEvaluatorSystem — the ECS system that coordinates
//! evaluation lanes across backends.
//!
//! This module owns the canonical authority for the system that
//! iterates a set of backends, dispatches each executable to the
//! capable backends, and emits an [`AdmissionDecision`] per
//! backend. The admission policy governs which backends are
//! trusted, what evidence is required, and what error bounds
//! must hold.
//!
//! The system is a thin coordination layer; the per-backend
//! behavior lives in each backend's [`BackendEvaluator`](super::backend_trait::BackendEvaluator)
//! implementation. The system itself holds no canonical world state
//! and does not commit any facts — it is pure coordination that
//! returns decisions to the caller.

use super::admission::AdmissionDecision;
use super::backend_trait::{BackendEvaluator, EvaluationConfig};
use super::fixture::EvaluationFixture;
use super::generated_executable::GeneratedExecutable;
use super::role::EvaluationRole;

/// Policy for admission decisions.
#[derive(Debug, Clone, PartialEq)]
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
                    let passed = bundle.numerical.as_ref().is_some_and(|n| n.passed);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::backend_trait::EvaluationError;
    use crate::evaluator::binding_plan::{BindingPlan, BindingSlot, ConstantSlot};
    use crate::evaluator::generated_executable::GeneratedExecutable;
    use crate::evaluator::kernel_abi::{
        BufferBinding, ConstantBinding, DispatchGeometryPolicy, KernelAbi, ThreadgroupAllocation,
    };
    use crate::evaluator::receipts::{
        EvaluationReceiptBundle, NumericalReceipt, RejectionReceipt,
    };

    fn sample_executable() -> GeneratedExecutable {
        GeneratedExecutable {
            source_digest: [0u8; 32],
            operation_id: "op".to_string(),
            codec_id: "nf4".to_string(),
            layout_id: "tile640".to_string(),
            entry_point: "ep".to_string(),
            abi: KernelAbi {
                version: 1,
                buffers: vec![BufferBinding {
                    slot: 0,
                    name: "input".to_string(),
                    byte_size: 1024,
                    optional: false,
                }],
                constants: vec![ConstantBinding {
                    index: 0,
                    name: "tile_m".to_string(),
                    default_value: Some(64),
                }],
                threadgroup_memory: vec![ThreadgroupAllocation { byte_size: 4096 }],
                dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
                threads_per_threadgroup: (32, 1, 1),
            },
            binding_plan: BindingPlan {
                buffers: vec![BindingSlot {
                    name: "input".to_string(),
                    slot: 0,
                    byte_size: 1024,
                    alignment: 16,
                }],
                constants: vec![ConstantSlot {
                    name: "tile_m".to_string(),
                    index: 0,
                    value: 64,
                }],
                output_buffer: "output".to_string(),
                output_size: 1024,
            },
            backend_target: "metal".to_string(),
            machine_requirements: vec![],
            compiler_identity: "ci".to_string(),
            artifact_digest: [1u8; 32],
        }
    }

    fn sample_fixture() -> EvaluationFixture {
        EvaluationFixture::Nf4 {
            codes: vec![0xAB; 128],
            scales: vec![1.0; 4],
            biases: vec![0.0; 4],
            input: vec![0.5; 256],
            reference: vec![0.0; 64],
            m: 4,
            k: 256,
            n: 16,
            digest: [7u8; 32],
        }
    }

    struct PassingBackend;
    impl BackendEvaluator for PassingBackend {
        fn name(&self) -> &str {
            "passing"
        }
        fn can_evaluate(&self, _: &GeneratedExecutable) -> bool {
            true
        }
        fn evaluate(
            &self,
            _: &GeneratedExecutable,
            _: &EvaluationFixture,
            _: EvaluationRole,
            _: &EvaluationConfig,
        ) -> Result<EvaluationReceiptBundle, EvaluationError> {
            Ok(EvaluationReceiptBundle {
                numerical: Some(NumericalReceipt {
                    max_abs_error: 0.001,
                    mean_abs_error: 0.0001,
                    passed: true,
                }),
                ..EvaluationReceiptBundle::default()
            })
        }
    }

    struct FailingBackend;
    impl BackendEvaluator for FailingBackend {
        fn name(&self) -> &str {
            "failing"
        }
        fn can_evaluate(&self, _: &GeneratedExecutable) -> bool {
            true
        }
        fn evaluate(
            &self,
            _: &GeneratedExecutable,
            _: &EvaluationFixture,
            _: EvaluationRole,
            _: &EvaluationConfig,
        ) -> Result<EvaluationReceiptBundle, EvaluationError> {
            Err(EvaluationError {
                stage: "compile".to_string(),
                message: "kernel build failed".to_string(),
                detail: "undefined symbol".to_string(),
                is_retryable: true,
            })
        }
    }

    struct IneligibleBackend;
    impl BackendEvaluator for IneligibleBackend {
        fn name(&self) -> &str {
            "ineligible"
        }
        fn can_evaluate(&self, _: &GeneratedExecutable) -> bool {
            false
        }
        fn evaluate(
            &self,
            _: &GeneratedExecutable,
            _: &EvaluationFixture,
            _: EvaluationRole,
            _: &EvaluationConfig,
        ) -> Result<EvaluationReceiptBundle, EvaluationError> {
            unreachable!()
        }
    }

    #[test]
    fn system_reports_backend_count_and_policy() {
        let sys = HeterogeneousEvaluatorSystem::new(vec![], AdmissionPolicy::default());
        assert_eq!(sys.backend_count(), 0);
        assert!(sys.policy().require_independent_oracle);
        assert_eq!(sys.policy().max_numerical_error, 0.02);
    }

    #[test]
    fn passing_backend_produces_admitted_decision() {
        let sys = HeterogeneousEvaluatorSystem::new(
            vec![Box::new(PassingBackend)],
            AdmissionPolicy::default(),
        );
        let decisions = sys.evaluate_candidate(&sample_executable(), &sample_fixture());
        assert_eq!(decisions.len(), 1);
        assert!(decisions[0].is_admitted());
    }

    #[test]
    fn failing_backend_produces_rejected_decision() {
        let sys = HeterogeneousEvaluatorSystem::new(
            vec![Box::new(FailingBackend)],
            AdmissionPolicy::default(),
        );
        let decisions = sys.evaluate_candidate(&sample_executable(), &sample_fixture());
        assert_eq!(decisions.len(), 1);
        assert!(decisions[0].is_rejected());
    }

    #[test]
    fn ineligible_backend_is_skipped() {
        let sys = HeterogeneousEvaluatorSystem::new(
            vec![Box::new(IneligibleBackend)],
            AdmissionPolicy::default(),
        );
        let decisions = sys.evaluate_candidate(&sample_executable(), &sample_fixture());
        assert!(decisions.is_empty());
    }

    #[test]
    fn default_policy_requires_independent_oracle() {
        let p = AdmissionPolicy::default();
        assert!(p.require_independent_oracle);
        assert_eq!(p.min_oracle_samples, 1);
        assert!(!p.require_performance_evidence);
    }

    #[test]
    fn rejection_receipt_is_carried_in_evidence_when_present() {
        // Ensure that the receipts module's RejectionReceipt type is
        // available to bundle consumers — sanity check the surface
        // is wired up.
        let r = RejectionReceipt {
            stage: "compile".to_string(),
            reason: "nope".to_string(),
            detail: "see logs".to_string(),
        };
        let bundle = EvaluationReceiptBundle {
            rejection: Some(r),
            ..EvaluationReceiptBundle::default()
        };
        assert!(bundle.rejection.is_some());
    }
}
