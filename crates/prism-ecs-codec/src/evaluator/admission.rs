//! AdmissionDecision — the typed outcome of an admission gate.
//!
//! This module owns the canonical authority for the decision an
//! admission gate emits after consuming an evaluation receipt
//! bundle. The decision is one of three shapes: admitted with
//! evidence, rejected with typed evidence, or deferred with a list
//! of missing evaluation roles.

use serde::{Deserialize, Serialize};

use super::generated_executable::GeneratedExecutable;
use super::receipts::EvaluationReceiptBundle;
use super::role::EvaluationRole;

/// Admission decision for a candidate executable on a backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AdmissionDecision {
    /// Accept this representation for the given executable and backend.
    Admitted {
        /// The admitted executable.
        executable: GeneratedExecutable,
        /// Supporting evidence.
        evidence: Vec<EvaluationReceiptBundle>,
        /// Confidence score (0.0–1.0).
        confidence: f64,
    },
    /// Reject with typed evidence.
    Rejected {
        /// Human-readable reason.
        reason: String,
        /// Supporting evidence.
        evidence: Vec<EvaluationReceiptBundle>,
    },
    /// Defer — needs more evaluation.
    Deferred {
        /// Missing evaluation roles that would be required for admission.
        missing_roles: Vec<EvaluationRole>,
        /// Additional detail.
        detail: String,
    },
}

impl AdmissionDecision {
    /// Returns true if this decision admits the candidate.
    pub fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted { .. })
    }

    /// Returns true if this decision rejects the candidate.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }

    /// Returns true if this decision defers for more evaluation.
    pub fn is_deferred(&self) -> bool {
        matches!(self, Self::Deferred { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::binding_plan::{BindingPlan, BindingSlot, ConstantSlot};
    use crate::evaluator::kernel_abi::{
        BufferBinding, ConstantBinding, DispatchGeometryPolicy, KernelAbi, ThreadgroupAllocation,
    };
    use crate::evaluator::receipts::EvaluationReceiptBundle;

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

    #[test]
    fn admitted_decision_is_admitted() {
        let d = AdmissionDecision::Admitted {
            executable: sample_executable(),
            evidence: vec![EvaluationReceiptBundle::default()],
            confidence: 0.95,
        };
        assert!(d.is_admitted());
        assert!(!d.is_rejected());
        assert!(!d.is_deferred());
    }

    #[test]
    fn rejected_decision_is_rejected() {
        let d = AdmissionDecision::Rejected {
            reason: "numerical error too high".to_string(),
            evidence: vec![],
        };
        assert!(!d.is_admitted());
        assert!(d.is_rejected());
        assert!(!d.is_deferred());
    }

    #[test]
    fn deferred_decision_is_deferred() {
        let d = AdmissionDecision::Deferred {
            missing_roles: vec![EvaluationRole::Oracle],
            detail: "needs oracle run".to_string(),
        };
        assert!(!d.is_admitted());
        assert!(!d.is_rejected());
        assert!(d.is_deferred());
    }

    #[test]
    fn decision_serializes() {
        let d = AdmissionDecision::Rejected {
            reason: "x".to_string(),
            evidence: vec![],
        };
        let json = serde_json::to_string(&d).expect("serialize");
        let restored: AdmissionDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, d);
    }
}
