//! Admission decision types for heterogeneous evaluation.

use super::generated_executable::GeneratedExecutable;
use super::receipts::EvaluationReceiptBundle;
use super::role::EvaluationRole;
use serde::{Deserialize, Serialize};

/// Admission decision for a candidate executable on a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
