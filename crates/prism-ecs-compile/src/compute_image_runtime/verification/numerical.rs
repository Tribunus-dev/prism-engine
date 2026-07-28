//! Numerical verification receipt — the canonical attestation that a
//! compiled executable's numerical behavior is within tolerance.

use serde::{Deserialize, Serialize};

/// Numerical verification receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalVerificationReceipt {
    /// Identity of the artifact verified (digest string).
    pub artifact_identity: String,
    /// Maximum absolute deviation observed against the reference.
    pub max_abs_deviation: f32,
    /// Mean absolute deviation observed against the reference.
    pub mean_abs_deviation: f32,
    /// Cosine similarity between reference and executable outputs.
    pub cosine_similarity: f32,
    /// Whether the executable passed the numerical tolerance.
    pub verified: bool,
}
