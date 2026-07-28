//! Residency verification receipt — the canonical attestation that a
//! compiled executable's residency plan is satisfiable on the target
//! device.

use serde::{Deserialize, Serialize};

/// Residency verification receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyVerificationReceipt {
    /// Identity of the artifact verified (digest string).
    pub artifact_identity: String,
    /// Whether the residency plan is satisfiable.
    pub residency_ok: bool,
    /// Total weight bytes required by the plan.
    pub total_weight_bytes: u64,
    /// Number of mandatory weight objects (cannot be evicted).
    pub mandatory_object_count: u32,
    /// Peak activation bytes required at any time.
    pub peak_activation_bytes: u64,
}
