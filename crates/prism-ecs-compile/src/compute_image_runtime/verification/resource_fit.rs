//! Resource-fit verification receipt — the canonical attestation that a
//! compiled executable's resource footprint fits within the target
//! device's capacity.

use serde::{Deserialize, Serialize};

/// Resource-fit verification receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceFitReceipt {
    /// Identity of the artifact verified (digest string).
    pub artifact_identity: String,
    /// Required peak memory in bytes.
    pub required_peak_memory_bytes: u64,
    /// Available peak memory in bytes.
    pub available_peak_memory_bytes: u64,
    /// Required persistent weight bytes.
    pub required_persistent_bytes: u64,
    /// Available persistent weight bytes.
    pub available_persistent_bytes: u64,
    /// Whether the executable fits within the device's resources.
    pub fits: bool,
}
