//! Common evidence receipt types shared across the reasoning evidence module.

use serde::{Deserialize, Serialize};

/// Header carried by every evidence receipt for provenance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceiptHeader {
    pub receipt_id: String,
    pub version: u32,
    pub created_at_unix_ms: u64,
}
