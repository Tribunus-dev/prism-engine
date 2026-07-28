//! Residency admission and execution receipts.

use serde::{Deserialize, Serialize};

/// Receipt attesting that a residency admission succeeded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyAdmissionReceipt {
    /// Identifier of the plan admitted.
    pub plan_id: String,
    /// Bytes required at admission time.
    pub required_bytes: u64,
    /// Bytes available at admission time.
    pub available_bytes: u64,
    /// Whether graceful degradation was allowed.
    pub graceful_degradation: bool,
    /// Admission timestamp (RFC3339 string).
    pub timestamp: String,
}

/// Receipt attesting to a completed residency execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyExecutionReceipt {
    /// Identifier of the plan executed.
    pub plan_id: String,
    /// Total bytes loaded during execution.
    pub bytes_loaded: u64,
    /// Total bytes evicted during execution.
    pub bytes_evicted: u64,
    /// Number of prefetch actions executed.
    pub prefetch_count: u32,
    /// Execution timestamp (RFC3339 string).
    pub timestamp: String,
}
