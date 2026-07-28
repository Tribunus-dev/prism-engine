//! Receipt types for executable compilation and admission.

use serde::{Deserialize, Serialize};

/// Opaque identifier for a compilation receipt.
pub type ReceiptId = String;

/// Compilation receipt for a single executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableCompilationReceipt {
    /// Receipt identifier.
    pub receipt_id: ReceiptId,
    /// Model name.
    pub model_name: String,
    /// Number of target profiles in the executable.
    pub profile_count: u32,
    /// Total weight bytes across all profiles.
    pub total_weight_bytes: u64,
    /// Compilation duration in milliseconds.
    pub compilation_duration_ms: u64,
    /// Hash of the executable's seal.
    pub seal_hash: String,
}
