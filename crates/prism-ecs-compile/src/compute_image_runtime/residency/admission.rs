//! Residency admission — types for admission decisions and refusal reasons.

use serde::{Deserialize, Serialize};

/// Result of a residency admission check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyAdmissionResult {
    /// Whether the admission succeeded.
    pub admitted: bool,
    /// If refused, the reason for refusal.
    pub refusal_reason: Option<ResidencyRefusalReason>,
    /// Admission contract applied.
    pub contract: Option<crate::compute_image_runtime::residency::plan::MemoryAdmissionContract>,
}

/// Reasons a residency admission may be refused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResidencyRefusalReason {
    /// Total resident bytes exceed the available budget.
    InsufficientMemory {
        /// Bytes required.
        required: u64,
        /// Bytes available.
        available: u64,
    },
    /// A required weight object's residency class is not satisfiable.
    UnresolvableResidencyClass {
        /// Class that could not be resolved.
        class: String,
    },
    /// A KV cache dimension constraint could not be satisfied.
    KvCacheUnsatisfiable,
}

/// Residency admission gate.
#[derive(Debug, Clone, Default)]
pub struct ResidencyAdmission;

impl ResidencyAdmission {
    /// Create a new admission gate.
    pub fn new() -> Self {
        Self
    }
}
