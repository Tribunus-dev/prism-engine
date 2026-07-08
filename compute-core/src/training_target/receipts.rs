//! Receipt types for training target generation and feedback processing.

use serde::{Deserialize, Serialize};

use super::gates::TrainingTargetStatus;

/// Receipt emitted when a TrainingTargetSpec is generated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTargetReceipt {
    pub spec_digest: String,
    pub source_policy_digest: String,
    pub generated_at_unix_ms: u64,
    pub target_count: usize,
    pub weight_target_count: usize,
    pub warnings: Vec<String>,
}

/// Receipt emitted when a TrainingFeedbackReport is processed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingFeedbackReceipt {
    pub report_digest: String,
    pub status: TrainingTargetStatus,
    pub total_items: usize,
    pub failed_items: usize,
    pub satisfied_targets: usize,
}
