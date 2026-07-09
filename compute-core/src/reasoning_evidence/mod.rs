//! Reasoning evidence — epistemic behavior receipts, distillation guardrails,
//! and promotion eligibility checks for model behavior evaluation.

pub mod distillation_guard;
pub mod epistemic;
pub mod receipts;

pub use distillation_guard::{
    DistillationGuardrail, DistillationSignalDecompositionReceipt, EpistemicThresholds,
    PromotionCheck,
};
pub use epistemic::{EpistemicBehaviorReceipt, EpistemicDegradationFlag, EpistemicMarkerSet};
pub use receipts::EvidenceReceiptHeader;

#[cfg(test)]
mod tests;
