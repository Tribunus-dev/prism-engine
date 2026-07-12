//! Training-aware compilation — targets, gates, feedback, and receipts.
//!
//! This module defines the types and logic for generating training targets
//! from compiler policies, evaluating evidence against gates, and producing
//! feedback for iterative quantization-aware training.

pub mod engram;
pub mod export;
pub mod feedback;
pub mod gates;
pub mod receipts;
pub mod resolve;
pub mod spec;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

// ── Re-exports — spec ──────────────────────────────────────────────────

pub use self::spec::{
    ActivationWeightedObjective, AttentionShapeTrainingTarget, EngramArtifact, EngramLookupParams,
    EngramLookupPolicy, EngramLookupReceipt, EngramTrainingTarget, KvCacheTrainingTarget,
    SpeculativeTrainingTarget, TrainingEvidenceGate, TrainingTargetPriority, TrainingTargetSpec,
    WeightTrainingTarget,
};

// ── Re-exports — gates ─────────────────────────────────────────────────

pub use self::gates::{
    QuantTrainingMethod, RequiredEvidenceLevel, TargetedLossTerm, TrainingFailureMode,
    TrainingTargetStatus, WeightTrainingGates,
};

// ── Re-exports — feedback ──────────────────────────────────────────────

pub use self::feedback::{TrainingFeedbackItem, TrainingFeedbackReport, TrainingFeedbackSummary};

// ── Re-exports — receipts ──────────────────────────────────────────────

pub use self::receipts::{TrainingFeedbackReceipt, TrainingTargetReceipt};
