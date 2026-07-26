//! Epistemic behavior receipts and marker sets.

use serde::{Deserialize, Serialize};

use super::receipts::EvidenceReceiptHeader;

/// Receipt capturing epistemic behavior characteristics for a model evaluation
/// session. Used by distillation guardrails to assess promotion eligibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicBehaviorReceipt {
    pub receipt_id: String,
    pub model_or_partition_id: String,
    pub evaluation_set_id: String,
    pub trace_count: u32,
    pub uncertainty_marker_rate: f64,
    pub self_correction_marker_rate: f64,
    pub average_reasoning_length: f64,
    pub ood_accuracy: Option<f64>,
    pub in_domain_accuracy: Option<f64>,
    pub degradation_flags: Vec<EpistemicDegradationFlag>,
    /// "Synthetic" | "Measured"
    pub evidence_kind: String,
    pub promotion_eligible: bool,
}

impl EpistemicBehaviorReceipt {
    /// Creates a new receipt with a header-based receipt_id.
    pub fn new(
        header: &EvidenceReceiptHeader,
        model_or_partition_id: impl Into<String>,
        evaluation_set_id: impl Into<String>,
        trace_count: u32,
        uncertainty_marker_rate: f64,
        self_correction_marker_rate: f64,
        average_reasoning_length: f64,
        evidence_kind: impl Into<String>,
    ) -> Self {
        Self {
            receipt_id: header.receipt_id.clone(),
            model_or_partition_id: model_or_partition_id.into(),
            evaluation_set_id: evaluation_set_id.into(),
            trace_count,
            uncertainty_marker_rate,
            self_correction_marker_rate,
            average_reasoning_length,
            ood_accuracy: None,
            in_domain_accuracy: None,
            degradation_flags: Vec::new(),
            evidence_kind: evidence_kind.into(),
            promotion_eligible: false,
        }
    }
}

/// Flags indicating which epistemic degradation patterns were observed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EpistemicDegradationFlag {
    SuppressedUncertaintyMarkers,
    ReasoningTraceCollapse,
    InDomainGainOodLoss,
    ShortcutStyleIncrease,
    SelfCorrectionDrop,
}

/// A curated marker set for detecting epistemic behavior in model outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicMarkerSet {
    pub marker_set_id: String,
    pub language: String,
    pub uncertainty_markers: Vec<String>,
    pub self_correction_markers: Vec<String>,
    pub reformulation_markers: Vec<String>,
}

impl Default for EpistemicMarkerSet {
    fn default() -> Self {
        Self {
            marker_set_id: "english_default_v1".into(),
            language: "en".into(),
            uncertainty_markers: vec![
                "I'm not sure".into(),
                "I think".into(),
                "maybe".into(),
                "perhaps".into(),
                "it depends".into(),
                "one possibility".into(),
                "not certain".into(),
                "might be".into(),
                "could be".into(),
            ],
            self_correction_markers: vec![
                "actually".into(),
                "wait".into(),
                "let me reconsider".into(),
                "on second thought".into(),
                "let me think".into(),
                "I need to rethink".into(),
                "let me check".into(),
            ],
            reformulation_markers: vec![
                "in other words".into(),
                "more precisely".into(),
                "let me rephrase".into(),
                "put differently".into(),
            ],
        }
    }
}
