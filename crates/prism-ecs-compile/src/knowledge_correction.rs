//! Sourced, temporal, reversible knowledge corrections for Living CImages.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelErrorClass {
    StaleFact,
    MissingFact,
    ReasoningFailure,
    ToolSelectionFailure,
    RetrievalFailure,
    EngramMisroute,
    QuantizationDrift,
    InstructionFailure,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMechanism {
    RetrievalRevision,
    EngramRevision,
    AdapterCorrection,
    WeightDelta,
    ToolPolicyRevision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefreshPolicy {
    Never,
    OnDemand,
    Daily,
    Weekly,
    BeforeUse,
    ExternalAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalKnowledgeContract {
    pub asserted_at: String,
    pub valid_from: Option<String>,
    pub expires_at: Option<String>,
    pub refresh_policy: RefreshPolicy,
    pub source_refs: Vec<String>,
    pub source_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedModelError {
    pub error_id: String,
    pub generation_digest: String,
    pub prompt_digest: String,
    pub observed_answer_digest: String,
    pub error_class: ModelErrorClass,
    pub description: String,
    pub workload_episode_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeCorrectionProposal {
    pub proposal_id: String,
    pub source_error_id: String,
    pub source_generation_digest: String,
    pub subject: String,
    pub old_claim_digest: Option<String>,
    pub corrected_claim: String,
    pub mechanism: CorrectionMechanism,
    pub temporal_contract: TemporalKnowledgeContract,
    pub correction_artifact_digest: Option<String>,
    pub locality_targets: Vec<String>,
    #[serde(default)]
    pub proposal_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeCorrectionEvaluation {
    pub proposal_digest: String,
    pub measured: bool,
    pub evaluation_fingerprint: String,
    pub corrected_prompt_accuracy: f64,
    pub paraphrase_accuracy: f64,
    pub multi_hop_accuracy: f64,
    pub locality_score: f64,
    pub contradiction_rate: f64,
    pub unrelated_regression: f64,
    pub agentic_success: f64,
    pub tool_call_correctness: f64,
    pub export_preservation_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeCorrectionPolicy {
    pub min_corrected_prompt_accuracy: f64,
    pub min_paraphrase_accuracy: f64,
    pub min_multi_hop_accuracy: f64,
    pub min_locality_score: f64,
    pub max_contradiction_rate: f64,
    pub max_unrelated_regression: f64,
    pub min_agentic_success: f64,
    pub min_tool_call_correctness: f64,
    pub require_sources: bool,
    pub require_expiry_for_external_authority: bool,
}

impl Default for KnowledgeCorrectionPolicy {
    fn default() -> Self {
        Self {
            min_corrected_prompt_accuracy: 1.0,
            min_paraphrase_accuracy: 0.95,
            min_multi_hop_accuracy: 0.90,
            min_locality_score: 0.95,
            max_contradiction_rate: 0.01,
            max_unrelated_regression: 0.01,
            min_agentic_success: 0.98,
            min_tool_call_correctness: 0.99,
            require_sources: true,
            require_expiry_for_external_authority: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeCorrectionReceipt {
    pub proposal_digest: String,
    pub admitted: bool,
    pub measured: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KnowledgeCorrectionError {
    #[error("proposal identity is incomplete")]
    MissingIdentity,
    #[error("knowledge correction lacks authoritative source evidence")]
    MissingSource,
    #[error("temporal contract is invalid")]
    InvalidTemporalContract,
    #[error("proposal digest mismatch")]
    DigestMismatch,
    #[error("evaluation is not authoritative")]
    Unmeasured,
}

impl KnowledgeCorrectionProposal {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.proposal_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, KnowledgeCorrectionError> {
        self.proposal_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), KnowledgeCorrectionError> {
        if self.proposal_id.is_empty() || self.source_error_id.is_empty() || self.source_generation_digest.is_empty() || self.subject.is_empty() || self.corrected_claim.is_empty() {
            return Err(KnowledgeCorrectionError::MissingIdentity);
        }
        if self.temporal_contract.source_refs.is_empty() || self.temporal_contract.source_digest.is_empty() {
            return Err(KnowledgeCorrectionError::MissingSource);
        }
        if matches!(self.temporal_contract.refresh_policy, RefreshPolicy::ExternalAuthority)
            && self.temporal_contract.expires_at.is_none()
        {
            return Err(KnowledgeCorrectionError::InvalidTemporalContract);
        }
        if !self.proposal_digest.is_empty() && self.proposal_digest != self.canonical_digest() {
            return Err(KnowledgeCorrectionError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn admit_knowledge_correction(
    proposal: &KnowledgeCorrectionProposal,
    evaluation: &KnowledgeCorrectionEvaluation,
    policy: &KnowledgeCorrectionPolicy,
) -> Result<KnowledgeCorrectionReceipt, KnowledgeCorrectionError> {
    proposal.verify()?;
    if !evaluation.measured || evaluation.evaluation_fingerprint.is_empty() || evaluation.proposal_digest != proposal.proposal_digest {
        return Err(KnowledgeCorrectionError::Unmeasured);
    }
    let mut reasons = Vec::new();
    if policy.require_sources && proposal.temporal_contract.source_refs.is_empty() { reasons.push("authoritative source evidence is missing".into()); }
    if policy.require_expiry_for_external_authority && matches!(proposal.temporal_contract.refresh_policy, RefreshPolicy::ExternalAuthority) && proposal.temporal_contract.expires_at.is_none() { reasons.push("external-authority correction lacks expiry".into()); }
    if evaluation.corrected_prompt_accuracy < policy.min_corrected_prompt_accuracy { reasons.push("corrected prompt accuracy below gate".into()); }
    if evaluation.paraphrase_accuracy < policy.min_paraphrase_accuracy { reasons.push("paraphrase accuracy below gate".into()); }
    if evaluation.multi_hop_accuracy < policy.min_multi_hop_accuracy { reasons.push("multi-hop accuracy below gate".into()); }
    if evaluation.locality_score < policy.min_locality_score { reasons.push("correction locality below gate".into()); }
    if evaluation.contradiction_rate > policy.max_contradiction_rate { reasons.push("contradiction rate exceeds gate".into()); }
    if evaluation.unrelated_regression > policy.max_unrelated_regression { reasons.push("unrelated regression exceeds gate".into()); }
    if evaluation.agentic_success < policy.min_agentic_success { reasons.push("agentic success below gate".into()); }
    if evaluation.tool_call_correctness < policy.min_tool_call_correctness { reasons.push("tool-call correctness below gate".into()); }
    let mut receipt = KnowledgeCorrectionReceipt { proposal_digest: proposal.proposal_digest.clone(), admitted: reasons.is_empty(), measured: true, reasons, receipt_digest: String::new() };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("knowledge correction canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
