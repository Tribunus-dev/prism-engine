//! Reversible region-scoped adapter training contracts for Living CImage generations.

use prism_ecs_ir::semantic_region::SemanticRegionId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    Lora,
    LowRankResidual,
    BiasOnly,
    RouterAdapter,
    EngramAccommodation,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterTrainingRequest {
    pub request_id: String,
    pub source_generation_digest: String,
    pub calibration_corpus_digest: String,
    pub adapter_kind: AdapterKind,
    pub target_regions: Vec<SemanticRegionId>,
    pub rank: Option<u32>,
    pub learning_rate: f64,
    pub max_steps: u32,
    pub max_trainable_bytes: u64,
    pub freeze_base_weights: bool,
    pub objective_weights: BTreeMap<String, f64>,
    #[serde(default)]
    pub request_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterArtifact {
    pub request_digest: String,
    pub artifact_digest: String,
    pub target_regions: Vec<SemanticRegionId>,
    pub trainable_bytes: u64,
    pub parameter_count: u64,
    pub storage_format: String,
    pub mergeable: bool,
    pub reversible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterTrainingReceipt {
    pub request_digest: String,
    pub artifact_digest: String,
    pub measured: bool,
    pub trainer_fingerprint: String,
    pub steps_completed: u32,
    pub final_loss: f64,
    pub task_success: f64,
    pub tool_call_correctness: f64,
    pub locality_score: f64,
    pub unrelated_regression: f64,
    pub admitted: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterAdmissionPolicy {
    pub max_trainable_bytes: u64,
    pub min_task_success: f64,
    pub min_tool_call_correctness: f64,
    pub min_locality_score: f64,
    pub max_unrelated_regression: f64,
    pub require_frozen_base: bool,
    pub require_reversible: bool,
}

impl Default for AdapterAdmissionPolicy {
    fn default() -> Self {
        Self {
            max_trainable_bytes: 256 * 1024 * 1024,
            min_task_success: 0.98,
            min_tool_call_correctness: 0.99,
            min_locality_score: 0.95,
            max_unrelated_regression: 0.01,
            require_frozen_base: true,
            require_reversible: true,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AdapterTrainingError {
    #[error("request identifier or generation digest is empty")]
    MissingIdentity,
    #[error("training request has no target regions")]
    MissingRegions,
    #[error("training parameters are invalid")]
    InvalidParameters,
    #[error("request digest mismatch")]
    DigestMismatch,
    #[error("training receipt is not authoritative")]
    Unmeasured,
}

impl AdapterTrainingRequest {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.request_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, AdapterTrainingError> {
        self.request_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), AdapterTrainingError> {
        if self.request_id.is_empty() || self.source_generation_digest.is_empty() || self.calibration_corpus_digest.is_empty() {
            return Err(AdapterTrainingError::MissingIdentity);
        }
        if self.target_regions.is_empty() {
            return Err(AdapterTrainingError::MissingRegions);
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 || self.max_steps == 0 || self.max_trainable_bytes == 0 {
            return Err(AdapterTrainingError::InvalidParameters);
        }
        if !self.request_digest.is_empty() && self.request_digest != self.canonical_digest() {
            return Err(AdapterTrainingError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn admit_adapter(
    request: &AdapterTrainingRequest,
    artifact: &AdapterArtifact,
    mut receipt: AdapterTrainingReceipt,
    policy: &AdapterAdmissionPolicy,
) -> Result<AdapterTrainingReceipt, AdapterTrainingError> {
    request.verify()?;
    if !receipt.measured || receipt.trainer_fingerprint.is_empty() {
        return Err(AdapterTrainingError::Unmeasured);
    }
    let mut reasons = Vec::new();
    if policy.require_frozen_base && !request.freeze_base_weights { reasons.push("base weights were not frozen".into()); }
    if policy.require_reversible && !artifact.reversible { reasons.push("adapter artifact is not reversible".into()); }
    if artifact.trainable_bytes > policy.max_trainable_bytes || artifact.trainable_bytes > request.max_trainable_bytes { reasons.push("adapter exceeds trainable-byte budget".into()); }
    if receipt.task_success < policy.min_task_success { reasons.push("task success below gate".into()); }
    if receipt.tool_call_correctness < policy.min_tool_call_correctness { reasons.push("tool-call correctness below gate".into()); }
    if receipt.locality_score < policy.min_locality_score { reasons.push("adapter locality below gate".into()); }
    if receipt.unrelated_regression > policy.max_unrelated_regression { reasons.push("unrelated regression exceeds gate".into()); }
    receipt.request_digest = request.request_digest.clone();
    receipt.artifact_digest = artifact.artifact_digest.clone();
    receipt.admitted = reasons.is_empty();
    receipt.reasons = reasons;
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("adapter training canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
