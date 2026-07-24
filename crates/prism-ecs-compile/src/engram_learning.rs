//! Versioned engram bank, routing, and model-accommodation contracts.

use crate::agentic_workload::AgenticWorkloadClass;
use prism_ecs_ir::semantic_region::SemanticRegionId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngramRepresentation {
    DenseVector,
    LowRankState,
    SparseCode,
    RecurrentState,
    RetrievalTemplate,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramEntry {
    pub id: String,
    pub semantic_scope: Vec<SemanticRegionId>,
    pub workload_classes: Vec<AgenticWorkloadClass>,
    pub representation: EngramRepresentation,
    pub retrieval_key_digest: String,
    pub payload_digest: String,
    pub utility_score: f64,
    pub interference_score: f64,
    pub access_count: u64,
    pub merge_candidates: Vec<String>,
    pub retired: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramRouterPolicy {
    pub policy_id: String,
    pub source_generation_digest: String,
    pub max_retrievals_per_step: u32,
    pub minimum_confidence: f64,
    pub workload_routes: BTreeMap<AgenticWorkloadClass, Vec<String>>,
    pub region_routes: BTreeMap<String, Vec<String>>,
    pub fallback_to_no_engram: bool,
    #[serde(default)]
    pub policy_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramGeneration {
    pub generation_id: String,
    pub parent_generation: Option<String>,
    pub source_cimage_generation_digest: String,
    pub entries: Vec<EngramEntry>,
    pub router_policy: EngramRouterPolicy,
    pub accommodation_adapter_digest: Option<String>,
    pub calibration_digest: String,
    pub admission_receipt_digest: Option<String>,
    #[serde(default)]
    pub generation_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramEvaluation {
    pub generation_digest: String,
    pub measured: bool,
    pub execution_fingerprint: String,
    pub task_success: f64,
    pub tool_call_correctness: f64,
    pub retrieval_precision: f64,
    pub retrieval_recall: f64,
    pub mean_utility_delta: f64,
    pub mean_interference_delta: f64,
    pub token_savings_ratio: f64,
    pub retry_reduction_ratio: f64,
    pub routing_stability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramAdmissionPolicy {
    pub min_task_success: f64,
    pub min_tool_call_correctness: f64,
    pub min_retrieval_precision: f64,
    pub min_mean_utility_delta: f64,
    pub max_mean_interference_delta: f64,
    pub min_routing_stability: f64,
    pub max_entries: usize,
}

impl Default for EngramAdmissionPolicy {
    fn default() -> Self {
        Self {
            min_task_success: 0.98,
            min_tool_call_correctness: 0.99,
            min_retrieval_precision: 0.95,
            min_mean_utility_delta: 0.0,
            max_mean_interference_delta: 0.01,
            min_routing_stability: 0.95,
            max_entries: 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramAdmissionReceipt {
    pub generation_digest: String,
    pub admitted: bool,
    pub measured: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EngramLearningError {
    #[error("generation identity is empty")]
    MissingIdentity,
    #[error("router policy is invalid")]
    InvalidRouter,
    #[error("engram metric is outside the valid range")]
    InvalidMetric,
    #[error("engram generation digest mismatch")]
    DigestMismatch,
    #[error("engram evaluation is not authoritative")]
    Unmeasured,
}

impl EngramRouterPolicy {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.policy_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, EngramLearningError> {
        if self.policy_id.is_empty()
            || self.source_generation_digest.is_empty()
            || self.max_retrievals_per_step == 0
            || !self.minimum_confidence.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_confidence)
        {
            return Err(EngramLearningError::InvalidRouter);
        }
        self.policy_digest = self.canonical_digest();
        Ok(self)
    }
}

impl EngramGeneration {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.generation_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, EngramLearningError> {
        if self.generation_id.is_empty()
            || self.source_cimage_generation_digest.is_empty()
            || self.calibration_digest.is_empty()
        {
            return Err(EngramLearningError::MissingIdentity);
        }
        self.router_policy = self.router_policy.clone().seal()?;
        for entry in &self.entries {
            for metric in [entry.utility_score, entry.interference_score] {
                if !metric.is_finite() {
                    return Err(EngramLearningError::InvalidMetric);
                }
            }
        }
        self.entries.sort_by(|a, b| a.id.cmp(&b.id));
        self.generation_digest = self.canonical_digest();
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), EngramLearningError> {
        if self.generation_digest != self.canonical_digest() {
            return Err(EngramLearningError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn admit_engram_generation(
    generation: &EngramGeneration,
    evaluation: &EngramEvaluation,
    policy: &EngramAdmissionPolicy,
) -> Result<EngramAdmissionReceipt, EngramLearningError> {
    generation.verify()?;
    if !evaluation.measured
        || evaluation.execution_fingerprint.is_empty()
        || evaluation.generation_digest != generation.generation_digest
    {
        return Err(EngramLearningError::Unmeasured);
    }
    let mut reasons = Vec::new();
    if generation.entries.len() > policy.max_entries {
        reasons.push("engram entry budget exceeded".into());
    }
    if evaluation.task_success < policy.min_task_success {
        reasons.push("task success below gate".into());
    }
    if evaluation.tool_call_correctness < policy.min_tool_call_correctness {
        reasons.push("tool-call correctness below gate".into());
    }
    if evaluation.retrieval_precision < policy.min_retrieval_precision {
        reasons.push("retrieval precision below gate".into());
    }
    if evaluation.mean_utility_delta < policy.min_mean_utility_delta {
        reasons.push("engram utility below gate".into());
    }
    if evaluation.mean_interference_delta > policy.max_mean_interference_delta {
        reasons.push("engram interference above gate".into());
    }
    if evaluation.routing_stability < policy.min_routing_stability {
        reasons.push("engram routing stability below gate".into());
    }
    let mut receipt = EngramAdmissionReceipt {
        generation_digest: generation.generation_digest.clone(),
        admitted: reasons.is_empty(),
        measured: true,
        reasons,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("engram canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
