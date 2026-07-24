//! Unified admission, promotion, shadow activation, and rollback gates for Living CImages.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementDomain {
    ProgressiveTernary,
    SpeculativeInference,
    KvCacheCompression,
    KvCacheCompaction,
    AdapterTraining,
    EngramLearning,
    KnowledgeCorrection,
    ModelExport,
    ExecutionGraphEvolution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DomainAdmissionEvidence {
    pub domain: RefinementDomain,
    pub artifact_digest: String,
    pub receipt_digest: String,
    pub measured: bool,
    pub admitted: bool,
    pub score: f64,
    pub hard_gate_failures: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingGenerationPromotionRequest {
    pub request_id: String,
    pub living_cimage_id: String,
    pub baseline_generation_digest: String,
    pub candidate_generation_digest: String,
    pub shadow_receipt_digest: String,
    pub domain_evidence: Vec<DomainAdmissionEvidence>,
    pub rollback_target_generation_digest: String,
    pub requested_by: String,
    #[serde(default)]
    pub request_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingGenerationPromotionPolicy {
    pub required_domains: Vec<RefinementDomain>,
    pub require_all_measured: bool,
    pub min_domain_score: f64,
    pub require_shadow_receipt: bool,
    pub require_rollback_target: bool,
    pub max_total_hard_gate_failures: usize,
}

impl Default for LivingGenerationPromotionPolicy {
    fn default() -> Self {
        Self {
            required_domains: Vec::new(),
            require_all_measured: true,
            min_domain_score: 0.0,
            require_shadow_receipt: true,
            require_rollback_target: true,
            max_total_hard_gate_failures: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingGenerationPromotionReceipt {
    pub request_digest: String,
    pub promoted_generation_digest: String,
    pub previous_generation_digest: String,
    pub rollback_target_generation_digest: String,
    pub admitted: bool,
    pub reasons: Vec<String>,
    pub domain_receipts: BTreeMap<String, String>,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LivingGenerationRollbackReceipt {
    pub living_cimage_id: String,
    pub from_generation_digest: String,
    pub to_generation_digest: String,
    pub cause: String,
    pub operator: String,
    pub rollback_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LivingPromotionError {
    #[error("promotion identity is incomplete")]
    MissingIdentity,
    #[error("promotion request digest mismatch")]
    DigestMismatch,
    #[error("required domain evidence is missing")]
    MissingDomain,
    #[error("rollback target is missing")]
    MissingRollback,
}

impl LivingGenerationPromotionRequest {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.request_digest.clear();
        canonical
            .domain_evidence
            .sort_by(|a, b| format!("{:?}", a.domain).cmp(&format!("{:?}", b.domain)));
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, LivingPromotionError> {
        self.request_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), LivingPromotionError> {
        if self.request_id.is_empty()
            || self.living_cimage_id.is_empty()
            || self.baseline_generation_digest.is_empty()
            || self.candidate_generation_digest.is_empty()
            || self.requested_by.is_empty()
        {
            return Err(LivingPromotionError::MissingIdentity);
        }
        if !self.request_digest.is_empty() && self.request_digest != self.canonical_digest() {
            return Err(LivingPromotionError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn evaluate_promotion(
    request: &LivingGenerationPromotionRequest,
    policy: &LivingGenerationPromotionPolicy,
) -> Result<LivingGenerationPromotionReceipt, LivingPromotionError> {
    request.verify()?;
    if policy.require_rollback_target && request.rollback_target_generation_digest.is_empty() {
        return Err(LivingPromotionError::MissingRollback);
    }
    let mut reasons = Vec::new();
    let mut domain_receipts = BTreeMap::new();
    for required in &policy.required_domains {
        if !request
            .domain_evidence
            .iter()
            .any(|evidence| &evidence.domain == required)
        {
            reasons.push(format!("missing required domain {required:?}"));
        }
    }
    let hard_failures = request
        .domain_evidence
        .iter()
        .map(|evidence| evidence.hard_gate_failures.len())
        .sum::<usize>();
    if hard_failures > policy.max_total_hard_gate_failures {
        reasons.push("hard-gate failure budget exceeded".into());
    }
    if policy.require_shadow_receipt && request.shadow_receipt_digest.is_empty() {
        reasons.push("shadow evaluation receipt is missing".into());
    }
    for evidence in &request.domain_evidence {
        domain_receipts.insert(
            format!("{:?}", evidence.domain),
            evidence.receipt_digest.clone(),
        );
        if !evidence.admitted {
            reasons.push(format!("domain {:?} is not admitted", evidence.domain));
        }
        if policy.require_all_measured && !evidence.measured {
            reasons.push(format!("domain {:?} is not measured", evidence.domain));
        }
        if evidence.score < policy.min_domain_score {
            reasons.push(format!("domain {:?} score below gate", evidence.domain));
        }
        if evidence.artifact_digest.is_empty() || evidence.receipt_digest.is_empty() {
            reasons.push(format!(
                "domain {:?} lacks artifact or receipt identity",
                evidence.domain
            ));
        }
    }
    let mut receipt = LivingGenerationPromotionReceipt {
        request_digest: request.request_digest.clone(),
        promoted_generation_digest: request.candidate_generation_digest.clone(),
        previous_generation_digest: request.baseline_generation_digest.clone(),
        rollback_target_generation_digest: request.rollback_target_generation_digest.clone(),
        admitted: reasons.is_empty(),
        reasons,
        domain_receipts,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

pub fn build_rollback_receipt(
    living_cimage_id: impl Into<String>,
    from_generation_digest: impl Into<String>,
    to_generation_digest: impl Into<String>,
    cause: impl Into<String>,
    operator: impl Into<String>,
) -> Result<LivingGenerationRollbackReceipt, LivingPromotionError> {
    let mut receipt = LivingGenerationRollbackReceipt {
        living_cimage_id: living_cimage_id.into(),
        from_generation_digest: from_generation_digest.into(),
        to_generation_digest: to_generation_digest.into(),
        cause: cause.into(),
        operator: operator.into(),
        rollback_digest: String::new(),
    };
    if receipt.living_cimage_id.is_empty()
        || receipt.from_generation_digest.is_empty()
        || receipt.to_generation_digest.is_empty()
        || receipt.cause.is_empty()
        || receipt.operator.is_empty()
    {
        return Err(LivingPromotionError::MissingIdentity);
    }
    receipt.rollback_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("living promotion canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
