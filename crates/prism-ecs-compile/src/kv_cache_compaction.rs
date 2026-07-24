//! Progressive KV-cache compaction policies for Living CImage refinement.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCompactionAlgorithm {
    PagePacking,
    PrefixDeduplication,
    SpanCoalescing,
    HeadGroupCompaction,
    SlidingWindowCompaction,
    SnapshotCompaction,
    SparseIndexCompaction,
    ResidualPageCompaction,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCompactionCandidate {
    pub candidate_id: String,
    pub source_generation_digest: String,
    pub algorithm: KvCompactionAlgorithm,
    pub page_size_tokens: u32,
    pub window_tokens: Option<u32>,
    pub retained_prefix_tokens: u32,
    pub head_group_size: Option<u32>,
    pub sparse_stride: Option<u32>,
    pub residual_precision: Option<String>,
    pub estimated_compacted_bytes: u64,
    pub estimated_rewrite_bytes: u64,
    pub estimated_index_bytes: u64,
    pub policy: BTreeMap<String, String>,
    #[serde(default)]
    pub candidate_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCompactionMeasurement {
    pub candidate_digest: String,
    pub measured: bool,
    pub execution_fingerprint: String,
    pub original_bytes: u64,
    pub compacted_bytes: u64,
    pub rewrite_bytes: u64,
    pub index_bytes: u64,
    pub compaction_latency_ms: f64,
    pub decode_latency_ms: f64,
    pub tokens_per_second: f64,
    pub long_context_recall: f64,
    pub instruction_retention: f64,
    pub tool_call_correctness: f64,
    pub agentic_success: f64,
    pub system_prompt_leakage: f64,
    pub fallback_frequency: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCompactionAdmissionPolicy {
    pub max_compacted_bytes: Option<u64>,
    pub max_rewrite_bytes: Option<u64>,
    pub max_index_bytes: Option<u64>,
    pub max_compaction_latency_ms: Option<f64>,
    pub max_decode_latency_ms: Option<f64>,
    pub min_long_context_recall: f64,
    pub min_instruction_retention: f64,
    pub min_tool_call_correctness: f64,
    pub min_agentic_success: f64,
    pub max_system_prompt_leakage: f64,
    pub max_fallback_frequency: f64,
}

impl Default for KvCompactionAdmissionPolicy {
    fn default() -> Self {
        Self {
            max_compacted_bytes: None,
            max_rewrite_bytes: None,
            max_index_bytes: None,
            max_compaction_latency_ms: None,
            max_decode_latency_ms: None,
            min_long_context_recall: 0.98,
            min_instruction_retention: 0.99,
            min_tool_call_correctness: 0.99,
            min_agentic_success: 0.98,
            max_system_prompt_leakage: 0.0,
            max_fallback_frequency: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCompactionReceipt {
    pub candidate_digest: String,
    pub source_generation_digest: String,
    pub admitted: bool,
    pub measured: bool,
    pub compression_ratio: f64,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvCompactionError {
    #[error("candidate source generation digest is empty")]
    MissingSourceGeneration,
    #[error("page size must be nonzero")]
    InvalidPageSize,
    #[error("candidate digest mismatch")]
    DigestMismatch,
    #[error("measurement is not authoritative")]
    Unmeasured,
}

impl KvCompactionCandidate {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.candidate_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, KvCompactionError> {
        self.candidate_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), KvCompactionError> {
        if self.source_generation_digest.is_empty() {
            return Err(KvCompactionError::MissingSourceGeneration);
        }
        if self.page_size_tokens == 0 {
            return Err(KvCompactionError::InvalidPageSize);
        }
        if !self.candidate_digest.is_empty() && self.candidate_digest != self.canonical_digest() {
            return Err(KvCompactionError::DigestMismatch);
        }
        Ok(())
    }
}

pub fn evaluate_kv_compaction(
    candidate: &KvCompactionCandidate,
    measurement: &KvCompactionMeasurement,
    policy: &KvCompactionAdmissionPolicy,
) -> Result<KvCompactionReceipt, KvCompactionError> {
    candidate.verify()?;
    if !measurement.measured || measurement.execution_fingerprint.is_empty() {
        return Err(KvCompactionError::Unmeasured);
    }
    let mut reasons = Vec::new();
    if let Some(limit) = policy.max_compacted_bytes {
        if measurement.compacted_bytes > limit { reasons.push("compacted bytes exceed budget".into()); }
    }
    if let Some(limit) = policy.max_rewrite_bytes {
        if measurement.rewrite_bytes > limit { reasons.push("rewrite bytes exceed budget".into()); }
    }
    if let Some(limit) = policy.max_index_bytes {
        if measurement.index_bytes > limit { reasons.push("index bytes exceed budget".into()); }
    }
    if let Some(limit) = policy.max_compaction_latency_ms {
        if measurement.compaction_latency_ms > limit { reasons.push("compaction latency exceeds budget".into()); }
    }
    if let Some(limit) = policy.max_decode_latency_ms {
        if measurement.decode_latency_ms > limit { reasons.push("decode latency exceeds budget".into()); }
    }
    if measurement.long_context_recall < policy.min_long_context_recall { reasons.push("long-context recall regressed".into()); }
    if measurement.instruction_retention < policy.min_instruction_retention { reasons.push("instruction retention regressed".into()); }
    if measurement.tool_call_correctness < policy.min_tool_call_correctness { reasons.push("tool-call correctness regressed".into()); }
    if measurement.agentic_success < policy.min_agentic_success { reasons.push("agentic success regressed".into()); }
    if measurement.system_prompt_leakage > policy.max_system_prompt_leakage { reasons.push("system-prompt leakage exceeded limit".into()); }
    if measurement.fallback_frequency > policy.max_fallback_frequency { reasons.push("fallback frequency exceeded limit".into()); }

    let compression_ratio = if measurement.compacted_bytes == 0 {
        f64::INFINITY
    } else {
        measurement.original_bytes as f64 / measurement.compacted_bytes as f64
    };
    let admitted = reasons.is_empty();
    let mut receipt = KvCompactionReceipt {
        candidate_digest: candidate.candidate_digest.clone(),
        source_generation_digest: candidate.source_generation_digest.clone(),
        admitted,
        measured: true,
        compression_ratio,
        reasons,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

pub fn propose_compaction_candidates(
    source_generation_digest: &str,
    original_bytes: u64,
) -> Vec<KvCompactionCandidate> {
    let specs = [
        (KvCompactionAlgorithm::PagePacking, 128, None, 0, None),
        (KvCompactionAlgorithm::PrefixDeduplication, 128, None, 256, None),
        (KvCompactionAlgorithm::SlidingWindowCompaction, 64, Some(4096), 256, None),
        (KvCompactionAlgorithm::HeadGroupCompaction, 128, None, 256, Some(4)),
        (KvCompactionAlgorithm::SparseIndexCompaction, 128, Some(8192), 256, Some(8)),
        (KvCompactionAlgorithm::Hybrid, 64, Some(4096), 512, Some(4)),
    ];
    specs
        .into_iter()
        .enumerate()
        .filter_map(|(index, (algorithm, page, window, prefix, heads))| {
            KvCompactionCandidate {
                candidate_id: format!("kv-compact-{index}"),
                source_generation_digest: source_generation_digest.into(),
                algorithm,
                page_size_tokens: page,
                window_tokens: window,
                retained_prefix_tokens: prefix,
                head_group_size: heads,
                sparse_stride: (index >= 4).then_some(8),
                residual_precision: (index == 5).then_some("int8".into()),
                estimated_compacted_bytes: original_bytes.saturating_mul(55 + index as u64 * 4) / 100,
                estimated_rewrite_bytes: original_bytes.saturating_mul(5 + index as u64) / 100,
                estimated_index_bytes: original_bytes.saturating_mul(1 + index as u64) / 100,
                policy: BTreeMap::new(),
                candidate_digest: String::new(),
            }.seal().ok()
        })
        .collect()
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("kv compaction canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_are_sealed_and_generation_bound() {
        let candidates = propose_compaction_candidates("generation-3", 1_000_000);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| !candidate.candidate_digest.is_empty()));
        assert!(candidates.iter().all(|candidate| candidate.source_generation_digest == "generation-3"));
    }

    #[test]
    fn admission_rejects_instruction_regression() {
        let candidate = propose_compaction_candidates("generation-3", 1_000_000).remove(0);
        let receipt = evaluate_kv_compaction(&candidate, &KvCompactionMeasurement {
            candidate_digest: candidate.candidate_digest.clone(), measured: true,
            execution_fingerprint: "real-run".into(), original_bytes: 1_000_000,
            compacted_bytes: 500_000, rewrite_bytes: 10_000, index_bytes: 5_000,
            compaction_latency_ms: 1.0, decode_latency_ms: 10.0, tokens_per_second: 50.0,
            long_context_recall: 1.0, instruction_retention: 0.8,
            tool_call_correctness: 1.0, agentic_success: 1.0,
            system_prompt_leakage: 0.0, fallback_frequency: 0.0,
        }, &KvCompactionAdmissionPolicy::default()).unwrap();
        assert!(!receipt.admitted);
    }
}
