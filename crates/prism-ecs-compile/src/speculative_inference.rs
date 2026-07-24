//! Speculative prefill and decoding contracts for Living CImage evolution.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculativePrefillStrategy {
    Disabled,
    TokenImportanceSubset {
        selector_model_digest: String,
        retain_ratio_bps: u32,
        preserve_positions: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculativeDecodeStrategy {
    Disabled,
    AutoregressiveDraft {
        draft_model_digest: String,
        max_draft_tokens: u32,
    },
    SemiAutoregressiveDspark {
        draft_model_digest: String,
        parallel_block: u32,
        sequential_tail: u32,
        confidence_schedule_digest: String,
    },
    BlockDiffusionDflash {
        draft_model_digest: String,
        block_size: u32,
        target_context_feature_digest: String,
    },
    DecoupledLongShort {
        dflash_digest: String,
        local_head_digest: String,
        block_size: u32,
    },
    VocabularySpeculation {
        draft_model_digest: String,
        vocabulary_selector_digest: String,
        vocabulary_budget: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationSchedule {
    Fixed {
        tokens: u32,
    },
    ConfidenceScheduled {
        min_tokens: u32,
        max_tokens: u32,
        survival_model_digest: String,
        throughput_profile_digest: String,
    },
    LoadAware {
        max_tokens: u32,
        queue_pressure_threshold_bps: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeInferenceCandidate {
    pub candidate_id: String,
    pub living_cimage_generation_digest: String,
    pub prefill: SpeculativePrefillStrategy,
    pub decode: SpeculativeDecodeStrategy,
    pub verification: VerificationSchedule,
    pub target_execution_graph_digest: String,
    pub kv_policy_digest: String,
    pub workload_class: String,
    pub expected_acceptance_length: Option<f64>,
    pub expected_ttft_ms: Option<f64>,
    pub expected_tokens_per_second: Option<f64>,
    pub candidate_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeMeasurement {
    pub candidate_digest: String,
    pub measured: bool,
    pub execution_fingerprint: String,
    pub workload_digest: String,
    pub ttft_ms: f64,
    pub tokens_per_second: f64,
    pub accepted_tokens_mean: f64,
    pub verification_waste_ratio: f64,
    pub draft_latency_ms: f64,
    pub verification_latency_ms: f64,
    pub quality_equivalent: bool,
    pub agentic_success_rate: f64,
    pub tool_call_correctness: f64,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeSearchBudget {
    pub max_candidates: usize,
    pub max_draft_models: usize,
    pub max_block_size: u32,
    pub max_verification_waste_ratio: f64,
    pub min_agentic_success_rate: f64,
    pub min_tool_call_correctness: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeSearchResult {
    pub workload_class: String,
    pub evaluated: Vec<SpeculativeMeasurement>,
    pub pareto_frontier: Vec<String>,
    pub selected_candidate_digest: Option<String>,
    pub rejection_reasons: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeInferenceReceipt {
    pub workload_class: String,
    pub selected_candidate_digest: Option<String>,
    pub candidate_count: usize,
    pub accepted_count: usize,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq)]
pub enum SpeculativeInferenceError {
    #[error("candidate references an empty Living CImage generation")]
    MissingGeneration,
    #[error("candidate references an empty execution graph")]
    MissingExecutionGraph,
    #[error("speculative block size exceeds search budget")]
    BlockBudgetExceeded,
    #[error("measurement is not authoritative")]
    NonAuthoritativeMeasurement,
    #[error("measurement violates quality or agentic gates")]
    QualityGateFailed,
}

impl SpeculativeInferenceCandidate {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.candidate_digest.clear();
        digest(&serde_json::to_vec(&canonical).expect("speculative candidate serialization"))
    }

    pub fn seal(mut self) -> Result<Self, SpeculativeInferenceError> {
        if self.living_cimage_generation_digest.is_empty() {
            return Err(SpeculativeInferenceError::MissingGeneration);
        }
        if self.target_execution_graph_digest.is_empty() {
            return Err(SpeculativeInferenceError::MissingExecutionGraph);
        }
        self.candidate_digest = self.canonical_digest();
        Ok(self)
    }

    pub fn block_size(&self) -> u32 {
        match self.decode {
            SpeculativeDecodeStrategy::SemiAutoregressiveDspark {
                parallel_block,
                sequential_tail,
                ..
            } => parallel_block + sequential_tail,
            SpeculativeDecodeStrategy::BlockDiffusionDflash { block_size, .. } => block_size,
            SpeculativeDecodeStrategy::DecoupledLongShort { block_size, .. } => block_size,
            SpeculativeDecodeStrategy::AutoregressiveDraft {
                max_draft_tokens, ..
            } => max_draft_tokens,
            SpeculativeDecodeStrategy::VocabularySpeculation { .. }
            | SpeculativeDecodeStrategy::Disabled => 0,
        }
    }
}

impl SpeculativeMeasurement {
    pub fn authoritative(
        &self,
        budget: &SpeculativeSearchBudget,
    ) -> Result<(), SpeculativeInferenceError> {
        if !self.measured
            || self.execution_fingerprint.is_empty()
            || self.receipt_digest.is_empty()
            || !self.ttft_ms.is_finite()
            || !self.tokens_per_second.is_finite()
            || self.ttft_ms <= 0.0
            || self.tokens_per_second <= 0.0
        {
            return Err(SpeculativeInferenceError::NonAuthoritativeMeasurement);
        }
        if !self.quality_equivalent
            || self.verification_waste_ratio > budget.max_verification_waste_ratio
            || self.agentic_success_rate < budget.min_agentic_success_rate
            || self.tool_call_correctness < budget.min_tool_call_correctness
        {
            return Err(SpeculativeInferenceError::QualityGateFailed);
        }
        Ok(())
    }
}

pub fn select_speculative_frontier(
    candidates: &[SpeculativeInferenceCandidate],
    measurements: &[SpeculativeMeasurement],
    budget: &SpeculativeSearchBudget,
    workload_class: &str,
) -> Result<SpeculativeSearchResult, SpeculativeInferenceError> {
    let by_digest: BTreeMap<_, _> = candidates
        .iter()
        .map(|candidate| (&candidate.candidate_digest, candidate))
        .collect();
    let mut accepted = Vec::new();
    let mut rejection_reasons = BTreeMap::new();
    for measurement in measurements {
        let Some(candidate) = by_digest.get(&measurement.candidate_digest) else {
            rejection_reasons
                .entry(measurement.candidate_digest.clone())
                .or_insert_with(Vec::new)
                .push("unknown candidate".into());
            continue;
        };
        if candidate.block_size() > budget.max_block_size {
            rejection_reasons
                .entry(measurement.candidate_digest.clone())
                .or_insert_with(Vec::new)
                .push("block budget exceeded".into());
            continue;
        }
        if let Err(error) = measurement.authoritative(budget) {
            rejection_reasons
                .entry(measurement.candidate_digest.clone())
                .or_insert_with(Vec::new)
                .push(error.to_string());
            continue;
        }
        accepted.push(measurement.clone());
    }
    accepted.sort_by(|a, b| {
        b.tokens_per_second
            .total_cmp(&a.tokens_per_second)
            .then_with(|| a.ttft_ms.total_cmp(&b.ttft_ms))
            .then_with(|| b.accepted_tokens_mean.total_cmp(&a.accepted_tokens_mean))
    });
    let frontier = accepted
        .iter()
        .map(|measurement| measurement.candidate_digest.clone())
        .collect::<Vec<_>>();
    Ok(SpeculativeSearchResult {
        workload_class: workload_class.into(),
        evaluated: accepted,
        selected_candidate_digest: frontier.first().cloned(),
        pareto_frontier: frontier,
        rejection_reasons,
    })
}

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dspark_candidate_seals_and_reports_block_size() {
        let candidate = SpeculativeInferenceCandidate {
            candidate_id: "dspark".into(),
            living_cimage_generation_digest: "generation".into(),
            prefill: SpeculativePrefillStrategy::Disabled,
            decode: SpeculativeDecodeStrategy::SemiAutoregressiveDspark {
                draft_model_digest: "draft".into(),
                parallel_block: 6,
                sequential_tail: 2,
                confidence_schedule_digest: "schedule".into(),
            },
            verification: VerificationSchedule::ConfidenceScheduled {
                min_tokens: 2,
                max_tokens: 8,
                survival_model_digest: "survival".into(),
                throughput_profile_digest: "throughput".into(),
            },
            target_execution_graph_digest: "graph".into(),
            kv_policy_digest: "kv".into(),
            workload_class: "agentic-code".into(),
            expected_acceptance_length: None,
            expected_ttft_ms: None,
            expected_tokens_per_second: None,
            candidate_digest: String::new(),
        }
        .seal()
        .unwrap();
        assert_eq!(candidate.block_size(), 8);
        assert!(!candidate.candidate_digest.is_empty());
    }

    #[test]
    fn non_authoritative_measurement_is_rejected() {
        let measurement = SpeculativeMeasurement {
            candidate_digest: "x".into(),
            measured: false,
            execution_fingerprint: String::new(),
            workload_digest: "w".into(),
            ttft_ms: 1.0,
            tokens_per_second: 1.0,
            accepted_tokens_mean: 1.0,
            verification_waste_ratio: 0.0,
            draft_latency_ms: 0.1,
            verification_latency_ms: 0.9,
            quality_equivalent: true,
            agentic_success_rate: 1.0,
            tool_call_correctness: 1.0,
            receipt_digest: String::new(),
        };
        let budget = SpeculativeSearchBudget {
            max_candidates: 10,
            max_draft_models: 3,
            max_block_size: 16,
            max_verification_waste_ratio: 0.3,
            min_agentic_success_rate: 0.95,
            min_tool_call_correctness: 0.99,
        };
        assert!(measurement.authoritative(&budget).is_err());
    }
}
