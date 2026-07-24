//! Evolutionary KV-cache compression for Living CImage generations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCompressionFamily {
    FullPrecision,
    Quantized {
        key_bits: u8,
        value_bits: u8,
        group_size: u32,
        residual_error_correction: bool,
    },
    HeavyHitter {
        recent_window: u32,
        heavy_hitter_budget: u32,
    },
    SnapshotSelection {
        observation_window: u32,
        retained_tokens: u32,
    },
    StreamingWindow {
        sink_tokens: u32,
        recent_tokens: u32,
    },
    TokenValueAware {
        retained_tokens: u32,
        decay_bps: u32,
    },
    KeyNorm {
        retained_tokens: u32,
    },
    HybridHeadPolicy {
        static_head_policy_digest: String,
        dynamic_head_policy_digest: String,
        total_budget_tokens: u32,
    },
    ThreeRing {
        recent_tokens: u32,
        heavy_hitter_tokens: u32,
        overview_tokens: u32,
    },
    PolarQuantized {
        bits: u8,
        qjl_error_bits: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvBudgetGranularity {
    Global,
    Layer,
    Head,
    SemanticRegion,
    Modality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheCompressionCandidate {
    pub candidate_id: String,
    pub living_cimage_generation_digest: String,
    pub family: KvCompressionFamily,
    pub granularity: KvBudgetGranularity,
    pub semantic_region_plan_digest: String,
    pub speculative_candidate_digest: Option<String>,
    pub target_execution_graph_digest: String,
    pub agentic_workload_class: String,
    pub memory_budget_bytes: u64,
    pub candidate_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvInstructionRetentionProbe {
    pub instruction_id: String,
    pub instruction_position: u32,
    pub retained: bool,
    pub leakage_detected: bool,
    pub compliance_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheMeasurement {
    pub candidate_digest: String,
    pub measured: bool,
    pub execution_fingerprint: String,
    pub workload_digest: String,
    pub peak_cache_bytes: u64,
    pub compression_ratio: f64,
    pub decode_tokens_per_second: f64,
    pub latency_ms: f64,
    pub long_context_recall: f64,
    pub agentic_success_rate: f64,
    pub tool_call_correctness: f64,
    pub instruction_retention: Vec<KvInstructionRetentionProbe>,
    pub multimodal_quality: Option<f64>,
    pub receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEvolutionBudget {
    pub max_candidates: usize,
    pub max_peak_cache_bytes: u64,
    pub min_long_context_recall: f64,
    pub min_agentic_success_rate: f64,
    pub min_tool_call_correctness: f64,
    pub min_instruction_compliance: f64,
    pub forbid_instruction_leakage: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEvolutionResult {
    pub workload_class: String,
    pub accepted: Vec<KvCacheMeasurement>,
    pub selected_candidate_digest: Option<String>,
    pub pareto_frontier: Vec<String>,
    pub rejection_reasons: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Error, PartialEq)]
pub enum KvEvolutionError {
    #[error("candidate references an empty Living CImage generation")]
    MissingGeneration,
    #[error("candidate references an empty execution graph")]
    MissingExecutionGraph,
    #[error("measurement is not authoritative")]
    NonAuthoritativeMeasurement,
    #[error("cache budget exceeded")]
    MemoryBudgetExceeded,
    #[error("behavioral quality gate failed")]
    BehavioralGateFailed,
    #[error("instruction retention or leakage gate failed")]
    InstructionSafetyFailed,
}

impl KvCacheCompressionCandidate {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.candidate_digest.clear();
        digest(&serde_json::to_vec(&canonical).expect("kv candidate serialization"))
    }

    pub fn seal(mut self) -> Result<Self, KvEvolutionError> {
        if self.living_cimage_generation_digest.is_empty() {
            return Err(KvEvolutionError::MissingGeneration);
        }
        if self.target_execution_graph_digest.is_empty() {
            return Err(KvEvolutionError::MissingExecutionGraph);
        }
        self.candidate_digest = self.canonical_digest();
        Ok(self)
    }
}

impl KvCacheMeasurement {
    pub fn authoritative(&self, budget: &KvEvolutionBudget) -> Result<(), KvEvolutionError> {
        if !self.measured
            || self.execution_fingerprint.is_empty()
            || self.receipt_digest.is_empty()
            || self.decode_tokens_per_second <= 0.0
            || !self.decode_tokens_per_second.is_finite()
            || self.latency_ms <= 0.0
            || !self.latency_ms.is_finite()
            || self.compression_ratio < 1.0
        {
            return Err(KvEvolutionError::NonAuthoritativeMeasurement);
        }
        if self.peak_cache_bytes > budget.max_peak_cache_bytes {
            return Err(KvEvolutionError::MemoryBudgetExceeded);
        }
        if self.long_context_recall < budget.min_long_context_recall
            || self.agentic_success_rate < budget.min_agentic_success_rate
            || self.tool_call_correctness < budget.min_tool_call_correctness
        {
            return Err(KvEvolutionError::BehavioralGateFailed);
        }
        if self.instruction_retention.iter().any(|probe| {
            !probe.retained
                || probe.compliance_score < budget.min_instruction_compliance
                || (budget.forbid_instruction_leakage && probe.leakage_detected)
        }) {
            return Err(KvEvolutionError::InstructionSafetyFailed);
        }
        Ok(())
    }
}

pub fn select_kv_frontier(
    candidates: &[KvCacheCompressionCandidate],
    measurements: &[KvCacheMeasurement],
    budget: &KvEvolutionBudget,
    workload_class: &str,
) -> KvEvolutionResult {
    let candidate_ids = candidates
        .iter()
        .map(|candidate| candidate.candidate_digest.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let mut accepted = Vec::new();
    let mut rejection_reasons = BTreeMap::new();
    for measurement in measurements {
        if !candidate_ids.contains(&measurement.candidate_digest) {
            rejection_reasons
                .entry(measurement.candidate_digest.clone())
                .or_insert_with(Vec::new)
                .push("unknown candidate".into());
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
        b.decode_tokens_per_second
            .total_cmp(&a.decode_tokens_per_second)
            .then_with(|| a.peak_cache_bytes.cmp(&b.peak_cache_bytes))
            .then_with(|| b.long_context_recall.total_cmp(&a.long_context_recall))
    });
    let frontier = accepted
        .iter()
        .map(|measurement| measurement.candidate_digest.clone())
        .collect::<Vec<_>>();
    KvEvolutionResult {
        workload_class: workload_class.into(),
        selected_candidate_digest: frontier.first().cloned(),
        pareto_frontier: frontier,
        accepted,
        rejection_reasons,
    }
}

pub type KvCacheCandidate = KvCacheCompressionCandidate;
pub type KvCacheEvaluationReceipt = KvCacheMeasurement;

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_leakage_blocks_promotion() {
        let measurement = KvCacheMeasurement {
            candidate_digest: "candidate".into(),
            measured: true,
            execution_fingerprint: "run".into(),
            workload_digest: "work".into(),
            peak_cache_bytes: 1024,
            compression_ratio: 4.0,
            decode_tokens_per_second: 20.0,
            latency_ms: 50.0,
            long_context_recall: 0.99,
            agentic_success_rate: 0.99,
            tool_call_correctness: 1.0,
            instruction_retention: vec![KvInstructionRetentionProbe {
                instruction_id: "system".into(),
                instruction_position: 0,
                retained: true,
                leakage_detected: true,
                compliance_score: 1.0,
            }],
            multimodal_quality: None,
            receipt_digest: "receipt".into(),
        };
        let budget = KvEvolutionBudget {
            max_candidates: 10,
            max_peak_cache_bytes: 2048,
            min_long_context_recall: 0.95,
            min_agentic_success_rate: 0.95,
            min_tool_call_correctness: 0.99,
            min_instruction_compliance: 0.95,
            forbid_instruction_leakage: true,
        };
        assert_eq!(
            measurement.authoritative(&budget),
            Err(KvEvolutionError::InstructionSafetyFailed)
        );
    }

    #[test]
    fn polar_quantized_candidate_is_sealed() {
        let candidate = KvCacheCompressionCandidate {
            candidate_id: "polar".into(),
            living_cimage_generation_digest: "generation".into(),
            family: KvCompressionFamily::PolarQuantized {
                bits: 3,
                qjl_error_bits: 1,
            },
            granularity: KvBudgetGranularity::Head,
            semantic_region_plan_digest: "regions".into(),
            speculative_candidate_digest: None,
            target_execution_graph_digest: "graph".into(),
            agentic_workload_class: "research".into(),
            memory_budget_bytes: 4096,
            candidate_digest: String::new(),
        }
        .seal()
        .unwrap();
        assert!(!candidate.candidate_digest.is_empty());
    }
}
