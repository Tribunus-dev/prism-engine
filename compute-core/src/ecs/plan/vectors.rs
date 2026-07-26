//! Model reference vectors — deterministic capture and comparison of
//! RawF32 reference output vs quantized cimage output.

use serde::{Deserialize, Serialize};

/// One log-probability entry for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobEntry {
    pub token: u32,
    pub logprob: f64,
}

/// Hidden state checkpoint at a specific token position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateCheckpoint {
    pub layer_index: u32,
    pub token_index: u32,
    pub nrmse_vs_reference: Option<f64>,
    pub max_abs_vs_reference: Option<f64>,
    pub cosine_vs_reference: Option<f64>,
}

/// A complete reference vector captured from a trusted source (RawF32 or BF16).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReferenceVector {
    pub vector_id: String,
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub prompt_tokens: Vec<u32>,
    pub prefill_chunk_size: usize,
    pub expected_greedy_tokens: Vec<u32>,
    pub logits_topk: Vec<Vec<LogprobEntry>>,
    pub hidden_checkpoints: Vec<HiddenStateCheckpoint>,
}

/// Acceptance gates for model equivalence decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceGates {
    /// Minimum top-1 agreement ratio (0.0 to 1.0).
    pub min_top1_agreement: f64,
    /// Minimum top-5 agreement ratio.
    pub min_top5_agreement: f64,
    /// Maximum acceptable logits KL divergence.
    pub max_logits_kl: f64,
    /// Maximum acceptable hidden state NRMSE.
    pub max_hidden_nrmse: f64,
    /// Maximum acceptable token generation divergence.
    pub max_divergent_tokens: usize,
}

impl AcceptanceGates {
    /// Default production quality gates.
    pub fn production_default() -> Self {
        Self {
            min_top1_agreement: 0.95,
            min_top5_agreement: 0.99,
            max_logits_kl: 0.008,
            max_hidden_nrmse: 0.005,
            max_divergent_tokens: 0,
        }
    }
    /// Relaxed gates for research/exploratory profiles.
    pub fn research_default() -> Self {
        Self {
            min_top1_agreement: 0.85,
            min_top5_agreement: 0.95,
            max_logits_kl: 0.05,
            max_hidden_nrmse: 0.02,
            max_divergent_tokens: 3,
        }
    }
}

/// Drift status from comparing a candidate against a baseline reference vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftStatus {
    WithinGates,
    LogitsKlExceeded { value: f64, gate: f64 },
    HiddenNrmseExceeded { value: f64, gate: f64 },
    Top1AgreementFailed { value: f64, gate: f64 },
    TokenDivergenceExceeded { count: usize, gate: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_gates() {
        let gates = AcceptanceGates::production_default();
        assert_eq!(gates.min_top1_agreement, 0.95);
        assert_eq!(gates.min_top5_agreement, 0.99);
        assert_eq!(gates.max_logits_kl, 0.008);
        assert_eq!(gates.max_hidden_nrmse, 0.005);
        assert_eq!(gates.max_divergent_tokens, 0);
    }

    #[test]
    fn test_research_gates() {
        let gates = AcceptanceGates::research_default();
        assert_eq!(gates.min_top1_agreement, 0.85);
        assert_eq!(gates.min_top5_agreement, 0.95);
        assert_eq!(gates.max_logits_kl, 0.05);
        assert_eq!(gates.max_hidden_nrmse, 0.02);
        assert_eq!(gates.max_divergent_tokens, 3);
    }

    #[test]
    fn test_drift_status_serialization() {
        let cases = [
            DriftStatus::WithinGates,
            DriftStatus::LogitsKlExceeded {
                value: 0.012,
                gate: 0.008,
            },
            DriftStatus::HiddenNrmseExceeded {
                value: 0.01,
                gate: 0.005,
            },
            DriftStatus::Top1AgreementFailed {
                value: 0.90,
                gate: 0.95,
            },
            DriftStatus::TokenDivergenceExceeded { count: 5, gate: 0 },
        ];
        for original in &cases {
            let json = serde_json::to_string(original).expect("serialize");
            let recovered: DriftStatus = serde_json::from_str(&json).expect("deserialize");
            // Roundtrip: re-serialize and compare strings.
            let json2 = serde_json::to_string(&recovered).expect("re-serialize");
            assert_eq!(json, json2, "roundtrip mismatch");
        }
    }
}
