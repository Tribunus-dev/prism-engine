//! Ternary substitution — first-class compiler pass for replacing primary codecs
//! with ternary-quantized weights when evidence gates are satisfied.
//!
//! Ternary is not a replacement for the primary codec across all tensors.
//! It is a **compiler optimization pass** that, for eligible tensor classes,
//! attempts to substitute ternary with a fallback path: if the ternary
//! candidate fails rollout gates, failing columns are restored to the primary
//! codec precision via residual rescue.
//!
//! # Architecture
//!
//! The substitution protocol is differential:
//!
//! 1. The tensor already has a valid primary codec (NF4 or INT8).
//! 2. The compiler tries ternary quantization on the same weights.
//! 3. If weight-space gates pass -> request ANE operator validation.
//! 4. If operator gates pass -> request rollout (logit/feature) validation.
//! 5. If rollout gates pass -> mark `ternary_substituted` in the cimage manifest.
//! 6. If rollout fails -> try residual rescue (restore failing columns to primary).
//! 7. If rescue passes -> mark `ternary_substituted_with_rescue`.
//! 8. If rescue fails -> keep the primary codec.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Matcher for eligible tensor classes ───────────────────────────────────

/// Matcher for ternary-eligible tensors — combines a TensorClass with an
/// optional tensor-key suffix pattern. This lets o_proj be eligible within
/// DecoderAttentionProjection while q_proj/k_proj/v_proj are not.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryEligibleMatcher {
    /// Tensor class name (must match a TensorClass variant name).
    pub class: String,
    /// Optional name suffix for the tensor key (e.g. "o_proj" matches
    /// ...self_attn.o_proj.weight).  If None, all tensors of this class
    /// are eligible.
    pub name_suffix: Option<String>,
}

// ── Ternary substitution config (from policy) ──────────────────────────────

/// Configuration for ternary substitution, sourced from `compiler_policy.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernarySubstitutionConfig {
    /// Whether ternary substitution is enabled at all.
    pub enabled: bool,
    /// Tensor classes eligible for ternary substitution (legacy, kept for
    /// backward compat).
    pub eligible_classes: Vec<String>,
    /// Eligible classes with optional name suffix matching (more precise
    /// than TensorClass alone).
    pub eligible_matchers: Vec<TernaryEligibleMatcher>,
    /// Gate thresholds for each evidence tier.
    pub gates: TernaryGates,
    /// Residual rescue configuration for columns that fail ternary.
    pub rescue: ResidualRescueConfig,
}

/// Gate thresholds for ternary substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryGates {
    /// Weight-space: maximum NRMSE for ternary.
    pub weight_nrmse_max: f64,
    /// Weight-space: maximum zero-collapse ratio (0.0–1.0).
    pub weight_zero_collapse_max: f64,
    /// Operator: maximum NRMSE for ternary output vs float reference.
    pub operator_nrmse_max: f64,
    /// Operator: minimum cosine similarity.
    pub operator_cosine_min: f64,
    /// Operator: maximum absolute error per output element.
    pub operator_max_abs_max: f64,
    /// Rollout: maximum logit drift (normalized).
    pub rollout_logit_drift_max: f64,
    /// Rollout: minimum worst-token cosine similarity.
    pub rollout_worst_token_cosine_min: f64,
}

impl Default for TernaryGates {
    fn default() -> Self {
        Self {
            weight_nrmse_max: 0.020,
            weight_zero_collapse_max: 0.85,
            operator_nrmse_max: 0.005,
            operator_cosine_min: 0.999,
            operator_max_abs_max: 2.0,
            rollout_logit_drift_max: 0.01,
            rollout_worst_token_cosine_min: 0.99,
        }
    }
}

/// Residual rescue config — when ternary fails rollout, restore failing
/// dimensions to the primary codec precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualRescueConfig {
    /// Rescue granularity (InputColumn, OutputRow, Tile, Group).
    pub granularity: RescueGranularity,
    /// Rescue format (F32Delta, F16Delta, PrimaryCodec).
    pub format: RescueFormat,
    /// Rescue selector (ActivationWeightedContribution, MaxWeightError).
    pub selector: RescueSelector,
    /// Maximum percentage of elements that can be rescued before the
    /// compression benefit is eliminated. Default 25%.
    pub max_rescue_fraction: f64,
    /// Gate for ternary-plus-rescue: max_abs_error must be below this.
    pub max_abs_gate_after_rescue: f64,
}

impl Default for ResidualRescueConfig {
    fn default() -> Self {
        Self {
            granularity: RescueGranularity::InputColumn,
            format: RescueFormat::PrimaryCodec,
            selector: RescueSelector::ActivationWeightedContribution,
            max_rescue_fraction: 0.25,
            max_abs_gate_after_rescue: 0.5,
        }
    }
}

/// Axis along which rescue operates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueGranularity {
    InputColumn,
    OutputRow,
    Tile,
    Group,
}

/// Storage format for rescued elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueFormat {
    F32Delta,
    F16Delta,
    PrimaryCodec,
}

/// How to select which elements to rescue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueSelector {
    MaxWeightError,
    ActivationWeightedContribution,
    OperatorMaxAbsGate,
}

// ── Substitution evidence types ────────────────────────────────────────────

/// Tiered evidence for a ternary substitution attempt on one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernarySubstitutionEvidence {
    /// Which tier gates this evidence covers.
    pub tier: EvidenceTier,
    /// Whether this evidence tier was evaluated.
    pub evaluated: bool,
    /// Whether this tier's gates were passed.
    pub passed: bool,
    /// Numeric metrics for this tier.
    pub metrics: HashMap<String, f64>,
    /// Error message if evaluation failed.
    pub error: Option<String>,
}

/// Evidence tier identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceTier {
    WeightSpace,
    Operator,
    Rollout,
}

/// Full evidence receipt for a ternary substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernarySubstitutionReceipt {
    /// Tensor key this substitution applies to.
    pub tensor_key: String,
    /// Tensor class.
    pub tensor_class: String,
    /// Primary codec family that was substituted.
    pub primary_family: String,
    /// Ternary parameters used.
    pub ternary_parameters: HashMap<String, serde_json::Value>,
    /// Evidence for each tier.
    pub evidence: Vec<TernarySubstitutionEvidence>,
    /// Final substitution outcome.
    pub outcome: SubstitutionOutcome,
    /// Residual rescue result, if applicable.
    pub rescue_result: Option<ResidualRescueResult>,
    /// Byte savings vs primary codec.
    pub bytes_saved: u64,
    /// Primary codec bytes (for comparison).
    pub primary_bytes: u64,
}

/// Outcome of a ternary substitution attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubstitutionOutcome {
    Substituted,
    SubstitutedWithRescue,
    Rejected,
    NotEligible,
}

/// Result of a residual rescue attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualRescueResult {
    pub rescued_count: usize,
    pub total_elements: usize,
    pub rescue_fraction: f64,
    pub max_abs_after_rescue: f64,
    pub gate_passed: bool,
    pub rescue_bytes: u64,
}

// ── Policy integration type ───────────────────────────────────────────────

/// Policy section for ternary substitution, embedded in compiler_policy.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryPolicySection {
    pub config: TernarySubstitutionConfig,
    pub default_stale: String,
    pub model_policies: Vec<TernaryModelPolicy>,
}

/// Per-model ternary substitution policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryModelPolicy {
    pub model_family: String,
    pub eligible_classes: Vec<String>,
    pub gates: TernaryGates,
    pub rescue: ResidualRescueConfig,
}

// ── Default config ────────────────────────────────────────────────────────

impl Default for TernarySubstitutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            eligible_classes: vec![
                "DecoderAttentionProjection".to_string(),
                "TokenEmbedding".to_string(),
            ],
            eligible_matchers: vec![
                TernaryEligibleMatcher {
                    class: "DecoderAttentionProjection".into(),
                    name_suffix: Some("o_proj".into()),
                },
                TernaryEligibleMatcher {
                    class: "TokenEmbedding".into(),
                    name_suffix: None,
                },
            ],
            gates: TernaryGates::default(),
            rescue: ResidualRescueConfig::default(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_config_default_roundtrip() {
        let config = TernarySubstitutionConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let decoded: TernarySubstitutionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.enabled, decoded.enabled);
        assert_eq!(config.eligible_matchers.len(), decoded.eligible_matchers.len());
        assert_eq!(config.gates.weight_nrmse_max, decoded.gates.weight_nrmse_max);
    }

    #[test]
    fn test_rescue_config_default() {
        let rescue = ResidualRescueConfig::default();
        assert_eq!(rescue.max_rescue_fraction, 0.25);
        assert_eq!(rescue.granularity, RescueGranularity::InputColumn);
        assert_eq!(rescue.selector, RescueSelector::ActivationWeightedContribution);
    }

    #[test]
    fn test_receipt_serialization() {
        let receipt = TernarySubstitutionReceipt {
            tensor_key: "test.weight".into(),
            tensor_class: "DecoderAttentionProjection".into(),
            primary_family: "NF4".into(),
            ternary_parameters: HashMap::new(),
            evidence: vec![],
            outcome: SubstitutionOutcome::NotEligible,
            rescue_result: None,
            bytes_saved: 0,
            primary_bytes: 1000,
        };
        let json = serde_json::to_string(&receipt).unwrap();
        let decoded: TernarySubstitutionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.tensor_key, "test.weight");
    }

    #[test]
    fn test_matcher_roundtrip() {
        let m = TernaryEligibleMatcher {
            class: "DecoderAttentionProjection".into(),
            name_suffix: Some("o_proj".into()),
        };
        let json = serde_json::to_string(&m).unwrap();
        let decoded: TernaryEligibleMatcher = serde_json::from_str(&json).unwrap();
        assert_eq!(m.class, decoded.class);
        assert_eq!(m.name_suffix, decoded.name_suffix);
    }
}
