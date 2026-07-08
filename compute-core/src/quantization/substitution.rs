//! Generalized substitution pipeline — tries ranked codec candidates against
//! evidence gates and uses the most aggressive one that passes.
//!
//! Each codec (FP16, INT8, NF4, SymInt4, Ternary) is a first-class
//! substitution candidate with its own gate profile. The pipeline tries
//! them in order and returns the first that passes all gates.
//!
//! # Protocol
//!
//! 1. Resolve the primary codec from the compiler policy.
//! 2. Build the substitution candidate list from the policy config.
//! 3. Try each candidate in order (most aggressive first):
//!    a. Pack weights with the candidate codec.
//!    b. Evaluate weight-space gate (NRMSE, zero-collapse).
//!    c. If passes: evaluate operator gate (CPU or ANE matmul).
//!    d. If operator fails: try residual rescue (restore failing columns).
//!    e. If rescue passes or operator passes: mark substituted.
//! 4. If no candidate passes: keep the primary codec.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ternary_substitution::ResidualRescueConfig;

// ── Substitution candidate ────────────────────────────────────────────────

/// A codec that can substitute for the primary, with gating thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionCandidate {
    /// Codec name (e.g. "FP16", "INT8", "NF4", "SymInt4", "Ternary").
    pub name: String,
    /// Codec-specific parameters (varies by family).
    pub parameters: HashMap<String, serde_json::Value>,
    /// Gate thresholds for this candidate.
    pub gates: SubstitutionGates,
    /// Whether this candidate requires activation-weighted optimization.
    pub requires_activation_profile: bool,
}

/// Gate thresholds for a substitution candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionGates {
    /// Weight-space: maximum NRMSE (None = no weight gate).
    pub weight_nrmse_max: Option<f64>,
    /// Weight-space: maximum zero-collapse ratio.
    pub weight_zero_collapse_max: Option<f64>,
    /// Operator: maximum NRMSE (None = no operator gate).
    pub operator_nrmse_max: Option<f64>,
    /// Operator: minimum cosine similarity.
    pub operator_cosine_min: Option<f64>,
    /// Operator: maximum absolute error per output element.
    pub operator_max_abs_max: Option<f64>,
    /// Whether this candidate needs ANE (or hardware) validation vs CPU-only.
    pub requires_hardware_validation: bool,
    /// Whether rollout validation is required for this candidate.
    pub requires_rollout_validation: bool,
}

impl Default for SubstitutionGates {
    fn default() -> Self {
        Self {
            weight_nrmse_max: None,
            weight_zero_collapse_max: None,
            operator_nrmse_max: None,
            operator_cosine_min: None,
            operator_max_abs_max: None,
            requires_hardware_validation: false,
            requires_rollout_validation: false,
        }
    }
}

// ── Substitution attempt result ───────────────────────────────────────────

/// Result of trying one substitution candidate on a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionAttempt {
    /// Candidate codec name.
    pub candidate: String,
    /// Weight-space evidence, if evaluated.
    pub weight_evidence: Option<SubstitutionEvidence>,
    /// Operator evidence, if evaluated.
    pub operator_evidence: Option<SubstitutionEvidence>,
    /// Residual rescue result, if applicable.
    pub rescue_result: Option<ResidualRescueResult>,
    /// Outcome of this attempt.
    pub outcome: SubstitutionOutcome,
    /// Byte savings vs primary codec.
    pub bytes_saved: u64,
    /// Primary codec bytes (for comparison).
    pub primary_bytes: u64,
}

/// Single tier of substitution evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionEvidence {
    /// Evidence tier.
    pub tier: EvidenceTier,
    /// Whether evaluated.
    pub evaluated: bool,
    /// Whether gates passed.
    pub passed: bool,
    /// Numeric metrics.
    pub metrics: HashMap<String, f64>,
    /// Error if evaluation failed.
    pub error: Option<String>,
}

/// Evidence tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceTier {
    WeightSpace,
    Operator,
    Rollout,
}

/// Outcome of a substitution attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubstitutionOutcome {
    /// Candidate passed all gates — fully substituted.
    Substituted,
    /// Candidate needed residual rescue — substituted with rescue.
    SubstitutedWithRescue,
    /// Candidate failed a gate that could not be rescued.
    Rejected,
    /// Candidate was not attempted.
    NotAttempted,
}

// ── Residual rescue ───────────────────────────────────────────────────────

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

// ── Gate profiles for each codec ──────────────────────────────────────────

/// Gate profiles for default substitution candidates.
impl SubstitutionCandidate {
    /// FP16: ~2x savings over F32, essentially lossless.
    /// No operator validation needed — FP16 is native ANE format.
    pub fn fp16() -> Self {
        Self {
            name: "FP16".into(),
            parameters: HashMap::new(),
            gates: SubstitutionGates {
                weight_nrmse_max: Some(0.001), // Tight — FP16 should be nearly exact
                weight_zero_collapse_max: Some(0.0),
                operator_nrmse_max: None,      // No operator gate — FP16 is lossless
                operator_cosine_min: None,
                operator_max_abs_max: None,
                requires_hardware_validation: false,
                requires_rollout_validation: false,
            },
            requires_activation_profile: false,
        }
    }

    /// INT8 g128: ~4x savings over F32.
    pub fn int8_g128() -> Self {
        Self {
            name: "INT8".into(),
            parameters: [("group_size".into(), serde_json::json!(128))]
                .into_iter().collect(),
            gates: SubstitutionGates {
                weight_nrmse_max: Some(0.01),
                weight_zero_collapse_max: Some(0.0),
                operator_nrmse_max: Some(0.002),
                operator_cosine_min: Some(0.999),
                operator_max_abs_max: Some(0.5),
                requires_hardware_validation: false,
                requires_rollout_validation: false,
            },
            requires_activation_profile: false,
        }
    }

    /// NF4 BitsAndBytes g32 ScaleOnly: ~8x savings over F32.
    pub fn nf4_bnb_g32() -> Self {
        Self {
            name: "NF4".into(),
            parameters: [
                ("codebook".into(), serde_json::json!("BitsAndBytesNf4")),
                ("group_size".into(), serde_json::json!(32)),
                ("affine_mode".into(), serde_json::json!("ScaleOnly")),
            ].into_iter().collect(),
            gates: SubstitutionGates {
                weight_nrmse_max: Some(0.10),
                weight_zero_collapse_max: Some(0.001),
                operator_nrmse_max: Some(0.002),
                operator_cosine_min: Some(0.999),
                operator_max_abs_max: Some(0.5),
                requires_hardware_validation: true,  // ANE validation recommended
                requires_rollout_validation: false,
            },
            requires_activation_profile: false,
        }
    }

    /// SymInt4 g32: ~8x savings over F32.
    pub fn sym_int4_g32() -> Self {
        Self {
            name: "SymInt4".into(),
            parameters: [
                ("group_size".into(), serde_json::json!(32)),
            ].into_iter().collect(),
            gates: SubstitutionGates {
                weight_nrmse_max: Some(0.10),
                weight_zero_collapse_max: Some(0.001),
                operator_nrmse_max: Some(0.002),
                operator_cosine_min: Some(0.999),
                operator_max_abs_max: Some(0.5),
                requires_hardware_validation: true,
                requires_rollout_validation: false,
            },
            requires_activation_profile: false,
        }
    }

    /// Ternary: ~16x savings over F32, strictest gates.
    pub fn ternary() -> Self {
        Self {
            name: "Ternary".into(),
            parameters: [
                ("sparsity_target".into(), serde_json::json!(0.5)),
                ("group_size".into(), serde_json::json!(256)),
            ].into_iter().collect(),
            gates: SubstitutionGates {
                weight_nrmse_max: Some(0.020),
                weight_zero_collapse_max: Some(0.85),
                operator_nrmse_max: Some(0.005),
                operator_cosine_min: Some(0.999),
                operator_max_abs_max: Some(2.0),
                requires_hardware_validation: true,
                requires_rollout_validation: true,
            },
            requires_activation_profile: false,
        }
    }
}

// ── Substitution config (from policy) ─────────────────────────────────────

/// Pipeline configuration for substitution candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstitutionPipelineConfig {
    /// Whether substitution is enabled.
    pub enabled: bool,
    /// Ordered list of candidates to try (most aggressive first).
    pub candidates: Vec<String>,
    /// Per-codec gate overrides (optional).
    pub gate_overrides: HashMap<String, SubstitutionGates>,
    /// Residual rescue config.
    pub rescue: ResidualRescueConfig,
}

impl Default for SubstitutionPipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            candidates: vec![
                "Ternary".into(),
                "SymInt4".into(),
                "NF4".into(),
                "INT8".into(),
                "FP16".into(),
            ],
            gate_overrides: HashMap::new(),
            rescue: ResidualRescueConfig::default(),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_profiles() {
        let f = SubstitutionCandidate::fp16();
        assert_eq!(f.name, "FP16");
        assert!(f.gates.weight_nrmse_max.unwrap() < 0.01);

        let t = SubstitutionCandidate::ternary();
        assert!(t.gates.weight_zero_collapse_max.unwrap() > 0.5);
        assert!(t.gates.requires_rollout_validation);
    }

    #[test]
    fn test_pipeline_config_default() {
        let cfg = SubstitutionPipelineConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.candidates.len(), 5);
        assert_eq!(cfg.candidates[0], "Ternary");
        assert_eq!(cfg.candidates[4], "FP16");
    }

    #[test]
    fn test_substitution_attempt_serde() {
        let attempt = SubstitutionAttempt {
            candidate: "FP16".into(),
            weight_evidence: Some(SubstitutionEvidence {
                tier: EvidenceTier::WeightSpace,
                evaluated: true,
                passed: true,
                metrics: [("nrmse".into(), 0.0003)].into_iter().collect(),
                error: None,
            }),
            operator_evidence: None,
            rescue_result: None,
            outcome: SubstitutionOutcome::Substituted,
            bytes_saved: 50000000,
            primary_bytes: 100000000,
        };
        let json = serde_json::to_string(&attempt).unwrap();
        let decoded: SubstitutionAttempt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.candidate, "FP16");
    }
}
