//! Graph-safe channel equalization for NF4.
//!
//! Scale migration: for a projection Y = XW, a diagonal scale D transforms
//! X' = XD and W' = D^{-1}W, preserving Y = X'W' = XDD^{-1}W = XW.
//! This is only valid at explicit linear boundaries where the adjacent
//! operations are also linear (not RMSNorm, RoPE, attention softmax, etc.).

use serde::{Deserialize, Serialize};

/// Whether a PhaseIR boundary is legal for scale migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryLegality {
    /// Legal: can safely absorb scales across this boundary.
    Legal,
    /// Illegal: non-linear operation blocks migration.
    Illegal { reason: &'static str },
    /// Conditional: legal only with manifest recording.
    Conditional { requires_inverse: bool },
}

/// A recorded scale migration operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleMigrationRecord {
    /// Source tensor name (the weights being transformed).
    pub source_tensor: String,
    /// Target tensor name (the adjacent weights receiving the inverse).
    pub target_tensor: Option<String>,
    /// The diagonal scale vector D used for migration.
    pub scale_diagonal: Vec<f32>,
    /// Inverse scale vector D^{-1} (element-wise reciprocal).
    pub inverse_diagonal: Vec<f32>,
    /// Whether the migration was actually applied.
    pub applied: bool,
}

/// Check if a pair of adjacent phase types can safely absorb scale migration.
pub fn is_legal_boundary(producer_phase_type: &str, consumer_phase_type: &str) -> BoundaryLegality {
    // Linear operations: safe to absorb scales
    let linear = [
        "QkvProjection",
        "OutputProjection",
        "FfnGate",
        "FfnUp",
        "FfnDown",
        "LoadTeacherRegion",
        "LoadStudentCandidate",
    ];

    // Non-linear operations: unsafe
    let nonlinear = [
        "AttentionSoftmax",
        "RmsNorm",
        "RoPE",
        "SiLU",
        "GELU",
        "ResidualAdd",
        "CausalConvolution",
        "SpatialPatchEmbedding",
        "GridAttention2D",
        "AdaptiveLayerNorm",
        "TimeStepEmbedding",
    ];

    let prod_is_linear = linear.iter().any(|l| producer_phase_type.contains(l));
    let cons_is_nonlinear = nonlinear.iter().any(|n| consumer_phase_type.contains(n));

    if prod_is_linear && !cons_is_nonlinear {
        BoundaryLegality::Conditional {
            requires_inverse: true,
        }
    } else if cons_is_nonlinear {
        BoundaryLegality::Illegal {
            reason: "consumer is non-linear",
        }
    } else {
        BoundaryLegality::Illegal {
            reason: "unsupported phase type pair",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_to_linear_is_legal() {
        let result = is_legal_boundary("QkvProjection", "OutputProjection");
        assert_eq!(
            result,
            BoundaryLegality::Conditional {
                requires_inverse: true
            }
        );
    }

    #[test]
    fn test_linear_to_nonlinear_is_illegal() {
        let result = is_legal_boundary("QkvProjection", "RmsNorm");
        assert_eq!(
            result,
            BoundaryLegality::Illegal {
                reason: "consumer is non-linear"
            }
        );
    }

    #[test]
    fn test_scale_migration_record_roundtrip() {
        let record = ScaleMigrationRecord {
            source_tensor: "layer0.q_proj.weight".into(),
            target_tensor: Some("layer0.input_layernorm.weight".into()),
            scale_diagonal: vec![1.0, 2.0, 3.0],
            inverse_diagonal: vec![1.0, 0.5, 0.333],
            applied: false,
        };
        assert_eq!(record.source_tensor, "layer0.q_proj.weight");
        assert!(!record.applied);
    }
}
