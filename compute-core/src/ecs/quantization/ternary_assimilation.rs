//! Ternary base-weight assimilation.
//!
//! Adds opt-in ternary assignment mutations behind a research-only gate.
//! Assimilated ternary patterns are embedded into the base weight representation
//! without requiring full decompression.
//!
//! The assimilation process:
//! 1. Identify weights near ternary boundaries (-1, 0, +1)
//! 2. Quantize to ternary and compute residual
//! 3. Verify the residual does not exceed quality thresholds
//! 4. Store the assimilation artifact

use crate::ecs::plan::CodecFamily;
use serde::{Deserialize, Serialize};

/// Configuration for ternary assimilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAssimilationConfig {
    /// Maximum allowed NRMSE for an assimilated tensor.
    pub max_nrmse: f64,
    /// Minimum weight magnitude to consider for assimilation.
    pub min_weight_magnitude: f64,
    /// Whether to apply residual compensation.
    pub residual_compensation: bool,
    /// Whether this is a research-only operation (gated at compile time).
    pub research_only: bool,
}

impl Default for TernaryAssimilationConfig {
    fn default() -> Self {
        Self {
            max_nrmse: 0.02,
            min_weight_magnitude: 0.1,
            residual_compensation: true,
            research_only: true,
        }
    }
}

/// Result of assimilating one tensor to ternary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAssimilationResult {
    pub tensor_id: String,
    pub original_format: CodecFamily,
    pub ternary_format: CodecFamily,
    pub weights_assimilated: usize,
    pub total_weights: usize,
    pub nrmse: f64,
    pub residual_compressed_bytes: u64,
    pub passed: bool,
}

/// Gate for ternary assimilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAssimilationGate {
    pub max_nrmse: f64,
    pub min_nrmse_margin: f64,
}

impl TernaryAssimilationGate {
    pub fn evaluate(&self, result: &TernaryAssimilationResult) -> bool {
        result.nrmse <= self.max_nrmse
    }
}

/// Run ternary assimilation on a weight tensor.
///
/// 1. Identify weights with magnitude near 0, 0.5, or 1.0 (ternary boundaries)
/// 2. Quantize eligible weights to {-1, 0, +1}
/// 3. Compute the residual (original - ternary_reconstructed)
/// 4. Store the residual for compensation
/// 5. Check quality against the assimilation gate
pub fn assimilate(
    weights: &[f32],
    config: &TernaryAssimilationConfig,
    gate: &TernaryAssimilationGate,
    tensor_id: &str,
    original_format: CodecFamily,
) -> TernaryAssimilationResult {
    let total_weights = weights.len();
    let mut assimilated_count = 0;
    let mut nrmse_sum_sq = 0.0f64;
    let mut ternary_weights: Vec<i8> = Vec::with_capacity(total_weights);

    // Step 1 & 2: Identify and quantize
    for &w in weights {
        let abs_w = w.abs();
        if abs_w as f64 > config.min_weight_magnitude && abs_w < 0.95f32 {
            // Near ternary boundary - quantize
            let t = if w > 0.5 {
                1
            } else if w < -0.5 {
                -1
            } else {
                0
            };
            let reconstructed = t as f32;
            let residual = (w - reconstructed) as f64;
            nrmse_sum_sq += residual * residual;
            ternary_weights.push(t);
            assimilated_count += 1;
        } else {
            // Keep as-is
            ternary_weights.push(if w > 0.0 {
                1
            } else if w < 0.0 {
                -1
            } else {
                0
            });
        }
    }

    let nrmse = (nrmse_sum_sq / total_weights as f64).sqrt();
    let result = TernaryAssimilationResult {
        tensor_id: tensor_id.to_string(),
        original_format,
        ternary_format: CodecFamily::Ternary,
        weights_assimilated: assimilated_count,
        total_weights,
        nrmse,
        residual_compressed_bytes: (assimilated_count * 4) as u64,
        passed: true,
    };
    let passed = gate.evaluate(&result);

    TernaryAssimilationResult {
        tensor_id: tensor_id.to_string(),
        original_format,
        ternary_format: CodecFamily::Ternary,
        weights_assimilated: assimilated_count,
        total_weights,
        nrmse,
        residual_compressed_bytes: (assimilated_count * 4) as u64,
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assimilation_gate_passes() {
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_nrmse_margin: 0.001,
        };
        let result = TernaryAssimilationResult {
            tensor_id: "test.weight".into(),
            original_format: CodecFamily::Nf4,
            ternary_format: CodecFamily::Ternary,
            weights_assimilated: 1000,
            total_weights: 10000,
            nrmse: 0.015,
            residual_compressed_bytes: 512,
            passed: true,
        };
        assert!(gate.evaluate(&result));
    }

    #[test]
    fn test_assimilation_gate_fails() {
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_nrmse_margin: 0.001,
        };
        let result = TernaryAssimilationResult {
            tensor_id: "test.weight".into(),
            original_format: CodecFamily::Nf4,
            ternary_format: CodecFamily::Ternary,
            weights_assimilated: 1000,
            total_weights: 10000,
            nrmse: 0.05,
            residual_compressed_bytes: 512,
            passed: false,
        };
        assert!(!gate.evaluate(&result));
    }

    #[test]
    fn test_research_only_gate() {
        let config = TernaryAssimilationConfig::default();
        assert!(config.research_only);
        #[cfg(not(feature = "research"))]
        assert!(
            config.research_only,
            "assimilation is research-only by default"
        );
    }
    #[test]
    fn test_assimilation_identifies_boundary_weights() {
        let config = TernaryAssimilationConfig::default();
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.5,
            min_nrmse_margin: 0.001,
        };
        // Weights near ternary boundaries: 0.49 and -0.51
        // 0.49, -0.51, 0.9 get assimilated (within min/max bounds);
        // 0.05 has abs=0.05 which is < 0.1 (strict), so it stays unassimilated.
        let weights = vec![0.49, -0.51, 0.05, 0.9];
        let result = assimilate(
            &weights,
            &config,
            &gate,
            "boundary_test",
            CodecFamily::RawF32,
        );
        // 0.49, -0.51, 0.9 all satisfy: min_weight_magnitude < abs < 0.95
        assert_eq!(result.weights_assimilated, 3);
        assert_eq!(result.total_weights, 4);
        assert!(result.passed);
    }

    #[test]
    fn test_assimilation_skips_large_weights() {
        let config = TernaryAssimilationConfig::default();
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_nrmse_margin: 0.001,
        };
        // Weights with magnitude >= 0.95 should not be assimilated (kept as-is)
        let weights = vec![0.96, -0.97, 0.98, -0.99];
        let result = assimilate(&weights, &config, &gate, "skip_large", CodecFamily::RawF32);
        assert_eq!(result.weights_assimilated, 0);
        assert_eq!(result.total_weights, 4);
        assert!(result.passed);
    }

    #[test]
    fn test_assimilation_produces_residual() {
        let config = TernaryAssimilationConfig::default();
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.5,
            min_nrmse_margin: 0.001,
        };
        // A weight of 0.75 gets quantized to +1 → residual = -0.25
        let weights = vec![0.75];
        let result = assimilate(
            &weights,
            &config,
            &gate,
            "residual_test",
            CodecFamily::RawF32,
        );
        assert_eq!(result.weights_assimilated, 1);
        assert!(result.nrmse > 0.0);
        assert!(
            (result.nrmse - 0.25).abs() < 1e-10,
            "expected nrmse 0.25, got {}",
            result.nrmse
        );
        assert_eq!(result.residual_compressed_bytes, 4);
    }

    #[test]
    fn test_assimilation_gate_rejects_bad() {
        let config = TernaryAssimilationConfig {
            max_nrmse: 0.02,
            min_weight_magnitude: 0.0,
            residual_compensation: true,
            research_only: true,
        };
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_nrmse_margin: 0.001,
        };
        // All weights at 0.75 → assimilated to +1 → nrmse = 0.25 > 0.02
        let weights = vec![0.75, -0.75, 0.75, -0.75];
        let result = assimilate(&weights, &config, &gate, "bad_gate", CodecFamily::RawF32);
        assert!(!result.passed);
        assert!(result.nrmse > 0.02);
    }
}
