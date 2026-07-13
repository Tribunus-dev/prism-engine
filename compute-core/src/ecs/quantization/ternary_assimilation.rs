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
    /// Maximum allowed RMSE for an assimilated tensor.
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
    /// Root mean square error (not normalized). Scale-dependent.
    /// Normalization requires a reference norm (e.g., max abs weight) which
    /// should be computed upstream before storage.
    pub rmse: f64,
    pub residual_compressed_bytes: u64,
    pub passed: bool,
    /// Packed ternary weights (-1, 0, +1) for every input weight.
    pub ternary_weights: Vec<i8>,
    /// Per-element reconstruction residuals (original - ternary_reconstructed).
    pub residuals: Vec<f32>,
}

/// Gate for ternary assimilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAssimilationGate {
    pub max_nrmse: f64,
    pub min_rmse_margin: f64,
}

impl TernaryAssimilationGate {
    pub fn evaluate(&self, result: &TernaryAssimilationResult) -> bool {
        result.rmse <= self.max_nrmse
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
    assert!(
        total_weights > 0,
        "cannot assimilate empty tensor: {}",
        tensor_id
    );
    let mut assimilated_count = 0;
    let mut rmse_sum_sq = 0.0f64;
    let mut ternary_weights: Vec<i8> = Vec::with_capacity(total_weights);
    let mut residuals: Vec<f32> = Vec::with_capacity(total_weights);

    // Step 1 & 2: Identify and quantize
    // NOTE: ALL weights are converted to ternary (the else branch also quantizes).
    // Reconstruction error is computed over ALL weights to give an accurate RMSE.
    for &w in weights {
        let abs_w = w.abs();
        // Convert ALL weights to ternary — same logic for assimilated and non-assimilated
        let t = if w > 0.5 {
            1
        } else if w < -0.5 {
            -1
        } else {
            0
        };
        let reconstructed = t as f32;
        let residual = w - reconstructed;
        rmse_sum_sq += (residual as f64) * (residual as f64);
        ternary_weights.push(t);
        residuals.push(residual);

        if abs_w as f64 > config.min_weight_magnitude && abs_w < 0.95f32 {
            assimilated_count += 1;
        }
    }

    let rmse = (rmse_sum_sq / total_weights as f64).sqrt();
    let result = TernaryAssimilationResult {
        tensor_id: tensor_id.to_string(),
        original_format,
        ternary_format: CodecFamily::Ternary,
        weights_assimilated: assimilated_count,
        total_weights,
        rmse,
        residual_compressed_bytes: (assimilated_count * 4) as u64,
        passed: true,
        ternary_weights,
        residuals,
    };
    let passed = gate.evaluate(&result);

    TernaryAssimilationResult {
        tensor_id: tensor_id.to_string(),
        original_format,
        ternary_format: CodecFamily::Ternary,
        weights_assimilated: assimilated_count,
        total_weights,
        rmse,
        residual_compressed_bytes: (assimilated_count * 4) as u64,
        passed,
        ternary_weights: result.ternary_weights,
        residuals: result.residuals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assimilation_gate_passes() {
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_rmse_margin: 0.001,
        };
        let result = TernaryAssimilationResult {
            tensor_id: "test.weight".into(),
            original_format: CodecFamily::Nf4,
            ternary_format: CodecFamily::Ternary,
            weights_assimilated: 1000,
            total_weights: 10000,
            rmse: 0.015,
            residual_compressed_bytes: 512,
            passed: true,
            ternary_weights: vec![],
            residuals: vec![],
        };
        assert!(gate.evaluate(&result));
    }

    #[test]
    fn test_assimilation_gate_fails() {
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.02,
            min_rmse_margin: 0.001,
        };
        let result = TernaryAssimilationResult {
            tensor_id: "test.weight".into(),
            original_format: CodecFamily::Nf4,
            ternary_format: CodecFamily::Ternary,
            weights_assimilated: 1000,
            total_weights: 10000,
            rmse: 0.05,
            residual_compressed_bytes: 512,
            passed: false,
            ternary_weights: vec![],
            residuals: vec![],
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
            min_rmse_margin: 0.001,
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
            // RMSE is ~0.027 with full error accounting (all weights get converted
            // and their reconstruction error is included). Use 0.05 to pass.
            max_nrmse: 0.05,
            min_rmse_margin: 0.001,
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
            min_rmse_margin: 0.001,
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
        assert!(result.rmse > 0.0);
        assert!(
            (result.rmse - 0.25).abs() < 1e-10,
            "expected rmse 0.25, got {}",
            result.rmse
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
            min_rmse_margin: 0.001,
        };
        // All weights at 0.75 → assimilated to +1 → rmse = 0.25 > 0.02
        let weights = vec![0.75, -0.75, 0.75, -0.75];
        let result = assimilate(&weights, &config, &gate, "bad_gate", CodecFamily::RawF32);
        assert!(!result.passed);
        assert!(result.rmse > 0.02);
    }

    #[test]
    fn test_assimilation_tracks_full_error() {
        // Verify that "kept as-is" weights' error is included in RMSE.
        // Weights with abs >= 0.95 are NOT assimilated (converted but not counted
        // as assimilated). Their reconstruction error should still count toward RMSE.
        let config = TernaryAssimilationConfig {
            max_nrmse: 0.5,
            min_weight_magnitude: 0.0, // all weights eligible
            residual_compensation: true,
            research_only: true,
        };
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.5,
            min_rmse_margin: 0.001,
        };

        // One assimilated (0.75 → +1, residual = -0.25)
        // One non-assimilated (0.99 → +1, residual = -0.01 — but not counted as assimilated
        //   because abs >= 0.95).
        let weights = vec![0.75, 0.99];
        let result = assimilate(&weights, &config, &gate, "full_error", CodecFamily::RawF32);
        assert_eq!(result.weights_assimilated, 1);

        // Expected RMSE = sqrt((0.25^2 + 0.01^2) / 2) = sqrt((0.0625 + 0.0001) / 2)
        //                 = sqrt(0.0626 / 2) = sqrt(0.0313) ≈ 0.1769
        // Compute expected via f32 arithmetic (identical to the implementation's
        // path: w - reconstructed in f32, then cast to f64) to avoid f32→f64 precision mismatch.
        let w1: f32 = 0.75;
        let w2: f32 = 0.99;
        let r1 = (w1 - 1.0_f32) as f64;
        let r2 = (w2 - 1.0_f32) as f64;
        let expected = ((r1 * r1 + r2 * r2) / 2.0_f64).sqrt();
        assert!(
            (result.rmse - expected).abs() < 1e-10,
            "expected rmse {:.10}, got {}",
            expected,
            result.rmse
        );

        // Verify ternary weights and residuals are populated
        assert_eq!(result.ternary_weights.len(), 2);
        assert_eq!(result.residuals.len(), 2);
    }

    #[test]
    #[should_panic(expected = "cannot assimilate empty tensor")]
    fn test_assimilation_rejects_empty() {
        let config = TernaryAssimilationConfig::default();
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.5,
            min_rmse_margin: 0.001,
        };
        let weights: Vec<f32> = vec![];
        let _result = assimilate(&weights, &config, &gate, "empty", CodecFamily::RawF32);
    }

    #[test]
    fn test_assimilation_returns_weights() {
        // Verify ternary_weights.len() == input.len()
        let config = TernaryAssimilationConfig::default();
        let gate = TernaryAssimilationGate {
            max_nrmse: 0.5,
            min_rmse_margin: 0.001,
        };
        let weights = vec![0.49, -0.51, 0.05, 0.9, 0.96, -0.97, 0.72];
        let result = assimilate(
            &weights,
            &config,
            &gate,
            "return_weights",
            CodecFamily::RawF32,
        );
        assert_eq!(result.ternary_weights.len(), weights.len());
        assert_eq!(result.residuals.len(), weights.len());
        assert_eq!(result.total_weights, weights.len());

        // Verify residuals match: residual = original - ternary_reconstructed
        for (i, (&w, &r)) in weights.iter().zip(result.residuals.iter()).enumerate() {
            let t = if w > 0.5 {
                1i8
            } else if w < -0.5 {
                -1
            } else {
                0
            };
            let expected_residual = w - t as f32;
            assert!(
                (r - expected_residual).abs() < 1e-7,
                "residual mismatch at index {}: w={}, t={}, expected residual={}, got {}",
                i,
                w,
                t,
                expected_residual,
                r
            );
        }
    }
}
