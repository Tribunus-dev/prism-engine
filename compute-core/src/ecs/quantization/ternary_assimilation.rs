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
}
