//! Sensitivity analysis system for per-dimension variance tracking.
//!
//! Before the main search, this system performs a cheap ternary probe on
//! each tensor to estimate how sensitive it is to quantization and operation
//! changes. The result drives budget classification: tensors that tolerate
//! aggressive quantization receive more compression, while sensitive tensors
//! get higher-precision treatment.

use crate::evolution::foundation::LogicalTensorId;
use prism_ecs_core::Component;

/// Per-tensor sensitivity analysis receipt.
///
/// Records the estimated variance for each dimension when a cheap probe
/// is applied to a logical tensor.
#[derive(Debug, Clone)]
pub struct TensorSensitivityReceipt {
    /// The logical tensor this receipt describes.
    pub tensor_id: LogicalTensorId,
    /// Estimated variance under format change. Higher = more sensitive.
    pub format_variance: f64,
    /// Estimated variance under operation change.
    pub operation_variance: f64,
    /// Estimated variance under tile geometry change.
    pub geometry_variance: f64,
    /// Estimated variance under memory configuration change.
    pub memory_variance: f64,
    /// Whether the probe was successful.
    pub probe_valid: bool,
}

impl TensorSensitivityReceipt {
    /// Overall sensitivity score: max across all dimensions.
    pub fn overall_sensitivity(&self) -> f64 {
        self.format_variance
            .max(self.operation_variance)
            .max(self.geometry_variance)
            .max(self.memory_variance)
    }

    /// Whether this tensor is sensitive enough to require high-precision.
    pub fn is_sensitive(&self, threshold: f64) -> bool {
        self.overall_sensitivity() > threshold
    }
}

/// Component attached to a tensor entity after sensitivity analysis.
impl Component for TensorSensitivityReceipt {}

/// Budget classification for a tensor.
///
/// Determines how aggressively the evolution pipeline can compress or
/// mutate this tensor's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchBudgetClass {
    /// High budget: can explore aggressive compression and mutations.
    Aggressive,
    /// Moderate budget: limited compression, conservative mutations.
    Conservative,
    /// Low budget: full precision, minimal mutation.
    Minimal,
}

/// Per-dimension variance tracking component.
///
/// Attached to an entity to record the tracked variance across each
/// dimension of the evolution search space.
#[derive(Debug, Clone)]
pub struct DimensionVariance {
    /// Variance per genome dimension (ordered as in CandidateGenome).
    pub per_dimension: Vec<f64>,
    /// Total variance across all dimensions.
    pub total_variance: f64,
}

impl Component for DimensionVariance {}

impl DimensionVariance {
    /// Create a new variance tracker with the given per-dimension values.
    pub fn new(per_dimension: Vec<f64>) -> Self {
        let total: f64 = per_dimension.iter().sum();
        Self {
            per_dimension,
            total_variance: total,
        }
    }
}

/// Sensitivity analysis system — runs cheap probes to classify tensors.
///
/// Attach to an entity in the ECS world. The system analyzes each tensor
/// by applying a cheap ternary probe and producing a `TensorSensitivityReceipt`.
#[derive(Debug, Clone)]
pub struct SensitivityAnalysisSystem {
    /// Default sensitivity threshold for classifying tensors as sensitive.
    pub default_threshold: f64,
}

impl SensitivityAnalysisSystem {
    pub fn new(threshold: f64) -> Self {
        Self {
            default_threshold: threshold,
        }
    }

    /// Run a cheap ternary probe on a tensor.
    ///
    /// The probe applies a small random perturbation to each dimension
    /// and estimates the output variance. This is a simplified model:
    /// real implementations would run a short kernel on a sample input.
    pub fn cheap_ternary_probe(
        &self,
        _tensor_id: &LogicalTensorId,
        _tensor_size: u64,
    ) -> TensorSensitivityReceipt {
        // Simplified probe: uses tensor size as a heuristic.
        // In production, this would run a short Metal/ANE kernel.
        let size_factor = (_tensor_size as f64).ln_1p() / 20.0;
        TensorSensitivityReceipt {
            tensor_id: _tensor_id.clone(),
            format_variance: size_factor * 0.3,
            operation_variance: size_factor * 0.2,
            geometry_variance: size_factor * 0.1,
            memory_variance: size_factor * 0.15,
            probe_valid: true,
        }
    }

    /// Classify a tensor into a budget class based on its sensitivity receipt.
    pub fn classify_budget(&self, receipt: &TensorSensitivityReceipt) -> SearchBudgetClass {
        let sensitivity = receipt.overall_sensitivity();
        if sensitivity > self.default_threshold * 1.5 {
            SearchBudgetClass::Minimal
        } else if sensitivity > self.default_threshold {
            SearchBudgetClass::Conservative
        } else {
            SearchBudgetClass::Aggressive
        }
    }

    /// Estimate the channel shift introduced by quantizing a tensor.
    ///
    /// Returns an approximate shift magnitude (0.0 = no shift, 1.0 = max shift).
    pub fn estimate_channel_shift(receipt: &TensorSensitivityReceipt) -> f64 {
        (receipt.format_variance + receipt.operation_variance).min(1.0)
    }

    /// Analyze all tracked dimensions and update variance tracking.
    ///
    /// Returns a vector of sensitivities, one per tensor.
    pub fn analyze_tensor_sensitivity(
        &self,
        tensors: &[(LogicalTensorId, u64)],
    ) -> Vec<(TensorSensitivityReceipt, SearchBudgetClass)> {
        tensors
            .iter()
            .map(|(id, size)| {
                let receipt = self.cheap_ternary_probe(id, *size);
                let budget = self.classify_budget(&receipt);
                (receipt, budget)
            })
            .collect()
    }
}

impl Default for SensitivityAnalysisSystem {
    fn default() -> Self {
        Self::new(0.3)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheap_probe_produces_valid_receipt() {
        let system = SensitivityAnalysisSystem::default();
        let tensor_id = LogicalTensorId("layer0.attn.q_proj".into());
        let receipt = system.cheap_ternary_probe(&tensor_id, 1024 * 1024);
        assert!(receipt.probe_valid);
        assert_eq!(receipt.tensor_id.0, "layer0.attn.q_proj");
    }

    #[test]
    fn classification_threshold() {
        let system = SensitivityAnalysisSystem::new(0.3);
        let low = TensorSensitivityReceipt {
            tensor_id: LogicalTensorId("t1".into()),
            format_variance: 0.1,
            operation_variance: 0.05,
            geometry_variance: 0.02,
            memory_variance: 0.03,
            probe_valid: true,
        };
        assert_eq!(system.classify_budget(&low), SearchBudgetClass::Aggressive);

        let high = TensorSensitivityReceipt {
            tensor_id: LogicalTensorId("t2".into()),
            format_variance: 0.8,
            operation_variance: 0.7,
            geometry_variance: 0.6,
            memory_variance: 0.5,
            probe_valid: true,
        };
        assert_eq!(
            system.classify_budget(&high),
            SearchBudgetClass::Minimal
        );
    }

    #[test]
    fn dimension_variance_tracking() {
        let dv = DimensionVariance::new(vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.01, 0.02, 0.03]);
        assert_eq!(dv.per_dimension.len(), 8);
        assert!((dv.total_variance - 1.11).abs() < 1e-9);
    }

    #[test]
    fn analyze_tensor_sensitivity_batch() {
        let system = SensitivityAnalysisSystem::default();
        let tensors = vec![
            (LogicalTensorId("t1".into()), 256 * 256),
            (LogicalTensorId("t2".into()), 1024 * 1024),
        ];
        let results = system.analyze_tensor_sensitivity(&tensors);
        assert_eq!(results.len(), 2);
        for (receipt, _budget) in &results {
            assert!(receipt.probe_valid);
        }
    }
}
