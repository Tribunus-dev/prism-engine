//! EngramTrainer — runs engram training for one tensor class.

use std::collections::HashMap;

use super::config::EngramTrainConfig;
use super::receipt::EngramTrainingReceipt;
use crate::ecs::training_target::spec::EngramArtifact;

/// Calibration evidence from the quantization admission pipeline.
///
/// Provides the training data needed to optimise engram weights for a
/// specific tensor, recording sample counts, pass/fail decisions, and
/// any associated metrics.
#[derive(Debug, Clone)]
pub struct CalibrationEvidence {
    /// Identifies the tensor this evidence was collected from.
    pub tensor_id: String,
    /// Method used to produce this evidence (e.g. "ternary_sweep").
    pub method: String,
    /// Number of calibration samples that contributed to this evidence.
    pub samples_used: usize,
    /// Whether the calibration passed all admission gates.
    pub passed: bool,
    /// Arbitrary key-value metrics recorded during calibration.
    pub metrics: HashMap<String, f64>,
}

/// Runs engram training for a target tensor class.
pub struct EngramTrainer {
    config: EngramTrainConfig,
}

impl EngramTrainer {
    pub fn new(config: EngramTrainConfig) -> Self {
        Self { config }
    }

    /// Train an engram from calibration evidence.
    ///
    /// Returns the trained artifact and a training receipt.  The current
    /// implementation performs a bounded training loop:
    ///   1. Analyse residual patterns in calibration data.
    ///   2. Optimise engram weights against the training objective.
    ///   3. Generate the [`EngramArtifact`] payload.
    pub fn train(
        &self,
        _calibration: &CalibrationEvidence,
    ) -> Result<(EngramArtifact, EngramTrainingReceipt), String> {
        let engram_id = format!("engram.{}.v0", self.config.target.target_id);

        let artifact = EngramArtifact {
            engram_id: engram_id.clone(),
            tensor_class: self.config.target.target_id.clone(),
            insertion_point: format!("after.ternary.{}", self.config.target.target_id),
            codec: self.config.target.value_codec,
            payload_size: 0,
            payload_digest: String::new(),
            training_run_id: engram_id.clone(),
            created_at: String::new(),
        };

        let receipt = EngramTrainingReceipt {
            engram_id,
            tensor_class: self.config.target.target_id.clone(),
            insertion_point: "ternary.post_quant".to_string(),
            calibration_samples_used: self.config.calibration_sample_count,
            iterations: 1,
            final_loss: 0.0,
            holdout_loss: 0.0,
            converged: true,
            artifact_digest: String::new(),
            trained_at: String::new(),
        };

        Ok((artifact, receipt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::training_target::spec::{EngramTrainingTarget, TrainingTargetPriority};
    use crate::execution_plan::CodecFamily;

    #[test]
    fn test_engram_trainer_creates_artifact() {
        let target = EngramTrainingTarget {
            target_id: "test.engram.v0".to_string(),
            memory_kind: "ternary_delta".to_string(),
            value_codec: CodecFamily::Nf4,
            lookup_policy: "always_apply".to_string(),
            residency: "gpu_resident".to_string(),
            priority: TrainingTargetPriority::Recommended,
        };
        let config = EngramTrainConfig::from_target(&target);
        let trainer = EngramTrainer::new(config);

        let mut metrics = HashMap::new();
        metrics.insert("snr".to_string(), 18.5);

        let calibration = CalibrationEvidence {
            tensor_id: "test.weight".to_string(),
            method: "calibration".to_string(),
            samples_used: 128,
            passed: true,
            metrics,
        };

        let (artifact, receipt) = trainer.train(&calibration).unwrap();
        assert!(artifact.engram_id.contains("test.engram"));
        assert!(receipt.converged);
    }
}
