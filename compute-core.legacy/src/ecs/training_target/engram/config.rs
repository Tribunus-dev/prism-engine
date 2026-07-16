use crate::ecs::training_target::spec::EngramTrainingTarget;

/// Configuration for engram training.
#[derive(Debug, Clone)]
pub struct EngramTrainConfig {
    pub target: EngramTrainingTarget,
    pub calibration_sample_count: usize,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub holdout_fraction: f64,
    pub seed: u64,
}

impl EngramTrainConfig {
    pub fn from_target(target: &EngramTrainingTarget) -> Self {
        Self {
            target: target.clone(),
            calibration_sample_count: 256,
            learning_rate: 0.01,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            holdout_fraction: 0.1,
            seed: 42,
        }
    }
}
