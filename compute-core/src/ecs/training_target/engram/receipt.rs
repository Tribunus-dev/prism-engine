use serde::{Deserialize, Serialize};

/// Receipt from an engram training run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramTrainingReceipt {
    pub engram_id: String,
    pub tensor_class: String,
    pub insertion_point: String,
    pub calibration_samples_used: usize,
    pub iterations: usize,
    pub final_loss: f64,
    pub holdout_loss: f64,
    pub converged: bool,
    pub artifact_digest: String,
    pub trained_at: String,
}
