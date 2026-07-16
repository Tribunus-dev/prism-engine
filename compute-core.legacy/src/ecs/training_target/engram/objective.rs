use serde::{Deserialize, Serialize};

/// Training objectives for engram optimization.
/// Plan Section 8 table:
/// - Retrieval: Activate the correct engram
/// - Reconstruction: Reproduce target latent patterns
/// - Task loss: Improve the intended behavior
/// - Sparsity: Minimize engram capacity and invocation frequency
/// - Interference: Preserve unrelated behavior
/// - Latency: Keep within runtime budget
/// - Quantization: Keep parameters executable in selected codec
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingObjectives {
    pub retrieval_weight: f64,
    pub reconstruction_weight: f64,
    pub task_loss_weight: f64,
    pub sparsity_penalty: f64,
    pub interference_weight: f64,
    pub latency_budget_ns: Option<u64>,
    pub target_codec: String,
}

impl Default for TrainingObjectives {
    fn default() -> Self {
        Self {
            retrieval_weight: 1.0,
            reconstruction_weight: 1.0,
            task_loss_weight: 0.5,
            sparsity_penalty: 0.01,
            interference_weight: 0.1,
            latency_budget_ns: None,
            target_codec: "f32".into(),
        }
    }
}

impl TrainingObjectives {
    pub fn compute_loss(&self, predictions: &[f32], targets: &[f32]) -> f64 {
        let mut mse = 0.0f64;
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let err = (*p - *t) as f64;
            mse += err * err;
        }
        mse / predictions.len() as f64
    }
}
