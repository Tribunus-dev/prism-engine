use prism_ecs_constitutional::canonical::identity::CorpusId;
use serde::{Deserialize, Serialize};

/// Engram training dataset — actual examples, activations, targets, holdouts.
/// Plan Section 8: "Engram training consumes actual examples, activations,
/// targets, and holdouts."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramTrainingDataset {
    pub corpus_id: CorpusId,
    pub train_examples: Vec<Vec<f32>>,
    pub train_targets: Vec<Vec<f32>>,
    pub validation_examples: Vec<Vec<f32>>,
    pub validation_targets: Vec<Vec<f32>>,
    pub holdout_examples: Vec<Vec<f32>>,
    pub holdout_targets: Vec<Vec<f32>>,
    /// Examples that should remain unchanged when the engram is applied.
    pub interference_examples: Vec<Vec<f32>>,
    pub activation_capture: Option<Vec<u8>>,
}
