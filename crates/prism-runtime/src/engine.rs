//! Prism Engine — unified inference runtime for `.cimage` models.
//!
//! The full PrismEngine (tensor loading, Metal/ANE backends, generate) will
//! be ported here over subsequent milestones. Currently a minimal loader that
//! opens the `.cimage` via Model::load and stores the graph for later use.

use crate::model::Model;
use prism_ecs_ir::model_graph::ModelGraph;
use std::path::Path;

/// Per-inference statistics returned by [`PrismEngine::generate`].
#[derive(Debug, Default)]
pub struct InferenceStats {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub total_time_ms: f64,
}

/// Inference engine for `.cimage` models.
///
/// Wraps the loaded [`Model`] together with a [`ModelGraph`] that describes
/// the compute topology.  `load()` parses the header and records tensor
/// metadata; payloads are read on demand.
pub struct PrismEngine {
    /// Wrapped model — tensor metadata and header info.
    pub model: Model,
    /// Compute graph describing the transformer architecture.
    pub graph: ModelGraph,
}

impl PrismEngine {
    /// Open a `.cimage` path and pair it with a `ModelGraph`.
    ///
    /// Parses the .cimage header via [`Model::load`] and stores the
    /// graph for subsequent inference dispatch.
    pub fn load(path: &Path, graph: ModelGraph) -> Result<Self, String> {
        let model = Model::load(path)?;
        Ok(PrismEngine { model, graph })
    }

    /// Enable Metal GPU acceleration.
    #[cfg(feature = "metal-dispatch")]
    pub fn with_metal(&mut self) -> Result<(), String> {
        eprintln!("[prism] Metal dispatch not yet wired in prism-runtime stub");
        Ok(())
    }

    /// Generate tokens from a prompt.
    pub fn generate(
        &mut self,
        _prompt: &[u32],
        _max_tokens: usize,
    ) -> Result<InferenceStats, String> {
        Err("prism-runtime engine stub: generate not yet implemented".to_string())
    }

    /// Return the hidden dimension of the embedding layer.
    pub fn embedding_dim(&self) -> u32 {
        for node in &self.graph.nodes {
            if let prism_ecs_ir::model_graph::ComputeNode::TokenEmbedding { hidden_dim, .. } = node
            {
                return *hidden_dim;
            }
        }
        896
    }
}
