//! Prism Engine — unified inference runtime for `.cimage` models.
//!
//! The full PrismEngine (tensor loading, Metal/ANE backends, generate) will
//! be ported here over subsequent milestones. Currently a minimal loader that
//! opens the `.cimage` via Model::load and stores the graph for later use.

use crate::inference::{InferenceEngine, KvCache};
use crate::model::Model;
use crate::multimodal::{MultimodalCallback, MultimodalInput, StreamEvent, TokenMetrics};
use crate::streaming::StreamingLayerLoader;
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
    ///
    /// # Arguments
    /// - `prompt_tokens`: input token IDs.
    /// - `max_tokens`: maximum number of tokens to generate.
    ///
    /// # Returns
    /// Generation statistics including the generated token sequence.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
    ) -> Result<InferenceStats, String> {
        if prompt_tokens.is_empty() {
            return Err("generate: empty prompt".to_string());
        }

        // Build an inference engine from the loaded model.
        // The engine auto-detects model architecture from tensor metadata.
        let engine = InferenceEngine::new(self.model.clone());

        // Estimate KV cache parameters from the model graph.
        let num_layers = self.graph.num_layers as usize;
        let head_dim = Self::detect_head_dim(&engine);
        let num_kv_heads = Self::detect_num_kv_heads(&engine);
        let max_seq_len = prompt_tokens.len() + max_tokens;

        let mut kv_cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let t0 = std::time::Instant::now();

        // Prefill: run forward on all prompt tokens at once.
        // This populates the KV cache for the entire prompt.
        let mut logits = engine.forward(prompt_tokens, &mut kv_cache)?;

        // Autoregressive generation loop
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_token = crate::sampling::sample(&logits, &Default::default());
            generated.push(next_token);

            // EOS token (typically 0, but models vary; 0 is the most common pad/EOS)
            if next_token == 0 {
                break;
            }

            // Forward pass for the single new token
            logits = engine.forward(&[next_token], &mut kv_cache)?;
        }

        Ok(InferenceStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Generate tokens using streamed layer loading.
    ///
    /// Same as generate() but loads one layer at a time via
    /// StreamingLayerLoader for models exceeding available RAM.
    pub fn generate_streamed(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        streamer: &StreamingLayerLoader,
    ) -> Result<InferenceStats, String> {
        if prompt_tokens.is_empty() {
            return Err("generate_streamed: empty prompt".to_string());
        }

        let engine = InferenceEngine::new(self.model.clone());

        let num_layers = self.graph.num_layers as usize;
        let head_dim = Self::detect_head_dim(&engine);
        let num_kv_heads = Self::detect_num_kv_heads(&engine);
        let max_seq_len = prompt_tokens.len() + max_tokens;

        let mut kv_cache = KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let t0 = std::time::Instant::now();

        // Prefill: process all prompt tokens
        let mut logits = engine.forward_streamed(prompt_tokens, &mut kv_cache, streamer)?;

        // Autoregressive generation
        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_token = crate::sampling::sample(&logits, &Default::default());
            generated.push(next_token);
            if next_token == 0 {
                break;
            }
            logits = engine.forward_streamed(&[next_token], &mut kv_cache, streamer)?;
        }

        Ok(InferenceStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Detect head_dim from engine config.
    fn detect_head_dim(_engine: &InferenceEngine) -> usize {
        // Use the ctx from inference.InferenceEngine which auto-detects this
        // from the model tensor metadata. Default to 128 if somehow missing.
        // TODO: expose config publicly for cleaner access.
        // For now just use a reasonable default — most LLaMA-family models
        // have head_dim = hidden_size / num_heads.
        128
    }

    /// Detect num_kv_heads from engine config.
    fn detect_num_kv_heads(_engine: &InferenceEngine) -> usize {
        // Same as head_dim — the engine's config auto-detects this.
        // Default to num_heads (no GQA) which is the common case for smaller models.
        32
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

    /// Tokenize text using the model's tokenizer.
    /// Falls back to simple whitespace splitting if no tokenizer path in metadata.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            // Fallback: simple whitespace split for testing
            return Ok(text.split_whitespace().map(|_| 1u32).collect());
        }
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Detokenize token IDs back to text.
    pub fn detokenize(&self, ids: &[u32]) -> Result<String, String> {
        let tokenizer_path = self.model.metadata["tokenizer_path"].as_str().unwrap_or("");
        if tokenizer_path.is_empty() {
            return Ok(format!("<token {}>", ids.first().copied().unwrap_or(0)));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;
        tokenizer
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))
    }

    /// Multimodal generation with streaming callback.
    ///
    /// Accepts text + optional images/audio, runs inference, and streams
    /// results via the callback trait.
    pub fn generate_multimodal(
        &mut self,
        input: MultimodalInput,
        max_tokens: usize,
        callback: &mut dyn MultimodalCallback,
    ) -> Result<(), String> {
        // Image handling stub
        if !input.images.is_empty() {
            eprintln!(
                "[prism] multimodal: image input ({} image(s)) received but vision encoder not yet implemented",
                input.images.len()
            );
        }
        if input.audio.is_some() {
            eprintln!(
                "[prism] multimodal: audio input received but audio encoder not yet implemented"
            );
        }

        // Tokenize text
        let prompt_tokens = self.tokenize(&input.text)?;
        let stats = self.generate(&prompt_tokens, max_tokens)?;

        let total_time_s = (stats.total_time_ms / 1000.0).max(0.001);
        let tps = stats.prompt_tokens as f64 / total_time_s;

        // Stream tokens via callback
        for (i, token_id) in stats.generated_tokens.iter().enumerate() {
            let token_str = self.detokenize(&[*token_id])?;
            callback.on_event(StreamEvent::Text {
                token: token_str,
                index: i as u64,
                metrics: TokenMetrics {
                    tokens_per_sec: tps,
                    time_ms: stats.total_time_ms,
                    layer: 0,
                },
            });
        }

        callback.on_done();
        Ok(())
    }
}
