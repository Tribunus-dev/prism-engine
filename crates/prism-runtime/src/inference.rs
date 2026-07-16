//! Inference engine — per-layer / per-tensor dispatch over a loaded
//! `.cimage` model.
//!
//! The engine routes each layer's forward pass through the format-aware
//! dispatch in [crate::metal] (when `metal-dispatch` is active) or
//! [crate::cpu] as a fallback.

use crate::model::Model;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use std::collections::HashMap;

/// A simple KV cache for transformer inference.
///
/// At minimum stores key and value tensors for each layer and head.
///
/// Fields:
/// - `k_cache[layer][head][seq][head_dim]`: cached keys
/// - `v_cache[layer][head][seq][head_dim]`: cached values
/// - `seq_len`: current cached sequence length
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Per-layer cached keys: k_cache[layer] is a flat `[head][seq][head_dim]`.
    pub k_cache: Vec<Vec<f32>>,
    /// Per-layer cached values: v_cache[layer] is a flat `[head][seq][head_dim]`.
    pub v_cache: Vec<Vec<f32>>,
    /// Current sequence length (number of tokens stored).
    pub seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (GQA support).
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let cache_size = num_layers;
        let per_layer_slots = num_kv_heads * max_seq_len * head_dim;
        KvCache {
            k_cache: vec![vec![0.0f32; per_layer_slots]; cache_size],
            v_cache: vec![vec![0.0f32; per_layer_slots]; cache_size],
            seq_len: 0,
            num_heads: 0,
            num_kv_heads,
            head_dim,
            num_layers,
            max_seq_len,
        }
    }

    /// Append a single token's keys and values for all layers.
    ///
    /// `k_tokens` and `v_tokens` are flattened per-layer arrays of size
    /// `num_kv_heads * head_dim`.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        if layer >= self.num_layers {
            return Err(format!(
                "layer {} out of range (max {})",
                layer, self.num_layers
            ));
        }
        let slot_offset = self.seq_len * self.num_kv_heads * self.head_dim;
        let end = slot_offset + k.len();
        if end > self.k_cache[layer].len() {
            return Err(format!(
                "KV cache overflow for layer {}: slot_offset={}, len={}, capacity={}",
                layer,
                slot_offset,
                k.len(),
                self.k_cache[layer].len()
            ));
        }
        self.k_cache[layer][slot_offset..end].copy_from_slice(k);
        self.v_cache[layer][slot_offset..end].copy_from_slice(v);
        Ok(())
    }

    /// Increment sequence length after all layers are appended.
    pub fn advance_seq(&mut self) {
        self.seq_len += 1;
    }
}

/// Sampling configuration for token generation.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
        }
    }
}

/// Per-layer forward result.
#[derive(Debug, Clone)]
pub struct LayerOutput {
    /// The hidden state after processing this layer.
    pub hidden: Vec<f32>,
}

/// Inference engine that dispatches per-layer forward passes.
pub struct InferenceEngine {
    /// The loaded model.
    pub model: Model,
    /// Per-layer/tensor format assignments (derived from model metadata
    /// or from an evolution CompilePlan).
    pub format_assignments: HashMap<String, TensorFormat>,
}

impl InferenceEngine {
    /// Create a new inference engine for the given model.
    ///
    /// Format assignments are initially empty — call
    /// [`set_format_assignment`] to populate them, or they default to
    /// [`TensorFormat::Fp16`].
    pub fn new(model: Model) -> Self {
        InferenceEngine {
            model,
            format_assignments: HashMap::new(),
        }
    }

    /// Assign an explicit format to a tensor.
    pub fn set_format_assignment(&mut self, tensor_name: &str, format: TensorFormat) {
        self.format_assignments
            .insert(tensor_name.to_string(), format);
    }

    /// Run a full forward pass over all layers.
    ///
    /// # Arguments
    /// - `tokens`: input token IDs.
    /// - `kv_cache`: mutable reference to the KV cache (stateful across calls).
    ///
    /// # Returns
    /// Logits for the next token (vocab-sized float vector).
    pub fn forward(&self, _tokens: &[u32], _kv_cache: &mut KvCache) -> Result<Vec<f32>, String> {
        // Phase 1: token embedding lookup
        // Phase 2: for each transformer layer:
        //   - self-attention (Q/K/V projection, RoPE, attn score, weighted sum, O projection)
        //   - residual add + rms-norm
        //   - MLP (gate/up/down projections + activation)
        //   - residual add + rms-norm
        // Phase 3: final rms-norm + lm_head
        //
        // Each matmul dispatches to metal::dispatch_matmul or cpu::matmul
        // based on the tensor's assigned format.
        //
        // TODO: wire per-layer dispatch once format_assignments are populated
        //       and layer graph is available.

        Err(
            "InferenceEngine::forward: not yet implemented — per-layer dispatch scaffolding"
                .to_string(),
        )
    }

    /// Run a single layer's forward pass.
    pub fn forward_layer(
        &self,
        _layer_idx: usize,
        _hidden: &[f32],
        _kv_cache: &mut KvCache,
    ) -> Result<LayerOutput, String> {
        Err("forward_layer: not yet implemented".to_string())
    }
}
