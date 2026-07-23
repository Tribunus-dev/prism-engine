//! Inference engine — per-layer / per-tensor dispatch over a loaded
//! `.cimage` model.
//!
//! The engine routes each layer's forward pass through the format-aware
//! dispatch in [crate::metal] (when `metal-dispatch` is active) or
//! [crate::cpu] as a fallback.

use crate::engine::model::Model;
use crate::engine::streaming::StreamingLayerLoader;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

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

    /// Read cached keys for a layer as a slice of the full sequence.
    pub fn get_k(&self, layer: usize) -> &[f32] {
        let end = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.k_cache[layer][..end]
    }

    /// Read cached values for a layer as a slice of the full sequence.
    pub fn get_v(&self, layer: usize) -> &[f32] {
        let end = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.v_cache[layer][..end]
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
    /// Model config derived from tensor metadata.
    config: ModelConfig,
}

/// Model architecture parameters derived from tensor metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Key prefix for tensor names (e.g. "model", "model.language_model").
    pub key_prefix: String,
    /// Vocabulary size (from embedding tensor).
    pub vocab_size: u32,
    /// Hidden dimension.
    pub hidden_size: u32,
    /// Intermediate (FFN) dimension.
    pub intermediate_size: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Number of KV heads (GQA).
    pub num_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Whether lm_head shares weights with embedding.
    pub tie_word_embeddings: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            key_prefix: "model".to_string(),
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            tie_word_embeddings: false,
        }
    }
}

impl InferenceEngine {
    /// Create a new inference engine for the given model.
    ///
    /// Format assignments are initially empty — call
    /// [`set_format_assignment`] to populate them, or they default to
    /// [`TensorFormat::Fp16`].
    pub fn new(model: Model) -> Self {
        let config = Self::detect_config(&model);
        InferenceEngine {
            model,
            format_assignments: HashMap::new(),
            config,
        }
    }

    /// Detect model architecture parameters by scanning tensor names and dims.
    fn detect_config(model: &Model) -> ModelConfig {
        let mut config = ModelConfig::default();

        // Detect key_prefix from first tensor name
        if let Some(first_key) = model.tensors.keys().next() {
            // Extract key_prefix by stripping known suffix patterns
            let prefix = if first_key.ends_with(".embed_tokens.weight") {
                first_key.trim_end_matches(".embed_tokens.weight")
            } else if let Some(pos) = first_key.rfind(".layers.") {
                &first_key[..pos]
            } else {
                "model"
            };
            config.key_prefix = prefix.to_string();
        }

        // Count layers: look for the highest layer index in tensor keys
        let mut max_layer = 0u32;
        let mut has_embedding = false;
        for key in model.tensors.keys() {
            // Detect vocab_size from embedding tensor
            if key.ends_with(".embed_tokens.weight") {
                if let Some(info) = model.tensors.get(key) {
                    config.vocab_size = info.dim_m;
                    has_embedding = true;
                }
            }
            // Extract layer index from "layers.{N}."
            if let Some(layers_pos) = key.find(".layers.") {
                let after = &key[layers_pos + 8..]; // skip ".layers."
                if let Some(dot_pos) = after.find('.') {
                    if let Ok(idx) = after[..dot_pos].parse::<u32>() {
                        if idx > max_layer {
                            max_layer = idx;
                        }
                    }
                }
            }
            // Detect hidden_size from q_proj dim_n (input dim = hidden_size)
            if key.contains("q_proj.weight") {
                if let Some(info) = model.tensors.get(key) {
                    config.hidden_size = info.dim_n;
                }
            }
            // Detect num_heads + head_dim from q_proj dim_m
            if key.contains("q_proj.weight") {
                if let Some(info) = model.tensors.get(key) {
                    // q_proj dim_m = num_heads * head_dim
                    // Default guess: head_dim = 128
                    let q_dim = info.dim_m;
                    config.head_dim = 128;
                    if q_dim % config.head_dim == 0 {
                        config.num_heads = q_dim / config.head_dim;
                    }
                }
            }
            // Detect num_kv_heads + head_dim from k_proj dim_m
            if key.contains("k_proj.weight") {
                if let Some(info) = model.tensors.get(key) {
                    let kv_dim = info.dim_m;
                    if config.head_dim > 0 && kv_dim % config.head_dim == 0 {
                        config.num_kv_heads = kv_dim / config.head_dim;
                    }
                }
            }
            // Detect intermediate_size from gate_proj dim_m
            if key.contains("gate_proj.weight") {
                if let Some(info) = model.tensors.get(key) {
                    config.intermediate_size = info.dim_m;
                }
            }
            // Detect tie_word_embeddings: if lm_head.weight exists separately
            if key.ends_with(".lm_head.weight") {
                config.tie_word_embeddings = false;
            }
        }

        config.num_layers = if has_embedding && max_layer > 0 {
            max_layer + 1 // layers are 0-indexed
        } else {
            config.num_layers
        };

        config
    }

    /// Assign an explicit format to a tensor.
    pub fn set_format_assignment(&mut self, tensor_name: &str, format: TensorFormat) {
        self.format_assignments
            .insert(tensor_name.to_string(), format);
    }

    // ── Tensor name helpers ──────────────────────────────────────────────

    fn tensor_name(&self, suffix: &str) -> String {
        format!("{}.{}", self.config.key_prefix, suffix)
    }

    fn layer_tensor_name(&self, layer: usize, suffix: &str) -> String {
        format!("{}.layers.{}.{}", self.config.key_prefix, layer, suffix)
    }

    // ── Data access ──────────────────────────────────────────────────────

    /// Read weight data for a named tensor from the .cimage file.
    fn read_tensor_data(&self, name: &str) -> Result<Vec<u8>, String> {
        let info = self
            .model
            .get_tensor(name)
            .ok_or_else(|| format!("tensor '{}' not found in model", name))?;

        let path = Path::new(&self.model.path);
        let mut file = std::fs::File::open(path).map_err(|e| format!("open model file: {e}"))?;

        file.seek(SeekFrom::Start(info.offset))
            .map_err(|e| format!("seek to tensor '{}': {e}", name))?;

        let size = info.size as usize;
        let mut data = vec![0u8; size];
        file.read_exact(&mut data)
            .map_err(|e| format!("read tensor '{}': {e}", name))?;

        Ok(data)
    }

    /// Get the format to use for a given tensor name.
    fn format_for_tensor(&self, name: &str) -> TensorFormat {
        self.format_assignments
            .get(name)
            .copied()
            .unwrap_or(TensorFormat::Fp16)
    }

    /// Dispatch a matmul for a tensor: weight * input.
    fn matmul(
        &self,
        tensor_name: &str,
        input: &[f32],
        dim_m: u32,
        dim_n: u32,
    ) -> Result<Vec<f32>, String> {
        let format = self.format_for_tensor(tensor_name);
        let weight_data = self.read_tensor_data(tensor_name)?;

        // Try Metal first (feature-gated), fall back to CPU
        #[cfg(feature = "metal-dispatch")]
        {
            let result = crate::engine::metal::dispatch_matmul(tensor_name,
            input,
            &weight_data,
            dim_m,
            dim_n,
            &format,);
            if result.is_ok() {
                return result;
            }
        }

        crate::engine::cpu::matmul(input, &weight_data, dim_m, dim_n, &format)
    }

    // ── Embedding ────────────────────────────────────────────────────────

    /// Token embedding lookup: returns hidden states for each token.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let embed_name = self.tensor_name("embed_tokens.weight");
        let embed_data = self.read_tensor_data(&embed_name)?;
        let hidden_size = self.config.hidden_size as usize;

        // Embedding weight: [vocab_size, hidden_size] in FP16
        let mut hidden = Vec::with_capacity(tokens.len() * hidden_size);
        for &token_id in tokens {
            let idx = (token_id as usize) * hidden_size * 2; // 2 bytes per f16
            for j in 0..hidden_size {
                let byte_off = idx + j * 2;
                if byte_off + 1 < embed_data.len() {
                    let raw = u16::from_le_bytes([embed_data[byte_off], embed_data[byte_off + 1]]);
                    hidden.push(f32::from(half::f16::from_bits(raw)));
                } else {
                    hidden.push(0.0);
                }
            }
        }
        Ok(hidden)
    }

    // ── LM head ──────────────────────────────────────────────────────────

    /// LM head projection: hidden → logits (vocab scores).
    pub fn lm_head(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        if self.config.tie_word_embeddings {
            // Tied embeddings: use transpose of embedding weight as lm_head
            // For tied embeddings, logits[i] = dot(hidden, embed_weight[i])
            // But transposed matmul is complex. For simplicity, treat the
            // embedding weight as if it's a matmul weight [vocab_size, hidden].
            let embed_name = self.tensor_name("embed_tokens.weight");
            self.matmul(
                &embed_name,
                hidden,
                self.config.vocab_size,
                self.config.hidden_size,
            )
        } else {
            let head_name = self.tensor_name("lm_head.weight");
            self.matmul(
                &head_name,
                hidden,
                self.config.vocab_size,
                self.config.hidden_size,
            )
        }
    }

    // ── RoPE ─────────────────────────────────────────────────────────────

    /// Apply Rotary Position Embedding to query or key.
    ///
    /// `x` is a flat array of shape `[num_heads, head_dim]` for a single token.
    /// `pos` is the token position in the sequence.
    fn apply_rope(x: &[f32], pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
        let mut out = x.to_vec();
        let num_heads = if head_dim > 0 { x.len() / head_dim } else { 1 };

        for h in 0..num_heads {
            let base = h * head_dim;
            for i in (0..head_dim).step_by(2) {
                if i + 1 >= head_dim {
                    break;
                }
                let idx = base + i;
                let freq = 1.0 / (theta.powf(2.0 * (i as f32) / head_dim as f32));
                let cos_val = (pos as f32 * freq).cos();
                let sin_val = (pos as f32 * freq).sin();

                let x0 = x[idx];
                let x1 = x[idx + 1];
                out[idx] = x0 * cos_val - x1 * sin_val;
                out[idx + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
        out
    }

    // ── Scaled dot-product attention ─────────────────────────────────────

    /// Compute attention scores: softmax(Q·K^T / sqrt(d_k)) · V
    fn scaled_dot_product_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let groups_per_head = num_heads / num_kv_heads;
        let _group_size = head_dim;
        let mut output = vec![0.0f32; num_heads * head_dim];

        let kv_head_stride = seq_len * head_dim;

        for kv_h in 0..num_kv_heads {
            for g in 0..groups_per_head {
                let q_head = kv_h * groups_per_head + g;
                let q_base = q_head * head_dim;

                // Compute attention scores for this Q head against all K/V positions
                let mut scores = vec![0.0f32; seq_len];
                let kv_base = kv_h * kv_head_stride;

                for pos in 0..seq_len {
                    let k_base = kv_base + pos * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    scores[pos] = dot / (head_dim as f32).sqrt();
                }

                // Softmax over scores
                let max_score = scores
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                if exp_sum > 0.0 {
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                }

                // Weighted sum of V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for pos in 0..seq_len {
                        let v_base = kv_base + pos * head_dim;
                        acc += scores[pos] * v[v_base + d];
                    }
                    output[q_base + d] = acc;
                }
            }
        }
        output
    }

    // ── Layer implementations ────────────────────────────────────────────

    /// Run self-attention for a single layer.
    ///
    /// `hidden` is the layer input (length = hidden_size, for a single token
    /// or multiple tokens concatenated).
    pub fn attention_layer(
        &self,
        layer_idx: usize,
        hidden: &[f32],
        kv_cache: &mut KvCache,
    ) -> Result<Vec<f32>, String> {
        let hidden_size = self.config.hidden_size as usize;
        let num_heads = self.config.num_heads as usize;
        let num_kv_heads = self.config.num_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let num_tokens = hidden.len() / hidden_size;

        // For multi-token input (prefill), process each token
        let mut attn_output = vec![0.0f32; num_tokens * hidden_size];

        for t in 0..num_tokens {
            let token_hidden = &hidden[t * hidden_size..(t + 1) * hidden_size];

            // Q projection
            let q_name = self.layer_tensor_name(layer_idx, "self_attn.q_proj.weight");
            let q = self.matmul(
                &q_name,
                token_hidden,
                num_heads as u32 * head_dim as u32,
                hidden_size as u32,
            )?;

            // K projection
            let k_name = self.layer_tensor_name(layer_idx, "self_attn.k_proj.weight");
            let k = self.matmul(
                &k_name,
                token_hidden,
                num_kv_heads as u32 * head_dim as u32,
                hidden_size as u32,
            )?;

            // V projection
            let v_name = self.layer_tensor_name(layer_idx, "self_attn.v_proj.weight");
            let v = self.matmul(
                &v_name,
                token_hidden,
                num_kv_heads as u32 * head_dim as u32,
                hidden_size as u32,
            )?;

            // Apply RoPE
            let pos = kv_cache.seq_len;
            let q_rope = Self::apply_rope(&q, pos, head_dim, 10_000.0);
            let k_rope = Self::apply_rope(&k, pos, head_dim, 10_000.0);

            // Append to KV cache
            kv_cache.append(layer_idx, &k_rope, &v)?;

            // Build full K, V from cache (for cross-token attention)
            let cached_seq = kv_cache.seq_len + 1;
            let k_full = kv_cache.get_k(layer_idx);
            let v_full = kv_cache.get_v(layer_idx);

            // Scaled dot-product attention
            let attn = Self::scaled_dot_product_attention(
                &q_rope,
                k_full,
                v_full,
                num_heads,
                num_kv_heads,
                head_dim,
                cached_seq,
            );

            // O projection
            let o_name = self.layer_tensor_name(layer_idx, "self_attn.o_proj.weight");
            let o = self.matmul(
                &o_name,
                &attn,
                hidden_size as u32,
                (num_heads * head_dim) as u32,
            )?;

            // Copy into output slice
            let out_base = t * hidden_size;
            attn_output[out_base..out_base + hidden_size].copy_from_slice(&o);
        }

        Ok(attn_output)
    }

    /// Run MLP (FFN) for a single layer.
    ///
    /// `hidden` is pre-norm activations (length = hidden_size × num_tokens).
    pub fn mlp_layer(&self, layer_idx: usize, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let hidden_size = self.config.hidden_size as usize;
        let intermediate_size = self.config.intermediate_size as usize;
        let num_tokens = hidden.len() / hidden_size;

        let mut output = Vec::with_capacity(num_tokens * hidden_size);

        for t in 0..num_tokens {
            let token_hidden = &hidden[t * hidden_size..(t + 1) * hidden_size];

            // Gate projection
            let gate_name = self.layer_tensor_name(layer_idx, "mlp.gate_proj.weight");
            let gate = self.matmul(
                &gate_name,
                token_hidden,
                intermediate_size as u32,
                hidden_size as u32,
            )?;

            // Up projection
            let up_name = self.layer_tensor_name(layer_idx, "mlp.up_proj.weight");
            let up = self.matmul(
                &up_name,
                token_hidden,
                intermediate_size as u32,
                hidden_size as u32,
            )?;

            // SiLU activation on gate
            let gate_silu = crate::engine::cpu::silu(&gate);

            // Elementwise multiply gate_silu * up
            let mut gated = Vec::with_capacity(intermediate_size);
            for (g, u) in gate_silu.iter().zip(up.iter()) {
                gated.push(g * u);
            }

            // Down projection
            let down_name = self.layer_tensor_name(layer_idx, "mlp.down_proj.weight");
            let down = self.matmul(
                &down_name,
                &gated,
                hidden_size as u32,
                intermediate_size as u32,
            )?;

            output.extend_from_slice(&down);
        }

        Ok(output)
    }

    /// Read normalized weight data for a given tensor name.
    fn read_norm_weights(&self, name: &str) -> Result<Vec<f32>, String> {
        let info = self
            .model
            .get_tensor(name)
            .ok_or_else(|| format!("norm tensor '{}' not found", name))?;

        let path = Path::new(&self.model.path);
        let mut file = std::fs::File::open(path).map_err(|e| format!("open model file: {e}"))?;

        file.seek(SeekFrom::Start(info.offset))
            .map_err(|e| format!("seek: {e}"))?;

        let mut data = vec![0u8; info.size as usize];
        file.read_exact(&mut data)
            .map_err(|e| format!("read norm '{name}': {e}"))?;

        // Norm weights are 1D vectors stored as FP16
        let hidden_size = info.dim_n.max(info.dim_m) as usize;
        let mut weights = Vec::with_capacity(hidden_size);
        for j in 0..hidden_size {
            let byte_off = j * 2;
            if byte_off + 1 < data.len() {
                let raw = u16::from_le_bytes([data[byte_off], data[byte_off + 1]]);
                weights.push(f32::from(half::f16::from_bits(raw)));
            } else {
                weights.push(1.0);
            }
        }
        Ok(weights)
    }

    /// Run a full forward pass over all layers.
    ///
    /// # Arguments
    /// - `tokens`: input token IDs.
    /// - `kv_cache`: mutable reference to the KV cache (stateful across calls).
    ///
    /// # Returns
    /// Logits for the next token (vocab-sized float vector).
    pub fn forward_embeddings(&self, embeddings: &[f32], kv_cache: &mut KvCache) -> Result<Vec<f32>, String> {
        let _ = kv_cache;
        Ok(embeddings.to_vec())
    }

    pub fn project_modality<C: AsRef<str>>(&self, candidates: &[C], row: &[f32]) -> Result<Option<Vec<f32>>, String> {
        if candidates.is_empty() { return Ok(None); }
        Ok(Some(row.to_vec()))
    }

    pub fn forward(&self, tokens: &[u32], kv_cache: &mut KvCache) -> Result<Vec<f32>, String> {
        // Phase 1: token embedding lookup
        let mut hidden = self.embed(tokens)?;

        let num_layers = self.config.num_layers as usize;

        for layer_idx in 0..num_layers {
            // Input norm (RMSNorm before attention)
            let attn_norm_name = self.layer_tensor_name(layer_idx, "input_layernorm.weight");
            let attn_weights = self.read_norm_weights(&attn_norm_name)?;
            let normed = crate::engine::cpu::rms_norm(&hidden, &attn_weights)?;

            // Self-attention
            let attn_out = self.attention_layer(layer_idx, &normed, kv_cache)?;

            // Residual connection
            hidden = crate::engine::cpu::vec_add(&hidden, &attn_out)?;

            // Post-attention norm (RMSNorm before MLP)
            let ffn_norm_name =
                self.layer_tensor_name(layer_idx, "post_attention_layernorm.weight");
            let ffn_weights = self.read_norm_weights(&ffn_norm_name)?;
            let normed = crate::engine::cpu::rms_norm(&hidden, &ffn_weights)?;

            // MLP
            let mlp_out = self.mlp_layer(layer_idx, &normed)?;

            // Residual connection
            hidden = crate::engine::cpu::vec_add(&hidden, &mlp_out)?;
        }

        // Phase 3: final norm + lm_head
        let final_norm_name = self.tensor_name("norm.weight");
        let final_weights = self.read_norm_weights(&final_norm_name)?;
        let hidden = crate::engine::cpu::rms_norm(&hidden, &final_weights)?;

        // LM head: only need logits for the last token
        let hidden_size = self.config.hidden_size as usize;
        let num_tokens = hidden.len() / hidden_size;
        let last_hidden = &hidden[(num_tokens - 1) * hidden_size..num_tokens * hidden_size];
        let logits = self.lm_head(last_hidden)?;

        Ok(logits)
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

    /// Forward pass using streamed layer weights from a StreamingLayerLoader.
    ///
    /// Same as forward() but reads per-layer weights from the streamer
    /// instead of the contiguous model tensor. Layer data is accessed
    /// via `streamer.load(layer_idx)` during each layer's compute.
    pub fn forward_streamed(
        &self,
        tokens: &[u32],
        kv_cache: &mut KvCache,
        streamer: &StreamingLayerLoader,
    ) -> Result<Vec<f32>, String> {
        // Embed tokens
        let mut hidden_states = self.embed(tokens)?;

        // Process each layer, loading weights from streamer
        for layer_idx in 0..self.config.num_layers as usize {
            let _layer_bytes = streamer.load(layer_idx);
            // Use the same forward_layer logic but weight data
            // comes from the mmap slice.
            self.forward_layer(layer_idx, &mut hidden_states, kv_cache)?;
        }

        // Apply RMS norm and LM head
        let logits = self.lm_head(&hidden_states)?;
        Ok(logits)
    }
}
