//! MLX-backed inference adapter for Prism Engine.
//!
//! Loads `.cimage` palettized weights, dequantizes to FP32, then uses
//! `mlx_rs::ops::quantize()` and `quantized_matmul()` for GPU inference.
//! All norms/attention/RoPE run on GPU via MLX — zero CPU round-trips.

#[cfg(feature = "prism-backend")]
pub(crate) mod mlx_adapter {
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::Instant;

    use mlx_rs::Array;
    use mlx_rs::ops;
    use mlx_rs::ops::indexing::IndexOp;

    use crate::quantization::cimage::CImageReader;
    use crate::lut::graph::{ComputeNode, ModelGraph, TensorRole};
    use crate::lut::engine::InferenceStats;

    // ── Weight structs ───────────────────────────────────────────────────

    struct QuantizedWeight {
        w: Array,        // packed u4
        scales: Array,   // FP16 scales per group
        biases: Array,   // FP16 biases per group
        out_dim: usize,
        in_dim: usize,
    }

    struct LayerWeights {
        q_proj: Option<QuantizedWeight>,
        k_proj: Option<QuantizedWeight>,
        v_proj: Option<QuantizedWeight>,
        o_proj: Option<QuantizedWeight>,
        fused_qkv: Option<QuantizedWeight>,
        gate_proj: Option<QuantizedWeight>,
        up_proj: Option<QuantizedWeight>,
        down_proj: Option<QuantizedWeight>,
    }

    pub struct MlxModel {
        tok_emb: Array,       // [vocab_size, dim] FP32
        norm_w: Array,        // [dim] FP32
        lm_head: Option<QuantizedWeight>,
        layers: Vec<LayerWeights>,
        // Model dimensions (extracted from graph)
        dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        vocab_size: usize,
        rope_theta: f32,
        norm_eps: f32,
    }

    // ── Load: cimage → FP32 arrays → MLX quantize ───────────────────────

    fn extract_config(graph: &ModelGraph) -> (UnifiedConfig) {
        // Build config from ComputeNode analysis
        let mut dim = 0u32;
        let mut vocab_size = 0u32;
        let mut n_layers = graph.num_layers;
        let mut n_heads = 0u32;
        let mut n_kv_heads = 0u32;
        let mut head_dim = 0u32;
        let mut hidden_dim = 0u32;
        let mut rope_theta = 10000.0f32;
        let mut norm_eps = 1e-6f32;

        for node in &graph.nodes {
            match node {
                ComputeNode::TokenEmbedding { hidden_dim, vocab_size: vs, .. } => {
                    dim = *hidden_dim;
                    vocab_size = *vs;
                }
                ComputeNode::PalettizedMatmul { role: TensorRole::GateProj, tensor, .. } => {
                    hidden_dim = tensor.dim_m;
                }
                ComputeNode::RotaryEmbedding { head_dim: hd, rope_theta: rt, .. } => {
                    head_dim = *hd;
                    rope_theta = *rt;
                }
                ComputeNode::Norm { eps, .. } => {
                    norm_eps = *eps;
                }
                _ => {}
            }
        }

        // Extract n_heads from first QProj
        for node in &graph.nodes {
            if let ComputeNode::PalettizedMatmul { role: TensorRole::QProj, tensor, .. } = node {
                let q_dim = tensor.dim_m as u32;
                n_heads = q_dim / head_dim.max(1);
                break;
            }
        }

        // Extract n_kv_heads from first KProj
        for node in &graph.nodes {
            if let ComputeNode::PalettizedMatmul { role: TensorRole::KProj, tensor, .. } = node {
                n_kv_heads = tensor.dim_m as u32 / head_dim.max(1);
                break;
            }
        }

        UnifiedConfig {
            dim, vocab_size, n_layers, n_heads, n_kv_heads,
            head_dim, hidden_dim, rope_theta, norm_eps,
        }
    }

    struct UnifiedConfig {
        dim: u32,
        vocab_size: u32,
        n_layers: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        hidden_dim: u32,
        rope_theta: f32,
        norm_eps: f32,
    }

    /// Load a tensor from .cimage as FP32 MLX Array.
    fn load_as_f32_array(reader: &CImageReader, data: &[u8], key: &str, shape: &[i32]) -> Result<Array, String> {
        let rec = reader.tensor(key).ok_or_else(|| format!("missing tensor: {key}"))?;
        let payload = &data[rec.offset as usize..][..rec.size as usize];
        let count = payload.len() / 2;
        let mut f32_vals = Vec::with_capacity(count);
        for i in 0..count {
            let bits = u16::from_le_bytes([payload[i*2], payload[i*2+1]]);
            f32_vals.push(half::f16::from_bits(bits).to_f32());
        }
        Ok(Array::from_slice(&f32_vals, shape))
    }

    /// Load a palettized weight from .cimage, dequantize to FP32, then
    /// MLX-quantize and return (quantized_w, scales, biases).
    fn load_quantized_weight(reader: &CImageReader, data: &[u8], key: &str,
        out_dim: usize, in_dim: usize) -> Result<QuantizedWeight, String> {
        let rec = reader.tensor(key).ok_or_else(|| format!("missing: {key}"))?;
        let payload = &data[rec.offset as usize..][..rec.size as usize];

        // Palettized layout: codebook_block (out_dim × 16 × 2 bytes FP16),
        // then indices_block (out_dim × in_dim/2 bytes, 4-bit packed)
        let cb_size = out_dim * 16 * 2;
        let idx_size = out_dim * (in_dim + 1) / 2;

        // Dequantize: for each row, lookup each 4-bit index in codebook
        let mut f32_flat = Vec::with_capacity(out_dim * in_dim);
        for row in 0..out_dim {
            let cb_offset = row * 32; // 16 × 2 bytes
            let mut codebook = [0.0f32; 16];
            for i in 0..16 {
                if cb_offset + i*2 + 1 < cb_size {
                    let bits = u16::from_le_bytes([payload[cb_offset + i*2], payload[cb_offset + i*2 + 1]]);
                    codebook[i] = half::f16::from_bits(bits).to_f32();
                }
            }
            let idx_offset = cb_size + row * ((in_dim + 1) / 2);
            for i in 0..in_dim {
                let byte_val = payload[idx_offset + i / 2];
                let idx = if i % 2 == 0 { byte_val & 0x0F } else { byte_val >> 4 };
                f32_flat.push(codebook[idx as usize]);
            }
        }

        let f32_arr = Array::from_slice(&f32_flat, &[out_dim as i32, in_dim as i32]);
        let (w, scales, biases) = ops::quantize(&f32_arr, 64, 4).map_err(|e| format!("quantize {key}: {e:?}"))?;

        Ok(QuantizedWeight { w, scales, biases, out_dim, in_dim })
    }

    // ── Public Engine ────────────────────────────────────────────────────

    pub struct MlxEngine {
        model: MlxModel,
    }

    impl MlxEngine {
        pub fn load(path: &Path, graph: ModelGraph) -> Result<Self, String> {
            let data = std::fs::read(path).map_err(|e| format!("read cimage: {e}"))?;
            let reader = CImageReader::open(path)?;
            let cfg = extract_config(&graph);

            // Token embedding
            let tok_emb = load_as_f32_array(&reader, &data, &find_key(&graph, "embed_tokens"),
                &[cfg.vocab_size as i32, cfg.dim as i32])?;

            // Norm weight
            let norm_w = load_norm(&reader, &data, &graph)?;

            // Per-layer weights
            let mut layers = Vec::with_capacity(cfg.n_layers as usize);
            for layer in 0..cfg.n_layers as usize {
                let q = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.self_attn.q_proj"), cfg.dim as usize, cfg.dim as usize);
                let k = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.self_attn.k_proj"), cfg.dim as usize, cfg.dim as usize);
                let v = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.self_attn.v_proj"), cfg.dim as usize, cfg.dim as usize);
                let o = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.self_attn.o_proj"), cfg.dim as usize, cfg.dim as usize);
                let gate = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.mlp.gate_proj"), cfg.hidden_dim as usize, cfg.dim as usize);
                let up = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.mlp.up_proj"), cfg.hidden_dim as usize, cfg.dim as usize);
                let down = load_opt_w(&reader, &data, &graph, &format!(".layers.{layer}.mlp.down_proj"), cfg.dim as usize, cfg.hidden_dim as usize);

                layers.push(LayerWeights { q_proj: q, k_proj: k, v_proj: v, o_proj: o,
                    gate_proj: gate, up_proj: up, down_proj: down, fused_qkv: None });
            }

            // LM head
            let lm_head = load_opt_w(&reader, &data, &graph, "lm_head", cfg.vocab_size as usize, cfg.dim as usize);

            Ok(Self {
                model: MlxModel {
                    tok_emb, norm_w, lm_head, layers,
                    dim: cfg.dim as usize,
                    n_layers: cfg.n_layers as usize,
                    n_heads: cfg.n_heads as usize,
                    n_kv_heads: cfg.n_kv_heads.max(1) as usize,
                    head_dim: cfg.head_dim as usize,
                    hidden_dim: cfg.hidden_dim as usize,
                    vocab_size: cfg.vocab_size as usize,
                    rope_theta: cfg.rope_theta,
                    norm_eps: cfg.norm_eps,
                },
            })
        }

        pub fn generate(&self, prompt: &[u32], max_tokens: usize) -> Result<InferenceStats, String> {
            let t0 = Instant::now();
            let mut generated = Vec::with_capacity(max_tokens);
            let mut pos: i64 = 0;

            // Embed first token
            let mut h = ops::indexing::take(&self.model.tok_emb, &Array::from_slice(&[prompt[0] as i32], &[1]), 0)
                .map_err(|e| format!("embed: {e:?}"))?;

            for step in 0..max_tokens {
                h = self.run_layer(&h, pos)?;

                // LM head
                let logits = match &self.model.lm_head {
                    Some(w) => mlx_matmul(&h, w),
                    None => h.slice(&[.., ..self.model.vocab_size.min(h.shape()[1])]).to_owned(),
                };

                let next = sample(&logits);
                generated.push(next);
                pos += 1;

                if next == 0 || next == 2 { break; }

                // Re-embed next token for next step
                if step + 1 < max_tokens {
                    h = ops::indexing::take(&self.model.tok_emb, &Array::from_slice(&[next as i32], &[1]), 0)
                        .map_err(|e| format!("embed: {e:?}"))?;
                }
            }

            let elapsed = t0.elapsed();
            Ok(InferenceStats {
                prompt_tokens: prompt.len(),
                generated_tokens: generated,
                total_time_ms: elapsed.as_secs_f64() * 1000.0,
            })
        }

        fn run_layer(&self, h: &Array, pos: i64) -> Result<Array, String> {
            let mut h = h.clone();
            let nh = self.model.n_heads;
            let nkv = self.model.n_kv_heads;
            let hd = self.model.head_dim;

            for layer in 0..self.model.n_layers {
                let lw = &self.model.layers[layer];

                // Pre-attention norm
                let h_norm = ops::rms_norm(&h, &self.model.norm_w, self.model.norm_eps)
                    .map_err(|e| format!("norm: {e:?}"))?;

                // QKV projections (FP16 matmul, GPU)
                let q = lw.q_proj.as_ref().map(|w| mlx_matmul(&h_norm, w)).unwrap();
                let k = lw.k_proj.as_ref().map(|w| mlx_matmul(&h_norm, w)).unwrap();
                let v = lw.v_proj.as_ref().map(|w| mlx_matmul(&h_norm, w)).unwrap();

                // Reshape for multi-head attention
                let q = q.reshape(&[1, nh as i32, hd as i32]);
                let k = k.reshape(&[1, nkv as i32, hd as i32]);
                let v = v.reshape(&[1, nkv as i32, hd as i32]);

                // RoPE
                let q = ops::rope(&q, pos as f32, hd as f32, self.model.rope_theta, 1.0, 0)
                    .map_err(|e| format!("rope: {e:?}"))?;
                let k = ops::rope(&k, pos as f32, hd as f32, self.model.rope_theta, 1.0, 0)
                    .map_err(|e| format!("rope: {e:?}"))?;

                // Scaled dot-product attention (GPU flash attention)
                let attn = ops::scaled_dot_product_attention(
                    Some(&q), Some(&k), Some(&v), None::<&Array>, None::<&Array>,
                    1.0 / (hd as f32).sqrt(), true, None::<&Array>, None,
                ).map_err(|e| format!("attn: {e:?}"))?;
                let attn = attn.reshape(&[1, (nh * hd) as i32]);

                // Output projection
                let o = lw.o_proj.as_ref().map(|w| mlx_matmul(&attn, w)).unwrap();
                h = h + &o;

                // Post-attention norm
                let h_norm = ops::rms_norm(&h, &self.model.norm_w, self.model.norm_eps)
                    .map_err(|e| format!("norm: {e:?}"))?;

                // MLP: gate = silu(x @ gate_proj), up = x @ up_proj, gate * up → down_proj
                let gate = lw.gate_proj.as_ref().map(|w| mlx_matmul(&h_norm, w)).unwrap();
                let up = lw.up_proj.as_ref().map(|w| mlx_matmul(&h_norm, w)).unwrap();
                let gate = ops::silu(&gate).map_err(|e| format!("silu: {e:?}"))?;
                let hidden = &gate * &up;
                let down = lw.down_proj.as_ref().map(|w| mlx_matmul(&hidden, w)).unwrap();
                h = h + &down;
            }

            // Final norm
            ops::rms_norm(&h, &self.model.norm_w, self.model.norm_eps)
                .map_err(|e| format!("final norm: {e:?}"))
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    fn mlx_matmul(x: &Array, w: &QuantizedWeight) -> Array {
        ops::quantized_matmul(x, &w.w, &w.scales, &w.biases, false, 64, 4, None)
            .expect("quantized_matmul")
    }

    fn sample(logits: &Array) -> u32 {
        let flat = logits.flatten();
        let (idx, _) = ops::argmax(&flat, 0, false).expect("argmax");
        // argmax returns 0-d array, extract the scalar
        idx.as_slice::<u32>().and_then(|s| s.first().copied()).unwrap_or(0)
    }

    fn find_key(graph: &ModelGraph, suffix: &str) -> String {
        for node in &graph.nodes {
            if let ComputeNode::TokenEmbedding { key, .. } = node {
                if key.contains(suffix) { return key.clone(); }
            }
        }
        suffix.to_string()
    }

    fn load_norm(reader: &CImageReader, data: &[u8], graph: &ModelGraph) -> Result<Array, String> {
        for node in &graph.nodes {
            if let ComputeNode::Norm { key: Some(k), .. } = node {
                return load_as_f32_array(reader, data, k, &[graph.num_layers as i32]);
            }
        }
        Ok(Array::from_slice(&[1.0f32], &[1]))
    }

    fn load_opt_w(reader: &CImageReader, data: &[u8], graph: &ModelGraph,
        suffix: &str, out_dim: usize, in_dim: usize) -> Option<QuantizedWeight> {
        for tb in graph.palettized_tensors() {
            if tb.key.contains(suffix) {
                return load_quantized_weight(reader, data, &tb.key, out_dim, in_dim).ok();
            }
        }
        None
    }
}
