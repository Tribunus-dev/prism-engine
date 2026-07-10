//! BitNet b1.58 CPU reference implementation.
//!
//! Pure Rust decoder layer reference using ternary-packed weights.
//! Designed for numerical validation against Metal GPU output.
//!
//! # Tensor storage convention
//!
//! The importer stores all weight tensors in `[in_features, out_features]` layout
//! (rows = in, cols = out).  This is the transpose of the layout that
//! `ternary_gemv_reference` expects ([out, in]).  We therefore provide our own
//! `bitnet_linear` function that correctly computes `output = input @ weight` for
//! `[in, out]` storage.

use crate::ternary::codec::{TernaryCodecError, TernaryPackedTensor};
use crate::ternary::pack::unpack_ternary_codes;

/// RMSNorm: `(x[i] / rms(x)) * weight[i]` with configurable epsilon.
///
/// Same as `rms_norm_f32` in `crate::ecs::cimage::mlp_reference` but public.
pub fn compute_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
    let rms = ((sum_sq / n) + eps as f64).sqrt() as f32;
    x.iter()
        .zip(weight.iter())
        .map(|(xi, wi)| (xi / rms) * wi)
        .collect()
}

/// SiLU activation: `x * sigmoid(x)`.
pub fn silu_activation(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Extract the weight vector from a single-row "layernorm" tensor.
///
/// The tensor has `rows=1, cols=dim, group_size=dim` (one group).
/// Unpacks the ternary codes and multiplies by the single f16 scale.
fn extract_layernorm_weight(tensor: &TernaryPackedTensor) -> Vec<f32> {
    let scale = tensor.scales[0].to_f32();
    let weights = unpack_ternary_codes(&tensor.codes, tensor.cols).expect("layernorm unpack");
    weights.iter().map(|&w| (w as f32) * scale).collect()
}

/// Extract f32 position IDs from the RawF32 position_ids tensor.
///
/// The tensor's `codes` buffer stores `seq_len` f32 values as raw LE bytes.
fn extract_position_ids(tensor: &TernaryPackedTensor, seq_len: usize) -> Vec<f32> {
    tensor.codes[..seq_len * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Linear projection for a weight stored `[in_features, out_features]`.
///
/// Computes `output[j] = Σᵢ input[i] × weight[i][j]`.
///
/// This is the correct GEMV for the importer's storage layout.
/// `ternary_gemv_reference` expects `[out, in]` and would misread the codes.
fn bitnet_linear(
    input: &[f32],
    tensor: &TernaryPackedTensor,
) -> Result<Vec<f32>, TernaryCodecError> {
    let in_features = tensor.rows; // stored [in, out]
    let out_features = tensor.cols;
    let bytes_per_row = tensor.groups_per_row * tensor.bytes_per_group;

    if input.len() < in_features {
        return Err(TernaryCodecError::PackingError(format!(
            "input length {} < in_features {}",
            input.len(),
            in_features
        )));
    }

    let mut output = vec![0.0f32; out_features];

    for i in 0..in_features {
        let row_scale_offset = i * tensor.groups_per_row;
        let codes_row_start = i * bytes_per_row;
        let input_i = input[i];

        for g in 0..tensor.groups_per_row {
            let col_start = g * tensor.group_size;
            let col_end = (col_start + tensor.group_size).min(out_features);
            let n_weights = col_end - col_start;

            let scale = tensor.scales[row_scale_offset + g].to_f32();
            let group_byte_offset = codes_row_start + g * tensor.bytes_per_group;
            let group_bytes =
                &tensor.codes[group_byte_offset..group_byte_offset + tensor.bytes_per_group];

            let ternary_weights = unpack_ternary_codes(group_bytes, n_weights)?;

            for (k, &w) in ternary_weights.iter().enumerate() {
                output[col_start + k] += input_i * (w as f32) * scale;
            }
        }
    }

    Ok(output)
}

/// Apply rotary position embeddings (RoPE) to a single token's Q or K vector.
///
/// `x` has length `num_heads * head_dim`.  Modifies in place.
fn apply_rope(x: &mut [f32], position: f32, num_heads: usize, head_dim: usize) {
    let base: f32 = 10000.0;
    for h in 0..num_heads {
        let start = h * head_dim;
        // For each pair (2i, 2i+1) within the head
        for i in 0..head_dim / 2 {
            let theta = position * base.powf(-(2.0 * i as f32) / head_dim as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let a = x[start + 2 * i];
            let b = x[start + 2 * i + 1];
            x[start + 2 * i] = a * cos_t - b * sin_t;
            x[start + 2 * i + 1] = a * sin_t + b * cos_t;
        }
    }
}

/// Full BitNet decoder layer reference using ternary-packed weights.
///
/// Processes `seq_len` tokens through the full attention + MLP block with
/// residual connections.
///
/// # Tensor ordering
///
/// | Index | Tensor               | Shape                          |
/// |-------|----------------------|--------------------------------|
/// | 0     | input_layernorm      | [1, hidden_dim] layernorm      |
/// | 1     | q_proj               | [hidden_dim, hidden_dim]       |
/// | 2     | k_proj               | [hidden_dim, kv_inner]         |
/// | 3     | v_proj               | [hidden_dim, kv_inner]         |
/// | 4     | o_proj               | [hidden_dim, hidden_dim]       |
/// | 5     | post_attention_ln    | [1, hidden_dim] layernorm      |
/// | 6     | gate_proj            | [hidden_dim, intermediate_dim] |
/// | 7     | up_proj              | [hidden_dim, intermediate_dim] |
/// | 8     | down_proj            | [intermediate_dim, hidden_dim] |
/// | 9     | position_ids         | [1, seq_len] RawF32            |
/// | 10    | rmsnorm_w (pre-rms)  | [1, hidden_dim] layernorm      |
pub fn bitnet_decoder_layer_reference(
    activations: &[f32],
    tensors: &[&TernaryPackedTensor],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    mut kv_cache: Option<(&mut Vec<Vec<f32>>, &mut Vec<Vec<f32>>)>,
) -> Vec<f32> {
    assert!(
        tensors.len() >= 11,
        "need at least 11 tensors for decoder layer"
    );

    let hidden_dim = activations.len() / seq_len;
    let _kv_inner = num_kv_heads * head_dim;
    let groups = num_heads / num_kv_heads;
    let eps = 1e-6_f32;

    // Pre-extract norm weights (these are 1-row ternary tensors with one group).
    let pre_rms_weight = extract_layernorm_weight(tensors[10]);
    let post_rms_weight = extract_layernorm_weight(tensors[5]);
    let positions = extract_position_ids(tensors[9], seq_len);

    let mut output = vec![0.0f32; activations.len()];

    for t in 0..seq_len {
        let offset = t * hidden_dim;
        let token_acts = &activations[offset..offset + hidden_dim];

        // 1. Input RMSNorm
        let normed = compute_rmsnorm(token_acts, &pre_rms_weight, eps);

        // 2. Q / K / V projections via bitnet_linear (handles [in, out] storage)
        let q = bitnet_linear(&normed, tensors[1]).expect("q_proj");
        let k = bitnet_linear(&normed, tensors[2]).expect("k_proj");
        let v = bitnet_linear(&normed, tensors[3]).expect("v_proj");

        // 3. Apply RoPE
        let pos = positions[t];
        let mut q_rope = q;
        let mut k_rope = k;
        apply_rope(&mut q_rope, pos, num_heads, head_dim);
        apply_rope(&mut k_rope, pos, num_kv_heads, head_dim);

        // 4. KV cache – append and select the appropriate KV window
        let (k_slice, v_slice): (&[Vec<f32>], &[Vec<f32>]);

        // Holding cells for the no-cache path (only attend to self).
        let no_cache_k: Vec<Vec<f32>>;
        let no_cache_v: Vec<Vec<f32>>;

        match &mut kv_cache {
            Some((kc, vc)) => {
                kc.push(k_rope);
                vc.push(v);
                k_slice = kc.as_slice();
                v_slice = vc.as_slice();
            }
            None => {
                no_cache_k = vec![k_rope];
                no_cache_v = vec![v];
                k_slice = &no_cache_k;
                v_slice = &no_cache_v;
            }
        }

        let total_kv = k_slice.len();
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        // 5. Attention scores (causal: only attend to positions ≤ current token)
        let mut attended = vec![0.0f32; hidden_dim];

        for h in 0..num_heads {
            let kv_idx = h / groups;
            let q_start = h * head_dim;
            let kv_start = kv_idx * head_dim;
            let q_head = &q_rope[q_start..q_start + head_dim];

            // Scores: dot(Q_head[t], K_cache[s][kv_idx*head_dim..]) / sqrt(d)
            let mut scores = Vec::with_capacity(total_kv);
            let mut max_score = f32::NEG_INFINITY;

            for s in 0..total_kv {
                let k_head = &k_slice[s][kv_start..kv_start + head_dim];
                let mut dot = 0.0_f32;
                for i in 0..head_dim {
                    dot += q_head[i] * k_head[i];
                }
                let score = dot * inv_sqrt_d;
                scores.push(score);
                if score > max_score {
                    max_score = score;
                }
            }

            // 6. Softmax
            let mut exp_sum = 0.0_f64;
            let mut exps = Vec::with_capacity(total_kv);
            for &s in &scores {
                let e = ((s - max_score) as f64).exp();
                exp_sum += e;
                exps.push(e);
            }

            // Weighted sum of V
            for i in 0..head_dim {
                let mut weighted = 0.0_f64;
                for s in 0..total_kv {
                    weighted += (exps[s] / exp_sum) * v_slice[s][kv_start + i] as f64;
                }
                attended[q_start + i] = weighted as f32;
            }
        }

        // 7. Output projection
        let o = bitnet_linear(&attended, tensors[4]).expect("o_proj");

        // 8. Residual add (first)
        let mut hidden_after_attn = Vec::with_capacity(hidden_dim);
        for i in 0..hidden_dim {
            hidden_after_attn.push(token_acts[i] + o[i]);
        }

        // 9. Post-attention RMSNorm
        let normed2 = compute_rmsnorm(&hidden_after_attn, &post_rms_weight, eps);

        // 10. Gate projection + SiLU
        let gate = bitnet_linear(&normed2, tensors[6]).expect("gate_proj");
        let gate_silu: Vec<f32> = gate.iter().copied().map(silu_activation).collect();

        // 11. Up projection
        let up = bitnet_linear(&normed2, tensors[7]).expect("up_proj");

        //  (gate × up element-wise)
        let gated: Vec<f32> = gate_silu
            .iter()
            .zip(up.iter())
            .map(|(g, u)| g * u)
            .collect();

        // 12. Down projection
        let down = bitnet_linear(&gated, tensors[8]).expect("down_proj");

        // 13. Residual add (second)
        let out_offset = t * hidden_dim;
        for i in 0..hidden_dim {
            output[out_offset + i] = hidden_after_attn[i] + down[i];
        }
    }

    output
}

/// BitNet logits projection via lm_head.
///
/// The `lm_head` is a `RawF32` ternary-like tensor whose `codes` buffer
/// stores f32 weight values as LE bytes — one `[vocab_size, hidden_dim]`
/// matrix in `[in, out]` layout (matching the importer convention).
///
/// Returns logits `[seq_len, vocab_size]` row-major.
pub fn bitnet_decoder_logits(
    activations: &[f32],
    lm_head: &TernaryPackedTensor,
    hidden_dim: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let seq_len = activations.len() / hidden_dim;
    let mut logits = vec![0.0f32; seq_len * vocab_size];

    // lm_head codes contain raw f32 weights in [in_features, out_features] layout
    // = [hidden_dim, vocab_size] in stored terms.
    let in_features = lm_head.rows; // == hidden_dim
    let out_features = lm_head.cols; // == vocab_size

    for t in 0..seq_len {
        let token_offset = t * hidden_dim;
        let token_act = &activations[token_offset..token_offset + hidden_dim];

        for j in 0..out_features {
            let mut dot = 0.0_f32;
            for i in 0..in_features {
                let byte_offset = (i * out_features + j) * 4;
                let weight = f32::from_le_bytes([
                    lm_head.codes[byte_offset],
                    lm_head.codes[byte_offset + 1],
                    lm_head.codes[byte_offset + 2],
                    lm_head.codes[byte_offset + 3],
                ]);
                dot += token_act[i] * weight;
            }
            logits[t * vocab_size + j] = dot;
        }
    }

    logits
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// Helper: a 1-row layernorm tensor (group_size = dim, single scale = 1.0).
    fn norm_tensor(dim: usize) -> TernaryPackedTensor {
        let codes = crate::ternary::pack::pack_ternary_codes(&vec![1_i8; dim]).unwrap();
        TernaryPackedTensor {
            rows: 1,
            cols: dim,
            group_size: dim,
            groups_per_row: 1,
            bytes_per_group: (dim + 3) / 4,
            codes,
            scales: vec![f16::from_f32(1.0)],
        }
    }

    /// Helper: a position_ids RawF32 tensor.
    fn position_ids_tensor(seq_len: usize) -> TernaryPackedTensor {
        let mut codes = Vec::with_capacity(seq_len * 4);
        for i in 0..seq_len {
            codes.extend_from_slice(&(i as f32).to_le_bytes());
        }
        TernaryPackedTensor {
            rows: 1,
            cols: seq_len,
            group_size: seq_len,
            groups_per_row: 1,
            bytes_per_group: 4,
            codes,
            scales: vec![],
        }
    }

    /// Helper: a ternary projection tensor with all weights = +1, scale = 1.0.
    ///
    /// `rows` = in_features, `cols` = out_features (importer convention).
    fn proj_tensor(rows: usize, cols: usize, group_size: usize) -> TernaryPackedTensor {
        let weights: Vec<i8> = vec![1_i8; rows * cols];
        let codes = crate::ternary::pack::pack_ternary_codes(&weights).unwrap();
        let groups_per_row = cols.div_ceil(group_size);
        let scales: Vec<f16> = (0..rows * groups_per_row)
            .map(|_| f16::from_f32(1.0))
            .collect();
        TernaryPackedTensor {
            rows,
            cols,
            group_size,
            groups_per_row,
            bytes_per_group: (group_size + 3) / 4,
            codes,
            scales,
        }
    }

    fn make_test_tensors(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        seq_len: usize,
        group_size: usize,
    ) -> Vec<TernaryPackedTensor> {
        let kv_inner = num_kv_heads * head_dim;

        vec![
            norm_tensor(hidden_dim),                               // 0  input_layernorm
            proj_tensor(hidden_dim, hidden_dim, group_size),       // 1  q_proj [in, out]
            proj_tensor(hidden_dim, kv_inner, group_size),         // 2  k_proj
            proj_tensor(hidden_dim, kv_inner, group_size),         // 3  v_proj
            proj_tensor(hidden_dim, hidden_dim, group_size),       // 4  o_proj
            norm_tensor(hidden_dim),                               // 5  post_attention_ln
            proj_tensor(hidden_dim, intermediate_dim, group_size), // 6  gate_proj
            proj_tensor(hidden_dim, intermediate_dim, group_size), // 7  up_proj
            proj_tensor(intermediate_dim, hidden_dim, group_size), // 8  down_proj
            position_ids_tensor(seq_len),                          // 9  position_ids
            norm_tensor(hidden_dim),                               // 10 rmsnorm_w
        ]
    }

    // ── Unit tests ─────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_basic() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0_f32; 4];
        let eps = 1e-6;
        let out = compute_rmsnorm(&x, &weight, eps);

        let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0_f64 + 1e-6_f64).sqrt() as f32;
        let expected: Vec<f32> = x.iter().map(|xi| xi / rms).collect();

        for (r, e) in out.iter().zip(&expected) {
            assert!((r - e).abs() < 1e-5, "got {r}, expected {e}");
        }
    }

    #[test]
    fn test_silu_activation_value() {
        assert!((silu_activation(0.0) - 0.0).abs() < 1e-6);
        let at_one = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!((silu_activation(1.0) - at_one).abs() < 1e-6);
        let at_minus_two = -2.0 / (1.0 + 2.0_f32.exp());
        assert!((silu_activation(-2.0) - at_minus_two).abs() < 1e-6);
    }

    #[test]
    fn test_decoder_layer_reference_runs() {
        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_dim / num_heads; // 16
        let intermediate_dim = 256;
        let seq_len = 4;
        let group_size = 16;

        let tensors = make_test_tensors(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
            group_size,
        );
        let refs: Vec<&TernaryPackedTensor> = tensors.iter().collect();
        let acts: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i as f32) / 100.0).sin())
            .collect();

        let out = bitnet_decoder_layer_reference(
            &acts,
            &refs,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            None,
        );

        assert_eq!(out.len(), seq_len * hidden_dim);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite outputs: {:?}",
            out.iter().filter(|v| !v.is_finite()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_decoder_layer_output_shape() {
        let hidden_dim = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let intermediate_dim = 128;
        let seq_len = 3;
        let group_size = 8;

        let tensors = make_test_tensors(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
            group_size,
        );
        let refs: Vec<&TernaryPackedTensor> = tensors.iter().collect();
        let acts: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i as f32) / 100.0).cos())
            .collect();

        let out = bitnet_decoder_layer_reference(
            &acts,
            &refs,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            None,
        );

        assert_eq!(out.len(), seq_len * hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_decoder_layer_with_kv_cache() {
        let hidden_dim = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let intermediate_dim = 128;
        let seq_len = 3;
        let group_size = 8;

        let tensors = make_test_tensors(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
            group_size,
        );
        let refs: Vec<&TernaryPackedTensor> = tensors.iter().collect();
        let acts: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i as f32) / 100.0).sin())
            .collect();

        // Baseline: no cache (each token attends only to itself).
        let no_cache = bitnet_decoder_layer_reference(
            &acts,
            &refs,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            None,
        );

        // Incremental: process one token at a time with cache.
        let mut kc: Vec<Vec<f32>> = Vec::new();
        let mut vc: Vec<Vec<f32>> = Vec::new();
        let mut cached = Vec::new();

        for t in 0..seq_len {
            let tok = &acts[t * hidden_dim..(t + 1) * hidden_dim];
            let out = bitnet_decoder_layer_reference(
                tok,
                &refs,
                num_heads,
                num_kv_heads,
                head_dim,
                1,
                Some((&mut kc, &mut vc)),
            );
            cached.extend_from_slice(&out);
        }

        assert_eq!(cached.len(), seq_len * hidden_dim);
        assert!(cached.iter().all(|v| v.is_finite()));
        assert_eq!(no_cache.len(), cached.len());
    }

    #[test]
    fn test_logits_shape() {
        let hidden_dim = 32;
        let vocab_size = 128;
        let seq_len = 2;

        // Build lm_head as RawF32: codes = f32 LE bytes in [in, out] layout.
        let in_feats = hidden_dim;
        let out_feats = vocab_size;
        let mut codes = Vec::with_capacity(in_feats * out_feats * 4);
        for i in 0..in_feats {
            for j in 0..out_feats {
                codes.extend_from_slice(&((i * out_feats + j) as f32 * 0.01).to_le_bytes());
            }
        }

        let lm_head = TernaryPackedTensor {
            rows: in_feats,
            cols: out_feats,
            group_size: hidden_dim,
            groups_per_row: 1,
            bytes_per_group: 4,
            codes,
            scales: vec![],
        };

        let acts: Vec<f32> = (0..seq_len * hidden_dim)
            .map(|i| ((i as f32) / 100.0).sin())
            .collect();

        let logits = bitnet_decoder_logits(&acts, &lm_head, hidden_dim, vocab_size);
        assert_eq!(logits.len(), seq_len * vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
