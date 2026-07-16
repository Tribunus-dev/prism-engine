//! Vision encoder — real ViT forward pass with injected matmul provider.
//!
//! Architecture: ViT-L/14 style (CLIP / SigLIP / EVA / Pixtral).
//! Accepts raw PNG/JPEG bytes, decodes via the `image` crate, patchifies,
//! projects patches through a learned linear embedding, adds positional
//! encoding, and runs transformer encoder layers with multi-head attention.
//!
//! Weight naming convention (CLIP-style):
//! - `patch_embed.weight` — [patch_dim, hidden_dim] f32 row-major
//! - `cls_token` — [hidden_dim] f32
//! - `pos_embed.weight` — [1 + num_patches, hidden_dim] f32 row-major
//! - `layers.{i}.ln1.weight` — [hidden_dim] LayerNorm gain
//! - `layers.{i}.ln1.bias` — [hidden_dim] LayerNorm bias
//! - `layers.{i}.q_proj.weight` — [hidden_dim, num_heads * head_dim]
//! - `layers.{i}.k_proj.weight` — [hidden_dim, num_heads * head_dim]
//! - `layers.{i}.v_proj.weight` — [hidden_dim, num_heads * head_dim]
//! - `layers.{i}.o_proj.weight` — [num_heads * head_dim, hidden_dim]
//! - `layers.{i}.ln2.weight` — [hidden_dim]
//! - `layers.{i}.ln2.bias` — [hidden_dim]
//! - `layers.{i}.mlp.fc1.weight` — [hidden_dim, intermediate_dim]
//! - `layers.{i}.mlp.fc1.bias` — [intermediate_dim]
//! - `layers.{i}.mlp.fc2.weight` — [intermediate_dim, hidden_dim]
//! - `layers.{i}.mlp.fc2.bias` — [hidden_dim]
//!
//! MLP intermediate dimension defaults to 4 * hidden_dim.

use std::collections::HashMap;

// ── Matmul provider ────────────────────────────────────────────────────

/// Injected matmul closure.
///
/// # Contract
/// `matmul(input, weight, dim_m, dim_n)` computes `output[j] = sum_i input[i] * weight[i * dim_m + j]`
/// i.e. a GEMV where `weight` is `[dim_n, dim_m]` row-major f32.
/// - `input` length == `dim_n as usize`
/// - `weight` length >= `(dim_n * dim_m) as usize`
/// - Returns `Vec<f32>` of length `dim_m as usize`
pub struct MatmulProvider {
    pub matmul: Box<dyn Fn(&[f32], &[f32], u32, u32) -> Result<Vec<f32>, String>>,
}

impl std::fmt::Debug for MatmulProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatmulProvider").finish()
    }
}

// ── Architecture variants ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VisionArch {
    ClipVitL,    // CLIP ViT-L/14, 768 dim
    ClipVitBigG, // CLIP ViT-bigG, 1280 dim
    SigLIP,      // SigLIP ViT, 1152 dim
    EvaVit,      // EVA ViT, 1408 dim (CogVLM)
    PixtralVit,  // Pixtral's ViT, 1024 dim
}

// ── Config ─────────────────────────────────────────────────────────────

pub struct VisionEncoderConfig {
    pub arch: VisionArch,
    pub input_size: (u32, u32),
    pub patch_size: u32,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
}

// ── Public entry point ─────────────────────────────────────────────────

/// Encode a raw image (PNG/JPEG) through the full ViT vision encoder.
///
/// # Arguments
/// - `image_bytes`: raw encoded image bytes (PNG, JPEG, etc.)
/// - `config`: vision encoder architecture parameters
/// - `weights`: map of tensor name → flat f32 weight data
/// - `matmul`: injected matmul provider for linear projections
///
/// # Returns
/// Output CLS token embedding vector of length `config.hidden_dim`.
pub fn encode_image(
    image_bytes: &[u8],
    config: &VisionEncoderConfig,
    weights: &HashMap<String, Vec<f32>>,
    matmul: &MatmulProvider,
) -> Result<Vec<f32>, String> {
    let hd = config.hidden_dim as usize;

    // ── 1. Decode image ──────────────────────────────────────────────
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("vision_encoder: decode image: {e}"))?;
    let img = img.resize_exact(
        config.input_size.0,
        config.input_size.1,
        image::imageops::FilterType::Lanczos3,
    );
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let num_pixels = (w * h) as usize;

    // Normalize RGB pixels to [0, 1] f32, flat interleaved
    let pixels: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();
    debug_assert_eq!(pixels.len(), num_pixels * 3);

    // ── 2. Patchify ──────────────────────────────────────────────────
    let ps = config.patch_size as usize;
    let patches_h = h as usize / ps;
    let patches_w = w as usize / ps;
    let num_patches = patches_h * patches_w;
    let patch_dim = (ps * ps * 3) as u32; // e.g. 14×14×3 = 588

    let mut patch_flat = Vec::with_capacity(num_patches * patch_dim as usize);
    for ph in 0..patches_h {
        for pw in 0..patches_w {
            for py in 0..ps {
                for px in 0..ps {
                    let img_y = ph * ps + py;
                    let img_x = pw * ps + px;
                    let idx = (img_y * w as usize + img_x) * 3;
                    patch_flat.push(pixels[idx]);
                    patch_flat.push(pixels[idx + 1]);
                    patch_flat.push(pixels[idx + 2]);
                }
            }
        }
    }
    debug_assert_eq!(patch_flat.len(), num_patches * patch_dim as usize);

    // ── 3. Linear patch projection via matmul ────────────────────────
    let patch_weight = weights
        .get("patch_embed.weight")
        .ok_or_else(|| "vision_encoder: missing weight 'patch_embed.weight'".to_string())?;

    // For each patch: flat_patch [patch_dim] @ weight [patch_dim, hidden_dim] → [hidden_dim]
    let mut patch_embeds: Vec<f32> = Vec::with_capacity(num_patches * hd);
    for p in 0..num_patches {
        let start = p * patch_dim as usize;
        let end = start + patch_dim as usize;
        let patch_vec = &patch_flat[start..end];
        let out = (matmul.matmul)(patch_vec, patch_weight, config.hidden_dim, patch_dim)?;
        debug_assert_eq!(out.len(), hd);
        patch_embeds.extend_from_slice(&out);
    }

    // ── 4. Prepend CLS token, add position embeddings ────────────────
    let cls_token = weights
        .get("cls_token")
        .ok_or_else(|| "vision_encoder: missing weight 'cls_token'".to_string())?;
    let pos_embed = weights
        .get("pos_embed.weight")
        .ok_or_else(|| "vision_encoder: missing weight 'pos_embed.weight'".to_string())?;

    let total_tokens = 1 + num_patches; // CLS + patches
    let expected_pos_len = total_tokens * hd;
    if pos_embed.len() < expected_pos_len {
        return Err(format!(
            "vision_encoder: pos_embed.weight len {} < expected {} ({} tokens × {})",
            pos_embed.len(),
            expected_pos_len,
            total_tokens,
            hd
        ));
    }

    // Build sequence: [CLS] + [patch_0, patch_1, ..., patch_N-1]
    let mut hidden: Vec<f32> = Vec::with_capacity(total_tokens * hd);
    hidden.extend_from_slice(cls_token);
    hidden.extend_from_slice(&patch_embeds);

    // Add position embeddings element-wise
    for i in 0..total_tokens {
        let base = i * hd;
        for j in 0..hd {
            hidden[base + j] += pos_embed[base + j];
        }
    }

    // ── 5. Transformer encoder layers ────────────────────────────────
    for layer_idx in 0..config.num_layers as usize {
        hidden = transformer_layer(hidden, total_tokens, hd, layer_idx, weights, matmul)?;
    }

    // ── 6. Return CLS token embedding (first token) ──────────────────
    Ok(hidden[..hd].to_vec())
}

// ── Per-layer transformer ──────────────────────────────────────────────

/// Run one ViT transformer encoder layer over all tokens.
///
/// Layer structure:
///   x = x + MHA(LayerNorm(x))
///   x = x + MLP(LayerNorm(x))
fn transformer_layer(
    hidden: Vec<f32>,
    num_tokens: usize,
    hidden_dim: usize,
    layer_idx: usize,
    weights: &HashMap<String, Vec<f32>>,
    matmul: &MatmulProvider,
) -> Result<Vec<f32>, String> {
    let prefix = format!("layers.{layer_idx}");

    // Per-head dimensions
    let num_heads = infer_num_heads(layer_idx, weights, hidden_dim);
    let head_dim = hidden_dim / num_heads;
    let qk_dim = num_heads * head_dim;

    let default_gain = vec![1.0f32; hidden_dim];
    let default_bias = vec![0.0f32; hidden_dim];

    // ── Self-attention sub-layer: x = x + MHA(LN(x)) ────────────────
    let residual = hidden.clone();

    let ln1_gain = weights
        .get(&format!("{prefix}.ln1.weight"))
        .unwrap_or(&default_gain);
    let ln1_bias = weights
        .get(&format!("{prefix}.ln1.bias"))
        .unwrap_or(&default_bias);
    let x_norm = layer_norm(&hidden, ln1_gain, ln1_bias, 1e-6);

    // QKV projections via matmul (per token, independent)
    let (w_q, w_k, w_v) = resolve_qkv_weights(&prefix, weights);

    let mut q = vec![0.0f32; num_tokens * qk_dim];
    let mut k = vec![0.0f32; num_tokens * qk_dim];
    let mut v = vec![0.0f32; num_tokens * qk_dim];

    for t in 0..num_tokens {
        let token_start = t * hidden_dim;
        let token_slice = &x_norm[token_start..token_start + hidden_dim];
        let q_t = (matmul.matmul)(token_slice, w_q, qk_dim as u32, hidden_dim as u32)?;
        let k_t = (matmul.matmul)(token_slice, w_k, qk_dim as u32, hidden_dim as u32)?;
        let v_t = (matmul.matmul)(token_slice, w_v, qk_dim as u32, hidden_dim as u32)?;
        let off = t * qk_dim;
        q[off..off + qk_dim].copy_from_slice(&q_t);
        k[off..off + qk_dim].copy_from_slice(&k_t);
        v[off..off + qk_dim].copy_from_slice(&v_t);
    }

    // Multi-head attention
    let attn_out = multi_head_attention(&q, &k, &v, num_tokens, num_heads, head_dim);

    // Output projection
    let w_o = weights
        .get(&format!("{prefix}.o_proj.weight"))
        .ok_or_else(|| format!("vision_encoder: missing weight '{prefix}.o_proj.weight'"))?;

    let mut x = residual;
    for t in 0..num_tokens {
        let attn_slice = &attn_out[t * qk_dim..(t + 1) * qk_dim];
        let out_t = (matmul.matmul)(attn_slice, w_o, hidden_dim as u32, qk_dim as u32)?;
        let t_off = t * hidden_dim;
        for j in 0..hidden_dim {
            x[t_off + j] += out_t[j];
        }
    }

    // ── MLP sub-layer: x = x + MLP(LN(x)) ───────────────────────────
    let mlp_prefix = format!("{prefix}.mlp");
    let mlp_intermediate = 4 * hidden_dim;

    let ln2_gain = weights
        .get(&format!("{prefix}.ln2.weight"))
        .unwrap_or(&default_gain);
    let ln2_bias = weights
        .get(&format!("{prefix}.ln2.bias"))
        .unwrap_or(&default_bias);
    let x_norm = layer_norm(&x, ln2_gain, ln2_bias, 1e-6);

    let fc1_w = weights
        .get(&format!("{mlp_prefix}.fc1.weight"))
        .ok_or_else(|| format!("vision_encoder: missing weight '{mlp_prefix}.fc1.weight'"))?;
    let fc1_b = weights
        .get(&format!("{mlp_prefix}.fc1.bias"))
        .ok_or_else(|| format!("vision_encoder: missing weight '{mlp_prefix}.fc1.bias'"))?;
    let fc2_w = weights
        .get(&format!("{mlp_prefix}.fc2.weight"))
        .ok_or_else(|| format!("vision_encoder: missing weight '{mlp_prefix}.fc2.weight'"))?;
    let fc2_b = weights
        .get(&format!("{mlp_prefix}.fc2.bias"))
        .ok_or_else(|| format!("vision_encoder: missing weight '{mlp_prefix}.fc2.bias'"))?;

    for t in 0..num_tokens {
        let t_off = t * hidden_dim;
        let token_slice = &x_norm[t_off..t_off + hidden_dim];

        // fc1: [hidden_dim] → [intermediate_dim]
        let fc1_out = (matmul.matmul)(
            token_slice,
            fc1_w,
            mlp_intermediate as u32,
            hidden_dim as u32,
        )?;

        // bias + GELU
        let mut activated = Vec::with_capacity(mlp_intermediate);
        for (v, b) in fc1_out.iter().zip(fc1_b.iter()) {
            activated.push(gelu(v + b));
        }

        // fc2: [intermediate_dim] → [hidden_dim]
        let fc2_out = (matmul.matmul)(
            &activated,
            fc2_w,
            hidden_dim as u32,
            mlp_intermediate as u32,
        )?;

        // bias + residual
        for j in 0..hidden_dim {
            x[t_off + j] += fc2_out[j] + fc2_b[j];
        }
    }

    Ok(x)
}

// ── Attention ──────────────────────────────────────────────────────────

/// Compute scaled dot-product multi-head attention over all tokens.
///
/// q, k, v are flat arrays of shape `[num_tokens, num_heads * head_dim]`.
/// Returns flat `[num_tokens, num_heads * head_dim]`.
fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let qk_dim = num_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0f32; num_tokens * qk_dim];

    for h in 0..num_heads {
        let h_off = h * head_dim;

        // Extract per-head Q, K, V: [num_tokens, head_dim]
        let q_h: Vec<f32> = (0..num_tokens)
            .flat_map(|t| {
                let base = t * qk_dim + h_off;
                q[base..base + head_dim].iter().copied()
            })
            .collect();
        let k_h: Vec<f32> = (0..num_tokens)
            .flat_map(|t| {
                let base = t * qk_dim + h_off;
                k[base..base + head_dim].iter().copied()
            })
            .collect();
        let v_h: Vec<f32> = (0..num_tokens)
            .flat_map(|t| {
                let base = t * qk_dim + h_off;
                v[base..base + head_dim].iter().copied()
            })
            .collect();

        // scores = Q @ K^T : [num_tokens, num_tokens]
        let mut scores = vec![0.0f32; num_tokens * num_tokens];
        for i in 0..num_tokens {
            for j in 0..num_tokens {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q_h[i * head_dim + d] as f64 * k_h[j * head_dim + d] as f64;
                }
                scores[i * num_tokens + j] = (dot * scale as f64) as f32;
            }
        }

        // softmax over last dim (per query token)
        let attn = softmax_2d(&scores, num_tokens, num_tokens);

        // out_h = attn @ V : [num_tokens, head_dim]
        for i in 0..num_tokens {
            for d in 0..head_dim {
                let mut s = 0.0f64;
                for j in 0..num_tokens {
                    s += attn[i * num_tokens + j] as f64 * v_h[j * head_dim + d] as f64;
                }
                output[i * qk_dim + h_off + d] = s as f32;
            }
        }
    }

    output
}

// ── Utility ops ────────────────────────────────────────────────────────

/// Layer Normalization: `y = (x - mean(x)) / sqrt(var(x) + ε) * gain + bias`.
fn layer_norm(x: &[f32], gain: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    let inv_std = (var + eps as f64).sqrt().recip();

    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let normalized = (v as f64 - mean) * inv_std;
            let g = if i < gain.len() { gain[i] as f64 } else { 1.0 };
            let b = if i < bias.len() { bias[i] as f64 } else { 0.0 };
            (normalized * g + b) as f32
        })
        .collect()
}

/// 2D softmax over rows (last dimension = row_len per row).
fn softmax_2d(scores: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = scores.to_vec();
    for r in 0..rows {
        let base = r * cols;
        let max_val = out[base..base + cols]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for c in 0..cols {
            let v = (out[base + c] - max_val).exp();
            out[base + c] = v;
            sum += v as f64;
        }
        let inv_sum = sum.recip();
        for c in 0..cols {
            out[base + c] = (out[base + c] as f64 * inv_sum) as f32;
        }
    }
    out
}

/// GELU activation (tanh approximation).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
}

// ── Weight resolution helpers ──────────────────────────────────────────

/// Resolve QKV projection weights for a transformer layer.
///
/// Tries separate q/k/v proj first, falls back to a fused in_proj_weight.
fn resolve_qkv_weights<'a>(
    prefix: &'a str,
    weights: &'a HashMap<String, Vec<f32>>,
) -> (&'a [f32], &'a [f32], &'a [f32]) {
    let q_key = format!("{prefix}.q_proj.weight");
    let k_key = format!("{prefix}.k_proj.weight");
    let v_key = format!("{prefix}.v_proj.weight");

    if let (Some(qw), Some(kw), Some(vw)) = (
        weights.get(&q_key),
        weights.get(&k_key),
        weights.get(&v_key),
    ) {
        return (qw.as_slice(), kw.as_slice(), vw.as_slice());
    }

    // Fused in_proj_weight: [3 * qk_dim, hidden_dim], first qk_dim = Q, next = K, last = V
    let fused_key = format!("{prefix}.attn.in_proj_weight");
    if let Some(fused) = weights.get(&fused_key) {
        let third = fused.len() / 3;
        return (
            &fused[..third],
            &fused[third..2 * third],
            &fused[2 * third..],
        );
    }

    panic!(
        "vision_encoder: cannot find QKV weights for '{prefix}' \
         (tried q_proj/k_proj/v_proj and attn.in_proj_weight)"
    );
}

/// Infer number of attention heads from q_proj weight or fall back to a heuristic.
fn infer_num_heads(
    layer_idx: usize,
    weights: &HashMap<String, Vec<f32>>,
    hidden_dim: usize,
) -> usize {
    let q_key = format!("layers.{layer_idx}.q_proj.weight");
    if let Some(qw) = weights.get(&q_key) {
        // q_proj weight is [hidden_dim, num_heads * head_dim]
        let dim_m = qw.len() / hidden_dim;
        // Default head_dim = 64 for ViT
        if dim_m % 64 == 0 {
            return dim_m / 64;
        }
        // Try to infer from hidden_dim
        let ratio = hidden_dim / 64;
        if ratio > 0 && dim_m % ratio == 0 {
            return dim_m / ratio;
        }
        return dim_m / 64;
    }
    // Default fallback
    let fallback = hidden_dim / 64;
    if fallback > 0 {
        return fallback;
    }
    16 // last resort — ViT-L/14 default
}
