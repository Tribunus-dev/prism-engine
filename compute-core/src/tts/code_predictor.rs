//! Code Predictor — predicts remaining 15 RVQ codebooks from Talker hidden states.
//!
//! Architecture:
//! - 5-layer transformer (self-attention + SwiGLU FFN) with RMSNorm + GQA
//! - 15 × 2048-class output projection heads for residual codebook prediction
//!
//! This module runs entirely on the CPU using the nf4tile640 dequantize+matmul
//! reference implementation. Metal dispatch via MPSGraph is reserved for a
//! follow-up.

use crate::nf4tile640::{self, Nf4Weights};
use bytemuck;

// ════════════════════════════════════════════════════════════════════════════
// Architecture Constants
// ════════════════════════════════════════════════════════════════════════════

/// Hidden dimension of the Talker and Code Predictor.
const HIDDEN_SIZE: usize = 2048;

/// Number of Code Predictor transformer layers.
const N_LAYERS: usize = 5;

/// Number of query heads in GQA self-attention.
const N_HEADS: usize = 16;

/// Number of key/value heads in GQA self-attention.
const N_KV_HEADS: usize = 2;

/// Dimension per attention head.
const HEAD_DIM: usize = 128; // HIDDEN_SIZE / N_HEADS

/// Number of GQA groups (query heads per KV head).
const HEADS_PER_GROUP: usize = N_HEADS / N_KV_HEADS; // 8

/// SwiGLU FFN intermediate dimension.
const INTERMEDIATE_SIZE: usize = 8192;

/// Number of residual codebooks predicted by this module (codebooks 1..=15).
const NUM_CODEBOOKS: usize = 15;

/// Total codebooks (0 from Talker + 15 from Code Predictor).
const TOTAL_CODEBOOKS: usize = 16;

/// RMSNorm epsilon.
const RMS_EPS: f32 = 1e-6;

/// Attention softmax scaling factor: 1 / sqrt(128.0) — pre-computed const.
const ATTN_SCALE: f32 = 0.08838834764831843;

/// Bytes per f32 value.
const F32_BYTES: usize = 4;

// ════════════════════════════════════════════════════════════════════════════
// Tile arithmetic helpers
// ════════════════════════════════════════════════════════════════════════════

/// Number of nf4tile640 tiles needed for a weight matrix of shape `[rows, cols]`.
fn tiles_for(rows: usize, cols: usize) -> usize {
    let tpr = cols.div_ceil(nf4tile640::TILE_ELEMENTS);
    rows * tpr
}

/// Byte size of packed codes for `tiles` tiles.
fn code_bytes(tiles: usize) -> usize {
    tiles * nf4tile640::PACKED_BYTES_PER_TILE
}

/// Number of f32 scale/bias values for `tiles` tiles.
fn scale_count(tiles: usize) -> usize {
    tiles * nf4tile640::SCALES_F32_PER_TILE
}

// ════════════════════════════════════════════════════════════════════════════
// Layer types
// ════════════════════════════════════════════════════════════════════════════

/// One layer of the Code Predictor transformer.
#[derive(Debug, Clone)]
struct CodePredictLayer {
    /// Self-attention Q projection: [HIDDEN_SIZE, N_HEADS * HEAD_DIM]
    attn_q: Nf4Weights,
    /// Self-attention K projection: [HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM]
    attn_k: Nf4Weights,
    /// Self-attention V projection: [HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM]
    attn_v: Nf4Weights,
    /// Self-attention output projection: [N_HEADS * HEAD_DIM, HIDDEN_SIZE]
    attn_o: Nf4Weights,
    /// SwiGLU FFN gate projection: [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    gate_proj: Nf4Weights,
    /// SwiGLU FFN up projection: [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    up_proj: Nf4Weights,
    /// SwiGLU FFN down projection: [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    down_proj: Nf4Weights,
    /// Pre-attention RMSNorm weight: [HIDDEN_SIZE]
    input_norm: Vec<f32>,
    /// Post-attention RMSNorm weight: [HIDDEN_SIZE]
    post_attn_norm: Vec<f32>,
}

// ════════════════════════════════════════════════════════════════════════════
// Core type
// ════════════════════════════════════════════════════════════════════════════

/// Code Predictor — predicts remaining 15 RVQ codebooks from Talker hidden states.
///
/// The Talker autoregressively generates the first RVQ codebook (codebook 0).
/// This module takes the Talker's final hidden states and predicts codebooks
/// 1 through 15, completing the 16-level residual vector quantisation.
pub struct TtsCodePredictor {
    /// 5 transformer layers.
    layers: Vec<CodePredictLayer>,
    /// 15 codebook prediction heads, each a [HIDDEN_SIZE, HIDDEN_SIZE] weight
    /// matrix producing 2048-class logits.
    output_proj: Vec<Nf4Weights>,
}

// ════════════════════════════════════════════════════════════════════════════
// Weight segment parsing
// ════════════════════════════════════════════════════════════════════════════

/// Parse one `Nf4Weights` from the three flat component blocks, advancing each
/// position tracker past the consumed data.
///
/// `codes` — flat packed-codes byte array (all tiles, all weights, tile-major).
/// `scales` — flat f32 scale array (same tile order).
/// `biases` — flat f32 bias array (same tile order).
/// `code_off` — byte cursor into `codes` (advanced by code_bytes(tiles)).
/// `scale_off` — f32-element cursor into `scales` (advanced by scale_count(tiles)).
/// `bias_off` — f32-element cursor into `biases` (advanced by scale_count(tiles)).
///
/// Returns the parsed weight and advances all three cursors past it.
fn parse_weights(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    code_off: &mut usize,
    scale_off: &mut usize,
    bias_off: &mut usize,
    rows: usize,
    cols: usize,
) -> Result<Nf4Weights, String> {
    let tiles = tiles_for(rows, cols);
    let need_code = code_bytes(tiles);
    let need_scales = scale_count(tiles);

    let code_end = *code_off + need_code;
    let scale_end = *scale_off + need_scales;
    let bias_end = *bias_off + need_scales;

    if code_end > codes.len() {
        return Err(format!(
            "segment codes too short: need {need_code} bytes at offset {}, have {}",
            *code_off,
            codes.len(),
        ));
    }
    if scale_end > scales.len() {
        return Err(format!(
            "segment scales too short: need {need_scales} f32 at offset {}, have {}",
            *scale_off,
            scales.len(),
        ));
    }
    if bias_end > biases.len() {
        return Err(format!(
            "segment biases too short: need {need_scales} f32 at offset {}, have {}",
            *bias_off,
            biases.len(),
        ));
    }

    let w = Nf4Weights {
        packed_codes: codes[*code_off..code_end].to_vec(),
        scales: scales[*scale_off..scale_end].to_vec(),
        biases: biases[*bias_off..bias_end].to_vec(),
        rows: rows as u32,
        cols: cols as u32,
    };

    *code_off = code_end;
    *scale_off = scale_end;
    *bias_off = bias_end;

    Ok(w)
}

// ════════════════════════════════════════════════════════════════════════════
// RMSNorm
// ════════════════════════════════════════════════════════════════════════════

/// Apply RMSNorm with learned weight.
///
/// `x` — [num_tokens, hidden_size] f32, row-major.
/// `weight` — [hidden_size] f32 scale.
/// `num_tokens` — number of rows in x.
/// `hidden_size` — number of cols in x / length of weight.
/// `out` — [num_tokens, hidden_size] f32 output, row-major.
fn rms_norm(x: &[f32], weight: &[f32], num_tokens: usize, hidden_size: usize, out: &mut [f32]) {
    for t in 0..num_tokens {
        let offset = t * hidden_size;
        let row = &x[offset..offset + hidden_size];

        // mean(x^2)
        let mut sq_sum = 0.0f32;
        for v in row.iter() {
            sq_sum += v * v;
        }
        let rms = (sq_sum / hidden_size as f32 + RMS_EPS).sqrt();

        // x / rms * weight
        for i in 0..hidden_size {
            out[offset + i] = row[i] / rms * weight[i];
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GQA self-attention
// ════════════════════════════════════════════════════════════════════════════

/// Compute GQA self-attention on CPU.
///
/// `q` — [num_tokens, N_HEADS * HEAD_DIM] f32 query vectors.
/// `k` — [num_tokens, N_KV_HEADS * HEAD_DIM] f32 key vectors.
/// `v` — [num_tokens, N_KV_HEADS * HEAD_DIM] f32 value vectors.
/// `num_tokens` — sequence length.
/// `out` — [num_tokens, N_HEADS * HEAD_DIM] f32 attention output.
fn gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_tokens: usize,
    out: &mut [f32],
) {
    let qk_dim = N_HEADS * HEAD_DIM; // 2048
    let kv_dim = N_KV_HEADS * HEAD_DIM; // 256

    // Per-head attention scores: [num_tokens, num_tokens].
    let mut scores = vec![0.0f32; num_tokens * num_tokens];

    for h in 0..N_HEADS {
        // Which KV head this Q head shares
        let kv_h = h / HEADS_PER_GROUP;

        // Q_h @ K_{kv_h}^T / sqrt(head_dim)
        for ti in 0..num_tokens {
            let qi_base = ti * qk_dim + h * HEAD_DIM;
            let row_scores = &mut scores[ti * num_tokens..(ti + 1) * num_tokens];
            for tj in 0..num_tokens {
                let kj_base = tj * kv_dim + kv_h * HEAD_DIM;
                let mut sum = 0.0f32;
                for d in 0..HEAD_DIM {
                    sum += q[qi_base + d] * k[kj_base + d];
                }
                row_scores[tj] = sum * ATTN_SCALE;
            }
        }

        // Softmax in-place
        for ti in 0..num_tokens {
            let row = &mut scores[ti * num_tokens..(ti + 1) * num_tokens];
            let max_val = row
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max_val).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for s in row.iter_mut() {
                *s *= inv_sum;
            }
        }

        // Weighted sum: out_h = softmax_scores @ V_{kv_h}
        for ti in 0..num_tokens {
            let oi_base = ti * qk_dim + h * HEAD_DIM;
            for d in 0..HEAD_DIM {
                let mut sum = 0.0f32;
                for tj in 0..num_tokens {
                    let vj_idx = tj * kv_dim + kv_h * HEAD_DIM + d;
                    sum += scores[ti * num_tokens + tj] * v[vj_idx];
                }
                out[oi_base + d] = sum;
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// SiLU activation
// ════════════════════════════════════════════════════════════════════════════

/// SiLU (swish) activation: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ════════════════════════════════════════════════════════════════════════════
// Forward pass helpers
// ════════════════════════════════════════════════════════════════════════════

/// Run one Code Predictor transformer layer.
///
/// `x` — [num_tokens, HIDDEN_SIZE] f32, row-major input.
/// Returns new [num_tokens, HIDDEN_SIZE] f32 row-major output.
fn forward_layer(layer: &CodePredictLayer, x: &[f32], num_tokens: usize) -> Result<Vec<f32>, String> {
    let mut residual = vec![0.0f32; num_tokens * HIDDEN_SIZE];
    let mut attn_out = vec![0.0f32; num_tokens * HIDDEN_SIZE];
    let mut normed = vec![0.0f32; num_tokens * HIDDEN_SIZE];
    let mut q = vec![0.0f32; num_tokens * N_HEADS * HEAD_DIM];
    let mut k = vec![0.0f32; num_tokens * N_KV_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; num_tokens * N_KV_HEADS * HEAD_DIM];

    // ── Pre-attention RMSNorm ────────────────────────────────────────────
    rms_norm(x, &layer.input_norm, num_tokens, HIDDEN_SIZE, &mut normed);

    // ── QKV projections ──────────────────────────────────────────────────
    nf4tile640::dequant_matmul_reference(
        &normed,
        &layer.attn_q.packed_codes,
        &layer.attn_q.scales,
        &layer.attn_q.biases,
        num_tokens,
        HIDDEN_SIZE,
        N_HEADS * HEAD_DIM,
        &mut q,
    )?;
    nf4tile640::dequant_matmul_reference(
        &normed,
        &layer.attn_k.packed_codes,
        &layer.attn_k.scales,
        &layer.attn_k.biases,
        num_tokens,
        HIDDEN_SIZE,
        N_KV_HEADS * HEAD_DIM,
        &mut k,
    )?;
    nf4tile640::dequant_matmul_reference(
        &normed,
        &layer.attn_v.packed_codes,
        &layer.attn_v.scales,
        &layer.attn_v.biases,
        num_tokens,
        HIDDEN_SIZE,
        N_KV_HEADS * HEAD_DIM,
        &mut v,
    )?;

    // ── GQA self-attention ───────────────────────────────────────────────
    gqa_attention(&q, &k, &v, num_tokens, &mut attn_out);

    // ── Output projection ────────────────────────────────────────────────
    residual.copy_from_slice(x);
    nf4tile640::dequant_matmul_reference(
        &attn_out,
        &layer.attn_o.packed_codes,
        &layer.attn_o.scales,
        &layer.attn_o.biases,
        num_tokens,
        N_HEADS * HEAD_DIM,
        HIDDEN_SIZE,
        &mut normed,
    )?;

    // Add residual: normed now holds the attention output + skip
    for i in 0..normed.len() {
        normed[i] += residual[i];
    }

    // ── Post-attention RMSNorm ───────────────────────────────────────────
    rms_norm(
        &normed,
        &layer.post_attn_norm,
        num_tokens,
        HIDDEN_SIZE,
        &mut residual,
    );

    // ── SwiGLU FFN ───────────────────────────────────────────────────────
    let mut gate = vec![0.0f32; num_tokens * INTERMEDIATE_SIZE];
    let mut up = vec![0.0f32; num_tokens * INTERMEDIATE_SIZE];
    let mut ffn_out = vec![0.0f32; num_tokens * HIDDEN_SIZE];

    nf4tile640::dequant_matmul_reference(
        &residual,
        &layer.gate_proj.packed_codes,
        &layer.gate_proj.scales,
        &layer.gate_proj.biases,
        num_tokens,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        &mut gate,
    )?;
    nf4tile640::dequant_matmul_reference(
        &residual,
        &layer.up_proj.packed_codes,
        &layer.up_proj.scales,
        &layer.up_proj.biases,
        num_tokens,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        &mut up,
    )?;

    // SiLU(gate) * up (element-wise)
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }

    // Down projection
    nf4tile640::dequant_matmul_reference(
        &gate,
        &layer.down_proj.packed_codes,
        &layer.down_proj.scales,
        &layer.down_proj.biases,
        num_tokens,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        &mut ffn_out,
    )?;

    // Add residual: normed was saved as the FFN input residual
    for i in 0..ffn_out.len() {
        ffn_out[i] += normed[i];
    }

    Ok(ffn_out)
}

// ════════════════════════════════════════════════════════════════════════════
// Top-level API
// ════════════════════════════════════════════════════════════════════════════

impl TtsCodePredictor {
    /// Factory: parse model weights from the cimage segment byte stream.
    ///
    /// The segment layout (contiguous byte stream, no padding):
    ///
    /// ```text
    /// [u32 little-endian: total_tiles across all nf4tile640 weights]
    /// [codes block:        total_tiles * PACKED_BYTES_PER_TILE bytes]
    /// [scales block:       total_tiles * SCALES_F32_PER_TILE f32 values]
    /// [biases block:       total_tiles * SCALES_F32_PER_TILE f32 values]
    /// [raw f32 block:      10 * HIDDEN_SIZE f32 values (5 layers × 2 norms)]
    /// ```
    ///
    /// All NF4 weights within the three tile-major blocks are stored in a
    /// fixed order matching the `parse_weights` calls below. The raw f32
    /// block stores the RMSNorm weight vectors packed consecutively.
    pub fn from_segments(weights: &[u8]) -> Result<Self, String> {
        if weights.len() < 4 {
            return Err(format!(
                "code_predictor segment too short: {} bytes",
                weights.len()
            ));
        }

        // The first 4 bytes encode the total tile count (little-endian u32).
        let total_tiles =
            u32::from_le_bytes([weights[0], weights[1], weights[2], weights[3]]) as usize;
        const HDR: usize = 4;

        // Compute bounds for the three component blocks
        let codes_end = HDR + total_tiles * nf4tile640::PACKED_BYTES_PER_TILE;
        let scales_end = codes_end + total_tiles * nf4tile640::SCALES_F32_PER_TILE * F32_BYTES;
        let biases_end = scales_end + total_tiles * nf4tile640::SCALES_F32_PER_TILE * F32_BYTES;
        let raw_norm_count = N_LAYERS * 2 * HIDDEN_SIZE;
        let total_needed = biases_end + raw_norm_count * F32_BYTES;

        if weights.len() < total_needed {
            return Err(format!(
                "code_predictor segment too short: have {} bytes, need {}",
                weights.len(),
                total_needed,
            ));
        }

        let codes_block = &weights[HDR..codes_end];
        let scales_block: &[f32] = bytemuck::cast_slice(&weights[codes_end..scales_end]);
        let biases_block: &[f32] = bytemuck::cast_slice(&weights[scales_end..biases_end]);
        let raw_norms_block: &[f32] = bytemuck::cast_slice(&weights[biases_end..total_needed]);

        // Track three separate cursors for the three flat blocks
        let mut co = 0; // byte offset into codes_block
        let mut so = 0; // f32 element offset into scales_block
        let mut bo = 0; // f32 element offset into biases_block
        let mut no = 0; // f32 element offset into raw_norms_block

        // ── Parse 5 layers ───────────────────────────────────────────────
        let mut layers = Vec::with_capacity(N_LAYERS);
        for _l in 0..N_LAYERS {
            let attn_q = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, N_HEADS * HEAD_DIM,
            )?;
            let attn_k = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM,
            )?;
            let attn_v = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM,
            )?;
            let attn_o = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                N_HEADS * HEAD_DIM, HIDDEN_SIZE,
            )?;
            let gate_proj = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, INTERMEDIATE_SIZE,
            )?;
            let up_proj = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, INTERMEDIATE_SIZE,
            )?;
            let down_proj = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                INTERMEDIATE_SIZE, HIDDEN_SIZE,
            )?;

            // Raw f32 norm weights follow all NF4 tiles
            let norm_end = no + HIDDEN_SIZE;
            let input_norm = raw_norms_block[no..norm_end].to_vec();
            no = norm_end;
            let post_end = no + HIDDEN_SIZE;
            let post_attn_norm = raw_norms_block[no..post_end].to_vec();
            no = post_end;

            layers.push(CodePredictLayer {
                attn_q,
                attn_k,
                attn_v,
                attn_o,
                gate_proj,
                up_proj,
                down_proj,
                input_norm,
                post_attn_norm,
            });
        }

        // ── Parse 15 output projection heads ─────────────────────────────
        let mut output_proj = Vec::with_capacity(NUM_CODEBOOKS);
        for _cb in 0..NUM_CODEBOOKS {
            let w = parse_weights(
                codes_block, scales_block, biases_block,
                &mut co, &mut so, &mut bo,
                HIDDEN_SIZE, HIDDEN_SIZE,
            )?;
            output_proj.push(w);
        }

        Ok(Self {
            layers,
            output_proj,
        })
    }

    /// Predict all 16 codebooks from Talker hidden states.
    ///
    /// `hidden` — [num_tokens, HIDDEN_SIZE] f32 — Talker's final hidden states
    /// for all autoregressive decode positions.
    ///
    /// Returns [num_tokens, TOTAL_CODEBOOKS] u32 — codebook indices per token.
    /// Codebook 0 is reserved as `0` (the Talker fills this via its argmax).
    /// Codebooks 1..=15 are predicted by this module.
    pub fn predict(&self, hidden: &[f32], num_tokens: usize) -> Result<Vec<u32>, String> {
        let expected = num_tokens * HIDDEN_SIZE;
        if hidden.len() != expected {
            return Err(format!(
                "CodePredictor input: hidden length {} != {} tokens * {} hidden_size",
                hidden.len(),
                num_tokens,
                HIDDEN_SIZE,
            ));
        }

        // ── Run the 5-layer transformer over hidden states ────────────────
        let mut x = hidden.to_vec();
        for layer in &self.layers {
            x = forward_layer(layer, &x, num_tokens)?;
        }
        // x is now [num_tokens, HIDDEN_SIZE] processed through all layers

        // ── Predict each of 15 codebooks via separate output heads ────────
        let mut logits = vec![0.0f32; num_tokens * HIDDEN_SIZE];
        // Allocation size: TOTAL_CODEBOOKS per token
        let mut result = vec![0u32; num_tokens * TOTAL_CODEBOOKS];

        for cb in 0..NUM_CODEBOOKS {
            let w = &self.output_proj[cb];

            // Project x through the codebook head:
            //   [num_tokens, 2048] @ [2048, 2048] -> [num_tokens, 2048] logits
            logits.fill(0.0f32);
            nf4tile640::dequant_matmul_reference(
                &x,
                &w.packed_codes,
                &w.scales,
                &w.biases,
                num_tokens,
                HIDDEN_SIZE,
                HIDDEN_SIZE,
                &mut logits,
            )?;

            // Argmax per token
            for t in 0..num_tokens {
                let offset = t * HIDDEN_SIZE;
                let slice = &logits[offset..offset + HIDDEN_SIZE];
                let max_idx = slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                // Codebook 0 (column 0) stays 0 — the Talker fills it.
                result[t * TOTAL_CODEBOOKS + cb + 1] = max_idx as u32;
            }
        }

        Ok(result)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Architecture constants are self-consistent.
    #[test]
    fn test_architecture_consistency() {
        assert_eq!(N_HEADS * HEAD_DIM, HIDDEN_SIZE);
        assert_eq!(N_HEADS / N_KV_HEADS, HEADS_PER_GROUP);
        assert!(N_HEADS % N_KV_HEADS == 0);
        assert_eq!(NUM_CODEBOOKS + 1, TOTAL_CODEBOOKS);
    }

    /// RMSNorm produces rows with unit RMS.
    #[test]
    fn test_rms_norm_unit_rms() {
        let num_tokens = 3;
        let mut x = vec![0.0f32; num_tokens * HIDDEN_SIZE];
        for i in 0..x.len() {
            x[i] = (i as f32) * 0.01 + 0.5;
        }
        let weight = vec![1.0f32; HIDDEN_SIZE];
        let mut out = vec![0.0f32; num_tokens * HIDDEN_SIZE];
        rms_norm(&x, &weight, num_tokens, HIDDEN_SIZE, &mut out);

        assert_eq!(out.len(), num_tokens * HIDDEN_SIZE);
        for t in 0..num_tokens {
            let offset = t * HIDDEN_SIZE;
            let mut sq_sum = 0.0f32;
            for i in 0..HIDDEN_SIZE {
                sq_sum += out[offset + i] * out[offset + i];
            }
            let rms = (sq_sum / HIDDEN_SIZE as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 0.01,
                "token {t}: RMS = {rms}, expected ~1.0"
            );
        }
    }

    /// SiLU on known values.
    #[test]
    fn test_silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(100.0) - 100.0).abs() < 1.0);
        assert!(silu(-100.0).abs() < 1e-6);
    }

    /// Uniform GQA input produces uniform output.
    #[test]
    fn test_gqa_attention_uniform() {
        let num_tokens = 2;
        let q = vec![0.1f32; num_tokens * N_HEADS * HEAD_DIM];
        let k = vec![0.2f32; num_tokens * N_KV_HEADS * HEAD_DIM];
        let v = vec![0.3f32; num_tokens * N_KV_HEADS * HEAD_DIM];
        let mut out = vec![0.0f32; num_tokens * N_HEADS * HEAD_DIM];
        gqa_attention(&q, &k, &v, num_tokens, &mut out);

        assert_eq!(out.len(), num_tokens * N_HEADS * HEAD_DIM);
        for i in 1..out.len() {
            assert!(
                (out[i] - out[0]).abs() < 1e-5,
                "uniform output not uniform at index {i}: {} vs {}",
                out[i],
                out[0],
            );
        }
    }

    /// from_segments rejects empty bytes.
    #[test]
    fn test_from_segments_empty() {
        assert!(TtsCodePredictor::from_segments(&[]).is_err());
    }

    /// from_segments rejects too-short bytes.
    #[test]
    fn test_from_segments_too_short() {
        assert!(TtsCodePredictor::from_segments(&[0u8; 3]).is_err());
    }

    /// predict rejects mismatched input length.
    #[test]
    fn test_predict_wrong_length_rejected() {
        // Build a minimal valid segment: total_tiles=0 since we don't need
        // actual weights for this test (the predict check happens before any
        // weight access when num_tokens is 0).
        let mut segment = Vec::new();
        segment.extend_from_slice(&0u32.to_le_bytes()); // total_tiles = 0
        // No NF4 tiles, just norms
        segment.extend_from_slice(&[0u8; 5 * 2 * HIDDEN_SIZE * F32_BYTES]);
        let predictor = TtsCodePredictor::from_segments(&segment).unwrap();

        // hidden is too short
        let hidden = vec![0.0f32; 10];
        let result = predictor.predict(&hidden, 1);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("hidden length"),
            "unexpected error: {err}"
        );
    }

    /// tiles_for arithmetic.
    #[test]
    fn test_tiles_arithmetic() {
        // [2048, 2048] = ceil(2048/640) * 2048 = 4 * 2048 = 8192 tiles
        // Actually: cols/640 = ceil(2048/640) = 4, rows=2048 => 2048*4 = 8192
        let t = tiles_for(HIDDEN_SIZE, HIDDEN_SIZE);
        assert_eq!(t, 8192);
        assert_eq!(code_bytes(t), 8192 * 320);
        assert_eq!(scale_count(t), 8192 * 5);
    }
}
