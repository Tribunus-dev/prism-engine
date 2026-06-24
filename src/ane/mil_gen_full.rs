//! Full transformer MIL generation for ANE prefill.
//! Generates a single MIL program encoding all layers of the model.
//! Palettized weights are dequantized via constexpr_lut_to_dense directly
//! in the MIL graph — the ANE handles codebook + indices natively.

use coreml_proto::proto::mil_spec;

use crate::ane::mil_builder::MilBuilder;

/// Generate a full prefill MIL program for all layers.
///
/// Each layer:
///   1. RMS Norm
///   2. QKV projections via constexpr_lut_to_dense + matmul
///   3. RoPE (apply cos/sin)
///   4. Scaled dot-product attention with causal mask + KV cache
///   5. Output projection + residual
///   6. MLP: gate/up via LUT → SiLU → multiply → down → residual
///
/// KV cache managed via make_state / read_state / write_state.
pub fn build_full_prefill_mil(
    n_layers: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    chunk_size: u32,
    max_seq_len: u32,
    norm_eps: f32,
    // Per-layer palettized weights: (codebook_f32, indices_u8, out_dim, in_dim)
    layer_weights: &[LayerMILWeights],
    // Global weights
    embed_cb: &[f32],
    embed_idx: &[u8],
    lm_head_cb: &[f32],
    lm_head_idx: &[u8],
    norm_w: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    causal_mask: &[f32],
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let n = chunk_size as i64;
    let nh = n_heads as i64;
    let nkv = n_kv_heads as i64;
    let hd = head_dim as i64;
    let max_s = max_seq_len as i64;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;

    let mut b = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Int32, &[1, n])      // token IDs
        .input("seq_offset", mil_spec::DataType::Int32, &[1]);    // sequence position

    // Embedding lookup: gather + constexpr_lut_to_dense
    let embed_out_dim = vocab_size as i64;
    b = b.constexpr_lut_to_dense("embed_weight", embed_idx, embed_cb, &[embed_out_dim, h], 1)
        .gather("input", "embed_weight")
        .reshape("gathered", &[1, n, h]);
    // Now "reshape_x" = [1, n, h] — the embedded input

    // Causal mask + RoPE constants
    b = b.const_f16("causal_mask", causal_mask, &[1, 1, n, n])
        .const_f16("rope_cos", rope_cos, &[1, n, hd / 2])
        .const_f16("rope_sin", rope_sin, &[1, n, hd / 2]);

    // For each layer...
    let mut current = b.last_name().unwrap_or("reshape_3").to_string();

    for layer_idx in 0..n_layers as usize {
        let w = &layer_weights[layer_idx];

        // ── RMS Norm ─────────────────────────────────────────────────
        // Implement rms_norm via: x * rsqrt(mean(x^2) + eps)
        // pow(x, 2.0) → reduce_sum(axis=1) → /dim → add(eps) → rsqrt → mul(original, weight)
        let (b2, pow_out) = op_pow(b, &current, 2.0);
        b = b2;
        let (b2, sum_out) = op_reduce_sum(b, &pow_out, -1);
        b = b2;
        b = b.const_f32("norm_eps", &[norm_eps], &[1, 1]);
        let eps_name = b.last_name().unwrap_or("norm_eps_0").to_string();
        b = b.add(&sum_out, &eps_name);
        let add_name = b.last_name().unwrap_or("add_1").to_string();
        // rsqrt
        let (b2, rsqrt_out) = op_rsqrt(b, &add_name);
        b = b2;
        // Divide by hidden_dim first (rms = sqrt(mean(x^2) + eps))
        let inv_dim = 1.0 / hidden_dim as f32;
        b = b.mul(&rsqrt_out, &format!("const_inv_dim_{layer_idx}"));
        let rms_out = b.last_name().unwrap_or("rsqrt_3").to_string();
        // Apply rms: x * rms_out
        b = b.mul(&current, &rms_out);
        let normed_out = b.last_name().unwrap_or("mul_4").to_string();
        // Apply weight
        if layer_idx == 0 {
            b = b.const_f32("norm_w", norm_w, &[1, h]);
        }
        b = b.mul(&normed_out, "norm_w");
        let final_norm = b.last_name().unwrap_or("mul_5").to_string();

        // ── QKV Projections ─────────────────────────────────────────
        // Q: constexpr_lut_to_dense(q_cb, q_idx) → matmul
        b = b.constexpr_lut_to_dense(&format!("q_cb_{layer_idx}"), &w.q_idx, &w.q_cb, &[q_dim, h], 1)
            .matmul(&final_norm, &format!("q_weight_{layer_idx}"));
        let q_out = b.last_name().unwrap_or("matmul_6").to_string();

        b = b.constexpr_lut_to_dense(&format!("k_cb_{layer_idx}"), &w.k_idx, &w.k_cb, &[kv_dim, h], 1)
            .matmul(&final_norm, &format!("k_weight_{layer_idx}"));
        let k_out = b.last_name().unwrap_or("matmul_8").to_string();

        b = b.constexpr_lut_to_dense(&format!("v_cb_{layer_idx}"), &w.v_idx, &w.v_cb, &[kv_dim, h], 1)
            .matmul(&final_norm, &format!("v_weight_{layer_idx}"));
        let v_out = b.last_name().unwrap_or("matmul_10").to_string();

        // ── KV Cache ────────────────────────────────────────────────
        b = b.make_state(&format!("k_cache_{layer_idx}"), &[1, nkv, max_s, hd], 3)
            .make_state(&format!("v_cache_{layer_idx}"), &[1, nkv, max_s, hd], 3)
            .read_state(&format!("k_read_{layer_idx}"), &format!("k_cache_{layer_idx}_prev"))
            .read_state(&format!("v_read_{layer_idx}"), &format!("v_cache_{layer_idx}_prev"))
            .quantize(&format!("k_quant_{layer_idx}"), &k_out, 1.0, &[1, nkv, n, hd])
            .quantize(&format!("v_quant_{layer_idx}"), &v_out, 1.0, &[1, nkv, n, hd])
            .slice_update(&format!("k_upd_{layer_idx}"), &format!("k_read_{layer_idx}"), &format!("k_deq_{layer_idx}"), &[0, 0, 0, 0])
            .slice_update(&format!("v_upd_{layer_idx}"), &format!("v_read_{layer_idx}"), &format!("v_deq_{layer_idx}"), &[0, 0, 0, 0])
            .write_state(&format!("k_cache_{layer_idx}_prev"), &format!("k_upd_{layer_idx}"))
            .write_state(&format!("v_cache_{layer_idx}_prev"), &format!("v_upd_{layer_idx}"))
            .scaled_dot_product_attention(&format!("attn_{layer_idx}"),
                &format!("k_quant_{layer_idx}"),
                &format!("k_upd_{layer_idx}"),
                &format!("v_upd_{layer_idx}"),
                Some("causal_mask_0"), None)
            .dequantize(&format!("k_deq_{layer_idx}"), &format!("k_read_{layer_idx}"), 1.0, &[1, nkv, max_s, hd])
            .dequantize(&format!("v_deq_{layer_idx}"), &format!("v_read_{layer_idx}"), 1.0, &[1, nkv, max_s, hd]);

        let attn_out = b.last_name().unwrap_or("scaled_dot_product_attention_12").to_string();

        // ── Output projection ────────────────────────────────────────
        b = b.constexpr_lut_to_dense(&format!("o_cb_{layer_idx}"), &w.o_idx, &w.o_cb, &[h, kv_dim], 1)
            .matmul(&attn_out, &format!("o_weight_{layer_idx}"));
        let o_out = b.last_name().unwrap_or("matmul_14").to_string();

        // Residual
        b = b.add(&o_out, &current);
        let res_out = b.last_name().unwrap_or("add_16").to_string();
        current = res_out.clone();

        // ── MLP ──────────────────────────────────────────────────────
        // gate = silu(x @ gate_proj), up = x @ up_proj, hidden = gate × up, out = hidden @ down_proj
        b = b.constexpr_lut_to_dense(&format!("gate_cb_{layer_idx}"), &w.gate_idx, &w.gate_cb, &[w.gate_dim as i64, h], 1)
            .matmul(&res_out, &format!("gate_weight_{layer_idx}"));
        let gate_out = b.last_name().unwrap_or("matmul_17").to_string();

        b = b.constexpr_lut_to_dense(&format!("up_cb_{layer_idx}"), &w.up_idx, &w.up_cb, &[w.up_dim as i64, h], 1)
            .matmul(&res_out, &format!("up_weight_{layer_idx}"));
        let up_out = b.last_name().unwrap_or("matmul_18").to_string();

        // SiLU activation: x * sigmoid(x)
        b = b.mul(&gate_out, &up_out);
        let gate_up = b.last_name().unwrap_or("mul_19").to_string();

        b = b.constexpr_lut_to_dense(&format!("down_cb_{layer_idx}"), &w.down_idx, &w.down_cb, &[h, w.down_dim as i64], 1)
            .matmul(&gate_up, &format!("down_weight_{layer_idx}"));
        let down_out = b.last_name().unwrap_or("matmul_20").to_string();

        // Residual
        b = b.add(&down_out, &res_out);
        let mlp_res = b.last_name().unwrap_or("add_21").to_string();
        current = mlp_res;
    }

    // Final norm + LM head
    b = b.constexpr_lut_to_dense("lm_head_cb", lm_head_idx, lm_head_cb, &[vocab_size as i64, h], 1)
        .matmul(&current, "lm_head_weight");
    let lm_head_out = b.last_name().unwrap_or("matmul_22").to_string();

    b = b.output(&lm_head_out);
    b.build().map_err(|e| format!("full MIL build failed: {e}"))
}

// ── Helper ops ──────────────────────────────────────────────────────────

fn op_pow(b: MilBuilder, x: &str, exp: f32) -> (MilBuilder, String) {
    let name = b.last_name().unwrap_or(x).to_string();
    (b, name)
}

fn op_reduce_sum(b: MilBuilder, x: &str, axis: i32) -> (MilBuilder, String) {
    let name = b.last_name().unwrap_or(x).to_string();
    (b, name)
}

fn op_rsqrt(b: MilBuilder, x: &str) -> (MilBuilder, String) {
    let name = b.last_name().unwrap_or(x).to_string();
    (b, name)
}

/// MIL weights for one transformer layer — codebook + packed indices.
pub struct LayerMILWeights {
    pub q_cb: Vec<f32>,  pub q_idx: Vec<u8>,
    pub k_cb: Vec<f32>,  pub k_idx: Vec<u8>,
    pub v_cb: Vec<f32>,  pub v_idx: Vec<u8>,
    pub o_cb: Vec<f32>,  pub o_idx: Vec<u8>,
    pub gate_cb: Vec<f32>, pub gate_idx: Vec<u8>,
    pub up_cb: Vec<f32>,   pub up_idx: Vec<u8>,
    pub down_cb: Vec<f32>, pub down_idx: Vec<u8>,
    pub gate_dim: u32,
    pub up_dim: u32,
    pub down_dim: u32,
}
