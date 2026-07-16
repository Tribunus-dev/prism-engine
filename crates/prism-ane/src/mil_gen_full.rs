//! Full transformer MIL generation for ANE prefill.
//! Generates a single MIL program encoding all layers of the model.
//! Palettized weights are dequantized via constexpr_lut_to_dense directly
//! in the MIL graph — the ANE handles codebook + indices natively.

use std::collections::HashMap;

use coreml_proto::proto::mil_spec;
use coreml_proto::proto::mil_spec::argument;

use crate::mil_builder::MilBuilder;
use crate::mil_helpers::{
    make_operation, named_arg, op_composite_silu, op_pow, op_reduce_sum, op_rsqrt,
    scalar_value_type, string_attr, tensor_type, value_type_tensor,
};

/// Generate a full prefill MIL program for all layers.
pub fn build_full_prefill_mil(
    n_layers: u32,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    _intermediate_dim: u32,
    vocab_size: u32,
    chunk_size: u32,
    max_seq_len: u32,
    norm_eps: f32,
    layer_weights: &[LayerMILWeights],
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
    let vocab = vocab_size as i64;

    let mut b = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Int32, &[1, n])
        .input("seq_offset", mil_spec::DataType::Int32, &[1]);

    // ── Embedding LUT ────────────────────────────────────────────────
    let r = op_lut_to_dense(b, "embed_weight", embed_idx, embed_cb, &[vocab, h]);
    b = r.0;
    let embed_weight = r.1;
    let r = op_gather(b, &embed_weight, "input", &[1, n * h]);
    b = r.0;
    let embed_1d = r.1;
    let r = op_reshape(b, &embed_1d, &[1, n, h]);
    b = r.0;
    let cur = r.1;

    // ── Constants: RoPE tables + causal mask ─────────────────────────
    let r = op_const_f16(b, "rope_cos", rope_cos, &[1, n, hd / 2]);
    b = r.0;
    let rope_cos_name = r.1;
    let r = op_const_f16(b, "rope_sin", rope_sin, &[1, n, hd / 2]);
    b = r.0;
    let rope_sin_name = r.1;
    let r = op_const_f16(b, "causal_mask", causal_mask, &[1, 1, n, n]);
    b = r.0;
    let causal_mask_name = r.1;
    let r = op_const_f16(b, "norm_w", norm_w, &[1, h]);
    b = r.0;
    let norm_w_name = r.1;

    // ── For each layer ───────────────────────────────────────────────
    let mut cur = cur;
    assert!(n_layers > 0);

    for layer_idx in 0..n_layers as usize {
        let w = &layer_weights[layer_idx];

        // 1. RMS Norm
        let r = rms_norm_3d(b, &cur, &norm_w_name, norm_eps, n, h);
        b = r.0;
        let normed = r.1;

        // 2. QKV projections via LUT+matmul
        let r = op_lut_to_dense(
            b,
            &format!("q_lut_{layer_idx}"),
            &w.q_idx,
            &w.q_cb,
            &[q_dim, h],
        );
        b = r.0;
        let q_weight_name = r.1;
        let r = op_matmul(b, &normed, &q_weight_name);
        b = r.0;
        let q_raw = r.1;

        let r = op_lut_to_dense(
            b,
            &format!("k_lut_{layer_idx}"),
            &w.k_idx,
            &w.k_cb,
            &[kv_dim, h],
        );
        b = r.0;
        let k_weight_name = r.1;
        let r = op_matmul(b, &normed, &k_weight_name);
        b = r.0;
        let k_raw = r.1;

        let r = op_lut_to_dense(
            b,
            &format!("v_lut_{layer_idx}"),
            &w.v_idx,
            &w.v_cb,
            &[kv_dim, h],
        );
        b = r.0;
        let v_weight_name = r.1;
        let r = op_matmul(b, &normed, &v_weight_name);
        b = r.0;
        let v_raw = r.1;

        // 3. RoPE
        let r = op_reshape(b, &q_raw, &[n * nh, hd]);
        b = r.0;
        let q_flat = r.1;
        let r = op_reshape(b, &k_raw, &[n * nkv, hd]);
        b = r.0;
        let k_flat = r.1;

        let r = op_repeat_interleave(b, &rope_cos_name, nh, &[n * nh, hd / 2]);
        b = r.0;
        let q_cos = r.1;
        let r = op_repeat_interleave(b, &rope_sin_name, nh, &[n * nh, hd / 2]);
        b = r.0;
        let q_sin = r.1;
        let r = op_repeat_interleave(b, &rope_cos_name, nkv, &[n * nkv, hd / 2]);
        b = r.0;
        let k_cos = r.1;
        let r = op_repeat_interleave(b, &rope_sin_name, nkv, &[n * nkv, hd / 2]);
        b = r.0;
        let k_sin = r.1;

        // Q RoPE: split last dim, rotate, concat
        let r = op_slice_last_dim(b, &q_flat, 0, hd / 2);
        b = r.0;
        let q_left = r.1;
        let r = op_slice_last_dim(b, &q_flat, hd / 2, hd / 2);
        b = r.0;
        let q_right = r.1;

        let r = op_mul(b, &q_left, &q_cos);
        b = r.0;
        let q_lc = r.1;
        let r = op_mul(b, &q_right, &q_sin);
        b = r.0;
        let q_rs = r.1;
        let r = op_sub(b, &q_lc, &q_rs);
        b = r.0;
        let q_new_left = r.1;

        let r = op_mul(b, &q_left, &q_sin);
        b = r.0;
        let q_ls = r.1;
        let r = op_mul(b, &q_right, &q_cos);
        b = r.0;
        let q_rc = r.1;
        let r = op_add(b, &q_ls, &q_rc);
        b = r.0;
        let q_new_right = r.1;

        let r = op_concat(b, &q_new_left, &q_new_right, -1);
        b = r.0;
        let q_rope = r.1;

        // K RoPE: split last dim, rotate, concat
        let r = op_slice_last_dim(b, &k_flat, 0, hd / 2);
        b = r.0;
        let k_left = r.1;
        let r = op_slice_last_dim(b, &k_flat, hd / 2, hd / 2);
        b = r.0;
        let k_right = r.1;

        let r = op_mul(b, &k_left, &k_cos);
        b = r.0;
        let k_lc = r.1;
        let r = op_mul(b, &k_right, &k_sin);
        b = r.0;
        let k_rs = r.1;
        let r = op_sub(b, &k_lc, &k_rs);
        b = r.0;
        let k_new_left = r.1;

        let r = op_mul(b, &k_left, &k_sin);
        b = r.0;
        let k_ls = r.1;
        let r = op_mul(b, &k_right, &k_cos);
        b = r.0;
        let k_rc = r.1;
        let r = op_add(b, &k_ls, &k_rc);
        b = r.0;
        let k_new_right = r.1;

        let r = op_concat(b, &k_new_left, &k_new_right, -1);
        b = r.0;
        let k_rope = r.1;

        // Reshape for SDPA
        let r = op_reshape(b, &q_rope, &[1, nh, n, hd]);
        b = r.0;
        let q_sdpa = r.1;
        let r = op_reshape(b, &k_rope, &[1, nkv, n, hd]);
        b = r.0;
        let k_sdpa = r.1;
        let r = op_reshape(b, &v_raw, &[1, nkv, n, hd]);
        b = r.0;
        let v_3d = r.1;

        // 4. KV Cache
        let r = op_make_state(b, &format!("k_cache_{layer_idx}"), &[1, nkv, max_s, hd]);
        b = r.0;
        let k_cache_state = r.1;
        let r = op_make_state(b, &format!("v_cache_{layer_idx}"), &[1, nkv, max_s, hd]);
        b = r.0;
        let v_cache_state = r.1;

        let r = op_read_state(b, &format!("k_read_{layer_idx}"), &k_cache_state);
        b = r.0;
        let k_cache_read = r.1;
        let r = op_read_state(b, &format!("v_read_{layer_idx}"), &v_cache_state);
        b = r.0;
        let v_cache_read = r.1;

        let r = op_quantize(
            b,
            &format!("k_quant_{layer_idx}"),
            &k_sdpa,
            1.0,
            &[1, nkv, n, hd],
        );
        b = r.0;
        let k_quant = r.1;
        let r = op_quantize(
            b,
            &format!("v_quant_{layer_idx}"),
            &v_3d,
            1.0,
            &[1, nkv, n, hd],
        );
        b = r.0;
        let v_quant = r.1;

        let r = op_slice_update(
            b,
            &format!("k_upd_{layer_idx}"),
            &k_cache_read,
            &k_quant,
            &[0, 0, 0, 0],
        );
        b = r.0;
        let k_cache_upd = r.1;
        let r = op_slice_update(
            b,
            &format!("v_upd_{layer_idx}"),
            &v_cache_read,
            &v_quant,
            &[0, 0, 0, 0],
        );
        b = r.0;
        let v_cache_upd = r.1;

        let r = op_write_state(b, &k_cache_state, &k_cache_upd);
        b = r.0;
        let r = op_write_state(b, &v_cache_state, &v_cache_upd);
        b = r.0;

        let r = op_dequantize(
            b,
            &format!("k_deq_{layer_idx}"),
            &k_cache_upd,
            1.0,
            &[1, nkv, max_s, hd],
        );
        b = r.0;
        let k_cache_fp = r.1;
        let r = op_dequantize(
            b,
            &format!("v_deq_{layer_idx}"),
            &v_cache_upd,
            1.0,
            &[1, nkv, max_s, hd],
        );
        b = r.0;
        let v_cache_fp = r.1;

        // 5. SDPA
        let r = op_scaled_dot_product_attention(
            b,
            &format!("attn_{layer_idx}"),
            &q_sdpa,
            &k_cache_fp,
            &v_cache_fp,
            Some(&causal_mask_name),
            None,
        );
        b = r.0;
        let attn_out = r.1;

        // 6. Output projection + residual
        let r = op_lut_to_dense(
            b,
            &format!("o_lut_{layer_idx}"),
            &w.o_idx,
            &w.o_cb,
            &[h, nh * hd],
        );
        b = r.0;
        let o_weight_name = r.1;
        let r = op_reshape(b, &attn_out, &[1, n, nh * hd]);
        b = r.0;
        let attn_flat = r.1;
        let r = op_matmul(b, &attn_flat, &o_weight_name);
        b = r.0;
        let o_proj = r.1;
        let r = op_add(b, &o_proj, &cur);
        b = r.0;
        let res1 = r.1;

        // 7. RMS Norm (second norm before MLP)
        let r = rms_norm_3d(b, &res1, &norm_w_name, norm_eps, n, h);
        b = r.0;
        let normed2 = r.1;

        // 8. MLP
        let r = op_lut_to_dense(
            b,
            &format!("gate_lut_{layer_idx}"),
            &w.gate_idx,
            &w.gate_cb,
            &[w.gate_dim as i64, h],
        );
        b = r.0;
        let gate_weight_name = r.1;
        let r = op_matmul(b, &normed2, &gate_weight_name);
        b = r.0;
        let gate_proj = r.1;
        let r = op_composite_silu(b, &gate_proj);
        b = r.0;
        let gate_act = r.1;

        let r = op_lut_to_dense(
            b,
            &format!("up_lut_{layer_idx}"),
            &w.up_idx,
            &w.up_cb,
            &[w.up_dim as i64, h],
        );
        b = r.0;
        let up_weight_name = r.1;
        let r = op_matmul(b, &normed2, &up_weight_name);
        b = r.0;
        let up_proj = r.1;

        let r = op_mul(b, &gate_act, &up_proj);
        b = r.0;
        let mlp_hidden = r.1;

        let r = op_lut_to_dense(
            b,
            &format!("down_lut_{layer_idx}"),
            &w.down_idx,
            &w.down_cb,
            &[h, w.down_dim as i64],
        );
        b = r.0;
        let down_weight_name = r.1;
        let r = op_matmul(b, &mlp_hidden, &down_weight_name);
        b = r.0;
        let down_proj = r.1;

        let r = op_add(b, &down_proj, &res1);
        b = r.0;
        let cur_next = r.1;
        cur = cur_next;
    }

    // ── Final RMS Norm + LM Head ──────────────────────────────────────
    let r = rms_norm_3d(b, &cur, &norm_w_name, norm_eps, n, h);
    b = r.0;
    let final_norm = r.1;

    let r = op_lut_to_dense(b, "lm_head_weight", lm_head_idx, lm_head_cb, &[vocab, h]);
    b = r.0;
    let lm_head_weight = r.1;
    let r = op_matmul(b, &final_norm, &lm_head_weight);
    b = r.0;
    let lm_out = r.1;

    let b = b.output(&lm_out);
    b.build().map_err(|e| format!("full MIL build failed: {e}"))
}

// ── High-Level Ops ──────────────────────────────────────────────────────

fn rms_norm_3d(
    b: MilBuilder,
    input: &str,
    norm_w: &str,
    eps: f32,
    _n: i64,
    h: i64,
) -> (MilBuilder, String) {
    let (b, pow_out) = op_pow(b, input, 2.0);
    let (b, sum_out) = op_reduce_sum(b, &pow_out, -1);
    let inv_h = 1.0 / h as f32;
    let (b, inv_h_name) = op_const_scalar(b, "inv_h", inv_h);
    let (b, mean_out) = op_mul(b, &sum_out, &inv_h_name);
    let (b, eps_name) = op_const_scalar(b, "rms_eps", eps);
    let (b, biased) = op_add(b, &mean_out, &eps_name);
    let (b, rms_out) = op_rsqrt(b, &biased);
    let (b, scaled) = op_mul(b, input, &rms_out);
    let (b, result) = op_mul(b, &scaled, norm_w);
    (b, result)
}

fn op_gather(
    b: MilBuilder,
    weight: &str,
    indices: &str,
    output_shape: &[i64],
) -> (MilBuilder, String) {
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, output_shape));
    let out_name = format!("gather_{}", b.ops().len());
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(weight));
    inputs.insert("indices".to_string(), named_arg(indices));
    let op = make_operation("gather", &out_name, inputs, &vt, HashMap::new());
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_const_f16(
    b: MilBuilder,
    name_hint: &str,
    values: &[f32],
    shape: &[i64],
) -> (MilBuilder, String) {
    let name = format!("{}_{}", name_hint, b.ops().len());
    let b = b.const_f16(&name, values, shape);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or(name);
    (b, out)
}

fn op_const_uint8(
    b: MilBuilder,
    name_hint: &str,
    values: &[u8],
    shape: &[i64],
) -> (MilBuilder, String) {
    let name = format!("{}_{}", name_hint, b.ops().len());
    let tt = tensor_type(mil_spec::DataType::Uint8, shape);
    let vt = value_type_tensor(tt.clone());
    let tv = mil_spec::TensorValue {
        value: Some(mil_spec::tensor_value::Value::Bytes(
            mil_spec::tensor_value::RepeatedBytes {
                values: values.to_vec(),
            },
        )),
    };
    let val = mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(vt.clone()),
        value: Some(mil_spec::value::Value::ImmediateValue(
            mil_spec::value::ImmediateValue {
                value: Some(mil_spec::value::immediate_value::Value::Tensor(tv)),
            },
        )),
    };
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&name));
    attrs.insert("val".to_string(), val);
    let op = make_operation("const", &name, HashMap::new(), &vt, attrs);
    let b = b.operation(op, Some((name.as_str(), vt)));
    (b, name)
}

fn op_lut_to_dense(
    b: MilBuilder,
    name_hint: &str,
    indices: &[u8],
    codebook: &[f32],
    out_shape: &[i64],
) -> (MilBuilder, String) {
    let r = op_const_f16(b, &format!("{name_hint}_cb"), codebook, &[out_shape[0], 16]);
    let b = r.0;
    let cb_name = r.1;
    let r = op_const_uint8(b, &format!("{name_hint}_idx"), indices, &[out_shape[0], 1]);
    let b = r.0;
    let idx_name = r.1;
    let out_name = format!("{name_hint}_dense_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, out_shape));
    let mut inputs = HashMap::new();
    inputs.insert("indices".to_string(), named_arg(&idx_name));
    inputs.insert("lut".to_string(), named_arg(&cb_name));
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&out_name));
    attrs.insert("vector_axis".to_string(), int64_attr(1));
    let op = make_operation("constexpr_lut_to_dense", &out_name, inputs, &vt, attrs);
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_matmul(b: MilBuilder, a: &str, b_name: &str) -> (MilBuilder, String) {
    let b = b.matmul(a, b_name);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_add(b: MilBuilder, a: &str, b_name: &str) -> (MilBuilder, String) {
    let b = b.add(a, b_name);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_mul(b: MilBuilder, a: &str, b_name: &str) -> (MilBuilder, String) {
    let b = b.mul(a, b_name);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_sub(b: MilBuilder, a: &str, b_name: &str) -> (MilBuilder, String) {
    let out_name = format!("sub_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, &[1, 1]));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(a));
    inputs.insert("y".to_string(), named_arg(b_name));
    let op = make_operation("sub", &out_name, inputs, &vt, HashMap::new());
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_concat(b: MilBuilder, a: &str, b_name: &str, axis: i64) -> (MilBuilder, String) {
    let out_name = format!("cat_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, &[1, 1]));
    let mut inputs = HashMap::new();
    let args = vec![
        argument::Binding {
            binding: Some(argument::binding::Binding::Name(a.to_string())),
        },
        argument::Binding {
            binding: Some(argument::binding::Binding::Name(b_name.to_string())),
        },
    ];
    inputs.insert("values".to_string(), mil_spec::Argument { arguments: args });
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&out_name));
    attrs.insert("axis".to_string(), int64_attr(axis));
    let op = make_operation("concat", &out_name, inputs, &vt, attrs);
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_slice_last_dim(b: MilBuilder, input: &str, start: i64, length: i64) -> (MilBuilder, String) {
    let out_name = format!("slice_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, &[1, 1]));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&out_name));
    attrs.insert("begin".to_string(), int64_attr(start));
    attrs.insert("end".to_string(), int64_attr(start + length));
    attrs.insert("stride".to_string(), int64_attr(1));
    attrs.insert("begin_mask".to_string(), int64_attr(0));
    attrs.insert("end_mask".to_string(), int64_attr(0));
    attrs.insert("squeeze_mask".to_string(), int64_attr(0));
    let op = make_operation("slice", &out_name, inputs, &vt, attrs);
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_repeat_interleave(
    b: MilBuilder,
    input: &str,
    repeat: i64,
    output_shape: &[i64],
) -> (MilBuilder, String) {
    let out_name = format!("repeat_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, output_shape));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&out_name));
    attrs.insert("rep".to_string(), int64_attr(repeat));
    attrs.insert("axis".to_string(), int64_attr(0));
    let op = make_operation("repeat", &out_name, inputs, &vt, attrs);
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_const_scalar(b: MilBuilder, name_hint: &str, val: f32) -> (MilBuilder, String) {
    op_const_f16(b, name_hint, &[val], &[1, 1])
}

fn op_reshape(b: MilBuilder, input: &str, shape: &[i64]) -> (MilBuilder, String) {
    let out_name = format!("reshape_{}", b.ops().len());
    let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, shape));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let mut attrs = HashMap::new();
    attrs.insert("name".to_string(), string_attr(&out_name));
    attrs.insert("shape".to_string(), int64s_attr(shape));
    let op = make_operation("reshape", &out_name, inputs, &vt, attrs);
    let b = b.operation(op, Some((out_name.as_str(), vt)));
    (b, out_name)
}

fn op_quantize(
    b: MilBuilder,
    name_hint: &str,
    input: &str,
    scale: f32,
    shape: &[i64],
) -> (MilBuilder, String) {
    let b = b.quantize(name_hint, input, scale, shape);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_dequantize(
    b: MilBuilder,
    name_hint: &str,
    input: &str,
    scale: f32,
    shape: &[i64],
) -> (MilBuilder, String) {
    let b = b.dequantize(name_hint, input, scale, shape);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_slice_update(
    b: MilBuilder,
    name_hint: &str,
    input: &str,
    source: &str,
    starts: &[i64],
) -> (MilBuilder, String) {
    let b = b.slice_update(name_hint, input, source, starts);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_make_state(b: MilBuilder, name_hint: &str, shape: &[i64]) -> (MilBuilder, String) {
    let b = b.make_state(name_hint, shape, 10);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_read_state(b: MilBuilder, name_hint: &str, state_ssa: &str) -> (MilBuilder, String) {
    let b = b.read_state(name_hint, state_ssa);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

fn op_write_state(b: MilBuilder, state_ssa: &str, value_ssa: &str) -> (MilBuilder, String) {
    let b = b.write_state(state_ssa, value_ssa);
    let out = format!("ws_{}", b.ops().len());
    (b, out)
}

fn op_scaled_dot_product_attention(
    b: MilBuilder,
    name_hint: &str,
    query: &str,
    key: &str,
    value: &str,
    mask: Option<&str>,
    scale: Option<f32>,
) -> (MilBuilder, String) {
    let b = b.scaled_dot_product_attention(name_hint, query, key, value, mask, scale);
    let out = b.last_name().map(|s| s.to_string()).unwrap_or_default();
    (b, out)
}

// ── Attribute helpers ───────────────────────────────────────────────────

fn int64_attr(val: i64) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(mil_spec::tensor_value::Value::LongInts(
            mil_spec::tensor_value::RepeatedLongInts { values: vec![val] },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Int64)),
        value: Some(mil_spec::value::Value::ImmediateValue(
            mil_spec::value::ImmediateValue {
                value: Some(mil_spec::value::immediate_value::Value::Tensor(int_tensor)),
            },
        )),
    }
}

fn int64s_attr(vals: &[i64]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(mil_spec::tensor_value::Value::LongInts(
            mil_spec::tensor_value::RepeatedLongInts {
                values: vals.to_vec(),
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: None,
        value: Some(mil_spec::value::Value::ImmediateValue(
            mil_spec::value::ImmediateValue {
                value: Some(mil_spec::value::immediate_value::Value::Tensor(int_tensor)),
            },
        )),
    }
}

// ── Layer weights struct ─────────────────────────────────────────────────

/// MIL weights for one transformer layer — codebook + packed indices.
pub struct LayerMILWeights {
    pub gate_dim: u32,
    pub up_dim: u32,
    pub down_dim: u32,
    pub q_cb: Vec<f32>,
    pub q_idx: Vec<u8>,
    pub k_cb: Vec<f32>,
    pub k_idx: Vec<u8>,
    pub v_cb: Vec<f32>,
    pub v_idx: Vec<u8>,
    pub o_cb: Vec<f32>,
    pub o_idx: Vec<u8>,
    pub gate_cb: Vec<f32>,
    pub gate_idx: Vec<u8>,
    pub up_cb: Vec<f32>,
    pub up_idx: Vec<u8>,
    pub down_cb: Vec<f32>,
    pub down_idx: Vec<u8>,
}
