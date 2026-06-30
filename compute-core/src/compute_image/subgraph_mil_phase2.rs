use crate::mil_builder::MilBuilder;
use coreml_proto::proto::mil_spec;

pub fn build_qkv_bundle_mil_palettized(
    input_name: &str,
    hidden_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    q_cb: &[f32],
    q_idx: &[u8],
    k_cb: &[f32],
    k_idx: &[u8],
    v_cb: &[f32],
    v_idx: &[u8],
) -> Result<mil_spec::Program, String> {
    let h = hidden_dim as i64;
    let q_dim = (n_heads * head_dim) as i64;
    let kv_dim = (n_kv_heads * head_dim) as i64;
    let b = MilBuilder::new("main").input(input_name, mil_spec::DataType::Float32, &[1, h]);
    let b = b
        .const_uint8("q_idx", q_idx, &[q_dim, h / 2])
        .const_f16("q_cb", q_cb, &[q_dim, 16]);
    let b = b
        .constexpr_lut_to_dense("q_weight", "q_idx_0", "q_cb_1", &[q_dim, h], 1)
        .matmul(input_name, "q_weight_2");
    let b = b
        .const_uint8("k_idx", k_idx, &[kv_dim, h / 2])
        .const_f16("k_cb", k_cb, &[kv_dim, 16]);
    let b = b
        .constexpr_lut_to_dense("k_weight", "k_idx_4", "k_cb_5", &[kv_dim, h], 1)
        .matmul(input_name, "k_weight_6");
    let b = b
        .const_uint8("v_idx", v_idx, &[kv_dim, h / 2])
        .const_f16("v_cb", v_cb, &[kv_dim, 16]);
    let b = b
        .constexpr_lut_to_dense("v_weight", "v_idx_8", "v_cb_9", &[kv_dim, h], 1)
        .matmul(input_name, "v_weight_10");
    b.output("matmul_3")
        .output("matmul_7")
        .output("matmul_11")
        .build()
        .map_err(|e| format!("QKV palettized MIL build failed: {e}"))
}

pub fn build_stateful_prefill_graph(
    chunk_size: u32,
    hidden_dim: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    causal_mask: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    q_cb: &[f32],
    q_idx: &[u8],
    k_cb: &[f32],
    k_idx: &[u8],
    v_cb: &[f32],
    v_idx: &[u8],
    kv_scale: f32,
) -> Result<mil_spec::Program, String> {
    let n = chunk_size as i64;
    let h = hidden_dim as i64;
    let hd = head_dim as i64;
    let kvh = n_kv_heads as i64;
    let max_s = max_seq_len as i64;
    let kv_dim = kvh * hd;
    let b = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Float32, &[1, n, h])
        .input("seq_offset", mil_spec::DataType::Int32, &[1])
        .const_f16("causal_mask", causal_mask, &[1, 1, n, n])
        .const_f16("rope_cos", rope_cos, &[1, n, hd / 2])
        .const_f16("rope_sin", rope_sin, &[1, n, hd / 2])
        .make_state("k_cache", &[1, kvh, max_s, hd], 4i32)
        .make_state("v_cache", &[1, kvh, max_s, hd], 4i32)
        .read_state("k_read", "k_cache_3")
        .read_state("v_read", "v_cache_4")
        .const_uint8("q_idx", q_idx, &[h, h / 2])
        .const_f16("q_cb", q_cb, &[h, 16]);
    let b = b
        .constexpr_lut_to_dense("q_weight", "q_idx_7", "q_cb_8", &[h, h], 1)
        .matmul("input", "q_weight_9");
    let b = b
        .const_uint8("k_idx", k_idx, &[kv_dim, h / 2])
        .const_f16("k_cb", k_cb, &[kv_dim, 16]);
    let b = b
        .constexpr_lut_to_dense("k_weight", "k_idx_11", "k_cb_12", &[kv_dim, h], 1)
        .matmul("input", "k_weight_13");
    let b = b
        .const_uint8("v_idx", v_idx, &[kv_dim, h / 2])
        .const_f16("v_cb", v_cb, &[kv_dim, 16]);
    let b = b
        .constexpr_lut_to_dense("v_weight", "v_idx_15", "v_cb_16", &[kv_dim, h], 1)
        .matmul("input", "v_weight_17");
    let b = b
        .mul("matmul_10", "rope_cos_1")
        .mul("matmul_14", "rope_cos_1");
    let b = b
        .dequantize("k_deq", "k_read_5", kv_scale, &[1, kvh, max_s, hd])
        .dequantize("v_deq", "v_read_6", kv_scale, &[1, kvh, max_s, hd]);
    let b = b
        .quantize("k_quant", "mul_19", kv_scale, &[1, kvh, n, hd])
        .quantize("v_quant", "matmul_18", kv_scale, &[1, kvh, n, hd]);
    let b = b
        .slice_update("k_upd", "k_deq_21", "k_quant_23", &[0, 0, 0, 0])
        .slice_update("v_upd", "v_deq_22", "v_quant_24", &[0, 0, 0, 0])
        .scaled_dot_product_attention(
            "attn",
            "mul_19",
            "k_upd_25",
            "v_upd_26",
            Some("causal_mask_0"),
            None,
        )
        .write_state("k_cache_3", "k_upd_25")
        .write_state("v_cache_4", "v_upd_26")
        .output("attn_27")
        .output("mul_19")
        .output("mul_20");
    b.build()
        .map_err(|e| format!("stateful prefill graph failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_qkv_bundle_palettized_acceptance() {
        let cb: Vec<f32> = vec![1.0; 8 * 16 ];
        let idx: Vec<u8> = vec![0; 8 * 16 / 2 ];
        let prog =
            build_qkv_bundle_mil_palettized("x", 8, 2, 2, 4, &cb, &idx, &cb, &idx, &cb, &idx)
                .unwrap();
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        assert_eq!(block.operations.len(), 12);
        assert_eq!(block.operations[2].r#type, "constexpr_lut_to_dense");
        assert_eq!(block.outputs, vec!["matmul_3", "matmul_7", "matmul_11"]);
    }

    #[test]
    fn build_quantize_dequantize_acceptance() {
        let mut b = MilBuilder::new("main");
        b = b.input("x", mil_spec::DataType::Float32, &[1, 4]);
        b = b.make_state("k_cache", &[1, 2, 64, 4], 4i32);
        let k_cache = b.last_name().expect("k_cache").to_string();
        b = b.read_state("kr", &k_cache);
        let k_read = b.last_name().expect("k_read").to_string();
        b = b.dequantize("kd", &k_read, 8.355, &[1, 2, 64, 4]);
        let k_deq = b.last_name().expect("k_deq").to_string();
        b = b.quantize("kq", &k_deq, 8.355, &[1, 2, 4, 4]);
        let k_quant = b.last_name().expect("k_quant").to_string();
        b = b.write_state(&k_cache, &k_quant);
        b = b.output(&k_quant);
        let prog = b.build().unwrap();
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let types: Vec<&str> = block.operations.iter().map(|o| o.r#type.as_str()).collect();
        assert!(types.contains(&"quantize"), "missing quantize op");
        assert!(types.contains(&"dequantize"), "missing dequantize op");
        assert!(types.contains(&"make_state"), "missing make_state");
        assert!(types.contains(&"read_state"), "missing read_state");
        assert!(types.contains(&"write_state"), "missing write_state");
    }
}
