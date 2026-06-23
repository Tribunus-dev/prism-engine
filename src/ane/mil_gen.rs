//! MIL program generator for ANE prefill graphs.
//!
//! Generates MIL (Core ML Intermediate Language) text that can be compiled
//! with `xcrun coremlcompiler` into a `.mlmodelc` bundle at model-pull time.
//! The compiled model is embedded as a blob in the .cimage for runtime loading.
//!
//! MIL text spec: https://apple.github.io/coremltools/source/coremltools-converters-mil.html

use std::path::Path;

/// Generate a MIL program for ANE prefill and write it to a .mil file.
/// Returns the path to the generated file.
pub fn generate_ane_prefill_mil(
    output_dir: &Path,
    num_layers: u32,
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    max_chunk: u32,
    tie_embeddings: bool,
) -> std::path::PathBuf {
    let mil_path = output_dir.join("ane_prefill.mil");
    let mut mil = String::new();

    // MIL header
    mil.push_str("// Auto-generated ANE prefill program\n");
    mil.push_str("// Architecture: Qwen2.5\n");
    mil.push_str("// Config: layers=");
    mil.push_str(&num_layers.to_string());
    mil.push_str(" hidden=");
    mil.push_str(&hidden_dim.to_string());
    mil.push_str(" heads=");
    mil.push_str(&num_heads.to_string());
    mil.push_str(" kv_heads=");
    mil.push_str(&num_kv_heads.to_string());
    mil.push_str(" head_dim=");
    mil.push_str(&head_dim.to_string());
    mil.push_str("\n\n");

    // Program signature
    mil.push_str("func prefill_chunk(");
    mil.push_str("input: tensor<fp16, [1, ");
    mil.push_str(&max_chunk.to_string());
    mil.push_str(", ");
    mil.push_str(&hidden_dim.to_string());
    mil.push_str("]>, state: state<fp32>) -> (\n");
    mil.push_str("    output: tensor<fp16, [1, 1, ");
    mil.push_str(&hidden_dim.to_string());
    mil.push_str("]>,\n");
    mil.push_str("    k_cache: tensor<fp16, [1, ");
    mil.push_str(&max_chunk.to_string());
    mil.push_str(", ");
    mil.push_str(&(num_kv_heads * head_dim).to_string());
    mil.push_str("]>,\n");
    mil.push_str("    v_cache: tensor<fp16, [1, ");
    mil.push_str(&max_chunk.to_string());
    mil.push_str(", ");
    mil.push_str(&(num_kv_heads * head_dim).to_string());
    mil.push_str("]>\n");
    mil.push_str(") {\n\n");

    // Weight declarations (loaded from .cimage weights dict)
    mil.push_str("    // Load weights from the model dictionary\n");
    mil.push_str("    let wts = weight_dict(\"model\");\n\n");

    mil.push_str("    // Embedding lookup (LUT dequant)\n");
    mil.push_str("    let embed_w = constexpr_lut_to_dense(wts[\"model.embed_tokens.weight\"]);\n");
    mil.push_str("    let h = embedding(input, embed_w);\n\n");

    // Transformer layers
    for l in 0..num_layers {
        mil.push_str(&format!("    // Layer {}\n", l));

        // RMS Norm
        mil.push_str(&format!(
            "    let norm_w = constexpr_lut_to_dense(wts[\"model.layers.{}.input_layernorm.weight\"]);\n", l
        ));
        mil.push_str("    let h = rms_norm(h, 1e-6);\n");

        // QKV projections
        mil.push_str(&format!(
            "    let q_w = constexpr_lut_to_dense(wts[\"model.layers.{}.self_attn.q_proj.weight\"]);\n", l
        ));
        mil.push_str(&format!(
            "    let k_w = constexpr_lut_to_dense(wts[\"model.layers.{}.self_attn.k_proj.weight\"]);\n", l
        ));
        mil.push_str(&format!(
            "    let v_w = constexpr_lut_to_dense(wts[\"model.layers.{}.self_attn.v_proj.weight\"]);\n", l
        ));
        mil.push_str("    let q = linear(h, q_w);\n");
        mil.push_str("    let k = linear(h, k_w);\n");
        mil.push_str("    let v = linear(h, v_w);\n");

        // RoPE
        mil.push_str(&format!(
            "    let q = rope(q, {}.0, {}.0, 10000.0);\n", head_dim, head_dim / 2
        ));
        mil.push_str(&format!(
            "    let k = rope(k, {}.0, {}.0, 10000.0);\n", head_dim, head_dim / 2
        ));

        // Read K/V state (appends to existing cache)
        mil.push_str(&format!(
            "    let k_state = read_state(state, \"k_{}\");\n", l
        ));
        mil.push_str(&format!(
            "    let v_state = read_state(state, \"v_{}\");\n", l
        ));
        mil.push_str("    let k = concat(k_state, k, dim=1);\n");
        mil.push_str("    let v = concat(v_state, v, dim=1);\n");

        // Write updated K/V back to state
        mil.push_str(&format!(
            "    let _ = write_state(state, \"k_{}\", k);\n", l
        ));
        mil.push_str(&format!(
            "    let _ = write_state(state, \"v_{}\", v);\n", l
        ));

        // Attention
        mil.push_str(&format!(
            "    let attn = scaled_dot_product_attention(q, k, v, {}.0, {}.0, {}.0, \"causal\");\n",
            num_heads, num_kv_heads, head_dim
        ));

        // Output projection
        mil.push_str(&format!(
            "    let o_w = constexpr_lut_to_dense(wts[\"model.layers.{}.self_attn.o_proj.weight\"]);\n", l
        ));
        mil.push_str("    let h_res = h + linear(attn, o_w);\n");

        // MLP
        mil.push_str(&format!(
            "    let gate_w = constexpr_lut_to_dense(wts[\"model.layers.{}.mlp.gate_proj.weight\"]);\n", l
        ));
        mil.push_str(&format!(
            "    let up_w = constexpr_lut_to_dense(wts[\"model.layers.{}.mlp.up_proj.weight\"]);\n", l
        ));
        mil.push_str(&format!(
            "    let down_w = constexpr_lut_to_dense(wts[\"model.layers.{}.mlp.down_proj.weight\"]);\n", l
        ));
        mil.push_str("    let gate = silu(linear(h_res, gate_w));\n");
        mil.push_str("    let up = linear(h_res, up_w);\n");
        mil.push_str("    let h = h_res + linear(gate * up, down_w);\n\n");
    }

    // Extract last token's hidden state as output
    mil.push_str("    // Output: last token position\n");
    mil.push_str(&format!(
        "    let output = slice(h, 0, {}, 1, 1);\n", max_chunk - 1
    ));

    // Extract K/V cache output (last chunk for handoff to Metal decode)
    mil.push_str(&format!(
        "    let k_cache = read_state(state, \"k_{}\");\n", num_layers - 1
    ));
    mil.push_str(&format!(
        "    let v_cache = read_state(state, \"v_{}\");\n", num_layers - 1
    ));

    mil.push_str("}\n");

    std::fs::write(&mil_path, &mil).expect("write MIL file");
    std::process::Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(&mil_path)
        .arg(&output_dir)
        .status()
        .ok();

    mil_path
}
