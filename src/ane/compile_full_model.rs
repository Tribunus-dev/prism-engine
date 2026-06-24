//! ANE prefill compilation pipeline.
//!
//! Orchestrates: safetensors weight loading → per-row uniform palettization →
//! MIL program generation → .mlpackage serialization → coremlcompiler invocation
//! → .mlmodelc blob embedding into .cimage.
//!
//! ## Weight loading
//!
//! Iterates all `.safetensors` shards in `safetensors_dir`, tries multiple
//! common HF key prefixes per weight to find each tensor:
//!
//! - `model.layers.{N}.self_attn.q_proj.weight`  (Llama / Mistral / Qwen2)
//! - `model.language_model.model.layers.{N}...`  (Qwen3_5 / Gemma4)
//! - `model.language_model.layers.{N}...`         (some hybrid exports)
//!
//! ## Palettization
//!
//! Uniform per-row 16-entry codebooks (f32 centroids) with packed 4-bit
//! indices.  Each row's 16 centroid values are stored as f32, then encoded
//! as f16 inside the MIL constexpr_lut_to_dense op.

use crate::ane::mil_gen_full::{self, LayerMILWeights};
use crate::ane::mlpackage::{self, ModelMeta};
use crate::ane::pack_mlmodelc;
use crate::lut::graph::{ComputeNode, ModelGraph};
use crate::quantization::cimage::cimage_append_blob;
use safetensors::SafeTensors;
use std::path::{Path, PathBuf};

// ── Config extraction ───────────────────────────────────────────────────

/// Model hyper-parameters extracted from [`ModelGraph`] compute nodes.
struct ModelCfg {
    vocab_size: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_layers: usize,
    intermediate_dim: usize,
    norm_eps: f32,
    rope_theta: f32,
}

/// Extract model configuration by scanning `ComputeNode` entries.
///
/// Follows the same pattern as `cfg_attn` / `cfg_rope` / `cfg_eps` helpers
/// in `cimage_engine.rs`.
fn extract_config(graph: &ModelGraph) -> ModelCfg {
    let mut vocab_size: usize = 151_936;
    let mut hidden_dim: usize = 4096;
    let mut n_heads: usize = 32;
    let mut n_kv_heads: usize = 8;
    let mut head_dim: usize = 128;
    let mut norm_eps: f32 = 1e-5;
    let mut rope_theta: f32 = 10_000.0;
    let intermediate_dim: usize = 11008;
    let n_layers = graph.num_layers as usize;

    for node in &graph.nodes {
        match *node {
            ComputeNode::TokenEmbedding {
                vocab_size: vs,
                hidden_dim: hd,
                ..
            } => {
                vocab_size = vs as usize;
                hidden_dim = hd as usize;
            }
            ComputeNode::ScaledDotProductAttention {
                num_heads,
                num_kv_heads,
                head_dim: hd,
            }
            | ComputeNode::LinearAttention {
                num_heads,
                num_kv_heads,
                head_dim: hd,
            } => {
                n_heads = num_heads as usize;
                n_kv_heads = num_kv_heads as usize;
                head_dim = hd as usize;
            }
            ComputeNode::Norm { eps, .. } => {
                norm_eps = eps;
            }
            ComputeNode::RotaryEmbedding { rope_theta: rt, .. } => {
                rope_theta = rt;
            }
            // Intermediate size is implicit from MLP projection shapes —
            // we derive it after loading the first layer's weights below.
            _ => {}
        }
    }

    // If SDpa/LinearAttention never appeared, derive from hidden_dim.
    if n_heads == 32 && hidden_dim != 4096 {
        n_heads = hidden_dim / 64;
        n_kv_heads = n_heads / 4;
        head_dim = 64;
    }

    ModelCfg {
        vocab_size,
        hidden_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        n_layers,
        intermediate_dim,
        norm_eps,
        rope_theta,
    }
}

// ── Safetensors loading ─────────────────────────────────────────────────

/// Load all `.safetensors` shards from a directory.
fn load_shards(dir: &Path) -> Result<Vec<(PathBuf, Vec<u8>, SafeTensors<'static>)>, String> {
    let mut shards: Vec<(PathBuf, Vec<u8>, SafeTensors<'static>)> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read safetensors dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let path = entry.path();
        let buf = std::fs::read(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let tensors = SafeTensors::deserialize(&buf)
            .map_err(|e| format!("deserialize {}: {e}", path.display()))?;
        // SAFETY: `buf` is owned by the tuple element — it lives as long as
        // the SafeTensors that borrows it.
        let st: SafeTensors<'static> = unsafe { std::mem::transmute(tensors) };
        shards.push((path, buf, st));
    }

    Ok(shards)
}

// ── Key-prefix search ───────────────────────────────────────────────────

/// Common HF weight key prefixes tried in order.
const KEY_PREFIXES: &[&str] = &[
    "model.layers.",
    "model.language_model.model.layers.",
    "model.language_model.layers.",
];

/// Top-level key suffixes (embed / norm / lm_head).
const TOP_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.language_model.model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
];

const NORM_KEYS: &[&str] = &[
    "model.norm.weight",
    "model.language_model.model.norm.weight",
    "model.language_model.norm.weight",
    "language_model.model.norm.weight",
];

const LM_HEAD_KEYS: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "model.language_model.model.lm_head.weight",
];

/// Build a layer weight key from prefix.
fn layer_key(prefix: &str, layer: usize, module: &str) -> String {
    format!("{prefix}{layer}.{module}.weight")
}

/// Try to find a weight tensor across all shards by exact key.
fn find_weight<'a>(
    shards: &'a [(PathBuf, Vec<u8>, SafeTensors<'static>)],
    key: &str,
) -> Option<&'a [f32]> {
    for (_, _, tensors) in shards {
        if let Ok(view) = tensors.tensor(key) {
            let raw = view.data();
            if raw.len() % 4 == 0 {
                let f32_data: &[f32] = bytemuck::cast_slice(raw);
                return Some(f32_data);
            }
        }
    }
    None
}

/// Find a layer weight by trying all known prefixes.
fn find_layer_weight<'a>(
    shards: &'a [(PathBuf, Vec<u8>, SafeTensors<'static>)],
    layer: usize,
    module: &str,
) -> Result<&'a [f32], String> {
    for prefix in KEY_PREFIXES {
        let key = layer_key(prefix, layer, module);
        if let Some(data) = find_weight(shards, &key) {
            return Ok(data);
        }
    }
    Err(format!(
        "layer {layer} weight not found for module `{module}`"
    ))
}

/// Find a top-level (non-layer) weight by trying each suffix.
fn find_top_weight<'a>(
    shards: &'a [(PathBuf, Vec<u8>, SafeTensors<'static>)],
    suffixes: &[&str],
) -> Option<&'a [f32]> {
    for suffix in suffixes {
        if let Some(data) = find_weight(shards, suffix) {
            return Some(data);
        }
    }
    None
}

/// Infer the 2D shape of a weight from safetensors metadata.
fn weight_shape<'a>(
    shards: &'a [(PathBuf, Vec<u8>, SafeTensors<'static>)],
    key: &str,
) -> Option<(usize, usize)> {
    for (_, _, tensors) in shards {
        if let Ok(view) = tensors.tensor(key) {
            let shape = view.shape();
            if shape.len() == 2 {
                return Some((shape[0], shape[1]));
            }
        }
    }
    None
}

/// Resolve 2D shape by trying all prefixes.
fn layer_weight_shape<'a>(
    shards: &'a [(PathBuf, Vec<u8>, SafeTensors<'static>)],
    layer: usize,
    module: &str,
) -> Option<(usize, usize)> {
    for prefix in KEY_PREFIXES {
        let key = layer_key(prefix, layer, module);
        if let Some(shape) = weight_shape(shards, &key) {
            return Some(shape);
        }
    }
    None
}

// ── Uniform palettization ───────────────────────────────────────────────

/// Palettize a weight matrix using uniform per-row quantization.
///
/// Algorithm per row:
/// 1. Find `[min, max]` of the row elements.
/// 2. Divide range into 16 equal-width bins → 16 centroids.
/// 3. Assign each element to its nearest centroid.
/// 4. Pack indices 2-per-byte (low nibble first).
///
/// Returns `(codebook_f32, packed_indices_u8)` where codebook is
/// `[out_dim × 16]` f32 centroid values.
fn palettize_weight(weights: &[f32], out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<u8>) {
    let idx_cols = (in_dim + 1) / 2;
    let mut codebook = vec![0.0f32; out_dim * 16];
    let mut indices = vec![0u8; out_dim * idx_cols];

    let mut centroids = [0.0f32; 16];

    for row in 0..out_dim {
        let start = row * in_dim;
        let row_slice = &weights[start..start + in_dim];

        // 1. Find min / max.
        let (row_min, row_max) = row_slice.iter().fold(
            (f32::MAX, f32::MIN),
            |(mn, mx), &v| (mn.min(v), mx.max(v)),
        );

        // 2. Build 16 uniform centroids.
        if row_max == row_min {
            for c in &mut centroids {
                *c = row_min;
            }
        } else {
            let step = (row_max - row_min) / 15.0;
            for (i, c) in centroids.iter_mut().enumerate() {
                *c = row_min + step * i as f32;
            }
        }

        // 3. Write codebook.
        let cb_row = row * 16;
        codebook[cb_row..cb_row + 16].copy_from_slice(&centroids);

        // 4. Assign & pack indices.
        let idx_row = row * idx_cols;
        for col in 0..in_dim {
            let val = row_slice[col];

            // Find nearest centroid (linear scan — 16 entries, trivially fast).
            let mut best = 0u8;
            let mut best_dist = f32::MAX;
            for (j, &c) in centroids.iter().enumerate() {
                let d = (val - c).abs();
                if d < best_dist {
                    best_dist = d;
                    best = j as u8;
                }
            }

            let byte_off = idx_row + col / 2;
            if col % 2 == 0 {
                // Low nibble.
                indices[byte_off] = (indices[byte_off] & 0xF0) | best;
            } else {
                // High nibble.
                indices[byte_off] = (indices[byte_off] & 0x0F) | (best << 4);
            }
        }
    }

    (codebook, indices)
}

// ── RoPE tables ─────────────────────────────────────────────────────────

/// Build RoPE cos/sin tables for a given head dimension and theta.
///
/// Returns `(cos_table, sin_table)` each as `[max_seq_len × head_dim]` f32.
fn build_rope_tables(head_dim: usize, rope_theta: f32, max_seq_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos_t = vec![0.0f32; max_seq_len * head_dim];
    let mut sin_t = vec![0.0f32; max_seq_len * head_dim];

    let half = head_dim / 2;

    for pos in 0..max_seq_len {
        let pos_f = pos as f32;
        for i in 0..half {
            let theta = pos_f / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
            let c = theta.cos();
            let s = theta.sin();
            cos_t[pos * head_dim + i] = c;
            sin_t[pos * head_dim + i] = s;
            cos_t[pos * head_dim + half + i] = c;
            sin_t[pos * head_dim + half + i] = s;
        }
    }

    (cos_t, sin_t)
}

// ── Causal mask ─────────────────────────────────────────────────────────

/// Build a causal (lower-triangular) attention mask.
///
/// Returns `[chunk_size × chunk_size]` f32 where `0.0` is allowed and
/// `f32::NEG_INFINITY` is masked.
fn build_causal_mask(chunk_size: usize) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; chunk_size * chunk_size];
    for i in 0..chunk_size {
        for j in 0..=i {
            mask[i * chunk_size + j] = 0.0;
        }
    }
    mask
}

// ── Main entry point ────────────────────────────────────────────────────

/// Compile a full ANE prefill model from raw safetensors weights.
///
/// Pipeline:
/// 1. Load FP32 weights from `.safetensors` shards in `safetensors_dir`.
/// 2. Palettize each weight matrix (uniform per-row, 16 centroids).
/// 3. Build RoPE cos/sin tables and causal mask.
/// 4. Generate MIL program via [`mil_gen_full::build_full_prefill_mil`].
/// 5. Write `.mlpackage` to a temporary directory.
/// 6. Invoke `xcrun coremlcompiler compile` to produce `.mlmodelc`.
/// 7. Pack the `.mlmodelc` directory and embed it into `cimage_path`.
pub fn compile_ane_prefill(
    model_name: &str,
    safetensors_dir: &Path,
    graph: &ModelGraph,
    cimage_path: &Path,
) -> Result<(), String> {
    let mut cfg = extract_config(graph);

    // ── 1. Load all safetensors shards ──────────────────────────────────
    let shards = load_shards(safetensors_dir)?;

    // We need to load the first layer's weights to determine intermediate_dim
    // (it isn't stored explicitly in the compute graph).  If not found,
    // fall back to a sensible heuristic.
    let first_gate_shape = layer_weight_shape(&shards, 0, "mlp.gate_proj");
    if let Some((_, gate_in)) = first_gate_shape {
        cfg.intermediate_dim = gate_in;
    } else {
        // Heuristic: common ratios.
        cfg.intermediate_dim = if cfg.hidden_dim <= 2048 {
            cfg.hidden_dim * 8 / 3 // ~2.67× (e.g. 2048→5461)
        } else {
            cfg.hidden_dim * 4 // 4× (e.g. 4096→16384)
        };
    }

    // ── 2. Load and palettize each layer's weights ──────────────────────
    let mut layer_weights: Vec<LayerMILWeights> = Vec::with_capacity(cfg.n_layers);

    for layer in 0..cfg.n_layers {
        let q_raw = find_layer_weight(&shards, layer, "self_attn.q_proj")?;
        let k_raw = find_layer_weight(&shards, layer, "self_attn.k_proj")?;
        let v_raw = find_layer_weight(&shards, layer, "self_attn.v_proj")?;
        let o_raw = find_layer_weight(&shards, layer, "self_attn.o_proj")?;
        let gate_raw = find_layer_weight(&shards, layer, "mlp.gate_proj")?;
        let up_raw = find_layer_weight(&shards, layer, "mlp.up_proj")?;
        let down_raw = find_layer_weight(&shards, layer, "mlp.down_proj")?;

        let q_shape = layer_weight_shape(&shards, layer, "self_attn.q_proj")
            .ok_or_else(|| format!("q_proj shape for layer {layer}"))?;
        let k_shape = layer_weight_shape(&shards, layer, "self_attn.k_proj")
            .ok_or_else(|| format!("k_proj shape for layer {layer}"))?;
        let v_shape = layer_weight_shape(&shards, layer, "self_attn.v_proj")
            .ok_or_else(|| format!("v_proj shape for layer {layer}"))?;
        let o_shape = layer_weight_shape(&shards, layer, "self_attn.o_proj")
            .ok_or_else(|| format!("o_proj shape for layer {layer}"))?;
        let gate_shape = layer_weight_shape(&shards, layer, "mlp.gate_proj")
            .ok_or_else(|| format!("gate_proj shape for layer {layer}"))?;
        let up_shape = layer_weight_shape(&shards, layer, "mlp.up_proj")
            .ok_or_else(|| format!("up_proj shape for layer {layer}"))?;
        let down_shape = layer_weight_shape(&shards, layer, "mlp.down_proj")
            .ok_or_else(|| format!("down_proj shape for layer {layer}"))?;

        // Palettize: produce (codebook_f32, indices_u8).
        let (q_cb, q_idx) = palettize_weight(q_raw, q_shape.0, q_shape.1);
        let (k_cb, k_idx) = palettize_weight(k_raw, k_shape.0, k_shape.1);
        let (v_cb, v_idx) = palettize_weight(v_raw, v_shape.0, v_shape.1);
        let (o_cb, o_idx) = palettize_weight(o_raw, o_shape.0, o_shape.1);
        let (gate_cb, gate_idx) = palettize_weight(gate_raw, gate_shape.0, gate_shape.1);
        let (up_cb, up_idx) = palettize_weight(up_raw, up_shape.0, up_shape.1);
        let (down_cb, down_idx) = palettize_weight(down_raw, down_shape.0, down_shape.1);

        layer_weights.push(LayerMILWeights {
            q_cb,
            q_idx,
            k_cb,
            k_idx,
            v_cb,
            v_idx,
            o_cb,
            o_idx,
            gate_cb,
            gate_idx,
            gate_dim: gate_shape.0 as u32,
            up_cb,
            up_idx,
            up_dim: up_shape.0 as u32,
            down_cb,
            down_idx,
            down_dim: down_shape.1 as u32,
        });
    }

    // ── 3. Load top-level weights ───────────────────────────────────────
    // Embedding table.
    let embed_raw = find_top_weight(&shards, TOP_KEYS).ok_or_else(|| {
        "embed_tokens.weight not found in safetensors".to_string()
    })?;
    let embed_n = cfg.vocab_size;
    let embed_d = cfg.hidden_dim;
    let (embed_cb, embed_idx) = palettize_weight(embed_raw, embed_n, embed_d);

    // Final RMS norm weight (1-D vector [hidden_dim]).
    let norm_raw = find_top_weight(&shards, NORM_KEYS).ok_or_else(|| {
        "norm.weight not found in safetensors".to_string()
    })?;

    // LM head (may be None if tied with embedding).
    let lm_head_data = find_top_weight(&shards, LM_HEAD_KEYS);
    let (lm_head_cb, lm_head_idx) = match lm_head_data {
        Some(raw) => {
            // Infer shape: the raw f32 slice length = vocab_size * hidden_dim.
            let lm_out = raw.len() / cfg.hidden_dim;
            palettize_weight(raw, lm_out, cfg.hidden_dim)
        }
        None => (vec![], vec![]),
    };

    // ── 4. Build RoPE tables and causal mask ────────────────────────────
    let max_seq_len = cfg.n_heads * cfg.head_dim; // reasonable upper bound
    let (rope_cos, rope_sin) = build_rope_tables(cfg.head_dim, cfg.rope_theta, max_seq_len);
    let causal_mask = build_causal_mask(max_seq_len);

    // ── 5. Generate MIL program ─────────────────────────────────────────
    let program = mil_gen_full::build_full_prefill_mil(
        cfg.n_layers as u32,
        cfg.hidden_dim as u32,
        cfg.n_heads as u32,
        cfg.n_kv_heads as u32,
        cfg.head_dim as u32,
        cfg.intermediate_dim as u32,
        cfg.vocab_size as u32,
        // chunk_size = max_seq_len (full prefill)
        max_seq_len as u32,
        max_seq_len as u32,
        cfg.norm_eps,
        &layer_weights,
        &embed_cb,
        &embed_idx,
        &lm_head_cb,
        &lm_head_idx,
        norm_raw,
        &rope_cos,
        &rope_sin,
        &causal_mask,
    )
    .map_err(|e| format!("MIL program generation failed: {e}"))?;

    // ── 6. Write .mlpackage ─────────────────────────────────────────────
    let tmp_dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let meta = ModelMeta {
        model_name: model_name.to_string(),
        function_name: "prefill".to_string(),
        short_description: format!("ANE prefill — {model_name}"),
        version: "1.0.0".to_string(),
        author: "Tribunus Compute".to_string(),
        output_name: "output".to_string(),
        inputs: vec![("x".to_string(), vec![1, cfg.hidden_dim as i64])],
        outputs: vec![("output".to_string(), vec![1, cfg.vocab_size as i64])],
    };

    let package_dir = mlpackage::write_mlpackage(program, tmp_dir.path(), &meta)?;

    // ── 7. Core ML compilation ──────────────────────────────────────────
    let mlmodelc_name = format!("{}.mlmodelc", meta.model_name);
    let mlmodelc_dir = tmp_dir.path().join(&mlmodelc_name);

    let status = std::process::Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(package_dir.to_str().unwrap())
        .arg(tmp_dir.path().to_str().unwrap())
        .status()
        .map_err(|e| format!("coremlcompiler launch: {e}"))?;

    if !status.success() {
        return Err("coremlcompiler compile failed".to_string());
    }

    if !mlmodelc_dir.exists() {
        return Err(format!(
            "compilation produced no .mlmodelc at {}",
            mlmodelc_dir.display()
        ));
    }

    // ── 8. Embed .mlmodelc into .cimage ─────────────────────────────────
    let blob_bytes = pack_mlmodelc(&mlmodelc_dir)?;
    cimage_append_blob(cimage_path, "mlmodelc", &blob_bytes)?;

    Ok(())
}
