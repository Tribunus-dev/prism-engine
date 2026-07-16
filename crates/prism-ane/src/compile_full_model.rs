//! ANE prefill compilation pipeline.
//!
//! Orchestrates: safetensors weight loading → per-row uniform palettization →
//! MIL program generation → .mlpackage serialization → coremlcompiler invocation
//! → .mlmodelc blob embedding into .cimage.

use std::path::{Path, PathBuf};

use crate::mil_gen_full::{self, LayerMILWeights};
use crate::mlpackage::{self, ModelMeta};
use crate::pack_mlmodelc;
use prism_engine::lut::graph::{ComputeNode, ModelGraph};
use prism_engine::quantization::cimage::cimage_append_blob;
use safetensors::SafeTensors;

// ── Config extraction ───────────────────────────────────────────────────

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
            _ => {}
        }
    }

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

/// A loaded safetensors shard with its backing data kept alive.
struct Shard {
    path: PathBuf,
    #[allow(dead_code)]
    data: Vec<u8>, // kept alive; SafeTensors borrows it
    tensors: SafeTensors<'static>,
}

/// Safety: the `data` Vec owns the bytes; `tensors` borrows it. Both fields stay
/// together in the `Shard` and are never separated.
fn load_shards(dir: &Path) -> Result<Vec<Shard>, String> {
    let mut shards: Vec<Shard> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {}", dir.display(), e))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "safetensors")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        let data = std::fs::read(&path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        // SafeTensors borrows data. We transmute the lifetime to 'static and
        // keep the Vec<u8> alive within the Shard for the same duration.
        let tensors: SafeTensors<'static> = unsafe {
            SafeTensors::deserialize(std::mem::transmute::<&[u8], &'static [u8]>(&data))
                .map_err(|e| format!("safetensors deser {}: {}", path.display(), e))?
        };
        shards.push(Shard {
            path,
            data,
            tensors,
        });
    }

    if shards.is_empty() {
        return Err(format!("no .safetensors files found in {}", dir.display()));
    }
    Ok(shards)
}

// ── Key-prefix search ───────────────────────────────────────────────────

const KEY_PREFIXES: &[&str] = &[
    "model.layers.",
    "model.language_model.model.layers.",
    "transformer.h.",
    "language_model.model.layers.",
    "model.language_model.layers.",
    "layers.",
];

const TOP_KEYS: &[&str] = &[
    "model.embed_tokens.weight",
    "model.language_model.model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
    "model.language_model.embed_tokens.weight",
];

const NORM_KEYS: &[&str] = &[
    "model.norm.weight",
    "model.language_model.model.norm.weight",
    "language_model.model.norm.weight",
    "transformer.ln_f.weight",
    "gpt_neox.final_layer_norm.weight",
    "model.language_model.ln_f.weight",
];

const LM_HEAD_KEYS: &[&str] = &[
    "lm_head.weight",
    "model.lm_head.weight",
    "model.language_model.model.lm_head.weight",
    "language_model.model.lm_head.weight",
    "model.language_model.lm_head.weight",
];

fn layer_key(prefix: &str, layer: usize, module: &str) -> String {
    format!("{prefix}{layer}.{module}.weight")
}

fn find_weight<'a>(shards: &'a [Shard], key: &str) -> Option<&'a [f32]> {
    shards.iter().find_map(|shard| {
        shard
            .tensors
            .tensor(key)
            .ok()
            .map(|t| bytemuck::cast_slice(t.data()))
    })
}

fn find_layer_weight<'a>(
    shards: &'a [Shard],
    layer: usize,
    module: &str,
) -> Result<&'a [f32], String> {
    for prefix in KEY_PREFIXES {
        let key = layer_key(prefix, layer, module);
        if let Some(w) = find_weight(shards, &key) {
            return Ok(w);
        }
    }
    Err(format!("weight layer {layer} {module} not found"))
}

fn find_top_weight<'a>(shards: &'a [Shard], suffixes: &[&str]) -> Option<&'a [f32]> {
    suffixes
        .iter()
        .find_map(|suffix| find_weight(shards, suffix))
}

fn weight_shape(shards: &[Shard], key: &str) -> Option<(usize, usize)> {
    shards.iter().find_map(|shard| {
        shard.tensors.tensor(key).ok().map(|t| {
            let s = t.shape();
            (s[0] as usize, s[1] as usize)
        })
    })
}

fn layer_weight_shape(shards: &[Shard], layer: usize, module: &str) -> Option<(usize, usize)> {
    for prefix in KEY_PREFIXES {
        let key = layer_key(prefix, layer, module);
        if let Some(s) = weight_shape(shards, &key) {
            return Some(s);
        }
    }
    None
}

// ── Uniform palettization ───────────────────────────────────────────────

fn palettize_weight(weights: &[f32], out_dim: usize, in_dim: usize) -> (Vec<f32>, Vec<u8>) {
    let n_centroids = 16usize;
    let bits_per_index = 4usize;
    let mut codebook = Vec::with_capacity(out_dim * n_centroids);
    let mut indices = vec![0u8; (out_dim * in_dim * bits_per_index + 7) / 8];

    for row in 0..out_dim {
        let start = row * in_dim;
        let end = start + in_dim;
        let row_vals = &weights[start..end];

        let min_val = row_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = if (max_val - min_val) > 1e-10 {
            max_val - min_val
        } else {
            1.0f32
        };

        for c in 0..n_centroids {
            codebook.push(min_val + range * (c as f32 + 0.5) / n_centroids as f32);
        }

        for col in 0..in_dim {
            let val = row_vals[col];
            let mut best_dist = f32::INFINITY;
            let mut best_idx = 0u8;
            for c in 0..n_centroids {
                let dist = (val - codebook[row * n_centroids + c]).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c as u8;
                }
            }
            let bit_pos = (row * in_dim + col) * bits_per_index;
            let byte_pos = bit_pos / 8;
            let offset = bit_pos % 8;
            indices[byte_pos] |= best_idx << offset;
        }
    }
    (codebook, indices)
}

// ── RoPE tables ─────────────────────────────────────────────────────────

fn build_rope_tables(head_dim: usize, rope_theta: f32, max_seq_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos_table = Vec::with_capacity(max_seq_len * head_dim);
    let mut sin_table = Vec::with_capacity(max_seq_len * head_dim);

    for pos in 0..max_seq_len {
        for i in 0..head_dim / 2 {
            let freq = (pos as f32) / (rope_theta.powf(2.0 * (2 * i) as f32 / head_dim as f32));
            cos_table.push(freq.cos());
            sin_table.push(freq.sin());
        }
    }
    (cos_table, sin_table)
}

// ── Causal mask ─────────────────────────────────────────────────────────

fn build_causal_mask(chunk_size: usize) -> Vec<f32> {
    let mut mask = Vec::with_capacity(chunk_size * chunk_size);
    for i in 0..chunk_size {
        for j in 0..chunk_size {
            mask.push(if i >= j { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    mask
}

// ── Main entry point ────────────────────────────────────────────────────

/// Compile a full ANE prefill model from raw safetensors weights.
pub fn compile_ane_prefill(
    model_name: &str,
    safetensors_dir: &Path,
    graph: &ModelGraph,
    cimage_path: &Path,
) -> Result<(), String> {
    let mut cfg = extract_config(graph);
    let shards = load_shards(safetensors_dir)?;

    let first_gate_shape = layer_weight_shape(&shards, 0, "mlp.gate_proj");
    if let Some((_, gate_in)) = first_gate_shape {
        cfg.intermediate_dim = gate_in;
    } else {
        cfg.intermediate_dim = if cfg.hidden_dim <= 2048 {
            cfg.hidden_dim * 8 / 3
        } else {
            cfg.hidden_dim * 4
        };
    }

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

    let embed_raw = find_top_weight(&shards, TOP_KEYS)
        .ok_or_else(|| "embed_tokens.weight not found".to_string())?;
    let embed_n = cfg.vocab_size;
    let embed_d = cfg.hidden_dim;
    let (embed_cb, embed_idx) = palettize_weight(embed_raw, embed_n, embed_d);

    let norm_raw =
        find_top_weight(&shards, NORM_KEYS).ok_or_else(|| "norm.weight not found".to_string())?;

    let lm_head_data = find_top_weight(&shards, LM_HEAD_KEYS);
    let (lm_head_cb, lm_head_idx) = match lm_head_data {
        Some(raw) => {
            let lm_out = raw.len() / cfg.hidden_dim;
            palettize_weight(raw, lm_out, cfg.hidden_dim)
        }
        None => (vec![], vec![]),
    };

    let max_seq_len = cfg.n_heads * cfg.head_dim;
    let (rope_cos, rope_sin) = build_rope_tables(cfg.head_dim, cfg.rope_theta, max_seq_len);
    let causal_mask = build_causal_mask(max_seq_len);

    let program = mil_gen_full::build_full_prefill_mil(
        cfg.n_layers as u32,
        cfg.hidden_dim as u32,
        cfg.n_heads as u32,
        cfg.n_kv_heads as u32,
        cfg.head_dim as u32,
        cfg.intermediate_dim as u32,
        cfg.vocab_size as u32,
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

    let blob_bytes = pack_mlmodelc(&mlmodelc_dir)?;
    cimage_append_blob(cimage_path, "mlmodelc", &blob_bytes)?;

    Ok(())
}
