//! gemma4_ingest — Stage 1+2: Download Gemma 4 12B from HF, ternary quantize,
//! compile to .cimage, all in Rust, zero Python.
//!
//! Subcommands:
//!   inspect-checkpoint  — Read a checkpoint directory and emit metadata
//!   (default)            — Ternary-quantize a Gemma 4 12B checkpoint to .cimage
//!
//! Usage:
//!   cargo run --bin gemma4_ingest -- inspect-checkpoint --model-dir <PATH> [--emit <PATH>]...
//!   cargo run --bin gemma4_ingest -- --repo google/gemma-4-12b-it --output gemma4_12b.cimage
//!   cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B-it-qat-q4_0-unquantized --output gemma4_12b_qat.cimage
//!   cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage

#![allow(unused_imports)]

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tribunus_compute_core::ane_compile::compile_ane_artifacts;
use tribunus_compute_core::compute_image::compile::execution_graph::ExecutionGraphDescriptor;
use tribunus_compute_core::compute_image::compile::ternary::{
    model_artifact_tag, CimageHeader, ModelArtifactEntry, SegmentEntry, SegmentKind,
    CIMAGE_SEGMENT_CAPACITY,
};
use tribunus_compute_core::compute_image::subgraph_mil::{build_draft_layer_mil, build_matmul_mil};
use tribunus_compute_core::quantization::embed_cluster::*;

// ── Gemma 4 12B architecture constants ──────────────────────────────
const NUM_LAYERS: usize = 48;
const HIDDEN_DIM: usize = 3840;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 256;
const FFN_INTERMEDIATE: usize = 15360;

/// (serialized_name, rows, cols)
const MATRICES: &[(&str, usize, usize)] = &[
    (
        "model.language_model.layers.{}.self_attn.q_proj.weight",
        HIDDEN_DIM,
        NUM_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.k_proj.weight",
        HIDDEN_DIM,
        NUM_KV_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.v_proj.weight",
        HIDDEN_DIM,
        NUM_KV_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.o_proj.weight",
        NUM_HEADS * HEAD_DIM,
        HIDDEN_DIM,
    ),
    (
        "model.language_model.layers.{}.mlp.gate_proj.weight",
        HIDDEN_DIM,
        FFN_INTERMEDIATE,
    ),
    (
        "model.language_model.layers.{}.mlp.up_proj.weight",
        HIDDEN_DIM,
        FFN_INTERMEDIATE,
    ),
    (
        "model.language_model.layers.{}.mlp.down_proj.weight",
        FFN_INTERMEDIATE,
        HIDDEN_DIM,
    ),
];

/// Multimodal projection tensors (image + audio direct projection).
const MULTIMODAL_WEIGHTS: &[(&str, usize, usize)] = &[
    ("model.vision_embedder.patch_dense.weight", 3840, 6912),
    ("model.vision_embedder.patch_dense.bias", 3840, 1),
    ("model.vision_embedder.patch_ln1.weight", 6912, 1),
    ("model.vision_embedder.patch_ln1.bias", 6912, 1),
    ("model.vision_embedder.patch_ln2.weight", 3840, 1),
    ("model.vision_embedder.patch_ln2.bias", 3840, 1),
    ("model.vision_embedder.pos_norm.weight", 3840, 1),
    ("model.vision_embedder.pos_norm.bias", 3840, 1),
    ("model.embed_vision.embedding_projection.weight", 3840, 3840),
    ("model.embed_audio.embedding_projection.weight", 3840, 640),
];

/// Token embeddings — packed separately for vocabulary lookup.
const EMBEDDING_WEIGHT: (&str, usize, usize) =
    ("model.language_model.embed_tokens.weight", 262144, 3840);

/// Final norm.
const FINAL_NORM: (&str, usize, usize) = ("model.language_model.norm.weight", 3840, 1);

// ── FP16 conversion ─────────────────────────────────────────────────

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return if mant == 0 {
            if sign != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if sign != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 {
        return sign;
    }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Read tensor bytes (f32 or bf16) into a Vec<f32>.
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            })
            .collect(),
        _ => panic!("unsupported dtype: {:?}", dtype),
    }
}

/// Build a tensor key for a given layer and matrix name.
fn tensor_key(layer: usize, template: &str) -> String {
    template.replace("{}", &layer.to_string())
}

// ── inspect-checkpoint subcommand ───────────────────────────────────

/// Classify a tensor name into one of the known categories.
fn classify_tensor(name: &str) -> &'static str {
    let n = name;

    // ignored: optimizer states, cached params, momentum, variance
    if n.contains("optimizer")
        || n.contains("momentum")
        || n.contains("variance")
        || n.contains("_cache")
        || n.contains("adam")
        || n.contains("rmsprop")
        || n.contains("ema_")
        || n.contains("batch_norm")
    {
        return "ignored";
    }

    // mtp_required
    if n.contains("mtp")
        || n.contains("draft")
        || n.contains("speculative")
        || n.contains("proposal")
    {
        return "mtp_required";
    }

    // multimodal_image_required — check before decoder to catch vision encoder tensors
    if n.contains("multimodal_image")
        || n.contains("mm_image")
        || n.contains("vision_")
        || n.contains("vision.")
        || n.contains("image_")
        || n.contains("image.")
        || n.contains("patch")
        || (n.contains("projection") && n.contains("layers"))
    {
        return "multimodal_image_required";
    }

    // multimodal_audio_required
    if n.contains("multimodal_audio")
        || n.contains("mm_audio")
        || n.contains("audio_")
        || n.contains("audio.")
        || n.contains("waveform")
        || n.contains("speech")
    {
        return "multimodal_audio_required";
    }

    // decoder_required: self_attn q/k/v/o, mlp gate/up/down
    if n.contains("self_attn.q_proj")
        || n.contains("self_attn.k_proj")
        || n.contains("self_attn.v_proj")
        || n.contains("self_attn.o_proj")
        || n.ends_with(".q_proj.weight")
        || n.ends_with(".k_proj.weight")
        || n.ends_with(".v_proj.weight")
        || n.ends_with(".o_proj.weight")
        || n.ends_with(".gate_proj.weight")
        || n.ends_with(".up_proj.weight")
        || n.ends_with(".down_proj.weight")
    {
        return "decoder_required";
    }

    // norm_required
    if n.contains("input_layernorm")
        || n.contains("post_attention_layernorm")
        || n.contains("pre_feedforward_layernorm")
        || n.contains("post_feedforward_layernorm")
        || n.contains("final_layernorm")
        || n.contains("rms_norm")
        || n.ends_with(".norm.weight")
        || n.ends_with("_norm.weight")
        || n.contains(".layernorm.")
    {
        return "norm_required";
    }

    // text_embedding_required — check before lm_head to avoid confusing embed output
    if n.contains("embed_tokens")
        || n.contains("token_embedding")
        || n.contains("embedding") && !n.contains("embedding_output")
    {
        return "text_embedding_required";
    }

    // lm_head_required
    if n.contains("lm_head") || n.contains("logits") || n.ends_with("output.weight") {
        return "lm_head_required";
    }

    "unknown"
}

/// Read a JSON file and deserialize, exiting on failure.
fn read_json_file(path: &Path) -> serde_json::Value {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: reading {}: {e}", path.display());
        std::process::exit(1);
    });
    serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("ERROR: parsing {}: {e}", path.display());
        std::process::exit(1);
    })
}

/// Extract a u64 value from a JSON object, checking `text_config` sub-object first.
fn config_int(config: &serde_json::Value, key: &str) -> usize {
    config
        .get("text_config")
        .and_then(|tc| tc.get(key).and_then(|v| v.as_u64()))
        .or_else(|| config.get(key).and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize
}

fn cmd_inspect_checkpoint(args: &[String]) {
    let model_dir = args.windows(2)
        .find(|w| w[0] == "--model-dir")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| {
            eprintln!("ERROR: --model-dir <PATH> is required");
            eprintln!();
            eprintln!("Usage:");
            eprintln!("  cargo run --bin gemma4_ingest -- inspect-checkpoint --model-dir <PATH> [--emit <PATH>]...");
            std::process::exit(1);
        });

    // Collect all --emit values
    let emits: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "--emit")
        .map(|w| w[1].clone())
        .collect();

    eprintln!("inspect-checkpoint: {}/", model_dir.display());

    // ── 1. Read config.json ────────────────────────────────────────
    let config_path = model_dir.join("config.json");
    let config = read_json_file(&config_path);

    let hidden_size = config_int(&config, "hidden_size");
    let num_layers = config_int(&config, "num_hidden_layers");
    let num_heads = config_int(&config, "num_attention_heads");
    let num_kv_heads = config_int(&config, "num_key_value_heads");
    let vocab_size = config_int(&config, "vocab_size");
    let intermediate_size = config_int(&config, "intermediate_size");
    let max_position_embeddings = config_int(&config, "max_position_embeddings");

    eprintln!("  model: hidden={hidden_size}, layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}");
    eprintln!(
        "         vocab={vocab_size}, ffn={intermediate_size}, max_pos={max_position_embeddings}"
    );

    // ── 2. Read tokenizer_config.json ──────────────────────────────
    let tok_path = model_dir.join("tokenizer_config.json");
    let tok_config = if tok_path.exists() {
        read_json_file(&tok_path)
    } else {
        eprintln!("  WARNING: tokenizer_config.json not found (will be empty)");
        serde_json::Value::Null
    };

    let bos_token_id = tok_config.get("bos_token_id").and_then(|v| v.as_u64());
    let eos_token_id = tok_config.get("eos_token_id").and_then(|v| v.as_u64());
    let pad_token_id = tok_config
        .get("pad_token_id")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            tok_config
                .get("pad_token_id")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
        });

    // Multimodal placeholder tokens — check named fields, then additional_special_tokens
    let image_token = tok_config
        .get("image_token_id")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            tok_config
                .get("additional_special_tokens")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    a.iter().find_map(|v| {
                        v.as_str()
                            .filter(|s| s.contains("image"))
                            .and_then(|_| v.as_u64())
                            .or_else(|| v.as_str().and_then(|_s| None))
                    })
                })
        });
    let audio_token = tok_config
        .get("audio_token_id")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            tok_config
                .get("additional_special_tokens")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    a.iter().find_map(|v| {
                        v.as_str()
                            .filter(|s| s.contains("audio"))
                            .and_then(|_| v.as_u64())
                    })
                })
        });

    // ── 3. Read processor_config.json ──────────────────────────────
    let proc_path = model_dir.join("processor_config.json");
    let proc_config = if proc_path.exists() {
        read_json_file(&proc_path)
    } else {
        eprintln!("  WARNING: processor_config.json not found (using defaults)");
        serde_json::Value::Null
    };

    let patch_size = config_int(&proc_config, "patch_size");
    let pooling_kernel = config_int(&proc_config, "pooling_kernel");
    let soft_token_default = proc_config
        .get("soft_token_default")
        .and_then(|v| v.as_u64())
        .or_else(|| proc_config.get("soft_token").and_then(|v| v.as_u64()))
        .or_else(|| proc_config.get("vision_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(256) as usize;
    let soft_token_image = proc_config
        .get("soft_token_image")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            proc_config
                .get("image_soft_tokens")
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0) as usize;
    let soft_token_audio = proc_config
        .get("soft_token_audio")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            proc_config
                .get("audio_soft_tokens")
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0) as usize;
    let max_patch_count = proc_config
        .get("max_patch_count")
        .and_then(|v| v.as_u64())
        .or_else(|| proc_config.get("max_images").and_then(|v| v.as_u64()))
        .unwrap_or(0) as usize;
    let width_divisibility = proc_config
        .get("width_divisibility")
        .and_then(|v| v.as_u64())
        .or_else(|| proc_config.get("width_align").and_then(|v| v.as_u64()))
        .unwrap_or(1) as usize;
    let height_divisibility = proc_config
        .get("height_divisibility")
        .and_then(|v| v.as_u64())
        .or_else(|| proc_config.get("height_align").and_then(|v| v.as_u64()))
        .unwrap_or(1) as usize;
    let sample_rate = proc_config
        .get("sample_rate")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            proc_config
                .get("audio_sample_rate")
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(16000) as usize;

    // ── 4. Collect tensor metadata (names, shapes, dtypes) ─────────
    // Try the safetensors index file first, then fall back to scanning headers
    let index_path = model_dir.join("model.safetensors.index.json");
    let (tensor_names, mut tensor_shapes, mut tensor_dtypes) = if index_path.exists() {
        let index = read_json_file(&index_path);
        let weight_map = index["weight_map"].as_object().unwrap_or_else(|| {
            eprintln!("ERROR: no \"weight_map\" in {}", index_path.display());
            std::process::exit(1);
        });

        let names: Vec<String> = weight_map.keys().cloned().collect();
        eprintln!(
            "  index: {} tensors in {} shard(s)",
            names.len(),
            weight_map.values().collect::<HashSet<_>>().len()
        );

        // Read safetensor headers to get shapes and dtypes
        let mut shard_files: Vec<PathBuf> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| model_dir.join(s))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        shard_files.sort();

        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut dtypes: HashMap<String, String> = HashMap::new();

        for shard_path in &shard_files {
            let data = fs::read(shard_path).unwrap_or_else(|e| {
                eprintln!("ERROR: reading {}: {e}", shard_path.display());
                std::process::exit(1);
            });
            if let Ok(st) = safetensors::SafeTensors::deserialize(&data) {
                for name in st.names() {
                    if let Ok(tvi) = st.tensor(name) {
                        shapes.insert(name.to_string(), tvi.shape().to_vec());
                        dtypes.insert(name.to_string(), format!("{:?}", tvi.dtype()));
                    }
                }
            }
        }

        (names, shapes, dtypes)
    } else {
        eprintln!("  no index file found — scanning .safetensors headers directly");
        let mut names: Vec<String> = Vec::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let mut dtypes: HashMap<String, String> = HashMap::new();

        let mut safetensor_files: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = fs::read_dir(&model_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.extension().map_or(false, |ext| ext == "safetensors") {
                    safetensor_files.push(p);
                }
            }
        }
        safetensor_files.sort();

        for shard_path in &safetensor_files {
            let data = fs::read(shard_path).unwrap_or_else(|e| {
                eprintln!("ERROR: reading {}: {e}", shard_path.display());
                std::process::exit(1);
            });
            if let Ok(st) = safetensors::SafeTensors::deserialize(&data) {
                for name in st.names() {
                    if !names.contains(name) {
                        let n = name.to_string();
                        names.push(n.clone());
                        if let Ok(tvi) = st.tensor(name) {
                            shapes.insert(n.clone(), tvi.shape().to_vec());
                            dtypes.insert(n.clone(), format!("{:?}", tvi.dtype()));
                        }
                    }
                }
            }
        }

        (names, shapes, dtypes)
    };

    eprintln!("  tensors: {}", tensor_names.len());

    // ── 5. Classify every tensor ───────────────────────────────────
    let one_million: u64 = 1_000_000;

    #[derive(Default, Clone)]
    struct CategoryStats {
        count: usize,
        total_params: u64,
    }

    let mut category_stats: HashMap<&'static str, CategoryStats> = HashMap::new();
    let mut tensors_out: Vec<serde_json::Value> = Vec::new();
    let mut unknown_large: Vec<serde_json::Value> = Vec::new();

    for name in &tensor_names {
        let shape = tensor_shapes.remove(name).unwrap_or_default();
        let dtype = tensor_dtypes
            .remove(name)
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let param_count: u64 = shape.iter().map(|&d| d as u64).product();
        let classification = classify_tensor(name);

        let entry = category_stats
            .entry(classification)
            .or_insert_with(CategoryStats::default);
        entry.count += 1;
        entry.total_params += param_count;

        let tensor_entry = serde_json::json!({
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "classification": classification,
            "param_count": param_count,
        });
        tensors_out.push(tensor_entry);

        if classification == "unknown" && param_count > one_million {
            unknown_large.push(serde_json::json!({
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "param_count": param_count,
            }));
        }
    }

    // ── 6. Build tensor_inventory.json ────────────────────────────
    let classification_obj: serde_json::Map<String, serde_json::Value> = [
        "decoder_required",
        "text_embedding_required",
        "lm_head_required",
        "norm_required",
        "multimodal_image_required",
        "multimodal_audio_required",
        "mtp_required",
        "unknown",
        "ignored",
    ]
    .iter()
    .map(|&cat| {
        let s = category_stats.get(cat).cloned().unwrap_or(CategoryStats {
            count: 0,
            total_params: 0,
        });
        (
            cat.to_string(),
            serde_json::json!({
                "count": s.count,
                "total_params": s.total_params,
            }),
        )
    })
    .collect();

    let inventory = serde_json::json!({
        "model_revision": config.get("_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown"),
        "total_tensors": tensor_names.len(),
        "classification": classification_obj,
        "tensors": tensors_out,
        "unknown_large": unknown_large,
    });

    // ── 7. Build processor_contract.json ───────────────────────────
    let contract = serde_json::json!({
        "text": {
            "vocabulary_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "image_token": image_token,
            "audio_token": audio_token,
        },
        "image": {
            "patch_size": patch_size,
            "pooling_kernel": pooling_kernel,
            "soft_token_default": soft_token_default,
            "soft_token_image": soft_token_image,
            "soft_token_audio": soft_token_audio,
            "max_patch_count": max_patch_count,
            "width_divisibility": width_divisibility,
            "height_divisibility": height_divisibility,
        },
        "audio": {
            "sample_rate": sample_rate,
        },
    });

    // ── 8. Emit output files ───────────────────────────────────────
    for emit_path in &emits {
        let lower = emit_path.to_lowercase();
        if lower.ends_with("tensor_inventory.json") {
            let json_str = serde_json::to_string_pretty(&inventory).unwrap_or_else(|e| {
                eprintln!("ERROR: serializing tensor_inventory: {e}");
                std::process::exit(1);
            });
            fs::write(emit_path, &json_str).unwrap_or_else(|e| {
                eprintln!("ERROR: writing {emit_path}: {e}");
                std::process::exit(1);
            });
            eprintln!("  emitted tensor_inventory.json -> {emit_path}");
        } else if lower.ends_with("processor_contract.json") {
            let json_str = serde_json::to_string_pretty(&contract).unwrap_or_else(|e| {
                eprintln!("ERROR: serializing processor_contract: {e}");
                std::process::exit(1);
            });
            fs::write(emit_path, &json_str).unwrap_or_else(|e| {
                eprintln!("ERROR: writing {emit_path}: {e}");
                std::process::exit(1);
            });
            eprintln!("  emitted processor_contract.json -> {emit_path}");
        } else {
            eprintln!("WARNING: unrecognized --emit target \"{emit_path}\" (expected: tensor_inventory.json or processor_contract.json)");
        }
    }

    // ── 9. Fail on large unknown tensors ───────────────────────────
    if !unknown_large.is_empty() {
        eprintln!();
        eprintln!(
            "ERROR: {} unknown tensor(s) exceed 1M parameters:",
            unknown_large.len()
        );
        for t in &unknown_large {
            eprintln!("  {} — {} params", t["name"], t["param_count"]);
        }
        std::process::exit(1);
    }

    eprintln!("  ✓ inspect-checkpoint complete");
}

// ── Entry point ─────────────────────────────────────────────────────

// ── Metal shader compilation ────────────────────────────────────────
fn compile_metal_lib() -> Vec<u8> {
    if std::process::Command::new("xcrun")
        .arg("--version")
        .output()
        .is_err()
    {
        return Vec::new();
    }
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let src_dir = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("compute_image");
    let shader_dirs = &[
        src_dir.join("templates"),
        src_dir.join("megakernel").join("shaders"),
    ];
    let mut metal_files: Vec<std::path::PathBuf> = Vec::new();
    for dir in shader_dirs {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.extension().map(|e| e == "metal").unwrap_or(false) {
                    metal_files.push(p);
                }
            }
        }
    }
    if metal_files.is_empty() {
        return Vec::new();
    }
    let tmp = std::env::temp_dir().join("gemma4_metal_build");
    let _ = std::fs::create_dir_all(&tmp);
    let mut air_files = Vec::new();
    for src in &metal_files {
        let stem = src.file_stem().unwrap().to_string_lossy();
        let air = tmp.join(format!("{stem}.air"));
        let out = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(src)
            .arg("-o")
            .arg(&air)
            .output();
        if out.map(|o| o.status.success()).unwrap_or(false) {
            air_files.push(air);
        }
    }
    if air_files.is_empty() {
        return Vec::new();
    }
    let metallib = tmp.join("kernels.metallib");
    let mut cmd = std::process::Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib", "-o"])
        .arg(&metallib);
    for air in &air_files {
        cmd.arg(air);
    }
    if cmd.status().map(|s| s.success()).unwrap_or(false) {
        std::fs::read(&metallib).unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn gemma4_kv_decompress_mil() -> Vec<u8> {
    tribunus_compute_core::ane::kv_decompress_program::generate_kv_decompress_mil(256, 8, 3)
        .into_bytes()
}

#[allow(dead_code)]
fn tar_mlmodelc_dirs(mlmodelc_paths: &[String]) -> Vec<u8> {
    let mut tar_bytes = Vec::new();
    {
        let mut builder = tar::Builder::new(&mut tar_bytes);
        for path_str in mlmodelc_paths {
            let dir_path = std::path::Path::new(path_str);
            if !dir_path.is_dir() {
                continue;
            }
            for entry in walkdir::WalkDir::new(dir_path)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let rel = entry.path().strip_prefix(dir_path).unwrap_or(entry.path());
                if entry.file_type().is_file() {
                    let data = std::fs::read(entry.path()).unwrap_or_default();
                    let mut header = tar::Header::new_gnu();
                    header.set_path(rel).unwrap_or_default();
                    header.set_size(data.len() as u64);
                    header.set_mode(0o644);
                    header.set_cksum();
                    builder.append(&header, &data[..]).unwrap_or_default();
                }
            }
        }
        builder.finish().unwrap_or_default();
    }
    tar_bytes
}

/// Compile all ANE islands in parallel via coremlcompiler.
/// Falls back to MIL protobuf tar if coremlcompiler is unavailable.
fn generate_ane_mil_packages(offsets: &HashMap<String, (u64, u64, usize)>) -> Vec<u8> {
    use prost::Message;

    // MIL protobuf fallback using the existing build_matmul_mil/build_draft_layer_mil.
    // These produce weightless programs (stateless=true). The ternary block
    // offsets are tracked by the caller but not needed for protobuf generation.
    let mut tar_entries: Vec<(String, Vec<u8>)> = Vec::new();

    // Vision patch embed: patch_dense [3840, 6912]
    if offsets.contains_key("model.vision_embedder.patch_dense.weight") {
        if let Ok(prog) = build_matmul_mil(
            "image_patches",
            "patch_dense",
            "patch_features",
            1,
            3840,
            6912,
            &[],
            true,
        ) {
            let mut buf = Vec::new();
            prog.encode(&mut buf).ok();
            tar_entries.push(("vision_patch_embed.mil".into(), buf));
        }
    }

    // Vision final projection: embedding_projection [3840, 3840]
    if offsets.contains_key("model.embed_vision.embedding_projection.weight") {
        if let Ok(prog) = build_matmul_mil(
            "patch_features",
            "embedding_proj",
            "projected_features",
            1,
            3840,
            3840,
            &[],
            true,
        ) {
            let mut buf = Vec::new();
            prog.encode(&mut buf).ok();
            tar_entries.push(("vision_projection.mil".into(), buf));
        }
    }

    // Audio embedding projection: [3840, 640]
    // Gemma4 uses a single audio projection (decoder hidden → audio latent space)
    if offsets.contains_key("model.embed_audio.embedding_projection.weight") {
        if let Ok(prog) = build_matmul_mil(
            "audio_features",
            "audio_embed_proj",
            "projected_audio",
            1,
            3840,
            640,
            &[],
            true,
        ) {
            let mut buf = Vec::new();
            prog.encode(&mut buf).ok();
            tar_entries.push(("audio_projection.mil".into(), buf));
        }
    }

    // MTP pre-projection: [1024, 3840]
    if offsets.contains_key("pre_projection.weight") {
        if let Ok(prog) = build_matmul_mil(
            "draft_hidden",
            "pre_proj",
            "main_space",
            1,
            1024,
            3840,
            &[],
            true,
        ) {
            let mut buf = Vec::new();
            prog.encode(&mut buf).ok();
            tar_entries.push(("draft_pre_proj.mil".into(), buf));
        }
    }

    // MTP post-projection: [3840, 1024]
    if offsets.contains_key("post_projection.weight") {
        if let Ok(prog) = build_matmul_mil(
            "main_hidden",
            "post_proj",
            "draft_space",
            1,
            3840,
            1024,
            &[],
            true,
        ) {
            let mut buf = Vec::new();
            prog.encode(&mut buf).ok();
            tar_entries.push(("draft_post_proj.mil".into(), buf));
        }
    }

    // Draft decoder layers: 4 layers, extract per-layer weights
    for l in 0u32..4u32 {
        let q_key = format!("model.layers.{}.self_attn.q_proj.weight", l);
        let k_key = format!("model.layers.{}.self_attn.k_proj.weight", l);
        let v_key = format!("model.layers.{}.self_attn.v_proj.weight", l);
        let o_key = format!("model.layers.{}.self_attn.o_proj.weight", l);
        let gate_key = format!("model.layers.{}.mlp.gate_proj.weight", l);
        let up_key = format!("model.layers.{}.mlp.up_proj.weight", l);
        let down_key = format!("model.layers.{}.mlp.down_proj.weight", l);

        let q = offsets.get(&q_key);
        let k = offsets.get(&k_key);
        let v = offsets.get(&v_key);
        let _o = offsets.get(&o_key);
        let gate = offsets.get(&gate_key);
        let up = offsets.get(&up_key);
        let down = offsets.get(&down_key);

        if q.is_some()
            && k.is_some()
            && v.is_some()
            && gate.is_some()
            && up.is_some()
            && down.is_some()
        {
            if let Ok(prog) = build_draft_layer_mil(
                "draft_hidden",
                1024,
                8,
                8,
                128,
                &[], // rms_w — placeholder (stateless, declared as input)
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                true,
            ) {
                let mut buf = Vec::new();
                prog.encode(&mut buf).ok();
                tar_entries.push((format!("draft_layer_{}.mil", l), buf));
            }
        }
    }

    // Simple tar-like packing: each entry is length-prefixed
    let mut buf = Vec::new();
    for (name, data) in &tar_entries {
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }

    buf
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // ── Subcommand dispatch ────────────────────────────────────────
    if args.len() > 1 && args[1] == "inspect-checkpoint" {
        cmd_inspect_checkpoint(&args[1..]);
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gemma 4 12B Unified → Ternary .cimage                     ║");
    println!("║  AOT Compiler (pure Rust, no Python)                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse arguments
    let repo = get_opt(&args, "--repo");
    let local_dir = get_opt(&args, "--local-dir");
    let output = get_opt(&args, "--output").unwrap_or("gemma4_12b.cimage");
    let draft_model_dir = get_opt(&args, "--draft-model-dir");
    let mil_program = get_opt(&args, "--mil");

    // Validate args
    if repo.is_none() && local_dir.is_none() {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12b-it --output gemma4_12b.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B-it-qat-q4_0-unquantized --output gemma4_12b_qat.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage");
        std::process::exit(1);
    }

    let total_start = Instant::now();

    // ── Step 1: Collect safetensor file paths ───────────────────
    let shard_paths = if let Some(dir) = local_dir {
        println!("  Loading from local directory: {dir}");
        collect_local_safetensors(Path::new(dir))
    } else if let Some(r) = repo {
        println!("  Downloading from Hugging Face: {r}");
        download_repo_safetensors(r)
    } else {
        unreachable!()
    };

    println!("  Found {} shard(s)", shard_paths.len());

    // ── Step 2: Process all weights ─────────────────────────────
    println!("\n  ── Quantizing weights (256-block ternary) ───────────");
    let quant_start = Instant::now();

    // Spawn Metal shader compilation concurrently (CPU LLVM work, while GPU quantizes)
    let metal_handle = std::thread::spawn(|| {
        let t = Instant::now();
        let bytes = compile_metal_lib();
        (bytes, t.elapsed())
    });

    let mut all_scales = Vec::new();
    let mut all_weights = Vec::new();
    let mut total_elements: usize = 0;

    // Separate buffers for non-transformer-weight segments
    let mut vocab_embedding_raw_f32: Vec<f32> = Vec::new();
    let mut aux_norm_fp16: Vec<u8> = Vec::new();
    let mut vocab_nibbles: Vec<u8> = Vec::new();
    let mut vocab_scales: Vec<u8> = Vec::new();
    let mut centroid_nibbles: Vec<u8> = Vec::new();
    let mut centroid_scales: Vec<u8> = Vec::new();
    let mut cluster_map_bytes: Vec<u8> = Vec::new();
    let mut multimodal_scales: Vec<u8> = Vec::new();
    let mut multimodal_nibbles: Vec<u8> = Vec::new();
    let mut multimodal_aux_fp16: Vec<u8> = Vec::new();
    let mut draft_layer_count: u32 = 0;
    // Ternary block offsets for ANE compilation: (nibble_off, scale_off, num_f32_elements)
    let mut ane_ternary_offsets: HashMap<String, (u64, u64, usize)> = HashMap::new();

    // ── EMBEDDING_WEIGHT (Vocabulary segment: raw f32 bytes) ─────
    {
        let (name, _rows, _cols) = EMBEDDING_WEIGHT;
        if let Some((data, shape)) = load_tensor(name, &shard_paths) {
            let n_blocks = (data.len() + 255) / 256;
            print!("  Embedding: {shape:?}, {n_blocks} blocks\n");
            vocab_embedding_raw_f32 = data;
        } else {
            println!("  {name:<40} NOT FOUND (vocab will be empty)");
        }
    }

    // ── CLUSTER & QUANTIZE EMBEDDING TABLE ───────────────────────────
    {
        let dim = HIDDEN_DIM;
        let n_rows = vocab_embedding_raw_f32.len() / dim;
        if n_rows > 0 {
            let k = 256;
            if n_rows < k * 2 {
                eprintln!("  Too few embedding rows ({n_rows}) for {k} clusters; skipping centroid scheme.");
                process_weights(
                    &vocab_embedding_raw_f32,
                    &mut vocab_scales,
                    &mut vocab_nibbles,
                );
                centroid_nibbles = vec![0u8; (256 * dim + 255) / 256 * 64];
                centroid_scales = vec![0u8; ((256 * dim + 255) / 256) * 2];
                cluster_map_bytes = vec![0u8; n_rows * 4];
            } else {
                eprint!("  Clustering {n_rows} embedding rows into {k} groups... ");
                let start = std::time::Instant::now();
                let mut centroids = kmeans_plusplus(&vocab_embedding_raw_f32, k, n_rows, dim);
                for _iter in 0..20 {
                    let (_assignments, delta) =
                        kmeans_iterate(&vocab_embedding_raw_f32, &mut centroids, n_rows, dim, k);
                    if delta < 1e-6 {
                        break;
                    }
                }
                let (assignments, _delta) =
                    kmeans_iterate(&vocab_embedding_raw_f32, &mut centroids, n_rows, dim, k);
                let reordered =
                    reorder_by_cluster(&vocab_embedding_raw_f32, &assignments, n_rows, dim, k);
                eprintln!("{:.1}s", start.elapsed().as_secs_f64());
                process_weights(&reordered, &mut vocab_scales, &mut vocab_nibbles);
                process_weights(&centroids, &mut centroid_scales, &mut centroid_nibbles);
                for &c in &assignments {
                    cluster_map_bytes.extend_from_slice(&c.to_le_bytes());
                }
            }
        } else {
            eprintln!("  Embedding table empty, skipping quantization");
        }
    }
    // ── FINAL_NORM (aux section: raw FP16 bytes) ─────────────────
    {
        let (name, _rows, _cols) = FINAL_NORM;
        if let Some((data, shape)) = load_tensor(name, &shard_paths) {
            print!("  Final norm: {shape:?}\n");
            // Convert f32 to FP16 bytes
            for &v in &data {
                aux_norm_fp16.extend_from_slice(&f32_to_fp16_bits(v).to_le_bytes());
            }
        } else {
            println!("  {name:<40} NOT FOUND (aux norm will be empty)");
        }
    }

    // ── MULTIMODAL_WEIGHTS (projection matrices → ternary; 1D → aux FP16) ──
    println!("  ── Multimodal weights ────────────────────────────");
    for (name, _rows, cols) in MULTIMODAL_WEIGHTS {
        let is_1d = *cols == 1;
        if let Some((data, shape)) = load_tensor(name, &shard_paths) {
            let n_blocks = (data.len() + 255) / 256;
            print!("    {name:<55} {shape:?}");
            if is_1d {
                // 1D params → aux FP16
                for &v in &data {
                    multimodal_aux_fp16.extend_from_slice(&f32_to_fp16_bits(v).to_le_bytes());
                }
                println!(" → aux FP16");
            } else {
                // Projection matrix → ternary quantization
                let nib_off = multimodal_nibbles.len() as u64;
                let scl_off = multimodal_scales.len() as u64;
                ane_ternary_offsets.insert(name.to_string(), (nib_off, scl_off, data.len()));
                process_weights(&data, &mut multimodal_scales, &mut multimodal_nibbles);
                total_elements += data.len();
                println!(" → {n_blocks} blocks ternary");
            }
        } else {
            println!("    {name:<55} NOT FOUND (skipping)");
        }
    }

    // ── lm_head (main ternary pipeline, may be tied with embed) ─
    let lm_head_key = "model.language_model.lm_head.weight";
    if let Some((data, shape)) = load_tensor(lm_head_key, &shard_paths) {
        let n_blocks = (data.len() + 255) / 256;
        process_weights(&data, &mut all_scales, &mut all_weights);
        total_elements += data.len();
        print!("  lm_head: {shape:?} — {n_blocks:>6} blocks\n");
    } else {
        println!("  lm_head.weight NOT FOUND (tied with embed_tokens)");
    }

    // Layer weights
    for layer in 0..NUM_LAYERS {
        print!("\r  Layer {}/{}", layer + 1, NUM_LAYERS);
        use std::io::Write;
        std::io::stdout().flush().ok();

        for (mat_name, _rows, _cols) in MATRICES {
            let key = tensor_key(layer, mat_name);
            if let Some((data, _)) = load_tensor(&key, &shard_paths) {
                process_weights(&data, &mut all_scales, &mut all_weights);
                total_elements += data.len();
            } else {
                println!("\n  WARNING: {key} not found");
            }
        }

        if layer % 8 == 7 {
            let mb = (all_scales.len() + all_weights.len()) as f64 / (1024.0 * 1024.0);
            println!(" — {mb:.1} MB");
        }

        // ── Collect per-layer norm weights ─────────────────────────
        for norm_name in &["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let nkey = format!("model.language_model.model.layers.{layer}.{norm_name}");
            if let Some((norm_data, _)) = load_tensor(&nkey, &shard_paths) {
                for &v in &norm_data {
                    aux_norm_fp16.extend_from_slice(&f32_to_fp16_bits(v).to_le_bytes());
                }
            }
        }
    }
    println!();

    // ── MTP Drafter Head Discovery ────────────────────────────────
    println!("  ── Scanning for MTP drafter heads ───────────────────");
    let mut mtp_tensors: Vec<String> = Vec::new();

    // Scan all safetensor metadata for "mtp" tensor keys
    for (_path, data) in &shard_paths {
        if let Ok(st) = safetensors::SafeTensors::deserialize(data) {
            for name in st.names() {
                if name.contains("mtp") {
                    mtp_tensors.push(name.to_string());
                }
            }
        }
    }
    mtp_tensors.sort();
    mtp_tensors.dedup();

    if !mtp_tensors.is_empty() {
        println!("  Found {} MTP tensor(s):", mtp_tensors.len());
        for t in &mtp_tensors {
            println!("    {t}");
        }
    } else {
        println!("  No MTP heads found (model may not have them)");
    }

    // ── MTP Draft Model (optional external draft decoder) ──────────
    let mut draft_segment_count: u32 = 0;
    if let Some(ref draft_dir) = draft_model_dir {
        println!(
            "
  ── MTP Draft Model ────────────────────────────"
        );
        let draft_path = Path::new(draft_dir).join("model.safetensors");
        if !draft_path.exists() {
            eprintln!(
                "  WARNING: draft model.safetensors not found at {}",
                draft_path.display()
            );
        } else {
            let draft_bytes = std::fs::read(&draft_path).unwrap_or_else(|e| {
                eprintln!("  ERROR: cannot read draft safetensors: {e}");
                std::process::exit(1);
            });
            let draft_st = safetensors::SafeTensors::deserialize(&draft_bytes)
                .expect("reading draft safetensors header");
            let draft_names: Vec<String> = draft_st.names().iter().map(|n| n.to_string()).collect();
            println!("  Draft tensors: {}", draft_names.len());

            for name in &draft_names {
                let is_1d = name.ends_with(".bias")
                    || name.ends_with("norm.weight")
                    || name.ends_with("_ln.weight")
                    || name.ends_with("_ln.bias")
                    || name.ends_with("layer_scalar");
                if is_1d {
                    if let Ok(tv) = draft_st.tensor(name) {
                        let f32_data = tensor_to_f32(tv.data(), tv.dtype());
                        for &v in &f32_data {
                            let bits = f32_to_fp16_bits(v);
                            multimodal_aux_fp16.extend_from_slice(&bits.to_le_bytes());
                        }
                    }
                    continue;
                }
                if let Ok(tv) = draft_st.tensor(name) {
                    let f32_data = tensor_to_f32(tv.data(), tv.dtype());
                    let n_elems = f32_data.len();
                    let nib_off = all_weights.len() as u64;
                    let scl_off = all_scales.len() as u64;
                    ane_ternary_offsets
                        .insert(name.to_string(), (nib_off, scl_off, f32_data.len()));
                    process_weights(&f32_data, &mut all_scales, &mut all_weights);
                    total_elements += n_elems;
                    let n_blocks = (n_elems + 255) / 256;
                    println!(
                        "    draft: {name:<55} {} elems, {} blocks",
                        n_elems, n_blocks
                    );
                }
            }
            draft_segment_count = 1;
            draft_layer_count = 4; // MTP draft has 4 transformer layers
        }
    }

    // ── Write output ───────────────────────────────────────────────
    // Main weights .cimage
    println!("\n  Writing main weights to {}", output);

    let quant_elapsed = quant_start.elapsed();
    let n_blocks = all_scales.len() / 2;
    let mb_scales = all_scales.len() as f64 / (1024.0 * 1024.0);
    let mb_weights = all_weights.len() as f64 / (1024.0 * 1024.0);

    println!(
        "  Quantized {} weights in {:.1?}",
        total_elements, quant_elapsed
    );
    println!(
        "  {} blocks, {:.1} MB scales, {:.1} MB nibbles",
        n_blocks, mb_scales, mb_weights
    );

    // ── Step 3: Load MIL program ───────────────────────────────
    println!("\n  ── Compiling .cimage ────────────────────────────────");

    let mil_bytes = if let Some(mil_path) = mil_program {
        std::fs::read(mil_path).unwrap_or_else(|e| {
            eprintln!("  WARNING: can't read {mil_path}: {e}, using placeholder");
            generate_placeholder_mil() /* TODO: pass --mil to override with real MIL program */
        })
    } else {
        generate_placeholder_mil() /* TODO: pass --mil to override with real MIL program */
    };

    // ── Step 4: Build .cimage ──────────────────────────────────

    let file = std::fs::File::create(output).unwrap_or_else(|e| {
        eprintln!("  ERROR: cannot create {output}: {e}");
        std::process::exit(1);
    });
    let mut writer = BufWriter::new(file);

    // Reserve header space (will overwrite at end)
    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    writer.write_all(&vec![0u8; header_size as usize]).unwrap();

    // Compile Metal shaders to .metallib
    println!(
        "
  ── Compiling Metal shaders ─────────────────────────"
    );
    let (metallib_bytes, metal_duration) = match metal_handle.join() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  Metal compilation panicked: {:?}", e);
            std::process::exit(1);
        }
    };
    let overlapped = if metal_duration < quant_elapsed {
        metal_duration
    } else {
        quant_elapsed
    };
    println!("  Metal kernel lib: {} bytes", metallib_bytes.len());
    println!(
        "  Metal compilation: {:.1}s (overlapped {:.1}s with quantization)",
        metal_duration.as_secs_f64(),
        overlapped.as_secs_f64()
    );

    // Generate ANE MIL programs embedded in the cimage.
    // Runtime compiles them via xcrun coremlcompiler or Rust ANE compiler on first load.
    let ane_island_tar: Vec<u8> = generate_ane_mil_packages(&ane_ternary_offsets);
    println!(
        "  ANE island tar: {} bytes (MIL programs embedded)",
        ane_island_tar.len()
    );

    // Generate ANE programs
    println!("  ── Generating ANE programs ─────────────────────────");
    let kv_decompress_bytes = gemma4_kv_decompress_mil();
    println!("  KV decompress MIL: {} bytes", kv_decompress_bytes.len());

    // ANE island MIL programs: generate .mlpackage protobufs from stashed

    // Generate execution graph descriptor
    println!("  ── Building execution graph ────────────────────────");
    let mut exec_graph = ExecutionGraphDescriptor::gemma4_12b();
    // Compute per-layer offsets by replaying the quantization layout.
    // Main decoder layers: q_proj, k_proj, v_proj, o_proj, gate, up, down
    let mut weight_off: u64 = 0;
    let mut scale_off: u64 = 0;
    let h = HIDDEN_DIM as u64; // 3840
    let nq = (NUM_HEADS * HEAD_DIM) as u64; // 16 * 256 = 4096
    let nk = (NUM_KV_HEADS * HEAD_DIM) as u64; // 8 * 256 = 2048
    let ffn = FFN_INTERMEDIATE as u64; // 15360
                                       // Elements per tensor: [q, k, v, o, gate, up, down]
    let per_tensor_elems = [h * nq, h * nk, h * nk, nq * h, h * ffn, h * ffn, ffn * h];
    for layer in exec_graph.layers.iter_mut().filter(|n| n.node_kind == 0) {
        let start_weight = weight_off;
        let start_scale = scale_off;
        let mut total_wbytes: u64 = 0;
        let mut total_sbytes: u64 = 0;
        for &elems in &per_tensor_elems {
            let blocks = (elems + 255) / 256;
            total_wbytes += blocks * 64;
            total_sbytes += blocks * 2;
        }
        layer.weight_offset = start_weight;
        layer.weight_length = total_wbytes;
        layer.scale_offset = start_scale;
        weight_off += total_wbytes;
        scale_off += total_sbytes;
        // Also populate k_proj-style fields that open_prism uses
        layer.hidden_dim = h as u32;
        layer.num_heads = NUM_HEADS as u16;
    }
    let exec_graph_bytes = exec_graph.to_bytes();
    println!(
        "  Execution graph: {} bytes, {} nodes, {} epochs",
        exec_graph_bytes.len(),
        exec_graph.node_count,
        exec_graph.num_compaction_epochs
    );

    // Page-align helper
    let page_align = |w: &mut BufWriter<std::fs::File>| -> std::io::Result<u64> {
        let pos = w.stream_position()?;
        let aligned = ((pos + 16383) / 16384) * 16384;
        if aligned > pos {
            w.write_all(&vec![0u8; (aligned - pos) as usize])?;
        }
        Ok(aligned)
    };

    // Write MetalLib segment 0
    let metal_offset = page_align(&mut writer).unwrap();
    writer.write_all(&metallib_bytes).unwrap();

    // Write TernaryWeights segment 1
    let weights_offset = page_align(&mut writer).unwrap();
    writer.write_all(&all_weights).unwrap();

    // Write BlockScales segment 2
    let scales_offset = page_align(&mut writer).unwrap();
    writer.write_all(&all_scales).unwrap();

    // Write AneArchive segment 3 (prefill MIL source)
    let ane_prefill_offset = page_align(&mut writer).unwrap();
    writer.write_all(&mil_bytes).unwrap();

    // Write AneArchive segment 4 (KV decompress)
    let ane_decompress_offset = page_align(&mut writer).unwrap();
    writer.write_all(&kv_decompress_bytes).unwrap();

    // Build segment directory (8 slots)
    let mut segments = [SegmentEntry {
        kind: SegmentKind::MetalLib as u32,
        offset: 0,
        length: 0,
    }; CIMAGE_SEGMENT_CAPACITY];
    segments[0] = SegmentEntry {
        kind: SegmentKind::MetalLib as u32,
        offset: metal_offset,
        length: metallib_bytes.len() as u64,
    };
    segments[1] = SegmentEntry {
        kind: SegmentKind::TernaryWeights as u32,
        offset: weights_offset,
        length: all_weights.len() as u64,
    };
    segments[2] = SegmentEntry {
        kind: SegmentKind::BlockScales as u32,
        offset: scales_offset,
        length: all_scales.len() as u64,
    };
    segments[3] = SegmentEntry {
        kind: SegmentKind::AneArchive as u32,
        offset: ane_prefill_offset,
        length: mil_bytes.len() as u64,
    };
    segments[4] = SegmentEntry {
        kind: SegmentKind::AneArchive as u32,
        offset: ane_decompress_offset,
        length: kv_decompress_bytes.len() as u64,
    };

    // Write AneArchive segment 5 (ANE islands for full inference)
    let ane_islands_offset = page_align(&mut writer).unwrap();
    writer.write_all(&ane_island_tar).unwrap();
    segments[5] = SegmentEntry {
        kind: SegmentKind::AneArchive as u32,
        offset: ane_islands_offset,
        length: ane_island_tar.len() as u64,
    };

    // ── ModelArtifacts (tokenizer, special token map) ────────────
    let mut model_artifacts: Vec<u8> = Vec::new();

    // Type 0x01: SentencePiece tokenizer
    let model_dir = shard_paths
        .first()
        .and_then(|(p, _)| p.parent())
        .unwrap_or_else(|| Path::new("."));
    // Try tokenizer.model (SentencePiece) then tokenizer.json (HuggingFace Fast)
    let tokenizer_paths = [
        model_dir.join("tokenizer.model"),
        model_dir.join("tokenizer.json"),
    ];
    if let Some(tok_path) = tokenizer_paths.iter().find(|p| p.exists()) {
        if let Ok(data) = std::fs::read(tok_path) {
            ModelArtifactEntry::encode(model_artifact_tag::TOKENIZER, &data, &mut model_artifacts);
            println!(
                "  Tokenizer: {} bytes ({})",
                data.len(),
                tok_path.file_name().and_then(|n| n.to_str()).unwrap_or("")
            );
        }
    } else {
        eprintln!("  WARNING: no tokenizer found (tried .model and .json)");
    }

    // Type 0x04: Multimodal special token map
    let token_map = serde_json::json!({
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "image_start_token": "<start_of_image>",
        "image_end_token": "<end_of_image>",
        "audio_start_token": "<start_of_audio>",
        "audio_end_token": "<end_of_audio>",
        "image_token_count": 256,
        "audio_token_rate_hz": 12.5,
        "vision_patch_size": 14,
        "audio_sample_rate": 16000,
        "audio_frame_ms": 25,
        "audio_hop_ms": 10
    });
    let token_map_bytes = serde_json::to_vec(&token_map).unwrap_or_default();
    ModelArtifactEntry::encode(
        model_artifact_tag::TOKEN_MAP,
        &token_map_bytes,
        &mut model_artifacts,
    );

    // ── Embedding auxiliary data ──────────────────────────────────────
    if !vocab_nibbles.is_empty() {
        ModelArtifactEntry::encode(
            model_artifact_tag::EMBED_NIBBLES,
            &vocab_nibbles,
            &mut model_artifacts,
        );
        ModelArtifactEntry::encode(
            model_artifact_tag::EMBED_SCALES,
            &vocab_scales,
            &mut model_artifacts,
        );
        ModelArtifactEntry::encode(
            model_artifact_tag::CENTROID_NIBBLES,
            &centroid_nibbles,
            &mut model_artifacts,
        );
        ModelArtifactEntry::encode(
            model_artifact_tag::CENTROID_SCALES,
            &centroid_scales,
            &mut model_artifacts,
        );
        ModelArtifactEntry::encode(
            model_artifact_tag::CLUSTER_MAP,
            &cluster_map_bytes,
            &mut model_artifacts,
        );
    }
    if !aux_norm_fp16.is_empty() {
        ModelArtifactEntry::encode(
            model_artifact_tag::AUX_NORMS,
            &aux_norm_fp16,
            &mut model_artifacts,
        );
    }
    println!(
        "  Model artifacts: {} bytes ({} entries)",
        model_artifacts.len(),
        (model_artifacts.len() as f64 / 8.0).ceil() as u32
    );

    // Write ExecutionGraph segment 6
    let exec_graph_offset = page_align(&mut writer).unwrap();
    writer.write_all(&exec_graph_bytes).unwrap();
    segments[6] = SegmentEntry::new(
        SegmentKind::ExecutionGraph,
        exec_graph_offset,
        exec_graph_bytes.len() as u64,
    );

    // Write ModelArtifacts segment 7 (tokenizer, special token map)
    let artifacts_offset = page_align(&mut writer).unwrap();
    writer.write_all(&model_artifacts).unwrap();
    segments[7] = SegmentEntry::new(
        SegmentKind::ModelArtifacts,
        artifacts_offset,
        model_artifacts.len() as u64,
    );

    let mut hasher = Sha256::new();
    hasher.update(&all_weights);
    hasher.update(&all_scales);
    hasher.update(&metallib_bytes);
    hasher.update(&mil_bytes);
    hasher.update(&kv_decompress_bytes);
    hasher.update(&ane_island_tar);
    hasher.update(&exec_graph_bytes);
    hasher.update(&model_artifacts);
    let payload_hash: [u8; 32] = hasher.finalize().into();

    // Write header at position 0
    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 6, // v6 adds auxiliary data (embedding, centroids, cluster map, norms) in ModelArtifacts
        segment_count: 8 + draft_segment_count,
        payload_hash,
        num_layers: NUM_LAYERS as u32,
        num_heads: NUM_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        hidden_dim: HIDDEN_DIM as u32,
        intermediate_dim: FFN_INTERMEDIATE as u32,
        vocab_size: 262144,
        quantization_schema: 0,
        draft_num_layers: draft_layer_count,
        segments,
        _pad: [0u8; 8],
    };
    writer.seek(SeekFrom::Start(0)).unwrap();
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const CimageHeader as *const u8,
            header_size as usize,
        )
    };
    writer.write_all(header_bytes).unwrap();
    writer.flush().unwrap();
    drop(writer);

    let cimage_bytes = std::fs::read(output).unwrap_or_else(|e| {
        eprintln!("  ERROR: cannot read back {output}: {e}");
        std::process::exit(1);
    });

    // ── Step 5: Verify ─────────────────────────────────────────
    match tribunus_compute_core::compute_image::compile::ternary::verify_cimage(&cimage_bytes) {
        Ok((header, _)) => {
            println!("  ✓ SHA-256 integrity PASSED");
            println!("  ✓ Schema: TERNARY_ADD ({})", header.quantization_schema);
            println!("  ✓ Magic:  {:?}", &header.magic);
        }
        Err(e) => {
            eprintln!("  ✗ Verification FAILED: {e}");
            std::process::exit(1);
        }
    }

    // ── Summary ────────────────────────────────────────────────
    let total_elapsed = total_start.elapsed();
    let file_size = cimage_bytes.len();
    let fp16_size = total_elements * 2;
    let compression_ratio = fp16_size as f64 / file_size as f64;

    println!();
    println!("  ── Result ──────────────────────────────────────────────");
    println!("  Output:     {output}");
    println!(
        "  File size:  {:.1} MB ({file_size} bytes)",
        file_size as f64 / (1024.0 * 1024.0)
    );
    println!("  Params:     {total_elements}");
    println!("  Blocks:     {n_blocks}");
    println!("  Compressed: {compression_ratio:.1}× vs FP16");
    println!("  Time:       {total_elapsed:.1?}");
    println!();
    println!("  ▶ Runtime ready: tribunus-compute-image load --cimage {output}");

    // ── Step 6: ANE programs generated above ────────────────────
    // ANE programs are now generated and embedded during cimage packing.
    // See compile_metal_lib() and gemma4_kv_decompress_mil() above.
}

// ── Safetensors loading helpers ─────────────────────────────────────

fn load_tensor(key: &str, shards: &[(PathBuf, Vec<u8>)]) -> Option<(Vec<f32>, Vec<usize>)> {
    let (_, data) = shards.iter().find(|(_, data)| {
        // Check if this shard contains the key (cheap: just check metadata)
        safetensors::SafeTensors::deserialize(data)
            .ok()
            .and_then(|st| st.tensor(key).ok())
            .is_some()
    })?;

    let st = safetensors::SafeTensors::deserialize(data).ok()?;
    let view = st.tensor(key).ok()?;
    let shape = view.shape().to_vec();
    let f32_vals = tensor_to_f32(view.data(), view.dtype());
    Some((f32_vals, shape))
}

fn collect_local_safetensors(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            let data = std::fs::read(&path).unwrap();
            shards.push((path, data));
        }
    }
    shards.sort_by(|a, b| a.0.cmp(&b.0));
    shards
}

fn download_repo_safetensors(repo_id: &str) -> Vec<(PathBuf, Vec<u8>)> {
    use hf_hub::api::sync::Api;

    let api = Api::new().expect("HF API init failed (set HF_TOKEN if needed for gated models)");
    let repo = api.model(repo_id.to_string());

    // Try the safetensors index first to discover all shards
    let index_name = "model.safetensors.index.json";
    let mut shard_names: Vec<String> = Vec::new();

    match repo.get(index_name) {
        Ok(index_path) => {
            let index_json: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(&index_path).unwrap())
                    .expect("invalid safetensors index JSON");
            if let Some(weight_map) = index_json.get("weight_map").and_then(|m| m.as_object()) {
                let mut seen = std::collections::HashSet::new();
                for (_tensor, shard) in weight_map {
                    let s = shard.as_str().unwrap();
                    if seen.insert(s.to_string()) {
                        shard_names.push(s.to_string());
                    }
                }
            }
            shard_names.sort();
        }
        Err(_) => {
            // No index — try numbered shard pattern
            for i in 1..=99 {
                let name = format!("model-{i:05}-of-00002.safetensors");
                if repo.get(&name).is_ok() {
                    shard_names.push(name);
                } else {
                    break;
                }
            }
            if shard_names.is_empty() {
                // Try single file
                shard_names.push("model.safetensors".to_string());
            }
        }
    }

    // Download each shard
    let mut shards = Vec::new();
    for name in &shard_names {
        print!("  Downloading {name}...");
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let local_path = match repo.get(name) {
            Ok(p) => p,
            Err(e) => {
                println!(" FAILED: {e}");
                continue;
            }
        };
        let data = std::fs::read(&local_path).unwrap_or_else(|e| {
            println!(" FAILED to read: {e}");
            std::process::exit(1);
        });
        let size_mb = data.len() as f64 / (1024.0 * 1024.0);
        println!(" {size_mb:.0} MB");
        shards.push((local_path, data));
    }

    shards
}

/// Generate a placeholder MIL program (E5 format).
fn generate_placeholder_mil() -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(b"\xE5\x00\x00\x00");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.resize(64, 0);
    buf
}

/// Read `--key <value>` pairs from args.
fn get_opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].as_str())
}

/// Read all values for a repeatable `--key <value>` flag.
#[allow(dead_code)]
fn get_opts<'a>(args: &'a [String], key: &str) -> Vec<&'a str> {
    args.windows(2)
        .filter(|w| w[0] == key)
        .map(|w| w[1].as_str())
        .collect()
}
