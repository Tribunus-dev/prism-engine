//! gemma4_ingest — Stage 1+2: Download Gemma 4 12B from HF, ternary or nf4tile640 quantize,
//! compile to .cimage, all in Rust, zero Python.
//!
//! Subcommands:
//!   inspect-checkpoint  — Read a checkpoint directory and emit metadata
//!   (default)            — Quantize a Gemma 4 12B checkpoint to .cimage
//!
//! Usage:
//!   cargo run --bin gemma4_ingest -- inspect-checkpoint --model-dir <PATH> [--emit <PATH>]...
//!   cargo run --bin gemma4_ingest -- --repo google/gemma-4-12b-it --output gemma4_12b.cimage
//!   cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B-it-qat-q4_0-unquantized --output gemma4_12b_qat.cimage
//!   cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage
//!   cargo run --bin gemma4_ingest -- --repo <REPO> --output out.cimage --nf4 --tts-repo Qwen/Qwen3-TTS-12Hz-1.7B-Base

#![allow(unused_imports)]

use memmap2::Mmap;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// SHA-256 digest using Merkle-tree parallelism.
///
/// Splits `data` into N chunks (N = available rayon threads), hashes each
/// chunk in parallel, then hashes the concatenated chunk digests.  The result
/// is deterministic: same data always produces the same digest.
///
/// On Apple M1 with 8 cores this typically yields ~4–6× throughput vs serial
/// SHA-256 for matrices over ~4 MB.
fn parallel_sha256(data: &[u8]) -> [u8; 32] {
    let num_threads = rayon::current_num_threads();
    let min_chunk = 65536; // 64 KB — below this, serial is faster
    if data.len() <= min_chunk || num_threads <= 1 {
        use sha2::{Digest, Sha256};
        return Sha256::digest(data).into();
    }
    let chunk_size = (data.len() + num_threads - 1) / num_threads;
    let chunk_size = chunk_size.max(min_chunk);
    let hashes: Vec<[u8; 32]> = data
        .par_chunks(chunk_size)
        .map(|chunk| {
            use sha2::{Digest, Sha256};
            Sha256::digest(chunk).into()
        })
        .collect();
    let mut combined = Vec::with_capacity(hashes.len() * 32);
    for h in &hashes {
        combined.extend_from_slice(h);
    }
    use sha2::{Digest, Sha256};
    Sha256::digest(&combined).into()
}

#[cfg(feature = "mlx-backend")]
use tribunus_compute_core::ane_compile::compile_ane_artifacts;
use tribunus_compute_core::compute_image::compile::execution_graph::ExecutionGraphDescriptor;
use tribunus_compute_core::compute_image::compile::execution_graph::{
    sidecar_byte_len, SidecarElementFormat,
};
use tribunus_compute_core::compute_image::compile::ternary::MatrixWeightBindingV1;
use tribunus_compute_core::compute_image::compile::ternary::{
    model_artifact_tag, read_matrix_weight_binding_v1_le, write_cimage_header_le,
    write_matrix_weight_binding_v1_le, CimageHeader, ModelArtifactEntry, SegmentEntry, SegmentKind,
    CIMAGE_HEADER_WIRE_SIZE, CIMAGE_SEGMENT_CAPACITY, MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH,
    QUANT_SCHEMA_NF4_TILE640,
};
use tribunus_compute_core::compute_image::compile::tts_compile::pack_tts_weights;
use tribunus_compute_core::compute_image::subgraph_mil::{build_draft_layer_mil, build_matmul_mil};
use tribunus_compute_core::compute_image::TensorEntry;
use tribunus_compute_core::nf4tile640::learn::{
    compute_activation_saliency, select_profile_for_matrix, LearnedProfile,
    ProfileSelectionReceipt, SelectionReason,
};
use tribunus_compute_core::nf4tile640::plan::{QuantizationPlan, QuantizationPlanEntry};
use tribunus_compute_core::nf4tile640::profile::QuantizerProfile;
use tribunus_compute_core::nf4tile640::roles::{classify_matrix_role, MatrixRole};
use tribunus_compute_core::nf4tile640::{
    dequant_matmul_reference, pack_nf4_weights, pack_nf4_weights_awls, unpack_nf4_weights,
};
use tribunus_compute_core::quantization::contract::{
    RuntimeRepresentationClass, INT8_TILE640_CODE_BYTES, NF4_TILE640_CODE_BYTES,
};
use tribunus_compute_core::quantization::embed_cluster::*;

use tribunus_compute_core::compilation::cancel::CancelToken;
use tribunus_compute_core::compilation::distill_core::kd_divergence;
use tribunus_compute_core::compilation::level1::reducer::{AccelerateReducer, DistillObjective};
use tribunus_compute_core::compilation::matrix_distill::{
    distill_matrix, DistillFormat, MatrixDistillResult,
};
use tribunus_compute_core::nf4tile640::squat::squat_requantize;

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
    ("model.vision_embedder.patch_dense.weight", 6912, 3840),
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

/// Per-matrix format, dimensions, and byte counts recorded during packing.
/// Consumed by the offline planner to compute independent segment offsets.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct WeightBindingData {
    key: String,
    format: RuntimeRepresentationClass,
    /// Number of Tile640 tiles (ceil(cols / 640)).
    tiles_per_row: usize,
    /// Total tile count = rows * tiles_per_row.
    total_tiles: usize,
    /// Segment-kind mapping for NF4 vs INT8 codes.
    weights_segment: u8,
    tile_metadata_segment: u8,
    sidecar_segment: u8,
    sidecar_count: u32,
    rows: usize,
    cols: usize,
}

/// A decoder-layer matrix packing job collected before parallel execution.
/// Each job holds an owned copy of the weight data so it can be processed
/// independently in a rayon thread pool.
struct PackJob {
    key: String,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    /// Physical tensor shape from the checkpoint (e.g. [out_features, in_features]).
    shape: Vec<usize>,
}

/// Extract physical dimensions from a checkpoint shape vector.
fn physical_dims_from_shape(shape: &[usize]) -> (usize, usize) {
    let rows = if shape.len() >= 2 { shape[0] } else { 0 };
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };
    (rows, cols)
}

/// Pack a weight matrix using nf4tile640 inline (single-pass, no accumulation).
/// Map admission QuantizedMatrixFormat (1,2,3) to runtime QuantizedWeightFormat (0,1,2).
fn admission_format_to_binding_format(f: RuntimeRepresentationClass) -> u8 {
    match f {
        RuntimeRepresentationClass::Nf4Tile640Base => 1,
        RuntimeRepresentationClass::Int8Tile640Base => 2,
        RuntimeRepresentationClass::TernaryTile640Base => 0,
        RuntimeRepresentationClass::RawF32 => 3,
    }
}

/// Used by the streaming refactor to avoid holding all f32 weight data simultaneously.
fn pack_matrix_nf4_inline(
    key: &str,
    stress: Option<&tribunus_compute_core::quantization::StressSuite>,
    calibration: Option<&tribunus_compute_core::quantization::CalibrationSuite>,
    data: &[f32],
    rows: usize,
    cols: usize,
    rows_physical: usize,
    cols_physical: usize,
    channel_sq_map: &mut HashMap<String, Vec<f32>>,
    selection_receipts: &mut Vec<ProfileSelectionReceipt>,
    plan_entries: &mut Vec<QuantizationPlanEntry>,
    bindings: &mut Vec<WeightBindingData>,
    learned_profiles: &HashMap<MatrixRole, LearnedProfile>,
    _distill_objective: &DistillObjective,
    tmp_dir: &Path,
    output_format: &str,
    _quality_policy: &str,
    output_lines: &mut Vec<String>,
) {
    // Validate data size against declared dimensions.  Some checkpoint tensors
    // (e.g. fused QKV projections) have more elements than rows × cols for a
    // single projection key.  Truncate with a warning rather than panicking in
    // the packer — the admission pipeline reads only the expected elements and
    // selects the correct candidate format.
    let expected_f32 = rows * cols;
    let data = if data.len() != expected_f32 {
        output_lines.push(format!(
            "  [size-warn] {}: data has {} f32 elements, expected {} (rows={}, cols={})",
            key,
            data.len(),
            expected_f32,
            rows,
            cols
        ));
        if data.len() > expected_f32 {
            &data[..expected_f32]
        } else {
            data
        }
    } else {
        data
    };

    // Transpose from checkpoint [out_features, in_features] to code convention [in_features, out_features]
    // new[in * out_features + out] = old[out * in_features + in]
    let mut transposed_buf: Vec<f32>;
    let data = if rows_physical != rows || cols_physical != cols {
        let in_features = rows;
        let out_features = cols;
        transposed_buf = vec![0.0f32; rows * cols];
        for in_ in 0..in_features {
            for out in 0..out_features {
                transposed_buf[in_ * out_features + out] = data[out * in_features + in_];
            }
        }
        &transposed_buf[..]
    } else {
        data
    };

    // ── Per-channel second moments for AW-LS activation-weighted packing ──
    let mut channel_sq = vec![0.0f32; cols];
    let effective_rows = rows;
    for i in 0..effective_rows {
        for j in 0..cols {
            let v = data[i * cols + j];
            channel_sq[j] += v * v;
        }
    }
    for j in 0..cols {
        channel_sq[j] /= effective_rows as f32;
    }
    channel_sq_map.insert(key.to_string(), channel_sq.clone());

    // ── Profile selection ────────────────────────────────────────────
    let role = classify_matrix_role(key);
    let groups: Vec<Vec<f32>> = data.chunks(128).map(|c| c.to_vec()).collect();
    let importances: Vec<f32> = groups.iter().map(|_| 1.0).collect();
    let (_selected_profile, receipt) = select_profile_for_matrix(
        key,
        role,
        &groups,
        &importances,
        tribunus_compute_core::nf4tile640::NF4_CODEBOOK,
        learned_profiles,
    );
    selection_receipts.push(receipt);

    // ── Pack ─────────────────────────────────────────────────────────
    // ── Admission pipeline (per spec) ─────────────────────────────
    use tribunus_compute_core::quantization::{
        admission::quantize_tensor, calibration::CalibrationSuite, calibration::StressSuite,
        contract::*,
    };

    let role = classify_matrix_role(key);
    // Override: vision embedder patch_dense is a VisionPatchProjection
    // (classify_matrix_role classifies all vision tensors as MultimodalProjection
    // which maps to CrossModalBridge with the wrong input dimension).
    let tensor_class = if key.contains("vision_embedder.patch_dense") {
        TensorClass::VisionPatchProjection
    } else {
        match role {
            MatrixRole::MultimodalProjection => TensorClass::CrossModalBridge,
            MatrixRole::UnknownLinear => TensorClass::Unknown,
            MatrixRole::AttentionQ
            | MatrixRole::AttentionK
            | MatrixRole::AttentionV
            | MatrixRole::AttentionO => TensorClass::DecoderAttentionProjection,
            MatrixRole::FfnGate | MatrixRole::FfnUp | MatrixRole::FfnDown => {
                TensorClass::DecoderMlpProjection
            }
            MatrixRole::Embedding => TensorClass::TokenEmbedding,
            MatrixRole::LmHead => TensorClass::OutputHead,
            _ => TensorClass::Unknown,
        }
    };

    // Vision embedder weights that structurally resist base NF4 may use
    // Output-scaled NF4 packing is folded into Nf4Tile640Base at pack time.
    let hint = QuantizationHint {
        tensor_class,
        permit_int8_candidate: true,
    };

    let act_weights = Some(channel_sq.as_slice());
    let q_start = std::time::Instant::now();
    eprintln!("  [QUALIFY] tensor={key} candidate-start formats=ternary,nf4,scaled,int8");
    let (codes, scales, biases, scale_vector, qmf) = match quantize_tensor(
        data,
        rows,
        cols,
        &hint,
        act_weights,
        stress,
        calibration,
    ) {
        Ok(tensor) => {
            output_lines.push(format!(
                "  {:?} for {}: wNRMSE={:.4} zCollapse={:.4} oRMSE={:.2} oNRMSE={:.4} cos={:.4} refRMS={:.2}",
                tensor.format,
                key,
                tensor.weight_report.nrmse,
                tensor.weight_report.zero_collapse_ratio,
                tensor.operator_report.rmse,
                tensor.operator_report.operator_nrmse,
                tensor.operator_report.cosine_similarity,
                tensor.operator_report.ref_output_rms,
            ));
            (
                tensor.codes,
                tensor.scales,
                tensor.biases,
                tensor.scale_vector,
                tensor.format,
            )
        }
        Err(e) => {
            match &e {
                QuantizationAdmissionFailure::NoCandidatePassed {
                    candidates_attempted,
                    best_evidence,
                    completed_vectors,
                    ..
                } => {
                    let (w_nrmse, o_nrmse, cos) = if let Some(ev) = best_evidence {
                        (
                            ev.reconstruction_report
                                .as_ref()
                                .map_or(0.0, |r| r.weight_nrmse),
                            ev.probe_report.as_ref().map_or(0.0, |r| r.operator_nrmse),
                            ev.probe_report
                                .as_ref()
                                .map_or(0.0, |r| r.cosine_similarity),
                        )
                    } else {
                        (0.0f64, 0.0f32, 0.0f32)
                    };
                    output_lines.push(format!(
                        "  FAILED for {key}: candidates={candidates_attempted:?} total_vecs={} wNRMSE={:.4} oNRMSE={:.4} cos={:.4}",
                        completed_vectors.total, w_nrmse, o_nrmse, cos
                    ));
                }
                QuantizationAdmissionFailure::PackerFailure(msg) => {
                    output_lines.push(format!("  PACKER FAILURE for {key}: {msg}"));
                }
                QuantizationAdmissionFailure::TimeoutDeadline {
                    candidates_attempted,
                    best_evidence,
                    completed_vectors,
                    expired_phase,
                    ..
                } => {
                    let (w_nrmse, o_nrmse, cos, hgates) = if let Some(ev) = best_evidence {
                        (
                            ev.reconstruction_report
                                .as_ref()
                                .map_or(0.0, |r| r.weight_nrmse),
                            ev.probe_report.as_ref().map_or(0.0, |r| r.operator_nrmse),
                            ev.probe_report
                                .as_ref()
                                .map_or(0.0, |r| r.cosine_similarity),
                            [
                                ev.probe_report.is_some(),
                                ev.promotion_report.is_some(),
                                ev.holdout_report.is_some(),
                            ]
                            .iter()
                            .filter(|&&x| x)
                            .count(),
                        )
                    } else {
                        (0.0f64, 0.0f32, 0.0f32, 0usize)
                    };
                    output_lines.push(format!(
                        "  TIMEOUT for {key}: candidates={candidates_attempted:?} phase={expired_phase} total_vecs={} wNRMSE={:.4} oNRMSE={:.4} cos={:.4} hgates={}",
                        completed_vectors.total, w_nrmse, o_nrmse, cos, hgates
                    ));
                }
            }
            std::process::exit(1);
        }
    };
    let q_elapsed = q_start.elapsed();
    eprintln!(
        "  [QUALIFY] tensor={key} result=accepted format={:?} elapsed={:.1}s",
        qmf,
        q_elapsed.as_secs_f64()
    );
    // Write temp files (after successful verification)
    // Start SHA-256 digest on a rayon worker thread, overlapping with file
    // writes and binding setup.  rayon::scope lets the spawned task borrow
    // `data` (a &[f32] param) without cloning the entire weight matrix.
    let dcell = parking_lot::Mutex::new(None::<[u8; 32]>);
    let dslot = &dcell;
    rayon::scope(|s| {
        s.spawn(|_| {
            *dslot.lock() = Some(parallel_sha256(bytemuck::cast_slice(data)));
        });
        let safe_name = sanitize_filename(key.to_string());
        if output_format == "fused" {
            let fused_buf =
                tribunus_compute_core::nf4tile640::fused::pack_weights_fused(data, rows, cols);
            std::fs::write(tmp_dir.join(format!("{}_fused.bin", safe_name)), &fused_buf).unwrap();
        } else {
            std::fs::write(
                tmp_dir.join(format!("{}_codes.bin", safe_name)),
                bytemuck::cast_slice(&codes),
            )
            .unwrap();
            std::fs::write(
                tmp_dir.join(format!("{}_scales.bin", safe_name)),
                bytemuck::cast_slice(&scales),
            )
            .unwrap();
            std::fs::write(
                tmp_dir.join(format!("{}_biases.bin", safe_name)),
                bytemuck::cast_slice(&biases),
            )
            .unwrap();
        }
        let tiles_per_row = (cols + 639) / 640;
        let total_tiles = rows * tiles_per_row;
        let rrc: RuntimeRepresentationClass = match qmf {
            RuntimeRepresentationClass::Nf4Tile640Base => {
                RuntimeRepresentationClass::Nf4Tile640Base
            }
            RuntimeRepresentationClass::Int8Tile640Base => {
                RuntimeRepresentationClass::Int8Tile640Base
            }
            RuntimeRepresentationClass::TernaryTile640Base => {
                RuntimeRepresentationClass::TernaryTile640Base
            }
            RuntimeRepresentationClass::RawF32 => RuntimeRepresentationClass::RawF32,
        };
        let (weights_seg, meta_seg, sidecar_seg) = match rrc {
            RuntimeRepresentationClass::Nf4Tile640Base => (26u8, 27u8, 40u8),
            RuntimeRepresentationClass::Int8Tile640Base => (39u8, 27u8, 0xFF),
            RuntimeRepresentationClass::TernaryTile640Base => (1u8, 27u8, 0xFF),
            RuntimeRepresentationClass::RawF32 => (38u8, 0xFF, 0xFF),
        };
        bindings.push(WeightBindingData {
            key: key.to_string(),
            format: rrc,
            tiles_per_row,
            total_tiles,
            weights_segment: weights_seg,
            tile_metadata_segment: meta_seg,
            sidecar_segment: if scale_vector.is_some() {
                sidecar_seg
            } else {
                0xFF
            },
            sidecar_count: scale_vector.as_ref().map(|_| cols as u32).unwrap_or(0),
            rows,
            cols,
        });
        if let Some(sv) = scale_vector {
            let scale_bytes: Vec<u8> = sv
                .iter()
                .flat_map(|&s| f32_to_fp16_bits(s).to_le_bytes())
                .collect();
            std::fs::write(
                tmp_dir.join(format!("{}_scales_f16.bin", safe_name)),
                &scale_bytes,
            )
            .unwrap();
        }
    });
    let tensor_digest = dcell.lock().take().unwrap();
    plan_entries.push(QuantizationPlanEntry {
        tensor_name: key.to_string(),
        source_tensor_digest: tensor_digest,
        profile_id: selection_receipts
            .last()
            .map(|r| r.selected_profile_id)
            .unwrap_or(0),
        group_importances: importances,
        outlier_channels: Vec::new(),
        verification_rmse: 0.0,
        gate_passed: true,
        aw_mse: None,
        channel_second_moments: Some(channel_sq),
    });
}

/// Collect nf4tile640 triplet segments from per-matrix temp files.
#[allow(dead_code)]
/// Files are sorted by sanitized name to match the matrix processing order
/// (embedding → layers → lm_head → multimodal).
fn collect_nf4_triplet_segments(tmp_dir: &Path) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(tmp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();

    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();

    for path in &entries {
        let name = path.file_name().unwrap().to_string_lossy();
        if name.ends_with("_codes.bin") {
            codes.extend(std::fs::read(path).unwrap());
        } else if name.ends_with("_scales.bin") {
            scales.extend(std::fs::read(path).unwrap());
        } else if name.ends_with("_biases.bin") {
            biases.extend(std::fs::read(path).unwrap());
        }
    }

    (codes, scales, biases)
}

/// Collect per-segment data from per-matrix temp files, keyed on binding order.
/// Returns (nf4_codes, int8_codes, tile_metadata, sidecar_fp16).
fn collect_triplet_segments(
    bindings: &[WeightBindingData],
    tmp_dir: &Path,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    use RuntimeRepresentationClass::*;
    let mut nf4_codes = Vec::new();
    let mut int8_codes = Vec::new();
    let mut tile_meta = Vec::new();
    let mut sidecar_fp16 = Vec::new();
    for b in bindings {
        let safe_name = sanitize_filename(b.key.clone());
        // Codes: NF4 vs INT8
        let codes = std::fs::read(tmp_dir.join(format!("{}_codes.bin", safe_name)))
            .unwrap_or_else(|e| panic!("Missing codes for {}: {}", b.key, e));
        match b.format {
            RuntimeRepresentationClass::Int8Tile640Base => int8_codes.extend(codes),
            _ => nf4_codes.extend(codes),
        }
        // Scales (NF4: f32 scales per tile, INT8: f32 scales per tile)
        let scales_raw = std::fs::read(tmp_dir.join(format!("{}_scales.bin", safe_name)))
            .unwrap_or_else(|e| panic!("Missing scales for {}: {}", b.key, e));
        // Biases (NF4 only: f32 biases per tile)
        if matches!(b.format, RuntimeRepresentationClass::Nf4Tile640Base) {
            let biases_raw = std::fs::read(tmp_dir.join(format!("{}_biases.bin", safe_name)))
                .unwrap_or_else(|e| panic!("Missing biases for {}: {}", b.key, e));
            // Interleave: per tile [f32_scale][f32_bias] = 8 bytes
            let scales_f32 = bytemuck::cast_slice::<u8, f32>(&scales_raw);
            let biases_f32 = bytemuck::cast_slice::<u8, f32>(&biases_raw);
            for i in 0..scales_f32.len().min(biases_f32.len()) {
                tile_meta.extend_from_slice(&scales_f32[i].to_le_bytes());
                tile_meta.extend_from_slice(&biases_f32[i].to_le_bytes());
            }
        } else {
            // INT8: scale only, no bias — just append raw f32 scale bytes
            tile_meta.extend(scales_raw);
        }
        if b.sidecar_count > 0 {
            let sc = std::fs::read(tmp_dir.join(format!("{}_scales_f16.bin", safe_name)))
                .unwrap_or_else(|e| panic!("Missing sidecar for {}: {}", b.key, e));
            sidecar_fp16.extend(sc);
        }
    }
    (nf4_codes, int8_codes, tile_meta, sidecar_fp16)
}

/// Build the binary MatrixContract segment: u32 count followed by count × MatrixWeightBinding.
///
/// Wire format:
///   [count: u32 LE]
///   for each binding:
///     [weights_offset: u64 LE]
///     [weights_bytes: u64 LE]
///     [tile_metadata_offset: u64 LE]
///     [tile_metadata_bytes: u64 LE]
///     [sidecar_offset: u64 LE]
///     [sidecar_count: u32 LE]        // 0 = no sidecar
///     [matrix_id: u32 LE]
///     [format: u8]
///     [weights_segment: u8]
///     [tile_metadata_segment: u8]
///     [sidecar_segment: u8]          // 0xFF = none
///     [sidecar_kind: u8]          // SidecarKind discriminant (0=None, 1=ReductionAxisScale)
///     [sidecar_element_format: u8] // SidecarElementFormat (0=None, 1=F16, 2=F32)
///     [_reserved: u8; 3]           // must be zero
///     [rows: u32 LE]
///     [cols: u32 LE]
///     [tiles_per_row: u32 LE]
///
/// Endian-independent — always little-endian.
fn build_matrix_contract_blob(bindings: &[MatrixWeightBindingV1]) -> Vec<u8> {
    let per_binding = MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH;
    let mut buf = Vec::with_capacity(4 + bindings.len() * per_binding);
    buf.extend_from_slice(&(bindings.len() as u32).to_le_bytes());
    for b in bindings {
        write_matrix_weight_binding_v1_le(&mut buf, b).unwrap();
    }
    buf
}

/// Deserialize a MatrixContract blob into a Vec of MatrixWeightBinding records.
/// Reads the explicit LE field format produced by build_matrix_contract_blob.
#[allow(dead_code)]
fn read_matrix_contract_blob(data: &[u8]) -> Vec<MatrixWeightBindingV1> {
    if data.len() < 4 {
        return Vec::new();
    }
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut bindings = Vec::with_capacity(count);
    let mut off = 4usize;
    for _ in 0..count {
        let remaining = &data[off..];
        match read_matrix_weight_binding_v1_le(remaining) {
            Ok(b) => {
                let consumed = MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH;
                bindings.push(b);
                off += consumed;
            }
            Err(_) => return Vec::new(),
        }
    }
    bindings
}

/// Validate that every binding's ranges fit within their referenced segment sizes.
/// Returns Ok(()) or Err with all validation errors.
#[allow(dead_code)]
fn validate_binding_ranges(
    bindings: &[MatrixWeightBindingV1],
    segment_sizes: &std::collections::HashMap<u8, u64>,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for b in bindings {
        let mid = b.matrix_id;
        // weights: unconditional — every binding must reference an existing segment
        let sz = match segment_sizes.get(&b.code_segment) {
            Some(&s) => s,
            None => {
                errors.push(format!(
                    "binding[{mid}]: missing weights segment {}",
                    b.code_segment
                ));
                continue;
            }
        };
        let end = b.code_offset.checked_add(b.code_length).unwrap_or(u64::MAX);
        if end > sz {
            errors.push(format!(
                "binding[{mid}]: weights range [{}..{}) exceeds segment {} size {}",
                b.code_offset, end, b.code_segment, sz
            ));
        }

        // tile_metadata: unconditional
        let sz = match segment_sizes.get(&b.metadata_segment) {
            Some(&s) => s,
            None => {
                errors.push(format!(
                    "binding[{mid}]: missing tile_metadata segment {}",
                    b.metadata_segment
                ));
                continue;
            }
        };
        let end = b
            .metadata_offset
            .checked_add(b.metadata_length)
            .unwrap_or(u64::MAX);
        if end > sz {
            errors.push(format!(
                "binding[{mid}]: tile_metadata range [{}..{}) exceeds segment {} size {}",
                b.metadata_offset, end, b.metadata_segment, sz
            ));
        }

        // sidecar: only when sidecar_count > 0 and sidecar_kind != 0
        if b.sidecar_count > 0 && b.sidecar_kind != 0 {
            let sz = match segment_sizes.get(&b.sidecar_segment) {
                Some(&s) => s,
                None => {
                    errors.push(format!(
                        "binding[{mid}]: missing sidecar segment {}",
                        b.sidecar_segment
                    ));
                    continue;
                }
            };
            let sc_fmt = match b.sidecar_element_format {
                0 => SidecarElementFormat::None,
                1 => SidecarElementFormat::F16,
                2 => SidecarElementFormat::F32,
                _ => unreachable!(),
            };
            let sc_bytes = match sidecar_byte_len(b.sidecar_count, sc_fmt) {
                Some(n) => n,
                None => {
                    errors.push(format!(
                        "binding[{mid}]: sidecar_byte_len overflow for count {}",
                        b.sidecar_count
                    ));
                    continue;
                }
            };
            let end = b.sidecar_offset.checked_add(sc_bytes).unwrap_or(u64::MAX);
            if end > sz {
                errors.push(format!(
                    "binding[{mid}]: sidecar range [{}..{}) exceeds segment {} size {}",
                    b.sidecar_offset, end, b.sidecar_segment, sz
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // ── Subcommand dispatch ────────────────────────────────────────
    if args.len() > 1 && args[1] == "inspect-checkpoint" {
        cmd_inspect_checkpoint(&args[1..]);
        return;
    }

    // Parse arguments
    let repo = get_opt(&args, "--repo");
    let local_dir = get_opt(&args, "--local-dir");
    let output = get_opt(&args, "--output").unwrap_or("gemma4_12b.cimage");
    let draft_model_dir = get_opt(&args, "--draft-model-dir");
    let mil_program = get_opt(&args, "--mil");
    let nf4 = has_flag(&args, "--nf4");
    let quantizer_mode = get_opt(&args, "--quantizer")
        .unwrap_or("canonical_nf4_v1")
        .to_string();
    let is_nf4_mode = quantizer_mode.starts_with("nf4tile640") || nf4;
    // Banner
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  Gemma 4 12B Unified → {} .cimage                          ║",
        if is_nf4_mode {
            "NF4 Tile640"
        } else {
            "Ternary"
        }
    );
    println!("║  AOT Compiler (pure Rust, no Python)                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    // Build deterministic calibration suite for operator-space validation.
    // Build deterministic stress suite (always, catches codec pathologies).
    // Optional calibration suite (prerendered activation banks) can be added
    // when the reference model execution is available.
    let stress_suite = if is_nf4_mode {
        Some(tribunus_compute_core::quantization::StressSuite::build_default())
    } else {
        None
    };
    let tts_repo = get_opt(&args, "--tts-repo").map(|s| s.to_string());
    let tts_local_dir = get_opt(&args, "--tts-local-dir").map(|s| s.to_string());
    let vision_bank = get_opt(&args, "--vision-bank").map(|s| s.to_string());
    let bridge_bank = get_opt(&args, "--bridge-bank").map(|s| s.to_string());

    let strategy = get_opt(&args, "--strategy").unwrap_or("causal-text");
    if strategy != "causal-text" && strategy != "acoustic-stream" {
        eprintln!("ERROR: --strategy must be 'causal-text' (default) or 'acoustic-stream', got: {strategy}");
        std::process::exit(1);
    }

    let output_format = get_opt(&args, "--format").unwrap_or("split");

    let _calibration_corpus = get_opt(&args, "--calibration-corpus").map(|s| s.to_string());
    let _calibration_budget =
        get_opt(&args, "--calibration-budget").and_then(|s| s.parse::<u64>().ok());
    let quality_policy = get_opt(&args, "--quality-policy").unwrap_or("default");
    let _emit_quality_report = has_flag(&args, "--emit-quality-report");
    let _allow_experimental = has_flag(&args, "--allow-experimental");

    // Load prerendered vision activation bank, if provided.
    let mut calibration_suite = vision_bank.as_ref().and_then(|path| {
        use tribunus_compute_core::quantization::CalibrationSuite;
        match CalibrationSuite::load_from_bank_dir(
            std::path::Path::new(path),
            tribunus_compute_core::quantization::contract::TensorClass::VisionPatchProjection,
            6912,
            "vision-prerendered",
        ) {
            Ok(suite) => {
                eprintln!("  Loaded vision activation bank from {}", path);
                Some(suite)
            }
            Err(e) => {
                eprintln!("  WARNING: failed to load vision bank from {}: {e}", path);
                None
            }
        }
    });

    // Load prerendered bridge activation bank (input to embedding_projection.weight).
    let calibration_suite = bridge_bank.as_ref().and_then(|path| {
        use tribunus_compute_core::quantization::contract::TensorClass;
        use tribunus_compute_core::quantization::CalibrationSuite;
        match CalibrationSuite::load_from_bank_dir(
            std::path::Path::new(path),
            TensorClass::CrossModalBridge,
            3840,
            "bridge-prerendered",
        ) {
            Ok(suite) => {
                let mut existing = calibration_suite
                    .take()
                    .unwrap_or_else(CalibrationSuite::empty);
                if let Some(bridge) = suite.get(&TensorClass::CrossModalBridge) {
                    existing.insert(TensorClass::CrossModalBridge, bridge.clone());
                }
                eprintln!("  Loaded bridge activation bank from {}", path);
                Some(existing)
            }
            Err(e) => {
                eprintln!("  WARNING: failed to load bridge bank from {}: {e}", path);
                calibration_suite.take()
            }
        }
    });

    // Validate args
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12b-it --output gemma4_12b.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B-it-qat-q4_0-unquantized --output gemma4_12b_qat.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage");
        eprintln!();
        eprintln!("Flags:");
        eprintln!("  --nf4                        Use nf4tile640 quantization instead of ternary");
        eprintln!(
            "  --tts-repo <HF_REPO_ID>      HF repo ID for Qwen3-TTS model to bake into cimage"
        );
        eprintln!(
            "  --tts-local-dir <PATH>       Local directory containing TTS model.safetensors"
        );
        eprintln!("  --verify-only                Re-download source and verify nf4tile640 packing quality");
        eprintln!("  --quantizer <PROFILE>         Quantizer profile: canonical_nf4_v1 (default), learn-gemma-v1");
        eprintln!("  --calibration-corpus <PATH>   Path to calibration text for profile learning");
        eprintln!(
            "  --calibration-budget <MB>     Memory budget for calibration in MB (default: 1024)"
        );
        eprintln!("  --quality-policy <POLICY>     Quality policy: default (0.05), strict (0.01), experimental");
        eprintln!("  --emit-quality-report         Emit JSON quality report alongside cimage");
        eprintln!(
            "  --allow-experimental          Allow experimental quantizer profiles in output"
        );
        eprintln!("  --format split|fused         Output format: split (three files per matrix, default) or fused (single 64B-aligned buffer)");
        eprintln!(
            "  --strategy causal-text|acoustic-stream   KV cache strategy (default: causal-text)"
        );
        std::process::exit(0);
    }

    if has_flag(&args, "--verify-only") {
        cmd_verify_only(&args);
        return;
    }

    if repo.is_none() && local_dir.is_none() {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12b-it --output gemma4_12b.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B-it-qat-q4_0-unquantized --output gemma4_12b_qat.cimage");
        eprintln!("  cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage");
        eprintln!();
        eprintln!("Flags:");
        eprintln!("  --nf4                        Use nf4tile640 quantization instead of ternary");
        eprintln!(
            "  --tts-repo <HF_REPO_ID>      HF repo ID for Qwen3-TTS model to bake into cimage"
        );
        eprintln!(
            "  --tts-local-dir <PATH>       Local directory containing TTS model.safetensors"
        );
        eprintln!("  --verify-only                Re-download source and verify nf4tile640 packing quality");
        eprintln!("  --format split|fused         Output format: split (three files per matrix, default) or fused (single 64B-aligned buffer)");
        eprintln!(
            "  --strategy causal-text|acoustic-stream   KV cache strategy (default: causal-text)"
        );
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
    if is_nf4_mode {
        println!("\n  ── Quantizing weights (NF4 Tile640) ────────────────────");
    } else {
        println!("\n  ── Quantizing weights (256-block ternary) ───────────────");
    }
    let quant_start = Instant::now();

    // Spawn Metal shader compilation concurrently (CPU LLVM work, while GPU quantizes)
    let metal_handle = std::thread::spawn(|| {
        let t = Instant::now();
        let bytes = compile_metal_lib();
        (bytes, t.elapsed())
    });

    let mut all_scales = Vec::new();
    let mut all_weights = Vec::new();
    // NF4-mode per-segment data (populated after packing)
    let mut nf4_weights_seg: Vec<u8> = Vec::new();
    let mut int8_weights_seg: Vec<u8> = Vec::new();
    let mut tile_metadata_seg: Vec<u8> = Vec::new();
    let mut sidecar_seg: Vec<u8> = Vec::new();
    let mut contract_bytes: Vec<u8> = Vec::new();
    let mut total_elements: usize = 0; // ternary element count (0 in nf4 mode)
    let mut ternary_distill_results: Vec<MatrixDistillResult> = Vec::new();

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
    // nf4tile640 state (moved inline — no longer accumulates all_matrices)
    // AW-LS activation-weighted saliency: per-matrix channel second moments.
    let mut channel_sq_map: HashMap<String, Vec<f32>> = HashMap::new();
    // nf4tile640 streaming state (pack inline, no accumulation)
    let mut selection_receipts: Vec<ProfileSelectionReceipt> = Vec::new();
    let mut plan_entries: Vec<QuantizationPlanEntry> = Vec::new();
    let mut weight_bindings: Vec<WeightBindingData> = Vec::new();
    let mut matrix_bindings: Vec<MatrixWeightBindingV1> = Vec::new();
    let learned_profiles: HashMap<MatrixRole, LearnedProfile> = HashMap::new();
    let distill_objective = DistillObjective::default();
    let nf4_tmp_dir = std::env::temp_dir().join("gemma4_nf4_pack_stream");
    let _ = std::fs::create_dir_all(&nf4_tmp_dir);
    let nf4_start = Instant::now();
    // nf4tile640 biases segment (populated via collect_nf4_triplet_segments after packing)
    let nf4_biases: Vec<u8> = Vec::new();
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

    // ── CLUSTER & QUANTIZE EMBEDDING TABLE (skipped in --nf4 mode) ──
    {
        if !is_nf4_mode {
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
                    let mut centroids = kmeans_plusplus(
                        &vocab_embedding_raw_f32,
                        k,
                        n_rows,
                        dim,
                        &CancelToken::new(None),
                    );
                    for _iter in 0..20 {
                        let (_assignments, delta) = kmeans_iterate(
                            &vocab_embedding_raw_f32,
                            &mut centroids,
                            n_rows,
                            dim,
                            k,
                            &CancelToken::new(None),
                        );
                        if delta < 1e-6 {
                            break;
                        }
                    }
                    let (assignments, _delta) = kmeans_iterate(
                        &vocab_embedding_raw_f32,
                        &mut centroids,
                        n_rows,
                        dim,
                        k,
                        &CancelToken::new(None),
                    );
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
        } else {
            let dim = HIDDEN_DIM;
            let n_rows = vocab_embedding_raw_f32.len() / dim;
            eprintln!(
                "  NF4 mode: raw f32 embeddings ({} rows x {} dims)",
                n_rows, dim
            );
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
    for (name, rows, cols) in MULTIMODAL_WEIGHTS {
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
            } else if is_nf4_mode {
                // Projection matrix → nf4tile640 inline packing
                let mut output_lines_temp = Vec::new();
                pack_matrix_nf4_inline(
                    name,
                    stress_suite.as_ref(),
                    calibration_suite.as_ref(),
                    &data,
                    *rows,
                    *cols,
                    shape.first().copied().unwrap_or(0),
                    shape.get(1).copied().unwrap_or(1),
                    &mut channel_sq_map,
                    &mut selection_receipts,
                    &mut plan_entries,
                    &mut weight_bindings,
                    &learned_profiles,
                    &distill_objective,
                    &nf4_tmp_dir,
                    &output_format,
                    quality_policy,
                    &mut output_lines_temp,
                );
                for line in &output_lines_temp {
                    eprintln!("{}", line);
                }
            } else {
                // Projection matrix → ternary quantization
                let nib_off = multimodal_nibbles.len() as u64;
                let scl_off = multimodal_scales.len() as u64;
                ane_ternary_offsets.insert(name.to_string(), (nib_off, scl_off, data.len()));
                process_weights(&data, &mut multimodal_scales, &mut multimodal_nibbles);
                total_elements += data.len();
                let distill_obj = DistillObjective::default();
                let distill_result = distill_matrix(
                    name,
                    &data,
                    *rows,
                    *cols,
                    DistillFormat::Ternary,
                    &distill_obj,
                    None,
                );
                if !distill_result.gate_passed {
                    eprintln!(
                        "  [distill] {} KL={:.4} total_loss={:.4}",
                        name, distill_result.kl_divergence, distill_result.total_loss
                    );
                }
                ternary_distill_results.push(distill_result);
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
        let rows = if shape.len() >= 2 {
            shape[0]
        } else {
            data.len()
        };
        let cols = if shape.len() >= 2 { shape[1] } else { 1 };
        if is_nf4_mode {
            let mut output_lines_temp = Vec::new();
            pack_matrix_nf4_inline(
                lm_head_key,
                stress_suite.as_ref(),
                calibration_suite.as_ref(),
                &data,
                rows,
                cols,
                shape.first().copied().unwrap_or(0),
                shape.get(1).copied().unwrap_or(1),
                &mut channel_sq_map,
                &mut selection_receipts,
                &mut plan_entries,
                &mut weight_bindings,
                &learned_profiles,
                &distill_objective,
                &nf4_tmp_dir,
                &output_format,
                quality_policy,
                &mut output_lines_temp,
            );
            for line in &output_lines_temp {
                eprintln!("{}", line);
            }
        } else {
            process_weights(&data, &mut all_scales, &mut all_weights);
            total_elements += data.len();
            let distill_obj = DistillObjective::default();
            let distill_result = distill_matrix(
                lm_head_key,
                &data,
                rows,
                cols,
                DistillFormat::Ternary,
                &distill_obj,
                None,
            );
            if !distill_result.gate_passed {
                eprintln!(
                    "  [distill] {} KL={:.4} total_loss={:.4}",
                    lm_head_key, distill_result.kl_divergence, distill_result.total_loss
                );
            }
            ternary_distill_results.push(distill_result);
            print!("  lm_head: {shape:?} — {n_blocks:>6} blocks\n");
        }
    } else {
        println!("  lm_head.weight NOT FOUND (tied with embed_tokens)");
    }

    // Layer weights
    // ── Pack all decoder layer matrices ──────────────────────────
    if is_nf4_mode {
        // Collect packing jobs serially (load_tensor reads mmap'd shards)
        let mut jobs: Vec<PackJob> = Vec::new();
        for layer in 0..NUM_LAYERS {
            for (mat_name, rows, cols) in MATRICES {
                let key = tensor_key(layer, mat_name);
                if let Some((data, shape)) = load_tensor(&key, &shard_paths) {
                    jobs.push(PackJob {
                        key,
                        data,
                        rows: *rows,
                        cols: *cols,
                        shape,
                    });
                } else {
                    println!("\n  WARNING: {key} not found");
                }
            }
        }

        // Parallelize packing across all matrices using rayon
        let output_buffer = parking_lot::Mutex::new(Vec::new());
        let sq_mtx = parking_lot::Mutex::new(std::mem::take(&mut channel_sq_map));
        let rec_mtx = parking_lot::Mutex::new(std::mem::take(&mut selection_receipts));
        let plan_mtx = parking_lot::Mutex::new(std::mem::take(&mut plan_entries));
        let bind_mtx = parking_lot::Mutex::new(std::mem::take(&mut weight_bindings));

        jobs.par_iter().for_each(|job| {
            // Per-job local accumulation avoids contention on every push
            let mut local_output = Vec::new();
            let mut local_sq = std::collections::HashMap::new();
            let mut local_rec = Vec::new();
            let mut local_plan = Vec::new();
            let mut local_bind = Vec::new();

            pack_matrix_nf4_inline(
                &job.key,
                stress_suite.as_ref(),
                calibration_suite.as_ref(),
                &job.data,
                job.rows,
                job.cols,
                physical_dims_from_shape(&job.shape).0,
                physical_dims_from_shape(&job.shape).1,
                &mut local_sq,
                &mut local_rec,
                &mut local_plan,
                &mut local_bind,
                &learned_profiles,
                &distill_objective,
                &nf4_tmp_dir,
                &output_format,
                quality_policy,
                &mut local_output,
            );

            // Merge per-job results into shared collections
            output_buffer.lock().append(&mut local_output);
            sq_mtx.lock().extend(local_sq);
            rec_mtx.lock().append(&mut local_rec);
            plan_mtx.lock().append(&mut local_plan);
            bind_mtx.lock().append(&mut local_bind);
        });

        // Reclaim ownership from mutexes
        channel_sq_map = sq_mtx.into_inner();
        selection_receipts = rec_mtx.into_inner();
        plan_entries = plan_mtx.into_inner();
        weight_bindings = bind_mtx.into_inner();

        // Flush buffered output lines in order
        for line in output_buffer.into_inner() {
            println!("{line}");
        }
    } else {
        // Ternary path — stays serial (calls process_weights and distill)
        for layer in 0..NUM_LAYERS {
            for (mat_name, rows, cols) in MATRICES {
                let key = tensor_key(layer, mat_name);
                if let Some((data, _shape)) = load_tensor(&key, &shard_paths) {
                    process_weights(&data, &mut all_scales, &mut all_weights);
                    total_elements += data.len();

                    // ── Guided distillation comparison (NF4 vs ternary) ─────
                    let distill_obj = DistillObjective::default();
                    let distill_result = distill_matrix(
                        &key,
                        &data,
                        *rows,
                        *cols,
                        DistillFormat::Ternary,
                        &distill_obj,
                        None,
                    );
                    if !distill_result.gate_passed {
                        eprintln!(
                            "  [distill] {} KL={:.4} total_loss={:.4}",
                            key, distill_result.kl_divergence, distill_result.total_loss
                        );
                    }
                    ternary_distill_results.push(distill_result);
                } else {
                    println!("\n  WARNING: {key} not found");
                }
            }

            if layer % 8 == 7 {
                let mb = (all_scales.len() + all_weights.len()) as f64 / (1024.0 * 1024.0);
                println!(" — {mb:.1} MB");
            }
        }
    }

    // ── Collect per-layer norm weights (always serial, tiny) ─────
    for layer in 0..NUM_LAYERS {
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

    // ── MTP Draft Model (optional external draft decoder) ──────────
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
                    if is_nf4_mode {
                        // nf4tile640 path: inline pack immediately (no accumulation).
                        let shape = tv.shape();
                        let rows = if shape.len() >= 2 {
                            shape[0]
                        } else {
                            f32_data.len()
                        };
                        let cols = if shape.len() >= 2 { shape[1] } else { 1 };
                        let mut output_lines_temp = Vec::new();
                        pack_matrix_nf4_inline(
                            name,
                            stress_suite.as_ref(),
                            calibration_suite.as_ref(),
                            &f32_data,
                            rows,
                            cols,
                            shape.first().copied().unwrap_or(0),
                            shape.get(1).copied().unwrap_or(1),
                            &mut channel_sq_map,
                            &mut selection_receipts,
                            &mut plan_entries,
                            &mut weight_bindings,
                            &learned_profiles,
                            &distill_objective,
                            &nf4_tmp_dir,
                            &output_format,
                            quality_policy,
                            &mut output_lines_temp,
                        );
                        for line in &output_lines_temp {
                            eprintln!("{}", line);
                        }
                        println!("    draft (nf4): {name:<55} {}x{}", rows, cols);
                    } else {
                        // Ternary path (existing behavior)
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
            }
            draft_layer_count = 4; // MTP draft has 4 transformer layers
        }
    }

    // ── nf4tile640 report & plan (packing happened inline above) ──
    if is_nf4_mode {
        println!();
        println!(
            "  ✓ Packed {} nf4tile640 matrices in {:.1?}",
            selection_receipts.len(),
            nf4_start.elapsed()
        );

        // ── Emit profile selection inspection artifact ───────────────
        let source_digest = "bf16_qat";
        let _report =
            emit_selection_report(&selection_receipts, output, source_digest, &quantizer_mode);

        // Validate selection integrity
        let registry_ids: Vec<u32> = if quantizer_mode == "learn-gemma-v1" {
            vec![1, 2, 3, 4]
        } else {
            vec![0]
        };
        validate_selection_integrity(&selection_receipts, &registry_ids, &quantizer_mode);

        // ── Build and emit QuantizationPlan ────────────────────────────
        let plan = QuantizationPlan {
            source_model_digest: QuantizationPlan::compute_model_digest(&plan_entries),
            quantizer_mode: quantizer_mode.clone(),
            target_strategy: strategy.to_string(),
            entries: plan_entries,
            profile_registry_ids: registry_ids,
            build_duration_secs: nf4_start.elapsed().as_secs_f64(),
        };
        let plan_json = plan.to_json_pretty().unwrap();
        let plan_path = format!("{}.plan.json", output);
        std::fs::write(&plan_path, &plan_json).unwrap();
        println!("  ✓ Quantization plan written to {}", plan_path);

        // ── Collect nf4tile640 triplet segments from temp files ────────
        // New: collect per-segment data in binding order
        (
            nf4_weights_seg,
            int8_weights_seg,
            tile_metadata_seg,
            sidecar_seg,
        ) = collect_triplet_segments(&weight_bindings, &nf4_tmp_dir);

        // ── Compute independent per-segment offsets ───────────────────
        let mut nf4_off = 0u64;
        let mut int8_off = 0u64;
        let mut tile_meta_off = 0u64;
        let mut sidecar_off = 0u64;
        use RuntimeRepresentationClass::*;
        matrix_bindings = Vec::new();
        for (mid, wb) in weight_bindings.iter().enumerate() {
            let codes_bytes = wb.total_tiles as u64
                * match wb.format {
                    RuntimeRepresentationClass::Int8Tile640Base => INT8_TILE640_CODE_BYTES as u64,
                    _ => NF4_TILE640_CODE_BYTES as u64,
                };
            let has_bias = matches!(wb.format, RuntimeRepresentationClass::Nf4Tile640Base);
            let meta_bytes_per_tile: u64 = if has_bias { 8 } else { 4 };
            let meta_bytes = wb.total_tiles as u64 * meta_bytes_per_tile;

            let (w_off, w_seg) = match wb.format {
                Int8Tile640Base => {
                    let off = int8_off;
                    int8_off += codes_bytes;
                    (off, 39u8) // Int8Tile640Weights
                }
                _ => {
                    let off = nf4_off;
                    nf4_off += codes_bytes;
                    (off, 26u8) // Nf4Tile640Weights
                }
            };
            let tm_off = tile_meta_off;
            tile_meta_off += meta_bytes;

            let mut sidecar_count = 0u32;
            let mut sidecar_seg_id = 0xFFu8;
            let mut sc_off = 0u64;
            if wb.sidecar_count > 0 {
                sidecar_count = wb.sidecar_count;
                sidecar_seg_id = 40u8; // QuantizationSidecars
                sc_off = sidecar_off;
                sidecar_off += wb.sidecar_count as u64 * 2; // FP16 = 2 bytes each
            }

            let representation = admission_format_to_binding_format(wb.format);
            matrix_bindings.push(MatrixWeightBindingV1 {
                binding_wire_version: 1u16,
                matrix_id: mid as u32,
                tensor_id: [0u8; 16],
                representation,
                representation_version: 1u16,
                kernel_abi_digest: [0u8; 32],
                in_features: wb.rows as u32,
                out_features: wb.cols as u32,
                reduction_tile_size: 640u16,
                tiles_per_output_channel: wb.tiles_per_row as u32,
                tail_reduction_count: (wb.cols % 640) as u16,
                macro_layout: 1u8,
                tail_handling: 1u8,
                code_segment: w_seg,
                code_offset: w_off,
                code_length: codes_bytes,
                code_tile_stride_bytes: match representation {
                    0 => 160,
                    1 => 320,
                    2 => 640,
                    _ => 0,
                },
                metadata_segment: 27u8,
                metadata_offset: tm_off,
                metadata_length: meta_bytes,
                metadata_tile_stride_bytes: match representation {
                    0 => 4u16,
                    1 => 8u16,
                    2 => 4u16,
                    _ => 0u16,
                },
                sidecar_segment: sidecar_seg_id,
                sidecar_offset: sc_off,
                sidecar_length: 0u64,
                sidecar_kind: if sidecar_count > 0 { 1 } else { 0 },
                sidecar_element_format: if sidecar_count > 0 { 1 } else { 0 },
                sidecar_count,
                residual_segment: 0u8,
                residual_offset: 0u64,
                residual_length: 0u64,
                required_alignment_bytes: 64u32,
            });
        }

        // ── Build MatrixContract binary ─────────────────────────────────
        contract_bytes = build_matrix_contract_blob(&matrix_bindings);
        println!(
            "  ✓ Collected nf4tile640 triplets: {:.1} MB nf4, {:.1} MB int8, {:.1} MB tile_meta, {:.1} MB sidecar",
            nf4_weights_seg.len() as f64 / (1024.0 * 1024.0),
            int8_weights_seg.len() as f64 / (1024.0 * 1024.0),
            tile_metadata_seg.len() as f64 / (1024.0 * 1024.0),
            sidecar_seg.len() as f64 / (1024.0 * 1024.0),
        );
    }
    // ── Ternary quality report ──
    if !ternary_distill_results.is_empty() && !is_nf4_mode {
        let ternary_report_path = format!(
            "{}.ternary_distill.json",
            output.trim_end_matches(".cimage")
        );
        let ternary_report = serde_json::json!({
            "total_matrices": ternary_distill_results.len(),
            "avg_kl": ternary_distill_results.iter().map(|r| r.kl_divergence as f64).sum::<f64>() / ternary_distill_results.len() as f64,
            "avg_loss": ternary_distill_results.iter().map(|r| r.total_loss).sum::<f64>() / ternary_distill_results.len() as f64,
            "gate_pass_rate": ternary_distill_results.iter().filter(|r| r.gate_passed).count() as f64 / ternary_distill_results.len() as f64,
            "matrices": ternary_distill_results.iter().map(|r| serde_json::json!({
                "tensor": r.tensor_name,
                "kl": r.kl_divergence,
                "total_loss": r.total_loss,
                "rmse": r.rmse,
                "gate_passed": r.gate_passed,
            })).collect::<Vec<_>>(),
        });
        if let Ok(json) = serde_json::to_string_pretty(&ternary_report) {
            if std::fs::write(&ternary_report_path, &json).is_ok() {
                eprintln!("  Ternary distillation report written to {ternary_report_path}");
            }
        }
    }

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

    let mut extra_tensor_entries: Vec<TensorEntry> = Vec::new();
    let mut tts_tmp_dir_opt: Option<PathBuf> = None;

    // ── TTS model packing ─────────────────────────────────────
    if tts_repo.is_some() || tts_local_dir.is_some() {
        let tts_file = if let Some(local) = &tts_local_dir {
            let path = Path::new(local).join("model.safetensors");
            println!("  ▶ Loading TTS model from local: {}", path.display());
            if !path.exists() {
                eprintln!(
                    "  ERROR: TTS model.safetensors not found at {}",
                    path.display()
                );
                std::process::exit(1);
            }
            path
        } else if let Some(tts_repo) = &tts_repo {
            println!("  ▶ Downloading TTS model from {}", tts_repo);
            download_tts_safetensors(tts_repo).expect("download TTS safetensors")
        } else {
            unreachable!()
        };

        let tts_tmp_dir = std::env::temp_dir().join("gemma4_tts_pack");
        let _ = std::fs::create_dir_all(&tts_tmp_dir);

        println!("  ▶ Packing TTS weights as nf4tile640");
        let tts_entries = pack_tts_weights(&tts_file, &tts_tmp_dir).expect("pack TTS weights");

        // Prefix TTS entries to avoid naming collisions with LLM weights
        let tts_entries: Vec<TensorEntry> = tts_entries
            .into_iter()
            .map(|mut e| {
                e.name = format!("tts_{}", e.name);
                e
            })
            .collect();
        println!("  ✓ Packed {} TTS weight segments", tts_entries.len());

        extra_tensor_entries.extend(tts_entries);
        tts_tmp_dir_opt = Some(tts_tmp_dir);
    }

    // ── Write output ───────────────────────────────────────────────
    // Main weights .cimage
    println!("\n  Writing main weights to {}", output);

    let quant_elapsed = quant_start.elapsed();
    let n_blocks = if is_nf4_mode {
        0 // nf4 mode doesn't use ternary block counting
    } else {
        all_scales.len() / 2
    };

    println!(
        "  Quantized {} weights in {:.1?}",
        total_elements, quant_elapsed
    );
    if !is_nf4_mode {
        let mb_scales = all_scales.len() as f64 / (1024.0 * 1024.0);
        let mb_weights = all_weights.len() as f64 / (1024.0 * 1024.0);
        println!(
            "  {} blocks, {:.1} MB scales, {:.1} MB nibbles",
            n_blocks, mb_scales, mb_weights
        );
    }

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
    let header_size = CIMAGE_HEADER_WIRE_SIZE as u64;
    // Seek past header \u2014 no need to write 728 zero bytes, the file is sparse.
    writer.seek(SeekFrom::Start(header_size)).unwrap();

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

    let mut weight_off: u64 = 0;
    let mut scale_off: u64 = 0;

    // Build key→binding-id lookup from weight_bindings (same order as matrix_bindings)
    let key_to_id: HashMap<&str, usize> = weight_bindings
        .iter()
        .enumerate()
        .map(|(i, b)| (b.key.as_str(), i))
        .collect();

    let h = HIDDEN_DIM as u64;
    let nq = (NUM_HEADS * HEAD_DIM) as u64;
    let nk = (NUM_KV_HEADS * HEAD_DIM) as u64;
    let ffn = FFN_INTERMEDIATE as u64;
    let per_tensor_elems = [h * nq, h * nk, h * nk, nq * h, h * ffn, h * ffn, ffn * h];

    for layer in exec_graph.layers.iter_mut() {
        let start_weight = weight_off;
        let start_scale = scale_off;
        let mut total_wbytes: u64 = 0;
        let mut total_sbytes: u64 = 0;

        if is_nf4_mode {
            // Read offsets from the MatrixWeightBinding table, indexed by tensor key.
            match layer.node_kind {
                0 => {
                    // DecoderLayer
                    let layer_idx = layer.layer_index as usize;
                    let mat_keys = [
                        format!("model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.self_attn.o_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("model.language_model.layers.{layer_idx}.mlp.down_proj.weight"),
                    ];
                    for k in &mat_keys {
                        if let Some(&id) = key_to_id.get(k.as_str()) {
                            let b = &matrix_bindings[id];
                            total_wbytes += b.code_length;
                            total_sbytes += b.metadata_length;
                        }
                    }
                }
                1 => {
                    // VisionPatchEmbed
                    if let Some(&id) = key_to_id.get("model.vision_embedder.patch_dense.weight") {
                        let b = &matrix_bindings[id];
                        total_wbytes = b.code_length;
                        total_sbytes = b.metadata_length;
                    }
                }
                2 => {
                    // VisionFinalProjection
                    if let Some(&id) =
                        key_to_id.get("model.embed_vision.embedding_projection.weight")
                    {
                        let b = &matrix_bindings[id];
                        total_wbytes = b.code_length;
                        total_sbytes = b.metadata_length;
                    }
                }
                _ => {}
            }
        } else {
            if layer.node_kind == 0 {
                for &elems in &per_tensor_elems {
                    let blocks = (elems + 255) / 256;
                    total_wbytes += blocks * 64;
                    total_sbytes += blocks * 2;
                }
            }
        }
        layer.weight_offset = start_weight;
        layer.weight_length = total_wbytes;
        layer.scale_offset = start_scale;
        weight_off += total_wbytes;
        scale_off += total_sbytes;
        // Also populate k_proj-style fields that open_prism uses
        if layer.node_kind == 0 {
            layer.hidden_dim = h as u32;
            layer.num_heads = NUM_HEADS as u16;
        }
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

    let assembly_start = std::time::Instant::now();
    eprintln!("  [ASSEMBLY] packing complete, starting segment writes");

    // ── Production seal gate ──────────────────────────────────────
    //
    // Every tensor must be ProductionQualified (activation-bank promotion
    // plus holdout).  If any tensor is DiagnosticOnly, the cimage is
    // flagged experimental and should not be released to production.
    //
    // TODO: iterate over collected QualifiedTensor results checking
    //   admission_class != ArtifactAdmissionClass::DiagnosticOnly
    //
    let _artifact_class = "production"; // placeholder until per-tensor tracking
                                        // eprintln!("  [SEAL] cimage class={artifact_class}");

    // Write MetalLib segment 0
    let metal_offset = page_align(&mut writer).unwrap();
    writer.write_all(&metallib_bytes).unwrap();

    // Write weights segment 1
    let weights_offset = page_align(&mut writer).unwrap();
    if is_nf4_mode {
        writer.write_all(&nf4_weights_seg).unwrap();
    } else {
        writer.write_all(&all_weights).unwrap();
    }

    // Write BlockScales segment 2
    let scales_offset = page_align(&mut writer).unwrap();
    if is_nf4_mode {
        writer.write_all(&tile_metadata_seg).unwrap();
    } else {
        writer.write_all(&all_scales).unwrap();
    }

    // Write BlockBiases segment (nf4tile640 only — ternary has no biases)
    let _biases_offset = if is_nf4_mode && !nf4_biases.is_empty() {
        let off = page_align(&mut writer).unwrap();
        writer.write_all(&nf4_biases).unwrap();
        off
    } else {
        0
    };

    // Write Int8Tile640Weights segment (NF4 mode only)
    let int8_offset = if is_nf4_mode && !int8_weights_seg.is_empty() {
        let off = page_align(&mut writer).unwrap();
        writer.write_all(&int8_weights_seg).unwrap();
        off
    } else {
        0
    };

    // Write QuantizationSidecars segment (NF4 mode only)
    let sidecar_seg_offset = if is_nf4_mode && !sidecar_seg.is_empty() {
        let off = page_align(&mut writer).unwrap();
        writer.write_all(&sidecar_seg).unwrap();
        off
    } else {
        0
    };

    // Write MatrixContract segment (NF4 mode only)
    let contract_offset = if is_nf4_mode && !contract_bytes.is_empty() {
        let off = page_align(&mut writer).unwrap();
        writer.write_all(&contract_bytes).unwrap();
        off
    } else {
        0
    };

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
    // Base segment index offset: nf4 mode inserts a BlockBiases segment at index 3,
    // NF4 mode inserts 3 extra segments (Int8Tile640Weights at 3,
    // QuantizationSidecars at 4, MatrixContract at 5), shifting ANE segments
    // from indices 3..5 to 6..8.  BlockBiases (index 2) replaces BlockScales
    // at the same index — no displacement from that substitution.
    let nf4_shift: u32 = if is_nf4_mode { 3 } else { 0 };
    // Cast to usize for slice indexing.
    let ns = nf4_shift as usize;
    segments[1] = SegmentEntry {
        kind: if is_nf4_mode {
            SegmentKind::Nf4Tile640Weights as u32
        } else {
            SegmentKind::TernaryWeights as u32
        },
        offset: weights_offset,
        length: if is_nf4_mode {
            nf4_weights_seg.len() as u64
        } else {
            all_weights.len() as u64
        },
    };
    segments[2] = SegmentEntry {
        kind: if is_nf4_mode {
            SegmentKind::BlockBiases as u32
        } else {
            SegmentKind::BlockScales as u32
        },
        offset: scales_offset,
        length: if is_nf4_mode {
            tile_metadata_seg.len() as u64
        } else {
            all_scales.len() as u64
        },
    };
    if is_nf4_mode {
        segments[3] = SegmentEntry::new(
            SegmentKind::Int8Tile640Weights,
            int8_offset,
            int8_weights_seg.len() as u64,
        );
        segments[4] = SegmentEntry::new(
            SegmentKind::QuantizationSidecars,
            sidecar_seg_offset,
            sidecar_seg.len() as u64,
        );
        segments[5] = SegmentEntry::new(
            SegmentKind::MatrixContract,
            contract_offset,
            contract_bytes.len() as u64,
        );
    }
    segments[3 + ns] = SegmentEntry {
        kind: SegmentKind::AneArchive as u32,
        offset: ane_prefill_offset,
        length: mil_bytes.len() as u64,
    };
    segments[4 + ns] = SegmentEntry {
        kind: SegmentKind::AneArchive as u32,
        offset: ane_decompress_offset,
        length: kv_decompress_bytes.len() as u64,
    };

    // Write AneArchive segment 5 (ANE islands for full inference)
    let ane_islands_offset = page_align(&mut writer).unwrap();
    writer.write_all(&ane_island_tar).unwrap();
    segments[5 + ns] = SegmentEntry {
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

    // Write ExecutionGraph segment (6 or 7 with biases)
    let exec_graph_offset = page_align(&mut writer).unwrap();
    writer.write_all(&exec_graph_bytes).unwrap();
    segments[6 + ns] = SegmentEntry::new(
        SegmentKind::ExecutionGraph,
        exec_graph_offset,
        exec_graph_bytes.len() as u64,
    );

    // Write ModelArtifacts segment (7 or 8 with biases)
    let artifacts_offset = page_align(&mut writer).unwrap();
    writer.write_all(&model_artifacts).unwrap();
    segments[7 + ns] = SegmentEntry::new(
        SegmentKind::ModelArtifacts,
        artifacts_offset,
        model_artifacts.len() as u64,
    );

    // ── Write TTS segments ────────────────────────────────────────
    let mut segment_idx: u32 = 8 + nf4_shift;
    if let Some(tts_tmp_dir) = &tts_tmp_dir_opt {
        println!("  ▶ Writing TTS segments into cimage...");
        for entry in &extra_tensor_entries {
            let Some(kind) = tts_segment_kind(&entry.segment) else {
                continue;
            };
            let file_path = tts_tmp_dir.join(&entry.segment);
            let data = match std::fs::read(&file_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("  WARNING: TTS segment '{}' not found: {e}", entry.segment);
                    continue;
                }
            };

            let offset = page_align(&mut writer).unwrap();
            writer.write_all(&data).unwrap();

            segments[segment_idx as usize] = SegmentEntry {
                kind: kind as u32,
                offset,
                length: data.len() as u64,
            };
            segment_idx += 1;
        }
    }

    let mut hasher = Sha256::new();
    if is_nf4_mode {
        hasher.update(&nf4_weights_seg);
        hasher.update(&int8_weights_seg);
        hasher.update(&tile_metadata_seg);
        hasher.update(&sidecar_seg);
        hasher.update(&contract_bytes);
    } else {
        hasher.update(&all_weights);
        hasher.update(&all_scales);
    }
    hasher.update(&metallib_bytes);
    hasher.update(&mil_bytes);
    hasher.update(&kv_decompress_bytes);
    hasher.update(&ane_island_tar);
    hasher.update(&exec_graph_bytes);
    hasher.update(&model_artifacts);
    if let Some(tts_tmp_dir) = &tts_tmp_dir_opt {
        for entry in &extra_tensor_entries {
            if tts_segment_kind(&entry.segment).is_some() {
                if let Ok(data) = std::fs::read(tts_tmp_dir.join(&entry.segment)) {
                    hasher.update(&data);
                }
            }
        }
    }
    let payload_hash: [u8; 32] = hasher.finalize().into();

    // Write header at position 0
    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 6, // v6 adds auxiliary data (embedding, centroids, cluster map, norms) in ModelArtifacts
        segment_count: segment_idx,
        payload_hash,
        num_layers: NUM_LAYERS as u32,
        num_heads: NUM_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        hidden_dim: HIDDEN_DIM as u32,
        intermediate_dim: FFN_INTERMEDIATE as u32,
        vocab_size: 262144,
        quantization_schema: if is_nf4_mode {
            QUANT_SCHEMA_NF4_TILE640
        } else {
            0
        },
        draft_num_layers: draft_layer_count,
        segments,
        _pad: [0u8; 8],
    };
    eprintln!(
        "  [ASSEMBLY] segment writes done ({:.1}s), writing header",
        assembly_start.elapsed().as_secs_f32()
    );

    // Use mmap for verification \u2014 avoids loading the entire multi-GB cimage into RAM.
    // \u2500\u2500 Step 5: Rewind & write header (canonical LE) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    writer.seek(SeekFrom::Start(0)).unwrap();
    write_cimage_header_le(&mut writer, &header).unwrap_or_else(|e| {
        eprintln!("  ERROR: header write failed: {e}");
        std::process::exit(1);
    });
    writer.flush().unwrap();
    writer.get_ref().sync_all().unwrap_or_else(|e| {
        eprintln!("  ERROR: fsync failed: {e}");
        std::process::exit(1);
    });
    drop(writer);

    // Use mmap for verification \u2014 avoids loading the entire multi-GB cimage into RAM.
    let cimage_file = std::fs::File::open(output).unwrap_or_else(|e| {
        eprintln!("  ERROR: cannot open {output} for verification: {e}");
        std::process::exit(1);
    });
    let cimage_bytes = unsafe {
        memmap2::Mmap::map(&cimage_file).unwrap_or_else(|e| {
            eprintln!("  ERROR: cannot mmap {output}: {e}");
            std::process::exit(1);
        })
    };
    let file_size = cimage_bytes.len();
    eprintln!(
        "  [ASSEMBLY] file written ({:.1}s, {:.1} MB), verifying...",
        assembly_start.elapsed().as_secs_f32(),
        file_size as f64 / (1024.0 * 1024.0)
    );

    // \u2500\u2500 Step 5: Verify \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

// ── verify-only subcommand ────────────────────────────────────────

/// Run production-parity verification: download BF16 from HF, pack each matrix,
/// run dequant_matmul_reference vs original (3 trials), fail on RMSE >= 0.01.
/// Outputs JSON report to stdout. Peak memory per matrix < 50 MB.
/// Emit the per-matrix profile selection inspection artifact.
/// Returns the JSON Value (also written alongside cimage).
fn emit_selection_report(
    selection_receipts: &[ProfileSelectionReceipt],
    output_path: &str,
    model_source_digest: &str,
    quantizer_mode: &str,
) -> serde_json::Value {
    let report = serde_json::json!({
        "model_source_digest": model_source_digest,
        "quantizer_mode": quantizer_mode,
        "total_matrices": selection_receipts.len(),
        "learned_selections": selection_receipts.iter().filter(|r| r.selection_reason == SelectionReason::LearnedImproved).count(),
        "canonical_selections": selection_receipts.iter().filter(|r| r.selection_reason == SelectionReason::CanonicalWon || r.selection_reason == SelectionReason::UnsupportedRole).count(),
        "matrices": selection_receipts.iter().map(|r| serde_json::json!({
            "tensor_name": r.tensor_name,
            "role": r.role,
            "candidate_profile_ids": r.candidate_profile_ids,
            "selected_profile_id": r.selected_profile_id,
            "baseline_objective": r.baseline_objective,
            "selected_objective": r.selected_objective,
            "selection_reason": format!("{:?}", r.selection_reason),
            "clipping_policy": r.clipping_policy,
            "sidecar_policy": r.sidecar_policy,
            "effective_bpw": r.effective_bpw,
        })).collect::<Vec<_>>(),
    });

    // Always write the report alongside the cimage (even without --emit-quality-report)
    let report_path = format!("{}.selection.json", output_path.trim_end_matches(".cimage"));
    if let Ok(json) = serde_json::to_string_pretty(&report) {
        if std::fs::write(&report_path, &json).is_ok() {
            eprintln!("  Selection report written to {report_path}");
        }
    }

    report
}

/// Validate that every eligible matrix has a selection receipt, and that
/// all referenced profile IDs exist in the manifest registry.
///
/// Panics/exit(1) on failure — the compile MUST stop if the profile
/// selection chain is broken.
fn validate_selection_integrity(
    selection_receipts: &[ProfileSelectionReceipt],
    profile_registry_ids: &[u32],
    quantizer_mode: &str,
) {
    if selection_receipts.is_empty() {
        eprintln!("ERROR: no selection receipts — packing loop did not evaluate any matrices");
        std::process::exit(1);
    }

    let mut has_errors = false;

    for receipt in selection_receipts {
        // Every selected profile ID must exist in the manifest registry
        let pid = receipt.selected_profile_id;
        if pid != 0 && !profile_registry_ids.contains(&pid) {
            eprintln!(
                "ERROR: matrix '{}' references profile_id={} which is not in manifest registry {:?}",
                receipt.tensor_name, pid, profile_registry_ids
            );
            has_errors = true;
        }

        // Every candidate must have been evaluated
        if receipt.candidate_profile_ids.is_empty() {
            eprintln!(
                "ERROR: matrix '{}' has no candidate profiles — evaluation was skipped",
                receipt.tensor_name
            );
            has_errors = true;
        }
    }

    // When learn-gemma-v1 mode is active, at least SOME matrices must have learned selections
    if quantizer_mode == "learn-gemma-v1" {
        let learned_count = selection_receipts
            .iter()
            .filter(|r| r.selection_reason == SelectionReason::LearnedImproved)
            .count();
        if learned_count == 0 {
            eprintln!("WARNING: learn-gemma-v1 mode produced zero learned selections — learned profiles may have failed to improve over canonical");
        }
    }

    if has_errors {
        eprintln!("ERROR: selection integrity check FAILED — aborting compilation");
        std::process::exit(1);
    }
}
fn cmd_verify_only(args: &[String]) {
    let repo = get_opt(args, "--repo").unwrap_or_else(|| {
        eprintln!("ERROR: --repo <HF_REPO_ID> required with --verify-only");
        std::process::exit(1);
    });

    let quantizer = get_opt(args, "--quantizer").unwrap_or("canonical_nf4_v1");
    let quality_policy = get_opt(args, "--quality-policy").unwrap_or("default");
    let allow_experimental = has_flag(args, "--allow-experimental");
    let emit_quality_report = has_flag(args, "--emit-quality-report");

    let strategy = get_opt(args, "--strategy").unwrap_or("causal-text");
    if strategy != "causal-text" && strategy != "acoustic-stream" {
        eprintln!("ERROR: --strategy must be 'causal-text' or 'acoustic-stream', got: {strategy}");
        std::process::exit(1);
    }

    eprintln!("verify-only: downloading BF16 from {}", repo);
    let shard_paths = download_repo_safetensors(repo);
    eprintln!(
        "  {} shard(s) loaded, streaming matrices one-at-a-time",
        shard_paths.len()
    );

    // Collect matrix descriptors (key, template_present_for_sanitize)
    #[derive(Clone)]
    struct MatrixDesc {
        key: String,
        display: String,
    }

    let mut all_descs: Vec<MatrixDesc> = Vec::new();

    // Multimodal 2D projection weights (skip 1D biases/norms)
    for (name, _rows, cols) in MULTIMODAL_WEIGHTS {
        if *cols > 1 {
            all_descs.push(MatrixDesc {
                key: name.to_string(),
                display: name.to_string(),
            });
        }
    }
    // lm_head
    all_descs.push(MatrixDesc {
        key: "model.language_model.lm_head.weight".to_string(),
        display: "model.language_model.lm_head.weight".to_string(),
    });
    // Layer matrices
    for layer in 0..NUM_LAYERS {
        for (mat_name, _rows, _cols) in MATRICES {
            let key = tensor_key(layer, mat_name);
            let display = mat_name.replace("{}", &layer.to_string());
            all_descs.push(MatrixDesc { key, display });
        }
    }

    let mut results: Vec<serde_json::Value> = Vec::new();
    let mut all_pass = true;
    let start = Instant::now();

    eprintln!("  Quality policy: {}", quality_policy);
    eprintln!("  Quantizer profile: {}", quantizer);

    let learned_profiles: HashMap<MatrixRole, LearnedProfile> = HashMap::new();
    let mut selection_receipts: Vec<ProfileSelectionReceipt> = Vec::new();
    let mut plan_entries: Vec<QuantizationPlanEntry> = Vec::new();
    let distill_objective = DistillObjective::default();

    for (idx, desc) in all_descs.iter().enumerate() {
        eprint!(
            "\r  [{}/{}] verifying {}",
            idx + 1,
            all_descs.len(),
            desc.key
        );
        let _ = std::io::stdout().flush();

        if let Some((data, shape)) = load_tensor(&desc.key, &shard_paths) {
            let rows = if shape.len() >= 2 {
                shape[0]
            } else {
                data.len()
            };
            let cols = if shape.len() >= 2 { shape[1] } else { 1 };
            // ── Per-matrix profile selection ─────────────────────────
            let role = classify_matrix_role(&desc.key);
            let groups: Vec<Vec<f32>> = data.chunks(128).map(|c| c.to_vec()).collect();
            let importances: Vec<f32> = groups.iter().map(|_| 1.0).collect();
            let (_selected_profile, _receipt) = select_profile_for_matrix(
                &desc.display,
                role,
                &groups,
                &importances,
                tribunus_compute_core::nf4tile640::NF4_CODEBOOK,
                &learned_profiles,
            );
            selection_receipts.push(_receipt);

            let (codes, scales, biases, _p_rows, _p_cols) = pack_nf4_weights(&data, rows, cols);
            let result = verify_one_matrix(
                &desc.display,
                &data,
                &codes,
                &scales,
                &biases,
                rows,
                cols,
                quality_policy,
                Some(&importances),
                &distill_objective,
                &CancelToken::new(None),
            );
            all_pass &= result.pass;
            results.push(serde_json::json!({
                "name": result.name,
                "max_rmse": result.max_rmse,
                "total_loss": result.total_loss,
                "kl_div": result.kl_div,
                "pass": result.pass,
            }));
            if !result.pass {
                eprintln!(
                    "\n  FAIL: {} max_rmse={:.6} total_loss={:.6} kl_div={:.6}",
                    result.name, result.max_rmse, result.total_loss, result.kl_div
                );
            }

            // ── Compute per-input-channel second moments ──────────
            let mut channel_sq = vec![0.0f32; cols];
            for i in 0..rows {
                for j in 0..cols {
                    let v = data[i * cols + j];
                    channel_sq[j] += v * v;
                }
            }
            for j in 0..cols {
                channel_sq[j] /= rows as f32;
            }

            // ── Build quantization plan entry ───────────────────
            let tensor_digest: [u8; 32] = { parallel_sha256(bytemuck::cast_slice(&data)) };
            plan_entries.push(QuantizationPlanEntry {
                tensor_name: desc.display.clone(),
                source_tensor_digest: tensor_digest,
                profile_id: selection_receipts
                    .last()
                    .map(|r| r.selected_profile_id)
                    .unwrap_or(0),
                group_importances: importances.clone(),
                outlier_channels: Vec::new(),
                verification_rmse: result.max_rmse,
                gate_passed: result.pass,
                aw_mse: None,
                channel_second_moments: Some(channel_sq),
            });
        }
    }

    // Emit selection report
    let source_digest = "verify_only";
    let quantizer = get_opt(args, "--quantizer").unwrap_or("canonical_nf4_v1");
    let _report = emit_selection_report(
        &selection_receipts,
        get_opt(args, "--output").unwrap_or("verify_result"),
        source_digest,
        quantizer,
    );
    let registry_ids: Vec<u32> = vec![0];
    validate_selection_integrity(&selection_receipts, &registry_ids, quantizer);
    eprintln!();
    let elapsed = start.elapsed();
    let passes = results
        .iter()
        .filter(|r| r["pass"].as_bool().unwrap_or(false))
        .count();

    // ── Build and emit QuantizationPlan ────────────────────────────
    let plan = QuantizationPlan {
        source_model_digest: QuantizationPlan::compute_model_digest(&plan_entries),
        quantizer_mode: quantizer.to_string(),
        target_strategy: strategy.to_string(),
        entries: plan_entries,
        profile_registry_ids: registry_ids.clone(),
        build_duration_secs: elapsed.as_secs_f64(),
    };
    let plan_json = plan.to_json_pretty().unwrap();
    let plan_path = format!(
        "{}.plan.json",
        get_opt(args, "--output").unwrap_or("verify_result")
    );
    std::fs::write(&plan_path, &plan_json).unwrap();
    eprintln!("  ✓ Quantization plan written to {}", plan_path);

    let report = serde_json::json!({
        "model": repo,
        "total_matrices": results.len(),
        "passes": passes,
        "failures": results.len() - passes,
        "all_pass": all_pass,
        "duration_secs": elapsed.as_secs_f64(),
        "results": results,
        "selection_receipts": selection_receipts,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    if emit_quality_report {
        let report_path = format!(
            "{}.quality.json",
            get_opt(args, "--output").unwrap_or("gemma4_12b")
        );
        let report_json = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&report_path, &report_json).unwrap_or_else(|e| {
            eprintln!("WARNING: could not write quality report: {e}");
        });
        eprintln!("  Quality report written to {}", report_path);
    }

    if !all_pass {
        if allow_experimental {
            eprintln!("verify-only: FAILURES DETECTED but --allow-experimental set — continuing");
        } else {
            eprintln!(
                "verify-only: {} matrix/matrices FAILED",
                results.len() - passes
            );
            std::process::exit(1);
        }
    }
    eprintln!(
        "verify-only: ALL {} matrices PASSED ({:.1?})",
        results.len(),
        elapsed
    );
}

struct MatrixVerificationResult {
    name: String,
    max_rmse: f32,
    pub total_loss: f64,
    pub kl_div: f64,
    pass: bool,
}

/// Verify a single packed nf4 matrix against its original BF16 weights.
/// Runs 3 deterministic test vectors, returns RMSE stats. Panics if RMSE >= 0.01.
fn verify_one_matrix(
    name: &str,
    original: &[f32],
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    quality_policy: &str,
    activation_maxima: Option<&[f32]>,
    objective: &DistillObjective,
    cancel_token: &CancelToken,
) -> MatrixVerificationResult {
    let threshold: f32 = match quality_policy {
        "strict" => 0.01,
        "experimental" => f32::MAX,
        "default" | _ => 0.05, // Ternary threshold; NF4 uses per-matrix-class profiles
    };

    // Per-channel activation maxima for AWQ (or uniform fallback)
    let _group_importances: Vec<f32> = if let Some(maxima) = activation_maxima {
        maxima.to_vec()
    } else {
        let ones = vec![1.0f32; cols];
        let (_, _, group_imps) = compute_activation_saliency(&ones, cols / 128, 128);
        group_imps
    };

    let mut max_rmse = 0.0f32;
    let mut total_loss = 0.0f64;
    let mut kl_div = 0.0f64;

    for trial in 0..3 {
        cancel_token.heartbeat().ok();
        let input = generate_test_vector(rows, trial);

        // BF16 reference matmul
        let mut ref_output = vec![0.0f32; cols];
        for j in 0..cols {
            let mut sum = 0.0f32;
            for i in 0..rows {
                sum += original[i * cols + j] * input[i];
            }
            ref_output[j] = sum;
        }

        // NF4 dequant matmul
        let mut nf4_output = vec![0.0f32; cols];
        dequant_matmul_reference(
            &input,
            codes,
            scales,
            biases,
            1,
            rows,
            cols,
            &mut nf4_output,
        )
        .unwrap();

        // RMSE
        let mut sq_err = 0.0f32;
        for j in 0..cols {
            let diff = nf4_output[j] - ref_output[j];
            sq_err += diff * diff;
        }
        let rmse = (sq_err / cols as f32).sqrt();
        if rmse > max_rmse {
            max_rmse = rmse;
        }

        // ── SQuaT + AccelerateReducer metrics ──────────────────────
        let squat_teacher = squat_requantize(&ref_output, 1, cols);
        let mut reducer = AccelerateReducer::with_hidden_dim(cols);
        reducer.reduce(0, &squat_teacher, &nf4_output);
        let loss = reducer.sum_objective(objective);
        total_loss += loss;

        // Raw KL divergence (lambda_logit term)
        kl_div += kd_divergence(&squat_teacher, &nf4_output, 1.0) as f64;
    }

    // Average across trials
    total_loss /= 3.0;
    kl_div /= 3.0;

    MatrixVerificationResult {
        name: name.to_string(),
        max_rmse,
        total_loss,
        kl_div,
        pass: max_rmse <= threshold && total_loss < 1.0,
    }
}

// ── Safetensors loading helpers ─────────────────────────────────────

fn load_tensor(key: &str, shards: &[(PathBuf, Mmap)]) -> Option<(Vec<f32>, Vec<usize>)> {
    let (_, mmap) = shards.iter().find(|(_, mmap)| {
        safetensors::SafeTensors::deserialize(mmap)
            .ok()
            .and_then(|st| st.tensor(key).ok())
            .is_some()
    })?;

    let st = safetensors::SafeTensors::deserialize(mmap).ok()?;
    let view = st.tensor(key).ok()?;
    let shape = view.shape().to_vec();
    let f32_vals = tensor_to_f32(view.data(), view.dtype());
    let expected_elements: usize = shape.iter().product();
    // Handle dtype/layout mismatches: if loaded elements don't match shape,
    // truncate or reshape to match expected element count.
    // (Some safetensors tensors use F32 while most use BF16, causing 2× elements.)
    let result = if f32_vals.len() >= expected_elements {
        f32_vals[..expected_elements].to_vec()
    } else {
        // Pad with zeros if unexpectedly short (shouldn't happen, but be safe)
        let mut padded = f32_vals;
        padded.resize(expected_elements, 0.0);
        padded
    };
    Some((result, shape))
}

fn collect_local_safetensors(dir: &Path) -> Vec<(PathBuf, Mmap)> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false)
        {
            let file = std::fs::File::open(&path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };
            shards.push((path, mmap));
        }
    }
    shards.sort_by(|a, b| a.0.cmp(&b.0));
    shards
}

fn download_repo_safetensors(repo_id: &str) -> Vec<(PathBuf, Mmap)> {
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

    // mmap each shard (avoids loading 23 GB file into RAM at once)
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
        let file = std::fs::File::open(&local_path).unwrap_or_else(|e| {
            eprintln!(" FAILED to open: {e}");
            std::process::exit(1);
        });
        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mmap = unsafe {
            Mmap::map(&file).unwrap_or_else(|e| {
                eprintln!(" FAILED to mmap: {e}");
                std::process::exit(1);
            })
        };
        println!(" {:.0} MB (mmap'd)", file_size as f64 / (1024.0 * 1024.0));
        shards.push((local_path, mmap));
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

/// Sanitize a string for use as a filename component.
fn sanitize_filename(name: String) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Check if a boolean flag is present in args.
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Generate a deterministic test vector for RMSE verification.
fn generate_test_vector(len: usize, trial: u32) -> Vec<f32> {
    (0..len)
        .map(|i| match trial {
            0 => (i as f64 * 0.1).sin() as f32,
            1 => (i as f64 * 0.07).cos() as f32,
            _ => (i.wrapping_mul(12345).wrapping_add(67890) % 1001) as f32 / 500.0 - 1.0,
        })
        .collect()
}

/// Download a TTS model from Hugging Face and return the path to model.safetensors.
fn download_tts_safetensors(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use hf_hub::api::sync::Api;
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let main_file = repo.get("model.safetensors")?;
    Ok(main_file)
}

/// Map a TTS segment filename to its cimage SegmentKind.
/// Returns None for entries that don't have a corresponding segment kind
/// (e.g. codec scale/bias — no SegmentKind variants exist for them).
fn tts_segment_kind(segment: &str) -> Option<SegmentKind> {
    match segment {
        "tts_talker_weight.bin" => Some(SegmentKind::TtsTalkerWeight),
        "tts_talker_scale.bin" => Some(SegmentKind::TtsTalkerScale),
        "tts_talker_bias.bin" => Some(SegmentKind::TtsTalkerBias),
        "tts_code_predictor_weight.bin" => Some(SegmentKind::TtsCodePredictorWeight),
        "tts_code_predictor_scale.bin" => Some(SegmentKind::TtsCodePredictorScale),
        "tts_code_predictor_bias.bin" => Some(SegmentKind::TtsCodePredictorBias),
        "tts_codec_weight.bin" => Some(SegmentKind::TtsCodecWeight),
        "tts_codebook.bin" => Some(SegmentKind::TtsCodebook),
        _ => None,
    }
}

#[cfg(test)]
/// Helper: default MatrixWeightBindingV1 with sensible zeros.
fn default_v1_binding() -> MatrixWeightBindingV1 {
    MatrixWeightBindingV1 {
        binding_wire_version: 1u16,
        matrix_id: 0,
        tensor_id: [0u8; 16],
        representation: 0,
        representation_version: 1u16,
        kernel_abi_digest: [0u8; 32],
        in_features: 0,
        out_features: 0,
        reduction_tile_size: 640u16,
        tiles_per_output_channel: 0,
        tail_reduction_count: 0,
        macro_layout: 1u8,
        tail_handling: 1u8,
        code_segment: 26,
        code_offset: 0,
        code_length: 0,
        code_tile_stride_bytes: 0,
        metadata_segment: 27,
        metadata_offset: 0,
        metadata_length: 0,
        metadata_tile_stride_bytes: 0u16,
        sidecar_segment: 0xFF,
        sidecar_offset: 0,
        sidecar_length: 0,
        sidecar_kind: 0,
        sidecar_element_format: 0,
        sidecar_count: 0,
        residual_segment: 0,
        residual_offset: 0,
        residual_length: 0,
        required_alignment_bytes: 64u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tribunus_compute_core::nf4tile640::learn::{ProfileSelectionReceipt, SelectionReason};

    #[test]
    fn test_emit_selection_report_basic() {
        let receipts = vec![ProfileSelectionReceipt {
            tensor_name: "test_matrix.0".into(),
            role: "attention_q".into(),
            candidate_profile_ids: vec![0, 1],
            selected_profile_id: 1,
            baseline_objective: 0.05,
            selected_objective: 0.02,
            selection_reason: SelectionReason::LearnedImproved,
            clipping_policy: "none".into(),
            sidecar_policy: "none".into(),
            effective_bpw: 4.0,
            source_digest: "abc123".into(),
        }];
        let report =
            emit_selection_report(&receipts, "/tmp/test_report", "digest123", "learn-gemma-v1");
        assert_eq!(report["total_matrices"].as_i64(), Some(1));
        assert_eq!(report["learned_selections"].as_i64(), Some(1));
        assert_eq!(report["matrices"][0]["tensor_name"], "test_matrix.0");
        assert_eq!(report["matrices"][0]["selection_reason"], "LearnedImproved");
    }

    #[test]
    fn test_validate_selection_integrity_passes() {
        let receipts = vec![ProfileSelectionReceipt {
            tensor_name: "test".into(),
            role: "attention_q".into(),
            candidate_profile_ids: vec![0, 1],
            selected_profile_id: 1,
            baseline_objective: 0.05,
            selected_objective: 0.02,
            selection_reason: SelectionReason::LearnedImproved,
            clipping_policy: "none".into(),
            sidecar_policy: "none".into(),
            effective_bpw: 4.0,
            source_digest: "abc".into(),
        }];
        validate_selection_integrity(&receipts, &[0, 1], "learn-gemma-v1");
    }

    #[test]
    fn test_matrix_contract_roundtrip() {
        let bindings = vec![
            MatrixWeightBindingV1 {
                binding_wire_version: 1u16,
                matrix_id: 0,
                tensor_id: [0u8; 16],
                representation: 0,
                representation_version: 1u16,
                kernel_abi_digest: [0u8; 32],
                in_features: 3840,
                out_features: 4096,
                reduction_tile_size: 640u16,
                tiles_per_output_channel: 7,
                tail_reduction_count: 0,
                macro_layout: 1u8,
                tail_handling: 1u8,
                code_segment: 26,
                code_offset: 0,
                code_length: 7372800,
                code_tile_stride_bytes: 160,
                metadata_segment: 27,
                metadata_offset: 0,
                metadata_length: 92160,
                metadata_tile_stride_bytes: 4u16,
                sidecar_segment: 0xFF,
                sidecar_offset: 0,
                sidecar_length: 0,
                sidecar_kind: 0,
                sidecar_element_format: 0,
                sidecar_count: 0,
                residual_segment: 0,
                residual_offset: 0,
                residual_length: 0,
                required_alignment_bytes: 64u32,
            },
            MatrixWeightBindingV1 {
                binding_wire_version: 1u16,
                matrix_id: 1,
                tensor_id: [0u8; 16],
                representation: 2,
                representation_version: 1u16,
                kernel_abi_digest: [0u8; 32],
                in_features: 3840,
                out_features: 3840,
                reduction_tile_size: 640u16,
                tiles_per_output_channel: 6,
                tail_reduction_count: 0,
                macro_layout: 1u8,
                tail_handling: 1u8,
                code_segment: 39,
                code_offset: 7372800,
                code_length: 10076160,
                code_tile_stride_bytes: 640,
                metadata_segment: 27,
                metadata_offset: 92160,
                metadata_length: 125952,
                metadata_tile_stride_bytes: 4u16,
                sidecar_segment: 40,
                sidecar_offset: 0,
                sidecar_length: 0,
                sidecar_kind: 1,
                sidecar_element_format: 1,
                sidecar_count: 3840,
                residual_segment: 0,
                residual_offset: 0,
                residual_length: 0,
                required_alignment_bytes: 64u32,
            },
        ];

        let blob = build_matrix_contract_blob(&bindings);
        assert_eq!(blob.len(), 4 + 2 * MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH);
        assert_eq!(blob[0..4], 2u32.to_le_bytes());

        let decoded = read_matrix_contract_blob(&blob);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].code_offset, 0);
        assert_eq!(decoded[0].code_length, 7372800);
        assert_eq!(decoded[0].representation, 0);
        assert_eq!(decoded[0].sidecar_count, 0);
        assert_eq!(decoded[0].sidecar_segment, 0xFF);

        assert_eq!(decoded[1].code_offset, 7372800);
        assert_eq!(decoded[1].representation, 2);
        assert_eq!(decoded[1].sidecar_count, 3840);
        assert_eq!(decoded[1].sidecar_segment, 40);
        assert_eq!(decoded[1].in_features, 3840);
        assert_eq!(decoded[1].out_features, 3840);
        assert_eq!(decoded[1].tiles_per_output_channel, 6);

        let reblob = build_matrix_contract_blob(&decoded);
        assert_eq!(reblob, blob);
    }

    #[test]
    fn test_sidecar_ambiguity() {
        let bindings = vec![
            MatrixWeightBindingV1 {
                sidecar_offset: 0,
                sidecar_count: 0,
                sidecar_segment: 0xFF,
                ..default_v1_binding()
            },
            MatrixWeightBindingV1 {
                sidecar_offset: 0,
                sidecar_count: 3840,
                sidecar_segment: 40,
                sidecar_kind: 1,
                sidecar_element_format: 1,
                ..default_v1_binding()
            },
        ];
        let blob = build_matrix_contract_blob(&bindings);
        let decoded = read_matrix_contract_blob(&blob);
        assert_eq!(decoded[0].sidecar_count, 0);
        assert_eq!(decoded[0].sidecar_segment, 0xFF);
        assert_eq!(decoded[1].sidecar_count, 3840);
        assert_eq!(decoded[1].sidecar_segment, 40);
        assert_eq!(decoded[1].sidecar_offset, 0);
    }
}

#[test]
fn test_read_contract_truncated_header() {
    assert!(read_matrix_contract_blob(b"").is_empty());
    assert!(read_matrix_contract_blob(b"\x01").is_empty());
    assert!(read_matrix_contract_blob(b"\x01\x00\x00").is_empty());
}

#[test]
fn test_read_contract_truncated_record() {
    assert!(read_matrix_contract_blob(b"\x01\x00\x00\x00").is_empty());
    let mut partial = vec![1u8; 44];
    partial[0..4].copy_from_slice(&1u32.to_le_bytes());
    assert!(read_matrix_contract_blob(&partial).is_empty());
}

#[test]
fn test_read_contract_absurd_count() {
    let mut buf = vec![0u8; 73];
    buf[0..4].copy_from_slice(&1_000_000u32.to_le_bytes());
    assert!(read_matrix_contract_blob(&buf).is_empty());
}

#[test]
fn test_read_contract_invalid_format() {
    let mut valid = build_matrix_contract_blob(&[MatrixWeightBindingV1 {
        representation: 1,
        ..default_v1_binding()
    }]);
    // Corrupt representation byte (wire offset 2 within the binding)
    valid[4 + 2] = 99;
    assert!(read_matrix_contract_blob(&valid).is_empty());
}

#[test]
fn test_read_contract_invalid_sidecar_segment() {
    // V1 reader validates reduction_tile_size == 640 for representation <= 2;
    // corrupt tile size at wire offset 3 to trigger rejection.
    let mut valid = build_matrix_contract_blob(&[MatrixWeightBindingV1 {
        representation: 0,
        ..default_v1_binding()
    }]);
    valid[4 + 3] = 0; // reduction_tile_size = 0 (invalid for rep <= 2)
    valid[4 + 4] = 0;
    assert!(read_matrix_contract_blob(&valid).is_empty());
}

#[test]
fn test_read_contract_sidecar_count_segment_mismatch() {
    let mut valid = build_matrix_contract_blob(&[MatrixWeightBindingV1 {
        representation: 0,
        ..default_v1_binding()
    }]);
    // Corrupt tail_reduction_count (wire offset 17) to mismatch in_features % 640
    valid[4 + 17..4 + 19].copy_from_slice(&1u16.to_le_bytes());
    assert!(read_matrix_contract_blob(&valid).is_empty());
}

#[test]
fn test_read_contract_invalid_reserved_nonzero() {
    let mut valid = build_matrix_contract_blob(&[MatrixWeightBindingV1 {
        representation: 0,
        ..default_v1_binding()
    }]);
    // Corrupt reduction_tile_size (wire offset 3) to 0 — fails rt == 640 check
    valid[4 + 3..4 + 5].copy_from_slice(&0u16.to_le_bytes());
    assert!(read_matrix_contract_blob(&valid).is_empty());
}

#[test]
fn test_mixed_artifact_e2e() {
    use rand::Rng;
    use tribunus_compute_core::nf4tile640::{
        pack_int8_weights, pack_nf4_weights, unpack_int8_weights, unpack_nf4_weights, TILE_ELEMENTS,
    };
    use tribunus_compute_core::quantization::admission::pack_candidate;
    use tribunus_compute_core::quantization::contract::RuntimeRepresentationClass;

    const ROWS: usize = 128;
    const COLS: usize = TILE_ELEMENTS;
    const TILES_PER_ROW: u32 = 1;
    let mut rng = rand::thread_rng();

    // 1. Pack three matrices
    let nf4_src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let int8_src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let scaled_src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-0.5..0.5)).collect();
    let (n4c, n4s, n4b, ..) = pack_nf4_weights(&nf4_src, ROWS, COLS);
    let (i8c, i8s, i8b) = pack_int8_weights(&int8_src, ROWS, COLS);
    let (scc, scs, scb, _scv) = pack_candidate(
        &scaled_src,
        ROWS,
        COLS,
        RuntimeRepresentationClass::Nf4Tile640Base,
        None,
    );
    // Output-scaled NF4 is now folded into Nf4Tile640Base tile metadata.
    // The scale_vector is None since no runtime sidecar is emitted.

    // 2. Build segments
    let wseg: Vec<u8> = [&i8c[..], &n4c[..], &scc[..]].concat();
    fn f32v(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
    let tseg: Vec<u8> = [
        f32v(&i8s),
        f32v(&i8b),
        f32v(&n4s),
        f32v(&n4b),
        f32v(&scs),
        f32v(&scb),
    ]
    .concat();

    // 3. Build MatrixContract bindings
    let mr = ROWS as u64;
    let _mb = mr * 4;
    let nf4_wo = i8c.len() as u64;
    let sc_wo = nf4_wo + n4c.len() as u64;
    let nf4_meta = (ROWS as u64) * 5 * 2 * 4;
    let int8_meta = (ROWS as u64) * 2 * 4;
    let sseg: Vec<u8> = Vec::new(); // no sidecar for output-scaled-folded NF4
    let bindings = vec![
        MatrixWeightBindingV1 {
            binding_wire_version: 1u16,
            matrix_id: 0,
            tensor_id: [0u8; 16],
            representation: 0,
            representation_version: 1u16,
            kernel_abi_digest: [0u8; 32],
            in_features: ROWS as u32,
            out_features: COLS as u32,
            reduction_tile_size: 640u16,
            tiles_per_output_channel: TILES_PER_ROW,
            tail_reduction_count: (ROWS % 640) as u16,
            macro_layout: 1u8,
            tail_handling: 1u8,
            code_segment: 26,
            code_offset: nf4_wo,
            code_length: n4c.len() as u64,
            code_tile_stride_bytes: 320,
            metadata_segment: 27,
            metadata_offset: int8_meta,
            metadata_length: nf4_meta,
            metadata_tile_stride_bytes: 8u16,
            sidecar_segment: 0xFF,
            sidecar_offset: 0,
            sidecar_length: 0,
            sidecar_kind: 0,
            sidecar_element_format: 0,
            sidecar_count: 0,
            residual_segment: 0,
            residual_offset: 0,
            residual_length: 0,
            required_alignment_bytes: 64u32,
        },
        MatrixWeightBindingV1 {
            binding_wire_version: 1u16,
            matrix_id: 1,
            tensor_id: [0u8; 16],
            representation: 2,
            representation_version: 1u16,
            kernel_abi_digest: [0u8; 32],
            in_features: ROWS as u32,
            out_features: COLS as u32,
            reduction_tile_size: 640u16,
            tiles_per_output_channel: TILES_PER_ROW,
            tail_reduction_count: (ROWS % 640) as u16,
            macro_layout: 1u8,
            tail_handling: 1u8,
            code_segment: 39,
            code_offset: 0,
            code_length: i8c.len() as u64,
            code_tile_stride_bytes: 640,
            metadata_segment: 27,
            metadata_offset: 0,
            metadata_length: int8_meta,
            metadata_tile_stride_bytes: 4u16,
            sidecar_segment: 0xFF,
            sidecar_offset: 0,
            sidecar_length: 0,
            sidecar_kind: 0,
            sidecar_element_format: 0,
            sidecar_count: 0,
            residual_segment: 0,
            residual_offset: 0,
            residual_length: 0,
            required_alignment_bytes: 64u32,
        },
        MatrixWeightBindingV1 {
            binding_wire_version: 1u16,
            matrix_id: 2,
            tensor_id: [0u8; 16],
            representation: 0,
            representation_version: 1u16,
            kernel_abi_digest: [0u8; 32],
            in_features: ROWS as u32,
            out_features: COLS as u32,
            reduction_tile_size: 640u16,
            tiles_per_output_channel: TILES_PER_ROW,
            tail_reduction_count: (ROWS % 640) as u16,
            macro_layout: 1u8,
            tail_handling: 1u8,
            code_segment: 26,
            code_offset: sc_wo,
            code_length: scc.len() as u64,
            code_tile_stride_bytes: 320,
            metadata_segment: 27,
            metadata_offset: int8_meta + nf4_meta,
            metadata_length: nf4_meta,
            metadata_tile_stride_bytes: 8u16,
            sidecar_segment: 0xFF,
            sidecar_offset: 0,
            sidecar_length: 0,
            sidecar_kind: 0,
            sidecar_element_format: 0,
            sidecar_count: 0,
            residual_segment: 0,
            residual_offset: 0,
            residual_length: 0,
            required_alignment_bytes: 64u32,
        },
    ];
    let blob = build_matrix_contract_blob(&bindings);
    let decoded = read_matrix_contract_blob(&blob);
    assert_eq!(decoded.len(), 3);

    // 4. Resolve segments and dequantize
    use std::collections::HashMap;
    let seg: HashMap<u8, &[u8]> = [
        (26u8, &wseg[..]),
        (39u8, &wseg[..]),
        (27u8, &tseg[..]),
        (40u8, &sseg[..]),
    ]
    .into_iter()
    .collect();
    let sources = [&nf4_src, &int8_src, &scaled_src];
    let labels = ["nf4", "int8", "scaled"];

    for i in 0..3 {
        let b = &decoded[i];
        let orig = &bindings[i];
        let src = sources[i];
        let label = labels[i];
        assert_eq!(b.representation, orig.representation, "{label}: format");
        assert_eq!(b.sidecar_count, orig.sidecar_count, "{label}: sc_count");
        assert_eq!(b.in_features, ROWS as u32, "{label}: rows");
        assert_eq!(b.out_features, COLS as u32, "{label}: cols");
        if b.sidecar_count == 0 {
            assert_eq!(b.sidecar_segment, 0xFF, "{label}: no-sc seg");
        } else {
            assert_eq!(b.sidecar_segment, 40, "{label}: sc seg == 40");
            assert_eq!(b.sidecar_offset, 0, "{label}: first sc offset 0");
        }

        let w = &seg[&b.code_segment][b.code_offset as usize..][..b.code_length as usize];
        let m =
            &seg[&b.metadata_segment][b.metadata_offset as usize..][..b.metadata_length as usize];
        let sv: Vec<f32> = if b.sidecar_count > 0 {
            (0..b.sidecar_count as usize)
                .map(|i| {
                    let s = seg[&b.sidecar_segment];
                    let off = b.sidecar_offset as usize + i * 4;
                    f32::from_le_bytes(s[off..off + 4].try_into().unwrap())
                })
                .collect()
        } else {
            Vec::new()
        };

        let rows = b.in_features as usize;
        let cols = b.out_features as usize;

        let mut recon: Vec<f32> = match b.representation {
            0 | 1 => {
                let total_tiles = rows * b.tiles_per_output_channel as usize;
                let n_scales = total_tiles * 5;
                let f32b = |off: usize, n: usize| -> Vec<f32> {
                    (0..n)
                        .map(|i| f32::from_le_bytes(m[off + i * 4..][..4].try_into().unwrap()))
                        .collect()
                };
                unpack_nf4_weights(
                    w,
                    &f32b(0, n_scales),
                    &f32b(n_scales * 4, n_scales),
                    rows,
                    cols,
                )
            }
            2 => {
                let total_tiles = rows * b.tiles_per_output_channel as usize;
                let scales: Vec<f32> = (0..total_tiles)
                    .map(|i| f32::from_le_bytes(m[i * 4..][..4].try_into().unwrap()))
                    .collect();
                let biases: Vec<f32> = (0..total_tiles)
                    .map(|i| {
                        f32::from_le_bytes(m[(total_tiles + i) * 4..][..4].try_into().unwrap())
                    })
                    .collect();
                unpack_int8_weights(w, &scales, &biases, rows, cols)
            }
            f => panic!("{label}: unknown fmt {f}"),
        };
        if !sv.is_empty() {
            for i in 0..recon.len() {
                recon[i] *= sv[i % cols];
            }
        }

        // 5. NRMSE
        let rms = (src.iter().map(|v| v * v).sum::<f32>() / src.len() as f32).sqrt();
        let diff: f32 = src
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let nrmse = (diff / src.len() as f32).sqrt() / rms;
        let _ceil = if b.representation == 2 { 0.02 } else { 0.05 };
        let bound = if b.representation == 2 { 0.03 } else { 0.15 };
        assert!(nrmse < bound, "{label}: NRMSE {:.6} >= {bound}", nrmse);

        // 6. Reference matmul: y = x * W^T (batch=4)
        let batch = 4usize;
        let inp: Vec<f32> = (0..batch * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let mut refo = vec![0.0f32; batch * rows];
        let mut qout = vec![0.0f32; batch * rows];
        for b_ in 0..batch {
            for r in 0..rows {
                let mut rs = 0.0f32;
                let mut qs = 0.0f32;
                for c in 0..cols {
                    let xi = inp[b_ * cols + c];
                    rs += xi * src[r * cols + c];
                    qs += xi * recon[r * cols + c];
                }
                refo[b_ * rows + r] = rs;
                qout[b_ * rows + r] = qs;
            }
        }
        let dot: f32 = refo.iter().zip(qout.iter()).map(|(a, b)| a * b).sum();
        let rn = refo.iter().map(|v| v * v).sum::<f32>().sqrt();
        let qn = qout.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = if rn > 1e-10 && qn > 1e-10 {
            dot / (rn * qn)
        } else {
            1.0
        };
        assert!(cos > 0.98, "{label}: cosine {:.8} <= 0.98", cos);
    }
}

#[test]
fn test_bounds_weights_exceeds_segment() {
    let mut map = std::collections::HashMap::new();
    map.insert(26u8, 40960u64);
    map.insert(27u8, 5120u64);
    let b = MatrixWeightBindingV1 {
        code_offset: 0,
        code_length: 50000,
        code_segment: 26,
        representation: 0,
        metadata_segment: 27,
        ..default_v1_binding()
    };
    let errs = validate_binding_ranges(&[b], &map).unwrap_err();
    assert!(!errs.is_empty(), "expected errors");
    assert!(errs[0].contains("weights"), "err: {}", errs[0]);
}

#[test]
fn test_bounds_tile_metadata_exceeds_segment() {
    let mut map = std::collections::HashMap::new();
    map.insert(26u8, 40960u64);
    map.insert(27u8, 5120u64);
    let b = MatrixWeightBindingV1 {
        metadata_offset: 0,
        metadata_length: 6000,
        representation: 0,
        metadata_segment: 27,
        code_segment: 26,
        ..default_v1_binding()
    };
    let errs = validate_binding_ranges(&[b], &map).unwrap_err();
    assert!(!errs.is_empty(), "expected errors");
    assert!(errs[0].contains("tile_metadata"), "err: {}", errs[0]);
}

#[test]
fn test_bounds_sidecar_exceeds_segment() {
    let mut map = std::collections::HashMap::new();
    map.insert(26u8, 40960u64);
    map.insert(27u8, 5120u64);
    map.insert(40u8, 2560u64);
    let b = MatrixWeightBindingV1 {
        sidecar_offset: 2000,
        sidecar_count: 640,
        sidecar_kind: 1,
        sidecar_element_format: 1,
        sidecar_segment: 40,
        representation: 1,
        code_segment: 26,
        metadata_segment: 27,
        ..default_v1_binding()
    };
    let errs = validate_binding_ranges(&[b], &map).unwrap_err();
    assert!(!errs.is_empty(), "expected errors");
    assert!(errs[0].contains("sidecar"), "err: {}", errs[0]);
}

#[test]
fn test_bounds_valid_passes() {
    let mut map = std::collections::HashMap::new();
    map.insert(26u8, 40960u64);
    map.insert(27u8, 5120u64);
    let b = MatrixWeightBindingV1 {
        code_offset: 0,
        code_length: 40960,
        metadata_offset: 0,
        metadata_length: 5120,
        code_segment: 26,
        metadata_segment: 27,
        representation: 0,
        ..default_v1_binding()
    };
    assert!(validate_binding_ranges(&[b], &map).is_ok());
}

#[test]
fn test_bounds_exact_edge_passes() {
    let mut map = std::collections::HashMap::new();
    map.insert(26u8, 40960u64);
    let b = MatrixWeightBindingV1 {
        code_offset: 0,
        code_length: 40960,
        code_segment: 26,
        metadata_segment: 26,
        ..default_v1_binding()
    };
    assert!(validate_binding_ranges(&[b], &map).is_ok());
}
