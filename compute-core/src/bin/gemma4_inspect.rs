//! gemma4_inspect — Inspect a Gemma 4 checkpoint and emit tensor inventory.
//!
//! Usage:
//!   cargo run --bin gemma4_inspect -- --model-dir ./gemma4-12B --emit inventory.json --emit processor_contract.json
//!
//! Reads config.json, tokenizer_config.json, processor_config.json, and
//! Safetensors index. Classifies every tensor using the Gemma4Unified schema.
//! Fails if any tensor classified as Unknown exceeds 1M parameters.

use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use tribunus_compute_core::compute_image::model_family::{
    classify_tensor_name, Gemma4UnifiedSchema, TensorClassification,
};

const UNKNOWN_PARAM_THRESHOLD: u64 = 1_000_000;

fn get_opt(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn get_opts(args: &[String], flag: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == flag && i + 1 < args.len() {
            result.push(args[i + 1].clone());
            i += 1;
        }
        i += 1;
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: gemma4_inspect --model-dir <PATH> [--emit <PATH>]...");
        eprintln!();
        eprintln!("  --model-dir <PATH>  Path to Gemma 4 checkpoint directory");
        eprintln!("  --emit <PATH>        Emit tensor_inventory.json or processor_contract.json");
        std::process::exit(1);
    }

    let model_dir = get_opt(&args, "--model-dir").unwrap_or_else(|| {
        eprintln!("ERROR: --model-dir is required");
        std::process::exit(1);
    });

    let emit_paths = get_opts(&args, "--emit");

    let dir = Path::new(&model_dir);
    if !dir.is_dir() {
        eprintln!("ERROR: model-dir not found: {}", model_dir);
        std::process::exit(1);
    }

    println!("Inspecting checkpoint: {}", model_dir);

    // ── Read config.json ──────────────────────────────────────────
    let config_path = dir.join("config.json");
    let config: serde_json::Value = serde_json::from_reader(BufReader::new(
        fs::File::open(&config_path).unwrap_or_else(|e| {
            eprintln!("ERROR: cannot read config.json: {}", e);
            std::process::exit(1);
        }),
    ))
    .unwrap_or_else(|e| {
        eprintln!("ERROR: invalid config.json: {}", e);
        std::process::exit(1);
    });

    let hidden_size = config["hidden_size"].as_u64().unwrap_or(0) as u32;
    let num_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as u32;
    let num_heads = config["num_attention_heads"].as_u64().unwrap_or(0) as u32;
    let num_kv_heads = config["num_key_value_heads"].as_u64().unwrap_or(0) as u32;
    let vocab_size = config["vocab_size"].as_u64().unwrap_or(0) as u32;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(0) as u32;

    println!("  hidden_size: {}", hidden_size);
    println!("  num_layers: {}", num_layers);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  vocab_size: {}", vocab_size);
    println!("  intermediate_size: {}", intermediate_size);

    // ── Build schema and validate ─────────────────────────────────
    let mut schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    schema.hidden_size = hidden_size;
    schema.num_layers = num_layers;
    schema.num_attention_heads = num_heads;
    schema.num_key_value_heads = num_kv_heads;
    schema.vocabulary_size = vocab_size;

    if let Err(e) = schema.validate_architecture() {
        eprintln!("WARNING: architecture validation: {}", e);
    }

    // ── Read Safetensors index ────────────────────────────────────
    let st_paths = [
        dir.join("model.safetensors.index.json"),
        dir.join("model.safetensors"),
    ];

    let mut tensor_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for st_path in &st_paths {
        if !st_path.exists() {
            continue;
        }
        if st_path.extension().map_or(false, |e| e == "json") {
            // Index file
            let index: serde_json::Value =
                serde_json::from_reader(BufReader::new(fs::File::open(st_path).unwrap()))
                    .unwrap_or_default();

            if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                for (name, _file) in weight_map {
                    // We don't have shapes in the index — need to read header of each shard
                    tensor_shapes.entry(name.clone()).or_insert(Vec::new());
                }
            }
        } else {
            // Single safetensors file — read header
            let file = fs::File::open(st_path).unwrap();
            let mut reader = BufReader::new(file);
            // Read 8-byte header size (little-endian u64)
            let mut header_len_bytes = [0u8; 8];
            use std::io::Read;
            reader.read_exact(&mut header_len_bytes).unwrap();
            let header_len = u64::from_le_bytes(header_len_bytes) as usize;
            let mut header_json = vec![0u8; header_len];
            reader.read_exact(&mut header_json).unwrap();
            let header: serde_json::Value =
                serde_json::from_slice(&header_json).unwrap_or_default();
            if let Some(tensors) = header.as_object() {
                for (name, info) in tensors {
                    if let Some(shape) = info.get("shape").and_then(|s| s.as_array()) {
                        let dims: Vec<usize> = shape
                            .iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect();
                        tensor_shapes.insert(name.clone(), dims);
                    }
                }
            }
        }
        break; // Use first available source
    }

    println!("  Found {} tensors", tensor_shapes.len());

    // ── Classify tensors ──────────────────────────────────────────
    let mut classification: HashMap<String, (usize, u64)> = HashMap::new();
    let mut tensor_list: Vec<serde_json::Value> = Vec::new();
    let mut unknown_large: Vec<serde_json::Value> = Vec::new();

    // Reject legacy vision tower
    let names: Vec<String> = tensor_shapes.keys().cloned().collect();
    if let Err(e) = schema.reject_legacy_vision_tower(&names) {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }

    for (name, shape) in &tensor_shapes {
        let cls = classify_tensor_name(name);
        let param_count: u64 = shape.iter().map(|&d| d as u64).product();

        let entry = classification
            .entry(cls.as_str().to_string())
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += param_count;

        tensor_list.push(serde_json::json!({
            "name": name,
            "shape": shape,
            "classification": cls.as_str(),
            "param_count": param_count,
        }));

        if cls == TensorClassification::Unknown && param_count > UNKNOWN_PARAM_THRESHOLD {
            unknown_large.push(serde_json::json!({
                "name": name,
                "shape": shape,
                "param_count": param_count,
            }));
        }
    }

    // Print classification summary
    println!("\n  Classification:");
    for (cls, (count, params)) in &classification {
        println!("    {}: {} tensors, {} params", cls, count, params);
    }

    // Fail on large unknowns
    if !unknown_large.is_empty() {
        eprintln!(
            "\nERROR: {} unknown tensor(s) exceed {} parameter threshold:",
            unknown_large.len(),
            UNKNOWN_PARAM_THRESHOLD
        );
        for t in &unknown_large {
            eprintln!("  {} ({:?})", t["name"], t["shape"]);
        }
        std::process::exit(1);
    }

    // ── Emit inventory ────────────────────────────────────────────
    let inventory = serde_json::json!({
        "model_revision": "",
        "total_tensors": tensor_shapes.len(),
        "classification": classification.iter().map(|(k, (c, p))| {
            (k, serde_json::json!({"count": c, "total_params": p}))
        }).collect::<std::collections::BTreeMap<_, _>>(),
        "tensors": tensor_list,
        "unknown_large": unknown_large,
    });

    // ── Build processor contract ──────────────────────────────────
    let tokenizer_path = dir.join("tokenizer_config.json");
    let tokenizer: serde_json::Value = if tokenizer_path.exists() {
        serde_json::from_reader(BufReader::new(fs::File::open(&tokenizer_path).unwrap()))
            .unwrap_or_default()
    } else {
        serde_json::Value::Null
    };

    let processor_path = dir.join("processor_config.json");
    let processor: serde_json::Value = if processor_path.exists() {
        serde_json::from_reader(BufReader::new(fs::File::open(&processor_path).unwrap()))
            .unwrap_or_default()
    } else {
        serde_json::Value::Null
    };

    let contract = serde_json::json!({
        "text": {
            "vocabulary_size": vocab_size,
            "bos_token_id": tokenizer.get("bos_token_id"),
            "eos_token_id": tokenizer.get("eos_token_id"),
            "pad_token_id": tokenizer.get("pad_token_id"),
        },
        "image": {
            "patch_size": processor.get("patch_size").or(processor.get("image_patch_size")),
            "soft_token_default": processor.get("soft_tokens").or(processor.get("default_soft_tokens")),
        },
        "audio": {
            "sample_rate": processor.get("sampling_rate"),
        },
    });

    // ── Write output files ────────────────────────────────────────
    for emit_path in &emit_paths {
        let path = Path::new(emit_path);
        let file_stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        match file_stem {
            "tensor_inventory" | "inventory" | "gemma4_tensor_inventory" => {
                fs::write(path, serde_json::to_string_pretty(&inventory).unwrap()).unwrap();
                println!("  Wrote tensor inventory to {}", emit_path);
            }
            "processor_contract" | "contract" | "gemma4_processor_contract" => {
                fs::write(path, serde_json::to_string_pretty(&contract).unwrap()).unwrap();
                println!("  Wrote processor contract to {}", emit_path);
            }
            _ => {
                // Generic: write inventory
                fs::write(path, serde_json::to_string_pretty(&inventory).unwrap()).unwrap();
                println!("  Wrote to {}", emit_path);
            }
        }
    }

    println!("\nDone.");
}
