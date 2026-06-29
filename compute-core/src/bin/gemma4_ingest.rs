//! gemma4_ingest — Stage 1+2: Download Gemma 4 12B from HF, ternary quantize,
//! compile to .cimage, all in Rust, zero Python.
//!
//! Usage:
//!   cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B --output gemma4_12b.cimage
//!   cargo run --bin gemma4_ingest -- --local-dir ./gemma4-12B --output gemma4_12b.cimage

#![allow(unused_imports)]

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tribunus_compute_core::compute_image::compile::ternary::TernaryCImageCompiler;

// ── Gemma 4 12B architecture constants ──────────────────────────────
const NUM_LAYERS: usize = 48;
const HIDDEN_DIM: usize = 3584;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 224;
const FFN_INTERMEDIATE: usize = 14336;

/// (serialized_name, rows, cols)
const MATRICES: &[(&str, usize, usize)] = &[
    ("self_attn.q_proj.weight",  HIDDEN_DIM, NUM_HEADS * HEAD_DIM),
    ("self_attn.k_proj.weight",  HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM),
    ("self_attn.v_proj.weight",  HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM),
    ("self_attn.o_proj.weight",  NUM_HEADS * HEAD_DIM, HIDDEN_DIM),
    ("mlp.gate_proj.weight",     HIDDEN_DIM, FFN_INTERMEDIATE),
    ("mlp.up_proj.weight",       HIDDEN_DIM, FFN_INTERMEDIATE),
    ("mlp.down_proj.weight",     FFN_INTERMEDIATE, HIDDEN_DIM),
];

// ── FP16 conversion ─────────────────────────────────────────────────

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 { return sign; }
    if exp == 0xFF {
        return if mant == 0 {
            if sign != 0 { 0xFC00 } else { 0x7C00 }
        } else { 0x7E00 };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if sign != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 { return sign; }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert a stream of f32 bytes to fp16 scale + 2-bit nibbles.
fn quantize_block(values: &[f32; 256]) -> (u16, [u8; 64]) {
    let max_mag = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let scale = if max_mag > 1e-12 { max_mag } else { 1.0 };
    let scale_fp16 = f32_to_fp16_bits(scale);

    let mut nibbles = [0u8; 64];
    for (i, chunk) in values.chunks_exact(4).enumerate() {
        let mut byte: u8 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            let snap = (v / scale).round().clamp(-1.0, 1.0) as i8;
            let nibble = match snap {
                1 => 0b01u8,
                -1 => 0b10u8,
                _ => 0b00u8,
            };
            byte |= nibble << (j * 2);
        }
        nibbles[i] = byte;
    }

    (scale_fp16, nibbles)
}

/// Process a flat weight array in 256-element blocks, append scales + nibbles.
fn process_weights(
    weights_f32: &[f32],
    scales_out: &mut Vec<u8>,
    weights_out: &mut Vec<u8>,
) {
    let padded = if weights_f32.len() % 256 == 0 {
        weights_f32.to_vec()
    } else {
        let n = ((weights_f32.len() + 255) / 256) * 256;
        let mut v = weights_f32.to_vec();
        v.resize(n, 0.0);
        v
    };

    for block in padded.chunks_exact(256) {
        let arr: [f32; 256] = {
            let mut b = [0.0f32; 256];
            b.copy_from_slice(block);
            b
        };
        let (scale, nibbles) = quantize_block(&arr);
        scales_out.extend_from_slice(&scale.to_le_bytes());
        weights_out.extend_from_slice(&nibbles);
    }
}

/// Read tensor bytes (f32 or bf16) into a Vec<f32>.
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let u = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((u as u32) << 16)
                })
                .collect()
        }
        _ => panic!("unsupported dtype: {:?}", dtype),
    }
}

/// Build a tensor key for a given layer and matrix name.
fn tensor_key(layer: usize, matrix_short: &str) -> String {
    let prefix = format!("model.language_model.layers.{layer}");
    if matrix_short.starts_with("self_attn.") {
        format!("{prefix}.{matrix_short}")
    } else {
        format!("{prefix}.mlp.{}", matrix_short.strip_prefix("mlp.").unwrap_or(matrix_short))
    }
}

// ── Entry point ─────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gemma 4 12B Unified → Ternary .cimage                     ║");
    println!("║  AOT Compiler (pure Rust, no Python)                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let repo = get_opt(&args, "--repo");
    let local_dir = get_opt(&args, "--local-dir");
    let output = get_opt(&args, "--output").unwrap_or("gemma4_12b.cimage");
    let mil_program = get_opt(&args, "--mil");

    // Validate args
    if repo.is_none() && local_dir.is_none() {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin gemma4_ingest -- --repo google/gemma-4-12B --output gemma4_12b.cimage");
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

    let mut all_scales = Vec::new();
    let mut all_weights = Vec::new();
    let mut total_elements: usize = 0;

    // Non-layer weights
    let non_layer_keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.lm_head.weight",
    ];

    for key in &non_layer_keys {
        if let Some((data, shape)) = load_tensor(key, &shard_paths) {
            let _rows = if shape.len() >= 2 { shape[0] } else { 1 };
            let _cols = if shape.len() >= 2 { shape[1] } else { data.len() };
            process_weights(&data, &mut all_scales, &mut all_weights);
            total_elements += data.len();
            // Simpler: just print the tensor shape
            let n_blocks = (data.len() + 255) / 256;
            let scale_kb = n_blocks as f64 * 2.0 / 1024.0;
            let weight_kb = n_blocks as f64 * 64.0 / 1024.0;
            print!("\r     {n_blocks:>6} blocks, {scale_kb:.1} KB scales, {weight_kb:.1} KB nibbles\n");
        } else {
            println!("  {key:<40} NOT FOUND (skipping)");
        }
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
                // Try alternate key format
                let alt_key = format!("model.language_model.layers.{layer}.{mat_name}");
                if let Some((data, _)) = load_tensor(&alt_key, &shard_paths) {
                    process_weights(&data, &mut all_scales, &mut all_weights);
                    total_elements += data.len();
                } else {
                    println!("\n  WARNING: {key} not found");
                }
            }
        }

        if layer % 8 == 7 {
            let mb = (all_scales.len() + all_weights.len()) as f64 / (1024.0 * 1024.0);
            println!(" — {mb:.1} MB");
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
        for t in &mtp_tensors { println!("    {t}"); }
    } else {
        println!("  No MTP heads found (model may not have them)");
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
        total_elements, quant_elapsed);
    println!("  {} blocks, {:.1} MB scales, {:.1} MB nibbles",
             n_blocks, mb_scales, mb_weights);

    // ── Step 3: Load MIL program ───────────────────────────────
    println!("\n  ── Compiling .cimage ────────────────────────────────");

    let mil_bytes = if let Some(mil_path) = mil_program {
        std::fs::read(mil_path).unwrap_or_else(|e| {
            eprintln!("  WARNING: can't read {mil_path}: {e}, using placeholder");
            generate_placeholder_mil()
        })
    } else {
        generate_placeholder_mil()
    };

    // ── Step 4: Build .cimage ──────────────────────────────────
    let compiler = TernaryCImageCompiler::new(
        mil_bytes,
        all_scales,
        all_weights,
        total_elements,
        NUM_LAYERS,
    );

    let (cimage_bytes, _layout) = compiler.write_to_file(output).unwrap_or_else(|e| {
        eprintln!("  ERROR: write .cimage failed: {e}");
        std::process::exit(1);
    });

    // ── Step 5: Verify ─────────────────────────────────────────
    match tribunus_compute_core::compute_image::compile::ternary::verify_prism_cimage(&cimage_bytes) {
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
    println!("  File size:  {:.1} MB ({file_size} bytes)", file_size as f64 / (1024.0 * 1024.0));
    println!("  Params:     {total_elements}");
    println!("  Blocks:     {n_blocks}");
    println!("  Compressed: {compression_ratio:.1}× vs FP16");
    println!("  Time:       {total_elapsed:.1?}");
    println!();
    println!("  ▶ Runtime ready: tribunus-compute-image load --cimage {output}");

    // ── Step 6: Compile ANE compaction model ─────────────────────
    println!("\n  ── Compiling ANE compaction model ─────────────────");
    compile_compaction_model(output);
}

// ── Safetensors loading helpers ─────────────────────────────────────

fn load_tensor(key: &str, shards: &[(PathBuf, Vec<u8>)])
    -> Option<(Vec<f32>, Vec<usize>)>
{
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
        if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
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
    args.windows(2)
        .find(|w| w[0] == key)
        .map(|w| w[1].as_str())
}

/// Compile ANE compaction gather model at build time.
/// Writes compiled .mlmodelc to {output}.compaction.modelc
fn compile_compaction_model(output: &str) {
    use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
    use tribunus_compute_core::mil_builder::MilBuilder;
    use tribunus_compute_core::mlpackage::{self, ModelMeta};
    use coreml_proto::proto::mil_spec;

    let target_count = 20480u32;
    let c = tribunus_compute_core::compute_image::compaction::align_dim(8 * 512, 2) as i64;
    let s_in = tribunus_compute_core::compute_image::compaction::align_dim(2048, 2) as i64;
    let s_out = tribunus_compute_core::compute_image::compaction::align_dim(target_count, 2) as i64;
    let tc = target_count as i64;

    let b = MilBuilder::new("main")
        .input("key_cache", mil_spec::DataType::Float16, &[1, c, 1, s_in])
        .input("value_cache", mil_spec::DataType::Float16, &[1, c, 1, s_in])
        .input("indices", mil_spec::DataType::Int32, &[tc])
        .gather("key_cache", "indices", 3)
        .gather("value_cache", "indices", 3);
    let b = b.output("gather_0").output("gather_1");
    let prog = b.build().expect("compaction MIL build");

    let meta = ModelMeta {
        model_name: "ane_compaction".into(),
        function_name: "main".into(),
        short_description: "ANE KV compaction gather".into(),
        version: "1.0.0".into(),
        author: "Prism Engine".into(),
        output_name: "gather_0".into(),
        inputs: vec![
            ("key_cache".into(), vec![1, c, 1, s_in]),
            ("value_cache".into(), vec![1, c, 1, s_in]),
            ("indices".into(), vec![tc]),
        ],
        outputs: vec![
            ("gather_0".into(), vec![1, c, 1, s_out]),
            ("gather_1".into(), vec![1, c, 1, s_out]),
        ],
    };

    let tmp = std::env::temp_dir().join("gemma4_ingest_compaction");
    let _ = std::fs::remove_dir_all(&tmp);
    let pkg_path = mlpackage::write_mlpackage(prog, &tmp, &meta)
        .expect("write compaction mlpackage");
    let receipt = compile_mlpackage(&pkg_path, &tmp, "ane_compaction", "cpuAndNeuralEngine", "macOS26")
        .expect("compile compaction model");
    let modelc_path = std::path::Path::new(&receipt.compiled_modelc_path);
    // Copy the .mlmodelc directory to a stable location
    let dest = format!("{}.compaction.modelc", output);
    let _ = std::fs::remove_dir_all(&dest);
    fn copy_dir(src: &std::path::Path, dst: &std::path::Path) {
        std::fs::create_dir_all(dst).unwrap();
        for entry in std::fs::read_dir(src).unwrap() {
            let e = entry.unwrap();
            let from = e.path();
            let to = dst.join(e.file_name());
            if from.is_dir() { copy_dir(&from, &to); }
            else { std::fs::copy(&from, &to).unwrap(); }
        }
    }
    copy_dir(modelc_path, std::path::Path::new(&dest));
    println!("  ANE compaction model: {} -> {}", modelc_path.display(), dest);
}
