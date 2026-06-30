//! AOT palette compiler with universal dequantization for Prism Engine.
//!
//! Takes a `ModelGraph`, iterates every `PalettizedMatmul` node, loads
//! weights in any format (F32/BF16/F16/U32 block-quantized), runs k-means
//! per row, builds split-block payloads, and writes a `.cimage` file.

use serde_json::json;
use std::fs::File;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::compute_image::compile::try_q8_0_ternary_pack_gpu;
use crate::config::build_execution_plan;
use crate::config::parse_config;
use crate::config_namespace::resolve_namespace;
use crate::lut::graph::{ModelGraph, TensorBlueprint};
use crate::quantization::cimage::CImageWriter;
use crate::quantization::palette::palettize_matrix;

pub struct CompiledTensor {
    pub key: String,
    pub dim_m: u32,
    pub dim_n: u32,
    pub payload: Vec<u8>,
    pub effective_bpp: f32,
}

/// Compile an entire model into a `.cimage` file.
pub fn compile_to_cimage(
    graph: &ModelGraph,
    safetensors_dir: &Path,
    output_path: &Path,
    config_path: &Path,
) -> Result<(), String> {
    let mut cimage = CImageWriter::new(output_path)?;
    let pal_tensors = graph.palettized_tensors();

    let shards = discover_safetensors(safetensors_dir)?;

    for tb in &pal_tensors {
        let t0 = std::time::Instant::now();
        let f32_vals = load_weight_f32(&shards, tb)?;
        let out_dim = tb.dim_m as usize;
        let in_dim = tb.dim_n as usize;

        eprint!("  [prism] {} ({}×{})... ", tb.key, out_dim, in_dim);

        let pal = palettize_matrix(&f32_vals, out_dim, in_dim, 16, 50);
        let bpp = pal.effective_bpp();

        let cb_bytes = pal.rows.len() * 16 * 2;
        let idx_bytes: usize = pal.rows.iter().map(|r| r.indices.len()).sum();
        let mut payload = Vec::with_capacity(cb_bytes + idx_bytes);
        for row in &pal.rows {
            for &cb_f32 in &row.codebook {
                let cb_f16 = half::f16::from_f32(cb_f32);
                payload.extend_from_slice(&cb_f16.to_bits().to_le_bytes());
            }
        }
        for row in &pal.rows {
            payload.extend_from_slice(&row.indices);
        }

        cimage.append_palettized(&tb.key, &payload, tb.dim_m, tb.dim_n)?;

        let elapsed = t0.elapsed();
        eprintln!("bpp={bpp:.3} {:.2}s", elapsed.as_secs_f64());
    }

    // Build and embed heterogeneous execution plan
    if let Ok(plan_json) = build_execution_plan_json(config_path, safetensors_dir) {
        cimage.set_execution_plan(plan_json);
    }

    cimage.finalize()?;
    eprintln!("[prism:compile] Done -> {}", output_path.display());
    Ok(())
}

/// Build the execution plan as a JSON string for embedding in the CImage.
fn build_execution_plan_json(config_path: &Path, _weights_dir: &Path) -> Result<String, String> {
    let (arch, _quant, _manifest) = parse_config(
        &config_path.to_string_lossy()
    ).map_err(|e| format!("config parse: {e}"))?;

    // Collect tensor names for namespace discovery
    let mut tensor_names: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(_weights_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                if let Ok(data) = std::fs::read(&path) {
                    if data.len() >= 8 {
                        let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap_or([0; 8]));
                        if header_len > 0 && (8 + header_len as usize) <= data.len() {
                            if let Ok(header) = serde_json::from_slice::<serde_json::Value>(&data[8..8 + header_len as usize]) {
                                if let Some(obj) = header.as_object() {
                                    tensor_names.extend(obj.keys().cloned());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    tensor_names.sort();

    let namespace = match resolve_namespace(&tensor_names) {
        Some(ns) => ns,
        None => return Err("could not resolve model namespace".into()),
    };

    let mut emitted_ids: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (i, name) in tensor_names.iter().enumerate() {
        emitted_ids.insert(name.clone(), i as u32);
    }

    let mut execution_plan = build_execution_plan(&arch, &namespace, &emitted_ids);
    execution_plan.apply_fusion_pass();

    serde_json::to_string(&execution_plan)
        .map_err(|e| format!("serialize execution plan: {e}"))
}

/// Compile to memory (no .cimage I/O).
pub fn compile_to_memory(
    graph: &ModelGraph,
    safetensors_dir: &Path,
) -> Result<HashMap<String, CompiledTensor>, String> {
    let shards = discover_safetensors(safetensors_dir)?;
    let mut results = HashMap::new();

    for tb in graph.palettized_tensors() {
        let f32_vals = load_weight_f32(&shards, tb)?;
        let out_dim = tb.dim_m as usize;
        let in_dim = tb.dim_n as usize;
        let pal = palettize_matrix(&f32_vals, out_dim, in_dim, 16, 50);
        let bpp = pal.effective_bpp();

        let cb_bytes = pal.rows.len() * 16 * 2;
        let idx_bytes: usize = pal.rows.iter().map(|r| r.indices.len()).sum();
        let mut payload = Vec::with_capacity(cb_bytes + idx_bytes);
        for row in &pal.rows {
            for &cb_f32 in &row.codebook {
                let cb_f16 = half::f16::from_f32(cb_f32);
                payload.extend_from_slice(&cb_f16.to_bits().to_le_bytes());
            }
        }
        for row in &pal.rows {
            payload.extend_from_slice(&row.indices);
        }

        results.insert(
            tb.key.clone(),
            CompiledTensor {
                key: tb.key.clone(),
                dim_m: tb.dim_m,
                dim_n: tb.dim_n,
                payload,
                effective_bpp: bpp as f32,
            },
        );
    }

    Ok(results)
}

// ── Safetensors helpers ─────────────────────────────────────────────────

fn discover_safetensors(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        if entry
            .path()
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            shards.push(entry.path());
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(format!("No .safetensors files in {}", dir.display()));
    }
    Ok(shards)
}

fn load_weight_f32(
    shards: &[std::path::PathBuf],
    tb: &TensorBlueprint,
) -> Result<Vec<f32>, String> {
    for shard_path in shards {
        let data =
            std::fs::read(shard_path).map_err(|e| format!("read {}: {e}", shard_path.display()))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| format!("parse {}: {e}", shard_path.display()))?;
        if let Ok(view) = tensors.tensor(&tb.key) {
            return tensor_to_f32(&tensors, &view, &tb.key);
        }
    }
    Err(format!("Tensor {} not found in any shard", tb.key))
}

/// Universal tensor-to-f32: handles F32, BF16, F16 natively,
/// and U32 block-quantized (NF4/INT4/INT8) via dequantization.
fn tensor_to_f32(
    tensors: &safetensors::SafeTensors<'_>,
    view: &safetensors::tensor::TensorView<'_>,
    key: &str,
) -> Result<Vec<f32>, String> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        Dtype::BF16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        Dtype::U32 => dequantize_mlx_block(tensors, key, view),
        _ => Err(format!("unsupported dtype {:?} for {}", view.dtype(), key)),
    }
}

/// NF4 exact quantile table (information-theoretic NormalFloat4).
const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39490527,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.07958029,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Dequantize U32 block-quantized weights (MLX/AF8/NF4 format).
///
/// Reads sibling `.scales` and `.biases` tensors recursively (handles F16/BF16),
/// then decodes packed U32 values back into f32 using the scale/bias per group.
fn dequantize_mlx_block(
    tensors: &safetensors::SafeTensors<'_>,
    key: &str,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<Vec<f32>, String> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    let scales_key = format!("{base}.scales");
    let biases_key = format!("{base}.biases");

    let packed: Vec<u32> = view
        .data()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Recursively load scales/biases
    let sv = tensors
        .tensor(&scales_key)
        .map_err(|_| format!("missing {scales_key}"))?;
    let scales = tensor_to_f32(tensors, &sv, &scales_key)?;
    let biases = match tensors.tensor(&biases_key) {
        Ok(bv) => tensor_to_f32(tensors, &bv, &biases_key)?,
        _ => vec![0.0; scales.len()],
    };

    let logical_n: usize = view.shape().iter().product();
    let group_size = logical_n / scales.len().max(1);
    let elements_per_word = if packed.len() > 0 {
        logical_n / packed.len()
    } else {
        8
    };
    let is_4bit = elements_per_word >= 8;

    let mut decoded = Vec::with_capacity(logical_n);
    let mut si = 0;
    let mut gc = 0usize;

    if is_4bit {
        for w in &packed {
            for i in 0..8 {
                let nibble = (*w >> (i * 4)) & 0x0F;
                let v = if key.contains("nf4") {
                    (NF4_LUT[nibble as usize] * scales[si]) + biases[si]
                } else {
                    ((nibble as f32) * scales[si]) + biases[si]
                };
                decoded.push(v);
                gc += 1;
                if gc >= group_size {
                    gc = 0;
                    si += 1;
                }
            }
        }
    } else {
        for w in &packed {
            for i in 0..4 {
                let byte = (*w >> (i * 8)) & 0xFF;
                decoded.push(((byte as f32) * scales[si]) + biases[si]);
                gc += 1;
                if gc >= group_size {
                    gc = 0;
                    si += 1;
                }
            }
        }
    }
    Ok(decoded)
}

// ── GGUF compilation ───────────────────────────────────────────────────

#[cfg(feature = "prism-backend")]
/// Compile a GGUF model file directly to a .cimage palettized format.
///
/// Parses the GGUF header, maps tensor names to HuggingFace-style conventions,
/// builds an execution graph, dequantizes weights (GGML → f32) and palettizes
/// each weight matrix via k-means clustering.
pub fn compile_gguf_to_cimage(
    gguf_path: &Path,
    output_path: &Path,
) -> Result<(), String> {
    use crate::gguf;

    // 1. Parse GGUF header → metadata + tensor inventory + architecture
    eprintln!("[gguf] parsing header...");
    let (metadata, tensors) = gguf::parse_gguf_header(gguf_path)?;
    let arch = gguf::extract_architecture(&metadata)?;
    eprintln!(
        "[gguf] arch={} layers={} hidden={}",
        arch.model_type, arch.num_hidden_layers, arch.hidden_size
    );

    // 2. Write a config.json to a temp directory for ModelGraph construction
    let tmp_dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let config_path = tmp_dir.path().join("config.json");
    // The GGUF metadata may omit head_dim. Compute it from the first layer's
    // Q projection tensor shape: Q_out = num_heads * head_dim → head_dim = Q_out / num_heads.
    let mut arch = arch;
    // Also infer num_key_value_heads from K projection shape.
    if let Some(q_tensor) = tensors.iter().find(|t| t.name.ends_with("attn_q.weight")) {
        if q_tensor.shape.len() >= 2 {
            let q_out = q_tensor.shape[1] as u32;
            let inferred = q_out / arch.num_attention_heads;
            if inferred > 0 && inferred != arch.head_dim {
                eprintln!("[gguf] inferred head_dim={inferred} from {}(out={q_out}, heads={})",
                    q_tensor.name, arch.num_attention_heads);
                arch.head_dim = inferred;
            }
        }
    }
    // Infer num_key_value_heads from the first K projection tensor shape.
    // The GGUF metadata stores head_count_kv as an array (variable per layer).
    if let Some(k_tensor) = tensors.iter().find(|t| t.name.ends_with("attn_k.weight")) {
        if k_tensor.shape.len() >= 2 {
            let k_out = k_tensor.shape[1] as u32;
            let inferred_kv = k_out / arch.head_dim;
            if inferred_kv > 0 && inferred_kv != arch.num_key_value_heads {
                eprintln!("[gguf] inferred kv_heads={inferred_kv} from {}(out={k_out}, head_dim={})",
                    k_tensor.name, arch.head_dim);
                arch.num_key_value_heads = inferred_kv;
    }
    }
    }
    write_gguf_config_json(&config_path, &arch, &metadata)?;

    // 3. Build the ModelGraph from the config
    let unified = crate::lut::graph::UnifiedConfig::from_file(&config_path)?;
    let graph = crate::lut::graph::ModelGraph::build(&unified);
    eprintln!(
        "[gguf] graph: {} layers, {} nodes",
        graph.num_layers,
        graph.nodes.len()
    );

    // 4. Build HF-name → GGUF-tensor map and collect all HF tensor names
    let arch_type = gguf::meta_str(&metadata, "general.architecture").unwrap_or("unknown");
    // Use indexes into the tensors vec since GgufTensorMeta is not Clone
    let mut hf_to_tensor_idx: HashMap<String, usize> = HashMap::new();
    let mut all_hf_names: Vec<String> = Vec::new();
    for (idx, t) in tensors.iter().enumerate() {
        if let Some(hf_name) = gguf::gguf_name_to_hf_name(&t.name, arch_type) {
            hf_to_tensor_idx.insert(hf_name.clone(), idx);
            all_hf_names.push(hf_name);
        }
    }
    all_hf_names.sort();

    eprintln!("[gguf] mapped {}/{} tensors to HF names", hf_to_tensor_idx.len(), tensors.len());

    // 5. Resolve namespace from HF-style names
    let namespace = resolve_namespace(&all_hf_names).ok_or_else(|| {
        format!(
            "could not resolve model namespace from {} mapped tensor names",
            all_hf_names.len()
        )
    })?;
    eprintln!("[gguf] namespace: {} (root={})", namespace.discovery, namespace.root);

    // 6. Compile each palettized tensor
    let mut cimage = CImageWriter::new(output_path)?;
    let pal_tensors = graph.palettized_tensors();
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    for (id, tb) in pal_tensors.iter().enumerate() {
        let t_idx = hf_to_tensor_idx.get(&tb.key).ok_or_else(|| {
            format!(
                "Tensor '{}' not found in GGUF file (mapped from graph key)",
                tb.key
            )
        })?;
        let meta = &tensors[*t_idx];

        // Verify shape consistency: GGUF stores [out_features, in_features]
        // GGUF stores [in_features, out_features]; graph expects [dim_m, dim_n] = [out, in]
        let gguf_in = meta.shape.first().copied().unwrap_or(1) as u32;
        let gguf_out = meta.shape.get(1).copied().unwrap_or(1) as u32;
        // Use the GGUF file's actual dimensions — they are the source of truth.
        // The graph's expected dims (from ModelGraph) assume uniform per-layer
        // projection sizes, but Gemma 4 shared KV layers vary (kv_heads=1 every
        // 5th layer doubles Q, shrinks K/V). The ANE path handles per-layer
        // shapes via its own fixed-shape contracts.
        let use_dim_m = gguf_out;
        let use_dim_n = gguf_in;
        if gguf_in != tb.dim_n || gguf_out != tb.dim_m {
            eprintln!("  [gguf] shape adjusted: {} graph [{}×{}] → GGUF [{gguf_in}×{gguf_out}]",
                meta.name, tb.dim_m, tb.dim_n);
        }

        // Read and dequantize the GGUF tensor to f32
        // Read raw Q8_0 bytes from the GGUF file (mmap'd at byte_offset)
        let raw_q8 = {
            let f = File::open(gguf_path)
                .map_err(|e| format!("Couldn't open GGUF: {e}"))?;
            let mmap = unsafe { memmap2::Mmap::map(&f) }
                .map_err(|e| format!("mmap: {e}"))?;
            let start = meta.byte_offset as usize;
            let end = start + meta.byte_size as usize;
            mmap[start..end].to_vec()
        };

        // Try GPU-accelerated Q8_0 → ternary tile640 pack.
        // If Metal is unavailable or the kernel fails, fall back to
        // CPU f32 dequant → transpose → palettize.
        let t0 = std::time::Instant::now();

        let gpu_result = try_q8_0_ternary_pack_gpu(&raw_q8, gguf_in, gguf_out);

        match gpu_result {
            Some((packed_u32, scales_f32, num_tiles)) => {
                // GPU succeeded: write ternary-packed data as raw u32 + f32 scales.
                let scales_name = format!("{}.scales", tb.key.trim_end_matches(".weight"));
                cimage.append_fp16(&tb.key, &packed_u32, use_dim_m, num_tiles * 32)?;
                cimage.append_fp16(&scales_name, &scales_f32, use_dim_m, num_tiles)?;
                emitted_ids.insert(tb.key.clone(), id as u32);
                eprintln!(
                    "  [gguf:gpu] {} ({}×{}) → ternary {} tiles {:.2}s",
                    meta.name, use_dim_m, use_dim_n, num_tiles,
                    t0.elapsed().as_secs_f64()
                );
            }
            None => {
                // GPU unavailable — fall back to CPU dequant + palettization
                let mut f32_vals = gguf::read_gguf_tensor_f32(gguf_path, meta)?;
                if use_dim_m > 1 && use_dim_n > 1 && f32_vals.len() > 1 {
                    let (d_in, d_out) = (gguf_in as usize, gguf_out as usize);
                    let mut t = vec![0.0f32; f32_vals.len()];
                    for i in 0..d_in {
                        let src_row_off = i * d_out;
                        for j in 0..d_out {
                            t[j * d_in + i] = f32_vals[src_row_off + j];
                        }
                    }
                    f32_vals = t;
                }
                let pal = palettize_matrix(&f32_vals, use_dim_m as usize, use_dim_n as usize, 16, 50);
                let bpp = pal.effective_bpp();
                let cb_bytes = pal.rows.len() * 16 * 2;
                let idx_bytes: usize = pal.rows.iter().map(|r| r.indices.len()).sum();
                let mut payload = Vec::with_capacity(cb_bytes + idx_bytes);
                for row in &pal.rows {
                    for &cb_f32 in &row.codebook {
                        let cb_f16 = half::f16::from_f32(cb_f32);
                        payload.extend_from_slice(&cb_f16.to_bits().to_le_bytes());
                    }
                }
                for row in &pal.rows {
                    payload.extend_from_slice(&row.indices);
                }
                cimage.append_palettized(&tb.key, &payload, use_dim_m, use_dim_n)?;
                emitted_ids.insert(tb.key.clone(), id as u32);
                eprintln!(
                    "  [gguf:cpu] {} ({}×{}) bpp={bpp:.3} {:.2}s",
                    meta.name, use_dim_m, use_dim_n,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }

    // 7. Build and embed execution plan
    let mut execution_plan = build_execution_plan(&arch, &namespace, &emitted_ids);
    execution_plan.apply_fusion_pass();
    let plan_json =
        serde_json::to_string(&execution_plan).map_err(|e| format!("serialize plan: {e}"))?;
    cimage.set_execution_plan(plan_json);

    // 8. Finalize .cimage
    cimage.finalize()?;
    eprintln!("[gguf:compile] Done -> {}", output_path.display());
    Ok(())
}

/// Write a HuggingFace-style config.json from GGUF metadata.
/// Used to build the ModelGraph and execution plan.
#[cfg(feature = "prism-backend")]
fn write_gguf_config_json(
    path: &Path,
    arch: &crate::config::TextArchitecture,
    _metadata: &[(String, String)],
) -> Result<(), String> {
    // Determine architectures field from model_type
    let architecture_name = match arch.model_type.as_str() {
        "gemma4" => "Gemma4ForCausalLM",
        "gemma" | "gemma2" => "GemmaForCausalLM",
        "llama" => "LlamaForCausalLM",
        "mistral" => "MistralForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" | "qwen3_5" => "Qwen3_5ForCausalLM",
        _ => "LlamaForCausalLM",
    };

    let config = json!({
        "architectures": [architecture_name],
        "model_type": arch.model_type,
        "hidden_size": arch.hidden_size,
        "intermediate_size": arch.intermediate_size,
        "num_attention_heads": arch.num_attention_heads,
        "num_key_value_heads": arch.num_key_value_heads,
        "head_dim": arch.head_dim,
        "num_hidden_layers": arch.num_hidden_layers,
        "vocab_size": arch.vocab_size,
        "max_position_embeddings": arch.max_position_embeddings,
        "rms_norm_eps": arch.rms_norm_eps,
        "tie_word_embeddings": arch.tie_word_embeddings,
        "rope_theta": arch.rope_local.theta,
        "attention_k_eq_v": arch.attention_k_eq_v,
        "sliding_window": arch.sliding_window,
    });

    let json_str =
        serde_json::to_string_pretty(&config).map_err(|e| format!("serialize config: {e}"))?;
    let mut f = fs::File::create(path).map_err(|e| format!("create config.json: {e}"))?;
    f.write_all(json_str.as_bytes())
        .map_err(|e| format!("write config.json: {e}"))?;
    Ok(())
}
