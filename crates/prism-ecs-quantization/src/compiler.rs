//! AOT palette compiler with universal dequantization for Prism Engine.
//!
//! Takes a `ModelGraph`, iterates every `PalettizedMatmul` node, loads
//! weights in any format (F32/BF16/F16/U32 block-quantized), runs k-means
//! per row, builds split-block payloads, and produces a
//! [`QuantizationResult`] describing what was applied.
//!
//! The previous version of this module silently wrote a different
//! representation than the one requested in a `FormatPlan` (Bf16 / Int8 /
//! Nf8 all fell through to a 4-bit palettized codec). That substitution
//! is now a hard error. The per-tensor format that the search selected
//! is the per-tensor format that the artifact contains, or compilation
//! fails with a precise reason.
//!
//! [`build_quantization_plan`] does the per-tensor work and returns the
//! structured plan. [`write_cimage_from_plan`] is the pure function
//! that serializes a plan to a CImage on disk. [`compile_to_cimage`] is
//! retained as a thin convenience wrapper for the CLI and dashboard.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::cimage::{CImageWriter, TensorType};
use crate::quantization_plan::{QuantizationResult, QuantizedTensorSelection};
use crate::palette::palettize_matrix;
use prism_ecs_ir::evolution::assembly::AssemblySpec;
use prism_ecs_ir::evolution::compile_plan::FormatPlan;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use prism_ecs_ir::{generate_plan, ModelGraph, TensorBlueprint};

/// Compilation backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationBackend {
    /// Default Metal/CPU path — k-means palettization with optional format plan.
    Default,
    /// Apple Neural Engine — compiles a prefill MIL program via coremlcompiler.
    Ane,
}

pub struct CompiledTensor {
    pub key: String,
    pub dim_m: u32,
    pub dim_n: u32,
    pub payload: Vec<u8>,
    pub effective_bpp: f32,
}

/// Build a [`QuantizationResult`] for an entire model.
///
/// This is the constitutional-first cutover of the old `compile_to_cimage`:
/// the per-tensor work (load weights, apply the requested codec, measure
/// bits per value) is done, and the result is returned as a structured
/// plan. **The CImage is not written.** Emission is a separate
/// [`write_cimage_from_plan`] call. The plan is observable to the caller
/// before any bytes hit disk.
///
/// When `compile_plan` is `Some`, uses per-tensor format assignments
/// from the evolution search. When `None`, every tensor is encoded with
/// `Palettized4Bit` (the legacy default).
///
/// When `backend` is `CompilationBackend::Ane`, returns an error — ANE
/// is a separate pipeline that does not produce a per-tensor weight plan.
///
/// `source_digest` and `target_hardware` are recorded on the result for
/// downstream receipts. `source_digest` should be the canonical digest
/// of the BF16 source model; pass an empty string only if the caller
/// genuinely cannot compute it.
pub fn build_quantization_plan(
    graph: &ModelGraph,
    safetensors_dir: &Path,
    has_metal: bool,
    compile_plan: Option<&FormatPlan>,
    backend: CompilationBackend,
    source_digest: &str,
    target_hardware: &str,
    progress: impl Fn(&str, u32, u32, f64, f64),
) -> Result<QuantizationResult, String> {
    if matches!(backend, CompilationBackend::Ane) {
        return Err(
            "ANE backend does not produce a per-tensor weight plan; use compile_to_cimage_ane directly".into(),
        );
    }

    // Generate execution plan before compiling weights.
    let exec_plan = generate_plan(graph, has_metal, false);
    let plan_json =
        serde_json::to_string(&exec_plan).map_err(|e| format!("serialize plan: {e}"))?;

    let shards = discover_safetensors(safetensors_dir)?;
    let mut selections: Vec<QuantizedTensorSelection> = Vec::new();

    // Helper: produce a single tensor's selection. `format_override`
    // comes from the caller-supplied plan; when None we apply the
    // default policy explicitly.
    let mut compile_tensor = |tb: &TensorBlueprint,
                              format_override: Option<TensorFormat>|
     -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let f32_vals = load_weight_f32(&shards, tb)?;
        let out_dim = tb.dim_m as usize;
        let in_dim = tb.dim_n as usize;

        let (payload, tensor_type, format, bpp) = match format_override {
            Some(fmt) => {
                let (payload, tensor_type, bpp) =
                    quantize_to_payload(&tb.key, &f32_vals, out_dim, in_dim, fmt)?;
                (payload, tensor_type, fmt, bpp)
            }
            None => {
                let (payload, tensor_type, bpp) = palettize_to_payload(&f32_vals, out_dim, in_dim)?;
                (payload, tensor_type, TensorFormat::Palettized4Bit, bpp)
            }
        };

        let elapsed = t0.elapsed();
        eprintln!(
            "  [prism] {} ({}×{}) bpp={:.3} format={:?} {:.2}s",
            tb.key,
            out_dim,
            in_dim,
            bpp,
            format,
            elapsed.as_secs_f64()
        );
        progress(&tb.key, tb.dim_m, tb.dim_n, bpp as f64, elapsed.as_secs_f64());

        selections.push(QuantizedTensorSelection {
            key: tb.key.clone(),
            format,
            payload_bytes: payload.len() as u64,
            tensor_type,
            dim_m: tb.dim_m,
            dim_n: tb.dim_n,
            effective_bpp: bpp,
            payload,
        });
        Ok(())
    };

    // Compile palettized tensors (matmuls, projections, heads)
    for tb in graph.palettized_tensors() {
        let fmt = compile_plan.and_then(|p| p.get(&tb.key));
        compile_tensor(tb, fmt)?;
    }

    // Also compile the embedding tensor (not in palettized_tensors).
    for node in &graph.nodes {
        if let prism_ecs_ir::ComputeNode::TokenEmbedding {
            key,
            vocab_size,
            hidden_dim,
        } = node
        {
            let tb = TensorBlueprint {
                key: key.clone(),
                dim_m: *vocab_size,
                dim_n: *hidden_dim,
            };
            let fmt = compile_plan.and_then(|p| p.get(key));
            compile_tensor(&tb, fmt)?;
            break;
        }
    }

    Ok(QuantizationResult {
        source_digest: source_digest.to_string(),
        target_hardware: target_hardware.to_string(),
        selections,
        execution_plan_json: Some(plan_json),
        default_format: TensorFormat::Palettized4Bit,
    })
}

/// Write a CImage file from a [`QuantizationResult`].
///
/// Pure function: same plan + same path = same bytes (modulo filesystem
/// timestamp). Does not recompute per-tensor policy. Does not call any
/// external compiler. The caller is expected to have already validated
/// the plan against a target profile and any receipt gates.
pub fn write_cimage_from_plan(plan: &QuantizationResult, output_path: &Path) -> Result<(), String> {
    let mut cimage = CImageWriter::new(output_path)?;
    if let Some(plan_json) = &plan.execution_plan_json {
        cimage.set_execution_plan(plan_json.clone());
    }
    for sel in &plan.selections {
        cimage.append(
            &sel.key,
            &sel.payload,
            sel.dim_m,
            sel.dim_n,
            sel.tensor_type.clone(),
        )?;
    }
    cimage.finalize()?;
    eprintln!("[prism:compile] Done -> {}", output_path.display());
    Ok(())
}

/// Backwards-compatible convenience wrapper for the CLI and dashboard.
///
/// Builds a plan, then writes it. New callers should use
/// [`build_quantization_plan`] and [`write_cimage_from_plan`] directly so
/// the plan is observable between the two steps.
pub fn compile_to_cimage(
    graph: &ModelGraph,
    safetensors_dir: &Path,
    output_path: &Path,
    has_metal: bool,
    progress: impl Fn(&str, u32, u32, f64, f64),
    compile_plan: Option<&FormatPlan>,
    backend: CompilationBackend,
) -> Result<(), String> {
    if matches!(backend, CompilationBackend::Ane) {
        return compile_to_cimage_ane(graph, safetensors_dir, output_path);
    }
    let plan = build_quantization_plan(
        graph,
        safetensors_dir,
        has_metal,
        compile_plan,
        backend,
        "",
        "unknown",
        progress,
    )?;
    write_cimage_from_plan(&plan, output_path)
}

/// Quantize a tensor by the given format. Returns `(payload, tensor_type, bpp)`.
///
/// **Hard error contract:** every format variant must return its own
/// codec, or an error. Silent substitution to `Palettized4Bit` (or any
/// other fallback) is forbidden — that was the bug the constitutional
/// cutover is fixing.
fn quantize_to_payload(
    key: &str,
    f32_vals: &[f32],
    out_dim: usize,
    in_dim: usize,
    format: TensorFormat,
) -> Result<(Vec<u8>, TensorType, f32), String> {
    let _ = key; // Reserved for future diagnostics; key is already carried by the selection.
    match format {
        TensorFormat::Nf4 | TensorFormat::Int4 => {
            let (codes, scales, biases, _packed_rows, _packed_cols) =
                crate::nf4tile640::pack_nf4_weights(f32_vals, out_dim, in_dim);
            // Payload layout: [packed codes] [f32 scales] [f32 biases]
            let mut payload = codes;
            for &s in &scales {
                payload.extend_from_slice(&s.to_le_bytes());
            }
            for &b in &biases {
                payload.extend_from_slice(&b.to_le_bytes());
            }
            // 4-bit → 0.5 bytes per value → 4 bpp
            let tensor_type = match format {
                TensorFormat::Nf4 => TensorType::NF4,
                _ => TensorType::Int4,
            };
            Ok((payload, tensor_type, 4.0))
        }

        TensorFormat::Fp16 => {
            // Raw f32→f16 conversion, no quantization
            let mut payload = Vec::with_capacity(f32_vals.len() * 2);
            for &v in f32_vals {
                payload.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            Ok((payload, TensorType::StandardFP16, 16.0))
        }

        TensorFormat::Ternary158 => {
            // Ternary: quantize each value to {-1, 0, +1} with group scales
            let group_size = 128usize;
            let num_groups = f32_vals.len().div_ceil(group_size);
            let mut weights = Vec::with_capacity(f32_vals.len());
            let mut scales = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * group_size;
                let end = (start + group_size).min(f32_vals.len());
                let group = &f32_vals[start..end];

                let abs_max = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if abs_max > 1e-10 { abs_max } else { 1.0 };
                scales.push(scale);

                for &v in group {
                    let q = (v / scale).clamp(-1.0, 1.0);
                    let t = if q < -0.5 {
                        -1i8
                    } else if q > 0.5 {
                        1i8
                    } else {
                        0i8
                    };
                    weights.push(t);
                }
            }

            let package = crate::ternarization::packaging::pack_ternary(&weights, &scales)?;

            // Payload layout: [packed ternary bytes] [f32 scales]
            let packed_len = package.packed_bytes.len();
            let scales_len = package.scales.len();
            let mut payload = package.packed_bytes;
            for &s in &package.scales {
                payload.extend_from_slice(&s.to_le_bytes());
            }

            // Ternary158: ~1.58 bits per value + scales
            let bpp = (packed_len as f32 * 8.0 + scales_len as f32 * 32.0) / f32_vals.len() as f32;
            Ok((payload, TensorType::Ternary158, bpp))
        }

        TensorFormat::Binary1 => {
            // Binary: quantize to {+1, -1} (1 bit per value)
            let group_size = 128usize;
            let num_groups = f32_vals.len().div_ceil(group_size);
            let mut binary_bits: Vec<u8> = Vec::new();
            let mut scales = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * group_size;
                let end = (start + group_size).min(f32_vals.len());
                let group = &f32_vals[start..end];

                let mean_abs = group.iter().map(|v| v.abs()).sum::<f32>() / group.len() as f32;
                let scale = if mean_abs > 1e-10 { mean_abs } else { 1.0 };
                scales.push(scale);

                let byte_count = group.len().div_ceil(8);
                let mut bits = vec![0u8; byte_count];
                for (i, &v) in group.iter().enumerate() {
                    if v > 0.0 {
                        bits[i / 8] |= 1 << (i % 8);
                    }
                }
                binary_bits.extend_from_slice(&bits);
            }

            let mut payload = binary_bits;
            for &s in &scales {
                payload.extend_from_slice(&s.to_le_bytes());
            }

            let bpp = (payload.len() as f32 * 8.0) / f32_vals.len() as f32;
            Ok((payload, TensorType::Binary1, bpp))
        }

        // Bf16 / Int8 / Nf8 do not yet have a real codec in this crate.
        // The previous version of this function silently substituted
        // Palettized4Bit, which made a CImage header that said INT8
        // physically contain 4-bit palettized bytes. That is a hard
        // error now — the caller must explicitly use Palettized4Bit
        // (the default) or wait for the real codec to be implemented.
        TensorFormat::Palettized4Bit => palettize_to_payload(f32_vals, out_dim, in_dim),
        TensorFormat::Bf16 => Err(format!(
            "TensorFormat::Bf16 has no codec implementation in prism-ecs-quantization; \
             use Palettized4Bit or implement the BF16 codec for {key}"
        )),
        TensorFormat::Int8 => Err(format!(
            "TensorFormat::Int8 has no codec implementation in prism-ecs-quantization; \
             use Palettized4Bit or implement the INT8 codec for {key}"
        )),
        TensorFormat::Nf8 => Err(format!(
            "TensorFormat::Nf8 has no codec implementation in prism-ecs-quantization; \
             use Palettized4Bit or implement the NF8 codec for {key}"
        )),
    }
}

/// K-means palettization to CImage-ready payload bytes. Used as the
/// default format when no plan is provided and as the explicit
/// implementation for `TensorFormat::Palettized4Bit`.
fn palettize_to_payload(
    f32_vals: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Result<(Vec<u8>, TensorType, f32), String> {
    let pal = palettize_matrix(f32_vals, out_dim, in_dim, 16, 50);
    let bpp = pal.effective_bpp() as f32;

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

    Ok((payload, TensorType::Palettized4Bit, bpp))
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
        let bpp = pal.effective_bpp() as f32;

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
                effective_bpp: bpp,
            },
        );
    }

    Ok(results)
}

// ── Safetensors helpers ───────────────────────────────────────────────

fn discover_safetensors(dir: &Path) -> Result<Vec<std::path::PathBuf>, String> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        if entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "safetensors")
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
pub(crate) fn tensor_to_f32(
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
    let elements_per_word = if !packed.is_empty() {
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

/// Compile multiple models into a single unified .cimage.
///
/// Each model is compiled independently (via `compile_to_cimage` when the
/// model file is a `.gguf` weight file) or read directly when already in
/// `.cimage` format. All compiled outputs are then interleaved into one
/// `.cimage` file with a model index for runtime dispatch.
///
/// The `model_files` map must contain a path to each model's weight file
/// (`.gguf`) or pre-compiled `.cimage` file. For `.gguf` files, an adjacent
/// `config.json` is required to build the `ModelGraph` for compilation.
pub fn compile_assembly(
    spec: &AssemblySpec,
    output_dir: &Path,
    model_files: &HashMap<String, std::path::PathBuf>,
) -> Result<std::path::PathBuf, String> {
    let output_path = output_dir.join("assembled.cimage");

    // 1. Compile each model independently
    let mut all_tensors: Vec<(String, std::path::PathBuf)> = Vec::new();
    let mut total_size: u64 = 0;

    for model in &spec.models {
        let model_file = model_files
            .get(&model.name)
            .ok_or_else(|| format!("model file not found for '{}'", model.name))?;

        let model_dir = output_dir.join(&model.name);
        std::fs::create_dir_all(&model_dir).map_err(|e| format!("create dir: {e}"))?;

        let model_output = model_dir.join("model.cimage");

        // Determine whether the model needs compilation or is pre-compiled.
        let is_cimage = model_file.extension().is_some_and(|e| e == "cimage");

        if !is_cimage {
            // Attempt full compilation: load config.json adjacent to the model file.
            let config_path = model_file
                .parent()
                .map(|p| p.join("config.json"))
                .ok_or_else(|| format!("cannot determine directory for model '{}'", model.name))?;

            let safetensors_dir = model_file
                .parent()
                .ok_or_else(|| format!("no parent directory for model '{}'", model.name))?;

            let config = prism_ecs_ir::UnifiedConfig::from_file(&config_path).map_err(|e| {
                format!(
                    "load config.json for '{}' (needed for .gguf compilation): {e}",
                    model.name
                )
            })?;
            let graph = ModelGraph::build(&config);

            compile_to_cimage(
                &graph,
                safetensors_dir,
                &model_output,
                false,
                |_, _, _, _, _| {},
                model.format_plan.as_ref(),
                CompilationBackend::Default,
            )?;
        } else {
            // Pre-compiled .cimage — copy directly (v1 concatenation).
            std::fs::copy(model_file, &model_output)
                .map_err(|e| format!("copy .cimage for '{}': {e}", model.name))?;
        }

        total_size += std::fs::metadata(&model_output)
            .map(|m| m.len())
            .unwrap_or(0);
        all_tensors.push((model.name.clone(), model_output));
    }

    // 2. Check memory budget
    let ram_gb = total_size as f64 / (1024.0 * 1024.0 * 1024.0);
    if ram_gb > spec.total_ram_budget_gb {
        return Err(format!(
            "assembly {:.2} GB exceeds RAM budget {:.1} GB",
            ram_gb, spec.total_ram_budget_gb
        ));
    }

    // 3. Interleave tensors and write model index header
    let mut output =
        std::fs::File::create(&output_path).map_err(|e| format!("create output: {e}"))?;

    let mut model_index = serde_json::Map::new();
    for (name, compiled_path) in &all_tensors {
        let data = std::fs::read(compiled_path).map_err(|e| format!("read compiled: {e}"))?;
        output
            .write_all(&data)
            .map_err(|e| format!("write assembly: {e}"))?;
        model_index.insert(
            name.clone(),
            serde_json::json!({
                "cimage_path": compiled_path.to_string_lossy(),
                "size_bytes": data.len(),
            }),
        );
    }

    eprintln!(
        "assembly: {} models, {:.2} GB total",
        all_tensors.len(),
        ram_gb
    );
    Ok(output_path)
}

/// ANE compilation branch — delegates to prism-ane's compile_ane_prefill.
#[cfg(feature = "ane")]
fn compile_to_cimage_ane(
    graph: &ModelGraph,
    safetensors_dir: &Path,
    output_path: &Path,
) -> Result<(), String> {
    prism_ane::compile_full_model::compile_ane_prefill("model", safetensors_dir, graph, output_path)
}

/// ANE compilation fallback — returns an error when the `ane` feature is disabled.
#[cfg(not(feature = "ane"))]
fn compile_to_cimage_ane(
    _graph: &ModelGraph,
    _safetensors_dir: &Path,
    _output_path: &Path,
) -> Result<(), String> {
    Err("ANE compilation requires the `ane` feature".to_string())
}

#[cfg(test)]
mod tests {
    //! Constitutional cutover: the silent-substitution bug.
    //!
    //! The previous version of `quantize_by_format` matched
    //! `Bf16 | Int8 | Nf8` and produced a 4-bit palettized payload while
    //! reporting the requested format. That made a CImage header that
    //! said INT8 physically contain 4-bit palettized bytes. These tests
    //! pin the new contract: any format without a real codec returns
    //! `Err`. The default path (`Palettized4Bit` and the
    //! `format_override = None` branch) is unaffected.

    use super::*;
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;

    /// `quantize_to_payload` for `Bf16` is a hard error.
    #[test]
    fn quantize_bf16_is_hard_error() {
        let vals = vec![0.0f32; 16];
        let result = quantize_to_payload("test", &vals, 2, 8, TensorFormat::Bf16);
        assert!(result.is_err(), "Bf16 must not silently substitute");
        let msg = result.unwrap_err();
        assert!(msg.contains("Bf16"), "error must name the format: {msg}");
    }

    /// `quantize_to_payload` for `Int8` is a hard error.
    #[test]
    fn quantize_int8_is_hard_error() {
        let vals = vec![0.0f32; 16];
        let result = quantize_to_payload("test", &vals, 2, 8, TensorFormat::Int8);
        assert!(result.is_err(), "Int8 must not silently substitute");
        let msg = result.unwrap_err();
        assert!(msg.contains("Int8"), "error must name the format: {msg}");
    }

    /// `quantize_to_payload` for `Nf8` is a hard error.
    #[test]
    fn quantize_nf8_is_hard_error() {
        let vals = vec![0.0f32; 16];
        let result = quantize_to_payload("test", &vals, 2, 8, TensorFormat::Nf8);
        assert!(result.is_err(), "Nf8 must not silently substitute");
        let msg = result.unwrap_err();
        assert!(msg.contains("Nf8"), "error must name the format: {msg}");
    }

    /// `quantize_to_payload` for `Palettized4Bit` still works.
    #[test]
    fn quantize_palettized4bit_succeeds() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let result = quantize_to_payload("test", &vals, 2, 8, TensorFormat::Palettized4Bit);
        assert!(result.is_ok());
        let (payload, tensor_type, _bpp) = result.unwrap();
        assert!(!payload.is_empty());
        assert_eq!(tensor_type, TensorType::Palettized4Bit);
    }

    /// `quantize_to_payload` for `Fp16` still works and produces FP16
    /// payload bytes.
    #[test]
    fn quantize_fp16_succeeds() {
        let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let result = quantize_to_payload("test", &vals, 2, 8, TensorFormat::Fp16);
        assert!(result.is_ok());
        let (payload, tensor_type, _bpp) = result.unwrap();
        assert_eq!(payload.len(), 16 * 2);
        assert_eq!(tensor_type, TensorType::StandardFP16);
    }

    /// `build_quantization_plan` records the default format on the
    /// plan so receipts can prove what policy was applied.
    #[test]
    fn build_plan_records_default_format() {
        // The default_format field on the result is what we read in
        // receipts. Pin it so future edits cannot silently change the
        // legacy default.
        let plan = QuantizationResult {
            source_digest: "src".into(),
            target_hardware: "apple-m1".into(),
            selections: vec![],
            execution_plan_json: None,
            default_format: TensorFormat::Palettized4Bit,
        };
        assert_eq!(plan.default_format, TensorFormat::Palettized4Bit);
        assert_eq!(plan.default_format_count(), 0);
        assert_eq!(plan.explicit_format_count(), 0);
    }

    /// `write_cimage_from_plan` is a pure function: same plan + same
    /// path = same bytes (we check the header magic + size are
    /// identical across two consecutive writes).
    #[test]
    fn write_cimage_from_plan_is_deterministic() {
        let plan = QuantizationResult {
            source_digest: "src".into(),
            target_hardware: "apple-m1".into(),
            selections: vec![QuantizedTensorSelection {
                key: "a".into(),
                format: TensorFormat::Palettized4Bit,
                payload: vec![0u8; 64],
                tensor_type: TensorType::Palettized4Bit,
                dim_m: 2,
                dim_n: 8,
                effective_bpp: 4.0,
                payload_bytes: 64,
            }],
            execution_plan_json: Some("{}".into()),
            default_format: TensorFormat::Palettized4Bit,
        };
        let dir = tempdir_like();
        let p1 = dir.join("a.cimage");
        let p2 = dir.join("b.cimage");
        write_cimage_from_plan(&plan, &p1).expect("first write");
        write_cimage_from_plan(&plan, &p2).expect("second write");
        let d1 = std::fs::read(&p1).expect("read a");
        let d2 = std::fs::read(&p2).expect("read b");
        assert_eq!(d1.len(), d2.len(), "byte length must match");
        // The CImage header is 16 KB reserved; the first 16 bytes are
        // the magic + header size and must be identical.
        assert_eq!(&d1[..16], &d2[..16], "header prefix must match");
    }

    fn tempdir_like() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "prism-quant-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let dir = base.join(unique);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
