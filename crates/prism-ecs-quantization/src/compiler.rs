//! AOT palette compiler with universal dequantization for Prism Engine.
//!
//! Takes a `ModelGraph`, iterates every `PalettizedMatmul` node, loads
//! weights in any format (F32/BF16/F16/U32 block-quantized), runs k-means
//! per row, builds split-block payloads, and writes a `.cimage` file.
//!
//! When a `FormatPlan` is provided, uses the per-tensor format assignment
//! from evolution search instead of uniform k-means palettization.

use std::collections::HashMap;
use std::path::Path;

use crate::cimage::{CImageWriter, TensorType};
use crate::palette::palettize_matrix;
use prism_ecs_ir::evolution::compile_plan::FormatPlan;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use prism_ecs_ir::{generate_plan, ModelGraph, TensorBlueprint};

pub struct CompiledTensor {
    pub key: String,
    pub dim_m: u32,
    pub dim_n: u32,
    pub payload: Vec<u8>,
    pub effective_bpp: f32,
}

/// Compile an entire model into a `.cimage` file.
///
/// When `compile_plan` is `Some`, uses per-tensor format assignments from the
/// evolution search instead of uniform k-means palettization.
pub fn compile_to_cimage(
    graph: &ModelGraph,
    safetensors_dir: &Path,
    output_path: &Path,
    has_metal: bool,
    progress: impl Fn(&str, u32, u32, f64, f64),
    compile_plan: Option<&FormatPlan>,
) -> Result<(), String> {
    // Generate execution plan before compiling weights.
    let plan = generate_plan(graph, has_metal, false);
    let plan_json = serde_json::to_string(&plan).map_err(|e| format!("serialize plan: {e}"))?;

    let mut cimage = CImageWriter::new(output_path)?;
    cimage.set_execution_plan(plan_json);

    let shards = discover_safetensors(safetensors_dir)?;

    // Helper: compile a single tensor with optional plan-based dispatch
    let compile_tensor = |cimage: &mut CImageWriter,
                          tb: &TensorBlueprint,
                          format: Option<TensorFormat>|
     -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let f32_vals = load_weight_f32(&shards, tb)?;
        let out_dim = tb.dim_m as usize;
        let in_dim = tb.dim_n as usize;

        eprint!("  [prism] {} ({}×{})... ", tb.key, out_dim, in_dim);

        match format {
            // With a plan-specified format, dispatch to the appropriate codec
            Some(fmt) => {
                let result = quantize_by_format(cimage, &tb.key, &f32_vals, out_dim, in_dim, fmt)?;
                let elapsed = t0.elapsed();
                eprintln!(
                    "bpp={:.3} format={:?} {:.2}s",
                    result.bpp,
                    fmt,
                    elapsed.as_secs_f64()
                );
                progress(
                    &tb.key,
                    tb.dim_m,
                    tb.dim_n,
                    result.bpp as f64,
                    elapsed.as_secs_f64(),
                );
            }
            // Without a plan: use uniform k-means palettization (legacy path)
            None => {
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
                progress(&tb.key, tb.dim_m, tb.dim_n, bpp, elapsed.as_secs_f64());
            }
        }
        Ok(())
    };

    // Compile palettized tensors (matmuls, projections, heads)
    for tb in graph.palettized_tensors() {
        let fmt = compile_plan.and_then(|p| p.get(&tb.key));
        compile_tensor(&mut cimage, tb, fmt)?;
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
            compile_tensor(&mut cimage, &tb, fmt)?;
            break;
        }
    }

    cimage.finalize()?;
    eprintln!("[prism:compile] Done -> {}", output_path.display());
    Ok(())
}

/// Result of a single tensor quantization operation.
struct QuantResult {
    bpp: f32,
}

/// Quantize a tensor by the given format and append to cimage.
fn quantize_by_format(
    cimage: &mut CImageWriter,
    key: &str,
    f32_vals: &[f32],
    out_dim: usize,
    in_dim: usize,
    format: TensorFormat,
) -> Result<QuantResult, String> {
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
            cimage.append(key, &payload, out_dim as u32, in_dim as u32, tensor_type)?;
            Ok(QuantResult { bpp: 4.0 })
        }

        TensorFormat::Fp16 => {
            // Raw f32→f16 conversion, no quantization
            let mut payload = Vec::with_capacity(f32_vals.len() * 2);
            for &v in f32_vals {
                payload.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
            }
            cimage.append_fp16(key, &payload, out_dim as u32, in_dim as u32)?;
            Ok(QuantResult { bpp: 16.0 })
        }

        TensorFormat::Ternary158 => {
            // Ternary: quantize each value to {-1, 0, +1} with group scales
            let group_size = 128usize;
            let num_groups = (f32_vals.len() + group_size - 1) / group_size;
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
            cimage.append(
                key,
                &payload,
                out_dim as u32,
                in_dim as u32,
                TensorType::Ternary158,
            )?;
            Ok(QuantResult { bpp })
        }

        TensorFormat::Binary1 => {
            // Binary: quantize to {+1, -1} (1 bit per value)
            let group_size = 128usize;
            let num_groups = (f32_vals.len() + group_size - 1) / group_size;
            let mut binary_bits: Vec<u8> = Vec::new();
            let mut scales = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * group_size;
                let end = (start + group_size).min(f32_vals.len());
                let group = &f32_vals[start..end];

                let mean_abs = group.iter().map(|v| v.abs()).sum::<f32>() / group.len() as f32;
                let scale = if mean_abs > 1e-10 { mean_abs } else { 1.0 };
                scales.push(scale);

                let byte_count = (group.len() + 7) / 8;
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
            cimage.append(
                key,
                &payload,
                out_dim as u32,
                in_dim as u32,
                TensorType::Binary1,
            )?;
            Ok(QuantResult { bpp })
        }

        // Fallback for formats not yet wired: use k-means palettization
        TensorFormat::Palettized4Bit
        | TensorFormat::Bf16
        | TensorFormat::Int8
        | TensorFormat::Nf8 => {
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

            cimage.append_palettized(key, &payload, out_dim as u32, in_dim as u32)?;
            Ok(QuantResult { bpp })
        }
    }
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
