//! ANE operator validation for QuantSweep candidates.
//!
//! Takes a candidate's reconstructed weights, compiles a MIL matmul graph
//! with activation and weights as dual IOSurface-backed inputs, executes on
//! the ANE via Core ML, and compares the output against a CPU float reference.
//! Returns operator-level metrics (operator NRMSE, cosine similarity, norm
//! ratio drift) that weight-space validation alone cannot provide.
//!
//! # Compilation cost
//!
//! Each unique `(in_features, out_features)` shape triggers one
//! coremlcompiler invocation (~0.5–2s). The compiled `.mlmodelc` is cached
//! at `/tmp/ane_opval_{in}x{out}.mlmodelc/`. Subsequent calls for the
//! same shape reuse the cached model and only swap weight data via IOSurface.
//!
//! # Gating
//!
//! Only available on macOS with `prism-backend` or `mlx-backend`.

use std::path::PathBuf;

use crate::arena::{Arena, DataType};
use crate::coreai_bridge::{CoreAiComputeUnits, CoreAiModel};
use crate::coreai_pipeline::compile_mlpackage;
use crate::mil_builder::MilBuilder;
use crate::mlpackage::{write_mlpackage, ModelMeta};

use coreml_proto::proto::mil_spec;

// The first matmul in a MIL graph gets SSA name "matmul_0".
// Since our model has exactly one matmul, this is the output name.
const OUTPUT_NAME: &str = "matmul_0";
const ACT_INPUT_NAME: &str = "activation";
const WT_INPUT_NAME: &str = "weights";

// ── FP16 conversion helpers ───────────────────────────────────────────────

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 255 {
        return (sign << 15) | 0x7C00;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let value = (mant as f32) * 2.0f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        f32::INFINITY
    } else {
        let normalized = 1.0f32 + (mant as f32) / 1024.0f32;
        let value = normalized * 2.0f32.powi((exp as i32) - 15);
        if sign != 0 { -value } else { value }
    }
}

// ── CPU reference matmul ──────────────────────────────────────────────────

/// CPU float reference: `output[j] = sum_i activation[i] * weights[j * in + i]`
/// Weights are row-major [out_features, in_features].
fn cpu_matmul(activation: &[f32], weights: &[f32], in_features: usize, out_features: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_features];
    for j in 0..out_features {
        let base = j * in_features;
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += activation[i] * weights[base + i];
        }
        output[j] = sum;
    }
    output
}

// ── Metrics type ──────────────────────────────────────────────────────────

/// Metrics from ANE operator validation, comparing ANE output against
/// a CPU float reference matmul.
#[derive(Debug, Clone, Copy)]
pub struct OperatorValidationMetrics {
    /// RMSE of the output activation vector (ANE vs reference).
    pub operator_rmse: f64,
    /// NRMSE = sqrt(SSE) / ||reference||.
    pub operator_nrmse: f64,
    /// Cosine similarity between ANE output and reference.
    pub cosine_similarity: f64,
    /// Ratio of output norms: ||ANE|| / ||reference||.
    pub norm_ratio_drift: f64,
    /// Maximum absolute element-wise error.
    pub max_abs_error: f64,
}

// ── Model compilation and caching ─────────────────────────────────────────

fn model_cache_path(in_features: u32, out_features: u32) -> PathBuf {
    PathBuf::from(std::env::temp_dir())
        .join(format!("ane_opval_{}x{}", in_features, out_features))
        .join("compiled")
        .join(format!("ane_opval_{}x{}.mlmodelc", in_features, out_features))
}

fn compile_or_load_model(
    in_features: u32,
    out_features: u32,
) -> Result<(CoreAiModel, Arena, Arena), String> {
    let cache_path = model_cache_path(in_features, out_features);

    if cache_path.exists() {
        let model = CoreAiModel::load_with_compute_units(
            &cache_path.to_string_lossy(),
            CoreAiComputeUnits::CpuAndNeuralEngine,
        )?;
        let weight_arena = Arena::new(in_features, out_features, DataType::Float16)
            .map_err(|e| format!("weight arena: {}", e))?;
        let output_arena = Arena::new(1, out_features, DataType::Float16)
            .map_err(|e| format!("output arena: {}", e))?;
        return Ok((model, weight_arena, output_arena));
    }

    // ── Build MIL program ──────────────────────────────────────────────
    let b = MilBuilder::new("main");
    let b = b.input(ACT_INPUT_NAME, mil_spec::DataType::Float16, &[1, in_features as i64]);
    let b = b.input(WT_INPUT_NAME, mil_spec::DataType::Float16, &[in_features as i64, out_features as i64]);
    let b = b.matmul(ACT_INPUT_NAME, WT_INPUT_NAME);
    let prog = b
        .output(OUTPUT_NAME)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: format!("ane_opval_{}x{}", in_features, out_features),
        function_name: "main".into(),
        short_description: format!("Operator validation matmul {}x{}", in_features, out_features),
        version: "1.0.0".into(),
        author: "tribunus-sweep".into(),
        output_name: OUTPUT_NAME.into(),
        inputs: vec![
            (ACT_INPUT_NAME.into(), vec![1, in_features as i64]),
            (WT_INPUT_NAME.into(), vec![in_features as i64, out_features as i64]),
        ],
        outputs: vec![(OUTPUT_NAME.into(), vec![1, out_features as i64])],
    };

    let model_dir = PathBuf::from(std::env::temp_dir())
        .join(format!("ane_opval_{}x{}", in_features, out_features));
    let mlpackage_dir = model_dir.join("model.mlpackage");
    let compiled_dir = model_dir.join("compiled");

    // Always try to compile (the mlpackage_dir guard is unsafe — a failed
    // compile leaves mlpackage_dir in place and blocks retry)
    if !cache_path.exists() {
        let _ = std::fs::create_dir_all(&compiled_dir);
        // Clean stale mlpackage if present
        if mlpackage_dir.exists() {
            std::fs::remove_dir_all(&mlpackage_dir).ok();
        }
        let written_package = write_mlpackage(prog, &mlpackage_dir, &meta)
            .map_err(|e| format!("mlpackage write: {}", e))?;

        let receipt = compile_mlpackage(
            &written_package,
            &compiled_dir,
            &meta.model_name,
            "cpuAndNeuralEngine",
            "iOS15",
        )
        .map_err(|e| format!("compile: {}", e))?;

        // Link to our canonical cache path
        let compiled = PathBuf::from(&receipt.compiled_modelc_path);
        if !cache_path.exists() && compiled.exists() {
            std::fs::rename(&compiled, &cache_path)
                .map_err(|e| format!("rename to cache: {}", e))?;
        }
    }

    let model = CoreAiModel::load_with_compute_units(
        &cache_path.to_string_lossy(),
        CoreAiComputeUnits::CpuAndNeuralEngine,
    )?;

    let weight_arena = Arena::new(in_features, out_features, DataType::Float16)
        .map_err(|e| format!("weight arena: {}", e))?;
    let output_arena = Arena::new(1, out_features, DataType::Float16)
        .map_err(|e| format!("output arena: {}", e))?;

    Ok((model, weight_arena, output_arena))
}

// ── Public API ────────────────────────────────────────────────────────────

/// Run ANE operator validation for a single candidate.
/// Pack an NF4 weight matrix using an explicit codebook and group size.
///
/// Unlike `pack_nf4_weights` (which uses the default Prism codebook with
/// group_size=128), this function accepts a codebook slice and group_size
/// parameter, matching the sweep runner's configurable NF4 codecs.
///
/// Returns (packed_codes, scales, biases). Scales use MaxAbs per group.
pub fn pack_nf4_weights_with_codebook(
    weights: &[f32],
    in_features: u32,
    out_features: u32,
    codebook: &[f32; 16],
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    const TILE_SIZE: usize = 640;
    let in_f = in_features as usize;
    let out_f = out_features as usize;
    let tiles_per_row = out_f.div_ceil(TILE_SIZE);
    let total_tiles = in_f * tiles_per_row;
    let num_groups = TILE_SIZE / group_size;
    let bytes_per_group = group_size / 2;
    let codes_per_tile = num_groups * bytes_per_group;

    let mut packed = vec![0u8; total_tiles * codes_per_tile];
    let mut scales = vec![0.0f32; total_tiles * num_groups];
    let mut biases = vec![0.0f32; total_tiles * num_groups];

    for tile_idx in 0..total_tiles {
        let row = tile_idx / tiles_per_row;
        let tile_col = tile_idx % tiles_per_row;
        let col_base = tile_col * TILE_SIZE;
        let code_off = tile_idx * codes_per_tile;
        let scale_off = tile_idx * num_groups;

        // Build tile from the weight matrix, padding with 0s beyond cols
        let mut tile = [0.0f32; TILE_SIZE];
        for j in 0..TILE_SIZE {
            let c = col_base + j;
            tile[j] = if c < out_f { weights[row * out_f + c] } else { 0.0 };
        }

        for g in 0..num_groups {
            let base = g * group_size;
            let max_abs = tile[base..base + group_size].iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let scale = if max_abs < 1e-30 { 1.0 } else { max_abs };
            scales[scale_off + g] = scale;
            biases[scale_off + g] = 0.0;

            for i in 0..(group_size / 2) {
                let v0 = tile[base + 2 * i] / scale;
                let v1 = tile[base + 2 * i + 1] / scale;
                let c0 = crate::nf4tile640::nf4_quantize_with_codebook(v0, codebook);
                let c1 = crate::nf4tile640::nf4_quantize_with_codebook(v1, codebook);
                packed[code_off + g * bytes_per_group + i] = c0 | (c1 << 4);
            }
        }
    }

    (packed, scales, biases)
}

/// Run ANE operator validation for a single candidate.
///
/// `reconstructed_weights` — f32 slice of `[out_features × in_features]`
/// elements (row-major, Prism canonical layout).
///
/// Compiles a MIL matmul graph once per unique shape (cached on disk),
/// then runs inference with a synthetic activation vector, comparing the
/// ANE output against a CPU float reference matmul.
pub fn validate_operator(
    reconstructed_weights: &[f32],
    in_features: u32,
    out_features: u32,
) -> Result<OperatorValidationMetrics, String> {
    let total = (out_features as usize) * (in_features as usize);
    if reconstructed_weights.len() != total {
        return Err(format!(
            "weight count mismatch: got {} expected {}",
            reconstructed_weights.len(),
            total
        ));
    }

    // ── Synthetic activation ───────────────────────────────────────────
    let pi = std::f32::consts::PI;
    let activation: Vec<f32> = (0..in_features as usize)
        .map(|i| ((i as f32) / (in_features as f32) * pi).sin())
        .collect();

    // ── CPU reference ──────────────────────────────────────────────────
    let reference = cpu_matmul(&activation, reconstructed_weights, in_features as usize, out_features as usize);
    let ref_norm: f64 = reference.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();

    // ── Compile or load model ──────────────────────────────────────────
    let (model, weight_arena, output_arena) = compile_or_load_model(in_features, out_features)?;

    // ── Write FP16 weights into IOSurface arena ────────────────────────
    weight_arena.lock().map_err(|e| format!("lock weights: {}", e))?;
    unsafe {
        let ptr = weight_arena.info.base_address as *mut u16;
        let in_f = in_features as usize;
        let out_f = out_features as usize;
        // Transpose: MIL weights are [in, out], data is [out, in]
        for i in 0..in_f {
            for j in 0..out_f {
                ptr.add(i * out_f + j).write(f32_to_f16_bits(reconstructed_weights[j * in_f + i]));
            }
        }
    }
    weight_arena.unlock().map_err(|e| format!("unlock weights: {}", e))?;

    // ── Create and fill activation arena ───────────────────────────────
    let act_arena = Arena::new(1, in_features, DataType::Float16)
        .map_err(|e| format!("activation arena: {}", e))?;
    act_arena.lock().map_err(|e| format!("lock activation: {}", e))?;
    unsafe {
        let ptr = act_arena.info.base_address as *mut u16;
        for (i, &a) in activation.iter().enumerate() {
            ptr.add(i).write(f32_to_f16_bits(a));
        }
    }
    act_arena.unlock().map_err(|e| format!("unlock activation: {}", e))?;

    // ── Run ANE prediction with dual inputs ────────────────────────────
    // We need &mut ArenaInfo for outputs. Create a mutable reference.
    let mut out_info = output_arena.info;
    model.predict_multi(
        &[ACT_INPUT_NAME, WT_INPUT_NAME],
        &[&act_arena.info, &weight_arena.info],
        &[OUTPUT_NAME],
        &mut [&mut out_info],
    ).map_err(|e| format!("predict_multi: {}", e))?;

    // ── Read back output ───────────────────────────────────────────────
    output_arena.lock().map_err(|e| format!("lock output: {}", e))?;
    let mut ane_output = vec![0.0f32; out_features as usize];
    unsafe {
        let ptr = out_info.base_address as *const u16;
        for i in 0..out_features as usize {
            ane_output[i] = f16_bits_to_f32(ptr.add(i).read());
        }
    }
    output_arena.unlock().map_err(|e| format!("unlock output: {}", e))?;

    // ── Compute metrics ────────────────────────────────────────────────
    let mut sq_err = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut dot_product = 0.0f64;
    let mut ane_norm_sq = 0.0f64;

    for j in 0..out_features as usize {
        let a = ane_output[j] as f64;
        let r = reference[j] as f64;
        let diff = a - r;
        sq_err += diff * diff;
        let abs_diff = diff.abs();
        if abs_diff > max_abs { max_abs = abs_diff; }
        dot_product += a * r;
        ane_norm_sq += a * a;
    }
    let ane_norm = ane_norm_sq.sqrt();

    let rmse = (sq_err / out_features as f64).sqrt();
    let nrmse = if ref_norm > 1e-30 {
        sq_err.sqrt() / ref_norm
    } else {
        0.0
    };
    let cosine = if ane_norm > 1e-30 && ref_norm > 1e-30 {
        dot_product / (ane_norm * ref_norm)
    } else {
        1.0
    };
    let norm_drift = if ref_norm > 1e-30 {
        ane_norm / ref_norm
    } else {
        1.0
    };

    Ok(OperatorValidationMetrics {
        operator_rmse: rmse,
        operator_nrmse: nrmse,
        cosine_similarity: cosine,
        norm_ratio_drift: norm_drift,
        max_abs_error: max_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_roundtrip() {
        let cases = [0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0, 0.000061, -0.000061];
        for &v in &cases {
            let bits = f32_to_f16_bits(v);
            let back = f16_bits_to_f32(bits);
            let diff = (back - v).abs();
            let rel = if v.abs() > 1e-10 { diff / v.abs() } else { diff };
            assert!(diff < 0.001 || rel < 0.01,
                "fp16 roundtrip: {} -> {:#06x} -> {}, diff={}", v, bits, back, diff);
        }
    }

    #[test]
    fn test_cpu_matmul_identity_2x2() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let act = vec![3.0, 5.0];
        let r = cpu_matmul(&act, &w, 2, 2);
        assert!((r[0] - 3.0).abs() < 1e-6);
        assert!((r[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_matmul_2x3() {
        // W[2x3] row-major: [[1,2,3],[4,5,6]]
        let w: Vec<f32> = (1..=6).map(|i| i as f32).collect();
        let act = vec![2.0, 3.0, 1.0];
        // output[0] = 2*1 + 3*4 + 1*... wait, in=3, out=2
        // W is [out=2, in=3]
        // output[0] = 2*1 + 3*2 + 1*3 = 11
        // output[1] = 2*4 + 3*5 + 1*6 = 29
        let r = cpu_matmul(&act, &w, 3, 2);
        assert!((r[0] - 11.0).abs() < 1e-6, "expected 11, got {}", r[0]);
        assert!((r[1] - 29.0).abs() < 1e-6, "expected 29, got {}", r[1]);
    }

    /// Integration test requiring ANE hardware. Marked ignore by default.
    #[ignore = "requires ANE hardware and Core ML compilation toolchain"]
    #[test]
    fn test_validate_operator_small() {
        let in_f = 64;
        let out_f = 64;
        // Identity-like weights: each output is sum of corresponding input portion
        let mut w = vec![0.0f32; (out_f * in_f) as usize];
        for j in 0..out_f as usize {
            for i in 0..in_f as usize {
                if i == j % in_f as usize {
                    w[j * in_f as usize + i] = 1.0;
                }
            }
        }
        // Not truly identity since [out_f=64, in_f=64] — OK for small test
        // Actually that IS identity. Reset to something that produces signal.
        let w: Vec<f32> = (0..(out_f * in_f) as usize)
            .map(|idx| ((idx % 64) as f32 - 32.0) * 0.01)
            .collect();

        let metrics = validate_operator(&w, in_f, out_f)
            .expect("operator validation should succeed");
        assert!(metrics.operator_rmse >= 0.0);
        assert!(metrics.cosine_similarity >= -1.0 && metrics.cosine_similarity <= 1.0);
        eprintln!("operator metrics: rmse={:.6} nrmse={:.6} cosine={:.6} drift={:.6} max_abs={:.6}",
            metrics.operator_rmse, metrics.operator_nrmse,
            metrics.cosine_similarity, metrics.norm_ratio_drift, metrics.max_abs_error);
    }
}
