//! Checkpoint-backed Level 1 validation for low-memory distillation.
//!
//! This module validates sampled ternary projections against the real teacher
//! checkpoint without loading the whole model into resident memory. It mmap's
//! the safetensors shards, loads one tensor at a time, samples a small set of
//! rows, compares dense vs ternary-projected outputs, and then releases the
//! tensor bytes before moving on.

use super::reducer::AccelerateReducer;
use crate::ecs::compute_image::compile::int4_pack::{
    quantize_to_ternary_block32, unpack_byte_5_trits,
};
use crate::ecs::compute_image::compile::source::{
    ensure_tensor_loaded, load_source, LoadedSource, SourceTensor,
};
use half::f16;
use std::path::Path;

const DEFAULT_LAYER_STRIDE: usize = 8;
const DEFAULT_ROW_SAMPLES: usize = 8;
const DEFAULT_INPUT_SAMPLES: usize = 2;

const DEFAULT_MAX_OUTPUT_MSE: f64 = 0.05;
const DEFAULT_MAX_RESIDUAL: f64 = 0.90;
const DEFAULT_MIN_COSINE: f64 = 0.70;

const VALIDATED_PROJECTIONS: &[&str] = &["q_proj", "o_proj", "gate_proj", "down_proj"];

#[derive(Debug, Clone)]
pub struct ProjectionValidationMetric {
    pub layer_index: usize,
    pub projection: String,
    pub output_mse: f64,
    pub cosine_similarity: f64,
    pub residual_relative_error: f64,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct CheckpointValidationResult {
    pub passed: bool,
    pub validated_layers: usize,
    pub validated_projections: usize,
    pub max_output_mse: f64,
    pub min_cosine_similarity: f64,
    pub max_residual_relative_error: f64,
    pub metrics: Vec<ProjectionValidationMetric>,
    pub failure_reason: Option<String>,
}

pub fn validate_teacher_checkpoint_against_ternary(
    checkpoint_dir: &Path,
) -> Result<CheckpointValidationResult, String> {
    let mut loaded = load_source(checkpoint_dir, true).map_err(|e| {
        format!(
            "load teacher checkpoint {}: {}",
            checkpoint_dir.display(),
            e
        )
    })?;

    let layer_stride = env_usize(
        "PRISM_DISTILL_VALIDATION_LAYER_STRIDE",
        DEFAULT_LAYER_STRIDE,
    );
    let row_samples = env_usize("PRISM_DISTILL_VALIDATION_ROW_SAMPLES", DEFAULT_ROW_SAMPLES).max(1);
    let input_samples = env_usize(
        "PRISM_DISTILL_VALIDATION_INPUT_SAMPLES",
        DEFAULT_INPUT_SAMPLES,
    )
    .max(1);

    let max_output_mse = env_f64(
        "PRISM_DISTILL_VALIDATION_MAX_OUTPUT_MSE",
        DEFAULT_MAX_OUTPUT_MSE,
    );
    let max_residual = env_f64(
        "PRISM_DISTILL_VALIDATION_MAX_RESIDUAL",
        DEFAULT_MAX_RESIDUAL,
    );
    let min_cosine = env_f64("PRISM_DISTILL_VALIDATION_MIN_COSINE", DEFAULT_MIN_COSINE);

    let total_layers = loaded.arch.num_hidden_layers as usize;
    let mut layer_indices: Vec<usize> = (0..total_layers).step_by(layer_stride.max(1)).collect();
    if total_layers > 0 && layer_indices.last().copied() != Some(total_layers - 1) {
        layer_indices.push(total_layers - 1);
    }

    let mut metrics = Vec::new();
    for &layer_index in &layer_indices {
        for &projection in VALIDATED_PROJECTIONS {
            let metric = validate_projection(
                &mut loaded,
                layer_index,
                projection,
                row_samples,
                input_samples,
            )?;
            metrics.push(metric);
        }
    }

    let validated_layers = layer_indices.len();
    let validated_projections = metrics.len();
    let observed_max_mse = metrics.iter().map(|m| m.output_mse).fold(0.0f64, f64::max);
    let observed_min_cosine = metrics
        .iter()
        .map(|m| m.cosine_similarity)
        .fold(1.0f64, f64::min);
    let observed_max_residual = metrics
        .iter()
        .map(|m| m.residual_relative_error)
        .fold(0.0f64, f64::max);

    let passed = validated_projections > 0
        && observed_max_mse <= max_output_mse
        && observed_min_cosine >= min_cosine
        && observed_max_residual <= max_residual;

    let failure_reason = if passed {
        None
    } else {
        Some(format!(
            "checkpoint-backed ternary validation exceeded thresholds: max_mse={:.6}, min_cosine={:.6}, max_residual={:.6}",
            observed_max_mse, observed_min_cosine, observed_max_residual
        ))
    };

    Ok(CheckpointValidationResult {
        passed,
        validated_layers,
        validated_projections,
        max_output_mse: observed_max_mse,
        min_cosine_similarity: observed_min_cosine,
        max_residual_relative_error: observed_max_residual,
        metrics,
        failure_reason,
    })
}

fn validate_projection(
    loaded: &mut LoadedSource,
    layer_index: usize,
    projection: &str,
    row_samples: usize,
    input_samples: usize,
) -> Result<ProjectionValidationMetric, String> {
    let tensor_name = projection_tensor_name(loaded, layer_index, projection);
    let tensor = load_tensor_bytes(loaded, &tensor_name)?;
    let shape = tensor.shape.clone();
    if shape.len() != 2 {
        release_tensor_bytes(loaded, &tensor_name);
        return Err(format!(
            "tensor {} has non-matrix shape {:?}",
            tensor_name, shape
        ));
    }

    let out_dim = shape[0] as usize;
    let in_dim = shape[1] as usize;
    let sampled_rows = sample_indices(out_dim, row_samples);
    let mut teacher_outputs = Vec::with_capacity(sampled_rows.len() * input_samples);
    let mut student_outputs = Vec::with_capacity(sampled_rows.len() * input_samples);

    for sample_index in 0..input_samples {
        let input = calibration_input(in_dim, layer_index, projection, sample_index);
        for &row_index in &sampled_rows {
            let row = read_row_f32(tensor, row_index, in_dim)?;
            teacher_outputs.push(dot(&row, &input) as f32);
            student_outputs.push(ternary_dot(&row, &input));
        }
    }

    release_tensor_bytes(loaded, &tensor_name);

    let mut reducer = AccelerateReducer::with_hidden_dim(teacher_outputs.len().max(1));
    reducer.reduce(0, &teacher_outputs, &student_outputs);

    Ok(ProjectionValidationMetric {
        layer_index,
        projection: projection.to_string(),
        output_mse: reducer.output_mse.unwrap_or(f64::INFINITY),
        cosine_similarity: reducer.cosine_similarity.unwrap_or(0.0),
        residual_relative_error: reducer.residual_relative_error.unwrap_or(f64::INFINITY),
        samples: teacher_outputs.len(),
    })
}

fn projection_tensor_name(loaded: &LoadedSource, layer_index: usize, projection: &str) -> String {
    let prefix = match projection {
        "gate_proj" | "up_proj" | "down_proj" => "mlp",
        _ => "self_attn",
    };
    format!(
        "{}.layers.{}.{}.{}.weight",
        loaded.namespace.root, layer_index, prefix, projection
    )
}

fn load_tensor_bytes<'a>(
    loaded: &'a mut LoadedSource,
    tensor_name: &str,
) -> Result<&'a SourceTensor, String> {
    let tensor = loaded
        .source_tensors
        .get_mut(tensor_name)
        .ok_or_else(|| format!("missing tensor in checkpoint: {}", tensor_name))?;
    for mmap in &loaded.mmap_bytes {
        ensure_tensor_loaded(tensor, mmap);
        if !tensor.data.is_empty() {
            break;
        }
    }
    if tensor.data.is_empty() {
        return Err(format!(
            "tensor {} could not be loaded from mmap",
            tensor_name
        ));
    }
    Ok(tensor)
}

fn release_tensor_bytes(loaded: &mut LoadedSource, tensor_name: &str) {
    if let Some(tensor) = loaded.source_tensors.get_mut(tensor_name) {
        tensor.data.clear();
        tensor.data.shrink_to_fit();
    }
}

fn read_row_f32(tensor: &SourceTensor, row_index: usize, cols: usize) -> Result<Vec<f32>, String> {
    let bytes_per_element = match tensor.dtype.as_str() {
        "BF16" | "BFloat16" | "F16" | "Float16" => 2usize,
        "F32" | "Float32" => 4usize,
        other => {
            return Err(format!(
                "unsupported tensor dtype {} for sampled projection validation",
                other
            ))
        }
    };
    let row_start = row_index
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(bytes_per_element))
        .ok_or_else(|| format!("row offset overflow for {}", tensor.name))?;
    let row_end = row_start + cols * bytes_per_element;
    if row_end > tensor.data.len() {
        return Err(format!(
            "row {} out of bounds for tensor {} ({} bytes requested, {} available)",
            row_index,
            tensor.name,
            row_end,
            tensor.data.len()
        ));
    }

    let row_bytes = &tensor.data[row_start..row_end];
    let row = match tensor.dtype.as_str() {
        "BF16" | "BFloat16" => row_bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
        "F16" | "Float16" => row_bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        "F32" | "Float32" => row_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        _ => unreachable!(),
    };
    Ok(row)
}

fn ternary_dot(row: &[f32], input: &[f32]) -> f32 {
    let len = row.len().min(input.len());
    let mut acc = 0.0f32;
    let mut offset = 0usize;

    while offset < len {
        let mut block = [0.0f32; 32];
        let block_len = (len - offset).min(32);
        block[..block_len].copy_from_slice(&row[offset..offset + block_len]);

        let quantized = quantize_to_ternary_block32(&block);
        let scale = f16::from_bits(quantized.block_scale).to_f32();

        for byte_idx in 0..6 {
            let mut digits = [0u8; 5];
            unpack_byte_5_trits(quantized.packed_trits[byte_idx], &mut digits);
            for (digit_offset, digit) in digits.iter().enumerate() {
                let idx = offset + byte_idx * 5 + digit_offset;
                if idx >= len {
                    break;
                }
                acc += ternary_value(*digit) * scale * input[idx];
            }
        }

        let tail = quantized.packed_trits[6];
        for tail_idx in 0..2 {
            let idx = offset + 30 + tail_idx;
            if idx >= len {
                break;
            }
            let digit = if tail_idx == 0 { tail % 3 } else { tail / 3 };
            acc += ternary_value(digit) * scale * input[idx];
        }

        offset += 32;
    }

    acc
}

fn ternary_value(digit: u8) -> f32 {
    match digit {
        0 => -1.0,
        1 => 0.0,
        2 => 1.0,
        _ => 0.0,
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum()
}

fn calibration_input(
    in_dim: usize,
    layer_index: usize,
    projection: &str,
    sample_index: usize,
) -> Vec<f32> {
    let projection_seed = projection
        .bytes()
        .fold(0u32, |acc, byte| acc.wrapping_mul(16777619) ^ byte as u32);
    let phase = ((layer_index as u32).wrapping_mul(31) ^ projection_seed ^ sample_index as u32)
        as f64
        * 0.013;
    (0..in_dim)
        .map(|i| (((i as f64) * 0.017 + phase).cos() * 0.1) as f32)
        .collect()
}

fn sample_indices(total: usize, count: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    if count >= total {
        return (0..total).collect();
    }
    let mut indices = Vec::with_capacity(count);
    for sample in 0..count {
        let numerator = sample * (total - 1);
        let denominator = (count - 1).max(1);
        indices.push(numerator / denominator);
    }
    indices.sort_unstable();
    indices.dedup();
    if indices.last().copied() != Some(total - 1) && indices.len() < count {
        indices.push(total - 1);
    }
    indices
}

fn env_usize(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_value)
}

fn env_f64(name: &str, default_value: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_indices_cover_endpoints() {
        assert_eq!(sample_indices(8, 3), vec![0, 3, 7]);
    }

    #[test]
    fn ternary_dot_matches_dense_for_exact_ternary_rows() {
        let row = vec![-2.0f32, 0.0, 2.0, -2.0, 0.0, 2.0];
        let input = vec![0.5f32, 1.0, -0.5, 0.25, -0.25, 0.75];
        let dense = dot(&row, &input) as f32;
        let ternary = ternary_dot(&row, &input);
        assert!((dense - ternary).abs() < 1e-5);
    }

    #[test]
    fn projection_name_uses_namespace_root() {
        let loaded = load_source(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("models/gemma4-12b-8bit"),
            true,
        );
        if let Ok(loaded) = loaded {
            let name = projection_tensor_name(&loaded, 0, "q_proj");
            assert!(name.ends_with(".layers.0.self_attn.q_proj.weight"));
        }
    }
}
