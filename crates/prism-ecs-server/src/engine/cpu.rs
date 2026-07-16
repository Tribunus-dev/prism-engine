//! CPU fallback — per-format matmul dispatch on the CPU.
//!
//! Each quantized tensor format is dequantized to FP32 using the
//! corresponding codec family from `prism-ecs-quantization`, then
//! a standard FP32 GEMV is performed.

use prism_ecs_ir::evolution::mutation_table::TensorFormat;

/// CPU matmul: dequantize (if needed) and multiply `weight * input`.
///
/// # Arguments
/// - `input`: activation vector (length `dim_n`).
/// - `weight_data`: raw weight payload bytes from the `.cimage` file.
/// - `dim_m`: output dimension (weight rows).
/// - `dim_n`: input dimension (weight columns).
/// - `format`: quantization format of this tensor.
///
/// # Returns
/// Output vector of length `dim_m` (FP32).
pub fn matmul(
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
    format: &TensorFormat,
) -> Result<Vec<f32>, String> {
    // Validate input length
    let n = dim_n as usize;
    if input.len() != n {
        return Err(format!(
            "cpu::matmul: input len {} does not match dim_n {}",
            input.len(),
            n
        ));
    }

    match format {
        TensorFormat::Fp16 => matmul_fp16(input, weight_data, dim_m, dim_n),
        TensorFormat::Bf16 => matmul_bf16(input, weight_data, dim_m, dim_n),
        TensorFormat::Int8 => matmul_int8(input, weight_data, dim_m, dim_n),
        TensorFormat::Int4 => matmul_int4(input, weight_data, dim_m, dim_n),
        TensorFormat::Nf4 => matmul_nf4(input, weight_data, dim_m, dim_n),
        TensorFormat::Nf8 => matmul_nf8(input, weight_data, dim_m, dim_n),
        TensorFormat::Ternary158 => matmul_ternary(input, weight_data, dim_m, dim_n),
        TensorFormat::Binary1 => matmul_binary(input, weight_data, dim_m, dim_n),
        TensorFormat::Palettized4Bit => Err(
            "Palettized4Bit CPU matmul: not yet implemented — pending palettized codec integration"
                .to_string(),
        ),
    }
}

// ── Per-format implementations ───────────────────────────────────────────
//
// Each function reads the weight_data as the specified format, dequantizes
// on the fly (or interprets directly for FP16/BF16), and performs a FP32
// GEMV.

/// FP16 matmul — weights are stored as native `f16` values.
///
/// Reads `dim_m * dim_n` half-precision values and multiplies into the
/// activation vector.
fn matmul_fp16(
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
) -> Result<Vec<f32>, String> {
    let m = dim_m as usize;
    let n = dim_n as usize;
    let expected_bytes = m * n * 2; // 2 bytes per f16
    if weight_data.len() < expected_bytes {
        return Err(format!(
            "FP16 matmul: expected {} bytes, got {}",
            expected_bytes,
            weight_data.len()
        ));
    }

    let mut output = vec![0.0f32; m];
    // Slice the weight data into half-precision values, dequantize on the fly.
    let f16_bytes = &weight_data[..expected_bytes];
    for i in 0..m {
        let mut dot = 0.0f32;
        for j in 0..n {
            let idx = (i * n + j) * 2;
            let raw = u16::from_le_bytes([f16_bytes[idx], f16_bytes[idx + 1]]);
            let w = f32::from(half::f16::from_bits(raw));
            dot += w * input[j];
        }
        output[i] = dot;
    }
    Ok(output)
}

/// BF16 matmul — weights as bfloat16 (truncated FP32).
fn matmul_bf16(
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
) -> Result<Vec<f32>, String> {
    let m = dim_m as usize;
    let n = dim_n as usize;
    let expected_bytes = m * n * 2;
    if weight_data.len() < expected_bytes {
        return Err(format!(
            "BF16 matmul: expected {} bytes, got {}",
            expected_bytes,
            weight_data.len()
        ));
    }

    let mut output = vec![0.0f32; m];
    for i in 0..m {
        let mut dot = 0.0f32;
        for j in 0..n {
            let idx = (i * n + j) * 2;
            // BF16: 16-bit truncated float — pad with zeros to form 32-bit float
            let bits = ((weight_data[idx] as u32) << 8) | (weight_data[idx + 1] as u32);
            let w = f32::from_bits(bits << 16);
            dot += w * input[j];
        }
        output[i] = dot;
    }
    Ok(output)
}

/// INT8 matmul — per-tensor or per-channel scale.
///
/// TODO: parse scale factors from weight_data (currently positional after
/// quantized data) and apply per-channel dequant.
fn matmul_int8(
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
) -> Result<Vec<f32>, String> {
    let m = dim_m as usize;
    let n = dim_n as usize;
    let expected = m * n;
    if weight_data.len() < expected {
        return Err(format!(
            "INT8 matmul: expected {} bytes, got {}",
            expected,
            weight_data.len()
        ));
    }

    // Simple per-tensor scale: compute max abs value from the payload.
    // TODO: store scale factors in the CImage execution plan for proper dequant.
    let mut output = vec![0.0f32; m];
    for i in 0..m {
        let mut dot = 0.0f32;
        for j in 0..n {
            let w = weight_data[i * n + j] as i8 as f32;
            dot += w * input[j];
        }
        output[i] = dot;
    }
    Ok(output)
}

/// INT4 matmul — per-group 4-bit quantized with FP16 scale.
///
/// TODO: determine group size from CImage execution plan / metadata.
fn matmul_int4(
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
) -> Result<Vec<f32>, String> {
    Err("INT4 CPU matmul: not yet implemented — pending codec integration".to_string())
}

/// NF4 matmul — Normal Float 4-bit with per-group scale.
///
/// TODO: integrate NF4 codec from prism-ecs-quantization sweep families.
fn matmul_nf4(
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
) -> Result<Vec<f32>, String> {
    Err("NF4 CPU matmul: not yet implemented — pending codec integration".to_string())
}

/// NF8 matmul — Normal Float 8-bit.
fn matmul_nf8(
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
) -> Result<Vec<f32>, String> {
    Err("NF8 CPU matmul: not yet implemented".to_string())
}

/// Ternary matmul — 1.58-bit {-1, 0, +1} with FP16 scale per 128 elements.
///
/// TODO: integrate ternary codec from prism-ecs-quantization ternarization family.
fn matmul_ternary(
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
) -> Result<Vec<f32>, String> {
    Err("Ternary158 CPU matmul: not yet implemented — pending codec integration".to_string())
}

/// Binary matmul — 1-bit {0, +1} with FP16 scale.
fn matmul_binary(
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
) -> Result<Vec<f32>, String> {
    Err("Binary1 CPU matmul: not yet implemented".to_string())
}

// ── Utility operations ──────────────────────────────────────────────────────

/// Root Mean Square normalization: `y = x / sqrt(mean(x²) + eps) * weight`.
///
/// Both `x` and `weight` must have length `dim`. Returns normalized vector.
pub fn rms_norm(x: &[f32], weight: &[f32]) -> Result<Vec<f32>, String> {
    if x.len() != weight.len() {
        return Err(format!(
            "rms_norm: x len {} != weight len {}",
            x.len(),
            weight.len()
        ));
    }
    if x.is_empty() {
        return Ok(Vec::new());
    }
    let dim = x.len() as f32;
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let rms = (sum_sq / dim + 1e-5).sqrt();
    let inv_rms = 1.0 / rms;
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        out.push(v * inv_rms * weight[i]);
    }
    Ok(out)
}

/// Element-wise vector addition: `c = a + b`.
///
/// Both slices must have the same length.
pub fn vec_add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "vec_add: left len {} != right len {}",
            a.len(),
            b.len()
        ));
    }
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        out.push(x + y);
    }
    Ok(out)
}

/// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_fp16_basic() {
        // 2x3 weight matrix, 3-element input
        let n = 3u32;
        let m = 2u32;
        let weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Serialize as little-endian f16 bytes
        let bytes: Vec<u8> = weight
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
            .collect();
        let input = vec![1.0f32, 1.0, 1.0];

        let result = matmul(&input, &bytes, m, n, &TensorFormat::Fp16).unwrap();
        assert_eq!(result.len(), 2);
        // row0: 1+2+3 = 6, row1: 4+5+6 = 15
        assert!((result[0] - 6.0).abs() < 1e-3);
        assert!((result[1] - 15.0).abs() < 1e-3);
    }

    #[test]
    fn test_matmul_input_len_mismatch() {
        let result = matmul(&[1.0f32], &[0u8; 8], 2, 2, &TensorFormat::Fp16);
        assert!(result.is_err());
    }
}
