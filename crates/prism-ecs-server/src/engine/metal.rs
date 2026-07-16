//! Metal dispatch — format-aware GPU kernel invocations.
//!
//! Gated behind `cfg(feature = "metal-dispatch")`. When disabled, the
//! dispatch falls through to the CPU path in [`crate::cpu`].
//!
//! When the feature is enabled but the kernel for a specific tensor
//! format is not yet implemented, the function returns an error with
//! the name of the unimplemented kernel.

/// Dispatch a matmul for a tensor with the given format, using Metal
/// when available, or falling back to CPU.
///
/// # Arguments
/// - `tensor_name`: name of the weight tensor being multiplied.
/// - `input`: the input activation vector (length `m`).
/// - `weight_data`: the quantized weight payload bytes from the `.cimage`.
/// - `dim_m`: output dimension.
/// - `dim_n`: input dimension (weight columns).
/// - `format`: quantization format of the weight tensor.
///
/// # Returns
/// Output vector of length `dim_m`.
#[cfg(feature = "metal-dispatch")]
pub fn dispatch_matmul(
    tensor_name: &str,
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
    format: &prism_ecs_ir::evolution::mutation_table::TensorFormat,
) -> Result<Vec<f32>, String> {
    match format {
        prism_ecs_ir::evolution::mutation_table::TensorFormat::Fp16
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Bf16
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Int8
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Int4
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Nf4
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Nf8
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Palettized4Bit
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Ternary158
        | prism_ecs_ir::evolution::mutation_table::TensorFormat::Binary1 => Err(format!(
            "Metal kernel for {format:?} not implemented — tensor {tensor_name} [{dim_m}×{dim_n}]"
        )),
    }
}

/// Dispatch matmul — Metal feature not enabled, delegate to CPU.
///
/// This path is used when `metal-dispatch` is not compiled in. It directly
/// calls the CPU matmul implementation (Accelerate / per-format dequant +
/// GEMV).
#[cfg(not(feature = "metal-dispatch"))]
pub fn dispatch_matmul(
    _tensor_name: &str,
    input: &[f32],
    weight_data: &[u8],
    dim_m: u32,
    dim_n: u32,
    format: &prism_ecs_ir::evolution::mutation_table::TensorFormat,
) -> Result<Vec<f32>, String> {
    crate::engine::cpu::matmul(input, weight_data, dim_m, dim_n, format)
}
