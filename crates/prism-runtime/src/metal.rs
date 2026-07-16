//! Metal dispatch — format-aware GPU kernel invocations.
//!
//! Gated behind `cfg(feature = "metal-dispatch")`. When disabled, imports
//! produce an empty module (the inference engine falls through to the CPU
//! path).

/// Dispatch a matmul for a tensor with the given format, using Metal.
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
    _tensor_name: &str,
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
    _format: &prism_ecs_ir::evolution::mutation_table::TensorFormat,
) -> Result<Vec<f32>, String> {
    Err("Metal GEMV dispatch not yet implemented — format-aware kernels (palettized GEMV, ternary, binary, NF4) are pending".to_string())
}

/// Stub: Metal dispatch when feature is off.
#[cfg(not(feature = "metal-dispatch"))]
pub fn dispatch_matmul(
    _tensor_name: &str,
    _input: &[f32],
    _weight_data: &[u8],
    _dim_m: u32,
    _dim_n: u32,
    _format: &prism_ecs_ir::evolution::mutation_table::TensorFormat,
) -> Result<Vec<f32>, String> {
    Err("Metal dispatch not enabled (compile with --features metal-dispatch)".to_string())
}
