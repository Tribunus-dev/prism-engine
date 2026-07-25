//! Certification — backend-vs-CPU-reference comparison of inference output.
//!
//! This module owns the canonical authority for verifying that a hardware
//! backend's inference output matches the CPU reference path within a
//! caller-supplied tolerance. It is invoked at admission time, at replay
//! time, and in nightly parity tests; it never mutates the model and never
//! drives execution beyond the single forward pass it certifies.
//!
//! All entry points assume the caller has already loaded a [`RuntimeModel`]
//! (see [`super::model`]) and supplies a `&dyn KernelBackend` for the
//! hardware path. The CPU reference path is internal to this module and
//! does not need to be visible to callers.

use prism_ecs_kernel::{
    BackendKind, CpuBackend, KernelBackend, KernelCompileRequest, KernelDispatchRequest,
    KernelVariant,
};

use super::model::RuntimeModel;
use super::{RuntimeError, decode_f32_output};

/// Outcome of comparing a backend's inference output against the CPU
/// reference implementation.
pub struct CertificationResult {
    /// Whether all output tensors matched within the specified tolerance.
    pub passed: bool,
    /// Maximum absolute error across all compared tensors.
    pub max_error: f32,
    /// Mean absolute error across all compared tensors.
    pub mean_error: f32,
    /// Names of tensors whose error exceeded the tolerance threshold.
    pub failed_tensors: Vec<String>,
}

/// Run CPU reference inference for certification.
///
/// Performs a forward pass using the CPU graph executor (when available)
/// and returns the raw logit vector. This is the correctness oracle that
/// all backend-accelerated paths are measured against.
///
/// The portable reference supports the canonical FP16, INT8, NF4, and
/// ternary Tile640 payload contracts. Unsupported custom variants still fail
/// closed rather than being mislabeled as CPU-compatible.
pub fn cpu_reference_inference(
    model: &RuntimeModel,
    input_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let (descriptor, inputs) = pack_fp16_gemv_inputs(model, input_tokens)?;
    let mut cpu_descriptor = descriptor;
    cpu_descriptor.backend = BackendKind::CPU;
    let cpu = CpuBackend;
    let artifact = cpu
        .compile(&KernelCompileRequest {
            source: b"prism-cpu-reference".to_vec(),
            descriptor: cpu_descriptor,
            source_path: None,
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    let output = cpu
        .dispatch(&KernelDispatchRequest {
            artifact,
            inputs,
            bindings: vec![],
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    decode_f32_output(output.outputs.first())
}

/// Run backend inference and compare with the CPU reference.
///
/// Dispatches inference on the given hardware backend, runs the CPU
/// reference path for the same inputs, compares every output tensor
/// element-wise within `tolerance`, and returns a [`CertificationResult`].
///
/// Certification currently supports the `FP16GEMV` and `FP16Matmul` kernel
/// contracts. It uses
/// identical packed inputs for both paths and reports numerical error rather
/// than treating successful dispatch as proof of correctness.
pub fn certify_inference(
    model: &RuntimeModel,
    input_tokens: &[u32],
    backend: &dyn KernelBackend,
    tolerance: f32,
) -> Result<CertificationResult, RuntimeError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(RuntimeError::ExecutionFailed(
            "certification tolerance must be finite and non-negative".into(),
        ));
    }
    let reference = cpu_reference_inference(model, input_tokens)?;
    let (descriptor, inputs) = pack_fp16_gemv_inputs(model, input_tokens)?;
    let name = descriptor.name.clone();
    let artifact = model.kernel_artifact(&name)?;
    let output = backend
        .dispatch(&KernelDispatchRequest {
            artifact,
            inputs,
            bindings: vec![],
        })
        .map_err(|error| RuntimeError::BackendError(error.to_string()))?;
    let actual = decode_f32_output(output.outputs.first())?;
    if actual.len() != reference.len() {
        return Err(RuntimeError::ExecutionFailed(format!(
            "certification output length mismatch: CPU {}, backend {}",
            reference.len(),
            actual.len()
        )));
    }
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    let mut failed = Vec::new();
    for (index, (expected, observed)) in reference.iter().zip(actual.iter()).enumerate() {
        let error = (expected - observed).abs();
        max_error = max_error.max(error);
        total_error += error;
        if error > tolerance {
            failed.push(format!("output[{index}]"));
        }
    }
    Ok(CertificationResult {
        passed: failed.is_empty(),
        max_error,
        mean_error: total_error / reference.len().max(1) as f32,
        failed_tensors: failed,
    })
}

fn pack_fp16_gemv_inputs(
    model: &RuntimeModel,
    input_tokens: &[u32],
) -> Result<(prism_ecs_kernel::KernelDescriptor, Vec<Vec<u8>>), RuntimeError> {
    if input_tokens.is_empty() {
        return Err(RuntimeError::ExecutionFailed(
            "inference requires input tokens".into(),
        ));
    }
    let name = model
        .kernel_descriptors
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| RuntimeError::KernelNotFound("no described kernels".into()))?;
    let artifact = model.kernel_artifact(&name)?;
    let descriptor = artifact
        .payloads
        .first()
        .ok_or_else(|| RuntimeError::KernelNotFound(name.clone()))?
        .descriptor
        .clone();
    if !matches!(
        descriptor.variant,
        KernelVariant::FP16GEMV
            | KernelVariant::FP16Matmul
            | KernelVariant::INT8Tile640
            | KernelVariant::NF4Tile640
            | KernelVariant::TernaryTile640(_)
    ) {
        return Err(RuntimeError::UnsupportedMode(format!(
            "CPU certification supports FP16GEMV, FP16Matmul, INT8Tile640, NF4Tile640, and TernaryTile640, got {:?}",
            descriptor.variant
        )));
    }
    let mut tensor_names = model.tensors.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    let first = tensor_names
        .first()
        .ok_or_else(|| RuntimeError::TensorNotFound("no kernel input tensor".into()))?;
    let first_data = model
        .tensors
        .get(first)
        .cloned()
        .ok_or_else(|| RuntimeError::TensorNotFound(first.clone()))?;
    if matches!(descriptor.variant, KernelVariant::FP16GEMV) {
        let token_bytes = input_tokens
            .iter()
            .flat_map(|token| (*token as f32).to_ne_bytes())
            .collect();
        Ok((descriptor, vec![first_data, token_bytes]))
    } else if matches!(descriptor.variant, KernelVariant::INT8Tile640) {
        let scales = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 weight scales tensor".into()))?;
        let input_scale = tensor_names
            .get(2)
            .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 input scale tensor".into()))?;
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("INT8 weights have no shape".into()))?;
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data,
                input_tokens
                    .iter()
                    .map(|token| *token as i8 as u8)
                    .collect(),
                model.tensors[scales].clone(),
                model.tensors[input_scale].clone(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else if matches!(descriptor.variant, KernelVariant::TernaryTile640(_)) {
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("ternary weights have no shape".into()))?;
        let pages = (record.dim_n as usize).div_ceil(640);
        let packed_len = record.dim_m as usize * pages * 4;
        let page_len = record.dim_m as usize * pages * 2;
        let lane_len = record.dim_m as usize * pages;
        if first_data.len() < packed_len + page_len + lane_len {
            return Err(RuntimeError::InvalidCImage(
                "ternary payload is truncated".into(),
            ));
        }
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data[..packed_len].to_vec(),
                input_tokens
                    .iter()
                    .flat_map(|token| half::f16::from_f32(*token as f32).to_le_bytes())
                    .collect(),
                first_data[packed_len..packed_len + page_len].to_vec(),
                first_data[packed_len + page_len..packed_len + page_len + lane_len].to_vec(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else if matches!(descriptor.variant, KernelVariant::NF4Tile640) {
        let scales = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weight scales tensor".into()))?;
        let biases = tensor_names
            .get(2)
            .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weight biases tensor".into()))?;
        let record = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("NF4 weights have no shape".into()))?;
        let dims = [record.dim_n, record.dim_m];
        Ok((
            descriptor,
            vec![
                first_data,
                input_tokens
                    .iter()
                    .flat_map(|token| (*token as f32).to_ne_bytes())
                    .collect(),
                model.tensors[scales].clone(),
                model.tensors[biases].clone(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    } else {
        let second = tensor_names
            .get(1)
            .ok_or_else(|| RuntimeError::TensorNotFound("no matrix B tensor".into()))?;
        let second_data = model
            .tensors
            .get(second)
            .cloned()
            .ok_or_else(|| RuntimeError::TensorNotFound(second.clone()))?;
        let a = model
            .tensor_records
            .get(first)
            .ok_or_else(|| RuntimeError::InvalidCImage("matrix A has no shape".into()))?;
        let b = model
            .tensor_records
            .get(second)
            .ok_or_else(|| RuntimeError::InvalidCImage("matrix B has no shape".into()))?;
        let dims = [a.dim_m, b.dim_n, a.dim_n];
        Ok((
            descriptor,
            vec![
                first_data,
                second_data,
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ],
        ))
    }
}
