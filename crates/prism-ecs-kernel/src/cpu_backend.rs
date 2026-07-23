//! Portable CPU reference backend for compiled kernel contracts.

use sha2::{Digest, Sha256};
use std::time::Instant;

use half::f16;

use crate::{
    BackendKind, KernelArtifact, KernelBackend, KernelCompileRequest, KernelDescriptor,
    KernelDispatchRequest, KernelError, KernelManifest, KernelMeasurement,
    KernelMeasurementRequest, KernelOutput, KernelPayload,
};

/// Portable CPU backend for deterministic compilation and FP16 GEMV execution.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl KernelBackend for CpuBackend {
    fn validate(&self, descriptor: &KernelDescriptor) -> Result<(), KernelError> {
        if descriptor.backend != BackendKind::CPU {
            return Err(KernelError::ValidationFailed(
                "descriptor is not CPU-targeted".into(),
            ));
        }
        Ok(())
    }

    fn compile(&self, request: &KernelCompileRequest) -> Result<KernelArtifact, KernelError> {
        self.validate(&request.descriptor)?;
        let mut source_hasher = Sha256::new();
        source_hasher.update(&request.source);
        let source_digest = hex::encode(source_hasher.finalize());
        let binary = format!("PRISM-CPU-REFERENCE\n{:?}", request.descriptor).into_bytes();
        let mut binary_hasher = Sha256::new();
        binary_hasher.update(&binary);
        let binary_digest = hex::encode(binary_hasher.finalize());
        let mut descriptor = request.descriptor.clone();
        descriptor.source_digest = source_digest;
        descriptor.binary_digest = binary_digest;
        let payload = KernelPayload {
            binary,
            descriptor: descriptor.clone(),
        };
        let manifest = KernelManifest {
            kernels: vec![descriptor],
            fusion_plan: None,
            manifest_digest: String::new(),
        };
        Ok(KernelArtifact {
            payloads: vec![payload],
            manifest,
            artifact_digest: hex::encode(Sha256::digest(b"prism-cpu-reference")),
        })
    }

    fn dispatch(&self, request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
        let start = Instant::now();
        let payload = request
            .artifact
            .payloads
            .first()
            .ok_or_else(|| KernelError::DispatchFailed("artifact has no CPU payload".into()))?;
        if matches!(
            payload.descriptor.variant,
            crate::KernelVariant::INT8Tile640
        ) {
            return dispatch_int8_gemv(&request.inputs);
        }
        if matches!(payload.descriptor.variant, crate::KernelVariant::NF4Tile640) {
            return dispatch_nf4_gemv(&request.inputs);
        }
        if matches!(
            payload.descriptor.variant,
            crate::KernelVariant::TernaryTile640(_)
        ) {
            return dispatch_ternary_gemv(&request.inputs);
        }
        if let crate::KernelVariant::Custom(name) = &payload.descriptor.variant {
            if name.starts_with("uop_") {
                return dispatch_uop_cpu(name, &request.inputs);
            }
        }
        if !matches!(
            payload.descriptor.variant,
            crate::KernelVariant::FP16GEMV | crate::KernelVariant::FP16Matmul
        ) {
            return Err(KernelError::DispatchFailed(format!(
                "CPU dispatch does not yet support {:?}",
                payload.descriptor.variant
            )));
        }
        if matches!(payload.descriptor.variant, crate::KernelVariant::FP16Matmul) {
            return dispatch_fp16_matmul(&request.inputs);
        }
        if request.inputs.len() < 2 {
            return Err(KernelError::BindingMismatch(
                "FP16 GEMV requires weights and input buffers".into(),
            ));
        }
        let weights = &request.inputs[0];
        let input = &request.inputs[1];
        if weights.len() % 2 != 0 || input.len() % 4 != 0 {
            return Err(KernelError::DispatchFailed(
                "unaligned FP16 GEMV buffers".into(),
            ));
        }
        let n = input.len() / 4;
        if n == 0 || weights.len() % (n * 2) != 0 {
            return Err(KernelError::DispatchFailed(
                "FP16 GEMV weight matrix does not match input width".into(),
            ));
        }
        let m = weights.len() / (n * 2);
        let input_values = input
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut output = vec![0.0f32; m];
        for (row, result) in output.iter_mut().enumerate() {
            let row_start = row * n * 2;
            for (col, input_value) in input_values.iter().enumerate() {
                let offset = row_start + col * 2;
                let weight = f16::from_le_bytes([weights[offset], weights[offset + 1]]).to_f32();
                *result += weight * input_value;
            }
        }
        Ok(KernelOutput {
            outputs: vec![output
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect()],
            dispatch_time_ns: start.elapsed().as_nanos() as u64,
        })
    }

    fn measure(
        &self,
        request: &KernelMeasurementRequest,
    ) -> Result<KernelMeasurement, KernelError> {
        if request.iterations == 0 {
            return Err(KernelError::MeasurementFailed(
                "iterations must be nonzero".into(),
            ));
        }
        let mut samples = Vec::with_capacity(request.iterations as usize);
        let bytes = request
            .artifact
            .payloads
            .first()
            .map(|payload| payload.binary.len())
            .unwrap_or(0);
        for _ in 0..request.iterations {
            let start = Instant::now();
            self.dispatch(&KernelDispatchRequest {
                artifact: request.artifact.clone(),
                inputs: request.inputs.clone(),
                bindings: vec![],
            })?;
            samples.push(start.elapsed().as_nanos() as f64);
        }
        let min_time_ns = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time_ns = samples.iter().copied().fold(0.0, f64::max);
        let avg_time_ns = samples.iter().sum::<f64>() / samples.len() as f64;
        Ok(KernelMeasurement {
            avg_time_ns,
            min_time_ns,
            max_time_ns,
            bandwidth_gbps: if avg_time_ns > 0.0 {
                bytes as f64 / avg_time_ns
            } else {
                0.0
            },
        })
    }

    fn name(&self) -> &str {
        "cpu-reference"
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, KernelError> {
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(KernelError::BindingMismatch(
            "UOp CPU buffers must be non-empty FP32".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn encode_f32(values: &[f32]) -> KernelOutput {
    KernelOutput {
        outputs: vec![values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()],
        dispatch_time_ns: 0,
    }
}

fn parse_shape(encoded: &str) -> Result<Vec<usize>, KernelError> {
    encoded
        .split('x')
        .map(|dimension| {
            dimension.parse::<usize>().map_err(|_| {
                KernelError::DispatchFailed(format!("invalid broadcast shape: {encoded}"))
            })
        })
        .collect()
}

fn broadcast_index(index: usize, output_shape: &[usize], input_shape: &[usize]) -> usize {
    if input_shape == output_shape {
        return index;
    }
    let rank_delta = output_shape.len() - input_shape.len();
    let mut input_index = 0;
    let mut stride = 1;
    for axis in (0..input_shape.len()).rev() {
        let output_axis = axis + rank_delta;
        let output_stride = output_shape[output_axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);
        let coordinate = (index / output_stride) % output_shape[output_axis];
        input_index += if input_shape[axis] == 1 {
            0
        } else {
            coordinate
        } * stride;
        stride *= input_shape[axis];
    }
    input_index
}

fn dispatch_broadcast_binary(name: &str, inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    let mut fields = name
        .strip_prefix("uop_broadcast_binary:")
        .unwrap_or_default()
        .split(':');
    let operation = fields.next().unwrap_or_default();
    let lhs_shape = parse_shape(fields.next().unwrap_or_default())?;
    let rhs_shape = parse_shape(fields.next().unwrap_or_default())?;
    let output_shape = parse_shape(fields.next().unwrap_or_default())?;
    let program = fields.next().unwrap_or_default();
    let lhs =
        decode_f32(inputs.first().ok_or_else(|| {
            KernelError::BindingMismatch("broadcast binary requires lhs".into())
        })?)?;
    let rhs =
        decode_f32(inputs.get(1).ok_or_else(|| {
            KernelError::BindingMismatch("broadcast binary requires rhs".into())
        })?)?;
    let elements = output_shape.iter().product::<usize>();
    let output = (0..elements)
        .map(|index| {
            let left = lhs[broadcast_index(index, &output_shape, &lhs_shape)];
            let right = rhs[broadcast_index(index, &output_shape, &rhs_shape)];
            let mut value = match operation {
                "add" => left + right,
                "mul" => left * right,
                "sub" => left - right,
                "div" => left / right,
                "maximum" => left.max(right),
                "minimum" => left.min(right),
                _ => left,
            };
            for unary in program.split(',').filter(|operation| !operation.is_empty()) {
                value = match unary {
                    "relu" => value.max(0.0),
                    "neg" => -value,
                    "exp" => value.exp(),
                    "sqrt" => value.sqrt(),
                    "abs" => value.abs(),
                    "log" => value.ln(),
                    "tanh" => value.tanh(),
                    "gelu" => {
                        0.5 * value
                            * (1.0 + (0.79788456 * (value + 0.044715 * value.powi(3))).tanh())
                    }
                    _ => value,
                };
            }
            value
        })
        .collect::<Vec<_>>();
    Ok(encode_f32(&output))
}

fn dispatch_uop_cpu(name: &str, inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    match name {
        _ if name.starts_with("uop_broadcast_binary:") => dispatch_broadcast_binary(name, inputs),
        "uop_where" => {
            let condition = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp where requires condition".into())
            })?)?;
            let when_true = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp where requires true value".into())
            })?)?;
            let when_false = decode_f32(inputs.get(2).ok_or_else(|| {
                KernelError::BindingMismatch("UOp where requires false value".into())
            })?)?;
            if condition.len() != when_true.len() || condition.len() != when_false.len() {
                return Err(KernelError::BindingMismatch(
                    "UOp where buffers must have equal lengths".into(),
                ));
            }
            Ok(encode_f32(
                &condition
                    .iter()
                    .zip(when_true)
                    .zip(when_false)
                    .map(|((condition, when_true), when_false)| {
                        if *condition != 0.0 {
                            when_true
                        } else {
                            when_false
                        }
                    })
                    .collect::<Vec<_>>(),
            ))
        }
        _ if name.starts_with("uop_elementwise_scalar:") => {
            let encoded = name.split(':').nth(1).unwrap_or_default();
            let mut fields = encoded.split('|');
            let operation = fields.next().unwrap_or_default();
            let scalar = fields
                .next()
                .and_then(|value| value.parse::<f32>().ok())
                .ok_or_else(|| {
                    KernelError::DispatchFailed("CPU scalar UOp variant has invalid value".into())
                })?;
            let scalar_left = fields.next() == Some("1");
            if !matches!(
                operation,
                "add" | "mul" | "sub" | "div" | "maximum" | "minimum"
            ) {
                return Err(KernelError::DispatchFailed(format!(
                    "CPU scalar UOp operation is unsupported: {operation}"
                )));
            }
            let input = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp scalar elementwise requires input".into())
            })?)?;
            let output = input
                .into_iter()
                .map(|value| match operation {
                    "add" => {
                        if scalar_left {
                            scalar + value
                        } else {
                            value + scalar
                        }
                    }
                    "mul" => value * scalar,
                    "sub" => {
                        if scalar_left {
                            scalar - value
                        } else {
                            value - scalar
                        }
                    }
                    "div" => {
                        if scalar_left {
                            scalar / value
                        } else {
                            value / scalar
                        }
                    }
                    "maximum" => value.max(scalar),
                    "minimum" => value.min(scalar),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_elementwise_program:") => {
            let program = name.split(':').nth(1).unwrap_or_default();
            let mut output = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp fused elementwise requires lhs".into())
            })?)?;
            let rhs = inputs.get(1).map(|bytes| decode_f32(bytes)).transpose()?;
            for operation in program.split(',') {
                let binary = matches!(
                    operation,
                    "add" | "mul" | "sub" | "div" | "maximum" | "minimum"
                );
                if binary {
                    let right = rhs.as_ref().ok_or_else(|| {
                        KernelError::BindingMismatch("UOp fused binary program requires rhs".into())
                    })?;
                    if output.len() != right.len() {
                        return Err(KernelError::BindingMismatch(
                            "UOp fused binary buffers must have equal lengths".into(),
                        ));
                    }
                }
                if !matches!(
                    operation,
                    "add"
                        | "mul"
                        | "sub"
                        | "div"
                        | "maximum"
                        | "minimum"
                        | "relu"
                        | "neg"
                        | "exp"
                        | "sqrt"
                        | "abs"
                        | "log"
                        | "tanh"
                        | "sin"
                        | "cos"
                        | "gelu"
                ) {
                    return Err(KernelError::DispatchFailed(format!(
                        "CPU UOp fused operation is unsupported: {operation}"
                    )));
                }
                output = output
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| match operation {
                        "add" => value + rhs.as_ref().unwrap()[index],
                        "mul" => value * rhs.as_ref().unwrap()[index],
                        "sub" => value - rhs.as_ref().unwrap()[index],
                        "div" => value / rhs.as_ref().unwrap()[index],
                        "maximum" => value.max(rhs.as_ref().unwrap()[index]),
                        "minimum" => value.min(rhs.as_ref().unwrap()[index]),
                        "relu" => value.max(0.0),
                        "neg" => -value,
                        "exp" => value.exp(),
                        "sqrt" => value.sqrt(),
                        "abs" => value.abs(),
                        "log" => value.ln(),
                        "tanh" => value.tanh(),
                        "sin" => value.sin(),
                        "cos" => value.cos(),
                        "gelu" => {
                            0.5 * value
                                * (1.0
                                    + (std::f32::consts::FRAC_2_SQRT_PI
                                        * (value + 0.044715 * value.powi(3)))
                                    .tanh())
                        }
                        _ => value,
                    })
                    .collect();
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_elementwise_binary:") => {
            let operation = name.split(':').nth(1).unwrap_or_default();
            let lhs = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp binary elementwise requires lhs".into())
            })?)?;
            let rhs = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp binary elementwise requires rhs".into())
            })?)?;
            if lhs.len() != rhs.len() {
                return Err(KernelError::BindingMismatch(
                    "UOp binary elementwise buffers must have equal lengths".into(),
                ));
            }
            if !matches!(
                operation,
                "add" | "mul" | "sub" | "div" | "maximum" | "minimum"
            ) {
                return Err(KernelError::DispatchFailed(format!(
                    "CPU UOp binary operation is unsupported: {operation}"
                )));
            }
            let output = lhs
                .into_iter()
                .zip(rhs)
                .map(|(left, right)| match operation {
                    "add" => left + right,
                    "mul" => left * right,
                    "sub" => left - right,
                    "div" => left / right,
                    "maximum" => left.max(right),
                    "minimum" => left.min(right),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_elementwise:") => {
            let program = name.split(':').nth(1).unwrap_or_default();
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp elementwise requires input".into())
            })?)?;
            let mut output = x;
            for operation in program.split(',') {
                if !matches!(
                    operation,
                    "relu"
                        | "neg"
                        | "exp"
                        | "sqrt"
                        | "abs"
                        | "log"
                        | "tanh"
                        | "sin"
                        | "cos"
                        | "gelu"
                ) {
                    return Err(KernelError::DispatchFailed(format!(
                        "CPU UOp elementwise operation is unsupported: {operation}"
                    )));
                }
                output = output
                    .into_iter()
                    .map(|value| match operation {
                        "relu" => value.max(0.0),
                        "neg" => -value,
                        "exp" => value.exp(),
                        "sqrt" => value.sqrt(),
                        "abs" => value.abs(),
                        "log" => value.ln(),
                        "tanh" => value.tanh(),
                        "sin" => value.sin(),
                        "cos" => value.cos(),
                        "gelu" => {
                            0.5 * value
                                * (1.0
                                    + (std::f32::consts::FRAC_2_SQRT_PI
                                        * (value + 0.044715 * value.powi(3)))
                                    .tanh())
                        }
                        _ => value,
                    })
                    .collect();
            }
            Ok(encode_f32(&output))
        }
        "uop_reduce_sum" => {
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp reduction requires input".into())
            })?)?;
            Ok(encode_f32(&[x.iter().sum()]))
        }
        "uop_reduce_max" => {
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp max reduction requires input".into())
            })?)?;
            Ok(encode_f32(&[x
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)]))
        }
        "uop_reduce_min" => {
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp min reduction requires input".into())
            })?)?;
            Ok(encode_f32(&[x
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)]))
        }
        _ if name.starts_with("uop_reduce_sum_axis:") => {
            let mut dims = name.split(':').skip(1).map(|value| value.parse::<usize>());
            let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp axis reduction variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp axis reduction requires input".into())
            })?)?;
            if x.len() != outer * reduce * inner {
                return Err(KernelError::BindingMismatch(
                    "UOp axis reduction input does not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; outer * inner];
            for out in 0..outer {
                for col in 0..inner {
                    output[out * inner + col] = (0..reduce)
                        .map(|step| x[(out * reduce + step) * inner + col])
                        .sum();
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_reduce_max_axis:") => {
            let mut dims = name.split(':').skip(1).map(|value| value.parse::<usize>());
            let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp max axis reduction variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp max axis reduction requires input".into())
            })?)?;
            if x.len() != outer * reduce * inner {
                return Err(KernelError::BindingMismatch(
                    "UOp max axis reduction input does not match dimensions".into(),
                ));
            }
            let mut output = vec![f32::NEG_INFINITY; outer * inner];
            for outer_index in 0..outer {
                for inner_index in 0..inner {
                    for step in 0..reduce {
                        output[outer_index * inner + inner_index] = output
                            [outer_index * inner + inner_index]
                            .max(x[(outer_index * reduce + step) * inner + inner_index]);
                    }
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_reduce_min_axis:") => {
            let mut dims = name.split(':').skip(1).map(|value| value.parse::<usize>());
            let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp min axis reduction variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp min axis reduction requires input".into())
            })?)?;
            if x.len() != outer * reduce * inner {
                return Err(KernelError::BindingMismatch(
                    "UOp min axis reduction input does not match dimensions".into(),
                ));
            }
            let mut output = vec![f32::INFINITY; outer * inner];
            for outer_index in 0..outer {
                for inner_index in 0..inner {
                    for step in 0..reduce {
                        output[outer_index * inner + inner_index] = output
                            [outer_index * inner + inner_index]
                            .min(x[(outer_index * reduce + step) * inner + inner_index]);
                    }
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_softmax_axis:") => {
            let mut dims = name.split(':').skip(1).map(|value| value.parse::<usize>());
            let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp softmax variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp softmax requires input".into())
            })?)?;
            if x.len() != outer * reduce * inner {
                return Err(KernelError::BindingMismatch(
                    "UOp softmax input does not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; x.len()];
            for out in 0..outer {
                for col in 0..inner {
                    let max = (0..reduce)
                        .map(|step| x[(out * reduce + step) * inner + col])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let denom: f32 = (0..reduce)
                        .map(|step| (x[(out * reduce + step) * inner + col] - max).exp())
                        .sum();
                    for step in 0..reduce {
                        let index = (out * reduce + step) * inner + col;
                        output[index] = (x[index] - max).exp() / denom;
                    }
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_attention_batched:") => {
            dispatch_uop_attention_cpu(name, inputs, true)
        }
        _ if name.starts_with("uop_attention:") => dispatch_uop_attention_cpu(name, inputs, false),
        _ if name.starts_with("uop_matmul:") => {
            let mut dims = name.split(':').skip(1).map(|value| value.parse::<usize>());
            let (Some(Ok(m)), Some(Ok(k)), Some(Ok(n))) = (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp matmul variant has invalid dimensions".into(),
                ));
            };
            let a =
                decode_f32(inputs.get(0).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp matmul requires A".into())
                })?)?;
            let b =
                decode_f32(inputs.get(1).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp matmul requires B".into())
                })?)?;
            if a.len() != m * k || b.len() != k * n {
                return Err(KernelError::BindingMismatch(
                    "UOp matmul buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; m * n];
            for row in 0..m {
                for col in 0..n {
                    output[row * n + col] = (0..k)
                        .map(|inner| a[row * k + inner] * b[inner * n + col])
                        .sum();
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_rms_norm:") => {
            let mut parts = name.split(':').skip(1);
            let (Some(Ok(rows)), Some(Ok(features)), Some(Ok(epsilon))) = (
                parts.next().map(|v| v.parse::<usize>()),
                parts.next().map(|v| v.parse::<usize>()),
                parts.next().map(|v| v.parse::<f32>()),
            ) else {
                return Err(KernelError::DispatchFailed(
                    "UOp RMSNorm variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.get(0).ok_or_else(|| {
                KernelError::BindingMismatch("UOp RMSNorm requires input".into())
            })?)?;
            let weight = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp RMSNorm requires weight".into())
            })?)?;
            if x.len() != rows * features || weight.len() != features {
                return Err(KernelError::BindingMismatch(
                    "UOp RMSNorm buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; x.len()];
            for row in 0..rows {
                let base = row * features;
                let mean =
                    x[base..base + features].iter().map(|v| v * v).sum::<f32>() / features as f32;
                let inv = (mean + epsilon).sqrt().recip();
                for col in 0..features {
                    output[base + col] = x[base + col] * inv * weight[col];
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_layer_norm:") => {
            let mut parts = name.split(':').skip(1);
            let (Some(Ok(rows)), Some(Ok(features)), Some(Ok(epsilon))) = (
                parts.next().map(|v| v.parse::<usize>()),
                parts.next().map(|v| v.parse::<usize>()),
                parts.next().map(|v| v.parse::<f32>()),
            ) else {
                return Err(KernelError::DispatchFailed(
                    "UOp LayerNorm variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.get(0).ok_or_else(|| {
                KernelError::BindingMismatch("UOp LayerNorm requires input".into())
            })?)?;
            let weight = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp LayerNorm requires weight".into())
            })?)?;
            let bias = decode_f32(inputs.get(2).ok_or_else(|| {
                KernelError::BindingMismatch("UOp LayerNorm requires bias".into())
            })?)?;
            if x.len() != rows * features || weight.len() != features || bias.len() != features {
                return Err(KernelError::BindingMismatch(
                    "UOp LayerNorm buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; x.len()];
            for row in 0..rows {
                let base = row * features;
                let mean = x[base..base + features].iter().sum::<f32>() / features as f32;
                let variance = x[base..base + features]
                    .iter()
                    .map(|value| {
                        let centered = *value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / features as f32;
                let inv = (variance + epsilon).sqrt().recip();
                for col in 0..features {
                    output[base + col] = (x[base + col] - mean) * inv * weight[col] + bias[col];
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_rope:") => {
            let mut dims = name.split(':').skip(1).map(|v| v.parse::<usize>());
            let (Some(Ok(rows)), Some(Ok(features))) = (dims.next(), dims.next()) else {
                return Err(KernelError::DispatchFailed(
                    "UOp RoPE variant has invalid dimensions".into(),
                ));
            };
            if features == 0 || features % 2 != 0 {
                return Err(KernelError::BindingMismatch(
                    "UOp RoPE feature dimension must be positive and even".into(),
                ));
            }
            let x =
                decode_f32(inputs.get(0).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp RoPE requires input".into())
                })?)?;
            let cos = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp RoPE requires cosine input".into())
            })?)?;
            let sin = decode_f32(inputs.get(2).ok_or_else(|| {
                KernelError::BindingMismatch("UOp RoPE requires sine input".into())
            })?)?;
            let half = features / 2;
            if x.len() != rows * features || cos.len() != rows * half || sin.len() != rows * half {
                return Err(KernelError::BindingMismatch(
                    "UOp RoPE buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; x.len()];
            for row in 0..rows {
                for pair in 0..half {
                    let base = row * features + pair * 2;
                    let angle = row * half + pair;
                    let a = x[base];
                    let b = x[base + 1];
                    output[base] = a * cos[angle] - b * sin[angle];
                    output[base + 1] = a * sin[angle] + b * cos[angle];
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_gather:") => {
            let mut dims = name.split(':').skip(1).map(|v| v.parse::<usize>());
            let (Some(Ok(rows)), Some(Ok(vocab)), Some(Ok(features))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp Gather variant has invalid dimensions".into(),
                ));
            };
            let weight = decode_f32(inputs.get(0).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Gather requires weight".into())
            })?)?;
            let indices = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Gather requires indices".into())
            })?)?;
            if weight.len() != vocab * features || indices.len() != rows {
                return Err(KernelError::BindingMismatch(
                    "UOp Gather buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; rows * features];
            for row in 0..rows {
                let index = indices[row];
                if !index.is_finite()
                    || index < 0.0
                    || index.fract() != 0.0
                    || index >= vocab as f32
                {
                    return Err(KernelError::BindingMismatch(
                        "UOp Gather index is out of range".into(),
                    ));
                }
                let source = index as usize * features;
                output[row * features..(row + 1) * features]
                    .copy_from_slice(&weight[source..source + features]);
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_scatter:") => {
            let mut dims = name.split(':').skip(1).map(|v| v.parse::<usize>());
            let (Some(Ok(rows)), Some(Ok(updates)), Some(Ok(features))) =
                (dims.next(), dims.next(), dims.next())
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp Scatter variant has invalid dimensions".into(),
                ));
            };
            let base = decode_f32(inputs.get(0).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Scatter requires base".into())
            })?)?;
            let indices = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Scatter requires indices".into())
            })?)?;
            let updates_value = decode_f32(inputs.get(2).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Scatter requires updates".into())
            })?)?;
            if base.len() != rows * features
                || indices.len() != updates
                || updates_value.len() != updates * features
            {
                return Err(KernelError::BindingMismatch(
                    "UOp Scatter buffers do not match dimensions".into(),
                ));
            }
            let mut output = base;
            for update in 0..updates {
                let index = indices[update];
                if !index.is_finite() || index < 0.0 || index.fract() != 0.0 || index >= rows as f32
                {
                    return Err(KernelError::BindingMismatch(
                        "UOp Scatter index is out of range".into(),
                    ));
                }
                let destination = index as usize * features;
                let source = update * features;
                output[destination..destination + features]
                    .copy_from_slice(&updates_value[source..source + features]);
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_ssm:") => {
            let mut dims = name.split(':').skip(1).map(|v| v.parse::<usize>());
            let (Some(Ok(rows)), Some(Ok(features))) = (dims.next(), dims.next()) else {
                return Err(KernelError::DispatchFailed(
                    "UOp SSM variant has invalid dimensions".into(),
                ));
            };
            let input =
                decode_f32(inputs.get(0).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp SSM requires input".into())
                })?)?;
            let decay =
                decode_f32(inputs.get(1).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp SSM requires decay".into())
                })?)?;
            let input_gain = decode_f32(inputs.get(2).ok_or_else(|| {
                KernelError::BindingMismatch("UOp SSM requires input gain".into())
            })?)?;
            let output_gain = decode_f32(inputs.get(3).ok_or_else(|| {
                KernelError::BindingMismatch("UOp SSM requires output gain".into())
            })?)?;
            if input.len() != rows * features
                || decay.len() != features
                || input_gain.len() != features
                || output_gain.len() != features
            {
                return Err(KernelError::BindingMismatch(
                    "UOp SSM buffers do not match dimensions".into(),
                ));
            }
            let mut state = vec![0.0; features];
            let mut output = vec![0.0; input.len()];
            for row in 0..rows {
                for feature in 0..features {
                    let index = row * features + feature;
                    state[feature] =
                        decay[feature] * state[feature] + input_gain[feature] * input[index];
                    output[index] = output_gain[feature] * state[feature];
                }
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_transpose:") => {
            let mut parts = name.strip_prefix("uop_transpose:").unwrap().split(':');
            let permutation = parts
                .next()
                .unwrap()
                .split(',')
                .map(|v| v.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    KernelError::DispatchFailed("UOp transpose permutation is invalid".into())
                })?;
            let input_shape = parts
                .next()
                .unwrap()
                .split('x')
                .map(|v| v.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    KernelError::DispatchFailed("UOp transpose input shape is invalid".into())
                })?;
            let output_shape = parts
                .next()
                .unwrap()
                .split('x')
                .map(|v| v.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    KernelError::DispatchFailed("UOp transpose output shape is invalid".into())
                })?;
            let x = decode_f32(inputs.first().ok_or_else(|| {
                KernelError::BindingMismatch("UOp transpose requires input".into())
            })?)?;
            if permutation.len() != input_shape.len()
                || permutation.iter().any(|axis| *axis >= input_shape.len())
            {
                return Err(KernelError::BindingMismatch(
                    "UOp transpose permutation rank mismatch".into(),
                ));
            }
            let input_strides: Vec<usize> = (0..input_shape.len())
                .map(|axis| input_shape[axis + 1..].iter().product())
                .collect();
            let mut output = vec![0.0; output_shape.iter().product()];
            for out_linear in 0..output.len() {
                let mut remainder = out_linear;
                let mut source_linear = 0;
                for out_axis in (0..output_shape.len()).rev() {
                    let coordinate = remainder % output_shape[out_axis];
                    remainder /= output_shape[out_axis];
                    source_linear += coordinate * input_strides[permutation[out_axis]];
                }
                output[out_linear] = x[source_linear];
            }
            Ok(encode_f32(&output))
        }
        _ if name.starts_with("uop_conv2d:") => {
            let mut dims = name.split(':').skip(1).map(|v| v.parse::<usize>());
            let (
                Some(Ok(batch)),
                Some(Ok(in_channels)),
                Some(Ok(height)),
                Some(Ok(width)),
                Some(Ok(out_channels)),
                Some(Ok(kernel_h)),
                Some(Ok(kernel_w)),
                Some(Ok(stride)),
                Some(Ok(padding)),
            ) = (
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
                dims.next(),
            )
            else {
                return Err(KernelError::DispatchFailed(
                    "UOp Conv2D variant has invalid dimensions".into(),
                ));
            };
            let x = decode_f32(inputs.get(0).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Conv2D requires input".into())
            })?)?;
            let weight = decode_f32(inputs.get(1).ok_or_else(|| {
                KernelError::BindingMismatch("UOp Conv2D requires weight".into())
            })?)?;
            let bias =
                decode_f32(inputs.get(2).ok_or_else(|| {
                    KernelError::BindingMismatch("UOp Conv2D requires bias".into())
                })?)?;
            let out_h = (height + 2 * padding - kernel_h) / stride + 1;
            let out_w = (width + 2 * padding - kernel_w) / stride + 1;
            if x.len() != batch * in_channels * height * width
                || weight.len() != out_channels * in_channels * kernel_h * kernel_w
                || bias.len() != out_channels
            {
                return Err(KernelError::BindingMismatch(
                    "UOp Conv2D buffers do not match dimensions".into(),
                ));
            }
            let mut output = vec![0.0; batch * out_channels * out_h * out_w];
            for b in 0..batch {
                for oc in 0..out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = bias[oc];
                            for ic in 0..in_channels {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let ih = oh * stride + kh;
                                        let iw = ow * stride + kw;
                                        if ih >= padding
                                            && iw >= padding
                                            && ih - padding < height
                                            && iw - padding < width
                                        {
                                            let sh = ih - padding;
                                            let sw = iw - padding;
                                            sum += x[((b * in_channels + ic) * height + sh)
                                                * width
                                                + sw]
                                                * weight[(((oc * in_channels + ic) * kernel_h
                                                    + kh)
                                                    * kernel_w)
                                                    + kw];
                                        }
                                    }
                                }
                            }
                            output[((b * out_channels + oc) * out_h + oh) * out_w + ow] = sum;
                        }
                    }
                }
            }
            Ok(encode_f32(&output))
        }
        _ => Err(KernelError::DispatchFailed(format!(
            "CPU UOp dispatch does not support {name}"
        ))),
    }
}

fn dispatch_uop_attention_cpu(
    name: &str,
    inputs: &[Vec<u8>],
    batched: bool,
) -> Result<KernelOutput, KernelError> {
    let mut parts = name.split(':').skip(1);
    let (batch, seq, head, scale) = if batched {
        let (Some(Ok(batch)), Some(Ok(seq)), Some(Ok(head)), Some(Ok(scale))) = (
            parts.next().map(|v| v.parse::<usize>()),
            parts.next().map(|v| v.parse::<usize>()),
            parts.next().map(|v| v.parse::<usize>()),
            parts.next().map(|v| v.parse::<f32>()),
        ) else {
            return Err(KernelError::DispatchFailed(
                "UOp batched attention variant has invalid dimensions".into(),
            ));
        };
        (batch, seq, head, scale)
    } else {
        let (Some(Ok(seq)), Some(Ok(head)), Some(Ok(scale))) = (
            parts.next().map(|v| v.parse::<usize>()),
            parts.next().map(|v| v.parse::<usize>()),
            parts.next().map(|v| v.parse::<f32>()),
        ) else {
            return Err(KernelError::DispatchFailed(
                "UOp attention variant has invalid dimensions".into(),
            ));
        };
        (1, seq, head, scale)
    };
    let q = decode_f32(
        inputs
            .get(0)
            .ok_or_else(|| KernelError::BindingMismatch("UOp attention requires Q".into()))?,
    )?;
    let k = decode_f32(
        inputs
            .get(1)
            .ok_or_else(|| KernelError::BindingMismatch("UOp attention requires K".into()))?,
    )?;
    let v = decode_f32(
        inputs
            .get(2)
            .ok_or_else(|| KernelError::BindingMismatch("UOp attention requires V".into()))?,
    )?;
    if q.len() != batch * seq * head || k.len() != q.len() || v.len() != q.len() {
        return Err(KernelError::BindingMismatch(
            "UOp attention buffers do not match dimensions".into(),
        ));
    }
    let mut output = vec![0.0; q.len()];
    for b in 0..batch {
        let base = b * seq * head;
        for query in 0..seq {
            let mut scores = vec![0.0; seq];
            for key in 0..seq {
                scores[key] = (0..head)
                    .map(|d| q[base + query * head + d] * k[base + key * head + d])
                    .sum::<f32>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
            let denom: f32 = weights.iter().sum();
            for d in 0..head {
                output[base + query * head + d] = (0..seq)
                    .map(|key| weights[key] / denom * v[base + key * head + d])
                    .sum();
            }
        }
    }
    Ok(encode_f32(&output))
}

/// CPU reference for the canonical packed ternary Tile640 ABI.
fn dispatch_ternary_gemv(inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    let start = Instant::now();
    if inputs.len() < 5 || inputs[1].len() % 2 != 0 || inputs[4].len() != 8 {
        return Err(KernelError::BindingMismatch(
            "ternary GEMV requires packed weights, FP16 input, page scales, lane scales, and dimensions".into(),
        ));
    }
    let input_dim = u32::from_ne_bytes(inputs[4][0..4].try_into().unwrap()) as usize;
    let output_dim = u32::from_ne_bytes(inputs[4][4..8].try_into().unwrap()) as usize;
    if input_dim == 0 || output_dim == 0 || inputs[1].len() != input_dim * 2 {
        return Err(KernelError::DispatchFailed(
            "ternary GEMV dimensions do not match input".into(),
        ));
    }
    let pages = input_dim.div_ceil(640);
    let words_per_row = pages * 32;
    let packed_len = output_dim * words_per_row * 4;
    let page_len = output_dim * pages * 2;
    let lane_len = output_dim * words_per_row;
    if inputs[0].len() != packed_len || inputs[2].len() != page_len || inputs[3].len() != lane_len {
        return Err(KernelError::DispatchFailed(
            "ternary GEMV payload lengths do not match dimensions".into(),
        ));
    }
    let input: Vec<f32> = inputs[1]
        .chunks_exact(2)
        .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
        .collect();
    let mut output = vec![0.0f32; output_dim];
    for (row, result) in output.iter_mut().enumerate() {
        for word_index in 0..words_per_row {
            let page = word_index / 32;
            let column_start = page * 640 + (word_index % 32) * 20;
            let word_offset = (row * words_per_row + word_index) * 4;
            let mut word =
                u32::from_le_bytes(inputs[0][word_offset..word_offset + 4].try_into().unwrap());
            let page_bits = u16::from_le_bytes(
                inputs[2][(row * pages + page) * 2..][..2]
                    .try_into()
                    .unwrap(),
            );
            let page_scale = f32::from(half::bf16::from_bits(page_bits));
            let lane_scale = inputs[3][row * words_per_row + word_index] as i8 as f32;
            let scale = page_scale * (lane_scale / 127.0);
            for offset in 0..20 {
                let column = column_start + offset;
                if column >= input_dim {
                    break;
                }
                match word % 3 {
                    1 => *result += input[column] * scale,
                    2 => *result -= input[column] * scale,
                    _ => {}
                }
                word /= 3;
            }
        }
    }
    Ok(KernelOutput {
        outputs: vec![output
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_fp16_matmul(inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    if inputs.len() < 3 {
        return Err(KernelError::BindingMismatch(
            "FP16 matmul requires A, B, and dimensions buffers".into(),
        ));
    }
    let dims = &inputs[2];
    if dims.len() < 12 {
        return Err(KernelError::DispatchFailed(
            "FP16 matmul dimensions buffer is truncated".into(),
        ));
    }
    let read_dim =
        |offset| u32::from_ne_bytes(dims[offset..offset + 4].try_into().unwrap()) as usize;
    let m = read_dim(0);
    let n = read_dim(4);
    let k = read_dim(8);
    if m == 0 || n == 0 || k == 0 || inputs[0].len() < m * k * 2 || inputs[1].len() < k * n * 2 {
        return Err(KernelError::DispatchFailed(
            "FP16 matmul dimensions do not match buffer lengths".into(),
        ));
    }
    let mut output = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut value = 0.0f32;
            for inner in 0..k {
                let a_offset = (row * k + inner) * 2;
                let b_offset = (inner * n + col) * 2;
                let a = f16::from_le_bytes([inputs[0][a_offset], inputs[0][a_offset + 1]]).to_f32();
                let b = f16::from_le_bytes([inputs[1][b_offset], inputs[1][b_offset + 1]]).to_f32();
                value += a * b;
            }
            output[row * n + col] = value;
        }
    }
    Ok(KernelOutput {
        outputs: vec![output
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_int8_gemv(inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    if inputs.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "INT8 GEMV requires weights, input, weight scales, input scale, and dimensions".into(),
        ));
    }
    let weights = &inputs[0];
    let input = &inputs[1];
    let weight_scales = &inputs[2];
    let input_scale = &inputs[3];
    let dims = &inputs[4];
    if input_scale.len() != 4 || dims.len() != 8 {
        return Err(KernelError::DispatchFailed(
            "invalid INT8 GEMV metadata".into(),
        ));
    }
    let input_dim = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let output_dim = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    if input_dim == 0
        || output_dim == 0
        || input.len() != input_dim
        || weights.len() != input_dim * output_dim
        || weight_scales.len() != output_dim * 4
    {
        return Err(KernelError::DispatchFailed(
            "INT8 GEMV dimensions do not match buffers".into(),
        ));
    }
    let activation_scale = f32::from_ne_bytes(input_scale[..4].try_into().unwrap());
    let mut output = vec![0.0f32; output_dim];
    for row in 0..output_dim {
        let mut acc = 0i32;
        for col in 0..input_dim {
            acc += (weights[row * input_dim + col] as i8) as i32 * (input[col] as i8) as i32;
        }
        let scale = f32::from_ne_bytes(weight_scales[row * 4..row * 4 + 4].try_into().unwrap());
        output[row] = acc as f32 * scale * activation_scale;
    }
    Ok(KernelOutput {
        outputs: vec![output
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_nf4_gemv(inputs: &[Vec<u8>]) -> Result<KernelOutput, KernelError> {
    const TILE: usize = 640;
    const GROUP: usize = 128;
    const GROUPS: usize = TILE / GROUP;
    const CODES_PER_TILE: usize = TILE / 2;
    const NF4: [f32; 16] = [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ];
    if inputs.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "NF4 GEMV requires codes, input, scales, biases, and dimensions".into(),
        ));
    }
    let codes = &inputs[0];
    let input = &inputs[1];
    let scales = &inputs[2];
    let biases = &inputs[3];
    let dims = &inputs[4];
    if dims.len() != 8 || input.len() % 4 != 0 {
        return Err(KernelError::DispatchFailed(
            "invalid NF4 GEMV metadata".into(),
        ));
    }
    let input_dim = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let output_dim = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    let tiles = input_dim.div_ceil(TILE);
    let expected_codes = output_dim * tiles * CODES_PER_TILE;
    let expected_groups = output_dim * tiles * GROUPS;
    if input_dim == 0
        || output_dim == 0
        || codes.len() != expected_codes
        || scales.len() != expected_groups * 4
        || biases.len() != expected_groups * 4
        || input.len() != input_dim * 4
    {
        return Err(KernelError::DispatchFailed(
            "NF4 GEMV dimensions do not match buffers".into(),
        ));
    }
    let activations = input
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut output = vec![0.0f32; output_dim];
    for row in 0..output_dim {
        let mut sum = 0.0f32;
        for col in 0..input_dim {
            let tile = col / TILE;
            let within_tile = col % TILE;
            let byte_index = (row * tiles + tile) * CODES_PER_TILE + within_tile / 2;
            let packed = codes[byte_index];
            let code = if within_tile % 2 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let group = within_tile / GROUP;
            let scale_index = (row * tiles + tile) * GROUPS + group;
            let scale = f32::from_ne_bytes(
                scales[scale_index * 4..scale_index * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            let bias = f32::from_ne_bytes(
                biases[scale_index * 4..scale_index * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            sum += (NF4[code as usize] * scale + bias) * activations[col];
        }
        output[row] = sum;
    }
    Ok(KernelOutput {
        outputs: vec![output
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()],
        dispatch_time_ns: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DispatchGeometry, KernelDispatchRequest, KernelVariant};

    fn uop_artifact(variant: &str) -> KernelArtifact {
        let descriptor = KernelDescriptor {
            name: "uop-test".into(),
            variant: KernelVariant::Custom(variant.into()),
            backend: BackendKind::CPU,
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: Vec::new(),
            dispatch_geometry: DispatchGeometry {
                threads_per_threadgroup: [1, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                threads_per_grid: [1, 1, 1],
            },
        };
        KernelArtifact {
            payloads: vec![KernelPayload {
                binary: Vec::new(),
                descriptor: descriptor.clone(),
            }],
            manifest: KernelManifest {
                kernels: vec![descriptor],
                fusion_plan: None,
                manifest_digest: String::new(),
            },
            artifact_digest: String::new(),
        }
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn dispatches_compiled_uop_matmul_and_rms_norm() {
        let matmul = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_matmul:2:2:2"),
                inputs: vec![bytes(&[1.0, 2.0, 3.0, 4.0]), bytes(&[2.0, 0.0, 1.0, 2.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = matmul.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![4.0, 4.0, 10.0, 8.0]);
        let norm = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_rms_norm:1:2:0.00001"),
                inputs: vec![bytes(&[3.0, 4.0]), bytes(&[2.0, 0.5])],
                bindings: Vec::new(),
            })
            .unwrap();
        let first = f32::from_ne_bytes(norm.outputs[0][..4].try_into().unwrap());
        assert!((first - 1.697056).abs() < 1e-3);
    }

    #[test]
    fn dispatches_shape_aware_broadcast_binary() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_broadcast_binary:add:2x3:3:2x3"),
                inputs: vec![
                    bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    bytes(&[10.0, 20.0, 30.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn dispatches_fused_broadcast_binary_and_unary_program() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_broadcast_binary:add:2x3:3:2x3:relu"),
                inputs: vec![
                    bytes(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]),
                    bytes(&[0.0, -3.0, 4.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0.0, 0.0, 1.0, 4.0, 0.0, 10.0]);
    }

    #[test]
    fn dispatches_compiled_uop_batched_attention() {
        let artifact = uop_artifact("uop_attention_batched:2:2:1:1");
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs: vec![
                    bytes(&[1.0, 0.0, 0.0, 1.0]),
                    bytes(&[1.0, 0.0, 0.0, 1.0]),
                    bytes(&[2.0, 4.0, 8.0, 16.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 4);
        assert!(values[0] > 2.0 && values[0] < 8.0);
        assert!(values[2] > values[0]);
    }

    #[test]
    fn dispatches_compiled_uop_reduce_max() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_reduce_max"),
                inputs: vec![bytes(&[-2.0, 7.0, 3.0, 1.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let value = f32::from_ne_bytes(output.outputs[0][..4].try_into().unwrap());
        assert_eq!(value, 7.0);
    }

    #[test]
    fn dispatches_compiled_uop_reduce_min() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_reduce_min"),
                inputs: vec![bytes(&[-2.0, 7.0, 3.0, 1.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let value = f32::from_ne_bytes(output.outputs[0][..4].try_into().unwrap());
        assert_eq!(value, -2.0);
    }

    #[test]
    fn dispatches_compiled_uop_reduce_max_axis() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_reduce_max_axis:2:3:1"),
                inputs: vec![bytes(&[1.0, 7.0, 3.0, 9.0, 2.0, 4.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![7.0, 9.0]);
    }

    #[test]
    fn dispatches_compiled_uop_reduce_min_axis() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_reduce_min_axis:2:3:1"),
                inputs: vec![bytes(&[1.0, 7.0, 3.0, 9.0, 2.0, 4.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn dispatches_compiled_uop_unary_elementwise() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_elementwise:gelu"),
                inputs: vec![bytes(&[-1.0, 0.0, 1.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!(values[0] < 0.0 && values[1].abs() < 1e-6 && values[2] > 0.0);
    }

    #[test]
    fn dispatches_fused_unary_elementwise_program() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_elementwise:neg,relu"),
                inputs: vec![bytes(&[-2.0, 1.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![2.0, 0.0]);
    }

    #[test]
    fn dispatches_binary_elementwise_with_rhs_order() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_elementwise_binary:sub"),
                inputs: vec![bytes(&[1.0, 5.0]), bytes(&[3.0, 2.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![-2.0, 3.0]);
    }

    #[test]
    fn dispatches_mixed_fused_elementwise_program() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_elementwise_program:add,relu"),
                inputs: vec![bytes(&[-3.0, 2.0]), bytes(&[1.0, -4.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0.0, 0.0]);
    }

    #[test]
    fn dispatches_scalar_elementwise_with_operand_order() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_elementwise_scalar:sub|2|1"),
                inputs: vec![bytes(&[1.0, 3.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, -1.0]);
    }

    #[test]
    fn dispatches_rope() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_rope:1:4"),
                inputs: vec![
                    bytes(&[1.0, 2.0, 3.0, 4.0]),
                    bytes(&[0.0, 1.0]),
                    bytes(&[1.0, 0.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![-2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn dispatches_gather() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_gather:2:3:2"),
                inputs: vec![bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), bytes(&[2.0, 0.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn dispatches_ssm() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_ssm:2:2"),
                inputs: vec![
                    bytes(&[1.0, 2.0, 3.0, 4.0]),
                    bytes(&[0.5, 0.25]),
                    bytes(&[1.0, 1.0]),
                    bytes(&[2.0, 4.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![2.0, 8.0, 7.0, 18.0]);
    }

    #[test]
    fn dispatches_layer_norm() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_layer_norm:1:2:0.00001"),
                inputs: vec![bytes(&[1.0, 3.0]), bytes(&[2.0, 0.5]), bytes(&[1.0, -1.0])],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!((values[0] + 1.0).abs() < 1e-3);
        assert!((values[1] + 0.5).abs() < 1e-3);
    }

    #[test]
    fn dispatches_conv2d() {
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact: uop_artifact("uop_conv2d:1:1:3:3:1:2:2:1:0"),
                inputs: vec![
                    bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                    bytes(&[1.0, 0.0, 0.0, 1.0]),
                    bytes(&[0.0]),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let values: Vec<f32> = output.outputs[0]
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn dispatches_nf4_tile640_gemv_with_group_scales_and_biases() {
        let input_dim = 640usize;
        let output_dim = 1usize;
        let codes = vec![0x88u8; input_dim / 2];
        let scales = vec![1.0f32; 5];
        let biases = vec![0.0f32; 5];
        let input = vec![1.0f32; input_dim];
        let dims = [input_dim as u32, output_dim as u32];
        let descriptor = KernelDescriptor {
            name: "nf4-test".into(),
            variant: KernelVariant::NF4Tile640,
            backend: BackendKind::CPU,
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: Vec::new(),
            dispatch_geometry: DispatchGeometry {
                threads_per_threadgroup: [1, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                threads_per_grid: [1, 1, 1],
            },
        };
        let artifact = KernelArtifact {
            payloads: vec![KernelPayload {
                binary: Vec::new(),
                descriptor: descriptor.clone(),
            }],
            manifest: KernelManifest {
                kernels: vec![descriptor],
                fusion_plan: None,
                manifest_digest: String::new(),
            },
            artifact_digest: String::new(),
        };
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs: vec![
                    codes,
                    input.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    scales.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    biases.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let value = f32::from_ne_bytes(output.outputs[0][..4].try_into().unwrap());
        assert!((value - 50.931392).abs() < 0.1, "got {value}");
    }

    #[test]
    fn dispatches_ternary_tile640_gemv_with_packed_abi() {
        let trits = 1u32 + 2 * 3 + 0 * 9 + 1 * 27;
        let descriptor = KernelDescriptor {
            name: "ternary-test".into(),
            variant: KernelVariant::TernaryTile640(crate::TernaryKernelAbi {
                page_size: 640,
                lane_size: 20,
                words_per_page: 32,
                scale_bits: 16,
                outlier_capacity: 0,
                pack_format: 0,
            }),
            backend: BackendKind::CPU,
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: Vec::new(),
            dispatch_geometry: DispatchGeometry {
                threads_per_threadgroup: [1, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                threads_per_grid: [1, 1, 1],
            },
        };
        let artifact = KernelArtifact {
            payloads: vec![KernelPayload {
                binary: Vec::new(),
                descriptor: descriptor.clone(),
            }],
            manifest: KernelManifest {
                kernels: vec![descriptor],
                fusion_plan: None,
                manifest_digest: String::new(),
            },
            artifact_digest: String::new(),
        };
        let dims = [4u32, 1u32];
        let output = CpuBackend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs: vec![
                    (0..32u32)
                        .map(|index| if index == 0 { trits } else { 0 })
                        .flat_map(|word| word.to_le_bytes())
                        .collect(),
                    [half::f16::from_f32(1.0); 4]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                    half::bf16::from_f32(1.0).to_bits().to_le_bytes().to_vec(),
                    [127u8; 32].to_vec(),
                    dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                ],
                bindings: Vec::new(),
            })
            .unwrap();
        let value = f32::from_ne_bytes(output.outputs[0][..4].try_into().unwrap());
        assert!((value - 1.0).abs() < 1e-6, "got {value}");
    }
}
