//! Metal dispatch implementation — unified GPU execution for FP16 and ternary kernels.
//!
//! This module will consolidate Metal dispatch functionality from:
//! - prism-ecs-server::engine::metal (FP16 dispatch)
//! - prism-ecs-quantization::bonsai_metal_dispatch (ternary Tile640 dispatch)
//!
//! The actual implementations will be moved here as part of Phase 9 unification.

#![cfg(all(target_os = "macos", feature = "metal-dispatch"))]

// FP16 kernel source (will be moved from prism-ecs-server)
pub const FP16_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fp16_gemv(
    device const half* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= m) return;
    float acc = 0.0f;
    for (uint col = 0; col < n; ++col) {
        acc += float(weights[row * n + col]) * input[col];
    }
    output[row] = acc;
}
"#;

// Ternary kernel source (will be moved from prism-ecs-server)
pub const TERNARY_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void ternary_tile640_gemv(
    device const uint* packed [[buffer(0)]], device const half* input [[buffer(1)]],
    device const ushort* page_scales [[buffer(2)]], device const char* lane_scales [[buffer(3)]],
    device half* output [[buffer(4)]], constant uint& n [[buffer(5)]], constant uint& m [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (row >= m) return;
    uint pages = (n + 639u) / 640u; uint words = pages * 32u; float acc = 0.0f;
    for (uint wi = tid; wi < words; wi += 64u) {
        uint page = wi / 32u; uint col0 = page * 640u + (wi % 32u) * 20u;
        uint word = packed[row * words + wi];
        float scale = as_type<float>(uint(page_scales[row * pages + page]) << 16) *
                      (float(lane_scales[row * words + wi]) / 32.0f);
        for (uint vi = 0; vi < 20u; ++vi) { uint col = col0 + vi; if (col >= n) break;
            uint trit = word % 3u; word /= 3u;
            if (trit == 1u) acc += float(input[col]) * scale;
            else if (trit == 2u) acc -= float(input[col]) * scale;
        }
    }
    acc = simd_sum(acc);
    if (tid == 0) output[row] = half(acc);
}
"#;

use std::time::Instant;

use metal::{Device, MTLResourceOptions, MTLSize};

use crate::{KernelDispatchRequest, KernelError, KernelOutput, KernelVariant};

/// Dispatch a compiled FP16 GEMV artifact through a reusable Metal command
/// queue. Inputs are the raw FP16 weight matrix followed by an FP32 vector;
/// the output is returned as an FP32 buffer.
pub fn dispatch_artifact(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("Metal artifact has no payload".into()))?;
    match &payload.descriptor.variant {
        KernelVariant::FP16GEMV => dispatch_gemv(request),
        KernelVariant::FP16Matmul => dispatch_matmul(request),
        KernelVariant::TernaryTile640(_) => dispatch_ternary(request),
        KernelVariant::INT8Tile640 => dispatch_int8_gemv(request),
        KernelVariant::NF4Tile640 => dispatch_nf4_gemv(request),
        KernelVariant::Custom(name)
            if name == "uop_elementwise"
                || name.starts_with("uop_elementwise:")
                || name.starts_with("uop_elementwise_binary:")
                || name.starts_with("uop_elementwise_program:")
                || name.starts_with("uop_elementwise_scalar:") =>
        {
            dispatch_uop_elementwise(request)
        }
        KernelVariant::Custom(name) if name == "uop_reduce_sum" => dispatch_uop_reduce_sum(request),
        KernelVariant::Custom(name) if name.starts_with("uop_reduce_sum_axis:") => {
            dispatch_uop_reduce_sum_axis(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_softmax_axis:") => {
            dispatch_uop_softmax_axis(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_attention:") => {
            dispatch_uop_attention(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_attention_batched:") => {
            dispatch_uop_attention_batched(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_rms_norm:") => {
            dispatch_uop_rms_norm(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_layer_norm:") => {
            dispatch_uop_layer_norm(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_rope:") => {
            dispatch_uop_rope(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_gather:") => {
            dispatch_uop_gather(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_scatter:") => {
            dispatch_uop_scatter(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_ssm:") => {
            dispatch_uop_ssm(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_conv2d:") => {
            dispatch_uop_conv2d(request, name)
        }
        KernelVariant::Custom(name) if name.starts_with("uop_matmul:") => {
            dispatch_uop_matmul(request, name)
        }
        variant => Err(KernelError::DispatchFailed(format!(
            "Metal dispatch does not yet support {variant:?}"
        ))),
    }
}

fn dispatch_uop_rms_norm(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp RMSNorm has no payload".into()))?;
    if request.inputs.len() < 2 || request.bindings.len() < 3 {
        return Err(KernelError::BindingMismatch(
            "UOp RMSNorm requires input, weight, and output bindings".into(),
        ));
    }
    let mut parts = variant.split(':').skip(1);
    let (Some(Ok(rows)), Some(Ok(features)), Some(Ok(epsilon))) = (
        parts.next().map(|value| value.parse::<usize>()),
        parts.next().map(|value| value.parse::<usize>()),
        parts.next().map(|value| value.parse::<f32>()),
    ) else {
        return Err(KernelError::DispatchFailed(
            "UOp RMSNorm variant has invalid dimensions".into(),
        ));
    };
    let x = &request.inputs[0];
    let weight = &request.inputs[1];
    if x.len() != rows * features * 4 || weight.len() != features * 4 {
        return Err(KernelError::BindingMismatch(
            "UOp RMSNorm buffers do not match variant dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_rms_norm", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let xb = make_buffer(x);
    let wb = make_buffer(weight);
    let out = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&xb), 0);
    encoder.set_buffer(1, Some(&wb), 0);
    encoder.set_buffer(2, Some(&out), 0);
    let width = pipeline.thread_execution_width().max(1);
    let grid = MTLSize {
        width: (rows * features) as u64,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid, group);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(out.contents() as *const u8, rows * features * 4) };
    let _ = epsilon;
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_conv2d(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp Conv2D has no payload".into()))?;
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "UOp Conv2D requires input, weight, bias, and output bindings".into(),
        ));
    }
    let mut dims = variant.split(':').skip(1).map(|v| v.parse::<usize>());
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
    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
    let elements = batch * out_channels * out_h * out_w;
    if request.inputs[0].len() != batch * in_channels * height * width * 4
        || request.inputs[1].len() != out_channels * in_channels * kernel_h * kernel_w * 4
        || request.inputs[2].len() != out_channels * 4
    {
        return Err(KernelError::BindingMismatch(
            "UOp Conv2D buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_conv2d", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let xb = make_buffer(&request.inputs[0]);
    let wb = make_buffer(&request.inputs[1]);
    let bb = make_buffer(&request.inputs[2]);
    let out = device.new_buffer((elements * 4) as u64, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&xb), 0);
    encoder.set_buffer(1, Some(&wb), 0);
    encoder.set_buffer(2, Some(&bb), 0);
    encoder.set_buffer(3, Some(&out), 0);
    let width_threads = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: elements as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: width_threads,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result = unsafe { std::slice::from_raw_parts(out.contents() as *const u8, elements * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_layer_norm(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp LayerNorm has no payload".into()))?;
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "UOp LayerNorm requires input, weight, bias, and output bindings".into(),
        ));
    }
    let mut parts = variant.split(':').skip(1);
    let (Some(Ok(rows)), Some(Ok(features)), Some(Ok(_epsilon))) = (
        parts.next().map(|v| v.parse::<usize>()),
        parts.next().map(|v| v.parse::<usize>()),
        parts.next().map(|v| v.parse::<f32>()),
    ) else {
        return Err(KernelError::DispatchFailed(
            "UOp LayerNorm variant has invalid dimensions".into(),
        ));
    };
    if request.inputs[0].len() != rows * features * 4
        || request.inputs[1].len() != features * 4
        || request.inputs[2].len() != features * 4
    {
        return Err(KernelError::BindingMismatch(
            "UOp LayerNorm buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_layer_norm", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let xb = make_buffer(&request.inputs[0]);
    let wb = make_buffer(&request.inputs[1]);
    let bb = make_buffer(&request.inputs[2]);
    let out = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&xb), 0);
    encoder.set_buffer(1, Some(&wb), 0);
    encoder.set_buffer(2, Some(&bb), 0);
    encoder.set_buffer(3, Some(&out), 0);
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * features) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(out.contents() as *const u8, rows * features * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_rope(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp RoPE has no payload".into()))?;
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "UOp RoPE requires input, cosine, sine, and output bindings".into(),
        ));
    }
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
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
    let half = features / 2;
    if request.inputs[0].len() != rows * features * 4
        || request.inputs[1].len() != rows * half * 4
        || request.inputs[2].len() != rows * half * 4
    {
        return Err(KernelError::BindingMismatch(
            "UOp RoPE buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_rope", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let xb = make_buffer(&request.inputs[0]);
    let cb = make_buffer(&request.inputs[1]);
    let sb = make_buffer(&request.inputs[2]);
    let out = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&xb), 0);
    encoder.set_buffer(1, Some(&cb), 0);
    encoder.set_buffer(2, Some(&sb), 0);
    encoder.set_buffer(3, Some(&out), 0);
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * half) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(out.contents() as *const u8, rows * features * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_gather(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp Gather has no payload".into()))?;
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(rows)), Some(Ok(vocab)), Some(Ok(features))) =
        (dims.next(), dims.next(), dims.next())
    else {
        return Err(KernelError::DispatchFailed(
            "UOp Gather variant has invalid dimensions".into(),
        ));
    };
    if request.inputs.len() < 2 || request.bindings.len() < 3 {
        return Err(KernelError::BindingMismatch(
            "UOp Gather requires weight, indices, and output bindings".into(),
        ));
    }
    if request.inputs[0].len() != vocab * features * 4 || request.inputs[1].len() != rows * 4 {
        return Err(KernelError::BindingMismatch(
            "UOp Gather buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_gather", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let weight = make_buffer(&request.inputs[0]);
    let indices = make_buffer(&request.inputs[1]);
    let output = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weight), 0);
    encoder.set_buffer(1, Some(&indices), 0);
    encoder.set_buffer(2, Some(&output), 0);
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * features) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(output.contents() as *const u8, rows * features * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_scatter(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp Scatter has no payload".into()))?;
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(rows)), Some(Ok(updates)), Some(Ok(features))) =
        (dims.next(), dims.next(), dims.next())
    else {
        return Err(KernelError::DispatchFailed(
            "UOp Scatter variant has invalid dimensions".into(),
        ));
    };
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "UOp Scatter requires base, indices, updates, and output bindings".into(),
        ));
    }
    if request.inputs[0].len() != rows * features * 4
        || request.inputs[1].len() != updates * 4
        || request.inputs[2].len() != updates * features * 4
    {
        return Err(KernelError::BindingMismatch(
            "UOp Scatter buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_scatter", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let base = make_buffer(&request.inputs[0]);
    let indices = make_buffer(&request.inputs[1]);
    let updates_buffer = make_buffer(&request.inputs[2]);
    let output = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&base), 0);
    encoder.set_buffer(1, Some(&indices), 0);
    encoder.set_buffer(2, Some(&updates_buffer), 0);
    encoder.set_buffer(3, Some(&output), 0);
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * features) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(output.contents() as *const u8, rows * features * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_ssm(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp SSM has no payload".into()))?;
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(rows)), Some(Ok(features))) = (dims.next(), dims.next()) else {
        return Err(KernelError::DispatchFailed(
            "UOp SSM variant has invalid dimensions".into(),
        ));
    };
    if request.inputs.len() < 4 || request.bindings.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "UOp SSM requires four inputs and an output binding".into(),
        ));
    }
    if request.inputs[0].len() != rows * features * 4
        || request.inputs[1].len() != features * 4
        || request.inputs[2].len() != features * 4
        || request.inputs[3].len() != features * 4
    {
        return Err(KernelError::BindingMismatch(
            "UOp SSM buffers do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_ssm", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let input = make_buffer(&request.inputs[0]);
    let decay = make_buffer(&request.inputs[1]);
    let input_gain = make_buffer(&request.inputs[2]);
    let output_gain = make_buffer(&request.inputs[3]);
    let output = device.new_buffer(
        (rows * features * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&decay), 0);
    encoder.set_buffer(2, Some(&input_gain), 0);
    encoder.set_buffer(3, Some(&output_gain), 0);
    encoder.set_buffer(4, Some(&output), 0);
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: (rows * features) as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    let result =
        unsafe { std::slice::from_raw_parts(output.contents() as *const u8, rows * features * 4) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: 0,
    })
}

fn dispatch_uop_elementwise(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp artifact has no payload".into()))?;
    if request.inputs.len() < 1
        || request.bindings.len() < 2
        || request.inputs[0].is_empty()
        || request.inputs[0].len() % 4 != 0
    {
        return Err(KernelError::BindingMismatch(
            "UOp dispatch requires an FP32 input and output binding".into(),
        ));
    }
    let has_rhs = payload.descriptor.binding_signature.len() == 3;
    if has_rhs && (request.inputs.len() < 2 || request.inputs[1].len() != request.inputs[0].len()) {
        return Err(KernelError::BindingMismatch(
            "UOp RHS buffer must match the primary input length".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_kernel", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let input = make_buffer(&request.inputs[0]);
    let rhs = has_rhs.then(|| make_buffer(&request.inputs[1]));
    let output = device.new_buffer(
        request.inputs[0].len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    if let Some(rhs) = &rhs {
        encoder.set_buffer(1, Some(rhs), 0);
        encoder.set_buffer(2, Some(&output), 0);
    } else {
        encoder.set_buffer(1, Some(&output), 0);
    }
    let elements = (request.inputs[0].len() / 4) as u64;
    let threads = pipeline.thread_execution_width().max(1) as u64;
    encoder.dispatch_threads(MTLSize::new(elements, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal UOp command failed".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(output.contents() as *const u8, request.inputs[0].len())
    };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_reduce_sum(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp reduction has no payload".into()))?;
    let input = request.inputs.first().ok_or_else(|| {
        KernelError::BindingMismatch("UOp reduction requires an FP32 input".into())
    })?;
    if input.is_empty() || input.len() % 4 != 0 || request.bindings.len() < 2 {
        return Err(KernelError::BindingMismatch(
            "UOp reduction requires an FP32 input and output binding".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_reduce_sum", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        input.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new(1, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal UOp reduction command failed".into(),
        ));
    }
    let bytes = unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u8, 4) };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_reduce_sum_axis(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload =
        request.artifact.payloads.first().ok_or_else(|| {
            KernelError::DispatchFailed("UOp axis reduction has no payload".into())
        })?;
    let input = request.inputs.first().ok_or_else(|| {
        KernelError::BindingMismatch("UOp axis reduction requires an FP32 input".into())
    })?;
    if input.is_empty() || input.len() % 4 != 0 || request.bindings.len() < 2 {
        return Err(KernelError::BindingMismatch(
            "UOp axis reduction requires FP32 input/output bindings".into(),
        ));
    }
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
        (dims.next(), dims.next(), dims.next())
    else {
        return Err(KernelError::DispatchFailed(
            "invalid axis reduction dimensions".into(),
        ));
    };
    if input.len() != outer * reduce * inner * 4 {
        return Err(KernelError::BindingMismatch(
            "axis reduction input does not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_reduce_sum_axis", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        input.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(
        (outer * inner * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new((outer * inner) as u64, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal axis reduction failed".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const u8, outer * inner * 4)
    };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_softmax_axis(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp softmax has no payload".into()))?;
    let input = request
        .inputs
        .first()
        .ok_or_else(|| KernelError::BindingMismatch("UOp softmax requires an FP32 input".into()))?;
    if input.is_empty() || input.len() % 4 != 0 || request.bindings.len() < 2 {
        return Err(KernelError::BindingMismatch(
            "UOp softmax requires FP32 input/output bindings".into(),
        ));
    }
    let mut dims = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(outer)), Some(Ok(reduce)), Some(Ok(inner))) =
        (dims.next(), dims.next(), dims.next())
    else {
        return Err(KernelError::DispatchFailed(
            "invalid softmax dimensions".into(),
        ));
    };
    let elements = outer * reduce * inner;
    if input.len() != elements * 4 {
        return Err(KernelError::BindingMismatch(
            "softmax input does not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_softmax_axis", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        input.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer =
        device.new_buffer(elements as u64 * 4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal softmax command failed".into(),
        ));
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u8, elements * 4) };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_attention_batched(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().ok_or_else(|| {
        KernelError::DispatchFailed("UOp batched attention has no payload".into())
    })?;
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "batched attention requires Q, K, V, and output bindings".into(),
        ));
    }
    let mut dims = variant.split(':').skip(1);
    let batch = dims
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention batch".into()))?;
    let seq = dims
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention sequence".into()))?;
    let head = dims
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention head".into()))?;
    let _scale = dims
        .next()
        .and_then(|value| value.parse::<f32>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention scale".into()))?;
    let elements = batch * seq * head;
    let bytes = elements * 4;
    if request.inputs[..3].iter().any(|input| input.len() != bytes) {
        return Err(KernelError::BindingMismatch(
            "batched attention inputs do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_attention_batched", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |input: &[u8]| {
        device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let q = make_buffer(&request.inputs[0]);
    let k = make_buffer(&request.inputs[1]);
    let v = make_buffer(&request.inputs[2]);
    let output = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&q), 0);
    encoder.set_buffer(1, Some(&k), 0);
    encoder.set_buffer(2, Some(&v), 0);
    encoder.set_buffer(3, Some(&output), 0);
    encoder.dispatch_threads(
        MTLSize::new(elements as u64, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal batched attention command failed".into(),
        ));
    }
    let result = unsafe { std::slice::from_raw_parts(output.contents() as *const u8, bytes) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_attention(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp attention has no payload".into()))?;
    if request.inputs.len() < 3 || request.bindings.len() < 4 {
        return Err(KernelError::BindingMismatch(
            "UOp attention requires Q, K, V, and output bindings".into(),
        ));
    }
    let mut values = variant.split(':').skip(1);
    let seq = values
        .next()
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention sequence".into()))?;
    let head = values
        .next()
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention head".into()))?;
    let _scale = values
        .next()
        .and_then(|v| v.parse::<f32>().ok())
        .ok_or_else(|| KernelError::DispatchFailed("invalid attention scale".into()))?;
    let bytes = seq * head * 4;
    if request.inputs[..3].iter().any(|input| input.len() != bytes) {
        return Err(KernelError::BindingMismatch(
            "attention inputs do not match dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_attention", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |input: &[u8]| {
        device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let q = make_buffer(&request.inputs[0]);
    let k = make_buffer(&request.inputs[1]);
    let v = make_buffer(&request.inputs[2]);
    let output = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&q), 0);
    encoder.set_buffer(1, Some(&k), 0);
    encoder.set_buffer(2, Some(&v), 0);
    encoder.set_buffer(3, Some(&output), 0);
    encoder.dispatch_threads(
        MTLSize::new((seq * head) as u64, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal attention command failed".into(),
        ));
    }
    let result = unsafe { std::slice::from_raw_parts(output.contents() as *const u8, bytes) };
    Ok(KernelOutput {
        outputs: vec![result.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_uop_matmul(
    request: &KernelDispatchRequest,
    variant: &str,
) -> Result<KernelOutput, KernelError> {
    let payload = request
        .artifact
        .payloads
        .first()
        .ok_or_else(|| KernelError::DispatchFailed("UOp matmul has no payload".into()))?;
    if request.inputs.len() < 2 || request.bindings.len() < 3 {
        return Err(KernelError::BindingMismatch(
            "UOp matmul requires A, B, and output bindings".into(),
        ));
    }
    let a = &request.inputs[0];
    let b = &request.inputs[1];
    if a.len() % 4 != 0 || b.len() % 4 != 0 || a.is_empty() || b.is_empty() {
        return Err(KernelError::BindingMismatch(
            "UOp matmul inputs must be non-empty FP32 buffers".into(),
        ));
    }
    let mut dimensions = variant
        .split(':')
        .skip(1)
        .map(|value| value.parse::<usize>());
    let (Some(Ok(m)), Some(Ok(k)), Some(Ok(n))) =
        (dimensions.next(), dimensions.next(), dimensions.next())
    else {
        return Err(KernelError::DispatchFailed(
            "UOp matmul variant has invalid dimensions".into(),
        ));
    };
    if a.len() != m * k * 4 || b.len() != k * n * 4 {
        return Err(KernelError::BindingMismatch(
            "UOp matmul buffers do not match variant dimensions".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("prism_matmul", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let a_buffer = make_buffer(a);
    let b_buffer = make_buffer(b);
    let output_buffer =
        device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&a_buffer), 0);
    encoder.set_buffer(1, Some(&b_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.dispatch_threads(
        MTLSize::new((m * n) as u64, 1, 1),
        MTLSize::new(pipeline.thread_execution_width().max(1) as u64, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal UOp matmul command failed".into(),
        ));
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u8, m * n * 4) };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_nf4_gemv(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().unwrap();
    if request.inputs.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "NF4 GEMV requires codes, input, scales, biases, and dimensions".into(),
        ));
    }
    let codes = &request.inputs[0];
    let input = &request.inputs[1];
    let scales = &request.inputs[2];
    let biases = &request.inputs[3];
    let dims = &request.inputs[4];
    if dims.len() != 8 || input.len() % 4 != 0 || scales.len() % 4 != 0 || biases.len() % 4 != 0 {
        return Err(KernelError::DispatchFailed(
            "invalid NF4 GEMV buffers".into(),
        ));
    }
    let input_dim = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let output_dim = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    let tiles = input_dim.div_ceil(640);
    if input_dim == 0
        || output_dim == 0
        || input.len() != input_dim * 4
        || codes.len() != output_dim * tiles * 320
        || scales.len() != output_dim * tiles * 5 * 4
        || biases.len() != output_dim * tiles * 5 * 4
    {
        return Err(KernelError::DispatchFailed(
            "NF4 GEMV dimensions do not match buffers".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("nf4_tile640_gemv", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let code_buffer = make_buffer(codes);
    let input_buffer = make_buffer(input);
    let scales_buffer = make_buffer(scales);
    let biases_buffer = make_buffer(biases);
    let dims_buffer = make_buffer(dims);
    let output_buffer = device.new_buffer(
        (output_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&code_buffer), 0);
    encoder.set_buffer(1, Some(&input_buffer), 0);
    encoder.set_buffer(2, Some(&scales_buffer), 0);
    encoder.set_buffer(3, Some(&biases_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&dims_buffer), 0);
    let threads = payload.descriptor.dispatch_geometry.threads_per_threadgroup[0].max(1) as u64;
    encoder.dispatch_threads(
        MTLSize::new(output_dim as u64, 1, 1),
        MTLSize::new(threads, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal NF4 command failed".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const u8, output_dim * 4)
    };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_int8_gemv(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().unwrap();
    if request.inputs.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "INT8 GEMV requires weights, input, weight scales, input scale, and dimensions".into(),
        ));
    }
    let weights = &request.inputs[0];
    let input = &request.inputs[1];
    let scales = &request.inputs[2];
    let input_scale = &request.inputs[3];
    let dims = &request.inputs[4];
    if input_scale.len() != 4 || dims.len() != 8 || scales.len() % 4 != 0 {
        return Err(KernelError::DispatchFailed(
            "invalid INT8 GEMV buffers".into(),
        ));
    }
    let input_dim = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let output_dim = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    if input_dim == 0
        || output_dim == 0
        || input.len() != input_dim
        || weights.len() != input_dim * output_dim
        || scales.len() != output_dim * 4
    {
        return Err(KernelError::DispatchFailed(
            "INT8 GEMV dimensions do not match buffers".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("int8_gemv", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let weight_buffer = make_buffer(weights);
    let input_buffer = make_buffer(input);
    let scale_buffer = make_buffer(scales);
    let input_scale_buffer = make_buffer(input_scale);
    let dims_buffer = make_buffer(dims);
    let output_buffer = device.new_buffer(
        (output_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weight_buffer), 0);
    encoder.set_buffer(1, Some(&input_buffer), 0);
    encoder.set_buffer(2, Some(&scale_buffer), 0);
    encoder.set_buffer(3, Some(&input_scale_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_buffer(5, Some(&dims_buffer), 0);
    let threads = payload.descriptor.dispatch_geometry.threads_per_threadgroup[0].max(1) as u64;
    encoder.dispatch_threads(
        MTLSize::new(output_dim as u64, 1, 1),
        MTLSize::new(threads, 1, 1),
    );
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal INT8 command failed".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const u8, output_dim * 4)
    };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_gemv(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().unwrap();
    if request.inputs.len() < 2 {
        return Err(KernelError::BindingMismatch(
            "FP16 GEMV requires weights and input buffers".into(),
        ));
    }
    let weights = &request.inputs[0];
    let input = &request.inputs[1];
    if input.len() % 4 != 0 || weights.len() % 2 != 0 {
        return Err(KernelError::DispatchFailed(
            "FP16 GEMV buffer lengths are not element aligned".into(),
        ));
    }
    let n = (input.len() / 4) as u32;
    let row_bytes = n as usize * 2;
    if n == 0 || weights.len() % row_bytes != 0 {
        return Err(KernelError::DispatchFailed(
            "FP16 GEMV weight matrix does not match input width".into(),
        ));
    }
    let m = (weights.len() / row_bytes) as u32;
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("fp16_gemv", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let weight_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const std::ffi::c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        input.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer((m as u64) * 4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weight_buffer), 0);
    encoder.set_buffer(1, Some(&input_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_bytes(3, 4, &m as *const _ as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &n as *const _ as *const std::ffi::c_void);
    let threads = payload.descriptor.dispatch_geometry.threads_per_threadgroup[0].max(1) as u64;
    encoder.dispatch_threads(MTLSize::new(m as u64, 1, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal command buffer completed with an error".into(),
        ));
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const u8, m as usize * 4)
    };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_matmul(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().unwrap();
    if request.inputs.len() < 3 {
        return Err(KernelError::BindingMismatch(
            "FP16 matmul requires A, B, and uint3 dimensions buffers".into(),
        ));
    }
    let a = &request.inputs[0];
    let b = &request.inputs[1];
    let dims = &request.inputs[2];
    if dims.len() != 12 || a.len() % 2 != 0 || b.len() % 2 != 0 {
        return Err(KernelError::DispatchFailed(
            "invalid FP16 matmul buffers".into(),
        ));
    }
    let m = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let n = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    let k = u32::from_ne_bytes(dims[8..12].try_into().unwrap()) as usize;
    if m == 0 || n == 0 || k == 0 || a.len() != m * k * 2 || b.len() != k * n * 2 {
        return Err(KernelError::DispatchFailed(
            "FP16 matmul dimensions do not match buffer lengths".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("matmul_fp16", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let a_buffer = device.new_buffer_with_data(
        a.as_ptr() as *const std::ffi::c_void,
        a.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buffer = device.new_buffer_with_data(
        b.as_ptr() as *const std::ffi::c_void,
        b.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer =
        device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);
    let dims_buffer = device.new_buffer_with_data(
        dims.as_ptr() as *const std::ffi::c_void,
        12,
        MTLResourceOptions::StorageModeShared,
    );
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&a_buffer), 0);
    encoder.set_buffer(1, Some(&b_buffer), 0);
    encoder.set_buffer(2, Some(&output_buffer), 0);
    encoder.set_buffer(3, Some(&dims_buffer), 0);
    let tx = payload.descriptor.dispatch_geometry.threads_per_threadgroup[0].max(1) as u64;
    let ty = payload.descriptor.dispatch_geometry.threads_per_threadgroup[1].max(1) as u64;
    encoder.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(tx, ty, 1));
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal matmul command failed".into(),
        ));
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u8, m * n * 4) };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}

fn dispatch_ternary(request: &KernelDispatchRequest) -> Result<KernelOutput, KernelError> {
    let payload = request.artifact.payloads.first().unwrap();
    if request.inputs.len() < 5 {
        return Err(KernelError::BindingMismatch(
            "ternary GEMV requires packed weights, input, page scales, lane scales, and dimensions"
                .into(),
        ));
    }
    let packed = &request.inputs[0];
    let input = &request.inputs[1];
    let page_scales = &request.inputs[2];
    let lane_scales = &request.inputs[3];
    let dims = &request.inputs[4];
    if input.len() % 2 != 0 || dims.len() != 8 {
        return Err(KernelError::DispatchFailed(
            "invalid ternary GEMV buffers".into(),
        ));
    }
    let in_dim = u32::from_ne_bytes(dims[0..4].try_into().unwrap()) as usize;
    let out_dim = u32::from_ne_bytes(dims[4..8].try_into().unwrap()) as usize;
    let pages = in_dim.div_ceil(640);
    if in_dim == 0
        || out_dim == 0
        || input.len() != in_dim * 2
        || packed.len() != out_dim * pages * 4
        || page_scales.len() != out_dim * pages * 2
        || lane_scales.len() < out_dim * pages
    {
        return Err(KernelError::DispatchFailed(
            "ternary GEMV dimensions do not match packed and scale buffers".into(),
        ));
    }
    let device = Device::system_default()
        .ok_or_else(|| KernelError::UnsupportedBackend("no Metal device found".into()))?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_data(&payload.binary)
        .map_err(KernelError::DispatchFailed)?;
    let function = library
        .get_function("ternary_tile640_gemv", None)
        .map_err(KernelError::DispatchFailed)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| KernelError::DispatchFailed(e.to_string()))?;
    let make_buffer = |bytes: &[u8]| {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let packed_buffer = make_buffer(packed);
    let input_buffer = make_buffer(input);
    let page_buffer = make_buffer(page_scales);
    let lane_buffer = make_buffer(lane_scales);
    let output_buffer =
        device.new_buffer((out_dim * 2) as u64, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&packed_buffer), 0);
    encoder.set_buffer(1, Some(&input_buffer), 0);
    encoder.set_buffer(2, Some(&page_buffer), 0);
    encoder.set_buffer(3, Some(&lane_buffer), 0);
    encoder.set_buffer(4, Some(&output_buffer), 0);
    encoder.set_bytes(
        5,
        4,
        &(in_dim as u32) as *const _ as *const std::ffi::c_void,
    );
    encoder.set_bytes(
        6,
        4,
        &(out_dim as u32) as *const _ as *const std::ffi::c_void,
    );
    encoder.dispatch_thread_groups(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
    encoder.end_encoding();
    let start = Instant::now();
    command.commit();
    command.wait_until_completed();
    if command.status() == metal::MTLCommandBufferStatus::Error {
        return Err(KernelError::DispatchFailed(
            "Metal ternary command failed".into(),
        ));
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(output_buffer.contents() as *const u8, out_dim * 2) };
    Ok(KernelOutput {
        outputs: vec![bytes.to_vec()],
        dispatch_time_ns: start.elapsed().as_nanos() as u64,
    })
}
