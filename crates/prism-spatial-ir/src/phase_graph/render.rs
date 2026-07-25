//! This module owns the canonical authority for the kernel renderer — the
//! `render_broadcast_index`, `render_kernel`, `render_binary`,
//! `render_extremum`, and `hex_digest` helpers that turn a `KernelGroup` into
//! deterministic target source for `LoweringTarget::Cpu`,
//! `LoweringTarget::Portable`, or `LoweringTarget::Metal`.
//! It does not own graph mutation, kernel op enumeration, or replay.

use sha2::{Digest, Sha256};

use crate::phase_graph::kernel_group::KernelGroup;
use crate::phase_graph::kernel_op::{BroadcastBinaryOperation, KernelOp, LoweringTarget};

pub(crate) fn render_broadcast_index(
    input_shape: &[u64],
    output_shape: &[u64],
    prefix: &str,
) -> (String, String) {
    let rank_delta = output_shape.len() - input_shape.len();
    let mut declarations = String::new();
    for axis in 0..output_shape.len() {
        declarations.push_str(&format!(
            "unsigned {prefix}c{axis} = (id / {}u) % {}u; ",
            output_shape[axis + 1..].iter().product::<u64>().max(1),
            output_shape[axis]
        ));
    }
    let mut expression = String::from("0u");
    for axis in 0..input_shape.len() {
        let output_axis = axis + rank_delta;
        if input_shape[axis] != 1 {
            let stride = input_shape[axis + 1..].iter().product::<u64>().max(1);
            expression.push_str(&format!(" + {prefix}c{output_axis} * {stride}u"));
        }
    }
    (declarations, expression)
}

pub(crate) fn render_kernel(group: &KernelGroup, target: LoweringTarget) -> (String, String) {
    if let Some((operation, lhs_shape, rhs_shape, output_shape)) = group.broadcast_binary_shape() {
        let (lhs_declarations, lhs_index) = render_broadcast_index(&lhs_shape, &output_shape, "l");
        let (rhs_declarations, rhs_index) = render_broadcast_index(&rhs_shape, &output_shape, "r");
        let expression = match operation {
            BroadcastBinaryOperation::Add => "lhs_value + rhs_value",
            BroadcastBinaryOperation::Mul => "lhs_value * rhs_value",
            BroadcastBinaryOperation::Sub => "lhs_value - rhs_value",
            BroadcastBinaryOperation::Div => "lhs_value / rhs_value",
            BroadcastBinaryOperation::Maximum => "max(lhs_value, rhs_value)",
            BroadcastBinaryOperation::Minimum => "min(lhs_value, rhs_value)",
        };
        let mut postlude = String::new();
        for op in group.ops_after_broadcast() {
            postlude.push_str(match op {
                KernelOp::Relu { .. } => " value = max(value, 0.0f);",
                KernelOp::Neg { .. } => " value = -value;",
                KernelOp::Exp { .. } => " value = expf(value);",
                KernelOp::Sqrt { .. } => " value = sqrtf(value);",
                KernelOp::Abs { .. } => " value = fabsf(value);",
                KernelOp::Log { .. } => " value = logf(value);",
                KernelOp::Tanh { .. } => " value = tanhf(value);",
                KernelOp::Sin { .. } => " value = sinf(value);",
                KernelOp::Cos { .. } => " value = cosf(value);",
                KernelOp::Gelu { .. } => " value = 0.5f * value * (1.0f + tanhf(0.79788456f * (value + 0.044715f * value * value * value)));",
                KernelOp::Pow { exponent, .. } => return {
                    let _ = exponent;
                    (String::new(), String::new())
                },
                _ => return (String::new(), String::new()),
            });
        }
        let elements = output_shape.iter().product::<u64>();
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_broadcast_binary(device const float* x [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ {lhs_declarations}{rhs_declarations} float lhs_value = x[{lhs_index}]; float rhs_value = rhs[{rhs_index}]; float value = {expression}; {postlude} output[id] = value; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_broadcast_binary(const float* x, const float* rhs, float* output, unsigned id) {{ if (id < {elements}u) {{ {lhs_declarations}{rhs_declarations} float lhs_value = x[{lhs_index}]; float rhs_value = rhs[{rhs_index}]; float value = {expression}; {postlude} output[id] = value; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let [KernelOp::Where {
        condition_shape,
        true_shape,
        false_shape,
        output_shape,
        ..
    }] = group.ops.as_slice()
    {
        let (condition_declarations, condition_index) =
            render_broadcast_index(condition_shape, output_shape, "c");
        let (true_declarations, true_index) = render_broadcast_index(true_shape, output_shape, "t");
        let (false_declarations, false_index) =
            render_broadcast_index(false_shape, output_shape, "f");
        let elements = output_shape.iter().product::<u64>();
        let source: String = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_where(device const float* condition [[buffer(0)]], device const float* when_true [[buffer(1)]], device const float* when_false [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ {condition_declarations}{true_declarations}{false_declarations} output[id] = condition[{condition_index}] != 0.0f ? when_true[{true_index}] : when_false[{false_index}]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_where(const float* condition, const float* when_true, const float* when_false, float* output, unsigned id) {{ if (id < {elements}u) {{ {condition_declarations}{true_declarations}{false_declarations} output[id] = condition[{condition_index}] != 0.0f ? when_true[{true_index}] : when_false[{false_index}]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let [KernelOp::Cast { from: _, to, .. }] = group.ops.as_slice() {
        let conversion = match to.as_str() {
            "f32" | "f16" | "bf16" => "v",
            "i8" => "(float)clamp((int)v, -128, 127)",
            "u8" => "(float)clamp((int)v, 0, 255)",
            "i32" => "(float)(int)clamp(v, -2147483648.0f, 2147483647.0f)",
            "u32" => "(float)(uint)clamp(v, 0.0f, 4294967295.0f)",
            // WAIVER: the `to` value is validated by `TinyGraph::validate`
            // for every `UOpKind::Cast` before the renderer runs, so this
            // match is exhaustive over the inputs the type system allows.
            _ => unreachable!("validated cast target"),
        };
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_cast(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ float v = x[id]; output[id] = {conversion}; }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_cast(const float* x, float* output, unsigned id) {{ float v = x[id]; output[id] = {conversion}; }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((permutation, input_shape, output_shape)) = group.transpose_shape() {
        let input_strides: Vec<usize> = (0..input_shape.len())
            .map(|axis| {
                input_shape[axis + 1..]
                    .iter()
                    .map(|d| *d as usize)
                    .product()
            })
            .collect();
        let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
        let output_elements: usize = output_dims.iter().product();
        let mut coord = String::new();
        let mut source_index = String::from("0");
        for (axis, source_axis) in permutation.iter().enumerate() {
            coord.push_str(&format!(
                "unsigned c{axis} = (id / {}u) % {}u; ",
                output_dims[axis + 1..].iter().product::<usize>().max(1),
                output_dims[axis]
            ));
            source_index.push_str(&format!(" + c{axis} * {}u", input_strides[*source_axis]));
        }
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_transpose(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ {coord} output[id] = x[{source_index}]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_transpose(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ {coord} output[id] = x[{source_index}]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features)) = group.ssm_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_ssm(device const float* input [[buffer(0)]], device const float* decay [[buffer(1)]], device const float* input_gain [[buffer(2)]], device const float* output_gain [[buffer(3)]], device float* output [[buffer(4)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ uint row = id / {features}u; uint feature = id % {features}u; float state = 0.0; for (uint step = 0; step <= row; ++step) state = decay[feature] * state + input_gain[feature] * input[step * {features}u + feature]; output[id] = output_gain[feature] * state; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_ssm(const float* input, const float* decay, const float* input_gain, const float* output_gain, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ unsigned row = id / {features}u; unsigned feature = id % {features}u; float state = 0.0f; for (unsigned step = 0; step <= row; ++step) state = decay[feature] * state + input_gain[feature] * input[step * {features}u + feature]; output[id] = output_gain[feature] * state; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, vocab, features)) = group.gather_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_gather(device const float* weight [[buffer(0)]], device const float* indices [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ uint row = id / {features}u; uint col = id % {features}u; uint index = uint(indices[row]); if (index < {vocab}u) output[id] = weight[index * {features}u + col]; else output[id] = 0.0; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_gather(const float* weight, const float* indices, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ unsigned row = id / {features}u; unsigned col = id % {features}u; unsigned index = (unsigned)indices[row]; output[id] = index < {vocab}u ? weight[index * {features}u + col] : 0.0f; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, updates, features)) = group.scatter_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_scatter(device const float* base [[buffer(0)]], device const float* indices [[buffer(1)]], device const float* updates [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ output[id] = base[id]; for (uint update = 0; update < {updates}u; ++update) {{ uint index = uint(indices[update]); if (index == id / {features}u) output[id] = updates[update * {features}u + id % {features}u]; }} }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_scatter(const float* base, const float* indices, const float* updates, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ output[id] = base[id]; for (unsigned update = 0; update < {updates}u; ++update) {{ unsigned index = (unsigned)indices[update]; if (index == id / {features}u) output[id] = updates[update * {features}u + id % {features}u]; }} }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features)) = group.rope_shape() {
        let elements = rows * (features / 2);
        let half = features / 2;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_rope(device const float* x [[buffer(0)]], device const float* cosv [[buffer(1)]], device const float* sinv [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {half}u; uint pair = id % {half}u; float c = cosv[id]; float s = sinv[id]; uint base = row * {features}u + pair * 2u; float a = x[base]; float b = x[base + 1u]; output[base] = a * c - b * s; output[base + 1u] = a * s + b * c; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_rope(const float* x, const float* cosv, const float* sinv, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {half}u; unsigned pair = id % {half}u; float c = cosv[id]; float s = sinv[id]; unsigned base = row * {features}u + pair * 2u; float a = x[base]; float b = x[base + 1u]; output[base] = a * c - b * s; output[base + 1u] = a * s + b * c; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((batch, seq, head, scale)) = group.batched_attention_shape() {
        let elements = batch * seq * head;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_attention_batched(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint batch = id / ({seq}u * {head}u); uint rem = id % ({seq}u * {head}u); uint query = rem / {head}u; uint dim = rem % {head}u; uint base = batch * {seq}u * {head}u; float scores[{seq}]; float max_v = -INFINITY; for (uint key = 0; key < {seq}u; ++key) {{ float score = 0.0; for (uint d = 0; d < {head}u; ++d) score += q[base + query * {head}u + d] * k[base + key * {head}u + d]; scores[key] = score * {scale}; max_v = max(max_v, scores[key]); }} float denom = 0.0; for (uint key = 0; key < {seq}u; ++key) denom += exp(scores[key] - max_v); float result = 0.0; for (uint key = 0; key < {seq}u; ++key) result += exp(scores[key] - max_v) / denom * v[base + key * {head}u + dim]; output[id] = result; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_attention_batched(const float* q, const float* k, const float* v, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned rem = id % ({seq}u * {head}u); unsigned query = rem / {head}u; unsigned dim = rem % {head}u; unsigned base = (id / ({seq}u * {head}u)) * {seq}u * {head}u; float scores[{seq}]; float max_v = -INFINITY; for (unsigned key = 0; key < {seq}u; ++key) {{ float score = 0.0f; for (unsigned d = 0; d < {head}u; ++d) score += q[base + query * {head}u + d] * k[base + key * {head}u + d]; scores[key] = score * {scale}f; if (scores[key] > max_v) max_v = scores[key]; }} float denom = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) denom += expf(scores[key] - max_v); float result = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) result += expf(scores[key] - max_v) / denom * v[base + key * {head}u + dim]; output[id] = result; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((seq, head, scale)) = group.attention_shape() {
        let elements = seq * head;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_attention(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint query = id / {head}u; uint dim = id % {head}u; float scores[{seq}]; float max_v = -INFINITY; for (uint key = 0; key < {seq}u; ++key) {{ float score = 0.0; for (uint d = 0; d < {head}u; ++d) score += q[query * {head}u + d] * k[key * {head}u + d]; scores[key] = score * {scale}; max_v = max(max_v, scores[key]); }} float denom = 0.0; for (uint key = 0; key < {seq}u; ++key) denom += exp(scores[key] - max_v); float result = 0.0; for (uint key = 0; key < {seq}u; ++key) result += exp(scores[key] - max_v) / denom * v[key * {head}u + dim]; output[id] = result; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_attention(const float* q, const float* k, const float* v, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned query = id / {head}u; unsigned dim = id % {head}u; float scores[{seq}]; float max_v = -INFINITY; for (unsigned key = 0; key < {seq}u; ++key) {{ float score = 0.0f; for (unsigned d = 0; d < {head}u; ++d) score += q[query * {head}u + d] * k[key * {head}u + d]; scores[key] = score * {scale}f; if (scores[key] > max_v) max_v = scores[key]; }} float denom = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) denom += expf(scores[key] - max_v); float result = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) result += expf(scores[key] - max_v) / denom * v[key * {head}u + dim]; output[id] = result; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features, epsilon)) = group.rms_norm_shape() {
        let elements = rows * features;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_rms_norm(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {features}u; float mean_sq = 0.0; for (uint i = 0; i < {features}u; ++i) {{ float v = x[row * {features}u + i]; mean_sq += v * v; }} mean_sq /= {features}u; output[id] = x[id] * rsqrt(mean_sq + {epsilon}); output[id] *= weight[id % {features}u]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_rms_norm(const float* x, const float* weight, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {features}u; float mean_sq = 0.0f; for (unsigned i = 0; i < {features}u; ++i) {{ float v = x[row * {features}u + i]; mean_sq += v * v; }} mean_sq /= {features}u; output[id] = x[id] / sqrtf(mean_sq + {epsilon}f) * weight[id % {features}u]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features, epsilon)) = group.layer_norm_shape() {
        let elements = rows * features;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_layer_norm(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* bias [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {features}u; float mean = 0.0; for (uint i = 0; i < {features}u; ++i) mean += x[row * {features}u + i]; mean /= {features}u; float variance = 0.0; for (uint i = 0; i < {features}u; ++i) {{ float centered = x[row * {features}u + i] - mean; variance += centered * centered; }} variance /= {features}u; output[id] = (x[id] - mean) * rsqrt(variance + {epsilon}) * weight[id % {features}u] + bias[id % {features}u]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_layer_norm(const float* x, const float* weight, const float* bias, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {features}u; float mean = 0.0f; for (unsigned i = 0; i < {features}u; ++i) mean += x[row * {features}u + i]; mean /= {features}u; float variance = 0.0f; for (unsigned i = 0; i < {features}u; ++i) {{ float centered = x[row * {features}u + i] - mean; variance += centered * centered; }} variance /= {features}u; output[id] = (x[id] - mean) / sqrtf(variance + {epsilon}f) * weight[id % {features}u] + bias[id % {features}u]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
    )) = group.conv2d_shape()
    {
        let out_h = (height + 2 * padding - kernel_h) / stride + 1;
        let out_w = (width + 2 * padding - kernel_w) / stride + 1;
        let elements = batch * out_channels * out_h * out_w;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_conv2d(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* bias [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint ow = id % {out_w}u; uint oh = (id / {out_w}u) % {out_h}u; uint oc = (id / ({out_h}u * {out_w}u)) % {out_channels}u; uint b = id / ({out_channels}u * {out_h}u * {out_w}u); float sum = bias[oc]; for (uint ic = 0; ic < {in_channels}u; ++ic) for (uint kh = 0; kh < {kernel_h}u; ++kh) for (uint kw = 0; kw < {kernel_w}u; ++kw) {{ int ih = int(oh * {stride}u + kh) - int({padding}u); int iw = int(ow * {stride}u + kw) - int({padding}u); if (ih >= 0 && iw >= 0 && ih < {height} && iw < {width}) sum += x[((b * {in_channels}u + ic) * {height}u + uint(ih)) * {width}u + uint(iw)] * weight[(((oc * {in_channels}u + ic) * {kernel_h}u + kh) * {kernel_w}u) + kw]; }} output[id] = sum; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_conv2d(const float* x, const float* weight, const float* bias, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned ow = id % {out_w}u; unsigned oh = (id / {out_w}u) % {out_h}u; unsigned oc = (id / ({out_h}u * {out_w}u)) % {out_channels}u; unsigned b = id / ({out_channels}u * {out_h}u * {out_w}u); float sum = bias[oc]; for (unsigned ic = 0; ic < {in_channels}u; ++ic) for (unsigned kh = 0; kh < {kernel_h}u; ++kh) for (unsigned kw = 0; kw < {kernel_w}u; ++kw) {{ int ih = (int)(oh * {stride}u + kh) - (int){padding}; int iw = (int)(ow * {stride}u + kw) - (int){padding}; if (ih >= 0 && iw >= 0 && ih < {height} && iw < {width}) sum += x[((b * {in_channels}u + ic) * {height}u + (unsigned)ih) * {width}u + (unsigned)iw] * weight[(((oc * {in_channels}u + ic) * {kernel_h}u + kh) * {kernel_w}u) + kw]; }} output[id] = sum; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((m, k, n)) = group.matmul_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_matmul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {m}u * {n}u) {{ uint row = id / {n}u; uint col = id % {n}u; float v = 0.0; for (uint inner = 0; inner < {k}u; ++inner) v += a[row * {k}u + inner] * b[inner * {n}u + col]; output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_matmul(const float* a, const float* b, float* output, unsigned id) {{ if (id < {m}u * {n}u) {{ unsigned row = id / {n}u; unsigned col = id % {n}u; float v = 0.0f; for (unsigned inner = 0; inner < {k}u; ++inner) v += a[row * {k}u + inner] * b[inner * {n}u + col]; output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_sum(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = 0.0; for (uint i = 0; i < {elements}; ++i) v += x[i]; output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_sum(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = 0.0f; for (unsigned i = 0; i < {elements}; ++i) v += x[i]; output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.max_reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_max(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = -INFINITY; for (uint i = 0; i < {elements}; ++i) v = max(v, x[i]); output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_max(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = -INFINITY; for (unsigned i = 0; i < {elements}; ++i) v = fmaxf(v, x[i]); output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.min_reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_min(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = INFINITY; for (uint i = 0; i < {elements}; ++i) v = min(v, x[i]); output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_min(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = INFINITY; for (unsigned i = 0; i < {elements}; ++i) v = fminf(v, x[i]); output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.softmax_shape() {
        let elements = outer * reduce * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_softmax_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint outer = id / ({reduce}u * {inner}u); uint rem = id % ({reduce}u * {inner}u); uint step = rem / {inner}u; uint inner = rem % {inner}u; float max_v = -INFINITY; for (uint i = 0; i < {reduce}u; ++i) max_v = max(max_v, x[(outer * {reduce}u + i) * {inner}u + inner]); float denom = 0.0; for (uint i = 0; i < {reduce}u; ++i) denom += exp(x[(outer * {reduce}u + i) * {inner}u + inner] - max_v); output[id] = exp(x[id] - max_v) / denom; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_softmax_axis(const float* x, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned outer = id / ({reduce}u * {inner}u); unsigned rem = id % ({reduce}u * {inner}u); unsigned inner = rem % {inner}u; float max_v = -INFINITY; for (unsigned i = 0; i < {reduce}u; ++i) {{ float v = x[(outer * {reduce}u + i) * {inner}u + inner]; if (v > max_v) max_v = v; }} float denom = 0.0f; for (unsigned i = 0; i < {reduce}u; ++i) denom += expf(x[(outer * {reduce}u + i) * {inner}u + inner] - max_v); output[id] = expf(x[id] - max_v) / denom; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_sum_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = 0.0; for (uint step = 0; step < {reduce}u; ++step) v += x[(outer * {reduce}u + step) * {inner}u + inner]; output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_sum_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = 0.0f; for (unsigned step = 0; step < {reduce}u; ++step) v += x[(outer * {reduce}u + step) * {inner}u + inner]; output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.max_axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_max_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = -INFINITY; for (uint step = 0; step < {reduce}u; ++step) v = max(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_max_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = -INFINITY; for (unsigned step = 0; step < {reduce}u; ++step) v = fmaxf(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.min_axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_min_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = INFINITY; for (uint step = 0; step < {reduce}u; ++step) v = min(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_reduce_min_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = INFINITY; for (unsigned step = 0; step < {reduce}u; ++step) v = fminf(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    let mut source = match target {
        LoweringTarget::Metal => {
            if group.requires_rhs() {
                String::from("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_kernel(device const float* x [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) { float v = x[id];")
            } else {
                String::from(
                    "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_kernel(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) { float v = x[id];",
                )
            }
        }
        LoweringTarget::Cpu | LoweringTarget::Portable => {
            if group.requires_rhs() {
                String::from(
                    "void prism_kernel(const float* x, const float* rhs, float* output, unsigned id) { float v = x[id];",
                )
            } else {
                String::from("void prism_kernel(const float* x, float* output, unsigned id) { float v = x[id];")
            }
        }
    };
    for op in &group.ops {
        match op {
            KernelOp::BroadcastBinary { .. } => {
                // WAIVER: shape-aware broadcast is handled by the dedicated
                // renderer arm at the top of this function, which is the
                // only path that constructs a `BroadcastBinary` group.
                unreachable!("shape-aware broadcast requires a dedicated kernel ABI")
            }
            KernelOp::ReduceSum { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMax { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMin { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceSumAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMaxAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMinAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::SoftmaxAxis { .. } => unreachable!("softmax uses the dedicated renderer"),
            KernelOp::Attention { .. } => unreachable!("attention uses the dedicated renderer"),
            KernelOp::AttentionBatched { .. } => {
                unreachable!("attention uses the dedicated renderer")
            }
            KernelOp::RmsNorm { .. } => unreachable!("rms norm uses the dedicated renderer"),
            KernelOp::LayerNorm { .. } => unreachable!("layer norm uses the dedicated renderer"),
            KernelOp::Rope { .. } => unreachable!("rope uses the dedicated renderer"),
            KernelOp::Gather { .. } => unreachable!("gather uses the dedicated renderer"),
            KernelOp::Scatter { .. } => unreachable!("scatter uses the dedicated renderer"),
            KernelOp::Ssm { .. } => unreachable!("ssm uses the dedicated renderer"),
            KernelOp::MatMul { .. } => unreachable!("matmul uses the dedicated renderer"),
            KernelOp::Conv2d { .. } => unreachable!("conv2d uses the dedicated renderer"),
            KernelOp::Where { .. } => unreachable!("where uses the dedicated renderer"),
            KernelOp::Transpose { .. } => unreachable!("transpose uses the dedicated renderer"),
            KernelOp::Add {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "+")),
            KernelOp::Mul {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "*")),
            KernelOp::Sub {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "-")),
            KernelOp::Div {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "/")),
            KernelOp::Maximum {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_extremum(*scalar, *scalar_left, "max")),
            KernelOp::Minimum {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_extremum(*scalar, *scalar_left, "min")),
            KernelOp::Relu { .. } => source.push_str(" v = v > 0.0 ? v : 0.0;"),
            KernelOp::Neg { .. } => source.push_str(" v = -v;"),
            KernelOp::Exp { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = exp(v);"
            } else {
                " v = expf(v);"
            }),
            KernelOp::Sqrt { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = sqrt(v);"
            } else {
                " v = sqrtf(v);"
            }),
            KernelOp::Abs { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = abs(v);"
            } else {
                " v = fabsf(v);"
            }),
            KernelOp::Log { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = log(v);"
            } else {
                " v = logf(v);"
            }),
            KernelOp::Tanh { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = tanh(v);"
            } else {
                " v = tanhf(v);"
            }),
            KernelOp::Sin { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = sin(v);"
            } else {
                " v = sinf(v);"
            }),
            KernelOp::Cos { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = cos(v);"
            } else {
                " v = cosf(v);"
            }),
            KernelOp::Gelu { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));"
            } else {
                " v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));"
            }),
            KernelOp::Pow { exponent, .. } => {
                source.push_str(&if matches!(target, LoweringTarget::Metal) {
                    format!(" v = pow(v, {exponent});")
                } else {
                    format!(" v = powf(v, {exponent}f);")
                })
            }
            KernelOp::Cast { to, .. } => source.push_str(match to.as_str() {
                "f32" | "f16" | "bf16" => "",
                "i8" => " v = fminf(fmaxf(truncf(v), -128.0f), 127.0f);",
                "u8" => " v = fminf(fmaxf(truncf(v), 0.0f), 255.0f);",
                "i32" => " v = fminf(fmaxf(truncf(v), -2147483648.0f), 2147483647.0f);",
                "u32" => " v = fminf(fmaxf(truncf(v), 0.0f), 4294967295.0f);",
                // WAIVER: the `to` value is validated by
                // `TinyGraph::validate` for every `UOpKind::Cast` before the
                // renderer runs, so this match is exhaustive over the inputs
                // the type system allows.
                _ => unreachable!("validated cast target"),
            }),
        }
    }
    source.push_str(" output[id] = v; }");
    let mut digest = Sha256::new();
    digest.update(source.as_bytes());
    (source, hex_digest(digest.finalize()))
}

pub(crate) fn render_binary(scalar: Option<f32>, scalar_left: bool, operator: &str) -> String {
    let scalar = scalar.map(|value| value.to_string());
    let left = if scalar.is_some() && scalar_left {
        scalar.as_deref().unwrap()
    } else {
        "v"
    };
    let right = if scalar.is_some() && scalar_left {
        "v"
    } else {
        scalar.as_deref().unwrap_or("rhs[id]")
    };
    format!(" v = {left} {operator} {right};")
}

pub(crate) fn render_extremum(scalar: Option<f32>, scalar_left: bool, function: &str) -> String {
    let scalar = scalar.map(|value| value.to_string());
    let left = if scalar.is_some() && scalar_left {
        scalar.as_deref().unwrap()
    } else {
        "v"
    };
    let right = if scalar.is_some() && scalar_left {
        "v"
    } else {
        scalar.as_deref().unwrap_or("rhs[id]")
    };
    format!(" v = {function}({left}, {right});")
}

pub(crate) fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
