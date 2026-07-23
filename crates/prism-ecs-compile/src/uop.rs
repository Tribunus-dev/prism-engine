//! Backend compilation bridge for the compact SpatialIR UOp capture.
//!
//! `prism-spatial-ir` remains backend-neutral. This module is the boundary
//! that turns its rendered kernel groups into Prism's existing KernelArtifact
//! contract, so the normal CImage writer and runtime can own them.

use prism_ecs_ir::cimage_types::TensorShape;
use prism_ecs_kernel::{
    BackendKind, BindingDataType, BindingSlot, BufferRole, CpuBackend, DispatchGeometry,
    KernelArtifact, KernelBackend, KernelCompileRequest, KernelDescriptor, KernelVariant,
    MetalBackend,
};
use prism_spatial_ir::{CapturePlan, FusionStrategy, LoweringTarget, TinyGraph};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Operations with a complete shape, renderer, artifact, and runtime
/// contract. Only these custom operations may lower directly.
pub const VALIDATED_CUSTOM_OPERATIONS: &[&str] = &[
    "add",
    "mul",
    "sub",
    "div",
    "maximum",
    "max",
    "minimum",
    "min",
    "relu",
    "neg",
    "exp",
    "sqrt",
    "abs",
    "attention",
    "conv2d",
    "log",
    "tanh",
    "sin",
    "cos",
    "gelu",
    "gather",
    "log_softmax",
    "layer_norm",
    "rms_norm",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "ssm",
    "scatter",
    "clamp",
    "transpose",
    "where",
    "cast",
    "pow",
];

/// Operations recognized as compiler candidates but not yet admitted to
/// executable custom lowering. Candidate discovery can use this list without
/// accidentally treating a speculative operation as validated.
pub const CUSTOM_OPERATION_CANDIDATES: &[&str] = &[
    // Recognized semantic operations whose backend contracts are deliberately
    // not admitted until their shape, renderer, and behavioral evidence are
    // complete.
    "flash_attention",
    "group_norm",
    "topk",
];

pub(crate) fn strategy_kernel_prefix(strategy: &str) -> String {
    let strategy = sanitize_strategy_id(strategy);
    format!("prism_uop_strategy_{}__", strategy)
}

fn sanitize_strategy_id(strategy: &str) -> String {
    let sanitized: String = strategy
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "strategy".into()
    } else {
        sanitized
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomOperationClass {
    Validated,
    Candidate,
}

pub fn classify_custom_operation(operation: &str) -> CustomOperationClass {
    let operation = operation.to_ascii_lowercase();
    if VALIDATED_CUSTOM_OPERATIONS.contains(&operation.as_str()) || operation.starts_with("cast_") {
        CustomOperationClass::Validated
    } else if CUSTOM_OPERATION_CANDIDATES.contains(&operation.as_str()) {
        CustomOperationClass::Candidate
    } else {
        // An unrecognized operation is still a candidate after it passes
        // structural validation; classification must not become a second
        // rejection mechanism.
        CustomOperationClass::Candidate
    }
}

/// Validate a custom operation's structural tensor contract before assigning
/// its admission class. Unknown names deliberately follow this same path and
/// become candidates instead of bypassing validation or creating a third
/// classification state.
pub fn validate_and_classify_custom_operation(
    operation: &str,
    shape: &prism_spatial_ir::graph::ShapeContract,
) -> Result<CustomOperationClass, String> {
    let normalized = operation.to_ascii_lowercase();
    validate_custom_contract(&normalized, shape)?;
    Ok(classify_custom_operation(&normalized))
}

fn validate_custom_contract(
    operation: &str,
    shape: &prism_spatial_ir::graph::ShapeContract,
) -> Result<(), String> {
    if operation.is_empty()
        || !operation.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
    {
        return Err(format!("custom operation name '{operation}' is malformed"));
    }
    if shape.in_shapes.is_empty() || shape.in_shapes.len() > 4 || shape.out_shapes.len() != 1 {
        return Err(format!(
            "custom operation '{operation}' has an invalid input/output arity"
        ));
    }
    if shape
        .in_shapes
        .iter()
        .chain(shape.out_shapes.iter())
        .any(|tensor| tensor.dims.is_empty() || tensor.dims.iter().any(|dimension| *dimension == 0))
    {
        return Err(format!(
            "custom operation '{operation}' has an invalid tensor shape"
        ));
    }
    if shape
        .in_shapes
        .iter()
        .chain(shape.out_shapes.iter())
        .any(|tensor| {
            tensor
                .dims
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .is_none()
        })
    {
        return Err(format!(
            "custom operation '{operation}' has a tensor shape whose element count overflows"
        ));
    }
    Ok(())
}

fn pow_exponent_from_metadata(
    metadata: Option<&prism_spatial_ir::graph::NodeMeta>,
) -> Result<f32, String> {
    let exponent = metadata
        .and_then(|meta| meta.pow_exponent)
        .ok_or_else(|| "custom pow requires a pow_exponent annotation".to_string())?;
    exponent
        .is_finite()
        .then_some(exponent)
        .ok_or_else(|| "custom pow exponent must be finite".to_string())
}

fn broadcast_shape_contract(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1; rank];
    for shape in shapes {
        let offset = rank.saturating_sub(shape.len());
        for (axis, dimension) in shape.iter().copied().enumerate() {
            let slot = &mut output[offset + axis];
            if *slot != dimension && *slot != 1 && dimension != 1 {
                return None;
            }
            *slot = (*slot).max(dimension);
        }
    }
    Some(output)
}

fn transpose_permutation(
    input: &TensorShape,
    output: &TensorShape,
    requested: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    if input.dims.len() != output.dims.len() {
        return Err("transpose requires equal input/output rank".into());
    }
    let permutation = requested.map(|value| value.to_vec()).unwrap_or_else(|| {
        output
            .dims
            .iter()
            .filter_map(|dimension| {
                input
                    .dims
                    .iter()
                    .position(|candidate| candidate == dimension)
            })
            .collect()
    });
    if permutation.len() != input.dims.len()
        || permutation
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != permutation.len()
        || permutation.iter().any(|axis| *axis >= input.dims.len())
        || permutation
            .iter()
            .enumerate()
            .any(|(axis, source)| output.dims[axis] != input.dims[*source])
    {
        return Err("transpose permutation is not a valid bijection for the shape contract".into());
    }
    Ok(permutation)
}

fn parse_cast_operation(operation: &str) -> Result<(&str, &str), String> {
    let (from, to) = if operation == "cast" {
        ("f32", "f32")
    } else if let Some(types) = operation.strip_prefix("cast_") {
        types.split_once("_to_").ok_or_else(|| {
            format!("custom cast '{operation}' must use cast_<source>_to_<target>")
        })?
    } else {
        return Err(format!("custom operation '{operation}' is not a cast"));
    };
    let supported =
        |dtype: &str| matches!(dtype, "f32" | "f16" | "bf16" | "i8" | "u8" | "i32" | "u32");
    if !supported(from) || !supported(to) {
        return Err(format!("unsupported cast dtype '{from}' to '{to}'"));
    }
    Ok((from, to))
}

pub fn compile_uop_capture(capture: &CapturePlan) -> Result<Vec<KernelArtifact>, String> {
    let mut artifacts = Vec::with_capacity(capture.kernels.len());
    for (index, kernel) in capture.kernels.iter().enumerate() {
        let (backend, compiler): (BackendKind, Box<dyn KernelBackend>) = match capture.target {
            LoweringTarget::Cpu | LoweringTarget::Portable => {
                (BackendKind::CPU, Box::new(CpuBackend))
            }
            LoweringTarget::Metal => (BackendKind::Metal, Box::new(MetalBackend::default())),
        };
        let is_reduction = kernel.group.is_reduction();
        let max_reduction = kernel.group.max_reduction();
        let min_reduction = kernel.group.min_reduction();
        let axis_reduction = kernel.group.axis_reduction();
        let max_axis_reduction = kernel.group.max_axis_reduction();
        let min_axis_reduction = kernel.group.min_axis_reduction();
        let softmax_shape = kernel.group.softmax_shape();
        let attention_shape = kernel.group.attention_shape();
        let batched_attention_shape = kernel.group.batched_attention_shape();
        let rms_norm_shape = kernel.group.rms_norm_shape();
        let layer_norm_shape = kernel.group.layer_norm_shape();
        let rope_shape = kernel.group.rope_shape();
        let gather_shape = kernel.group.gather_shape();
        let scatter_shape = kernel.group.scatter_shape();
        let ssm_shape = kernel.group.ssm_shape();
        let conv2d_shape = kernel.group.conv2d_shape();
        let matmul_shape = kernel.group.matmul_shape();
        let transpose_shape = kernel.group.transpose_shape();
        let broadcast_binary_shape = kernel.group.broadcast_binary_shape();
        let input_elements = kernel
            .group
            .input_elements()
            .or(kernel.output_elements)
            .unwrap_or(64) as u32;
        let output_elements = matmul_shape
            .map(|(m, _, n)| (m * n) as u32)
            .or_else(|| rope_shape.map(|(rows, features)| (rows * features) as u32))
            .or_else(|| gather_shape.map(|(rows, _, features)| (rows * features) as u32))
            .or_else(|| scatter_shape.map(|(rows, _, features)| (rows * features) as u32))
            .or_else(|| ssm_shape.map(|(rows, features)| (rows * features) as u32))
            .or_else(|| {
                transpose_shape
                    .as_ref()
                    .map(|(_, _, output)| output.iter().product::<u64>() as u32)
            })
            .or_else(|| {
                broadcast_binary_shape
                    .as_ref()
                    .map(|(_, _, _, output)| output.iter().product::<u64>() as u32)
            })
            .or_else(|| {
                conv2d_shape.map(
                    |(
                        batch,
                        _,
                        height,
                        width,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding,
                    )| {
                        (batch
                            * out_channels
                            * ((height + 2 * padding - kernel_h) / stride + 1)
                            * ((width + 2 * padding - kernel_w) / stride + 1))
                            as u32
                    },
                )
            })
            .or(kernel.output_elements.map(|elements| elements as u32))
            .unwrap_or(input_elements);
        let mut binding_signature = vec![BindingSlot {
            index: 0,
            role: BufferRole::Input,
            data_type: BindingDataType::Float32,
        }];
        if kernel.group.requires_rhs() || matmul_shape.is_some() {
            binding_signature.push(BindingSlot {
                index: 1,
                role: BufferRole::Input,
                data_type: BindingDataType::Float32,
            });
        }
        if kernel.group.requires_tertiary_input() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if rms_norm_shape.is_some() {
            binding_signature.push(BindingSlot {
                index: 1,
                role: BufferRole::Input,
                data_type: BindingDataType::Float32,
            });
        }
        if layer_norm_shape.is_some() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if rope_shape.is_some() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if gather_shape.is_some() {
            binding_signature.push(BindingSlot {
                index: 1,
                role: BufferRole::Input,
                data_type: BindingDataType::Float32,
            });
        }
        if scatter_shape.is_some() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if ssm_shape.is_some() {
            for index in 1..=3 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if conv2d_shape.is_some() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        if attention_shape.is_some() || batched_attention_shape.is_some() {
            for index in 1..=2 {
                binding_signature.push(BindingSlot {
                    index,
                    role: BufferRole::Input,
                    data_type: BindingDataType::Float32,
                });
            }
        }
        binding_signature.push(BindingSlot {
            index: binding_signature.len() as u32,
            role: BufferRole::Output,
            data_type: BindingDataType::Float32,
        });
        let descriptor = KernelDescriptor {
            name: format!("prism_uop_{index}"),
            variant: KernelVariant::Custom(
                if let Some((operation, lhs, rhs, output)) = &broadcast_binary_shape {
                    let operation = format!("{operation:?}").to_lowercase();
                    let program = kernel.group.broadcast_program().unwrap_or_default();
                    format!(
                        "uop_broadcast_binary:{operation}:{}:{}:{}:{program}",
                        lhs.iter().map(u64::to_string).collect::<Vec<_>>().join("x"),
                        rhs.iter().map(u64::to_string).collect::<Vec<_>>().join("x"),
                        output
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join("x")
                    )
                } else if kernel.group.requires_tertiary_input() {
                    "uop_where".into()
                } else if matmul_shape.is_some() {
                    let (m, k, n) = matmul_shape.unwrap();
                    format!("uop_matmul:{m}:{k}:{n}")
                } else if let Some((outer, reduce, inner)) = axis_reduction {
                    format!("uop_reduce_sum_axis:{outer}:{reduce}:{inner}")
                } else if let Some((outer, reduce, inner)) = max_axis_reduction {
                    format!("uop_reduce_max_axis:{outer}:{reduce}:{inner}")
                } else if let Some((outer, reduce, inner)) = min_axis_reduction {
                    format!("uop_reduce_min_axis:{outer}:{reduce}:{inner}")
                } else if let Some((outer, reduce, inner)) = softmax_shape {
                    format!("uop_softmax_axis:{outer}:{reduce}:{inner}")
                } else if let Some((seq, head, scale)) = attention_shape {
                    format!("uop_attention:{seq}:{head}:{scale}")
                } else if let Some((batch, seq, head, scale)) = batched_attention_shape {
                    format!("uop_attention_batched:{batch}:{seq}:{head}:{scale}")
                } else if let Some((rows, features, epsilon)) = rms_norm_shape {
                    format!("uop_rms_norm:{rows}:{features}:{epsilon}")
                } else if let Some((rows, features, epsilon)) = layer_norm_shape {
                    format!("uop_layer_norm:{rows}:{features}:{epsilon}")
                } else if let Some((rows, features)) = rope_shape {
                    format!("uop_rope:{rows}:{features}")
                } else if let Some((rows, vocab, features)) = gather_shape {
                    format!("uop_gather:{rows}:{vocab}:{features}")
                } else if let Some((rows, updates, features)) = scatter_shape {
                    format!("uop_scatter:{rows}:{updates}:{features}")
                } else if let Some((rows, features)) = ssm_shape {
                    format!("uop_ssm:{rows}:{features}")
                } else if let Some((
                    batch,
                    in_channels,
                    height,
                    width,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride,
                    padding,
                )) = conv2d_shape
                {
                    format!("uop_conv2d:{batch}:{in_channels}:{height}:{width}:{out_channels}:{kernel_h}:{kernel_w}:{stride}:{padding}")
                } else if let Some((permutation, input, output)) = &transpose_shape {
                    format!(
                        "uop_transpose:{}:{}:{}",
                        permutation
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                        input
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join("x"),
                        output
                            .iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join("x")
                    )
                } else if max_reduction.is_some() {
                    "uop_reduce_max".into()
                } else if min_reduction.is_some() {
                    "uop_reduce_min".into()
                } else if is_reduction {
                    "uop_reduce_sum".into()
                } else if let Some(scalar) = kernel.group.scalar_elementwise_variant() {
                    format!("uop_elementwise_scalar:{scalar}")
                } else if kernel.group.ops.len() > 1 {
                    if let Some(program) = kernel.group.elementwise_program() {
                        format!("uop_elementwise_program:{program}")
                    } else {
                        "uop_elementwise".into()
                    }
                } else if let Some(program) = kernel.group.unary_elementwise_program() {
                    format!("uop_elementwise:{program}")
                } else if let Some(operation) = kernel.group.binary_elementwise_variant() {
                    format!("uop_elementwise_binary:{operation}")
                } else {
                    "uop_elementwise".into()
                },
            ),
            backend,
            source_digest: kernel.source_digest.clone(),
            binary_digest: String::new(),
            binding_signature,
            dispatch_geometry: DispatchGeometry {
                threads_per_threadgroup: [if is_reduction { 1 } else { 64 }, 1, 1],
                threadgroups_per_grid: [1, 1, 1],
                threads_per_grid: [
                    if axis_reduction.is_some()
                        || max_axis_reduction.is_some()
                        || min_axis_reduction.is_some()
                    {
                        (axis_reduction
                            .or(max_axis_reduction)
                            .or(min_axis_reduction)
                            .map(|(outer, _, inner)| outer * inner)
                            .unwrap()) as u32
                    } else if is_reduction {
                        1
                    } else if matmul_shape.is_some()
                        || transpose_shape.is_some()
                        || broadcast_binary_shape.is_some()
                    {
                        output_elements
                    } else if let Some((rows, features)) = rope_shape {
                        (rows * (features / 2)) as u32
                    } else if let Some((rows, _, features)) = gather_shape {
                        (rows * features) as u32
                    } else if let Some((rows, features)) = ssm_shape {
                        (rows * features) as u32
                    } else if let Some((outer, _, inner)) = axis_reduction {
                        (outer * inner) as u32
                    } else if let Some((outer, reduce, inner)) = softmax_shape {
                        (outer * reduce * inner) as u32
                    } else if let Some((seq, head, _)) = attention_shape {
                        (seq * head) as u32
                    } else if let Some((batch, seq, head, _)) = batched_attention_shape {
                        (batch * seq * head) as u32
                    } else if let Some((rows, features, _)) = rms_norm_shape {
                        (rows * features) as u32
                    } else if let Some((rows, features, _)) = layer_norm_shape {
                        (rows * features) as u32
                    } else if let Some((
                        batch,
                        _,
                        height,
                        width,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding,
                    )) = conv2d_shape
                    {
                        (batch
                            * out_channels
                            * ((height + 2 * padding - kernel_h) / stride + 1)
                            * ((width + 2 * padding - kernel_w) / stride + 1))
                            as u32
                    } else {
                        input_elements
                    },
                    1,
                    1,
                ],
            },
        };
        let artifact = compiler
            .compile(&KernelCompileRequest {
                source: kernel.source.as_bytes().to_vec(),
                descriptor,
                source_path: None,
            })
            .map_err(|error| format!("compile UOp kernel {index}: {error}"))?;
        artifacts.push(artifact);
    }
    Ok(artifacts)
}

/// Lower and compile one concrete fusion strategy through the normal artifact
/// boundary. Keeping this operation here makes strategy evaluation produce
/// the same executable artifacts as the ordinary UOp path; callers do not
/// need to lower a strategy manually and risk measuring a different capture
/// than the one that will be published.
pub fn compile_uop_graph_with_strategy(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategy: &FusionStrategy,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    let capture = graph
        .lower_with_fusion_strategy(target, strategy)
        .map_err(|error| format!("lower UOp graph with {strategy:?}: {error}"))?;
    let artifacts = compile_uop_capture(&capture)?;
    Ok((capture, artifacts))
}

/// Compile every requested strategy as an independent executable candidate.
/// Each candidate owns its capture and artifact vector, so comparing them at
/// runtime cannot accidentally share a fused layout or compiler payload.
pub fn compile_uop_graph_strategies(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
) -> Result<Vec<(FusionStrategy, CapturePlan, Vec<KernelArtifact>)>, String> {
    let mut stable_ids = std::collections::BTreeSet::new();
    for (index, strategy) in strategies.iter().enumerate() {
        if strategies[..index].contains(strategy) {
            return Err(format!(
                "duplicate UOp fusion strategy at index {index}: {strategy:?}"
            ));
        }
        if !stable_ids.insert(strategy.stable_id()) {
            return Err(format!(
                "duplicate UOp strategy runtime namespace '{}' at index {index}",
                strategy.stable_id()
            ));
        }
    }
    strategies
        .iter()
        .map(|strategy| {
            let (capture, artifacts) = compile_uop_graph_with_strategy(graph, target, strategy)?;
            Ok((strategy.clone(), capture, artifacts))
        })
        .collect()
}

/// Measure compiled UOp strategy candidates through repeated CPU/portable
/// dispatch. This is intentionally below the workload model: callers can run
/// it once for each realtime or batch scenario and feed the returned timings
/// into SpatialIR's measured strategy evaluator.
pub fn benchmark_uop_graph_strategies(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
    inputs: &BTreeMap<String, Vec<f32>>,
    iterations: usize,
) -> Result<Vec<prism_spatial_ir::FusionMeasurement>, String> {
    if !matches!(target, LoweringTarget::Cpu | LoweringTarget::Portable) {
        return Err("UOp CPU benchmark only supports CPU and portable targets".into());
    }
    let candidates = compile_and_validate_uop_graph_strategies(graph, target, strategies, inputs)?;
    let iterations = iterations.max(1);
    candidates
        .iter()
        .enumerate()
        .map(|(candidate_index, (_, capture, _))| {
            let program = UOpCompiledProgram::compile(capture.clone())?;
            let start = Instant::now();
            for _ in 0..iterations {
                program
                    .dispatch_cpu(inputs)
                    .map_err(|error| format!("strategy benchmark dispatch failed: {error}"))?;
            }
            let elapsed = (start.elapsed().as_nanos() as u64 / iterations as u64).max(1);
            let materialized_bytes = capture.memory_plan.slot_count as u64 * 4;
            Ok(prism_spatial_ir::FusionMeasurement {
                candidate_index,
                latency_ns: elapsed,
                materialized_bytes,
            })
        })
        .collect()
}

/// Benchmark UOp candidates through a caller-owned backend runner. This is
/// the heterogeneous counterpart to [`benchmark_uop_graph_strategies`]: the
/// runner may submit Metal, XDNA, or another device-specific executable while
/// the candidate compilation and measurement schema remain identical.
pub fn benchmark_uop_graph_strategies_with_runner<F>(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
    mut run: F,
) -> Result<Vec<prism_spatial_ir::FusionMeasurement>, String>
where
    F: FnMut(usize, &FusionStrategy, &CapturePlan) -> Result<(u64, u64), String>,
{
    let mut candidates = compile_uop_graph_strategies(graph, target, strategies)?;
    for (_, capture, artifacts) in &mut candidates {
        *artifacts = compile_and_validate_uop_capture(capture)?;
    }
    candidates
        .iter()
        .enumerate()
        .map(|(candidate_index, (strategy, capture, _))| {
            let (latency_ns, materialized_bytes) = run(candidate_index, strategy, capture)?;
            Ok(prism_spatial_ir::FusionMeasurement {
                candidate_index,
                latency_ns,
                materialized_bytes,
            })
        })
        .collect()
}

/// Measured strategy timings partitioned by workload scenario. The caller
/// supplies scenario-shaped inputs, allowing realtime and batched inference
/// to be benchmarked against the same candidate set without conflating their
/// latency distributions.
#[derive(Debug, Clone)]
pub struct UOpWorkloadMeasurement {
    pub scenario: prism_spatial_ir::WorkloadScenario,
    pub measurements: Vec<prism_spatial_ir::FusionMeasurement>,
}

/// The measured execution policy for one concrete workload scenario.
///
/// Keeping the scenario beside the selected strategy is intentional: realtime
/// single-token inference and larger batched execution are allowed to choose
/// different kernels without collapsing their evidence into one global winner.
#[derive(Debug, Clone)]
pub struct UOpWorkloadSelection {
    pub scenario: prism_spatial_ir::WorkloadScenario,
    pub strategy_id: String,
    pub measurement: prism_spatial_ir::FusionMeasurement,
}

/// Select one measured strategy independently for every workload scenario.
pub fn select_measured_uop_workloads(
    strategies: &[FusionStrategy],
    workloads: &[UOpWorkloadMeasurement],
) -> Result<Vec<UOpWorkloadSelection>, String> {
    workloads
        .iter()
        .map(|workload| {
            workload
                .scenario
                .validate()
                .map_err(|error| format!("scenario {:?}: {error}", workload.scenario))?;
            let (strategy_id, measurement) =
                select_measured_uop_strategy(strategies, &workload.measurements)
                    .map_err(|error| format!("scenario {:?}: {error}", workload.scenario))?;
            Ok(UOpWorkloadSelection {
                scenario: workload.scenario,
                strategy_id,
                measurement,
            })
        })
        .collect()
}

/// Select the lowest-cost measured candidate while preserving its stable
/// runtime namespace. The same score is used by SpatialIR's fusion evaluator
/// so a measured UOp choice cannot diverge from artifact policy selection.
pub fn select_measured_uop_strategy(
    strategies: &[FusionStrategy],
    measurements: &[prism_spatial_ir::FusionMeasurement],
) -> Result<(String, prism_spatial_ir::FusionMeasurement), String> {
    if strategies.len() != measurements.len() || strategies.is_empty() {
        return Err("UOp strategy measurements do not match candidate set".into());
    }
    if measurements
        .iter()
        .enumerate()
        .any(|(index, measurement)| measurement.candidate_index != index)
    {
        return Err("UOp strategy measurements have inconsistent candidate indices".into());
    }
    if measurements
        .iter()
        .map(|measurement| measurement.candidate_index)
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != measurements.len()
    {
        return Err("UOp strategy measurements contain duplicate candidate indices".into());
    }
    if measurements
        .iter()
        .any(|measurement| measurement.latency_ns == 0)
    {
        return Err("UOp strategy measurements contain a zero-latency sample".into());
    }
    let (index, measurement) = measurements
        .iter()
        .enumerate()
        .min_by_key(|(_, measurement)| {
            measurement
                .latency_ns
                .saturating_add(measurement.materialized_bytes / 100)
        })
        .ok_or_else(|| "UOp strategy measurements are empty".to_string())?;
    Ok((
        strategies[index].stable_id().to_string(),
        measurement.clone(),
    ))
}

pub fn benchmark_uop_graph_workloads<F>(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
    scenarios: &[prism_spatial_ir::WorkloadScenario],
    mut inputs_for: F,
    iterations: usize,
) -> Result<Vec<UOpWorkloadMeasurement>, String>
where
    F: FnMut(prism_spatial_ir::WorkloadScenario) -> BTreeMap<String, Vec<f32>>,
{
    scenarios
        .iter()
        .copied()
        .map(|scenario| {
            scenario.validate()?;
            let inputs = inputs_for(scenario);
            let measurements =
                benchmark_uop_graph_strategies(graph, target, strategies, &inputs, iterations)?;
            Ok(UOpWorkloadMeasurement {
                scenario,
                measurements,
            })
        })
        .collect()
}

/// Scenario-partitioned heterogeneous benchmarking. Each invocation gets a
/// fresh scenario-shaped input map and can submit candidates to a device
/// runtime through the same runner contract used by single-workload tests.
pub fn benchmark_uop_graph_workloads_with_runner<F, I>(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
    scenarios: &[prism_spatial_ir::WorkloadScenario],
    mut inputs_for: I,
    mut run: F,
) -> Result<Vec<UOpWorkloadMeasurement>, String>
where
    I: FnMut(prism_spatial_ir::WorkloadScenario) -> BTreeMap<String, Vec<f32>>,
    F: FnMut(
        prism_spatial_ir::WorkloadScenario,
        usize,
        &FusionStrategy,
        &CapturePlan,
        &BTreeMap<String, Vec<f32>>,
    ) -> Result<(u64, u64), String>,
{
    scenarios
        .iter()
        .copied()
        .map(|scenario| {
            scenario.validate()?;
            let inputs = inputs_for(scenario);
            let candidates = if matches!(target, LoweringTarget::Cpu | LoweringTarget::Portable) {
                compile_and_validate_uop_graph_strategies(graph, target, strategies, &inputs)?
            } else {
                compile_uop_graph_strategies(graph, target, strategies)?
            };
            let measurements = candidates
                .iter()
                .enumerate()
                .map(|(candidate_index, (strategy, capture, _))| {
                    let artifacts = compile_and_validate_uop_capture(capture)?;
                    let _ = artifacts;
                    let (latency_ns, materialized_bytes) =
                        run(scenario, candidate_index, strategy, capture, &inputs)?;
                    if latency_ns == 0 {
                        return Err(format!(
                            "workload runner returned a zero-latency sample for candidate {candidate_index}"
                        ));
                    }
                    Ok(prism_spatial_ir::FusionMeasurement {
                        candidate_index,
                        latency_ns,
                        materialized_bytes,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(UOpWorkloadMeasurement {
                scenario,
                measurements,
            })
        })
        .collect()
}

/// Compile a strategy set and require every portable candidate to agree with
/// the graph's behavioral oracle. This is the admission gate used before a
/// measured candidate can be promoted into a validated execution policy.
pub fn compile_and_validate_uop_graph_strategies(
    graph: &TinyGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
    inputs: &BTreeMap<String, Vec<f32>>,
) -> Result<Vec<(FusionStrategy, CapturePlan, Vec<KernelArtifact>)>, String> {
    let expected = graph
        .execute_f32(inputs)
        .map_err(|error| format!("reference validation failed: {error}"))?;
    let candidates = compile_uop_graph_strategies(graph, target, strategies)?;
    if matches!(target, LoweringTarget::Cpu | LoweringTarget::Portable) {
        for (strategy, capture, _) in &candidates {
            let program = UOpCompiledProgram::compile(capture.clone())?;
            let actual = program
                .dispatch_cpu(inputs)
                .map_err(|error| format!("{strategy:?} CPU validation failed: {error}"))?
                .outputs;
            if actual != expected {
                return Err(format!(
                    "{strategy:?} failed behavioral equivalence against the reference graph"
                ));
            }
        }
    }
    Ok(candidates)
}

/// Build a compact executable UOp capture from a SpatialGraph MatMul shape.
///
/// SpatialGraph deliberately stores operation class and tensor contracts
/// separately from backend details. This adapter is therefore intentionally
/// limited to the operation whose semantics are fully represented by that
/// contract; generic `Elementwise` nodes require an operation annotation
/// before they can be lowered without guessing.
pub fn compile_spatial_matmul(
    m: usize,
    k: usize,
    n: usize,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    if m == 0 || k == 0 || n == 0 {
        return Err("SpatialGraph MatMul dimensions must be non-zero".into());
    }
    let mut graph = prism_spatial_ir::TinyGraph::default();
    let a = graph.add(
        prism_spatial_ir::UOpKind::Input { name: "a".into() },
        vec![],
        vec![m as u64, k as u64],
    );
    let b = graph.add(
        prism_spatial_ir::UOpKind::Input { name: "b".into() },
        vec![],
        vec![k as u64, n as u64],
    );
    let product = graph.add(
        prism_spatial_ir::UOpKind::MatMul { m, k, n },
        vec![a, b],
        vec![m as u64, n as u64],
    );
    graph.add(
        prism_spatial_ir::UOpKind::Output { name: "out".into() },
        vec![product],
        vec![m as u64, n as u64],
    );
    let capture = graph
        .lower(target)
        .map_err(|error| format!("lower SpatialGraph MatMul: {error}"))?;
    let artifacts = compile_and_validate_uop_capture(&capture)?;
    Ok((capture, artifacts))
}

/// Lower a concrete SpatialGraph compute node when its contract is sufficient
/// to determine semantics. MatMul is currently the first such operation.
pub fn compile_spatial_node(
    node: &prism_spatial_ir::SpatialNode,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    compile_spatial_node_with_metadata(node, None, target)
}

/// Lower a SpatialGraph node using optional semantic annotations.
pub fn compile_spatial_node_with_metadata(
    node: &prism_spatial_ir::SpatialNode,
    metadata: Option<&prism_spatial_ir::graph::NodeMeta>,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Custom(operation),
        shape,
        ..
    } = node
    {
        let operation = operation.to_ascii_lowercase();
        let classification = validate_and_classify_custom_operation(&operation, shape)?;
        if operation == "pow" {
            let input = shape
                .in_shapes
                .first()
                .ok_or_else(|| "custom pow requires one input".to_string())?;
            let output = shape
                .out_shapes
                .first()
                .ok_or_else(|| "custom pow requires one output".to_string())?;
            if shape.in_shapes.len() != 1 || input.dims != output.dims {
                return Err("custom pow must preserve shape and have one input".into());
            }
            let exponent = pow_exponent_from_metadata(metadata)?;
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let input_id = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "input".into(),
                },
                vec![],
                input.dims.iter().map(|dim| *dim as u64).collect(),
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Pow { exponent },
                vec![input_id],
                output.dims.iter().map(|dim| *dim as u64).collect(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                output.dims.iter().map(|dim| *dim as u64).collect(),
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom pow: {error}"))?;
            let artifacts = compile_and_validate_uop_capture(&capture)?;
            return Ok((capture, artifacts));
        }
        match classification {
            CustomOperationClass::Validated => {}
            CustomOperationClass::Candidate => {
                return Err(format!(
                    "custom operation '{operation}' is a candidate, not validated"
                ));
            }
        }
        if operation == "transpose" {
            let input = shape
                .in_shapes
                .first()
                .ok_or("custom transpose requires one input")?;
            let output = shape.out_shapes.first().unwrap();
            let permutation = transpose_permutation(
                input,
                output,
                metadata.and_then(|meta| meta.permutation.as_deref()),
            )?;
            let dims = input.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let output_dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims,
            );
            graph.add(
                prism_spatial_ir::UOpKind::Transpose { permutation },
                vec![x],
                output_dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![prism_spatial_ir::UOpId(1)],
                output_dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom transpose: {error}"))?;
            let artifacts = compile_and_validate_uop_capture(&capture)?;
            return Ok((capture, artifacts));
        }
        if operation == "where" {
            let [condition, when_true, when_false] = shape.in_shapes.as_slice() else {
                return Err("custom where requires condition, true, and false shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let broadcast =
                broadcast_shape_contract(&[&condition.dims, &when_true.dims, &when_false.dims])
                    .ok_or_else(|| {
                        "custom where shapes are not broadcast-compatible".to_string()
                    })?;
            if broadcast != output.dims {
                return Err("custom where output shape does not match broadcast shape".into());
            }
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let c = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "condition".into(),
                },
                vec![],
                dims.clone(),
            );
            let t = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "when_true".into(),
                },
                vec![],
                dims.clone(),
            );
            let f = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "when_false".into(),
                },
                vec![],
                dims.clone(),
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Where,
                vec![c, t, f],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom where: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "clamp" {
            let [input, lower, upper] = shape.in_shapes.as_slice() else {
                return Err("custom clamp requires input, lower, and upper shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            if input.dims != output.dims || lower.dims != output.dims || upper.dims != output.dims {
                return Err("custom clamp shapes must match".into());
            }
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let lo = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "lower".into(),
                },
                vec![],
                dims.clone(),
            );
            let hi = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "upper".into(),
                },
                vec![],
                dims.clone(),
            );
            let lower_value = graph.add(
                prism_spatial_ir::UOpKind::Maximum,
                vec![x, lo],
                dims.clone(),
            );
            let result = graph.add(
                prism_spatial_ir::UOpKind::Minimum,
                vec![lower_value, hi],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![result],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom clamp: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "softmax" || operation == "log_softmax" {
            let [input] = shape.in_shapes.as_slice() else {
                return Err(format!("custom {operation} requires one input shape"));
            };
            let output = shape.out_shapes.first().unwrap();
            if input.dims != output.dims || input.dims.is_empty() {
                return Err(format!(
                    "custom {operation} shapes must match and be nonempty"
                ));
            }
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let softmax = graph.add(
                prism_spatial_ir::UOpKind::SoftmaxAxis {
                    axis: input.dims.len() - 1,
                },
                vec![x],
                dims.clone(),
            );
            let value = if operation == "log_softmax" {
                graph.add(prism_spatial_ir::UOpKind::Log, vec![softmax], dims.clone())
            } else {
                softmax
            };
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom {operation}: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "attention" {
            let [q, k, v] = shape.in_shapes.as_slice() else {
                return Err("custom attention requires Q, K, and V shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            if q.dims != k.dims || q.dims != v.dims || q.dims != output.dims {
                return Err("custom attention Q/K/V/output shapes must match".into());
            }
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let q_value = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "q".into() },
                vec![],
                dims.clone(),
            );
            let k_value = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "k".into() },
                vec![],
                dims.clone(),
            );
            let v_value = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "v".into() },
                vec![],
                dims.clone(),
            );
            let value = match output.dims.as_slice() {
                [seq, head] if *seq > 0 && *head > 0 => prism_spatial_ir::UOpKind::Attention {
                    seq: *seq,
                    head: *head,
                    scale: 1.0 / (*head as f32).sqrt(),
                },
                [batch, seq, head] if *batch > 0 && *seq > 0 && *head > 0 => {
                    prism_spatial_ir::UOpKind::AttentionBatched {
                        batch: *batch,
                        seq: *seq,
                        head: *head,
                        scale: 1.0 / (*head as f32).sqrt(),
                    }
                }
                _ => return Err("custom attention requires rank-2 or rank-3 tensors".into()),
            };
            let result = graph.add(value, vec![q_value, k_value, v_value], dims.clone());
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![result],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom attention: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "conv2d" {
            let [input, weight, bias] = shape.in_shapes.as_slice() else {
                return Err("custom conv2d requires input, weight, and bias shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let [batch, in_channels, height, width] = input.dims.as_slice() else {
                return Err("custom conv2d input must be NCHW".into());
            };
            let [out_channels, weight_in, kernel_h, kernel_w] = weight.dims.as_slice() else {
                return Err("custom conv2d weight must be OIHW".into());
            };
            let [bias_channels] = bias.dims.as_slice() else {
                return Err("custom conv2d bias must be rank-1".into());
            };
            let stride = metadata
                .and_then(|meta| meta.convolution_stride)
                .unwrap_or(1);
            let padding = metadata
                .and_then(|meta| meta.convolution_padding)
                .unwrap_or(0);
            if stride == 0
                || *weight_in != *in_channels
                || *bias_channels != *out_channels
                || *height + 2 * padding < *kernel_h
                || *width + 2 * padding < *kernel_w
            {
                return Err("custom conv2d input contract is inconsistent".into());
            }
            let expected = vec![
                *batch,
                *out_channels,
                (*height + 2 * padding - *kernel_h) / stride + 1,
                (*width + 2 * padding - *kernel_w) / stride + 1,
            ];
            if output.dims != expected {
                return Err("custom conv2d output contract is inconsistent".into());
            }
            let input_dims = input.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let weight_dims = weight
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let bias_dims = bias.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let output_dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                input_dims,
            );
            let w = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "weight".into(),
                },
                vec![],
                weight_dims,
            );
            let b = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "bias".into(),
                },
                vec![],
                bias_dims,
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Conv2d {
                    batch: *batch,
                    in_channels: *in_channels,
                    height: *height,
                    width: *width,
                    out_channels: *out_channels,
                    kernel_h: *kernel_h,
                    kernel_w: *kernel_w,
                    stride,
                    padding,
                },
                vec![x, w, b],
                output_dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                output_dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom conv2d: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "rms_norm" || operation == "rmsnorm" {
            let [input, weight] = shape.in_shapes.as_slice() else {
                return Err("custom rms_norm requires input and weight shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let [rows, features] = output.dims.as_slice() else {
                return Err("custom rms_norm requires rank-2 output".into());
            };
            if input.dims != output.dims || weight.dims != vec![*features] {
                return Err("custom rms_norm shape contract is inconsistent".into());
            }
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let w = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "weight".into(),
                },
                vec![],
                vec![*features as u64],
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::RmsNorm {
                    rows: *rows,
                    features: *features,
                    epsilon: 1e-5,
                },
                vec![x, w],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom rms_norm: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "ssm" {
            let [input, decay, input_gain, output_gain] = shape.in_shapes.as_slice() else {
                return Err(
                    "custom ssm requires input, decay, input gain, and output gain shapes".into(),
                );
            };
            let output = shape.out_shapes.first().unwrap();
            let [rows, features] = input.dims.as_slice() else {
                return Err("custom ssm input must be rank-2".into());
            };
            if output.dims != input.dims
                || decay.dims != vec![*features]
                || input_gain.dims != vec![*features]
                || output_gain.dims != vec![*features]
            {
                return Err("custom ssm shape contract is inconsistent".into());
            }
            let input_dims = input.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let feature_dims = vec![*features as u64];
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "input".into(),
                },
                vec![],
                input_dims.clone(),
            );
            let decay_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "decay".into(),
                },
                vec![],
                feature_dims.clone(),
            );
            let input_gain_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "input_gain".into(),
                },
                vec![],
                feature_dims.clone(),
            );
            let output_gain_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "output_gain".into(),
                },
                vec![],
                feature_dims,
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Ssm {
                    rows: *rows,
                    features: *features,
                },
                vec![x, decay_value, input_gain_value, output_gain_value],
                input_dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                input_dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom ssm: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "gather" {
            let [weight, indices] = shape.in_shapes.as_slice() else {
                return Err("custom gather requires weight and indices shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let [vocab, features] = weight.dims.as_slice() else {
                return Err("custom gather weight must be rank-2".into());
            };
            let [rows] = indices.dims.as_slice() else {
                return Err("custom gather indices must be rank-1".into());
            };
            if *vocab == 0 || *features == 0 || output.dims != vec![*rows, *features] {
                return Err("custom gather shape contract is inconsistent".into());
            }
            let weight_dims = weight
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let index_dims = indices
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let output_dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let weight_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "weight".into(),
                },
                vec![],
                weight_dims,
            );
            let index_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "indices".into(),
                },
                vec![],
                index_dims,
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Gather {
                    rows: *rows,
                    vocab: *vocab,
                    features: *features,
                },
                vec![weight_value, index_value],
                output_dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                output_dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom gather: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "scatter" {
            let [base, indices, updates] = shape.in_shapes.as_slice() else {
                return Err("custom scatter requires base, indices, and update shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let [rows, features] = base.dims.as_slice() else {
                return Err("custom scatter base must be rank-2".into());
            };
            let [update_rows] = indices.dims.as_slice() else {
                return Err("custom scatter indices must be rank-1".into());
            };
            if *rows == 0
                || *features == 0
                || *update_rows == 0
                || output.dims != base.dims
                || updates.dims != vec![*update_rows, *features]
            {
                return Err("custom scatter shape contract is inconsistent".into());
            }
            let base_dims = base.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let index_dims = indices
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let update_dims = updates
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let base_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "base".into(),
                },
                vec![],
                base_dims.clone(),
            );
            let index_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "indices".into(),
                },
                vec![],
                index_dims,
            );
            let update_value = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "updates".into(),
                },
                vec![],
                update_dims,
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Scatter {
                    rows: *rows,
                    updates: *update_rows,
                    features: *features,
                },
                vec![base_value, index_value, update_value],
                base_dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                base_dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom scatter: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "layer_norm" || operation == "layernorm" {
            let [input, weight, bias] = shape.in_shapes.as_slice() else {
                return Err("custom layer_norm requires input, weight, and bias shapes".into());
            };
            let output = shape.out_shapes.first().unwrap();
            let [rows, features] = output.dims.as_slice() else {
                return Err("custom layer_norm requires rank-2 output".into());
            };
            if input.dims != output.dims
                || weight.dims != vec![*features]
                || bias.dims != vec![*features]
            {
                return Err("custom layer_norm shape contract is inconsistent".into());
            }
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let w = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "weight".into(),
                },
                vec![],
                vec![*features as u64],
            );
            let b = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "bias".into(),
                },
                vec![],
                vec![*features as u64],
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::LayerNorm {
                    rows: *rows,
                    features: *features,
                    epsilon: 1e-5,
                },
                vec![x, w, b],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom layer_norm: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        if operation == "cast" || operation.starts_with("cast_") {
            let (from, to) = parse_cast_operation(&operation)?;
            let [input] = shape.in_shapes.as_slice() else {
                return Err("custom cast requires one input shape".into());
            };
            let output = shape.out_shapes.first().unwrap();
            if input.dims != output.dims {
                return Err("custom cast must preserve shape".into());
            }
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let dims = output
                .dims
                .iter()
                .map(|dim| *dim as u64)
                .collect::<Vec<_>>();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let value = graph.add(
                prism_spatial_ir::UOpKind::Cast {
                    from: from.into(),
                    to: to.into(),
                },
                vec![x],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![value],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower SpatialGraph cast: {error}"))?;
            let artifacts = compile_and_validate_uop_capture(&capture)?;
            return Ok((capture, artifacts));
        }
        if matches!(operation.as_str(), "sigmoid" | "silu" | "softplus") {
            let [input] = shape.in_shapes.as_slice() else {
                return Err("custom sigmoid requires one input shape".into());
            };
            let output = shape
                .out_shapes
                .first()
                .ok_or_else(|| "custom sigmoid has no output shape".to_string())?;
            if input.dims != output.dims {
                return Err("custom sigmoid input/output shapes must match".into());
            }
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let dims = input.dims.iter().map(|dim| *dim as u64).collect::<Vec<_>>();
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let neg = graph.add(prism_spatial_ir::UOpKind::Neg, vec![x], dims.clone());
            let exp = graph.add(prism_spatial_ir::UOpKind::Exp, vec![neg], dims.clone());
            let one = graph.add(
                prism_spatial_ir::UOpKind::Const { value: 1.0 },
                vec![],
                vec![1],
            );
            let denom = graph.add(prism_spatial_ir::UOpKind::Add, vec![one, exp], dims.clone());
            let sigmoid = graph.add(
                prism_spatial_ir::UOpKind::Div,
                vec![one, denom],
                dims.clone(),
            );
            let result = if operation == "silu" {
                graph.add(
                    prism_spatial_ir::UOpKind::Mul,
                    vec![x, sigmoid],
                    dims.clone(),
                )
            } else if operation == "softplus" {
                let exp_x = graph.add(prism_spatial_ir::UOpKind::Exp, vec![x], dims.clone());
                let one_plus_exp = graph.add(
                    prism_spatial_ir::UOpKind::Add,
                    vec![one, exp_x],
                    dims.clone(),
                );
                graph.add(
                    prism_spatial_ir::UOpKind::Log,
                    vec![one_plus_exp],
                    dims.clone(),
                )
            } else {
                sigmoid
            };
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![result],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower custom sigmoid: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        let metadata = prism_spatial_ir::graph::NodeMeta {
            elementwise_op: Some(operation),
            ..Default::default()
        };
        return compile_spatial_elementwise(shape, Some(&metadata), target);
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Reshape,
        shape,
        ..
    } = node
    {
        let [input] = shape.in_shapes.as_slice() else {
            return Err("SpatialGraph Reshape requires exactly one input shape".into());
        };
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Reshape has no output shape".to_string())?;
        let input_elements = input
            .dims
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| "SpatialGraph Reshape input element count overflows".to_string())?;
        let output_elements = output
            .dims
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| "SpatialGraph Reshape output element count overflows".to_string())?;
        if input_elements != output_elements {
            return Err("SpatialGraph Reshape must preserve element count".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let input_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "input".into(),
            },
            vec![],
            input.dims.iter().map(|dim| *dim as u64).collect(),
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![input_id],
            output.dims.iter().map(|dim| *dim as u64).collect(),
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph Reshape: {error}"))?;
        return Ok((capture, Vec::new()));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
        shape,
        ..
    } = node
    {
        return compile_spatial_elementwise(shape, metadata, target);
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Attention,
        shape,
        ..
    } = node
    {
        return compile_spatial_attention(shape, target);
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::RoPE,
        shape,
        ..
    } = node
    {
        let [input, cos, sin] = shape.in_shapes.as_slice() else {
            return Err("SpatialGraph RoPE requires input, cosine, and sine shapes".into());
        };
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph RoPE has no output shape".to_string())?;
        let [rows, features] = output.dims.as_slice() else {
            return Err("SpatialGraph RoPE requires rank-2 output".into());
        };
        if *features == 0
            || *features % 2 != 0
            || input.dims != vec![*rows, *features]
            || cos.dims != vec![*rows, *features / 2]
            || sin.dims != cos.dims
        {
            return Err("SpatialGraph RoPE shape contract is inconsistent".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let dims = vec![*rows as u64, *features as u64];
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            dims.clone(),
        );
        let cos_id = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "cos".into() },
            vec![],
            vec![*rows as u64, (*features / 2) as u64],
        );
        let sin_id = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "sin".into() },
            vec![],
            vec![*rows as u64, (*features / 2) as u64],
        );
        let rope = graph.add(
            prism_spatial_ir::UOpKind::Rope {
                rows: *rows,
                features: *features,
            },
            vec![x, cos_id, sin_id],
            dims.clone(),
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![rope],
            dims,
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph RoPE: {error}"))?;
        return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Gather,
        shape,
        ..
    } = node
    {
        let [weight, indices] = shape.in_shapes.as_slice() else {
            return Err("SpatialGraph Gather requires weight and indices".into());
        };
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Gather has no output shape".to_string())?;
        let [vocab, features] = weight.dims.as_slice() else {
            return Err("SpatialGraph Gather weight must be rank-2".into());
        };
        let [rows] = indices.dims.as_slice() else {
            return Err("SpatialGraph Gather indices must be rank-1".into());
        };
        if *vocab == 0 || *features == 0 || output.dims != vec![*rows, *features] {
            return Err("SpatialGraph Gather shape contract is inconsistent".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let weight_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![*vocab as u64, *features as u64],
        );
        let indices_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![*rows as u64],
        );
        let gather = graph.add(
            prism_spatial_ir::UOpKind::Gather {
                rows: *rows,
                vocab: *vocab,
                features: *features,
            },
            vec![weight_id, indices_id],
            vec![*rows as u64, *features as u64],
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![gather],
            vec![*rows as u64, *features as u64],
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph Gather: {error}"))?;
        return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::SSM,
        shape,
        ..
    } = node
    {
        let [input, decay, input_gain, output_gain] = shape.in_shapes.as_slice() else {
            return Err(
                "SpatialGraph SSM requires input, decay, input gain, and output gain".into(),
            );
        };
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph SSM has no output shape".to_string())?;
        let [rows, features] = input.dims.as_slice() else {
            return Err("SpatialGraph SSM input must be rank-2".into());
        };
        if output.dims != vec![*rows, *features]
            || decay.dims != vec![*features]
            || input_gain.dims != vec![*features]
            || output_gain.dims != vec![*features]
        {
            return Err("SpatialGraph SSM shape contract is inconsistent".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let dims = vec![*rows as u64, *features as u64];
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "input".into(),
            },
            vec![],
            dims.clone(),
        );
        let decay_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "decay".into(),
            },
            vec![],
            vec![*features as u64],
        );
        let input_gain_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "input_gain".into(),
            },
            vec![],
            vec![*features as u64],
        );
        let output_gain_id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "output_gain".into(),
            },
            vec![],
            vec![*features as u64],
        );
        let scan = graph.add(
            prism_spatial_ir::UOpKind::Ssm {
                rows: *rows,
                features: *features,
            },
            vec![x, decay_id, input_gain_id, output_gain_id],
            dims.clone(),
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![scan],
            dims,
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph SSM: {error}"))?;
        return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Normalization,
        shape,
        ..
    } = node
    {
        if matches!(
            metadata
                .and_then(|meta| meta.normalization_op.as_deref())
                .or_else(|| metadata.and_then(|meta| meta.elementwise_op.as_deref())),
            Some("layer_norm" | "layernorm")
        ) {
            let output = shape
                .out_shapes
                .first()
                .ok_or_else(|| "SpatialGraph LayerNorm has no output shape".to_string())?;
            let [rows, features] = output.dims.as_slice() else {
                return Err("SpatialGraph LayerNorm requires rank-2 output".into());
            };
            let [input, weight, bias] = shape.in_shapes.as_slice() else {
                return Err("SpatialGraph LayerNorm requires input, weight, and bias".into());
            };
            if input.dims != vec![*rows, *features]
                || weight.dims != vec![*features]
                || bias.dims != vec![*features]
            {
                return Err("SpatialGraph LayerNorm shape contract is inconsistent".into());
            }
            let mut graph = prism_spatial_ir::TinyGraph::default();
            let dims = vec![*rows as u64, *features as u64];
            let x = graph.add(
                prism_spatial_ir::UOpKind::Input { name: "x".into() },
                vec![],
                dims.clone(),
            );
            let w = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "weight".into(),
                },
                vec![],
                vec![*features as u64],
            );
            let b = graph.add(
                prism_spatial_ir::UOpKind::Input {
                    name: "bias".into(),
                },
                vec![],
                vec![*features as u64],
            );
            let norm = graph.add(
                prism_spatial_ir::UOpKind::LayerNorm {
                    rows: *rows,
                    features: *features,
                    epsilon: 1e-5,
                },
                vec![x, w, b],
                dims.clone(),
            );
            graph.add(
                prism_spatial_ir::UOpKind::Output { name: "out".into() },
                vec![norm],
                dims,
            );
            let capture = graph
                .lower(target)
                .map_err(|error| format!("lower SpatialGraph LayerNorm: {error}"))?;
            return Ok((capture.clone(), compile_and_validate_uop_capture(&capture)?));
        }
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Normalization has no output shape".to_string())?;
        let [rows, features] = output.dims.as_slice() else {
            return Err("SpatialGraph RMSNorm requires rank-2 output".into());
        };
        let [input, weight] = shape.in_shapes.as_slice() else {
            return Err("SpatialGraph RMSNorm requires input and weight".into());
        };
        if input.dims != vec![*rows, *features] || weight.dims != vec![*features] {
            return Err("SpatialGraph RMSNorm shape contract is inconsistent".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            vec![*rows as u64, *features as u64],
        );
        let w = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![*features as u64],
        );
        let norm = graph.add(
            prism_spatial_ir::UOpKind::RmsNorm {
                rows: *rows,
                features: *features,
                epsilon: 1e-5,
            },
            vec![x, w],
            vec![*rows as u64, *features as u64],
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![norm],
            vec![*rows as u64, *features as u64],
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph RMSNorm: {error}"))?;
        let artifacts = compile_and_validate_uop_capture(&capture)?;
        return Ok((capture, artifacts));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Softmax,
        shape,
        ..
    } = node
    {
        let input = shape
            .in_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Softmax has no input shape".to_string())?;
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Softmax has no output shape".to_string())?;
        if input.dims != output.dims || input.dims.is_empty() {
            return Err("SpatialGraph Softmax shape contract is inconsistent".into());
        }
        let dims: Vec<u64> = input.dims.iter().map(|dim| *dim as u64).collect();
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            dims.clone(),
        );
        let softmax = graph.add(
            prism_spatial_ir::UOpKind::SoftmaxAxis {
                axis: dims.len() - 1,
            },
            vec![x],
            dims.clone(),
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![softmax],
            dims,
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph Softmax: {error}"))?;
        let artifacts = compile_and_validate_uop_capture(&capture)?;
        return Ok((capture, artifacts));
    }
    if let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::Convolution,
        shape,
        ..
    } = node
    {
        let [input, weight, bias] = shape.in_shapes.as_slice() else {
            return Err("SpatialGraph Conv2D requires input, weight, and bias".into());
        };
        let output = shape
            .out_shapes
            .first()
            .ok_or_else(|| "SpatialGraph Conv2D has no output shape".to_string())?;
        let stride = metadata
            .and_then(|meta| meta.convolution_stride)
            .unwrap_or(1);
        let padding = metadata
            .and_then(|meta| meta.convolution_padding)
            .unwrap_or(0);
        if stride == 0 {
            return Err("SpatialGraph Conv2D stride must be nonzero".into());
        }
        let [batch, in_channels, height, width] = input.dims.as_slice() else {
            return Err("SpatialGraph Conv2D input must be NCHW".into());
        };
        let [out_channels, weight_in, kernel_h, kernel_w] = weight.dims.as_slice() else {
            return Err("SpatialGraph Conv2D weight must be OIHW".into());
        };
        if *height + 2 * padding < *kernel_h || *width + 2 * padding < *kernel_w {
            return Err("SpatialGraph Conv2D kernel does not fit padded input".into());
        }
        let out_h = (*height + 2 * padding - *kernel_h) / stride + 1;
        let out_w = (*width + 2 * padding - *kernel_w) / stride + 1;
        if *weight_in != *in_channels
            || bias.dims != vec![*out_channels]
            || output.dims != vec![*batch, *out_channels, out_h, out_w]
        {
            return Err("SpatialGraph Conv2D shape contract is inconsistent".into());
        }
        let mut graph = prism_spatial_ir::TinyGraph::default();
        let x = graph.add(
            prism_spatial_ir::UOpKind::Input { name: "x".into() },
            vec![],
            input.dims.iter().map(|dim| *dim as u64).collect(),
        );
        let w = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            weight.dims.iter().map(|dim| *dim as u64).collect(),
        );
        let b = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            bias.dims.iter().map(|dim| *dim as u64).collect(),
        );
        let conv = graph.add(
            prism_spatial_ir::UOpKind::Conv2d {
                batch: *batch,
                in_channels: *in_channels,
                height: *height,
                width: *width,
                out_channels: *out_channels,
                kernel_h: *kernel_h,
                kernel_w: *kernel_w,
                stride,
                padding,
            },
            vec![x, w, b],
            output.dims.iter().map(|dim| *dim as u64).collect(),
        );
        graph.add(
            prism_spatial_ir::UOpKind::Output { name: "out".into() },
            vec![conv],
            output.dims.iter().map(|dim| *dim as u64).collect(),
        );
        let capture = graph
            .lower(target)
            .map_err(|error| format!("lower SpatialGraph Conv2D: {error}"))?;
        let artifacts = compile_and_validate_uop_capture(&capture)?;
        return Ok((capture, artifacts));
    }
    let prism_spatial_ir::SpatialNode::Compute {
        kind: prism_spatial_ir::graph::ComputeKind::MatMul,
        shape,
        ..
    } = node
    else {
        return Err("SpatialGraph node lacks a lossless UOp lowering contract".into());
    };
    let a = shape.in_shapes.first().map(|shape| shape.dims.as_slice());
    let b = shape.in_shapes.get(1).map(|shape| shape.dims.as_slice());
    let c = shape.out_shapes.first().map(|shape| shape.dims.as_slice());
    let (Some([m, k]), Some([k_rhs, n]), Some([m_out, n_out])) = (a, b, c) else {
        return Err("SpatialGraph MatMul must have rank-2 input/output shapes".into());
    };
    if k != k_rhs || m != m_out || n != n_out {
        return Err("SpatialGraph MatMul shape contract is inconsistent".into());
    }
    compile_spatial_matmul(*m, *k, *n, target)
}

fn compile_spatial_attention(
    shape: &prism_spatial_ir::graph::ShapeContract,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    let inputs = shape.in_shapes.as_slice();
    let output = shape
        .out_shapes
        .first()
        .ok_or_else(|| "SpatialGraph Attention has no output shape".to_string())?;
    let [q, k, v] = inputs else {
        return Err("SpatialGraph Attention requires three inputs".into());
    };
    if k.dims != q.dims || v.dims != q.dims || output.dims != q.dims {
        return Err("SpatialGraph Attention shape contract is inconsistent".into());
    }
    let mut graph = prism_spatial_ir::TinyGraph::default();
    let dims: Vec<u64> = q.dims.iter().map(|dim| *dim as u64).collect();
    let kind = match q.dims.as_slice() {
        [seq, head] if *head > 0 => prism_spatial_ir::UOpKind::Attention {
            seq: *seq,
            head: *head,
            scale: 1.0 / (*head as f32).sqrt(),
        },
        [batch, seq, head] if *head > 0 => prism_spatial_ir::UOpKind::AttentionBatched {
            batch: *batch,
            seq: *seq,
            head: *head,
            scale: 1.0 / (*head as f32).sqrt(),
        },
        _ => return Err("SpatialGraph Attention requires rank-2 or rank-3 Q shape".into()),
    };
    let q_id = graph.add(
        prism_spatial_ir::UOpKind::Input { name: "q".into() },
        vec![],
        dims.clone(),
    );
    let k_id = graph.add(
        prism_spatial_ir::UOpKind::Input { name: "k".into() },
        vec![],
        dims.clone(),
    );
    let v_id = graph.add(
        prism_spatial_ir::UOpKind::Input { name: "v".into() },
        vec![],
        dims.clone(),
    );
    let attention = graph.add(kind, vec![q_id, k_id, v_id], dims.clone());
    graph.add(
        prism_spatial_ir::UOpKind::Output { name: "out".into() },
        vec![attention],
        dims,
    );
    let capture = graph
        .lower(target)
        .map_err(|error| format!("lower SpatialGraph Attention: {error}"))?;
    let artifacts = compile_and_validate_uop_capture(&capture)?;
    Ok((capture, artifacts))
}

/// Lower a linear SpatialGraph dataflow into one executable UOp capture.
///
/// Memory nodes become named inputs, compute-to-compute edges reuse the
/// producer value, and graph exits become named outputs. The adapter rejects
/// barriers, repeated regions, and unsupported compute kinds rather than
/// silently changing graph semantics.
pub fn compile_spatial_graph(
    spatial: &prism_spatial_ir::SpatialGraph,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    let graph = build_tiny_graph_from_spatial(spatial)?;
    let capture = graph
        .lower(target)
        .map_err(|error| format!("lower SpatialGraph: {error}"))?;
    let artifacts = compile_and_validate_uop_capture(&capture)?;
    Ok((capture, artifacts))
}

/// Lower one SpatialGraph into the compact graph so callers can materialize
/// multiple fusion strategies from the exact same semantic graph.
pub fn compile_spatial_graph_strategies(
    spatial: &prism_spatial_ir::SpatialGraph,
    target: LoweringTarget,
    strategies: &[FusionStrategy],
) -> Result<Vec<(FusionStrategy, CapturePlan, Vec<KernelArtifact>)>, String> {
    let graph = build_tiny_graph_from_spatial(spatial)?;
    let inputs = graph
        .ops
        .iter()
        .filter_map(|op| {
            let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
                return None;
            };
            let elements = op.shape.iter().try_fold(1usize, |count, dimension| {
                count.checked_mul(*dimension as usize)
            })?;
            Some((name.clone(), vec![0.0; elements]))
        })
        .collect::<BTreeMap<_, _>>();
    compile_and_validate_uop_graph_strategies(&graph, target, strategies, &inputs)
}

fn build_tiny_graph_from_spatial(
    spatial: &prism_spatial_ir::SpatialGraph,
) -> Result<TinyGraph, String> {
    let mut graph = prism_spatial_ir::TinyGraph::default();
    let mut values = std::collections::HashMap::new();
    let mut input_values = std::collections::HashMap::new();
    let mut compute_nodes = Vec::new();
    for node_id in spatial
        .topological_sort()
        .ok_or_else(|| "SpatialGraph contains a cycle".to_string())?
    {
        let node = spatial
            .get_node(node_id)
            .ok_or_else(|| format!("SpatialGraph node {node_id} is missing"))?;
        let prism_spatial_ir::SpatialNode::Compute { kind, shape, .. } = node else {
            continue;
        };
        let mut incoming = spatial.incoming_edges(node_id);
        incoming.sort_by_key(|edge| edge.sink_input_idx);
        let mut sources = Vec::with_capacity(incoming.len());
        for edge in incoming {
            let source_node = spatial
                .get_node(edge.source)
                .ok_or_else(|| format!("edge source {} is missing", edge.source))?;
            let value = match source_node {
                prism_spatial_ir::SpatialNode::Memory { region, .. } => {
                    if let Some(value) = input_values.get(&edge.source) {
                        *value
                    } else {
                        let value = graph.add(
                            prism_spatial_ir::UOpKind::Input {
                                name: format!("memory_{}", edge.source.0),
                            },
                            vec![],
                            region.shape.dims.iter().map(|dim| *dim as u64).collect(),
                        );
                        input_values.insert(edge.source, value);
                        value
                    }
                }
                prism_spatial_ir::SpatialNode::Compute { .. } => *values
                    .get(&edge.source)
                    .ok_or_else(|| format!("compute source {} was not lowered", edge.source))?,
                _ => return Err(format!("unsupported source node {}", edge.source)),
            };
            sources.push(value);
        }
        let output_shape = shape
            .out_shapes
            .first()
            .ok_or_else(|| format!("compute node {node_id} has no output shape"))?;
        let output_dims: Vec<u64> = output_shape.dims.iter().map(|dim| *dim as u64).collect();
        // The model graph represents RoPE tables as an implicit runtime
        // resource for the common single-input form. Materialize those
        // resources as explicit UOp inputs so the lowering contract remains
        // three-operand (input, cosine, sine) without rejecting valid model
        // graphs that omit table edges.
        if matches!(kind, prism_spatial_ir::graph::ComputeKind::RoPE) && sources.len() == 1 {
            let (rows, features) = match output_shape.dims.as_slice() {
                [features] => (1, *features),
                [rows, features] => (*rows, *features),
                _ => (0, 0),
            };
            if rows > 0 && features > 0 && features % 2 == 0 {
                let table_shape = vec![rows as u64, (features / 2) as u64];
                let cos = graph.add(
                    prism_spatial_ir::UOpKind::Input {
                        name: format!("rope_cos_{}", node_id.0),
                    },
                    vec![],
                    table_shape.clone(),
                );
                let sin = graph.add(
                    prism_spatial_ir::UOpKind::Input {
                        name: format!("rope_sin_{}", node_id.0),
                    },
                    vec![],
                    table_shape,
                );
                sources.extend([cos, sin]);
            }
        }
        // The abstract model graph uses Gather for token embedding while
        // carrying the embedding lookup tables outside the spatial edge
        // contract. When only the already-gathered activation is present,
        // preserve it as the graph value; a concrete source adapter supplies
        // the indexed gather when weights and indices are available.
        if matches!(kind, prism_spatial_ir::graph::ComputeKind::Gather)
            && sources.len() == 1
            && graph.ops[sources[0].0 as usize].shape == output_dims
        {
            values.insert(node_id, sources[0]);
            compute_nodes.push(node_id);
            continue;
        }
        if matches!(kind, prism_spatial_ir::graph::ComputeKind::Reshape) {
            let input_shape = shape
                .in_shapes
                .first()
                .ok_or_else(|| format!("Reshape node {node_id} has no input shape"))?;
            let input_elements = input_shape
                .dims
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| format!("Reshape node {node_id} element count overflows"))?;
            let output_elements = output_shape
                .dims
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or_else(|| format!("Reshape node {node_id} element count overflows"))?;
            if input_elements != output_elements || sources.len() != 1 {
                return Err(format!(
                    "Reshape node {node_id} must preserve element count and have one input"
                ));
            }
            values.insert(node_id, sources[0]);
            compute_nodes.push(node_id);
            continue;
        }
        if let prism_spatial_ir::graph::ComputeKind::Custom(operation) = kind {
            let operation = operation.to_ascii_lowercase();
            if validate_and_classify_custom_operation(&operation, shape)?
                == CustomOperationClass::Candidate
            {
                return Err(format!(
                    "custom operation '{operation}' is a candidate, not validated"
                ));
            }
            if operation == "where" {
                let [condition, when_true, when_false] = sources.as_slice() else {
                    return Err(format!("custom where node {node_id} needs three inputs"));
                };
                let broadcast = broadcast_shape_contract(
                    &shape
                        .in_shapes
                        .iter()
                        .map(|tensor| tensor.dims.as_slice())
                        .collect::<Vec<_>>(),
                )
                .ok_or_else(|| {
                    format!("custom where node {node_id} shapes are not broadcast-compatible")
                })?;
                if broadcast != output_shape.dims {
                    return Err(format!(
                        "custom where node {node_id} output shape does not match broadcast shape"
                    ));
                }
                let value = graph.add(
                    prism_spatial_ir::UOpKind::Where,
                    vec![*condition, *when_true, *when_false],
                    output_dims,
                );
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if operation == "clamp" {
                let [input, lower, upper] = sources.as_slice() else {
                    return Err(format!("custom clamp node {node_id} needs three inputs"));
                };
                if shape
                    .in_shapes
                    .iter()
                    .any(|tensor| tensor.dims != output_shape.dims)
                {
                    return Err(format!("custom clamp node {node_id} shapes must match"));
                }
                let lower_value = graph.add(
                    prism_spatial_ir::UOpKind::Maximum,
                    vec![*input, *lower],
                    output_dims.clone(),
                );
                let value = graph.add(
                    prism_spatial_ir::UOpKind::Minimum,
                    vec![lower_value, *upper],
                    output_dims,
                );
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if operation == "softmax" || operation == "log_softmax" {
                let [source] = sources.as_slice() else {
                    return Err(format!("custom {operation} node {node_id} needs one input"));
                };
                let input_shape = shape.in_shapes.first().unwrap();
                if input_shape.dims != output_shape.dims || input_shape.dims.is_empty() {
                    return Err(format!(
                        "custom {operation} node {node_id} shapes must match"
                    ));
                }
                let softmax = graph.add(
                    prism_spatial_ir::UOpKind::SoftmaxAxis {
                        axis: input_shape.dims.len() - 1,
                    },
                    vec![*source],
                    output_dims.clone(),
                );
                let value = if operation == "log_softmax" {
                    graph.add(prism_spatial_ir::UOpKind::Log, vec![softmax], output_dims)
                } else {
                    softmax
                };
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if operation == "scatter" {
                let [base, indices, updates] = sources.as_slice() else {
                    return Err(format!("custom scatter node {node_id} needs three inputs"));
                };
                let [base_shape, index_shape, update_shape] = shape.in_shapes.as_slice() else {
                    return Err(format!(
                        "custom scatter node {node_id} has invalid input shapes"
                    ));
                };
                let [rows, features] = output_shape.dims.as_slice() else {
                    return Err(format!(
                        "custom scatter node {node_id} output must be rank-2"
                    ));
                };
                let [update_rows] = index_shape.dims.as_slice() else {
                    return Err(format!(
                        "custom scatter node {node_id} indices must be rank-1"
                    ));
                };
                if base_shape.dims != output_shape.dims
                    || update_shape.dims != vec![*update_rows, *features]
                    || *rows == 0
                    || *features == 0
                    || *update_rows == 0
                {
                    return Err(format!(
                        "custom scatter node {node_id} shape contract is inconsistent"
                    ));
                }
                let value = graph.add(
                    prism_spatial_ir::UOpKind::Scatter {
                        rows: *rows,
                        updates: *update_rows,
                        features: *features,
                    },
                    vec![*base, *indices, *updates],
                    output_dims,
                );
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if operation == "cast" || operation.starts_with("cast_") {
                let (from, to) = parse_cast_operation(&operation)?;
                let [input] = shape.in_shapes.as_slice() else {
                    return Err(format!("custom cast node {node_id} needs one input"));
                };
                if input.dims != output_shape.dims {
                    return Err(format!("custom cast node {node_id} must preserve shape"));
                }
                let value = graph.add(
                    prism_spatial_ir::UOpKind::Cast {
                        from: from.into(),
                        to: to.into(),
                    },
                    vec![sources[0]],
                    output_dims,
                );
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if operation == "pow" {
                let [input] = shape.in_shapes.as_slice() else {
                    return Err(format!("custom pow node {node_id} needs one input"));
                };
                if input.dims != output_shape.dims || sources.len() != 1 {
                    return Err(format!("custom pow node {node_id} must preserve shape"));
                }
                let exponent = pow_exponent_from_metadata(spatial.get_annotations(node_id))?;
                let value = graph.add(
                    prism_spatial_ir::UOpKind::Pow { exponent },
                    vec![sources[0]],
                    output_dims,
                );
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
            if matches!(operation.as_str(), "sigmoid" | "silu" | "softplus") {
                let [source] = sources.as_slice() else {
                    return Err(format!("custom sigmoid node {node_id} needs one input"));
                };
                let input_shape = shape.in_shapes.first().unwrap();
                if input_shape.dims != output_shape.dims {
                    return Err(format!("custom sigmoid node {node_id} shapes must match"));
                }
                let neg = graph.add(
                    prism_spatial_ir::UOpKind::Neg,
                    vec![*source],
                    output_dims.clone(),
                );
                let exp = graph.add(
                    prism_spatial_ir::UOpKind::Exp,
                    vec![neg],
                    output_dims.clone(),
                );
                let one = graph.add(
                    prism_spatial_ir::UOpKind::Const { value: 1.0 },
                    vec![],
                    vec![1],
                );
                let denom = graph.add(
                    prism_spatial_ir::UOpKind::Add,
                    vec![one, exp],
                    output_dims.clone(),
                );
                let sigmoid = graph.add(
                    prism_spatial_ir::UOpKind::Div,
                    vec![one, denom],
                    output_dims.clone(),
                );
                let value = if operation == "silu" {
                    graph.add(
                        prism_spatial_ir::UOpKind::Mul,
                        vec![*source, sigmoid],
                        output_dims,
                    )
                } else if operation == "softplus" {
                    let exp_x = graph.add(
                        prism_spatial_ir::UOpKind::Exp,
                        vec![*source],
                        output_dims.clone(),
                    );
                    let one_plus_exp = graph.add(
                        prism_spatial_ir::UOpKind::Add,
                        vec![one, exp_x],
                        output_dims.clone(),
                    );
                    graph.add(
                        prism_spatial_ir::UOpKind::Log,
                        vec![one_plus_exp],
                        output_dims,
                    )
                } else {
                    sigmoid
                };
                values.insert(node_id, value);
                compute_nodes.push(node_id);
                continue;
            }
        }
        let uop_kind = match kind {
            prism_spatial_ir::graph::ComputeKind::MatMul => {
                let [a, b] = sources.as_slice() else {
                    return Err(format!("MatMul node {node_id} needs two inputs"));
                };
                let _ = (a, b);
                let [m, n] = output_shape.dims.as_slice() else {
                    return Err(format!("MatMul node {node_id} needs rank-2 output"));
                };
                let input_shape = shape
                    .in_shapes
                    .first()
                    .ok_or_else(|| format!("MatMul node {node_id} has no lhs shape"))?;
                let [_, k] = input_shape.dims.as_slice() else {
                    return Err(format!("MatMul node {node_id} needs rank-2 lhs"));
                };
                prism_spatial_ir::UOpKind::MatMul {
                    m: *m,
                    k: *k,
                    n: *n,
                }
            }
            prism_spatial_ir::graph::ComputeKind::Elementwise => {
                let operation = spatial
                    .get_annotations(node_id)
                    .and_then(|meta| meta.elementwise_op.as_deref())
                    .ok_or_else(|| {
                        format!("Elementwise node {node_id} has no operation annotation")
                    })?
                    .to_ascii_lowercase();
                match operation.as_str() {
                    "add" if sources.len() == 2 => prism_spatial_ir::UOpKind::Add,
                    "mul" if sources.len() == 2 => prism_spatial_ir::UOpKind::Mul,
                    "sub" if sources.len() == 2 => prism_spatial_ir::UOpKind::Sub,
                    "div" if sources.len() == 2 => prism_spatial_ir::UOpKind::Div,
                    "maximum" | "max" if sources.len() == 2 => prism_spatial_ir::UOpKind::Maximum,
                    "minimum" | "min" if sources.len() == 2 => prism_spatial_ir::UOpKind::Minimum,
                    "relu" if sources.len() == 1 => prism_spatial_ir::UOpKind::Relu,
                    "neg" if sources.len() == 1 => prism_spatial_ir::UOpKind::Neg,
                    "exp" if sources.len() == 1 => prism_spatial_ir::UOpKind::Exp,
                    "sqrt" if sources.len() == 1 => prism_spatial_ir::UOpKind::Sqrt,
                    "abs" if sources.len() == 1 => prism_spatial_ir::UOpKind::Abs,
                    "log" if sources.len() == 1 => prism_spatial_ir::UOpKind::Log,
                    "tanh" if sources.len() == 1 => prism_spatial_ir::UOpKind::Tanh,
                    "sin" if sources.len() == 1 => prism_spatial_ir::UOpKind::Sin,
                    "cos" if sources.len() == 1 => prism_spatial_ir::UOpKind::Cos,
                    "gelu" if sources.len() == 1 => prism_spatial_ir::UOpKind::Gelu,
                    "pow" if sources.len() == 1 => {
                        let exponent = spatial
                            .get_annotations(node_id)
                            .and_then(|meta| meta.pow_exponent)
                            .ok_or_else(|| {
                                format!("Elementwise node {node_id} has no pow_exponent annotation")
                            })?;
                        if !exponent.is_finite() {
                            return Err(format!(
                                "Elementwise node {node_id} pow exponent must be finite"
                            ));
                        }
                        prism_spatial_ir::UOpKind::Pow { exponent }
                    }
                    operation => {
                        return Err(format!("unsupported Elementwise operation '{operation}'"))
                    }
                }
            }
            prism_spatial_ir::graph::ComputeKind::Convolution => {
                let [input_shape, weight_shape, bias_shape] = shape.in_shapes.as_slice() else {
                    return Err(format!(
                        "Conv2D node {node_id} needs input, weight, and bias"
                    ));
                };
                let [batch, in_channels, height, width] = input_shape.dims.as_slice() else {
                    return Err(format!("Conv2D node {node_id} input must be NCHW"));
                };
                let [out_channels, weight_in, kernel_h, kernel_w] = weight_shape.dims.as_slice()
                else {
                    return Err(format!("Conv2D node {node_id} weight must be OIHW"));
                };
                let stride = spatial
                    .get_annotations(node_id)
                    .and_then(|meta| meta.convolution_stride)
                    .unwrap_or(1);
                let padding = spatial
                    .get_annotations(node_id)
                    .and_then(|meta| meta.convolution_padding)
                    .unwrap_or(0);
                if stride == 0 {
                    return Err(format!("Conv2D node {node_id} stride must be nonzero"));
                }
                if *height + 2 * padding < *kernel_h || *width + 2 * padding < *kernel_w {
                    return Err(format!(
                        "Conv2D node {node_id} kernel does not fit padded input"
                    ));
                }
                if *weight_in != *in_channels || bias_shape.dims != vec![*out_channels] {
                    return Err(format!(
                        "Conv2D node {node_id} input contracts are inconsistent"
                    ));
                }
                let expected = vec![
                    *batch,
                    *out_channels,
                    (*height + 2 * padding - *kernel_h) / stride + 1,
                    (*width + 2 * padding - *kernel_w) / stride + 1,
                ];
                if output_shape.dims != expected {
                    return Err(format!(
                        "Conv2D node {node_id} output contract is inconsistent"
                    ));
                }
                prism_spatial_ir::UOpKind::Conv2d {
                    batch: *batch,
                    in_channels: *in_channels,
                    height: *height,
                    width: *width,
                    out_channels: *out_channels,
                    kernel_h: *kernel_h,
                    kernel_w: *kernel_w,
                    stride,
                    padding,
                }
            }
            prism_spatial_ir::graph::ComputeKind::Normalization => {
                let operation = spatial
                    .get_annotations(node_id)
                    .and_then(|meta| {
                        meta.normalization_op
                            .as_deref()
                            .or(meta.elementwise_op.as_deref())
                    })
                    .unwrap_or("rms_norm")
                    .to_ascii_lowercase();
                if operation == "layer_norm" || operation == "layernorm" {
                    let [rows, features] = output_shape.dims.as_slice() else {
                        return Err(format!("LayerNorm node {node_id} needs rank-2 output"));
                    };
                    if sources.len() != 3 {
                        return Err(format!(
                            "LayerNorm node {node_id} needs input, weight, and bias"
                        ));
                    }
                    prism_spatial_ir::UOpKind::LayerNorm {
                        rows: *rows,
                        features: *features,
                        epsilon: 1e-5,
                    }
                } else if operation != "rms_norm" && operation != "rmsnorm" {
                    return Err(format!("unsupported Normalization operation '{operation}'"));
                } else {
                    let [rows, features] = output_shape.dims.as_slice() else {
                        return Err(format!("RMSNorm node {node_id} needs rank-2 output"));
                    };
                    if sources.len() != 2 {
                        return Err(format!("RMSNorm node {node_id} needs input and weight"));
                    }
                    prism_spatial_ir::UOpKind::RmsNorm {
                        rows: *rows,
                        features: *features,
                        epsilon: 1e-5,
                    }
                }
            }
            prism_spatial_ir::graph::ComputeKind::Softmax => {
                let input = shape
                    .in_shapes
                    .first()
                    .ok_or_else(|| format!("Softmax node {node_id} has no input shape"))?;
                if input.dims != output_shape.dims || input.dims.is_empty() {
                    return Err(format!("Softmax node {node_id} has inconsistent shapes"));
                }
                prism_spatial_ir::UOpKind::SoftmaxAxis {
                    axis: input.dims.len() - 1,
                }
            }
            prism_spatial_ir::graph::ComputeKind::Attention => {
                let [q, k, v] = sources.as_slice() else {
                    return Err(format!("Attention node {node_id} needs three inputs"));
                };
                let _ = (q, k, v);
                match output_shape.dims.as_slice() {
                    [seq, head] if *head > 0 => prism_spatial_ir::UOpKind::Attention {
                        seq: *seq,
                        head: *head,
                        scale: 1.0 / (*head as f32).sqrt(),
                    },
                    [batch, seq, head] if *head > 0 => {
                        prism_spatial_ir::UOpKind::AttentionBatched {
                            batch: *batch,
                            seq: *seq,
                            head: *head,
                            scale: 1.0 / (*head as f32).sqrt(),
                        }
                    }
                    _ => {
                        return Err(format!(
                            "Attention node {node_id} needs rank-2 or rank-3 output"
                        ))
                    }
                }
            }
            prism_spatial_ir::graph::ComputeKind::RoPE => {
                let input = shape
                    .in_shapes
                    .first()
                    .ok_or_else(|| format!("RoPE node {node_id} needs an input"))?;
                let (rows, features) = match output_shape.dims.as_slice() {
                    [features] => (1, *features),
                    [rows, features] => (*rows, *features),
                    _ => return Err(format!("RoPE node {node_id} needs rank-1 or rank-2 output")),
                };
                let (cos_dims, sin_dims) = if shape.in_shapes.len() == 1 {
                    (vec![rows, features / 2], vec![rows, features / 2])
                } else if let [input, cos, sin] = shape.in_shapes.as_slice() {
                    if input.dims != vec![rows, features] {
                        return Err(format!(
                            "RoPE node {node_id} shape contract is inconsistent"
                        ));
                    }
                    (cos.dims.clone(), sin.dims.clone())
                } else {
                    return Err(format!("RoPE node {node_id} needs three inputs"));
                };
                if features == 0
                    || features % 2 != 0
                    || (input.dims != vec![rows, features] && input.dims != vec![features])
                    || cos_dims != vec![rows, features / 2]
                    || sin_dims != cos_dims
                {
                    return Err(format!(
                        "RoPE node {node_id} shape contract is inconsistent"
                    ));
                }
                prism_spatial_ir::UOpKind::Rope { rows, features }
            }
            prism_spatial_ir::graph::ComputeKind::Gather => {
                let [weight, indices] = shape.in_shapes.as_slice() else {
                    return Err(format!("Gather node {node_id} needs weight and indices"));
                };
                let [vocab, features] = weight.dims.as_slice() else {
                    return Err(format!("Gather node {node_id} weight must be rank-2"));
                };
                let [rows] = indices.dims.as_slice() else {
                    return Err(format!("Gather node {node_id} indices must be rank-1"));
                };
                if *vocab == 0 || *features == 0 || output_shape.dims != vec![*rows, *features] {
                    return Err(format!(
                        "Gather node {node_id} shape contract is inconsistent"
                    ));
                }
                prism_spatial_ir::UOpKind::Gather {
                    rows: *rows,
                    vocab: *vocab,
                    features: *features,
                }
            }
            prism_spatial_ir::graph::ComputeKind::SSM => {
                let [input, decay, input_gain, output_gain] = shape.in_shapes.as_slice() else {
                    return Err(format!("SSM node {node_id} needs four inputs"));
                };
                let [rows, features] = input.dims.as_slice() else {
                    return Err(format!("SSM node {node_id} input must be rank-2"));
                };
                if output_shape.dims != vec![*rows, *features]
                    || decay.dims != vec![*features]
                    || input_gain.dims != vec![*features]
                    || output_gain.dims != vec![*features]
                {
                    return Err(format!("SSM node {node_id} shape contract is inconsistent"));
                }
                prism_spatial_ir::UOpKind::Ssm {
                    rows: *rows,
                    features: *features,
                }
            }
            prism_spatial_ir::graph::ComputeKind::Custom(operation) => {
                let operation = operation.to_ascii_lowercase();
                match validate_and_classify_custom_operation(&operation, shape)? {
                    CustomOperationClass::Validated => {}
                    CustomOperationClass::Candidate => {
                        return Err(format!(
                            "custom operation '{operation}' is a candidate, not validated"
                        ));
                    }
                }
                if operation.starts_with("cast_") {
                    let (from, to) = parse_cast_operation(&operation)?;
                    if sources.len() != 1 || shape.in_shapes[0].dims != output_shape.dims {
                        return Err(format!(
                            "custom cast node {node_id} must preserve shape and have one input"
                        ));
                    }
                    let value = graph.add(
                        prism_spatial_ir::UOpKind::Cast {
                            from: from.into(),
                            to: to.into(),
                        },
                        vec![sources[0]],
                        output_dims,
                    );
                    values.insert(node_id, value);
                    compute_nodes.push(node_id);
                    continue;
                }
                match operation.as_str() {
                    "cast" => {
                        let (from, to) = parse_cast_operation(&operation)?;
                        let value = graph.add(
                            prism_spatial_ir::UOpKind::Cast {
                                from: from.into(),
                                to: to.into(),
                            },
                            vec![sources[0]],
                            output_dims,
                        );
                        values.insert(node_id, value);
                        compute_nodes.push(node_id);
                        continue;
                    }
                    "add" if sources.len() == 2 => prism_spatial_ir::UOpKind::Add,
                    "mul" if sources.len() == 2 => prism_spatial_ir::UOpKind::Mul,
                    "sub" if sources.len() == 2 => prism_spatial_ir::UOpKind::Sub,
                    "div" if sources.len() == 2 => prism_spatial_ir::UOpKind::Div,
                    "maximum" | "max" if sources.len() == 2 => prism_spatial_ir::UOpKind::Maximum,
                    "minimum" | "min" if sources.len() == 2 => prism_spatial_ir::UOpKind::Minimum,
                    "relu" if sources.len() == 1 => prism_spatial_ir::UOpKind::Relu,
                    "neg" if sources.len() == 1 => prism_spatial_ir::UOpKind::Neg,
                    "exp" if sources.len() == 1 => prism_spatial_ir::UOpKind::Exp,
                    "sqrt" if sources.len() == 1 => prism_spatial_ir::UOpKind::Sqrt,
                    "abs" if sources.len() == 1 => prism_spatial_ir::UOpKind::Abs,
                    "log" if sources.len() == 1 => prism_spatial_ir::UOpKind::Log,
                    "tanh" if sources.len() == 1 => prism_spatial_ir::UOpKind::Tanh,
                    "sin" if sources.len() == 1 => prism_spatial_ir::UOpKind::Sin,
                    "cos" if sources.len() == 1 => prism_spatial_ir::UOpKind::Cos,
                    "gelu" if sources.len() == 1 => prism_spatial_ir::UOpKind::Gelu,
                    "pow" if sources.len() == 1 => prism_spatial_ir::UOpKind::Pow {
                        exponent: pow_exponent_from_metadata(spatial.get_annotations(node_id))?,
                    },
                    "transpose" if sources.len() == 1 => {
                        let input = shape
                            .in_shapes
                            .first()
                            .ok_or("custom transpose requires one input")?;
                        let output = shape.out_shapes.first().unwrap();
                        prism_spatial_ir::UOpKind::Transpose {
                            permutation: transpose_permutation(input, output, None)?,
                        }
                    }
                    _ => {
                        return Err(format!(
                            "validated custom operation '{operation}' is missing a lowering"
                        ))
                    }
                }
            }
            other => return Err(format!("unsupported SpatialGraph compute kind {other:?}")),
        };
        let value = graph.add(uop_kind, sources, output_dims);
        values.insert(node_id, value);
        compute_nodes.push(node_id);
    }
    for node_id in compute_nodes {
        if spatial.exit_points().contains(&node_id) {
            let value = values[&node_id];
            let shape = match spatial.get_node(node_id).unwrap() {
                prism_spatial_ir::SpatialNode::Compute { shape, .. } => {
                    shape.out_shapes[0].dims.clone()
                }
                _ => unreachable!(),
            };
            graph.add(
                prism_spatial_ir::UOpKind::Output {
                    name: format!("output_{node_id}"),
                },
                vec![value],
                shape.iter().map(|dim| *dim as u64).collect(),
            );
        }
    }
    Ok(graph)
}

fn compile_spatial_elementwise(
    shape: &prism_spatial_ir::graph::ShapeContract,
    metadata: Option<&prism_spatial_ir::graph::NodeMeta>,
    target: LoweringTarget,
) -> Result<(CapturePlan, Vec<KernelArtifact>), String> {
    let operation = metadata
        .and_then(|meta| meta.elementwise_op.as_deref())
        .ok_or_else(|| {
            "SpatialGraph Elementwise node has no elementwise_op annotation".to_string()
        })?;
    let output_shape = shape
        .out_shapes
        .first()
        .ok_or_else(|| "SpatialGraph Elementwise node has no output shape".to_string())?;
    let element_count = output_shape.dims.iter().product::<usize>();
    if element_count == 0 || shape.in_shapes.is_empty() {
        return Err("SpatialGraph Elementwise node has an empty shape contract".into());
    }
    let mut graph = prism_spatial_ir::TinyGraph::default();
    let mut inputs = Vec::new();
    for (index, input_shape) in shape.in_shapes.iter().enumerate() {
        let id = graph.add(
            prism_spatial_ir::UOpKind::Input {
                name: format!("input_{index}"),
            },
            vec![],
            input_shape.dims.iter().map(|dim| *dim as u64).collect(),
        );
        inputs.push(id);
    }
    let kind = if operation == "pow" {
        if inputs.len() != 1 {
            return Err("SpatialGraph Elementwise pow requires one input".into());
        }
        let exponent = metadata.and_then(|meta| meta.pow_exponent).ok_or_else(|| {
            "SpatialGraph Elementwise pow has no pow_exponent annotation".to_string()
        })?;
        if !exponent.is_finite() {
            return Err("SpatialGraph Elementwise pow exponent must be finite".into());
        }
        prism_spatial_ir::UOpKind::Pow { exponent }
    } else {
        match operation {
            "add" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Add,
            "mul" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Mul,
            "sub" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Sub,
            "div" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Div,
            "maximum" | "max" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Maximum,
            "minimum" | "min" if inputs.len() == 2 => prism_spatial_ir::UOpKind::Minimum,
            "relu" | "neg" | "exp" | "sqrt" | "abs" | "log" | "tanh" | "sin" | "cos" | "gelu"
                if inputs.len() == 1 =>
            {
                match operation {
                    "relu" => prism_spatial_ir::UOpKind::Relu,
                    "neg" => prism_spatial_ir::UOpKind::Neg,
                    "exp" => prism_spatial_ir::UOpKind::Exp,
                    "sqrt" => prism_spatial_ir::UOpKind::Sqrt,
                    "abs" => prism_spatial_ir::UOpKind::Abs,
                    "log" => prism_spatial_ir::UOpKind::Log,
                    "sin" => prism_spatial_ir::UOpKind::Sin,
                    "cos" => prism_spatial_ir::UOpKind::Cos,
                    "gelu" => prism_spatial_ir::UOpKind::Gelu,
                    _ => prism_spatial_ir::UOpKind::Tanh,
                }
            }
            _ => {
                return Err(format!(
                    "unsupported SpatialGraph Elementwise operation '{operation}'"
                ))
            }
        }
    };
    let value = graph.add(
        kind,
        inputs,
        output_shape.dims.iter().map(|dim| *dim as u64).collect(),
    );
    graph.add(
        prism_spatial_ir::UOpKind::Output { name: "out".into() },
        vec![value],
        output_shape.dims.iter().map(|dim| *dim as u64).collect(),
    );
    let capture = graph
        .lower(target)
        .map_err(|error| format!("lower SpatialGraph Elementwise: {error}"))?;
    let artifacts = compile_and_validate_uop_capture(&capture)?;
    Ok((capture, artifacts))
}

/// Publication gate for a compiled capture. It verifies that backend
/// descriptors identify the exact rendered source and that every artifact is
/// structurally complete before CImage sealing.
pub fn compile_and_validate_uop_capture(
    capture: &CapturePlan,
) -> Result<Vec<KernelArtifact>, String> {
    capture.validate()?;
    let artifacts = compile_uop_capture(capture)?;
    if artifacts.len() != capture.kernels.len() {
        return Err("backend returned a different number of UOp artifacts".into());
    }
    for (index, (artifact, kernel)) in artifacts.iter().zip(&capture.kernels).enumerate() {
        let descriptor = artifact
            .manifest
            .kernels
            .first()
            .ok_or_else(|| format!("UOp artifact {index} has no manifest kernel"))?;
        if descriptor.source_digest != kernel.source_digest {
            return Err(format!("UOp artifact {index} source digest mismatch"));
        }
        let expected_bindings = if kernel.group.ssm_shape().is_some() {
            5
        } else if kernel.group.requires_tertiary_input() {
            4
        } else if kernel.group.attention_shape().is_some()
            || kernel.group.batched_attention_shape().is_some()
            || kernel.group.layer_norm_shape().is_some()
            || kernel.group.rope_shape().is_some()
            || kernel.group.conv2d_shape().is_some()
            || kernel.group.ssm_shape().is_some()
        {
            4
        } else if kernel.group.scatter_shape().is_some() {
            4
        } else if kernel.group.requires_rhs()
            || kernel.group.matmul_shape().is_some()
            || kernel.group.rms_norm_shape().is_some()
            || kernel.group.gather_shape().is_some()
        {
            3
        } else {
            2
        };
        if descriptor.binding_signature.len() != expected_bindings {
            return Err(format!("UOp artifact {index} binding count mismatch"));
        }
        if descriptor
            .binding_signature
            .first()
            .map(|binding| binding.role)
            != Some(prism_ecs_kernel::BufferRole::Input)
            || descriptor
                .binding_signature
                .last()
                .map(|binding| binding.role)
                != Some(prism_ecs_kernel::BufferRole::Output)
        {
            return Err(format!(
                "UOp artifact {index} binding roles do not match capture ABI"
            ));
        }
        if artifact.payloads.is_empty() || artifact.artifact_digest.is_empty() {
            return Err(format!("UOp artifact {index} is incomplete"));
        }
    }
    Ok(artifacts)
}

#[derive(Debug, Clone)]
pub struct UOpCompiledProgram {
    pub capture: CapturePlan,
    pub artifacts: Vec<KernelArtifact>,
}

impl UOpCompiledProgram {
    pub fn from_cimage(reader: &crate::cimage::CImageReader) -> Result<Self, String> {
        let capture = reader.uop_capture()?;
        Self::from_cimage_with_prefix(reader, capture, "prism_uop_")
    }

    /// Load one strategy-indexed UOp capture and its namespaced kernel set.
    pub fn from_cimage_strategy(
        reader: &crate::cimage::CImageReader,
        strategy: &str,
    ) -> Result<Self, String> {
        let capture = reader.uop_capture_for_strategy(strategy)?;
        let prefix = strategy_kernel_prefix(strategy);
        Self::from_cimage_with_prefix(reader, capture, &prefix)
    }

    fn from_cimage_with_prefix(
        reader: &crate::cimage::CImageReader,
        capture: CapturePlan,
        prefix: &str,
    ) -> Result<Self, String> {
        let mut entries: Vec<_> = reader.header.kernels.keys().cloned().collect();
        entries.sort_by_key(|name| {
            name.strip_prefix(prefix)
                .and_then(|suffix| suffix.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
        entries.retain(|name| {
            name.starts_with(prefix)
                && !(prefix == "prism_uop_" && name.starts_with("prism_uop_strategy_"))
        });
        if entries.len() != capture.kernels.len() {
            return Err("CImage UOp kernel count does not match capture".into());
        }
        let mut artifacts = Vec::with_capacity(entries.len());
        for (index, name) in entries.iter().enumerate() {
            let descriptor = reader
                .header
                .kernels
                .get(name)
                .and_then(|record| record.descriptor.clone())
                .ok_or_else(|| format!("CImage UOp kernel '{name}' has no descriptor"))?;
            if descriptor.source_digest != capture.kernels[index].source_digest {
                return Err(format!("CImage UOp kernel '{name}' source digest mismatch"));
            }
            let expected_bindings = if capture.kernels[index].group.ssm_shape().is_some() {
                5
            } else if capture.kernels[index].group.attention_shape().is_some()
                || capture.kernels[index]
                    .group
                    .batched_attention_shape()
                    .is_some()
                || capture.kernels[index].group.layer_norm_shape().is_some()
                || capture.kernels[index].group.rope_shape().is_some()
                || capture.kernels[index].group.conv2d_shape().is_some()
                || capture.kernels[index].group.ssm_shape().is_some()
            {
                4
            } else if capture.kernels[index].group.scatter_shape().is_some() {
                4
            } else if capture.kernels[index].group.requires_rhs()
                || capture.kernels[index].group.matmul_shape().is_some()
                || capture.kernels[index].group.rms_norm_shape().is_some()
                || capture.kernels[index].group.gather_shape().is_some()
            {
                3
            } else {
                2
            };
            if descriptor.binding_signature.len() != expected_bindings
                || descriptor
                    .binding_signature
                    .first()
                    .map(|binding| binding.role)
                    != Some(prism_ecs_kernel::BufferRole::Input)
                || descriptor
                    .binding_signature
                    .last()
                    .map(|binding| binding.role)
                    != Some(prism_ecs_kernel::BufferRole::Output)
            {
                return Err(format!("CImage UOp kernel '{name}' binding ABI mismatch"));
            }
            let binary = reader.load_kernel(name)?;
            artifacts.push(prism_ecs_kernel::KernelArtifact {
                payloads: vec![prism_ecs_kernel::KernelPayload {
                    binary,
                    descriptor: descriptor.clone(),
                }],
                manifest: prism_ecs_kernel::KernelManifest {
                    kernels: vec![descriptor],
                    fusion_plan: None,
                    manifest_digest: String::new(),
                },
                artifact_digest: String::new(),
            });
        }
        Ok(Self { capture, artifacts })
    }
}

#[derive(Debug, Default, Clone)]
pub struct UOpCompileCache {
    entries: Arc<Mutex<HashMap<String, Vec<KernelArtifact>>>>,
}

impl UOpCompileCache {
    pub fn compile(&self, capture: CapturePlan) -> Result<UOpCompiledProgram, String> {
        let key = capture.digest();
        if let Some(artifacts) = self
            .entries
            .lock()
            .map_err(|_| "UOp compile cache is poisoned".to_string())?
            .get(&key)
            .cloned()
        {
            return Ok(UOpCompiledProgram { capture, artifacts });
        }
        let artifacts = compile_and_validate_uop_capture(&capture)?;
        self.entries
            .lock()
            .map_err(|_| "UOp compile cache is poisoned".to_string())?
            .insert(key, artifacts.clone());
        Ok(UOpCompiledProgram { capture, artifacts })
    }

    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    /// Remove one compiled capture from the cache. This is used when a
    /// backend compiler, device ABI, or calibration generation is retired;
    /// retaining the graph capture must not force reuse of stale artifacts.
    pub fn invalidate(&self, capture_digest: &str) -> Result<bool, String> {
        Ok(self
            .entries
            .lock()
            .map_err(|_| "UOp compile cache is poisoned".to_string())?
            .remove(capture_digest)
            .is_some())
    }

    /// Drop all compiled backend artifacts while leaving callers free to
    /// retain and recapture their immutable `CapturePlan` values.
    pub fn clear(&self) -> Result<(), String> {
        self.entries
            .lock()
            .map_err(|_| "UOp compile cache is poisoned".to_string())?
            .clear();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct UOpDispatchResult {
    pub outputs: BTreeMap<String, Vec<f32>>,
    pub receipt: prism_spatial_ir::ExecutionReceipt,
}

impl UOpCompiledProgram {
    fn materialize_reshape_aliases(&self, values: &mut BTreeMap<prism_spatial_ir::UOpId, Vec<u8>>) {
        let mut changed = true;
        while changed {
            changed = false;
            for op in &self.capture.graph.ops {
                if !matches!(op.kind, prism_spatial_ir::UOpKind::Reshape)
                    || values.contains_key(&op.id)
                {
                    continue;
                }
                if let Some(source) = op.src.first().and_then(|source| values.get(source)) {
                    values.insert(op.id, source.clone());
                    changed = true;
                }
            }
        }
    }

    fn validate_runtime_inputs(&self, inputs: &BTreeMap<String, Vec<f32>>) -> Result<(), String> {
        for op in &self.capture.graph.ops {
            let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
                continue;
            };
            let expected = op
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .ok_or_else(|| format!("input '{name}' shape overflows element count"))?;
            let actual = inputs
                .get(name)
                .ok_or_else(|| format!("missing UOp input '{name}'"))?
                .len();
            if actual != expected {
                return Err(format!(
                    "UOp input '{name}' has {actual} elements; capture requires {expected}"
                ));
            }
        }
        Ok(())
    }

    pub fn compile(capture: CapturePlan) -> Result<Self, String> {
        let artifacts = compile_and_validate_uop_capture(&capture)?;
        Ok(Self { capture, artifacts })
    }

    pub fn dispatch(
        &self,
        inputs: &BTreeMap<String, Vec<f32>>,
    ) -> Result<UOpDispatchResult, String> {
        if matches!(
            self.capture.target,
            LoweringTarget::Cpu | LoweringTarget::Portable
        ) {
            return self.dispatch_cpu(inputs);
        }
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        if matches!(self.capture.target, LoweringTarget::Metal) {
            return self.dispatch_metal_capture(inputs);
        }
        let outputs = execute_uop_reference(&self.capture, inputs)?;
        struct NoopExecutor;
        impl prism_spatial_ir::CaptureExecutor for NoopExecutor {
            fn dispatch(
                &mut self,
                _command_id: u32,
                _kernel: &prism_spatial_ir::LoweredKernel,
            ) -> Result<(), String> {
                Ok(())
            }
            fn synchronize(&mut self, _command_id: u32) -> Result<(), String> {
                Ok(())
            }
        }
        let mut executor = NoopExecutor;
        let receipt = self.capture.replay(&mut executor)?;
        self.capture.validate_receipt(&receipt)?;
        Ok(UOpDispatchResult { outputs, receipt })
    }

    /// Execute the scheduled compiled artifacts sequentially on the CPU
    /// backend, retaining each kernel result as a graph value for later
    /// groups. This is the deterministic backend replay boundary used by
    /// portable CImages and test environments without a GPU.
    pub fn dispatch_cpu(
        &self,
        inputs: &BTreeMap<String, Vec<f32>>,
    ) -> Result<UOpDispatchResult, String> {
        if !matches!(
            self.capture.target,
            LoweringTarget::Cpu | LoweringTarget::Portable
        ) {
            return Err("CPU UOp replay requires a CPU or Portable capture target".into());
        }
        if self.artifacts.len() != self.capture.kernels.len() {
            return Err("CPU UOp replay artifact count does not match capture".into());
        }
        self.validate_runtime_inputs(inputs)?;
        let mut values: BTreeMap<prism_spatial_ir::UOpId, Vec<u8>> = BTreeMap::new();
        let schedule = self
            .capture
            .graph
            .schedule()
            .map_err(|error| format!("CPU UOp replay schedule: {error}"))?;
        let command_position: BTreeMap<_, _> = schedule
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, position))
            .collect();
        let output_sources: std::collections::BTreeSet<_> = self
            .capture
            .graph
            .ops
            .iter()
            .filter_map(|op| match op.kind {
                prism_spatial_ir::UOpKind::Output { .. } => op.src.first().copied(),
                _ => None,
            })
            .collect();
        for op in &self.capture.graph.ops {
            match &op.kind {
                prism_spatial_ir::UOpKind::Input { name } => {
                    let data = inputs
                        .get(name)
                        .ok_or_else(|| format!("missing UOp input '{name}'"))?;
                    values.insert(
                        op.id,
                        data.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                    );
                }
                prism_spatial_ir::UOpKind::Const { value } => {
                    let elements = op
                        .shape
                        .iter()
                        .map(|dimension| *dimension as usize)
                        .product::<usize>();
                    let bytes = value.to_ne_bytes();
                    let mut data = Vec::with_capacity(elements * bytes.len());
                    for _ in 0..elements {
                        data.extend_from_slice(&bytes);
                    }
                    values.insert(op.id, data);
                }
                _ => {}
            }
        }
        self.materialize_reshape_aliases(&mut values);
        for (index, kernel) in self.capture.kernels.iter().enumerate() {
            let op_ids = kernel.group.op_ids();
            let op_set: std::collections::BTreeSet<_> = op_ids.iter().copied().collect();
            let mut source_ids = Vec::new();
            for op_id in &op_ids {
                let op = self
                    .capture
                    .graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == *op_id)
                    .ok_or_else(|| format!("kernel {index} references missing UOp {op_id:?}"))?;
                for source in &op.src {
                    if !op_set.contains(source) && !source_ids.contains(source) {
                        source_ids.push(*source);
                    }
                }
            }
            let scalar_variant = self.artifacts[index]
                .manifest
                .kernels
                .first()
                .is_some_and(|descriptor| matches!(&descriptor.variant, KernelVariant::Custom(name) if name.starts_with("uop_elementwise_scalar:")));
            if scalar_variant {
                source_ids.retain(|source| {
                    self.capture
                        .graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == *source)
                        .is_some_and(|op| {
                            !matches!(op.kind, prism_spatial_ir::UOpKind::Const { .. })
                        })
                });
            }
            let mut buffers = Vec::with_capacity(source_ids.len());
            for source in source_ids {
                buffers.push(
                    values
                        .get(&source)
                        .cloned()
                        .ok_or_else(|| format!("kernel {index} source {source:?} has no value"))?,
                );
            }
            let bindings = self.artifacts[index]
                .manifest
                .kernels
                .first()
                .map(|descriptor| descriptor.binding_signature.clone())
                .unwrap_or_default();
            let output = CpuBackend
                .dispatch(&prism_ecs_kernel::KernelDispatchRequest {
                    artifact: self.artifacts[index].clone(),
                    inputs: buffers,
                    bindings,
                })
                .map_err(|error| format!("CPU UOp kernel {index}: {error}"))?;
            let final_id = op_ids
                .last()
                .ok_or_else(|| format!("kernel {index} is empty"))?;
            values.insert(
                *final_id,
                output
                    .outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("CPU UOp kernel {index} returned no output"))?,
            );
            self.materialize_reshape_aliases(&mut values);
            let final_position = command_position
                .get(final_id)
                .copied()
                .ok_or_else(|| format!("kernel {index} final UOp has no schedule position"))?;
            for allocation in &self.capture.memory_plan.allocations {
                if allocation.last_command <= final_position
                    && !output_sources.contains(&allocation.value)
                {
                    values.remove(&allocation.value);
                }
            }
        }
        let mut outputs = BTreeMap::new();
        for op in &self.capture.graph.ops {
            if let prism_spatial_ir::UOpKind::Output { name } = &op.kind {
                let source = op
                    .src
                    .first()
                    .ok_or_else(|| format!("output '{name}' has no source"))?;
                let bytes = values
                    .get(source)
                    .ok_or_else(|| format!("output '{name}' source has no value"))?;
                let expected_bytes = op
                    .shape
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count.checked_mul(*dimension as usize)
                    })
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| format!("output '{name}' shape overflows byte count"))?;
                if bytes.len() != expected_bytes {
                    return Err(format!(
                        "output '{name}' has {} bytes; capture requires {expected_bytes}",
                        bytes.len()
                    ));
                }
                if bytes.len() % 4 != 0 {
                    return Err(format!("output '{name}' is not FP32-aligned"));
                }
                outputs.insert(
                    name.clone(),
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                );
            }
        }
        struct NoopExecutor;
        impl prism_spatial_ir::CaptureExecutor for NoopExecutor {
            fn dispatch(
                &mut self,
                _: u32,
                _: &prism_spatial_ir::LoweredKernel,
            ) -> Result<(), String> {
                Ok(())
            }
            fn synchronize(&mut self, _: u32) -> Result<(), String> {
                Ok(())
            }
        }
        let mut executor = NoopExecutor;
        let receipt = self.capture.replay(&mut executor)?;
        self.capture.validate_receipt(&receipt)?;
        Ok(UOpDispatchResult { outputs, receipt })
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub fn dispatch_metal(&self, input: Vec<u8>, rhs: Option<Vec<u8>>) -> Result<Vec<u8>, String> {
        use prism_ecs_kernel::{KernelBackend, KernelDispatchRequest, MetalBackend};
        let artifact = self
            .artifacts
            .first()
            .ok_or_else(|| "UOp program has no compiled artifact".to_string())?
            .clone();
        let mut inputs = vec![input];
        if let Some(rhs) = rhs {
            inputs.push(rhs);
        }
        let bindings = artifact
            .manifest
            .kernels
            .first()
            .ok_or_else(|| "UOp artifact has no descriptor".to_string())?
            .binding_signature
            .clone();
        let output = MetalBackend::default()
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs,
                bindings,
            })
            .map_err(|error| format!("Metal UOp dispatch: {error}"))?;
        output
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| "Metal UOp dispatch returned no output".into())
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub fn dispatch_metal_capture(
        &self,
        inputs: &BTreeMap<String, Vec<f32>>,
    ) -> Result<UOpDispatchResult, String> {
        use prism_ecs_kernel::{KernelBackend, KernelDispatchRequest, MetalBackend};
        self.validate_runtime_inputs(inputs)?;
        let mut values: BTreeMap<prism_spatial_ir::UOpId, Vec<u8>> = BTreeMap::new();
        let schedule = self
            .capture
            .graph
            .schedule()
            .map_err(|error| format!("Metal UOp replay schedule: {error}"))?;
        let command_position: BTreeMap<_, _> = schedule
            .iter()
            .enumerate()
            .map(|(position, id)| (*id, position))
            .collect();
        let output_sources: std::collections::BTreeSet<_> = self
            .capture
            .graph
            .ops
            .iter()
            .filter_map(|op| match op.kind {
                prism_spatial_ir::UOpKind::Output { .. } => op.src.first().copied(),
                _ => None,
            })
            .collect();
        for op in &self.capture.graph.ops {
            match &op.kind {
                prism_spatial_ir::UOpKind::Input { name } => {
                    let data = inputs
                        .get(name)
                        .ok_or_else(|| format!("missing UOp input '{name}'"))?;
                    values.insert(
                        op.id,
                        data.iter().flat_map(|value| value.to_ne_bytes()).collect(),
                    );
                }
                prism_spatial_ir::UOpKind::Const { value } => {
                    let elements = op
                        .shape
                        .iter()
                        .map(|dimension| *dimension as usize)
                        .product::<usize>();
                    let bytes = value.to_ne_bytes();
                    let mut data = Vec::with_capacity(elements * bytes.len());
                    for _ in 0..elements {
                        data.extend_from_slice(&bytes);
                    }
                    values.insert(op.id, data);
                }
                _ => {}
            }
        }
        self.materialize_reshape_aliases(&mut values);
        for (index, kernel) in self.capture.kernels.iter().enumerate() {
            let op_ids = kernel.group.op_ids();
            let op_set: std::collections::BTreeSet<_> = op_ids.iter().copied().collect();
            let mut source_ids = Vec::new();
            for op_id in &op_ids {
                let op = self
                    .capture
                    .graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == *op_id)
                    .ok_or_else(|| format!("kernel {index} references missing UOp {op_id:?}"))?;
                for source in &op.src {
                    if !op_set.contains(source) && !source_ids.contains(source) {
                        source_ids.push(*source);
                    }
                }
            }
            let scalar_variant = self.artifacts[index].manifest.kernels.first().is_some_and(|descriptor| matches!(&descriptor.variant, KernelVariant::Custom(name) if name.starts_with("uop_elementwise_scalar:")));
            if scalar_variant {
                source_ids.retain(|source| {
                    self.capture
                        .graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == *source)
                        .is_some_and(|op| {
                            !matches!(op.kind, prism_spatial_ir::UOpKind::Const { .. })
                        })
                });
            }
            let buffers = source_ids
                .into_iter()
                .map(|source| {
                    values
                        .get(&source)
                        .cloned()
                        .ok_or_else(|| format!("kernel {index} source {source:?} has no value"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let descriptor = self.artifacts[index]
                .manifest
                .kernels
                .first()
                .ok_or_else(|| format!("kernel {index} has no descriptor"))?;
            let output = MetalBackend::default()
                .dispatch(&KernelDispatchRequest {
                    artifact: self.artifacts[index].clone(),
                    inputs: buffers,
                    bindings: descriptor.binding_signature.clone(),
                })
                .map_err(|error| format!("Metal UOp kernel {index}: {error}"))?;
            values.insert(
                *op_ids
                    .last()
                    .ok_or_else(|| format!("kernel {index} is empty"))?,
                output
                    .outputs
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("Metal UOp kernel {index} returned no output"))?,
            );
            self.materialize_reshape_aliases(&mut values);
            let final_id = op_ids
                .last()
                .ok_or_else(|| format!("kernel {index} is empty"))?;
            let final_position = command_position
                .get(final_id)
                .copied()
                .ok_or_else(|| format!("kernel {index} final UOp has no schedule position"))?;
            for allocation in &self.capture.memory_plan.allocations {
                if allocation.last_command <= final_position
                    && !output_sources.contains(&allocation.value)
                {
                    values.remove(&allocation.value);
                }
            }
        }
        let mut outputs = BTreeMap::new();
        for op in &self.capture.graph.ops {
            if let prism_spatial_ir::UOpKind::Output { name } = &op.kind {
                let source = op
                    .src
                    .first()
                    .ok_or_else(|| format!("output '{name}' has no source"))?;
                let bytes = values
                    .get(source)
                    .ok_or_else(|| format!("output '{name}' source has no value"))?;
                let expected_bytes = op
                    .shape
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count.checked_mul(*dimension as usize)
                    })
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or_else(|| format!("output '{name}' shape overflows byte count"))?;
                if bytes.len() != expected_bytes {
                    return Err(format!(
                        "output '{name}' has {} bytes; capture requires {expected_bytes}",
                        bytes.len()
                    ));
                }
                if bytes.len() % 4 != 0 {
                    return Err(format!("output '{name}' is not FP32-aligned"));
                }
                outputs.insert(
                    name.clone(),
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                );
            }
        }
        struct NoopExecutor;
        impl prism_spatial_ir::CaptureExecutor for NoopExecutor {
            fn dispatch(
                &mut self,
                _: u32,
                _: &prism_spatial_ir::LoweredKernel,
            ) -> Result<(), String> {
                Ok(())
            }
            fn synchronize(&mut self, _: u32) -> Result<(), String> {
                Ok(())
            }
        }
        let mut executor = NoopExecutor;
        let receipt = self.capture.replay(&mut executor)?;
        self.capture.validate_receipt(&receipt)?;
        Ok(UOpDispatchResult { outputs, receipt })
    }
}

/// Run the capture through the deterministic behavioral oracle used to
/// validate backend artifacts before hardware replay.
pub fn execute_uop_reference(
    capture: &CapturePlan,
    inputs: &BTreeMap<String, Vec<f32>>,
) -> Result<BTreeMap<String, Vec<f32>>, String> {
    capture
        .graph
        .execute_f32(inputs)
        .map_err(|error| format!("execute UOp reference: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_spatial_ir::{TinyGraph, UOpKind};
    use std::collections::BTreeMap;

    #[test]
    fn compiles_portable_capture_into_kernel_artifact() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![64]);
        let constant = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let add = graph.add(UOpKind::Add, vec![input, constant], vec![64]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![add], vec![64]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifacts = compile_uop_capture(&capture).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].payloads.len(), 1);
        assert_eq!(artifacts[0].manifest.kernels[0].backend, BackendKind::CPU);
        assert!(!artifacts[0].payloads[0].binary.is_empty());
    }

    #[test]
    fn runtime_rejects_input_geometry_that_does_not_match_capture() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2, 3]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2, 3]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();

        let error = program
            .dispatch_cpu(&BTreeMap::from([(String::from("x"), vec![0.0; 5])]))
            .unwrap_err();
        assert!(error.contains("requires 6"));
    }

    #[test]
    fn strategy_specific_captures_cross_artifact_and_dispatch_boundaries() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![2]);
        let strategy = prism_spatial_ir::FusionStrategy::PerOperation;
        let (capture, artifacts) =
            compile_uop_graph_with_strategy(&graph, LoweringTarget::Portable, &strategy).unwrap();
        assert_eq!(artifacts.len(), 2);
        let program = UOpCompiledProgram::compile(capture).unwrap();
        assert_eq!(program.artifacts.len(), 2);
        let result = program
            .dispatch_cpu(&BTreeMap::from([("x".into(), vec![-1.0, 1.0])]))
            .unwrap();
        assert_eq!(result.outputs["y"].len(), 2);
        assert_eq!(result.outputs["y"][0], 1.0);
        assert!((result.outputs["y"][1] - std::f32::consts::E).abs() < 1e-6);
    }

    #[test]
    fn compiled_cpu_replay_materializes_reshape_aliases() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let reshaped = graph.add(UOpKind::Reshape, vec![input], vec![3, 2]);
        let relu = graph.add(UOpKind::Relu, vec![reshaped], vec![3, 2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![3, 2]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch_cpu(&BTreeMap::from([(
                String::from("x"),
                vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],
            )]))
            .unwrap();
        assert_eq!(result.outputs["y"], vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    }

    #[test]
    fn compiled_cpu_replay_materializes_shaped_constants() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let constant = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![4]);
        let sum = graph.add(UOpKind::Add, vec![input, constant], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![sum], vec![4]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch_cpu(&BTreeMap::from([(
                String::from("x"),
                vec![1.0, 2.0, 3.0, 4.0],
            )]))
            .unwrap();
        assert_eq!(result.outputs["y"], vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn benchmark_uop_strategies_measures_repeated_dispatches() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![8]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![8]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![8]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![8]);
        let strategies = [
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
            prism_spatial_ir::FusionStrategy::PerOperation,
            prism_spatial_ir::FusionStrategy::PersistentMegakernel {
                search_generation: 1,
            },
        ];
        let inputs = BTreeMap::from([(String::from("x"), vec![1.0; 8])]);
        let measurements = benchmark_uop_graph_strategies(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            &inputs,
            2,
        )
        .unwrap();
        assert_eq!(measurements.len(), strategies.len());
        assert!(measurements
            .iter()
            .all(|measurement| measurement.latency_ns > 0));
        let runner_measurements = benchmark_uop_graph_strategies_with_runner(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            |index, strategy, capture| {
                assert_eq!(strategy.stable_id(), strategies[index].stable_id());
                assert!(!capture.kernels.is_empty());
                Ok((index as u64 + 1, capture.kernels.len() as u64))
            },
        )
        .unwrap();
        assert_eq!(runner_measurements.len(), strategies.len());
    }

    #[test]
    fn benchmark_uop_workloads_keeps_realtime_and_batch_measurements_separate() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![8]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![8]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![8]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![8]);
        let strategies = [
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
            prism_spatial_ir::FusionStrategy::PerOperation,
            prism_spatial_ir::FusionStrategy::PersistentMegakernel {
                search_generation: 1,
            },
        ];
        let scenarios = [
            prism_spatial_ir::WorkloadScenario {
                realtime: true,
                batch_size: 1,
                sequence_length: 1,
            },
            prism_spatial_ir::WorkloadScenario {
                realtime: false,
                batch_size: 8,
                sequence_length: 16,
            },
        ];
        let results = benchmark_uop_graph_workloads(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            &scenarios,
            |_| BTreeMap::from([(String::from("x"), vec![1.0; 8])]),
            1,
        )
        .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].scenario.realtime);
        assert!(!results[1].scenario.realtime);
        assert_eq!(results[0].measurements.len(), strategies.len());
        assert_eq!(results[1].measurements.len(), strategies.len());
        let (selected, _) =
            select_measured_uop_strategy(&strategies, &results[0].measurements).unwrap();
        assert!(strategies
            .iter()
            .any(|strategy| strategy.stable_id() == selected));
        let selections = select_measured_uop_workloads(&strategies, &results).unwrap();
        assert_eq!(selections.len(), 2);
        assert_eq!(selections[0].scenario, scenarios[0]);
        assert_eq!(selections[1].scenario, scenarios[1]);
        assert!(strategies
            .iter()
            .any(|strategy| strategy.stable_id() == selections[0].strategy_id));
        assert!(strategies
            .iter()
            .any(|strategy| strategy.stable_id() == selections[1].strategy_id));
        assert!(select_measured_uop_strategy(
            &strategies,
            &[prism_spatial_ir::FusionMeasurement {
                candidate_index: 7,
                latency_ns: 1,
                materialized_bytes: 0,
            }]
        )
        .is_err());
        assert!(select_measured_uop_strategy(
            &strategies,
            &[prism_spatial_ir::FusionMeasurement {
                candidate_index: 0,
                latency_ns: 0,
                materialized_bytes: 0,
            }]
        )
        .is_err());
        let invalid_workload = UOpWorkloadMeasurement {
            scenario: prism_spatial_ir::WorkloadScenario {
                realtime: true,
                batch_size: 2,
                sequence_length: 1,
            },
            measurements: results[0].measurements.clone(),
        };
        assert!(
            select_measured_uop_workloads(&strategies, &[invalid_workload])
                .unwrap_err()
                .contains("realtime workload batch size must be one")
        );
        let heterogeneous = benchmark_uop_graph_workloads_with_runner(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            &scenarios,
            |_| BTreeMap::from([(String::from("x"), vec![2.0; 8])]),
            |scenario, index, _, capture, inputs| {
                assert_eq!(inputs.get("x").map(Vec::len), Some(8));
                assert!(index < strategies.len());
                assert!(scenario.batch_size > 0);
                assert!(!capture.kernels.is_empty());
                Ok((1, 0))
            },
        )
        .unwrap();
        assert_eq!(heterogeneous.len(), scenarios.len());
        assert!(benchmark_uop_graph_workloads_with_runner(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            &scenarios[..1],
            |_| BTreeMap::from([(String::from("x"), vec![2.0; 8])]),
            |_, _, _, _, _| Ok((0, 0)),
        )
        .unwrap_err()
        .contains("zero-latency"));
    }

    #[test]
    fn compiles_a_complete_strategy_candidate_set() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
        let strategies = vec![
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::InterleavedFused {
                stages: vec![
                    vec![prism_spatial_ir::FusableOp::FpGemv],
                    vec![prism_spatial_ir::FusableOp::Silu],
                ],
            },
            prism_spatial_ir::FusionStrategy::PerOperation,
            prism_spatial_ir::FusionStrategy::PersistentMegakernel {
                search_generation: 3,
            },
        ];
        let candidates =
            compile_uop_graph_strategies(&graph, LoweringTarget::Portable, &strategies).unwrap();
        assert_eq!(candidates.len(), strategies.len());
        assert_eq!(candidates[0].1.kernels.len(), 1);
        assert_eq!(candidates[1].1.kernels.len(), 2);
        assert_eq!(candidates[3].1.kernels.len(), 1);
        assert!(candidates[3].1.replay.persistent);
        assert!(candidates
            .iter()
            .all(|(_, _, artifacts)| !artifacts.is_empty()));

        let validated = compile_and_validate_uop_graph_strategies(
            &graph,
            LoweringTarget::Portable,
            &strategies,
            &BTreeMap::from([(String::from("x"), vec![-1.0, 0.0, 1.0, 2.0])]),
        )
        .unwrap();
        assert_eq!(validated.len(), strategies.len());
        let persistent_program = UOpCompiledProgram::compile(validated[3].1.clone()).unwrap();
        let persistent_output = persistent_program
            .dispatch_cpu(&BTreeMap::from([(
                String::from("x"),
                vec![-1.0, 0.0, 1.0, 2.0],
            )]))
            .unwrap();
        assert_eq!(persistent_output.outputs["y"].len(), 4);
    }

    #[test]
    fn kernel_free_reshape_capture_replays_as_a_storage_alias() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let reshaped = graph.add(UOpKind::Reshape, vec![input], vec![3, 2]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![reshaped],
            vec![3, 2],
        );
        let program = UOpCompiledProgram::compile(
            graph
                .lower(LoweringTarget::Portable)
                .expect("reshape graph should lower"),
        )
        .expect("kernel-free reshape capture should compile");
        assert!(program.capture.kernels.is_empty());
        assert!(program.artifacts.is_empty());
        let result = program
            .dispatch_cpu(&BTreeMap::from([(
                String::from("x"),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )]))
            .expect("reshape alias should replay");
        assert_eq!(result.outputs["y"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(result.receipt.command_ids.is_empty());
    }

    #[test]
    fn rejects_duplicate_strategy_candidates() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
        let strategies = [
            prism_spatial_ir::FusionStrategy::StandardFused,
            prism_spatial_ir::FusionStrategy::StandardFused,
        ];
        let error = compile_uop_graph_strategies(&graph, LoweringTarget::Portable, &strategies)
            .unwrap_err();
        assert!(error.contains("duplicate UOp fusion strategy"));

        let distinct_interleaved_names = [
            prism_spatial_ir::FusionStrategy::InterleavedFused { stages: vec![] },
            prism_spatial_ir::FusionStrategy::InterleavedFused {
                stages: vec![vec![prism_spatial_ir::FusableOp::FpGemv]],
            },
        ];
        let error = compile_uop_graph_strategies(
            &graph,
            LoweringTarget::Portable,
            &distinct_interleaved_names,
        )
        .unwrap_err();
        assert!(error.contains("runtime namespace"));
    }

    #[test]
    fn executes_capture_through_reference_boundary() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-2.0, 3.0]);
        let outputs = execute_uop_reference(&capture, &inputs).unwrap();
        assert_eq!(outputs["y"], vec![0.0, 3.0]);
    }

    #[test]
    fn two_input_capture_gets_matching_kernel_bindings() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![2]);
        let sum = graph.add(UOpKind::Add, vec![x, y], vec![2]);
        graph.add(UOpKind::Output { name: "z".into() }, vec![sum], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifacts = compile_uop_capture(&capture).unwrap();
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_elementwise_binary:add".into())
        );
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 3);
        assert_eq!(
            artifacts[0].manifest.kernels[0].binding_signature[2].role,
            BufferRole::Output
        );
    }

    #[test]
    fn scalar_elementwise_publishes_value_and_operand_order() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let constant = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let sub = graph.add(UOpKind::Sub, vec![constant, x], vec![2]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![sub], vec![2]);
        let artifacts =
            compile_uop_capture(&graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_elementwise_scalar:sub|2|1".into())
        );
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 2);
    }

    #[test]
    fn compiled_program_dispatches_scalar_capture_without_constant_binding() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let constant = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let sub = graph.add(UOpKind::Sub, vec![constant, x], vec![2]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![sub], vec![2]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch_cpu(&BTreeMap::from([("x".into(), vec![1.0, 3.0])]))
            .unwrap();
        assert_eq!(result.outputs["out"], vec![1.0, -1.0]);
    }

    #[test]
    fn publication_gate_checks_source_provenance() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifacts = compile_and_validate_uop_capture(&capture).unwrap();
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn reduction_capture_publishes_shape_aware_kernel_variant() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let sum = graph.add(UOpKind::ReduceSum, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![sum], vec![1]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifacts = compile_and_validate_uop_capture(&capture).unwrap();
        let descriptor = &artifacts[0].manifest.kernels[0];
        assert_eq!(
            descriptor.variant,
            KernelVariant::Custom("uop_reduce_sum".into())
        );
        assert_eq!(descriptor.dispatch_geometry.threads_per_grid, [1, 1, 1]);
        assert_eq!(descriptor.binding_signature.len(), 2);
    }

    #[test]
    fn matmul_capture_publishes_dimensions_and_three_buffer_abi() {
        let mut graph = TinyGraph::default();
        let a = graph.add(UOpKind::Input { name: "a".into() }, vec![], vec![2, 3]);
        let b = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![3, 2]);
        let product = graph.add(UOpKind::MatMul { m: 2, k: 3, n: 2 }, vec![a, b], vec![2, 2]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![product],
            vec![2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifact = compile_and_validate_uop_capture(&capture)
            .unwrap()
            .remove(0);
        let descriptor = &artifact.manifest.kernels[0];
        assert_eq!(
            descriptor.variant,
            KernelVariant::Custom("uop_matmul:2:3:2".into())
        );
        assert_eq!(descriptor.dispatch_geometry.threads_per_grid, [4, 1, 1]);
        assert_eq!(descriptor.binding_signature.len(), 3);
    }

    #[test]
    fn axis_reduction_capture_publishes_dimensions() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let sum = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![sum], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifact = compile_and_validate_uop_capture(&capture)
            .unwrap()
            .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_reduce_sum_axis:2:3:1".into())
        );
        assert_eq!(
            artifact.manifest.kernels[0]
                .dispatch_geometry
                .threads_per_grid,
            [2, 1, 1]
        );
    }

    #[test]
    fn softmax_capture_publishes_axis_dimensions() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let softmax = graph.add(UOpKind::SoftmaxAxis { axis: 1 }, vec![input], vec![2, 3]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![softmax],
            vec![2, 3],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifact = compile_and_validate_uop_capture(&capture)
            .unwrap()
            .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_softmax_axis:2:3:1".into())
        );
        assert_eq!(
            artifact.manifest.kernels[0]
                .dispatch_geometry
                .threads_per_grid,
            [6, 1, 1]
        );
    }

    #[test]
    fn attention_capture_publishes_four_buffer_fused_abi() {
        let mut graph = TinyGraph::default();
        let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2]);
        let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2]);
        let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2]);
        let attention = graph.add(
            UOpKind::Attention {
                seq: 2,
                head: 2,
                scale: 0.5,
            },
            vec![q, k, v],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![attention],
            vec![2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifact = compile_and_validate_uop_capture(&capture)
            .unwrap()
            .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_attention:2:2:0.5".into())
        );
        assert_eq!(artifact.manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn batched_attention_capture_publishes_batch_geometry() {
        let mut graph = TinyGraph::default();
        let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2, 1]);
        let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2, 1]);
        let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2, 1]);
        let attention = graph.add(
            UOpKind::AttentionBatched {
                batch: 2,
                seq: 2,
                head: 1,
                scale: 1.0,
            },
            vec![q, k, v],
            vec![2, 2, 1],
        );
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![attention],
            vec![2, 2, 1],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifact = compile_and_validate_uop_capture(&capture)
            .unwrap()
            .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_attention_batched:2:2:1:1".into())
        );
        assert_eq!(
            artifact.manifest.kernels[0]
                .dispatch_geometry
                .threads_per_grid,
            [4, 1, 1]
        );
    }

    #[test]
    fn spatial_matmul_adapter_preserves_shape_contract() {
        let (capture, artifacts) =
            compile_spatial_matmul(2, 3, 2, LoweringTarget::Portable).unwrap();
        assert_eq!(capture.graph_op_count, 4);
        assert_eq!(artifacts.len(), 1);
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_matmul:2:3:2".into())
        );
    }

    #[test]
    fn spatial_node_adapter_accepts_only_consistent_matmul_contracts() {
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(7),
            kind: prism_spatial_ir::graph::ComputeKind::MatMul,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 3] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![3, 2] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (_, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn annotated_spatial_elementwise_adapter_lowers_relu() {
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(8),
            kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.elementwise_op = Some("relu".into());
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0].source.contains("v > 0.0"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn annotated_spatial_elementwise_adapter_lowers_pow_with_metadata() {
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(81),
            kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.elementwise_op = Some("pow".into());
        metadata.pow_exponent = Some(2.0);
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("powf(v, 2f)"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_elementwise() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(18),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("relu".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("v = v > 0.0"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_silu_composition() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(181),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("silu".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("expf")));
        assert!(capture.kernels.len() >= 2);
        assert_eq!(artifacts.len(), capture.kernels.len());
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_softplus_composition() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(182),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("softplus".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("logf")));
        assert_eq!(artifacts.len(), capture.kernels.len());
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_log_softmax_composition() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(187),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("log_softmax".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2, 3] }],
                vec![TensorShape { dims: vec![2, 3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("prism_softmax_axis")));
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("logf")));
        assert_eq!(artifacts.len(), capture.kernels.len());
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_softmax() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(193),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("softmax".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2, 3] }],
                vec![TensorShape { dims: vec![2, 3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_softmax_axis"));
        assert!(!capture.kernels[0].source.contains("logf"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_ssm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(194),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("ssm".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![4] },
                    TensorShape { dims: vec![4] },
                    TensorShape { dims: vec![4] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_ssm"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_gather() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(195),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("gather".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![8, 4] },
                    TensorShape { dims: vec![2] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_gather"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_scatter() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(196),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("scatter".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![4, 2] },
                    TensorShape { dims: vec![2] },
                    TensorShape { dims: vec![2, 2] },
                ],
                vec![TensorShape { dims: vec![4, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_scatter"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_rms_norm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(188),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("rms_norm".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![4] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_rms_norm"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_layer_norm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(189),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("layer_norm".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![4] },
                    TensorShape { dims: vec![4] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_layer_norm"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_attention() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(191),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("attention".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![2, 4] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_attention"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_conv2d() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(192),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("conv2d".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape {
                        dims: vec![1, 2, 4, 4],
                    },
                    TensorShape {
                        dims: vec![3, 2, 3, 3],
                    },
                    TensorShape { dims: vec![3] },
                ],
                vec![TensorShape {
                    dims: vec![1, 3, 2, 2],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let metadata = prism_spatial_ir::graph::NodeMeta {
            convolution_stride: Some(1),
            convolution_padding: Some(0),
            ..Default::default()
        };
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("prism_conv2d"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_clamp() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let shape = TensorShape { dims: vec![2] };
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(183),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("clamp".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![shape.clone(), shape.clone(), shape.clone()],
                vec![shape],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("max")));
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("min")));
        assert_eq!(artifacts.len(), capture.kernels.len());
    }

    #[test]
    fn custom_spatial_node_adapter_lowers_validated_where() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let shape = TensorShape { dims: vec![3] };
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(184),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("where".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![shape.clone(), shape.clone(), shape.clone()],
                vec![shape],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0].source.contains("condition"));
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_where".into())
        );
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 4);
        let mut inputs = BTreeMap::new();
        inputs.insert("condition".into(), vec![0.0, 1.0, -2.0]);
        inputs.insert("when_true".into(), vec![10.0, 20.0, 30.0]);
        inputs.insert("when_false".into(), vec![1.0, 2.0, 3.0]);
        let output = execute_uop_reference(&capture, &inputs).unwrap();
        assert_eq!(output["out"], vec![1.0, 20.0, 30.0]);
    }

    #[test]
    fn custom_where_accepts_trailing_dimension_broadcast() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(194),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("where".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 1] },
                    TensorShape { dims: vec![2, 3] },
                    TensorShape { dims: vec![3] },
                ],
                vec![TensorShape { dims: vec![2, 3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_where"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_cast_validates_dtype_contract_and_executes_quantization() {
        use prism_ecs_ir::cimage_types::TensorShape;
        assert_eq!(
            classify_custom_operation("cast_f32_to_i8"),
            CustomOperationClass::Validated
        );
        assert!(!CUSTOM_OPERATION_CANDIDATES.contains(&"cast"));
        let shape = TensorShape { dims: vec![3] };
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(185),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("cast_f32_to_i8".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(vec![shape.clone()], vec![shape]),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_cast"));
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_elementwise".into())
        );
        let output = execute_uop_reference(
            &capture,
            &BTreeMap::from([("x".into(), vec![-2.9, 3.7, 300.0])]),
        )
        .unwrap();
        assert_eq!(output["out"], vec![-2.0, 3.0, 127.0]);
    }

    #[test]
    fn custom_cast_rejects_unsupported_dtype_instead_of_identity() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(186),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("cast_f32_to_fp8".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let error = compile_spatial_node(&node, LoweringTarget::Portable).unwrap_err();
        assert!(error.contains("unsupported cast dtype"));
    }

    #[test]
    fn custom_operation_lists_separate_validated_and_candidate_sets() {
        use std::collections::HashSet;

        let validated = VALIDATED_CUSTOM_OPERATIONS
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let candidates = CUSTOM_OPERATION_CANDIDATES
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(validated.len(), VALIDATED_CUSTOM_OPERATIONS.len());
        assert_eq!(candidates.len(), CUSTOM_OPERATION_CANDIDATES.len());
        assert!(validated.is_disjoint(&candidates));
        assert!(candidates.contains("flash_attention"));
        assert!(candidates.contains("group_norm"));
        assert!(candidates.contains("topk"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"relu"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"sigmoid"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"log_softmax"));
        assert!(!CUSTOM_OPERATION_CANDIDATES.contains(&"where"));
        assert_eq!(
            classify_custom_operation("where"),
            CustomOperationClass::Validated
        );
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"clamp"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"silu"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"softplus"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"transpose"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"rms_norm"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"layer_norm"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"softmax"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"ssm"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"gather"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"attention"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"conv2d"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"scatter"));
        assert!(VALIDATED_CUSTOM_OPERATIONS.contains(&"pow"));
        assert_eq!(
            classify_custom_operation("rms_norm"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("layer_norm"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("attention"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("conv2d"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("softmax"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("ssm"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("gather"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("scatter"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("transpose"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("sigmoid"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("RMS_NORM"),
            CustomOperationClass::Validated
        );
        assert_eq!(
            classify_custom_operation("not_an_operation"),
            CustomOperationClass::Candidate
        );
        assert_eq!(
            classify_custom_operation("flash_attention"),
            CustomOperationClass::Candidate
        );
        assert_eq!(
            classify_custom_operation("pow"),
            CustomOperationClass::Validated
        );
    }

    #[test]
    fn custom_pow_requires_and_uses_exponent_metadata() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(191),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("pow".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![3] }],
                vec![TensorShape { dims: vec![3] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let missing = compile_spatial_node(&node, LoweringTarget::Portable).unwrap_err();
        assert!(missing.contains("pow_exponent"));
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.pow_exponent = Some(2.0);
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("powf(v, 2f)"));
        assert_eq!(artifacts.len(), 1);
    }

    #[test]
    fn custom_transpose_lowers_nontrivial_permutation_from_metadata() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(190),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("transpose".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape {
                    dims: vec![2, 3, 4],
                }],
                vec![TensorShape {
                    dims: vec![4, 2, 3],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let metadata = prism_spatial_ir::graph::NodeMeta {
            permutation: Some(vec![2, 0, 1]),
            ..Default::default()
        };
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert_eq!(
            capture
                .graph
                .ops
                .iter()
                .find(|op| matches!(op.kind, prism_spatial_ir::UOpKind::Transpose { .. }))
                .unwrap()
                .shape,
            vec![4, 2, 3]
        );
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_transpose:2,0,1:2x3x4:4x2x3".into())
        );
    }

    #[test]
    fn unknown_custom_operation_is_contract_validated_before_admission() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(19),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("future_op".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let error = compile_spatial_node(&node, LoweringTarget::Portable).unwrap_err();
        assert!(error.contains("candidate"));
        assert_eq!(
            validate_and_classify_custom_operation(
                "future_op",
                match &node {
                    prism_spatial_ir::SpatialNode::Compute { shape, .. } => shape,
                    _ => unreachable!(),
                },
            )
            .unwrap(),
            CustomOperationClass::Candidate
        );

        let malformed = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(20),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("future-op".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let error = compile_spatial_node(&malformed, LoweringTarget::Portable).unwrap_err();
        assert!(error.contains("malformed"));

        let overflowing = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(21),
            kind: prism_spatial_ir::graph::ComputeKind::Custom("future_op".into()),
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape {
                    dims: vec![usize::MAX, 2],
                }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let error = compile_spatial_node(&overflowing, LoweringTarget::Portable).unwrap_err();
        assert!(error.contains("overflows"));
    }

    #[test]
    fn annotated_spatial_elementwise_adapter_lowers_gelu() {
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(10),
            kind: prism_spatial_ir::graph::ComputeKind::Elementwise,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.elementwise_op = Some("gelu".into());
        let (capture, _) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("tanhf"));
    }

    #[test]
    fn spatial_normalization_adapter_lowers_rms_norm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(11),
            kind: prism_spatial_ir::graph::ComputeKind::Normalization,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 4] },
                    TensorShape { dims: vec![4] },
                ],
                vec![TensorShape { dims: vec![2, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_rms_norm"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 3);
    }

    #[test]
    fn spatial_convolution_adapter_lowers_conv2d() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(13),
            kind: prism_spatial_ir::graph::ComputeKind::Convolution,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape {
                        dims: vec![1, 1, 3, 3],
                    },
                    TensorShape {
                        dims: vec![1, 1, 2, 2],
                    },
                    TensorShape { dims: vec![1] },
                ],
                vec![TensorShape {
                    dims: vec![1, 1, 2, 2],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_conv2d"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn spatial_node_adapter_lowers_rope() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(15),
            kind: prism_spatial_ir::graph::ComputeKind::RoPE,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![1, 4] },
                    TensorShape { dims: vec![1, 2] },
                    TensorShape { dims: vec![1, 2] },
                ],
                vec![TensorShape { dims: vec![1, 4] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_rope"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn spatial_node_adapter_lowers_gather() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(16),
            kind: prism_spatial_ir::graph::ComputeKind::Gather,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![3, 2] },
                    TensorShape { dims: vec![2] },
                ],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_gather"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 3);
    }

    #[test]
    fn spatial_node_adapter_lowers_ssm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(17),
            kind: prism_spatial_ir::graph::ComputeKind::SSM,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 2] },
                    TensorShape { dims: vec![2] },
                    TensorShape { dims: vec![2] },
                    TensorShape { dims: vec![2] },
                ],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_ssm"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 5);
    }

    #[test]
    fn spatial_node_adapter_lowers_reshape_without_kernel() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(14),
            kind: prism_spatial_ir::graph::ComputeKind::Reshape,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![TensorShape { dims: vec![2, 3] }],
                vec![TensorShape { dims: vec![3, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels.is_empty());
        assert!(artifacts.is_empty());
        assert_eq!(capture.graph.ops.last().unwrap().shape, vec![3, 2]);
    }

    #[test]
    fn compiled_program_dispatches_conv2d_artifact() {
        let mut graph = TinyGraph::default();
        let x = graph.add(
            UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 1, 3, 3],
        );
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![1, 1, 2, 2],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![1],
        );
        let conv = graph.add(
            UOpKind::Conv2d {
                batch: 1,
                in_channels: 1,
                height: 3,
                width: 3,
                out_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
                stride: 1,
                padding: 0,
            },
            vec![x, weight, bias],
            vec![1, 1, 2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![conv],
            vec![1, 1, 2, 2],
        );
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch(&BTreeMap::from([
                (
                    "x".into(),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ),
                ("weight".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("bias".into(), vec![0.0]),
            ]))
            .unwrap();
        assert_eq!(result.outputs["out"], vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn compiled_program_dispatches_gather_artifact() {
        let mut graph = TinyGraph::default();
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let gather = graph.add(
            UOpKind::Gather {
                rows: 2,
                vocab: 3,
                features: 2,
            },
            vec![weight, indices],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![gather],
            vec![2, 2],
        );
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch(&BTreeMap::from([
                ("weight".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ("indices".into(), vec![2.0, 0.0]),
            ]))
            .unwrap();
        assert_eq!(result.outputs["out"], vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn compiled_program_dispatches_scatter_artifact() {
        let mut graph = TinyGraph::default();
        let base = graph.add(
            UOpKind::Input {
                name: "base".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let updates = graph.add(
            UOpKind::Input {
                name: "updates".into(),
            },
            vec![],
            vec![2, 2],
        );
        let scatter = graph.add(
            UOpKind::Scatter {
                rows: 3,
                updates: 2,
                features: 2,
            },
            vec![base, indices, updates],
            vec![3, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![scatter],
            vec![3, 2],
        );
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch(&BTreeMap::from([
                ("base".into(), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
                ("indices".into(), vec![1.0, 1.0]),
                ("updates".into(), vec![7.0, 8.0, 9.0, 10.0]),
            ]))
            .unwrap();
        assert_eq!(result.outputs["out"], vec![0.0, 0.0, 9.0, 10.0, 2.0, 2.0]);
    }

    #[test]
    fn compiled_program_dispatches_ssm_artifact() {
        let mut graph = TinyGraph::default();
        let input = graph.add(
            UOpKind::Input {
                name: "input".into(),
            },
            vec![],
            vec![2, 2],
        );
        let decay = graph.add(
            UOpKind::Input {
                name: "decay".into(),
            },
            vec![],
            vec![2],
        );
        let input_gain = graph.add(
            UOpKind::Input {
                name: "input_gain".into(),
            },
            vec![],
            vec![2],
        );
        let output_gain = graph.add(
            UOpKind::Input {
                name: "output_gain".into(),
            },
            vec![],
            vec![2],
        );
        let scan = graph.add(
            UOpKind::Ssm {
                rows: 2,
                features: 2,
            },
            vec![input, decay, input_gain, output_gain],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![scan],
            vec![2, 2],
        );
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch(&BTreeMap::from([
                ("input".into(), vec![1.0, 2.0, 3.0, 4.0]),
                ("decay".into(), vec![0.5, 0.25]),
                ("input_gain".into(), vec![1.0, 1.0]),
                ("output_gain".into(), vec![2.0, 4.0]),
            ]))
            .unwrap();
        assert_eq!(result.outputs["out"], vec![2.0, 8.0, 7.0, 18.0]);
        assert!(result.receipt.replayed);
    }

    #[test]
    fn annotated_spatial_convolution_preserves_stride_and_padding() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(14),
            kind: prism_spatial_ir::graph::ComputeKind::Convolution,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape {
                        dims: vec![1, 1, 3, 3],
                    },
                    TensorShape {
                        dims: vec![1, 1, 2, 2],
                    },
                    TensorShape { dims: vec![1] },
                ],
                vec![TensorShape {
                    dims: vec![1, 1, 2, 2],
                }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.convolution_stride = Some(2);
        metadata.convolution_padding = Some(1);
        let (capture, _) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("prism_conv2d"));
        assert_eq!(
            capture.graph.ops.iter().find_map(|op| match op.kind {
                UOpKind::Conv2d {
                    stride, padding, ..
                } => Some((stride, padding)),
                _ => None,
            }),
            Some((2, 1))
        );
    }

    #[test]
    fn annotated_spatial_normalization_adapter_lowers_layer_norm() {
        use prism_ecs_ir::cimage_types::TensorShape;
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(12),
            kind: prism_spatial_ir::graph::ComputeKind::Normalization,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    TensorShape { dims: vec![1, 2] },
                    TensorShape { dims: vec![2] },
                    TensorShape { dims: vec![2] },
                ],
                vec![TensorShape { dims: vec![1, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::MemoryBound,
        };
        let mut metadata = prism_spatial_ir::graph::NodeMeta::default();
        metadata.normalization_op = Some("layer_norm".into());
        let (capture, artifacts) =
            compile_spatial_node_with_metadata(&node, Some(&metadata), LoweringTarget::Portable)
                .unwrap();
        assert!(capture.kernels[0].source.contains("prism_layer_norm"));
        assert_eq!(artifacts[0].manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn layer_norm_capture_publishes_four_buffer_abi() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1, 2]);
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![2],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![2],
        );
        let norm = graph.add(
            UOpKind::LayerNorm {
                rows: 1,
                features: 2,
                epsilon: 1e-5,
            },
            vec![x, weight, bias],
            vec![1, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![norm],
            vec![1, 2],
        );
        let artifact =
            compile_and_validate_uop_capture(&graph.lower(LoweringTarget::Portable).unwrap())
                .unwrap()
                .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_layer_norm:1:2:0.00001".into())
        );
        assert_eq!(artifact.manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn conv2d_capture_publishes_shape_specialized_abi() {
        let mut graph = TinyGraph::default();
        let x = graph.add(
            UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 1, 3, 3],
        );
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![1, 1, 2, 2],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![1],
        );
        let conv = graph.add(
            UOpKind::Conv2d {
                batch: 1,
                in_channels: 1,
                height: 3,
                width: 3,
                out_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
                stride: 1,
                padding: 0,
            },
            vec![x, weight, bias],
            vec![1, 1, 2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![conv],
            vec![1, 1, 2, 2],
        );
        let artifact =
            compile_and_validate_uop_capture(&graph.lower(LoweringTarget::Portable).unwrap())
                .unwrap()
                .remove(0);
        assert_eq!(
            artifact.manifest.kernels[0].variant,
            KernelVariant::Custom("uop_conv2d:1:1:3:3:1:2:2:1:0".into())
        );
        assert_eq!(artifact.manifest.kernels[0].binding_signature.len(), 4);
    }

    #[test]
    fn spatial_attention_adapter_uses_head_dimension_scale() {
        let node = prism_spatial_ir::SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(9),
            kind: prism_spatial_ir::graph::ComputeKind::Attention,
            shape: prism_spatial_ir::graph::ShapeContract::new(
                vec![
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 2] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 2] },
                    prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 2] },
                ],
                vec![prism_ecs_ir::cimage_types::TensorShape { dims: vec![2, 2] }],
            ),
            intensity: prism_spatial_ir::graph::ComputeIntensity::ComputeBound,
        };
        let (capture, artifacts) = compile_spatial_node(&node, LoweringTarget::Portable).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert!(capture.kernels[0].source.contains("prism_attention"));
    }

    #[test]
    fn spatial_graph_adapter_wires_memory_edges_into_matmul() {
        use prism_ecs_ir::cimage_types::TensorShape;
        use prism_spatial_ir::graph::{
            ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
            SpatialEdge, SpatialEdgeId, SpatialNode,
        };
        let mut spatial = prism_spatial_ir::SpatialGraph::new();
        let a = spatial.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(1),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2, 3] },
                element_size: 4,
                strides: vec![],
            },
        });
        let b = spatial.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(2),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![3, 2] },
                element_size: 4,
                strides: vec![],
            },
        });
        let matmul = spatial.add_node(SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(3),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 3] },
                    TensorShape { dims: vec![3, 2] },
                ],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        spatial.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: a,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![2, 3] }),
        });
        spatial.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: b,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 1,
            shape: Some(TensorShape { dims: vec![3, 2] }),
        });
        let (capture, artifacts) =
            compile_spatial_graph(&spatial, LoweringTarget::Portable).unwrap();
        assert_eq!(capture.graph_op_count, 4);
        assert_eq!(artifacts.len(), 1);
        assert!(capture.kernels[0].source.contains("prism_matmul"));
    }

    #[test]
    fn spatial_graph_adapter_lowers_reshape_as_a_metadata_alias() {
        use prism_ecs_ir::cimage_types::TensorShape;
        use prism_spatial_ir::graph::{
            ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
            SpatialEdge, SpatialEdgeId, SpatialNode,
        };
        let mut spatial = prism_spatial_ir::SpatialGraph::new();
        let input = spatial.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(10),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2, 3] },
                element_size: 4,
                strides: vec![],
            },
        });
        let reshape = spatial.add_node(SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(11),
            kind: ComputeKind::Reshape,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![2, 3] }],
                vec![TensorShape { dims: vec![3, 2] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        spatial.add_edge(SpatialEdge {
            id: SpatialEdgeId(10),
            source: input,
            sink: reshape,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![2, 3] }),
        });

        let (capture, artifacts) =
            compile_spatial_graph(&spatial, LoweringTarget::Portable).unwrap();
        assert!(capture.kernels.is_empty());
        assert!(artifacts.is_empty());
        assert_eq!(capture.graph_op_count, 2);
        assert_eq!(capture.graph.ops.last().unwrap().shape, vec![3, 2]);
    }

    #[test]
    fn spatial_graph_adapter_lowers_validated_custom_sigmoid() {
        use prism_ecs_ir::cimage_types::TensorShape;
        use prism_spatial_ir::graph::{
            ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
            SpatialEdge, SpatialEdgeId, SpatialNode,
        };
        let mut spatial = prism_spatial_ir::SpatialGraph::new();
        let input = spatial.add_node(SpatialNode::Memory {
            id: prism_spatial_ir::SpatialNodeId(30),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2] },
                element_size: 4,
                strides: vec![],
            },
        });
        let sigmoid = spatial.add_node(SpatialNode::Compute {
            id: prism_spatial_ir::SpatialNodeId(31),
            kind: ComputeKind::Custom("sigmoid".into()),
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![2] }],
                vec![TensorShape { dims: vec![2] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        spatial.add_edge(SpatialEdge {
            id: SpatialEdgeId(30),
            source: input,
            sink: sigmoid,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![2] }),
        });

        let (capture, artifacts) =
            compile_spatial_graph(&spatial, LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 2);
        assert_eq!(artifacts.len(), 2);
        assert!(capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("expf")));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_capture_source_compiles_with_xcrun() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Metal).unwrap();
        let artifacts = compile_and_validate_uop_capture(&capture).unwrap();
        assert_eq!(artifacts[0].manifest.kernels[0].backend, BackendKind::Metal);
        assert!(!artifacts[0].payloads[0].binary.is_empty());
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn metal_capture_dispatches_fp32_elementwise_kernel() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Metal).unwrap()).unwrap();
        let input = [(-1.0f32).to_ne_bytes(), 3.0f32.to_ne_bytes()].concat();
        let output = program.dispatch_metal(input, None).unwrap();
        let values: Vec<f32> = output
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![0.0, 3.0]);
    }

    #[test]
    fn compiled_program_dispatches_with_receipt() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-1.0, 4.0]);
        let result = program.dispatch(&inputs).unwrap();
        assert_eq!(result.outputs["y"], vec![0.0, 4.0]);
        assert!(result.receipt.replayed);
    }

    #[test]
    fn compiled_program_dispatches_cpu_artifacts_and_intermediates() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let neg = graph.add(UOpKind::Neg, vec![input], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![neg], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let program =
            UOpCompiledProgram::compile(graph.lower(LoweringTarget::Portable).unwrap()).unwrap();
        let result = program
            .dispatch_cpu(&BTreeMap::from([("x".into(), vec![-2.0, 1.0])]))
            .unwrap();
        assert_eq!(result.outputs["y"], vec![2.0, 0.0]);
        assert!(result.receipt.replayed);
    }

    #[test]
    fn fused_mixed_elementwise_publishes_linear_program() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![2]);
        let add = graph.add(UOpKind::Add, vec![x, y], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![add], vec![2]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let artifacts = compile_uop_capture(&capture).unwrap();
        assert_eq!(
            artifacts[0].manifest.kernels[0].variant,
            KernelVariant::Custom("uop_elementwise_program:add,relu".into())
        );
    }

    #[test]
    fn compile_cache_reuses_identical_capture_artifacts() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let cache = UOpCompileCache::default();
        let first = cache.compile(capture.clone()).unwrap();
        let second = cache.compile(capture).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(first.artifacts, second.artifacts);
    }

    #[test]
    fn compile_cache_supports_backend_artifact_invalidation() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let digest = capture.digest();
        let cache = UOpCompileCache::default();
        cache.compile(capture).unwrap();
        assert!(cache.invalidate(&digest).unwrap());
        assert!(!cache.invalidate(&digest).unwrap());
        assert_eq!(cache.len(), 0);
    }
}
