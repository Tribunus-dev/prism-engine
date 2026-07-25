//! Internal dispatch layer — token-to-logits orchestration.
//!
//! This module owns the canonical authority for taking input tokens and
//! producing logits. It is the *only* path that owns input packing, the
//! AOT-plan / UOp-program / CPU-fallback decision, and the typed
//! per-element decoder for plan output tensors. The public
//! `run_batch` / `run_prefill` / `run_decode` methods in [`super::run`]
//! are thin wrappers around [`dispatch_tokens`].
//!
//! No public API lives here: callers go through [`super::UnifiedRuntime`]
//! and the public run methods. The functions in this module are
//! `pub(super)` so the orchestrator and its submodules can wire the
//! internal flow together.

use std::collections::HashMap;

use prism_ecs_kernel::{
    AccelerateBackend, CpuBackend, KernelBackend, KernelDispatchRequest, KernelVariant, MetalBackend,
};
use prism_spatial_ir::execution_plan::{FusedScheduleStep, TensorBinding};
use prism_spatial_ir::{BufferStorage, ResolvedBuffer, WorkloadScenario};

use crate::uop::UOpCompiledProgram;

use super::super::ane_backend::EmbeddedAneRouteBackend;
use super::super::binding::CImageBindingResolver;
use super::super::certification::cpu_reference_inference;
use super::super::kernel_dispatch::KernelRouteDispatcher;
use super::super::model::RuntimeModel;
use super::super::RuntimeError;
use super::super::{AotScheduler, RoutedExecutor};
use super::{ExecutionMode, UnifiedRuntime};

pub(super) fn dispatch_tokens(
    runtime: &mut UnifiedRuntime,
    input_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let sequence_length = if runtime.mode == ExecutionMode::Batch {
        runtime
            .requested_batch_size
            .map(|batch_size| (input_tokens.len() / batch_size.max(1) as usize).max(1) as u32)
            .unwrap_or(input_tokens.len().max(1) as u32)
    } else {
        input_tokens.len().max(1) as u32
    };
    runtime.last_workload_selection = runtime.workload_profile_for_dispatch(sequence_length).cloned();
    let scenario = WorkloadScenario {
        realtime: matches!(
            runtime.mode,
            ExecutionMode::RealtimePrefill | ExecutionMode::RealtimeDecode
        ),
        batch_size: if matches!(
            runtime.mode,
            ExecutionMode::RealtimePrefill | ExecutionMode::RealtimeDecode
        ) {
            1
        } else {
            runtime.requested_batch_size.unwrap_or(1).max(1)
        },
        sequence_length: sequence_length.max(1),
    };

    if runtime
        .active_execution_plan()
        .is_some_and(|plan| !plan.fused_steps.is_empty())
    {
        if let Ok(output) = dispatch_heterogeneous_plan_for_tokens(runtime, &scenario, input_tokens)
        {
            return Ok(output);
        }
    }

    if let Some(program) = selected_uop_program(runtime, sequence_length) {
        if uop_program_accepts_tokens(runtime, program, input_tokens.len()) {
            return dispatch_uop_tokens(runtime, program, input_tokens);
        }
    }
    if runtime.backend.is_none() {
        // Keep the runtime usable on hosts without an attached hardware
        // backend. The same canonical packing and CPU kernel contracts
        // used by certification provide the reference execution path.
        return cpu_reference_inference(&runtime.model, input_tokens);
    }
    // WAIVER: `runtime.backend` was checked for `is_none()` on the
    // previous branch (the `if runtime.backend.is_none()` block above
    // returns `cpu_reference_inference`). The `expect` here is the
    // structurally-`Some` arm of the same branch. Pre-existing pattern
    // that survived the orchestrator decomposition.
    let backend = runtime.backend.as_ref().expect("backend checked above");
    let name = runtime
        .model
        .kernel_descriptors
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| RuntimeError::KernelNotFound("no described kernels".into()))?;
    let artifact = runtime.model.kernel_artifact(&name)?;
    let descriptor = &artifact.payloads[0].descriptor;
    let token_bytes: Vec<u8> = input_tokens
        .iter()
        .flat_map(|t| (*t as f32).to_ne_bytes())
        .collect();
    let mut tensor_names = runtime.model.tensors.keys().cloned().collect::<Vec<_>>();
    tensor_names.sort();
    let inputs = match &descriptor.variant {
        KernelVariant::FP16GEMV => {
            let weights_name = tensor_names
                .first()
                .ok_or_else(|| RuntimeError::TensorNotFound("no weights".into()))?;
            // WAIVER: `weights_name` was just resolved from
            // `tensor_names.first()`, and the previous line has already
            // returned `TensorNotFound` if `tensor_names` was empty.
            // The `tensors.get` lookup is infallible by construction.
            // Pre-existing — survived the orchestrator decomposition.
            let weights = runtime.model.tensors.get(weights_name).unwrap();
            vec![weights.clone(), token_bytes]
        }
        KernelVariant::QuantizedGEMV => {
            let weights_name = tensor_names.first().ok_or_else(|| {
                RuntimeError::TensorNotFound("no quantized GEMV weights".into())
            })?;
            let weights = runtime.model.tensors.get(weights_name).ok_or_else(|| {
                RuntimeError::TensorNotFound(format!("missing tensor {weights_name:?}"))
            })?;
            let record = runtime.model.tensor_records.get(weights_name).ok_or_else(|| {
                RuntimeError::InvalidCImage("quantized GEMV weights have no shape".into())
            })?;
            let dims = [record.dim_m, record.dim_n];
            vec![
                weights.clone(),
                input_tokens
                    .iter()
                    .flat_map(|token| (*token as f32).to_ne_bytes())
                    .collect(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ]
        }
        KernelVariant::FP16Matmul => {
            let a_name = tensor_names
                .first()
                .ok_or_else(|| RuntimeError::TensorNotFound("no matrix A".into()))?;
            let b_name = tensor_names
                .get(1)
                .ok_or_else(|| RuntimeError::TensorNotFound("no matrix B".into()))?;
            let a_shape = runtime.model.tensor_records.get(a_name).ok_or_else(|| {
                RuntimeError::InvalidCImage("matrix A has no shape".into())
            })?;
            let b_shape = runtime.model.tensor_records.get(b_name).ok_or_else(|| {
                RuntimeError::InvalidCImage("matrix B has no shape".into())
            })?;
            let dims = [a_shape.dim_m, b_shape.dim_n, a_shape.dim_n];
            vec![
                runtime.model.tensors[a_name].clone(),
                runtime.model.tensors[b_name].clone(),
                dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            ]
        }
        KernelVariant::INT8Tile640 => {
            let weights_name = tensor_names
                .first()
                .ok_or_else(|| RuntimeError::TensorNotFound("no INT8 weights".into()))?;
            let scales_name = tensor_names.get(1).ok_or_else(|| {
                RuntimeError::TensorNotFound("no INT8 weight scales".into())
            })?;
            let input_scale_name = tensor_names.get(2).ok_or_else(|| {
                RuntimeError::TensorNotFound("no INT8 input scale".into())
            })?;
            let record = runtime.model.tensor_records.get(weights_name).ok_or_else(|| {
                RuntimeError::InvalidCImage("INT8 weights have no shape".into())
            })?;
            let dims = [record.dim_n, record.dim_m];
            vec![
                runtime.model.tensors[weights_name].clone(),
                input_tokens
                    .iter()
                    .map(|token| *token as i8 as u8)
                    .collect(),
                runtime.model.tensors[scales_name].clone(),
                runtime.model.tensors[input_scale_name].clone(),
                dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            ]
        }
        KernelVariant::NF4Tile640 => {
            let weights_name = tensor_names
                .first()
                .ok_or_else(|| RuntimeError::TensorNotFound("no NF4 weights".into()))?;
            let scales_name = tensor_names.get(1).ok_or_else(|| {
                RuntimeError::TensorNotFound("no NF4 weight scales".into())
            })?;
            let biases_name = tensor_names.get(2).ok_or_else(|| {
                RuntimeError::TensorNotFound("no NF4 weight biases".into())
            })?;
            let record = runtime.model.tensor_records.get(weights_name).ok_or_else(|| {
                RuntimeError::InvalidCImage("NF4 weights have no shape".into())
            })?;
            let tiles = (record.dim_n as usize).div_ceil(640);
            let groups = tiles * 5;
            let expected_codes = record.dim_m as usize * tiles * 320;
            let expected_metadata = record.dim_m as usize * groups * 4;
            if runtime.model.tensors[weights_name].len() != expected_codes
                || runtime.model.tensors[scales_name].len() != expected_metadata
                || runtime.model.tensors[biases_name].len() != expected_metadata
            {
                return Err(RuntimeError::InvalidCImage(
                    "NF4 Tile640 payload or group metadata is truncated".into(),
                ));
            }
            let dims = [record.dim_n, record.dim_m];
            vec![
                runtime.model.tensors[weights_name].clone(),
                input_tokens
                    .iter()
                    .flat_map(|token| (*token as f32).to_ne_bytes())
                    .collect(),
                runtime.model.tensors[scales_name].clone(),
                runtime.model.tensors[biases_name].clone(),
                dims.iter().flat_map(|value| value.to_ne_bytes()).collect(),
            ]
        }
        KernelVariant::TernaryTile640(_) => {
            let name = tensor_names
                .first()
                .ok_or_else(|| RuntimeError::TensorNotFound("no ternary weights".into()))?;
            let record = runtime.model.tensor_records.get(name).ok_or_else(|| {
                RuntimeError::InvalidCImage("ternary weights have no shape".into())
            })?;
            let input_half: Vec<u8> = input_tokens
                .iter()
                .flat_map(|t| half::f16::from_f32(*t as f32).to_le_bytes())
                .collect();
            let pages = (record.dim_n as usize).div_ceil(640);
            let packed_len = record.dim_m as usize * pages * 4;
            let page_len = record.dim_m as usize * pages * 2;
            let lane_len = record.dim_m as usize * pages;
            let packed = &runtime.model.tensors[name];
            if packed.len() < packed_len + page_len + lane_len {
                return Err(RuntimeError::InvalidCImage(
                    "ternary payload is truncated".into(),
                ));
            }
            let dims = [record.dim_n, record.dim_m];
            vec![
                packed[..packed_len].to_vec(),
                input_half,
                packed[packed_len..packed_len + page_len].to_vec(),
                packed[packed_len + page_len..packed_len + page_len + lane_len].to_vec(),
                dims.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            ]
        }
        variant => {
            return Err(RuntimeError::UnsupportedMode(format!(
                "runtime input packing for {variant:?} is not implemented"
            )))
        }
    };
    let output = backend
        .dispatch(&KernelDispatchRequest {
            artifact,
            inputs,
            bindings: vec![],
        })
        .map_err(|e| RuntimeError::BackendError(e.to_string()))?
        .outputs
        .into_iter()
        .next()
        .ok_or_else(|| RuntimeError::ExecutionFailed("backend returned no output".into()))?;
    if output.len() % 4 != 0 {
        return Err(RuntimeError::ExecutionFailed(
            "backend output is not FP32".into(),
        ));
    }
    // WAIVER: backend output is f32-aligned (the
    // `output.len() % 4 != 0` check above returns early on
    // mis-alignment). The `try_into().unwrap()` is structurally
    // infallible. Pre-existing — survived the orchestrator decomposition.
    Ok(output
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

pub(super) fn dispatch_heterogeneous_plan_for_tokens(
    runtime: &UnifiedRuntime,
    scenario: &WorkloadScenario,
    input_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let plan = runtime
        .active_execution_plan()
        .ok_or_else(|| RuntimeError::UnsupportedMode("CImage has no AOT execution plan".into()))?
        .try_specialize_for_workload(*scenario)
        .map_err(RuntimeError::UnsupportedMode)?;
    if plan.fused_steps.is_empty() {
        return Err(RuntimeError::UnsupportedMode(
            "AOT execution plan is empty for this workload".into(),
        ));
    }
    let mut resolver = CImageBindingResolver {
        model: &runtime.model,
        runtime_outputs: preseed_plan_inputs(runtime, &plan.fused_steps, input_tokens)?,
    };

    let mut ane = EmbeddedAneRouteBackend {
        runtime,
        outputs: HashMap::new(),
    };
    let accelerate = AccelerateBackend;
    let metal = MetalBackend::new();
    let cpu = CpuBackend;
    let routes = KernelRouteDispatcher {
        model: &runtime.model,
        ane: &mut ane,
        accelerate: &accelerate,
        metal: &metal,
        cpu: &cpu,
        xdna: None,
    };
    let mut routed = RoutedExecutor { routes };
    AotScheduler::replay_resolved(&plan, &mut resolver, &mut routed)
        .map_err(RuntimeError::ExecutionFailed)?;
    extract_plan_logits(&plan.fused_steps, resolver.runtime_outputs).map_err(RuntimeError::InvalidCImage)
}

pub(super) fn preseed_plan_inputs(
    runtime: &UnifiedRuntime,
    steps: &[FusedScheduleStep],
    input_tokens: &[u32],
) -> Result<HashMap<String, ResolvedBuffer>, RuntimeError> {
    let mut seeded: HashMap<String, ResolvedBuffer> = HashMap::new();
    for step in steps {
        for binding in &step.input_tensors {
            if runtime.model.tensors.contains_key(&binding.name) {
                continue;
            }
            if seeded.contains_key(&binding.name) {
                continue;
            }
            let payload = decode_dispatch_tokens(binding, input_tokens).map_err(|error| {
                RuntimeError::InvalidCImage(format!(
                    "failed to bind runtime input '{}' for heterogeneous plan: {error}",
                    binding.name
                ))
            })?;
            let element_size = match binding.element_type.as_str() {
                "fp32" | "int32" => 4usize,
                "fp16" => 2usize,
                "int8" => 1usize,
                _ => {
                    return Err(RuntimeError::UnsupportedMode(format!(
                        "unsupported plan input element type {:?}",
                        binding.element_type
                    )))
                }
            };
            let expected_elements = if binding.shape.is_empty() {
                input_tokens.len()
            } else {
                binding
                    .shape
                    .iter()
                    .fold(1u64, |size, extent| size.saturating_mul(*extent))
                    as usize
            };
            if payload.len() != expected_elements.saturating_mul(element_size) {
                return Err(RuntimeError::InvalidCImage(format!(
                    "plan input '{}' payload size mismatch: has {} bytes, expected {} bytes",
                    binding.name,
                    payload.len(),
                    expected_elements.saturating_mul(element_size)
                )));
            }
            seeded.insert(
                binding.name.clone(),
                ResolvedBuffer {
                    name: binding.name.clone(),
                    element_type: binding.element_type.clone(),
                    region: "unified-memory".into(),
                    byte_length: payload.len(),
                    zero_copy: false,
                    file_offset: None,
                    storage: BufferStorage::RuntimeOwned,
                    shape: binding.shape.clone(),
                    payload: Some(payload),
                },
            );
        }
    }
    Ok(seeded)
}

pub(super) fn extract_plan_logits(
    _steps: &[FusedScheduleStep],
    outputs: HashMap<String, ResolvedBuffer>,
) -> Result<Vec<f32>, String> {
    if outputs.is_empty() {
        return Err("AOT plan produced no runtime outputs".into());
    }
    let mut final_step: Option<&FusedScheduleStep> = None;
    for step in _steps {
        if final_step
            .as_ref()
            .is_none_or(|best| step.step_id > best.step_id)
        {
            final_step = Some(step);
        }
    }
    let Some(step) = final_step else {
        return Err("AOT plan has no execution steps to derive output tensor".into());
    };
    let mut logits = Vec::new();
    for output in &step.output_tensors {
        if let Some(buffer) = outputs.get(&output.name) {
            let payload = buffer
                .payload
                .as_ref()
                .ok_or_else(|| format!("AOT output buffer '{}' has no payload", output.name))?;
            logits.extend(decode_tensor_payload(
                payload,
                output.element_type.as_str(),
                output.name.as_str(),
            )?);
        }
    }
    if logits.is_empty() {
        return Err(format!(
            "AOT plan produced no output from step {}",
            step.step_id
        ));
    }
    Ok(logits)
}

pub fn selected_uop_program(
    runtime: &UnifiedRuntime,
    sequence_length: u32,
) -> Option<&UOpCompiledProgram> {
    let fallback = runtime.model.uop_program.as_ref();
    let Some(plan) = runtime.active_execution_plan() else {
        return fallback;
    };
    let realtime = matches!(
        runtime.mode,
        ExecutionMode::RealtimePrefill | ExecutionMode::RealtimeDecode
    );
    let batch_size = if realtime {
        1
    } else if let Some(batch_size) = runtime.requested_batch_size {
        batch_size
    } else {
        plan.batch_size.max(1)
    };
    let scenario = WorkloadScenario {
        realtime,
        batch_size,
        sequence_length: sequence_length.max(1),
    };
    let measured_strategy = runtime.measured_strategy_for_scenario(scenario);
    if let Some(strategy_id) = measured_strategy {
        return runtime
            .model
            .uop_strategy_programs
            .get(strategy_id)
            .or(fallback);
    }
    let Some(strategy) = plan.selected_workload_strategy(scenario) else {
        return fallback;
    };
    runtime
        .model
        .uop_strategy_programs
        .get(strategy.stable_id())
        .or(fallback)
}

pub(super) fn dispatch_uop_tokens(
    runtime: &UnifiedRuntime,
    program: &UOpCompiledProgram,
    input_tokens: &[u32],
) -> Result<Vec<f32>, RuntimeError> {
    let mut inputs = std::collections::BTreeMap::new();
    for op in &program.capture.graph.ops {
        let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
            continue;
        };
        let values = if let Some(payload) = runtime.model.tensors.get(name) {
            if payload.len() % std::mem::size_of::<f32>() != 0 {
                return Err(RuntimeError::InvalidCImage(format!(
                    "UOp tensor input {name:?} is not FP32-aligned"
                )));
            }
            // WAIVER: f32-aligned 4-byte chunks produced by
            // `chunks_exact` after the `payload.len() % 4 == 0` check.
            // Structurally infallible. Pre-existing — survived the
            // orchestrator decomposition.
            payload
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        } else {
            input_tokens.iter().map(|token| *token as f32).collect()
        };
        let expected = op
            .shape
            .iter()
            .try_fold(1usize, |size, dimension| {
                size.checked_mul(*dimension as usize)
            })
            .ok_or_else(|| {
                RuntimeError::InvalidCImage(format!("UOp input {name:?} shape overflows"))
            })?;
        if values.len() != expected {
            return Err(RuntimeError::ExecutionFailed(format!(
                "UOp input {name:?} has {} values, expected {expected}",
                values.len()
            )));
        }
        inputs.insert(name.clone(), values);
    }
    let result = program
        .dispatch(&inputs)
        .map_err(RuntimeError::ExecutionFailed)?;
    result
        .outputs
        .into_values()
        .next()
        .ok_or_else(|| RuntimeError::ExecutionFailed("UOp program produced no outputs".into()))
}

pub(super) fn uop_program_accepts_tokens(
    runtime: &UnifiedRuntime,
    program: &UOpCompiledProgram,
    token_count: usize,
) -> bool {
    let token_inputs = program.capture.graph.ops.iter().filter(|op| {
        let prism_spatial_ir::UOpKind::Input { name } = &op.kind else {
            return false;
        };
        !runtime.model.tensors.contains_key(name)
    });
    let mut saw_token_input = false;
    let all_match = token_inputs.clone().all(|op| {
        saw_token_input = true;
        op.shape.iter().try_fold(1usize, |size, dimension| {
            size.checked_mul(*dimension as usize)
        }) == Some(token_count)
    });
    saw_token_input && all_match
}

// ── Local helpers (dispatch-only) ───────────────────────────────────────

pub(super) fn argmax_token(logits: &[f32]) -> u32 {
    let mut best_index: usize = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in logits.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best_index = index;
        }
    }
    best_index as u32
}

pub(super) fn decode_dispatch_tokens(
    binding: &TensorBinding,
    input_tokens: &[u32],
) -> Result<Vec<u8>, String> {
    match binding.element_type.as_str() {
        "fp32" => Ok(input_tokens
            .iter()
            .flat_map(|token| (*token as f32).to_le_bytes())
            .collect()),
        "fp16" => Ok(input_tokens
            .iter()
            .flat_map(|token| half::f16::from_f32(*token as f32).to_le_bytes())
            .collect()),
        "int32" => Ok(input_tokens
            .iter()
            .flat_map(|token| (*token as i32).to_le_bytes())
            .collect()),
        "int8" => Ok(input_tokens
            .iter()
            .map(|token| *token as i8 as u8)
            .collect()),
        other => Err(format!("unsupported plan input element type '{other}'")),
    }
}

pub(super) fn decode_tensor_payload(
    payload: &[u8],
    element_type: &str,
    name: &str,
) -> Result<Vec<f32>, String> {
    // WAIVER: every `try_into().unwrap()` below is guarded by the
    // matching `chunks_exact(N)` where N is the element byte size
    // (4 for f32 / i32, 2 for f16). The chunk slices are infallible
    // casts to `[u8; N]`. Pre-existing — survived the orchestrator
    // decomposition. Conversion to a checked error would require
    // passing a length context through the AOT plan loader, which is
    // out of scope for the runtime decomposition.
    match element_type {
        "fp32" => Ok(payload
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()),
        "fp16" => Ok(payload
            .chunks_exact(2)
            .map(|chunk| half::f16::from_le_bytes(chunk.try_into().unwrap()).to_f32())
            .collect()),
        "int32" => Ok(payload
            .chunks_exact(4)
            .map(|chunk| (i32::from_le_bytes(chunk.try_into().unwrap())) as f32)
            .collect()),
        "int8" => Ok(payload.iter().map(|value| (*value as i8) as f32).collect()),
        other => Err(format!(
            "unsupported AOT output element type '{other}' for '{name}'"
        )),
    }
}

// ── Touch the imports so dead-code lints don't fail on the inner types ──
// (RuntimeModel is used through the UnifiedRuntime's `model` field, but
// listing it here keeps the import path live in case the dispatch layer
// is later extracted to a free-standing helper.)
#[allow(dead_code)]
fn _model_anchor(_m: &RuntimeModel) {}
