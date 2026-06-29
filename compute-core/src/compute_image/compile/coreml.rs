//! Core ML subgraph decomposition pass.
//!
//! Splits transformer blocks into ANE subgraphs and CPU lanes during
//! ComputeImage build. Ops where Accelerate benchmarks show <10µs are
//! pulled out of the Core ML subgraph and run on the CPU lane concurrently
//! over the shared unified arena.
//!
//! ## Workflow
//!
//! 1. [`candidate_subgraphs`] lists the known transformer subgraphs (MLP
//!    block, RMS norm + QKV projections, output projection, etc.).
//! 2. [`decompose_subgraph`] consults the [`ConcurrencyPlan`] lane assignments
//!    to decide which ops belong on the ANE (Core ML) vs the CPU (Accelerate).
//!    Ops with no plan data fall back to a tiny-op heuristic (<10µs).
//! 3. [`compile_subgraph`] builds a MIL program for the ANE-bound ops,
//!    serialises it as an `.mlpackage`, and runs `xcrun coremlcompiler`
//!    to produce a `.mlmodelc` ready for IOSurface-backed inference.

use coreml_proto::proto::mil_spec;
use std::collections::HashMap;
use std::path::Path;
use mil_spec::ValueType;
use crate::compute_image::hw_assessment::ConcurrencyPlan;
use crate::compute_image::subgraph_mil;
use crate::coreml_pipeline;
use crate::mil_builder::MilBuilder;
use crate::mlpackage::{self, ModelMeta};

// ── Known subgraph catalog ──────────────────────────────────────────────────

/// A decomposed subgraph assignment produced by [`decompose_subgraph`].
#[derive(Clone, Debug)]
pub struct SubgraphDecomposition {
    pub subgraph_name: String,
    /// Ops assigned to Core ML (ANE).
    pub coreml_ops: Vec<String>,
    /// Ops assigned to Accelerate (CPU).
    pub accelerate_ops: Vec<String>,
    /// Whether this subgraph was actually compiled.
    pub compiled: bool,
    /// Path to the compiled .mlmodelc (relative to ComputeImage).
    pub modelc_path: Option<String>,
}

/// Candidate transformer subgraphs for Core ML compilation.
///
/// Each entry pairs a human-readable subgraph name with the list of op types
/// it encompasses in execution order.  The decomposition pass uses these as
/// input templates; individual ops may be promoted to the CPU lane when the
/// concurrency plan indicates Accelerate can handle them faster.
pub fn candidate_subgraphs() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "mlp_block",
            vec![
                "gate_proj",
                "silu_mul",
                "up_proj",
                "mul",
                "down_proj",
                "residual_add",
            ],
        ),
        (
            "rmsnorm_qkv",
            vec!["rms_norm", "q_proj", "k_proj", "v_proj", "rope"],
        ),
        ("output_proj", vec!["lm_head"]),
        (
            "ffn_output",
            vec![
                "gate_proj",
                "silu_mul",
                "up_proj",
                "mul",
                "down_proj",
                "rms_norm",
                "lm_head",
            ],
        ),
        ("qkv_bundle", vec!["q_proj", "k_proj", "v_proj"]),
    ]
}

/// Ops considered "tiny" — Accelerate finishes them in well under 10µs so
/// pulling them out of the Core ML subgraph reduces ANE dispatch overhead
/// without adding meaningful CPU cost.
const TINY_OPS: &[&str] = &["residual_add", "rms_norm", "silu_mul"];

// ── Decomposition ───────────────────────────────────────────────────────────

/// Decompose a transformer subgraph into Core ML and Accelerate assignments.
///
/// The decision is driven by the [`ConcurrencyPlan`] produced during hardware
/// assessment:
/// * Ops the plan assigned to `"accelerate_cpu"` go to the CPU lane.
/// * Ops with no plan data are classified by the tiny-op heuristic (ops in
///   [`TINY_OPS`] go to Accelerate; everything else goes to Core ML).
///
/// Both lane assignments execute concurrently over the same
/// [`UnifiedExecutionArena`](crate::backend::unified_arena::UnifiedExecutionArena).
pub fn decompose_subgraph(
    name: &str,
    ops: &[&str],
    concurrency_plan: &ConcurrencyPlan,
) -> SubgraphDecomposition {
    let mut coreml_ops = Vec::new();
    let mut accelerate_ops = Vec::new();

    // Build a fast lookup: op -> estimated latency on accelerate_cpu lane.
    let accelerate_latencies: HashMap<&str, u64> = concurrency_plan
        .concurrent_assignments
        .iter()
        .filter(|a| a.lane == "accelerate_cpu")
        .flat_map(|a| {
            let lat = a.estimated_latency_ns;
            a.ops.iter().map(move |op| (op.as_str(), lat))
        })
        .collect();

    for op in ops {
        // If the concurrency plan says this op lives on accelerate_cpu, trust it.
        if concurrency_plan
            .concurrent_assignments
            .iter()
            .filter(|a| a.lane == "accelerate_cpu")
            .any(|a| a.ops.iter().any(|o| o == op))
        {
            accelerate_ops.push(op.to_string());
            continue;
        }

        // Fallback heuristic when the concurrency plan has no lane assignment.
        // Ops the Accelerate micro-benchmarks complete in <10µs are candidates
        // for the CPU lane.
        let plan_latency_ns = accelerate_latencies.get(op).copied();
        let is_tiny = match plan_latency_ns {
            Some(ns) => ns < 10_000, // <10µs → Accelerate
            None => TINY_OPS.contains(op),
        };

        if is_tiny {
            accelerate_ops.push(op.to_string());
        } else {
            coreml_ops.push(op.to_string());
        }
    }

    SubgraphDecomposition {
        subgraph_name: name.to_string(),
        coreml_ops,
        accelerate_ops,
        compiled: false,
        modelc_path: None,
    }
}

// ── Subgraph compilation ────────────────────────────────────────────────────

/// Compile a decomposed subgraph against the ANE via coremlcompiler.
///
/// Builds a MIL program from the op list, serialises it as an `.mlpackage`,
/// and invokes `xcrun coremlcompiler compile` to produce a `.mlmodelc`
/// bundle.  The returned path is valid for
/// [`CoreMlModel::load`](crate::coreml_bridge::CoreMlModel::load).
///
/// ## Shape limitations
///
/// Core ML requires fixed input shapes at compile time.  Callers MUST
/// provide every dimension that affects the subgraph's intermediate shapes
/// via `input_shapes`.  Dynamic subgraphs (variable-batch, variable-sequence)
/// are not yet supported — the MIL builder hard-codes the provided shapes.
///
/// ## Errors
///
/// Returns `Err` when MIL construction fails (unknown op type, SSA
/// verification error), mlpackage serialisation fails, or `xcrun
/// coremlcompiler` exits with a non-zero status.
pub fn compile_subgraph(
    name: &str,
    _coreml_ops: &[String],
    input_shapes: &HashMap<String, Vec<i64>>,
    weights: &HashMap<String, Vec<f32>>,
    output_dir: &Path,
) -> Result<String, String> {
    // Extract dimensions from input_shapes.
    let hidden_dim = input_shapes
        .get("hidden")
        .and_then(|s| s.first().copied())
        .unwrap_or(4096) as u32;
    let intermediate_dim = input_shapes
        .get("intermediate")
        .and_then(|s| s.first().copied())
        .unwrap_or(11008) as u32;
    let n_heads = input_shapes
        .get("num_attention_heads")
        .and_then(|s| s.first().copied())
        .unwrap_or(32) as u32;
    let n_kv_heads = input_shapes
        .get("num_key_value_heads")
        .and_then(|s| s.first().copied())
        .unwrap_or(n_heads as i64) as u32;
    let head_dim = input_shapes
        .get("head_dim")
        .and_then(|s| s.first().copied())
        .unwrap_or((hidden_dim / n_heads) as i64) as u32;
    let vocab_dim = input_shapes
        .get("vocab")
        .and_then(|s| s.first().copied())
        .unwrap_or(32000) as u32;

    // Phase 1: Build MIL program using the correct concrete builder.
    // Dispatch by subgraph name rather than interpreting op-name strings.
    let program = match name {
        "matmul" => {
            let k = input_shapes
                .get("k")
                .and_then(|s| s.first().copied())
                .unwrap_or(64) as u32;
            let n = input_shapes
                .get("n")
                .and_then(|s| s.first().copied())
                .unwrap_or(64) as u32;
            let weight_values = weights
                .get("weight_values")
                .map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_matmul_mil("x", "w", "out", 1, k, n, weight_values)?
        }
        "mlp_block" => {
            let gate_w = weights
                .get("gate_w")
                .map_or(&[] as &[f32], |v| v.as_slice());
            let up_w = weights.get("up_w").map_or(&[] as &[f32], |v| v.as_slice());
            let down_w = weights
                .get("down_w")
                .map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_mlp_block_mil("x",
            hidden_dim,
            intermediate_dim,
            gate_w,
            up_w,
            down_w,)?
        }
        "rmsnorm_qkv" => {
            let rms_w = weights.get("rms_w").map_or(&[] as &[f32], |v| v.as_slice());
            let q_w = weights.get("q_w").map_or(&[] as &[f32], |v| v.as_slice());
            let k_w = weights.get("k_w").map_or(&[] as &[f32], |v| v.as_slice());
            let v_w = weights.get("v_w").map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_rmsnorm_qkv_mil("x", hidden_dim, n_heads, n_kv_heads, head_dim, rms_w, q_w, k_w, v_w,)?
        }
        "output_proj" => {
            let weight_values = weights
                .get("weight_values")
                .map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_output_proj_mil("x", hidden_dim, vocab_dim, weight_values)?
        }
        "ffn_output" => {
            let gate_w = weights
                .get("gate_w")
                .map_or(&[] as &[f32], |v| v.as_slice());
            let up_w = weights.get("up_w").map_or(&[] as &[f32], |v| v.as_slice());
            let down_w = weights
                .get("down_w")
                .map_or(&[] as &[f32], |v| v.as_slice());
            let lm_head_w = weights
                .get("lm_head_w")
                .map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_ffn_output_mil("x",
            hidden_dim,
            intermediate_dim,
            vocab_dim,
            gate_w,
            up_w,
            down_w,
            lm_head_w,)?
        }
        "qkv_bundle" => {
            let q_w = weights.get("q_w").map_or(&[] as &[f32], |v| v.as_slice());
            let k_w = weights.get("k_w").map_or(&[] as &[f32], |v| v.as_slice());
            let v_w = weights.get("v_w").map_or(&[] as &[f32], |v| v.as_slice());
            subgraph_mil::build_qkv_bundle_mil("x", hidden_dim, n_heads, n_kv_heads, head_dim, q_w, k_w, v_w,)?
        }
        _ => {
            return Err(format!(
                "unknown subgraph kind '{}' — no MIL builder registered",
                name
            ));
        }
    };

    // ── ANE Hardware Validation ───────────────────────────────────
    // Before serializing, verify the compiled program satisfies all
    // ANE hardware invariants.  If this fails, the .mlmodelc would
    // silently fall back to CPU/GPU — catch it here at compile time.
    validate_ane_program(&program).map_err(|e| {
        format!("ANE validation failed for subgraph '{}': {}", name, e)
    })?;

    // Phase 2: Write .mlpackage
    let mlpackage_dir = output_dir.join(format!("{}.mlpackage", name));
    let _ = std::fs::create_dir_all(&mlpackage_dir);

    // We need to know what SSA names the builder produces for outputs.
    let output_names = subgraph_mil::subgraph_output_names(name);
    // Use the first output name as the model output (or a fallback).
    let output_name = output_names.first().unwrap_or(&"output").to_string();

    let meta = ModelMeta {
        model_name: format!("tribunus-subgraph-{}", name),
        function_name: name.to_string(),
        short_description: format!("Core ML subgraph: {}", name),
        version: "1.0".into(),
        author: "Tribunus Compute".into(),
        output_name: output_name.clone(),
        inputs: vec![
            ("x".into(), vec![1, hidden_dim as i64]),
            ("residual".into(), vec![1, hidden_dim as i64]),
        ],
        outputs: vec![(output_name, vec![1, hidden_dim as i64])],
    };

    let written_path = mlpackage::write_mlpackage(program, &mlpackage_dir, &meta)
        .map_err(|e| format!("mlpackage write failed: {e}"))?;

    // Phase 3: Compile via coremlcompiler
    let receipt = coreml_pipeline::compile_mlpackage(
        &written_path,
        output_dir,
        name,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("coremlcompiler failed: {e}"))?;

    Ok(receipt.compiled_modelc_path)
}

// ── MIL program construction ────────────────────────────────────────────────

/// Build a MIL [`mil_spec::Program`] from the given list of op types.
///
/// Each op type is translated to its corresponding `MilBuilder` method:
///
/// | Op type          | MIL operation                          |
/// |------------------|----------------------------------------|
/// | `gate_proj`      | `matmul(input, weight_gate)`           |
/// | `up_proj`        | `matmul(input, weight_up)`             |
/// | `down_proj`      | `matmul(input, weight_down)`           |
/// | `q_proj`         | `matmul(input, weight_q)`              |
/// | `k_proj`         | `matmul(input, weight_k)`              |
/// | `v_proj`         | `matmul(input, weight_v)`              |
/// | `lm_head`        | `matmul(input, weight_lm_head)`        |
/// | `mul`            | `mul(input_a, input_b)`                |
/// | `silu_mul`       | silu then mul (two ops)                |
/// | `residual_add`   | `add(input, residual)`                 |
/// | `rms_norm`       | raw MIL operation stub                 |
/// | `rope`           | identity pass-through stub             |
///
/// When an op type is not recognised an error is returned.
#[allow(unused_variables)]
#[allow(dead_code)]
fn build_mil_program(
    name: &str,
    ops: &[String],
    input_shapes: &HashMap<String, Vec<i64>>,
) -> Result<coreml_proto::proto::mil_spec::Program, String> {
    let hidden_dim = input_shapes
        .get("hidden")
        .and_then(|s| s.first().copied())
        .unwrap_or(4096);
    let intermediate_dim = input_shapes
        .get("intermediate")
        .and_then(|s| s.first().copied())
        .unwrap_or(11008);

    let mut builder = MilBuilder::new(name)
        .input("x", mil_spec::DataType::Float32, &[1, hidden_dim])
        .input("residual", mil_spec::DataType::Float32, &[1, hidden_dim])
        .set_opset("CoreML9");

    // Register constant weight placeholders for every projection in the op list.
    // Real weights are supplied at inference time through IOSurface-backed
    // arena views; zero-filled buffers of the correct shape ensure the MIL
    // graph type-checks during compilation.
    let weight_dims: HashMap<&str, Vec<i64>> = [
        ("gate_proj", vec![hidden_dim, intermediate_dim]),
        ("up_proj", vec![hidden_dim, intermediate_dim]),
        ("down_proj", vec![intermediate_dim, hidden_dim]),
        ("q_proj", vec![hidden_dim, hidden_dim]),
        ("k_proj", vec![hidden_dim, hidden_dim]),
        ("v_proj", vec![hidden_dim, hidden_dim]),
        ("lm_head", vec![hidden_dim, hidden_dim]),
    ]
    .into_iter()
    .collect();

    for (&wt, dims) in &weight_dims {
        if ops.iter().any(|o| o == wt) {
            let n: usize = dims.iter().map(|d| *d as usize).product();
            builder = builder.const_f32(wt, &vec![0.0f32; n], dims);
        }
    }

    if ops.iter().any(|o| o == "rms_norm") {
        let norm_shape = vec![hidden_dim];
        let n: usize = hidden_dim as usize;
        builder = builder.const_f32("norm_weight", &vec![1.0f32; n], &norm_shape);
    }

    // Walk ops in order, chaining SSA values.
    let mut current = "x".to_string();
    for op in ops {
        match op.as_str() {
            "gate_proj" | "up_proj" | "down_proj" | "q_proj" | "k_proj" | "v_proj" | "lm_head" => {
                builder = builder.matmul(&current, op);
                // MilBuilder assigns a fresh SSA name internally; we need
                // to recover it.  The builder doesn't expose the last-assigned
                // name, so we approximate by tracking that the latest operation
                // with a known pattern produced an SSA value named
                // "{op_type}_{counter}".  For compilation stubs this is
                // sufficient — real wiring requires the full builder state.
                current = format!("matmul_{}", op);
            }
            "silu_mul" => {
                // MIL has no fused SiLU-mul.  Emit as two ops:
                //   s = silu(x)
                //   out = mul(s, second_input)
                // For a standalone silu we emit via the raw Operation API.
                // The exact protobuf shape depends on mil_spec::Operation
                // construction which varies by opset version.
                current = format!("silu_mul_{}", op);
            }
            "mul" => {
                // Elementwise multiply — uses the last value twice (input * gate).
                builder = builder.mul(&current, &current);
                current = format!("elem_mul");
            }
            "residual_add" => {
                builder = builder.add(&current, "residual");
                current = format!("residual_add");
            }
            "rms_norm" => {
                // RMS norm is not exposed as a MilBuilder helper yet.
                // Emit as an identity pass-through for graph validation.
                current = format!("rms_norm_{}", op);
            }
            "rope" => {
                // RoPE is not expressible via the current MilBuilder helpers.
                // Emit as an identity pass-through for graph validation.
                current = format!("rope_{}", op);
            }
            other => {
                return Err(format!(
                    "unknown op type '{}' in subgraph '{}'; cannot build MIL program",
                    other, name
                ));
            }
        }
    }

    builder = builder.output(&current);
    builder
        .build()
        .map_err(|e| format!("MIL build error for subgraph '{}': {e}", name))
}
/// Validator for ANE hardware compatibility invariants.
///
/// Enforces four rules before coremlcompiler runs:
/// 1. No `linear` or `matmul` ops — must be 1×1 Conv2d
/// 2. Every output tensor shape is [B, C, 1, S] (4D with spatial dim = 1)
/// 3. Trailing dimension stride is 64-byte aligned (Float16 = dim*2 % 64 == 0)
/// 4. Weight inputs tagged `Uint8` are followed by a `cast` + `mul` with scale
pub fn validate_ane_program(program: &mil_spec::Program) -> Result<(), String> {
    let mut errors: Vec<String> = Vec::new();

    for (func_name, func) in &program.functions {
        let active_block = func.block_specializations.get(&func.opset);
        let block = match active_block {
            Some(b) => b,
            None => continue,
        };

        for op in &block.operations {
            let op_type = op.r#type.to_lowercase();

            // Rule 1: No linear or matmul operations.
            if op_type == "linear" || op_type == "matmul" {
                errors.push(format!(
                    "[ANE-HW] Function '{}': forbidden op '{}' — must be replaced with 1x1 Conv2d",
                    func_name, op.r#type
                ));
            }

            // Rule 2 & 3: Check output tensor shapes.
            for out in &op.outputs {
                if let Some(ref vt) = out.r#type {
                    if let Some(mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
                        let dims: Vec<i64> = tt.dimensions.iter().filter_map(|d| {
                            d.dimension.as_ref().and_then(|dim| {
                                if let mil_spec::dimension::Dimension::Constant(c) = dim {
                                    Some(c.size as i64)
                                } else {
                                    None
                                }
                            })
                        }).collect();

                        if dims.len() == 4 {
                            // Rule 3: trailing dimension * 2 (Float16) must be 64-byte aligned.
                            let trailing = dims[3] as u64;
                            if (trailing * 2) % 64 != 0 {
                                errors.push(format!(
                                    "[ANE-HW] Function '{}': op '{}' output '{}' trailing dim {} * 2 = {} not 64-byte aligned",
                                    func_name, op.r#type, out.name, trailing, trailing * 2
                                ));
                            }
                        } else if dims.len() > 0 {
                            // Rule 2: non-scalar tensors must be 4D [B, C, 1, S].
                            errors.push(format!(
                                "[ANE-HW] Function '{}': op '{}' output '{}' has {}D shape {:?}, must be 4D [B, C, 1, S]",
                                func_name, op.r#type, out.name, dims.len(), dims
                            ));
                        }
                    }
                }
            }
        }
    }

    // Rule 4: Check for Uint8 inputs followed by cast + mul.
    // Scan for "const" ops with Uint8 type, then verify downstream cast exists.
    let const_u8_names: Vec<String> = {
        let mut names = Vec::new();
        for func in program.functions.values() {
            let active_block = func.block_specializations.get(&func.opset);
            let block = match active_block { Some(b) => b, None => continue };
            for op in &block.operations {
                if op.r#type.to_lowercase() == "const" {
                    for out in &op.outputs {
                        if let Some(ref vt) = out.r#type {
                            if let Some(mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
                                if tt.data_type == mil_spec::DataType::Uint8 as i32
                                    || tt.data_type == mil_spec::DataType::Int8 as i32
                                {
                                    names.push(out.name.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        names
    };

    if !const_u8_names.is_empty() {
        // Verify each Uint8 constant has a downstream cast + mul chain.
        for u8_name in &const_u8_names {
            let mut found_cast = false;
            for func in program.functions.values() {
                let active_block = func.block_specializations.get(&func.opset);
                let block = match active_block { Some(b) => b, None => continue };
                for op in &block.operations {
                    if op.r#type.to_lowercase() == "cast" {
                        for inp in op.inputs.values() {
                            if let Some(arg_name) = inp.arguments.first().and_then(|b| {
                                b.binding.as_ref().and_then(|b| {
                                    if let mil_spec::argument::binding::Binding::Name(n) = b {
                                        Some(n.as_str())
                                    } else {
                                        None
                                    }
                                })
                            }) {
                                if arg_name == u8_name {
                                    found_cast = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if !found_cast {
                errors.push(format!(
                    "[ANE-HW] Uint8 constant '{}' has no downstream cast(Float16) — INT8 memory bypass not active",
                    u8_name
                ));
            }
        }
    }

    if errors.is_empty() {
        println!("✅ ANE Structural Optimization Invariants Validated.");
        Ok(())
    } else {
        let msg = errors.join("\n");
        Err(format!(
            "❌ ANE Hardware Validation FAILED ({} errors):\n{}",
            errors.len(),
            msg
        ))
    }
}

/// Compile all ANE fused islands with zero-filled weight placeholders.
///
/// Iterates over [`crate::config::AneFusedIsland`] entries in the execution
/// plan and compiles each as a Core ML subgraph. The ANE model is stateless
/// and reads weights from the shared ternary-quantized .cimage at runtime;
/// the MIL programs use zero-filled placeholders for type-checking only.
/// Each island's `.mlmodelc` is written to
/// `output_dir / island.modelc_relpath`.
pub fn compile_ane_islands(
    execution_plan: &crate::config::ModelExecutionPlan,
    arch: &crate::config::TextArchitecture,
    output_dir: &std::path::Path,
) -> Result<(), String> {
    fn placeholder_f32(rows: u32, cols: u32) -> Vec<f32> {
        vec![0.0f32; (rows as usize) * (cols as usize)]
    }
    let gm = arch.hidden_size;
    let im = arch.intermediate_size;
    let vm = arch.vocab_size;
    let hd = arch.head_dim;
    let nq = arch.num_attention_heads;
    let nk = arch.num_key_value_heads;
    let qk = hd * nq;
    let kk = hd * nk;

    for island in &execution_plan.fused_ane_islands {

        // ── Build weights HashMap ──────────────────────────────────────
        let mut weights: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        match island.subgraph_kind.as_str() {
            "mlp_block" => {
                weights.insert("gate_w".to_string(), placeholder_f32(gm, im));
                weights.insert("up_w".to_string(), placeholder_f32(gm, im));
                weights.insert("down_w".to_string(), placeholder_f32(im, gm));
            }
            "qkv_bundle" | "rmsnorm_qkv" => {
                weights.insert("q_w".to_string(), placeholder_f32(gm, qk));
                weights.insert("k_w".to_string(), placeholder_f32(gm, kk));
                weights.insert("v_w".to_string(), placeholder_f32(gm, kk));
            }
            "output_proj" => {
                weights.insert("lm_head_w".to_string(), placeholder_f32(gm, vm));
            }
            "ffn_output" => {
                weights.insert("gate_w".to_string(), placeholder_f32(gm, im));
                weights.insert("up_w".to_string(), placeholder_f32(gm, im));
                weights.insert("down_w".to_string(), placeholder_f32(im, gm));
                weights.insert("lm_head_w".to_string(), placeholder_f32(gm, vm));
            }
            "matmul" => {
                // No named weight — uses input activations only.
            }
            other => {
                return Err(format!(
                    "unknown subgraph_kind '{}' for island '{}'",
                    other, island.island_id
                ));
            }
        }

        // Add rms weight for rmsnorm_qkv
        if island.subgraph_kind == "rmsnorm_qkv" {
            weights.insert("rms_w".to_string(), vec![0.0f32; gm as usize]);
        }

        // ── Build input_shapes ────────────────────────────────────────
        let mut shapes: std::collections::HashMap<String, Vec<i64>> =
            std::collections::HashMap::new();
        shapes.insert("hidden".to_string(), vec![arch.hidden_size as i64]);
        shapes.insert(
            "intermediate".to_string(),
            vec![arch.intermediate_size as i64],
        );
        shapes.insert("vocab".to_string(), vec![arch.vocab_size as i64]);
        shapes.insert("head_dim".to_string(), vec![arch.head_dim as i64]);
        shapes.insert("n_heads".to_string(), vec![arch.num_attention_heads as i64]);
        shapes.insert(
            "n_kv_heads".to_string(),
            vec![arch.num_key_value_heads as i64],
        );

        // ── Build ops from subgraph_kind ───────────────────────────────
        let ops: Vec<String> = match island.subgraph_kind.as_str() {
            "mlp_block" => vec![
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            "qkv_bundle" => vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ],
            "rmsnorm_qkv" => vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
            ],
            "output_proj" => vec!["lm_head".to_string()],
            "ffn_output" => vec![
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
                "lm_head".to_string(),
            ],
            "matmul" => vec!["matmul".to_string()],
            other => {
                return Err(format!(
                    "unknown subgraph_kind '{}' for island '{}'",
                    other, island.island_id
                ));
            }
        };

        // ── Compile subgraph with weights ──────────────────────────────
        let modelc_path = compile_subgraph(&island.island_id, &ops, &shapes, &weights, output_dir)?;
        eprintln!(
            "[compile_coreml] compiled {} → {}/{}",
            island.island_id,
            output_dir.display(),
            island.modelc_relpath
        );
        let _ = modelc_path; // consumed by compile_subgraph
    }

    Ok(())
}
