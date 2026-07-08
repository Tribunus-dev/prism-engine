//! Fusion lowering — translate a fused group into an `AccelerateRayonProgram`.
//!
//! The lowering pass takes a group descriptor (produced by the fusion
//! scheduler) and produces a concrete program that the CPU runtime can
//! execute: a sequence of ops with a parallel strategy, Accelerate call
//! specs, and a scratch plan for intermediate buffers.
//!
//! Types in this module are self-contained to avoid coupling to the
//! evolving `execution_plan::fusion` types. A bridge function converts
//! between the external `FusedGroup` and the local `LoweredGroup` at
//! the integration boundary.

use crate::cpu_runtime::capabilities::{CpuBackendCapability, CpuProgramOp};
use crate::cpu_runtime::rayon_strategy::{CpuScratchPlan, RayonStrategy};
use crate::cpu_runtime::receipts::CpuLoweringReceipt;
use serde::{Deserialize, Serialize};

// ── LoweredOp ───────────────────────────────────────────────────────────────

/// A single operation within a lowered CPU program — the self-contained
/// representation of what the runtime will execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredOp {
    /// What kind of operation this is.
    pub op: CpuProgramOp,
    /// Human-readable name (from the original fused group).
    pub step_name: String,
    /// Indices into the program's buffer table for inputs.
    pub input_buffers: Vec<usize>,
    /// Indices into the program's buffer table for outputs.
    pub output_buffers: Vec<usize>,
}

// ── LoweredGroup ────────────────────────────────────────────────────────────

/// A fused group ready for CPU lowering — the self-contained input to
/// [`accel_rayon_lower`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoweredGroup {
    /// Stable identifier from the fusion scheduler.
    pub group_id: usize,
    /// The member ops in execution order.
    pub ops: Vec<LoweredOp>,
    /// Codec family used by this group (e.g. "RawF32", "Fp16", "Int8").
    pub codec: String,
    /// Estimated total materialization size in bytes.
    pub estimated_bytes: u64,
}

// ── AccelerateCallSpec ──────────────────────────────────────────────────────

/// Describes a single Accelerate framework call (vDSP, BLAS, BNNS) within
/// a fused CPU program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateCallSpec {
    /// The Accelerate function name (e.g. `cblas_sgemm`, `vDSP_vsq`).
    pub function_name: String,
    /// Indices into the program's buffer table for input arguments.
    pub input_args: Vec<usize>,
    /// Indices into the program's buffer table for output arguments.
    pub output_args: Vec<usize>,
    /// Whether a CPU memory barrier / sync is required before the result
    /// can be consumed by a subsequent op.
    pub requires_sync: bool,
}

// ── AccelerateRayonProgram ─────────────────────────────────────────────────

/// A lowered program for the Accelerate + Rayon CPU execution backend.
///
/// Produced by [`accel_rayon_lower`] from a [`LoweredGroup`]. Contains
/// everything the runtime needs to execute the group: the op sequence,
/// parallel strategy, individual Accelerate calls, and a scratch plan
/// for temporary buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateRayonProgram {
    /// Unique identifier for this program within the execution plan.
    pub program_id: String,

    /// The sequence of CPU operations to execute.
    pub ops: Vec<CpuProgramOp>,

    /// Strategy for subdividing work across Rayon threads.
    pub parallel_strategy: RayonStrategy,

    /// Individual Accelerate framework calls that make up the program.
    pub accelerate_calls: Vec<AccelerateCallSpec>,

    /// Scratch buffer plan for intermediate values.
    pub scratch_plan: CpuScratchPlan,

    /// Whether the program guarantees deterministic (bit-identical) results
    /// regardless of thread count or scheduling.
    pub deterministic: bool,
}

// ── CpuLoweringError ──────────────────────────────────────────────────────

/// Errors that can occur when lowering a fused group into a CPU program.
#[derive(Debug, Clone)]
pub enum CpuLoweringError {
    /// An operation kind is not supported by the CPU backend.
    UnsupportedOp(String),
    /// A codec is not supported by the CPU backend.
    UnsupportedCodec(String),
    /// A dense materialization exceeds the backend's size budget.
    MaterializationTooLarge {
        requested: u64,
        max: u64,
    },
    /// A required scratch plan could not be derived.
    MissingScratchPlan(String),
    /// An internal lowering error.
    Internal(String),
}

impl std::fmt::Display for CpuLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOp(op) => write!(f, "unsupported CPU op: {op}"),
            Self::UnsupportedCodec(codec) => write!(f, "unsupported CPU codec: {codec}"),
            Self::MaterializationTooLarge { requested, max } => {
                write!(f, "materialization {requested} bytes exceeds CPU limit {max}")
            }
            Self::MissingScratchPlan(detail) => write!(f, "missing scratch plan: {detail}"),
            Self::Internal(detail) => write!(f, "internal lowering error: {detail}"),
        }
    }
}

impl std::error::Error for CpuLoweringError {}

// ── Inference helpers ──────────────────────────────────────────────────────

/// Map an op name string to the corresponding `CpuProgramOp` if supported.
fn parse_cpu_op(name: &str) -> Option<CpuProgramOp> {
    match name {
        "Matmul" | "matmul" => Some(CpuProgramOp::Matmul),
        "RmsNorm" | "rms_norm" => Some(CpuProgramOp::RmsNorm),
        "Gelu" | "gelu" | "SiLU" | "silu" => Some(CpuProgramOp::Gelu),
        "AddResidual" | "ResidualAdd" | "add" | "Add" => Some(CpuProgramOp::AddResidual),
        "AttentionScore" | "attention_score" => Some(CpuProgramOp::AttentionScore),
        _ => None,
    }
}

/// Infer a parallel strategy from group characteristics.
fn infer_parallel_strategy(has_heavy_matmul: bool) -> RayonStrategy {
    if has_heavy_matmul {
        RayonStrategy::Static {
            num_threads: 0,
            chunk_size: 128,
        }
    } else {
        RayonStrategy::WorkStealing
    }
}

// ── Lowering entry point ────────────────────────────────────────────────────

/// Lower a [`LoweredGroup`] into an [`AccelerateRayonProgram`].
///
/// # Errors
///
/// Returns `CpuLoweringError` when:
/// - Any op in the group is not in `capability`'s supported ops.
/// - The codec is not supported.
/// - The estimated materialization size exceeds the backend's budget.
///
/// # Determinism
///
/// A program is marked deterministic when all ops are element-wise or
/// reduction-only (no non-associative accumulation order sensitivity).
pub fn accel_rayon_lower(
    group: &LoweredGroup,
    capability: &CpuBackendCapability,
) -> Result<AccelerateRayonProgram, CpuLoweringError> {
    // Validate group is non-empty.
    if group.ops.is_empty() {
        return Err(CpuLoweringError::Internal(
            "cannot lower an empty fused group".into(),
        ));
    }

    // ── Validate ops ────────────────────────────────────────────────────
    let mut cpu_ops: Vec<CpuProgramOp> = Vec::with_capacity(group.ops.len());

    for op in &group.ops {
        if !capability.supports_op(&op.op) {
            return Err(CpuLoweringError::UnsupportedOp(format!("{:?}", op.op)));
        }
        cpu_ops.push(op.op);
    }

    // ── Validate codec ──────────────────────────────────────────────────
    if !capability.supports_codec(&group.codec) {
        return Err(CpuLoweringError::UnsupportedCodec(group.codec.clone()));
    }

    // ── Check materialization budget ─────────────────────────────────────
    capability
        .check_materialization_budget(group.estimated_bytes)
        .map_err(|_| CpuLoweringError::MaterializationTooLarge {
            requested: group.estimated_bytes,
            max: capability.max_dense_bytes,
        })?;

    // ── Infer strategy and scratch ───────────────────────────────────────
    let has_heavy_matmul = cpu_ops
        .iter()
        .any(|op| matches!(op, CpuProgramOp::Matmul));
    let parallel_strategy = infer_parallel_strategy(has_heavy_matmul);

    // ── Build accelerator call specs ─────────────────────────────────────
    let mut accelerate_calls: Vec<AccelerateCallSpec> = Vec::new();
    for op in &group.ops {
        let call = AccelerateCallSpec {
            function_name: format!("cpu_{}", op.step_name),
            input_args: op.input_buffers.clone(),
            output_args: op.output_buffers.clone(),
            requires_sync: true,
        };
        accelerate_calls.push(call);
    }

    // ── Determinism ──────────────────────────────────────────────────────
    let deterministic = if has_heavy_matmul {
        matches!(parallel_strategy, RayonStrategy::Static { .. })
    } else {
        true
    };

    let program_id = format!("cpu_fused_{}", group.group_id);

    Ok(AccelerateRayonProgram {
        program_id,
        ops: cpu_ops,
        parallel_strategy,
        accelerate_calls,
        scratch_plan: CpuScratchPlan::single_spill(1024), // conservative minimum
        deterministic,
    })
}

// ── Convenience bridge ─────────────────────────────────────────────────────

/// Build a [`LoweredGroup`] from a name-based op sequence (for test use and
/// for integration with the fusion scheduler before the real types stabilise).
pub fn build_lowered_group(
    group_id: usize,
    op_names: &[&str],
    codec: &str,
    estimated_bytes: u64,
) -> Result<LoweredGroup, CpuLoweringError> {
    let mut ops: Vec<LoweredOp> = Vec::with_capacity(op_names.len());
    for (i, name) in op_names.iter().enumerate() {
        let op = parse_cpu_op(name)
            .ok_or_else(|| CpuLoweringError::UnsupportedOp(name.to_string()))?;
        ops.push(LoweredOp {
            op,
            step_name: name.to_string(),
            input_buffers: vec![i * 2],
            output_buffers: vec![i * 2 + 1],
        });
    }
    Ok(LoweredGroup {
        group_id,
        ops,
        codec: codec.to_string(),
        estimated_bytes,
    })
}

/// Generate a `CpuLoweringReceipt` from a lowered program and original group.
pub fn build_lowering_receipt(
    program: &AccelerateRayonProgram,
    group_id: usize,
) -> CpuLoweringReceipt {
    CpuLoweringReceipt {
        program_id: program.program_id.clone(),
        group_id,
        op_count: program.ops.len(),
        parallel_strategy: program.parallel_strategy,
        accelerate_call_count: program.accelerate_calls.len(),
        scratch_bytes: program.scratch_plan.spill_buffer_size as u64,
        deterministic: program.deterministic,
        warnings: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_cap() -> CpuBackendCapability {
        let mut c = CpuBackendCapability::default();
        crate::cpu_runtime::capabilities::register_accelerate_rayon_capabilities(&mut c);
        c
    }

    #[test]
    fn lower_rmsnorm_succeeds() {
        let cap = cpu_cap();
        let group = build_lowered_group(0, &["RmsNorm"], "RawF32", 4096).unwrap();
        let program = accel_rayon_lower(&group, &cap).unwrap();
        assert_eq!(program.ops.len(), 1);
        assert!(matches!(program.ops[0], CpuProgramOp::RmsNorm));
    }

    #[test]
    fn lower_matmul_gelu_succeeds() {
        let cap = cpu_cap();
        let group =
            build_lowered_group(1, &["Matmul", "Gelu"], "RawF32", 64 * 1024).unwrap();
        let program = accel_rayon_lower(&group, &cap).unwrap();
        assert_eq!(program.ops.len(), 2);
        assert!(matches!(program.ops[0], CpuProgramOp::Matmul));
        assert!(matches!(program.ops[1], CpuProgramOp::Gelu));
    }

    #[test]
    fn lower_unsupported_op_fails() {
        let cap = cpu_cap();
        let result = build_lowered_group(99, &["BridgeProjection"], "RawF32", 4096);
        assert!(
            result.is_err(),
            "BridgeProjection should be unsupported by CPU"
        );
        assert!(
            matches!(result.unwrap_err(), CpuLoweringError::UnsupportedOp(_)),
            "expected UnsupportedOp error"
        );
    }

    #[test]
    fn lower_unsupported_codec_fails() {
        let cap = cpu_cap();
        let group = build_lowered_group(2, &["RmsNorm"], "Nf4", 4096).unwrap();
        let result = accel_rayon_lower(&group, &cap);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), CpuLoweringError::UnsupportedCodec(_)),
            "expected UnsupportedCodec error for Nf4"
        );
    }

    #[test]
    fn lower_empty_group_fails() {
        let cap = cpu_cap();
        let group = LoweredGroup {
            group_id: 0,
            ops: vec![],
            codec: "RawF32".into(),
            estimated_bytes: 0,
        };
        let result = accel_rayon_lower(&group, &cap);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), CpuLoweringError::Internal(_)),
            "expected Internal error for empty group"
        );
    }
}
