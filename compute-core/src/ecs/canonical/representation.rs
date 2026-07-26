//! RepresentationPlan — decides how each tensor is stored and executed.
//!
//! The representation planner consumes ModelIr and produces a plan that
//! the rest of the compiler respects. No silent RawF32 fallback is allowed
//! unless the policy explicitly permits it.

use std::collections::BTreeMap;

use super::model_ir::TensorId;

/// How a single tensor is represented in the compiled artifact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorRepresentation {
    /// IEEE 754 single-precision.
    Fp32,
    /// IEEE 754 half-precision (BF16 on GPU).
    Bf16,
    /// IEEE 754 half-precision (FP16).
    Fp16,
    /// INT8 with per-block scaling.
    Int8Block(u32), // group_size
    /// INT4 with per-block scaling.
    Int4Block(u32), // group_size
    /// NF4 with tile640 layout, group_size.
    Nf4Tile640(u32), // group_size
    /// Ternary {-1,0,+1} with tile640 layout.
    TernaryTile640,
    /// Scaled reduction axis — for special attention/aggregation kernels.
    ScaledReductionAxis,
    /// Packed BitNet b1.58 format.
    PackedBitNet,
}

/// Description of residual columns that were rescued with a higher-precision format.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualPlan {
    /// Number of residual columns.
    pub columns: usize,
    /// The format used for rescued columns.
    pub rescue_format: TensorRepresentation,
    /// Fraction of total columns rescued (0.0–1.0).
    pub rescue_fraction: f64,
}

/// A single tensor's representation decision.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRepresentationEntry {
    pub tensor_id: TensorId,
    pub representation: TensorRepresentation,
    pub residual: Option<ResidualPlan>,
    /// Byte size in the compiled format.
    pub compiled_byte_size: u64,
    /// NRMSE vs the source fp32 weights.
    pub weight_nrmse: f64,
}

/// Receipt from the calibration subsystem.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationReceipt {
    pub method: String,
    pub samples_used: usize,
    pub passed: bool,
}

/// Receipt from the admission (packing) subsystem.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionReceipt {
    pub candidate_count: usize,
    pub selected_format: TensorRepresentation,
    pub weight_nrmse: f64,
    pub operator_nrmse: Option<f64>,
    pub operator_cosine: Option<f64>,
    pub passed: bool,
}

/// The complete representation plan for a model.
///
/// Every tensor that appears in the compiled artifact has an entry.
/// The plan is produced by the RepresentationPlanner and consumed by
/// the execution graph lowerer and the cimage packer.
#[derive(Debug, Clone, PartialEq)]
pub struct RepresentationPlan {
    /// Per-tensor representation decision, keyed by TensorId.
    pub tensors: BTreeMap<TensorId, TensorRepresentationEntry>,
    /// Receipt from the calibration pass.
    pub calibration_receipt: Option<CalibrationReceipt>,
    /// Receipt from the admission pass.
    pub admission_receipt: Option<AdmissionReceipt>,
    /// Whether all tensors fell back to RawF32.
    pub all_raw_f32: bool,
}

/// Policy that controls fallback behavior.
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackPolicy {
    /// Reject compilation if any tensor cannot use its target format.
    Reject,
    /// Allow falling back to a higher-precision format.
    AllowHigherPrecision,
    /// Allow falling back to CPU-only execution for problematic tensors.
    AllowCpuRegion,
}
