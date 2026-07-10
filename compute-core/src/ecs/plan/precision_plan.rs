//! Mixed-precision planning types for the fusion compiler pipeline.
//!
//! A PrecisionPlan specifies how a fused group's weights and activations
//! may use different codec families (precisions) to trade accuracy for
//! performance. The plan is authored by the policy resolver and consumed
//! by `BackendCapabilityRegistry::evaluate()` during fusion scheduling.

use serde::{Deserialize, Serialize};

use crate::ecs::plan::CodecFamily;
use crate::training_target::RequiredEvidenceLevel;

// ── PrecisionPlan ────────────────────────────────────────────────────────

/// A complete mixed-precision plan for one fused group or layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPlan {
    pub plan_id: String,
    pub scope: PrecisionScope,
    pub default_codec: CodecFamily,
    pub overrides: Vec<PrecisionOverride>,
    pub selection_basis: PrecisionSelectionBasis,
    pub evidence_level: RequiredEvidenceLevel,
}

// ── PrecisionScope ───────────────────────────────────────────────────────

/// How broadly a precision plan applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionScope {
    WholeTensor,
    TensorFamily,
    LayerRange,
    Tile,
    Group,
    InputAxisSlice,
    OutputAxisSlice,
    Expert,
    FusedGroup,
}

// ── PrecisionOverride ────────────────────────────────────────────────────

/// A single precision override for a specific selection of tensors/tiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverride {
    pub selector: PrecisionSelector,
    pub codec: CodecFamily,
    pub reason: PrecisionOverrideReason,
    pub byte_cost: u64,
    pub expected_error_reduction: Option<f64>,
}

// ── PrecisionSelector ────────────────────────────────────────────────────

/// Selector identifying which tiles or groups an override applies to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionSelector {
    TileIds(Vec<u32>),
    GroupIds(Vec<u32>),
    InputColumns(Vec<u32>),
    OutputRows(Vec<u32>),
    LayerRange { start: u32, end: u32 },
    TopErrorTiles { fraction: f64 },
    OutlierColumns { max_fraction: f64 },
    ActivationWeightedTopK { fraction: f64 },
}

// ── PrecisionOverrideReason ──────────────────────────────────────────────

/// Why a particular precision override was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionOverrideReason {
    OperatorTailRescue,
    ActivationWeightedOutlier,
    ZeroCollapseRescue,
    ByteSavingsFallback,
    BackendCompatibility,
    RawF32Required,
}

// ── PrecisionSelectionBasis ──────────────────────────────────────────────

/// What kind of evidence the precision plan is based on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionSelectionBasis {
    StaticPolicy,
    WeightError,
    OperatorError,
    ActivationWeightedError,
    OutlierMagnitude,
    ZeroCollapseRisk,
    HardwareProfile,
    LearnedProfile,
}
