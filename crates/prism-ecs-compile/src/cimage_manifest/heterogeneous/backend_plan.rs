//! Backend capability analysis and region formation.
//!
//! This module owns the **backend-eligibility layer**: the
//! [`PhaseCapabilityMatrix`] that records, for every phase, which
//! lanes support it (and at what cost), and the
//! [`RegionFormationDecision`] that records whether the compiler
//! chose to merge adjacent phases into a fused region or keep them
//! separate for concurrency.
//!
//! These are **plan descriptors**, not live evidence — the runtime
//! consumes them to decide which lane to dispatch each phase on and
//! which region boundary to honor. The actual lane selection is
//! re-validated by the per-lane admission gates before dispatch.

use serde::{Deserialize, Serialize};

use super::phase_ir::PhaseId;
use super::shared::{ActivationAbi, ContentHash, ExecutionLane};

/// Every phase receives a capability record for all three lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCapabilityMatrix {
    pub phase_id: PhaseId,
    pub metal: LaneCapability,
    pub ane: LaneCapability,
    pub accelerate: LaneCapability,
}

/// Lane capability for a single phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaneCapability {
    /// The lane supports this phase directly with the given cost and
    /// ABI.
    Supported {
        estimated_cost: CompileCostEstimate,
        required_abi: ActivationAbi,
        required_artifacts: Vec<ArtifactRequirement>,
    },
    /// Supported but requires materialization (data transfer) at
    /// boundaries.
    SupportedWithMaterialization {
        estimated_cost: CompileCostEstimate,
        materialization: MaterializationPlan,
        required_abi: ActivationAbi,
    },
    /// Not supported — records the reason explicitly.
    Unsupported { reason: UnsupportedReason },
}

/// Why a lane cannot execute a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsupportedReason {
    OperatorNotImplemented(String),
    ShapeOutOfRange(String),
    NumericalContractUnsatisfied(String),
    DynamicShapeUnsupported(String),
    ResourceConstraint(String),
    QualificationFailed(String),
    Other(String),
}

/// Compile-time cost estimate for a phase on a lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCostEstimate {
    pub expected_ns: u64,
    pub memory_bytes: u64,
    pub compute_intensity: f64,
    pub confidence: CostConfidence,
}

/// Confidence level of a cost estimate.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CostConfidence {
    Measured,
    Profiled,
    Estimated,
    Speculative,
}

/// An artifact requirement (e.g., a compiled .mlmodelc or .metallib).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    pub artifact_kind: ArtifactKind,
    pub artifact_id: String,
    pub content_hash: ContentHash,
}

/// Kind of compiled artifact.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactKind {
    CoreAiModel,
    MetalLibrary,
    MetalKernel,
    AccelerateRoutine,
    WeightPack,
    ArenaPlan,
}

/// How tensor data crosses device boundaries.
///
/// The runtime looks up this plan to pick the right zero-copy
/// sharing path, IOSurface-backed pointer binding, or explicit host
/// copy when materializing a value at a phase boundary.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MaterializationPlan {
    /// Zero-copy IOSurface — the preferred mode.
    IOSurfaceShared,
    /// IOSurface-backed pointer binding through MLMultiArray.
    IOSurfacePointerBackedMultiArray,
    /// Explicit host-side copy.
    HostCopy,
    /// IOSurface pixel buffer (CVPixelBuffer) binding.
    IOSurfacePixelBuffer,
}

// ── Region formation ──────────────────────────────────────────────────────

/// Identifies a region in the compiler's region formation analysis.
pub type RegionId = u64;

/// Result of a region formation decision — whether to merge adjacent
/// phases into a fused region or keep them separate for concurrency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionFormationDecision {
    pub region_id: RegionId,
    pub merged_phases: Vec<PhaseId>,
    pub selected_lane_candidates: Vec<ExecutionLane>,
    pub fusion_gain_ns: u64,
    pub lost_overlap_ns: u64,
    pub decision: RegionDecision,
}

/// Whether a region formation was accepted or rejected.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegionDecision {
    Fused,
    KeptSeparate,
}
