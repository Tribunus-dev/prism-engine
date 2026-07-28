//! Heterogeneous execution image types — pure data type definitions
//! for the compiler-emitted multi-lane execution image.
//!
//! The engine-coupled implementations (the actual `HeterogeneousImageBuilder`,
//! which dispatches to Metal/MLX/ANE/Accelerate, and the full
//! `HeterogeneousExecutionImage` with its `CompiledPhaseGraph`,
//! `CompiledResourcePlan`, etc.) stay engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/heterogeneous/types.rs`.
//!
//! This file declares the public type identifiers and the data-only
//! taxonomy. Engine-coupled data lives behind the legacy path.

use serde::{Deserialize, Serialize};

/// Opaque phase identifier.
pub type PhaseId = u64;

/// Opaque value identifier.
pub type ValueId = u64;

/// Opaque operator identifier.
pub type OperatorId = String;

/// Top-level heterogeneous execution image (canonical type identity;
/// the full data definition lives engine-side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionImage {
    /// Format version.
    pub image_version: u32,
    /// Model identity.
    pub model_identity: ModelIdentity,
    /// Graph digest.
    pub graph_digest: u64,
}

/// Identity of the imported model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Model name.
    pub model_name: String,
    /// Model family.
    pub model_family: String,
    /// Model variant.
    pub model_variant: String,
    /// Canonical graph hash.
    pub canonical_graph_hash: u64,
    /// Compile timestamp (RFC3339 string).
    pub compile_timestamp: String,
    /// Compiler version.
    pub compiler_version: String,
}

/// Canonical phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraph {
    /// Phase nodes.
    pub phases: Vec<PhaseNode>,
    /// Phase edges.
    pub edges: Vec<PhaseEdge>,
    /// Phase values.
    pub values: Vec<PhaseValue>,
}

/// A single executable semantic region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseNode {
    /// Phase identifier.
    pub phase_id: PhaseId,
    /// Phase kind.
    pub kind: PhaseKind,
    /// Operator identifiers.
    pub operators: Vec<OperatorId>,
    /// Input value identifiers.
    pub inputs: Vec<ValueId>,
    /// Output value identifiers.
    pub outputs: Vec<ValueId>,
    /// Shape contract.
    pub shape_contract: ShapeContract,
    /// Numerical contract.
    pub numerical_contract: NumericalContract,
}

/// A directed edge between two phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEdge {
    /// Source phase identifier.
    pub from_phase: PhaseId,
    /// Target phase identifier.
    pub to_phase: PhaseId,
    /// Edge kind.
    pub kind: PhaseEdgeKind,
    /// Carried value identifier.
    pub value: ValueId,
}

/// A value flowing through the phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseValue {
    /// Value identifier.
    pub value_id: ValueId,
    /// Source phase.
    pub producer: PhaseId,
    /// Consumers.
    pub consumers: Vec<PhaseId>,
}

/// Phase kind taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseKind {
    /// Attention phase.
    Attention,
    /// MLP phase.
    Mlp,
    /// Normalization phase.
    Norm,
    /// Embedding phase.
    Embedding,
    /// Sampling phase.
    Sample,
    /// Custom phase.
    Custom,
}

/// Dependency class for a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyClass {
    /// Strictly sequential.
    Sequential,
    /// Can be parallelized.
    Parallel,
    /// Barrier (all consumers must wait).
    Barrier,
}

/// Shape contract for a phase's input or output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeContract {
    /// Dimensions.
    pub dims: Vec<u32>,
}

/// Numerical contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalContract {
    /// Element type as a string.
    pub element_type: String,
    /// Tolerance for numerical verification.
    pub tolerance: f64,
}

/// Edge kind between two phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseEdgeKind {
    /// Direct data dependency.
    DataDependency,
    /// Control dependency (e.g., barrier).
    ControlDependency,
}

/// Lane capability matrix for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCapabilityMatrix {
    /// Phase identifier.
    pub phase_id: PhaseId,
    /// Lane capabilities.
    pub lanes: Vec<LaneCapability>,
}

/// Lane capability assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaneCapability {
    /// Lane is fully supported.
    Supported,
    /// Lane is supported with caveats.
    SupportedWithCaveats,
    /// Lane is unsupported.
    Unsupported(UnsupportedReason),
}

/// Reason a lane is unsupported.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsupportedReason {
    /// Hardware feature missing.
    MissingHardware,
    /// Driver version too old.
    DriverTooOld,
    /// Numerical policy violation.
    NumericalPolicyViolation,
    /// Memory budget exceeded.
    MemoryBudgetExceeded,
}

/// Compile cost estimate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCostEstimate {
    /// Compile time in milliseconds.
    pub compile_time_ms: u64,
    /// Peak memory during compile in bytes.
    pub peak_compile_memory_bytes: u64,
    /// Cost confidence.
    pub confidence: CostConfidence,
}

/// Cost confidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostConfidence {
    /// High confidence.
    High,
    /// Medium confidence.
    Medium,
    /// Low confidence.
    Low,
}
