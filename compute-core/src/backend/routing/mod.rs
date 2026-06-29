//! Deterministic heterogeneous routing types.

pub mod lanes;
pub mod policy;


pub use lanes::*;
pub use policy::*;

use serde::{Deserialize, Serialize};

pub use super::DType;

// ── Identity types ────────────────────────────────────────────────────────

/// Identifies a logical tensor across backend boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

/// Identifies a logical operation in the Tribunus-owned execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

/// Identifies a specific backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub u32);

/// Identifies a sealed route profile (deterministic backend assignment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteProfileId(pub u64);

/// Identifies a compiled backend artifact (e.g. Core ML model, packed layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendArtifactId(pub u64);

/// Identifies a specific materialization of a tensor on a particular backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorMaterializationId(pub u64);

/// Identifies a compiled graph region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompiledRegionHandle {
    /// Slot index into the backend's compiled region array.
    pub slot: u32,
    /// Generation counter, bumped on eviction/replacement.
    pub generation: u32,
}

/// Identifies an evaluation group (synchronization fence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvaluationGroupId(pub u64);

/// Machine profile identity (model + hardware + thermal state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MachineProfileId(pub u64);

/// Evidence digest — content-addressed proof of a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvidenceDigest(pub String);

// ── Substrate ─────────────────────────────────────────────────────────────

/// Requested compute substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedSubstrate {
    Cpu,
    Gpu,
    NeuralEngine,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

/// Observed compute substrate — `Unknown` until native instrumentation
/// provides defensible placement evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Substrate {
    Cpu,
    Gpu,
    NeuralEngine,
    Unknown,
}

// ── Operation descriptor ──────────────────────────────────────────────────

/// Logical shape before any physical layout is applied.
#[derive(Debug, Clone)]
pub struct LogicalShape {
    pub dims: Vec<u32>,
}

/// Physical layout (row-major, column-major, packed, etc.).
#[derive(Debug, Clone)]
pub enum PhysicalLayout {
    RowMajor,
    ColumnMajor,
    PackedU32 { group_size: u32, bits: u8 },
    Custom(String),
}

/// Quantization contract carried through the operation.
#[derive(Debug, Clone)]
pub struct QuantizationContract {
    pub bits: u8,
    pub group_size: u32,
    pub symmetric: bool,
}

/// Tensor shape descriptor.
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dims: Vec<u32>,
}

/// Execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
    Conditioning,
    Qualification,
}

/// Operation family for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationFamily {
    QuantizedMatmul,
    Matmul,
    RmsNorm,
    RoPE,
    Silu,
    Add,
    Multiply,
    Softmax,
    Transpose,
    Reshape,
    IndexSelect,
    Sampling,
    Reduction,
    LayoutTransform,
    Checksum,
    MlpBlock,
    AttentionBlock,
    DecoderLayer,
    PrefillFragment,
}

pub type OperationContractDigest = EvidenceDigest;

/// Policy for correctness checkpointing.
#[derive(Debug, Clone)]
pub enum CorrectnessCheckpointPolicy {
    None,
    CompareAgainstAuthority { tolerance: f64 },
    Checksum { digest: EvidenceDigest },
}

/// Complete descriptor for a single logical operation.
#[derive(Debug, Clone)]
pub struct OperationDescriptor {
    pub operation_id: OperationId,
    pub family: OperationFamily,
    pub layer_index: Option<u32>,
    pub phase: Phase,
    pub logical_shape: LogicalShape,
    pub physical_layout: PhysicalLayout,
    pub input_dtypes: Vec<DType>,
    pub output_dtype: DType,
    pub quantization: Option<QuantizationContract>,
    pub expected_output_shape: TensorShape,
    pub correctness_checkpoint: CorrectnessCheckpointPolicy,
}

// ── Tensor version ────────────────────────────────────────────────────────

/// Version counter for a logical tensor (incremented on mutation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorVersion(pub u64);

// ── Route profile ──────────────────────────────────────────────────────────

/// One routed operation in a deterministic profile.
#[derive(Debug, Clone)]
pub struct RoutedOperation {
    pub operation_id: OperationId,
    pub operation_contract: OperationContractDigest,
    pub backend: BackendId,
    pub requested_substrate: RequestedSubstrate,
    pub backend_artifact: Option<BackendArtifactId>,
    pub input_materializations: Vec<TensorMaterializationId>,
    pub output_materialization: TensorMaterializationId,
    pub evaluation_group: EvaluationGroupId,
    pub fallback_policy: FallbackPolicy,
}

/// What to do when the routed backend cannot execute.
#[derive(Debug, Clone)]
pub enum FallbackPolicy {
    FailClosed,
    FallbackTo(BackendId),
    RetryOnce(BackendId),
}

/// Manifest of backend-specific artifacts referenced by a route profile.
#[derive(Debug, Clone)]
pub struct BackendArtifactManifest {
    pub coreml: Vec<BackendArtifactId>,
    pub accelerate: Vec<BackendArtifactId>,
    pub mlx: Vec<BackendArtifactId>,
}

/// A sealed, deterministic route profile — compiled, not improvised.
#[derive(Debug, Clone)]
pub struct ComputeRouteProfile {
    pub profile_id: RouteProfileId,
    pub logical_image_hash: EvidenceDigest,
    pub artifact_root_hash: EvidenceDigest,
    pub machine_profile: MachineProfileId,
    pub operations: Vec<RoutedOperation>,
    pub transfers: Vec<TensorTransferPlan>,
    pub backend_artifacts: BackendArtifactManifest,
    /// Single source of truth for evaluation boundaries — supersedes
    /// both SynchronizationGroup and EvaluationGroupPlan.
    pub execution_boundaries: Vec<SealedExecutionBoundaryPlan>,
    pub evidence_basis: Vec<EvidenceDigest>,
}

// ── Graph region descriptor ───────────────────────────────────────────────

/// A stable subgraph region (e.g. MLP block, attention block, decoder layer).
#[derive(Debug, Clone)]
pub struct GraphRegion {
    pub region_id: u64,
    pub family: OperationFamily,
    pub operations: Vec<OperationId>,
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,
    pub shape_constraints: Vec<TensorShape>,
}
