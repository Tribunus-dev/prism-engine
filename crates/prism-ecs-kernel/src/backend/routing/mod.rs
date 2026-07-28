//! Routing types — backend identity, operation descriptors, execution receipts,
//! graph regions, and supporting types for deterministic heterogeneous routing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Backend identity ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub u32);

pub const BACKEND_METAL: BackendId = BackendId(0);
pub const BACKEND_ACCELERATE: BackendId = BackendId(1);
pub const BACKEND_ANE: BackendId = BackendId(2);
pub const BACKEND_MLX: BackendId = BackendId(3);
pub const BACKEND_MEGAKERNEL: BackendId = BackendId(4);

// ── Operation types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationFamily {
    // Primitive ops
    Matmul,
    QuantizedMatmul,
    RmsNorm,
    RoPE,
    Attention,
    Convolution,
    Elementwise,
    Activation,
    Normalization,
    Pooling,
    Reduction,
    Reshape,
    Transpose,
    Concat,
    Slice,
    Pad,
    Gather,
    Scatter,
    Broadcast,
    View,
    // Accelerate-specific elementwise ops
    Add,
    Multiply,
    Silu,
    Softmax,
    Sampling,
    LayoutTransform,
    Checksum,
    IndexSelect,
    // Attention-family projection ops
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
    // Graph-region / compiled-region ops
    AttentionBlock,
    MlpBlock,
    DecoderLayer,
    PrefillFragment,
    VisionEncode,
    AudioEncode,
    MultimodalProject,
    // Custom extension
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    Prefill,
    Decode,
    Both,
    Qualification,
}

// ── Shape / layout types ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalShape {
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicalLayout {
    RowMajor,
    ColumnMajor,
    PackedU32 { group_size: u32, bits: u8 },
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub dims: Vec<u64>,
}

// ── Correctness / quality-gate types ────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CorrectnessCheckpointPolicy {
    None,
    Exact,
    Approximate { atol: f64, rtol: f64 },
}

// ── Tensor identity ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorVersion(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorMaterializationId(pub u64);

// ── Evidence / digest ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDigest(pub String);

// ── Backend artifact identity ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendArtifactId(pub u64);

// ── Quantization config (used in OperationDescriptor) ───────────────────

/// Quantization contract — what format the weights use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationContract {
    pub scheme: String,
    pub group_size: u32,
    pub bits: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub codec: String,
    pub group_size: u32,
    pub bits: u8,
}

// ── Tensor handle ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorHandle {
    pub slot: u32,
    pub generation: u32,
}

// ── Compiled region ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompiledRegionHandle(pub u64);

// ── Substrate types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestedSubstrate {
    Cpu,
    Gpu,
    NeuralEngine,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Substrate {
    Cpu,
    Gpu,
    NeuralEngine,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

// ── Backend version ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendVersion {
    pub backend_name: String,
    pub version: String,
    pub git_commit: Option<String>,
}

// ── Operation descriptor ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationDescriptor {
    pub operation_id: OperationId,
    pub family: OperationFamily,
    pub layer_index: Option<u32>,
    pub phase: Phase,
    pub logical_shape: LogicalShape,
    pub physical_layout: PhysicalLayout,
    pub input_dtypes: Vec<DType>,
    pub output_dtype: DType,
    pub quantization: Option<QuantizationConfig>,
    pub expected_output_shape: TensorShape,
    pub correctness_checkpoint: CorrectnessCheckpointPolicy,
}

// ── Execution receipt ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendExecutionReceipt {
    pub operation_id: OperationId,
    pub backend_id: BackendId,
    pub backend_version: BackendVersion,
    pub requested_substrate: Option<RequestedSubstrate>,
    pub observed_substrate: Option<Substrate>,
    pub graph_build_ns: Option<u64>,
    pub compile_ns: Option<u64>,
    pub queue_wait_ns: Option<u64>,
    pub submit_ns: Option<u64>,
    pub execution_ns: Option<u64>,
    pub synchronization_ns: Option<u64>,
    pub total_wall_ns: u64,
    pub bytes_read: Option<u64>,
    pub bytes_written: Option<u64>,
    pub temporary_bytes: Option<u64>,
    pub active_memory_before: Option<u64>,
    pub active_memory_after: Option<u64>,
    pub cache_memory_before: Option<u64>,
    pub cache_memory_after: Option<u64>,
    pub transfer_in_ns: Option<u64>,
    pub transfer_out_ns: Option<u64>,
    pub fallback_occurred: bool,
}

// ── Graph region ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRegion {
    pub region_id: u64,
    pub family: OperationFamily,
    pub operations: Vec<OperationId>,
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,
    pub shape_constraints: Vec<TensorShape>,
    pub inputs: HashMap<String, TensorHandle>,
    pub outputs: HashMap<String, TensorHandle>,
    pub tensor_bindings: HashMap<String, u64>,
}

// ── Evaluation-boundary / routing-plan types ────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConversionKind {
    SharedReference,
    Copy,
    Quantize,
    Dequantize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SynchronizationPolicy {
    None,
    Barrier,
    Fence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EvaluationGroupId(pub u64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationPolicy {
    BackendLazy,
    ExplicitRegion,
    Eager {
        release_inputs_after_use: bool,
        prohibit_deferred_nodes: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationGroupCardinality {
    Fixed(usize),
    PerOperation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionBoundaryPlan {
    pub group_id: EvaluationGroupId,
    pub backend_id: BackendId,
    pub operations: Vec<OperationId>,
    pub materialized_outputs: Vec<TensorId>,
    pub policy: EvaluationPolicy,
    pub synchronization: SynchronizationPolicy,
    pub release_after: Vec<OperationId>,
    pub content_digest: Option<EvidenceDigest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedExecutionBoundaryPlan {
    pub plan: ExecutionBoundaryPlan,
    pub sha256: EvidenceDigest,
}

impl SealedExecutionBoundaryPlan {
    pub fn seal(plan: ExecutionBoundaryPlan) -> Self {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        plan.group_id.0.hash(&mut h);
        plan.backend_id.0.hash(&mut h);
        for op in &plan.operations {
            op.0.hash(&mut h);
        }
        Self {
            plan,
            sha256: EvidenceDigest(format!("{:x}", h.finish())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorTransferPlan {
    pub source_backend: BackendId,
    pub destination_backend: BackendId,
    pub tensor_id: TensorId,
    pub source_layout: PhysicalLayout,
    pub destination_layout: PhysicalLayout,
    pub conversion: ConversionKind,
    pub expected_bytes: u64,
    pub synchronization_before: bool,
    pub synchronization_after: bool,
}

// ── DType ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    I2,
    U8,
    U32,
    I32,
}

// ── Submodules ───────────────────────────────────────────────────────────

pub mod lanes;
pub mod policy;
