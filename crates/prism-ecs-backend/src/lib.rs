//! Root-level backend types for prism-engine.
//!
//! This crate defines the canonical types used by all backends:
//! tensor handles, data types, operation descriptors, traits, etc.
//!
//! Compute-core backends re-export these via `pub use prism_ecs_backend::*;`.

pub mod completion;
pub mod routing;

// Re-export types from routing that used to be at root level.
pub use routing::{
    BackendExecutionReceipt, BackendVersion, ConversionKind, CorrectnessCheckpointPolicy, DType,
    EvaluationGroupCardinality, EvaluationGroupId, EvaluationPolicy, EvidenceDigest,
    ExecutionBoundaryPlan, GraphRegion, LogicalShape, OperationDescriptor, OperationFamily,
    OperationId, Phase, PhysicalLayout, QuantizationConfig, RequestedSubstrate,
    SealedExecutionBoundaryPlan, Substrate, SynchronizationPolicy, TensorHandle, TensorId,
    TensorMaterializationId, TensorShape, TensorTransferPlan, TensorVersion, BACKEND_ACCELERATE,
    BACKEND_ANE, BACKEND_MEGAKERNEL, BACKEND_METAL, BACKEND_MLX,
};

// ═════════════════════════════════════════════════════════════════════════════
// Backend trait types — migrated from the original prism-ecs-backend root.
// ═════════════════════════════════════════════════════════════════════════════

/// Opaque handle for quantized weight storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizedWeightHandle {
    pub slot: u32,
    pub generation: u32,
}

/// Parameters for a plain f32 matmul.
#[derive(Debug, Clone)]
pub struct MatmulOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub transpose_a: bool,
    pub transpose_b: bool,
}

/// Parameters for a quantized matmul.
#[derive(Debug, Clone)]
pub struct QuantizedMatmulOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
}

impl QuantizedMatmulOp {
    pub fn new(m: usize, n: usize, k: usize, group_size: usize) -> Self {
        Self {
            m,
            n,
            k,
            group_size,
        }
    }
}

/// Kernel semantic identity for ABI-versioned dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelSemanticId(pub String);

/// Parameters for RMS normalization.
#[derive(Debug, Clone)]
pub struct RmsNormOp {
    pub dim: usize,
    pub eps: f32,
    pub head_dim: i32,
}

/// Parameters for Rotary Position Embedding.
#[derive(Debug, Clone)]
pub struct RoPEOp {
    pub dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
    pub head_dim: i32,
    pub positions: Vec<u32>,
}

/// Static capability flags that a backend reports.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub supports_fp32: bool,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub supports_bf16_native: bool,
    pub backend_name: String,
    pub max_allocations: usize,
    pub max_total_bytes: u64,
}

/// Receipt returned by [`TensorBackend::evaluate`].
#[derive(Debug, Clone)]
pub struct EvaluationReceipt {
    pub elapsed_ns: u64,
    pub flops_estimate: Option<f64>,
}

/// Receipt returned by [`TensorBackend::read_f32`].
#[derive(Debug, Clone)]
pub struct ReadbackReceipt {
    pub data: Vec<f32>,
    pub forced_eval: bool,
    pub sync_ns: u64,
    pub observed_substrate: Option<Substrate>,
}

// ═════════════════════════════════════════════════════════════════════════════
// TensorBackend trait
// ═════════════════════════════════════════════════════════════════════════════

/// Trait that every tensor compute backend must implement — mutable methods
/// since each backend owns its own slot map / resource pool.
pub trait TensorBackend: Send + Sync {
    /// ── Creation ───────────────────────────────────────────────────────────
    fn create_f32(&mut self, data: &[f32], shape: &[i32]) -> Result<TensorHandle, String>;
    fn create_u32(&mut self, data: &[u32], shape: &[i32]) -> Result<TensorHandle, String>;
    fn create_f32_from_bf16_bits(
        &mut self,
        data: &[u16],
        shape: &[i32],
    ) -> Result<TensorHandle, String>;
    fn create_owned_from_bytes(
        &mut self,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String>;

    /// ── Ops ────────────────────────────────────────────────────────────────
    fn quantized_matmul(
        &mut self,
        op: &QuantizedMatmulOp,
        x: TensorHandle,
        w: QuantizedWeightHandle,
        scales: TensorHandle,
        biases: TensorHandle,
    ) -> Result<TensorHandle, String>;

    fn matmul(
        &mut self,
        op: &MatmulOp,
        a: TensorHandle,
        b: TensorHandle,
    ) -> Result<TensorHandle, String>;

    fn rms_norm(
        &mut self,
        op: &RmsNormOp,
        x: TensorHandle,
        weight: TensorHandle,
    ) -> Result<TensorHandle, String>;

    fn rope(&mut self, op: &RoPEOp, x: TensorHandle) -> Result<TensorHandle, String>;

    fn add(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String>;

    fn multiply(&mut self, a: TensorHandle, b: TensorHandle) -> Result<TensorHandle, String>;

    fn silu(&mut self, x: TensorHandle) -> Result<TensorHandle, String>;

    fn transpose(&mut self, x: TensorHandle, dims: &[i32]) -> Result<TensorHandle, String>;

    fn reshape(&mut self, x: TensorHandle, shape: &[i32]) -> Result<TensorHandle, String>;

    fn softmax(&mut self, x: TensorHandle, axis: i32) -> Result<TensorHandle, String>;

    fn index_select(&mut self, x: TensorHandle, indices: &[u32]) -> Result<TensorHandle, String>;

    fn concatenate(&mut self, tensors: &[TensorHandle], axis: i32) -> Result<TensorHandle, String>;

    fn cast(&mut self, x: TensorHandle, dtype: DType) -> Result<TensorHandle, String>;

    fn slice(
        &mut self,
        x: TensorHandle,
        offset: &[i32],
        size: &[i32],
    ) -> Result<TensorHandle, String>;

    /// ── Query / lifecycle ──────────────────────────────────────────────────
    fn evaluate(
        &mut self,
        op: &OperationDescriptor,
        inputs: &[TensorHandle],
        output: &TensorHandle,
    ) -> Result<EvaluationReceipt, String>;

    fn read_f32(&mut self, handle: TensorHandle) -> Result<ReadbackReceipt, String>;

    fn shape(&self, handle: TensorHandle) -> Result<Vec<i32>, String>;

    fn release(&mut self, handle: TensorHandle) -> Result<(), String>;

    fn active_memory(&self) -> u64;

    fn bind_external(
        &mut self,
        owner_token: u64,
        data: &[u8],
        shape: &[i32],
        dtype: DType,
    ) -> Result<TensorHandle, String>;

    fn backend_capabilities(&self) -> BackendCapabilities;

    /// Return the backend's name.
    fn name(&self) -> &'static str;
}

// ═════════════════════════════════════════════════════════════════════════════
// CompiledRegionBackend trait
// ═════════════════════════════════════════════════════════════════════════════

/// Trait for backends that can execute pre-compiled subgraph regions.
pub trait CompiledRegionBackend: Send + Sync {
    /// Execute a pre-compiled region identified by `region_id`.
    fn execute_region(
        &self,
        region_id: &GraphRegion,
        inputs: &[TensorHandle],
        outputs: &[TensorHandle],
    ) -> Result<EvaluationReceipt, String>;
}

// Re-export completion types for consumer convenience.
pub use completion::{ComellationToken, Completer, ComputationToken};
