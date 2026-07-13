//! ExecutionGraph — the execution-oriented graph produced from ModelIr + RepresentationPlan.
//!
//! Describes what must execute, but not yet which exact Metal function
//! implements it. This is the shared input for megakernel planning,
//! per-layer planning, fused-region planning, CPU fallback, and ANE
//! subgraph extraction.

use serde::{Deserialize, Serialize};

/// Unique identifier for a region within an execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub usize);

/// Identifies which execution lane (backend) a region targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionLane {
    /// Apple GPU via Metal.
    MetalGpu,
    /// Apple Neural Engine.
    Ane,
    /// CPU fallback (reference).
    Cpu,
    /// AMD GPU via ROCm.
    Rocm,
    /// Intel GPU via Level Zero.
    LevelZero,
    /// MLX framework (Apple GPU).
    Mlx,
}

/// A value that flows between execution operations (a buffer reference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferValue {
    pub name: String,
    pub byte_size: u64,
    pub tensor_id: Option<super::model_ir::TensorId>,
}

/// A single executable operation within a region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOp {
    pub name: String,
    pub kind: ExecutionOpKind,
    pub inputs: Vec<BufferValue>,
    pub outputs: Vec<BufferValue>,
    pub attributes: std::collections::HashMap<String, String>,
}

/// Kinds of executable operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionOpKind {
    RmsNorm,
    LayerNorm,
    Linear,
    QuantizedLinear,
    Attention,
    RoPE,
    SiLU,
    Mul,
    Add,
    Softmax,
    RotaryEmbedding,
    Gather,
    ScalarAdd,
    Scale,
    Fp32Dequant,
    Nf4Dequant,
    Int8Dequant,
    TernaryDequant,
    Other(String),
}

/// Constraints that guide fusion decisions for a region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionConstraints {
    /// Maximum operations that can be fused into one kernel.
    pub max_fused_ops: Option<usize>,
    /// Whether the region must be a single fused kernel.
    pub force_fused: bool,
    /// Whether the region must remain unfused (individual kernels).
    pub force_unfused: bool,
}

/// A single execution region — a group of operations that execute together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRegion {
    pub id: RegionId,
    pub name: String,
    pub operations: Vec<ExecutionOp>,
    pub target_lane: ExecutionLane,
    pub fusion_constraints: FusionConstraints,
    pub inputs: Vec<BufferValue>,
    pub outputs: Vec<BufferValue>,
}

/// A directed edge between execution regions (data dependency).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionEdge {
    pub source_region: RegionId,
    pub source_output: String,
    pub target_region: RegionId,
    pub target_input: String,
}

/// Plan for runtime state (KV cache, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStatePlan {
    pub max_context_tokens: usize,
    pub kv_cache_bytes_per_token: u64,
    pub total_kv_cache_bytes: u64,
}

/// Plan for memory allocation across regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub total_activation_bytes: u64,
    pub total_weight_bytes: u64,
    pub arena_region_count: usize,
}

/// ExecutionGraph — the complete execution-oriented representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub regions: Vec<ExecutionRegion>,
    pub edges: Vec<ExecutionEdge>,
    pub state: RuntimeStatePlan,
    pub memory: MemoryPlan,
}

impl ExecutionGraph {
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}
