//! Compilation planning types: tensor dispositions, planned segments, and the
//! complete CompilationPlan produced before payload emission.

use serde::{Deserialize, Serialize};

/// Disposition of a tensor in the compiled image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorDisposition {
    /// No physical payload; another tensor is the canonical storage.
    AliasOnly { canonical_tensor_id: u32 },
    /// Bytes copied unchanged into destination segment.
    RelocateAndAlign,
    /// Source bytes can be directly referenced (external-source profile).
    PreserveInPlace,
    /// Small metadata tensor that should be transformed on CPU.
    CpuTransform { recipe: String },
    /// Large data-parallel tensor that should be transformed on GPU.
    GpuTransform { recipe: String },
    /// Tensor participates in Core ML backend island.
    CoreMlLoweringInput,
    /// Not emitted (e.g., unused multimodal wrapper in text-only profile).
    DiscardWithReason { reason: String },
}

/// A single tensor's identity and placement in the compiled image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedTensor {
    pub id: u32,
    pub name: String,
    pub disposition: TensorDisposition,
    pub source_shard: String,
    pub source_offset: u64,
    pub source_byte_length: u64,
    pub destination_segment: String,
    pub destination_offset: u64,
    pub destination_byte_length: u64,
    pub logical_dtype: String,
    pub logical_shape: Vec<u32>,
}

/// A planned binary segment containing tensors in execution order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedSegment {
    pub id: String,
    pub filename: String,
    pub byte_size: u64,
    pub kind: String,
    pub tensor_count: usize,
}

/// A complete, validated, immutable compilation plan.
/// Produced by the planning phase before any payload emission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationPlan {
    pub model_identity: String,
    pub source_config_hash: String,
    pub source_shard_hashes: Vec<String>,
    pub tensor_table: Vec<PlannedTensor>,
    pub segments: Vec<PlannedSegment>,
    pub total_source_bytes: u64,
    pub total_image_bytes: u64,
}
