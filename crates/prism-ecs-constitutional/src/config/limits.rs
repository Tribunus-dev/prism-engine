//! Compilation planning types: tensor dispositions, planned segments, and
//! the complete `CompilationPlan` produced before payload emission.
//!
//! Authority: the canonical [`TensorDisposition`], [`PlannedTensor`],
//! [`PlannedSegment`], and [`CompilationPlan`] types — the immutable
//! output of the planning phase that the emission phase consumes. No
//! engine-internal types are referenced.

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
    CoreAiLoweringInput,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_tensor_serde_round_trip() {
        let t = PlannedTensor {
            id: 1,
            name: "model.embed_tokens.weight".into(),
            disposition: TensorDisposition::PreserveInPlace,
            source_shard: "model-00001-of-00002.safetensors".into(),
            source_offset: 0,
            source_byte_length: 1024,
            destination_segment: "persistent".into(),
            destination_offset: 0,
            destination_byte_length: 1024,
            logical_dtype: "F32".into(),
            logical_shape: vec![128, 64],
        };
        let j = serde_json::to_string(&t).unwrap();
        let back: PlannedTensor = serde_json::from_str(&j).unwrap();
        assert_eq!(back.id, 1);
        assert_eq!(back.name, t.name);
    }

    #[test]
    fn planned_segment_serde_round_trip() {
        let s = PlannedSegment {
            id: "layer_0".into(),
            filename: "layer_0.bin".into(),
            byte_size: 4096,
            kind: "decoder_layer".into(),
            tensor_count: 8,
        };
        let j = serde_json::to_string(&s).unwrap();
        let back: PlannedSegment = serde_json::from_str(&j).unwrap();
        assert_eq!(back.id, "layer_0");
        assert_eq!(back.tensor_count, 8);
    }

    #[test]
    fn compilation_plan_default_is_empty() {
        let p = CompilationPlan {
            model_identity: "test".into(),
            source_config_hash: "abc".into(),
            source_shard_hashes: vec![],
            tensor_table: vec![],
            segments: vec![],
            total_source_bytes: 0,
            total_image_bytes: 0,
        };
        assert!(p.tensor_table.is_empty());
    }
}
