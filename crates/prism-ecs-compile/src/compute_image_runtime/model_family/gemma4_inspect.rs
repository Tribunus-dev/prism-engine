//! Gemma 4 checkpoint inspector — pure data types.
//!
//! The real `inspect_gemma4_checkpoint` (which reads `config.json` and
//! parses the safetensors shards) lives engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/model_family/gemma4_inspect.rs`.

use serde::{Deserialize, Serialize};

/// Model configuration parsed from `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name.
    pub model_name: String,
    /// Model type.
    pub model_type: String,
    /// Hidden size.
    pub hidden_size: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Number of hidden layers.
    pub num_hidden_layers: u32,
}

/// Source identity (config hash, shard hashes, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    /// Config content hash.
    pub config_hash: String,
    /// Shard content hashes.
    pub shard_hashes: Vec<String>,
    /// Tokenizer content hashes.
    pub tokenizer_hashes: Vec<String>,
}

/// Stream of source shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceShardStream {
    /// Shard paths.
    pub shard_paths: Vec<String>,
    /// Shard content hashes.
    pub shard_hashes: Vec<String>,
}

/// Serialized schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSchema {
    /// Schema name.
    pub schema_name: String,
    /// Schema version.
    pub schema_version: u32,
    /// Schema content (JSON).
    pub content: String,
}

/// A tensor entry in the inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<u32>,
    /// Tensor dtype.
    pub dtype: String,
    /// Tensor byte size.
    pub byte_size: u64,
}

/// Per-class summary of the tensor inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorClassSummary {
    /// Number of tensors in this class.
    pub count: u32,
    /// Total bytes in this class.
    pub byte_size: u64,
}

/// Tensor inventory — every tensor classified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventory {
    /// Tensor entries.
    pub entries: Vec<TensorEntry>,
    /// Per-class summary.
    pub class_summary: std::collections::HashMap<String, TensorClassSummary>,
}

/// Result of inspecting a Gemma 4 checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4Inspection {
    /// Model configuration.
    pub config: ModelConfig,
    /// Serialized schema.
    pub schema: SerializedSchema,
    /// Tensor inventory.
    pub inventory: TensorInventory,
    /// Source identity.
    pub source_identity: SourceIdentity,
}

/// Engine-side stub: the real implementation lives at the legacy path.
pub fn inspect_gemma4_checkpoint(
    _source_dir: &std::path::Path,
) -> Result<Gemma4Inspection, String> {
    Err("inspect_gemma4_checkpoint is engine-coupled; use the legacy path or call the engine binary directly".to_string())
}
