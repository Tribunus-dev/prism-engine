//! ModelIr — the canonical, platform-independent model representation.
//!
//! Represents model semantics, not device execution. Every model frontend
//! (GGUF, safetensors, HuggingFace) produces the same IR for semantically
//! equivalent sources.
//!
//! No Metal entry-point names, buffer indices, or device-specific layout
//! information belongs in ModelIr.

use std::collections::HashMap;

/// Stable identity for a model instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelIdentity {
    /// Human-readable name (e.g. "gemma-4-12b").
    pub name: String,
    /// Optional revision or git commit from the source.
    pub revision: Option<String>,
}

/// Identifies a specific model architecture family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArchitectureId(pub String);

/// Architecture-specific configuration parsed from the source model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfiguration {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub partial_rope_dim: Option<usize>,
    pub tie_word_embeddings: bool,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub num_mtp_heads: Option<usize>,
    pub mtp_hidden_size: Option<usize>,
    pub mtp_intermediate_size: Option<usize>,
}

/// Unique identifier for a tensor within a ModelIr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub usize);

/// Describes a single tensor in the model: its name, shape, and data source.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDescriptor {
    pub id: TensorId,
    pub name: String,
    pub shape: Vec<usize>,
    pub byte_size: u64,
    /// Whether this tensor's data is lazily loaded from the source.
    pub is_lazy: bool,
}

/// All tensors in the model, indexed by name and id.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorCatalogue {
    pub by_id: Vec<TensorDescriptor>,
    pub by_name: HashMap<String, TensorId>,
}

impl TensorCatalogue {
    pub fn get_by_name(&self, name: &str) -> Option<&TensorDescriptor> {
        self.by_name.get(name).and_then(|id| self.by_id.get(id.0))
    }

    pub fn get_by_id(&self, id: TensorId) -> Option<&TensorDescriptor> {
        self.by_id.get(id.0)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// A single named logical operation in the model graph.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalOp {
    /// Unique operation name within the graph.
    pub name: String,
    /// The operation kind (matmul, norm, attention, etc.).
    pub kind: LogicalOpKind,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Layer index for per-layer ops; None for global ops.
    pub layer_index: Option<u32>,
    /// Per-operation attributes (shape info, parameters).
    pub attributes: HashMap<String, String>,
}

/// Kinds of logical operations in the model graph.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOpKind {
    Embedding,
    RmsNorm,
    LayerNorm,
    QProjection,
    KProjection,
    VProjection,
    QkNorm,
    RoPE,
    Attention,
    OProjection,
    GateProjection,
    UpProjection,
    SiLU,
    GateMul,
    DownProjection,
    ResidualAdd,
    FinalNorm,
    LmHead,
    Softmax,
    Softcap,
    Argmax,
    Concat,
    Reshape,
    Transpose,
    Cast,
    Other(String),
}

/// The logical operation graph of the model — backend-neutral.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalGraph {
    /// All operations in execution order (pre-order traversal).
    pub ops: Vec<LogicalOp>,
    /// Tensor names that are model inputs.
    pub inputs: Vec<String>,
    /// Tensor names that are model outputs.
    pub outputs: Vec<String>,
}

/// Describes the tokenizer associated with the model.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerDescriptor {
    pub tokenizer_type: String,
    pub vocab_size: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

/// Provenance information about where the model was loaded from.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceProvenance {
    pub source_type: SourceType,
    pub source_path: String,
    pub file_digests: Vec<(String, String)>, // (file_path, sha256_hex)
}

/// The type of source the model was loaded from.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceType {
    Gguf,
    Safetensors,
    HuggingFace,
    CImageV0,
    CImage,
}

/// ModelIr — the canonical model representation.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelIr {
    pub identity: ModelIdentity,
    pub architecture: ArchitectureId,
    pub configuration: ModelConfiguration,
    pub tensors: TensorCatalogue,
    pub graph: LogicalGraph,
    pub tokenizer: TokenizerDescriptor,
    pub source_provenance: SourceProvenance,
}

impl ModelIr {
    pub fn layer_count(&self) -> usize {
        self.configuration.num_hidden_layers
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}
