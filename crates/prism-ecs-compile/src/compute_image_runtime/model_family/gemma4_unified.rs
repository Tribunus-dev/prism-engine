//! Gemma 4 Unified model-family schema — pure data types and pure
//! algorithms.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::multimodal::{
    AudioProcessorContractV1, ImageProcessorContractV1,
};

/// Hidden size for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_HIDDEN_SIZE: u32 = 3840;
/// Number of layers for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_NUM_LAYERS: u32 = 48;
/// Number of attention heads for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS: u32 = 16;
/// Number of KV heads for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_NUM_KV_HEADS: u32 = 8;
/// Vocabulary size for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_VOCABULARY_SIZE: u32 = 262144;
/// Intermediate size for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_INTERMEDIATE_SIZE: u32 = 15360;
/// Head dim for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_HEAD_DIM: u32 = 256;

/// Tensor classification categories produced by checkpoint inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorClassification {
    /// Decoder weights required.
    DecoderRequired,
    /// Text embedding weights required.
    TextEmbeddingRequired,
    /// LM head weights required.
    LmHeadRequired,
    /// Norm weights required.
    NormRequired,
    /// Multimodal image projection weights required.
    MultimodalImageRequired,
    /// Multimodal audio projection weights required.
    MultimodalAudioRequired,
    /// MTP (multi-token-prediction) weights required.
    MtpRequired,
    /// Tensor classification unknown.
    Unknown,
    /// Tensor classified as ignored.
    Ignored,
}

impl TensorClassification {
    /// String identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DecoderRequired => "decoder_required",
            Self::TextEmbeddingRequired => "text_embedding_required",
            Self::LmHeadRequired => "lm_head_required",
            Self::NormRequired => "norm_required",
            Self::MultimodalImageRequired => "multimodal_image_required",
            Self::MultimodalAudioRequired => "multimodal_audio_required",
            Self::MtpRequired => "mtp_required",
            Self::Unknown => "unknown",
            Self::Ignored => "ignored",
        }
    }

    /// Whether this tensor is required.
    pub fn is_required(&self) -> bool {
        !matches!(self, Self::Unknown | Self::Ignored)
    }
}

/// Gemma 4 Unified schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4UnifiedSchema {
    /// Hidden size.
    pub hidden_size: u32,
    /// Number of layers.
    pub num_hidden_layers: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Intermediate size.
    pub intermediate_size: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Image processor contract.
    pub image_contract: Option<ImageProcessorContractV1>,
    /// Audio processor contract.
    pub audio_contract: Option<AudioProcessorContractV1>,
}
