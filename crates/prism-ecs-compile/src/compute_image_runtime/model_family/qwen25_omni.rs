//! Qwen2.5-Omni model-family schema — pure data types and pure
//! algorithms.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::multimodal::{
    AudioProcessorContractV1, ImageProcessorContractV1,
};

// ── Qwen2.5-Omni 7B architecture constants ──────────────────────

/// Hidden size for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_HIDDEN_SIZE: u32 = 3584;
/// Number of layers for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_NUM_LAYERS: u32 = 28;
/// Number of attention heads for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_NUM_ATTENTION_HEADS: u32 = 28;
/// Number of KV heads for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_NUM_KV_HEADS: u32 = 4;
/// Vocabulary size for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_VOCABULARY_SIZE: u32 = 151936;
/// Intermediate size for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_INTERMEDIATE_SIZE: u32 = 18944;
/// Head dim for Qwen2.5-Omni 7B.
pub const QWEN25_OMNI_7B_HEAD_DIM: u32 = 128;

/// Vision encoder hidden size.
pub const QWEN25_OMNI_VISION_HIDDEN: u32 = 1152;
/// Number of vision encoder layers.
pub const QWEN25_OMNI_VISION_LAYERS: u32 = 27;
/// Number of vision encoder attention heads.
pub const QWEN25_OMNI_VISION_HEADS: u32 = 16;
/// Vision encoder patch size.
pub const QWEN25_OMNI_VISION_PATCH_SIZE: u32 = 14;
/// Vision encoder input image size.
pub const QWEN25_OMNI_VISION_IMAGE_SIZE: u32 = 448;

/// Audio encoder sample rate.
pub const QWEN25_OMNI_AUDIO_SAMPLE_RATE: u32 = 16000;

/// Tensor classification for Qwen2.5-Omni checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qwen25OmniTensorClass {
    /// Decoder weights required.
    DecoderRequired,
    /// Text embedding weights required.
    TextEmbeddingRequired,
    /// LM head weights required.
    LmHeadRequired,
    /// Norm weights required.
    NormRequired,
    /// Vision encoder weights required.
    VisionEncoderRequired,
    /// Vision projector weights required.
    VisionProjectorRequired,
    /// Audio encoder weights required.
    AudioEncoderRequired,
    /// Audio projector weights required.
    AudioProjectorRequired,
    /// Talker LM weights required.
    TalkerRequired,
    /// MTP weights required.
    MtpRequired,
    /// Tensor classification unknown.
    Unknown,
    /// Tensor classified as ignored.
    Ignored,
}

impl Qwen25OmniTensorClass {
    /// String identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DecoderRequired => "decoder_required",
            Self::TextEmbeddingRequired => "text_embedding_required",
            Self::LmHeadRequired => "lm_head_required",
            Self::NormRequired => "norm_required",
            Self::VisionEncoderRequired => "vision_encoder_required",
            Self::VisionProjectorRequired => "vision_projector_required",
            Self::AudioEncoderRequired => "audio_encoder_required",
            Self::AudioProjectorRequired => "audio_projector_required",
            Self::TalkerRequired => "talker_required",
            Self::MtpRequired => "mtp_required",
            Self::Unknown => "unknown",
            Self::Ignored => "ignored",
        }
    }
}

/// Qwen2.5-Omni schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen25OmniSchema {
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
    /// Head dim.
    pub head_dim: u32,
    /// Image processor contract.
    pub image_contract: Option<ImageProcessorContractV1>,
    /// Audio processor contract.
    pub audio_contract: Option<AudioProcessorContractV1>,
}
