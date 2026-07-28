//! Qwen2.5-Omni model-family schema.
//!
//! Encodes architectural constants for Qwen2.5-Omni 7B and provides
//! checkpoint tensor name classification.
//!
//! Qwen2.5-Omni uses a ViT-based vision encoder + audio encoder + Thinker LM
//! + Talker LM. This differs fundamentally from Gemma 4's encoder-free
//! direct projection. The ViT encoder path is served by `ViTVisionEncoderAdapter`.

use crate::ecs::legacy_compute_image_core::multimodal::{AudioProcessorContractV1, ImageProcessorContractV1};

// ── Qwen2.5-Omni 7B architecture constants ──────────────────────
// Based on config.json for Qwen/Qwen2.5-Omni-7B
pub const QWEN25_OMNI_7B_HIDDEN_SIZE: u32 = 3584;
pub const QWEN25_OMNI_7B_NUM_LAYERS: u32 = 28;
pub const QWEN25_OMNI_7B_NUM_ATTENTION_HEADS: u32 = 28;
pub const QWEN25_OMNI_7B_NUM_KV_HEADS: u32 = 4;
pub const QWEN25_OMNI_7B_VOCABULARY_SIZE: u32 = 151936;
pub const QWEN25_OMNI_7B_INTERMEDIATE_SIZE: u32 = 18944;
pub const QWEN25_OMNI_7B_HEAD_DIM: u32 = 128;

// Vision encoder constants
pub const QWEN25_OMNI_VISION_HIDDEN: u32 = 1152;
pub const QWEN25_OMNI_VISION_LAYERS: u32 = 27;
pub const QWEN25_OMNI_VISION_HEADS: u32 = 16;
pub const QWEN25_OMNI_VISION_PATCH_SIZE: u32 = 14;
pub const QWEN25_OMNI_VISION_IMAGE_SIZE: u32 = 448;

// Audio encoder constants
pub const QWEN25_OMNI_AUDIO_SAMPLE_RATE: u32 = 16000;

// ── Tensor classification ───────────────────────────────────────

/// Tensor classification for Qwen2.5-Omni checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen25OmniTensorClass {
    DecoderRequired,
    TextEmbeddingRequired,
    LmHeadRequired,
    NormRequired,
    VisionEncoderRequired,
    VisionProjectorRequired,
    AudioEncoderRequired,
    AudioProjectorRequired,
    TalkerRequired,
    MtpRequired,
    Unknown,
    Ignored,
}

impl Qwen25OmniTensorClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Qwen25OmniTensorClass::DecoderRequired => "decoder_required",
            Qwen25OmniTensorClass::TextEmbeddingRequired => "text_embedding_required",
            Qwen25OmniTensorClass::LmHeadRequired => "lm_head_required",
            Qwen25OmniTensorClass::NormRequired => "norm_required",
            Qwen25OmniTensorClass::VisionEncoderRequired => "vision_encoder_required",
            Qwen25OmniTensorClass::VisionProjectorRequired => "vision_projector_required",
            Qwen25OmniTensorClass::AudioEncoderRequired => "audio_encoder_required",
            Qwen25OmniTensorClass::AudioProjectorRequired => "audio_projector_required",
            Qwen25OmniTensorClass::TalkerRequired => "talker_required",
            Qwen25OmniTensorClass::MtpRequired => "mtp_required",
            Qwen25OmniTensorClass::Unknown => "unknown",
            Qwen25OmniTensorClass::Ignored => "ignored",
        }
    }

    pub fn is_required(&self) -> bool {
        !matches!(
            self,
            Qwen25OmniTensorClass::Unknown | Qwen25OmniTensorClass::Ignored
        )
    }
}

/// Map a Qwen2.5-Omni checkpoint tensor name to its classification.
/// Qwen uses HF Transformers naming conventions with `model.*` prefix.
pub fn classify_qwen25_omni_tensor(name: &str) -> Qwen25OmniTensorClass {
    let lower = name.to_lowercase();

    // Decoder (Thinker LM) weights
    if lower.contains("self_attn.q_proj")
        || lower.contains("self_attn.k_proj")
        || lower.contains("self_attn.v_proj")
        || lower.contains("self_attn.o_proj")
        || lower.contains("mlp.gate_proj")
        || lower.contains("mlp.up_proj")
        || lower.contains("mlp.down_proj")
    {
        return Qwen25OmniTensorClass::DecoderRequired;
    }

    // Text embeddings
    if lower.contains("embed_tokens") {
        return Qwen25OmniTensorClass::TextEmbeddingRequired;
    }

    // LM head
    if lower.contains("lm_head") {
        return Qwen25OmniTensorClass::LmHeadRequired;
    }

    // Norms
    if lower.contains("layernorm")
        || lower.contains("rms_norm")
        || lower.contains("norm.weight")
        || lower.contains("norm.bias")
    {
        return Qwen25OmniTensorClass::NormRequired;
    }

    // Vision encoder (ViT)
    if lower.contains("visual")
        || lower.contains("vision_tower.vision_model")
        || lower.contains("vision_model.embeddings")
        || lower.contains("vision_model.encoder")
    {
        return Qwen25OmniTensorClass::VisionEncoderRequired;
    }

    // Vision projector (ViT → Thinker LM)
    if lower.contains("vision_projection") || lower.contains("visual_projector") {
        return Qwen25OmniTensorClass::VisionProjectorRequired;
    }

    // Audio encoder
    if lower.contains("audio_tower")
        || lower.contains("audio_encoder")
        || lower.contains("speech_encoder")
    {
        return Qwen25OmniTensorClass::AudioEncoderRequired;
    }

    // Audio projector
    if lower.contains("audio_projection") || lower.contains("audio_projector") {
        return Qwen25OmniTensorClass::AudioProjectorRequired;
    }

    // Talker LM
    if lower.contains("talker") {
        return Qwen25OmniTensorClass::TalkerRequired;
    }

    // MTP / speculative decoding
    if lower.contains("mtp") {
        return Qwen25OmniTensorClass::MtpRequired;
    }

    // Optimizer states
    if lower.contains("optimizer") || lower.contains("adam") || lower.contains("exp_avg") {
        return Qwen25OmniTensorClass::Ignored;
    }

    Qwen25OmniTensorClass::Unknown
}

/// Schema describing a Qwen2.5-Omni model instance.
#[derive(Debug, Clone)]
pub struct Qwen25OmniSchema {
    pub model_revision: String,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocabulary_size: u32,
    pub supports_text: bool,
    pub supports_image: bool,
    pub supports_audio: bool,
    pub image_contract: Option<ImageProcessorContractV1>,
    pub audio_contract: Option<AudioProcessorContractV1>,
}

impl Qwen25OmniSchema {
    /// Create the known schema for Qwen2.5-Omni 7B.
    pub fn omni_7b() -> Self {
        Self {
            model_revision: String::new(),
            hidden_size: QWEN25_OMNI_7B_HIDDEN_SIZE,
            num_layers: QWEN25_OMNI_7B_NUM_LAYERS,
            num_attention_heads: QWEN25_OMNI_7B_NUM_ATTENTION_HEADS,
            num_key_value_heads: QWEN25_OMNI_7B_NUM_KV_HEADS,
            vocabulary_size: QWEN25_OMNI_7B_VOCABULARY_SIZE,
            supports_text: true,
            supports_image: true,
            supports_audio: true,
            image_contract: None,
            audio_contract: None,
        }
    }

    /// Validate that a loaded checkpoint matches the expected 7B architecture.
    pub fn validate_architecture(&self) -> Result<(), String> {
        if self.hidden_size != QWEN25_OMNI_7B_HIDDEN_SIZE {
            return Err(format!(
                "hidden_size mismatch: expected {}, got {}",
                QWEN25_OMNI_7B_HIDDEN_SIZE, self.hidden_size
            ));
        }
        if self.num_layers != QWEN25_OMNI_7B_NUM_LAYERS {
            return Err(format!(
                "num_layers mismatch: expected {}, got {}",
                QWEN25_OMNI_7B_NUM_LAYERS, self.num_layers
            ));
        }
        if self.vocabulary_size != QWEN25_OMNI_7B_VOCABULARY_SIZE {
            return Err(format!(
                "vocabulary_size mismatch: expected {}, got {}",
                QWEN25_OMNI_7B_VOCABULARY_SIZE, self.vocabulary_size
            ));
        }
        Ok(())
    }
}
