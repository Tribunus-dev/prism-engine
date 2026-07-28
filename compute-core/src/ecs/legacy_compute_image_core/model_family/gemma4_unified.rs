//! Gemma 4 Unified model-family schema.
//!
//! Encodes the known architectural constants for Gemma 4 12B Unified
//! and maps checkpoint tensor names to Prism logical tensor roles.

use crate::ecs::legacy_compute_image_core::multimodal::{AudioProcessorContractV1, ImageProcessorContractV1};

/// Hard-coded architectural constants for Gemma 4 12B Unified.
pub const GEMMA4_12B_UNIFIED_HIDDEN_SIZE: u32 = 3840;
pub const GEMMA4_12B_UNIFIED_NUM_LAYERS: u32 = 48;
pub const GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS: u32 = 16;
pub const GEMMA4_12B_UNIFIED_NUM_KV_HEADS: u32 = 8;
pub const GEMMA4_12B_UNIFIED_VOCABULARY_SIZE: u32 = 262144;
pub const GEMMA4_12B_UNIFIED_INTERMEDIATE_SIZE: u32 = 15360;
pub const GEMMA4_12B_UNIFIED_HEAD_DIM: u32 = 256;

/// Tensor classification categories produced by checkpoint inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorClassification {
    DecoderRequired,
    TextEmbeddingRequired,
    LmHeadRequired,
    NormRequired,
    MultimodalImageRequired,
    MultimodalAudioRequired,
    MtpRequired,
    Unknown,
    Ignored,
}

impl TensorClassification {
    pub fn as_str(&self) -> &'static str {
        match self {
            TensorClassification::DecoderRequired => "decoder_required",
            TensorClassification::TextEmbeddingRequired => "text_embedding_required",
            TensorClassification::LmHeadRequired => "lm_head_required",
            TensorClassification::NormRequired => "norm_required",
            TensorClassification::MultimodalImageRequired => "multimodal_image_required",
            TensorClassification::MultimodalAudioRequired => "multimodal_audio_required",
            TensorClassification::MtpRequired => "mtp_required",
            TensorClassification::Unknown => "unknown",
            TensorClassification::Ignored => "ignored",
        }
    }

    pub fn is_required(&self) -> bool {
        !matches!(
            self,
            TensorClassification::Unknown | TensorClassification::Ignored
        )
    }
}

/// Map a checkpoint tensor name to its classification.
pub fn classify_tensor_name(name: &str) -> TensorClassification {
    let lower = name.to_lowercase();

    // Ignore metadata tensors
    if lower.contains("__metadata__") {
        return TensorClassification::Ignored;
    }

    // Decoder weights
    if lower.contains("self_attn.q_proj")
        || lower.contains("self_attn.k_proj")
        || lower.contains("self_attn.v_proj")
        || lower.contains("self_attn.o_proj")
        || lower.contains("mlp.gate_proj")
        || lower.contains("mlp.up_proj")
        || lower.contains("mlp.down_proj")
    {
        return TensorClassification::DecoderRequired;
    }

    // Text embeddings
    if lower.contains("embed_tokens") {
        return TensorClassification::TextEmbeddingRequired;
    }

    // LM head
    if lower.contains("lm_head_projection") || lower.contains("lm_head") {
        return TensorClassification::LmHeadRequired;
    }
    // Multimodal image (before Norms so pos_norm stays image)

    if lower.contains("vision_embedder") || lower.contains("embed_vision") {
        return TensorClassification::MultimodalImageRequired;
    }

    // Multimodal audio
    if lower.contains("embed_audio") {
        return TensorClassification::MultimodalAudioRequired;
    }

    // MTP (before Norms so mtp_norm stays MTP)
    if lower.contains("mtp_projection") || lower.contains("mtp_norm") {
        return TensorClassification::MtpRequired;
    }

    // Norms
    if lower.contains("layernorm")
        || lower.contains("layer_scalar")
        || lower.contains("q_norm")
        || lower.contains("k_norm")
        || lower.contains("norm")
    {
        return TensorClassification::NormRequired;
    }

    TensorClassification::Unknown
}

/// Schema describing a Gemma 4 Unified model instance.
#[derive(Debug, Clone)]
pub struct Gemma4UnifiedSchema {
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

impl Gemma4UnifiedSchema {
    /// Create the known schema for Gemma 4 12B Unified.
    pub fn gemma4_12b_unified() -> Self {
        Self {
            model_revision: String::new(), // filled from checkpoint
            hidden_size: GEMMA4_12B_UNIFIED_HIDDEN_SIZE,
            num_layers: GEMMA4_12B_UNIFIED_NUM_LAYERS,
            num_attention_heads: GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS,
            num_key_value_heads: GEMMA4_12B_UNIFIED_NUM_KV_HEADS,
            vocabulary_size: GEMMA4_12B_UNIFIED_VOCABULARY_SIZE,
            supports_text: true,
            supports_image: true,
            supports_audio: true,
            image_contract: None,
            audio_contract: None,
        }
    }

    /// Validate that a loaded checkpoint matches the expected 12B Unified architecture.
    pub fn validate_architecture(&self) -> Result<(), String> {
        if self.hidden_size != GEMMA4_12B_UNIFIED_HIDDEN_SIZE {
            return Err(format!(
                "hidden_size mismatch: expected {}, got {}",
                GEMMA4_12B_UNIFIED_HIDDEN_SIZE, self.hidden_size
            ));
        }
        if self.num_layers != GEMMA4_12B_UNIFIED_NUM_LAYERS {
            return Err(format!(
                "num_layers mismatch: expected {}, got {}",
                GEMMA4_12B_UNIFIED_NUM_LAYERS, self.num_layers
            ));
        }
        if self.num_attention_heads != GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS {
            return Err(format!(
                "num_attention_heads mismatch: expected {}, got {}",
                GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS, self.num_attention_heads
            ));
        }
        if self.num_key_value_heads != GEMMA4_12B_UNIFIED_NUM_KV_HEADS {
            return Err(format!(
                "num_key_value_heads mismatch: expected {}, got {}",
                GEMMA4_12B_UNIFIED_NUM_KV_HEADS, self.num_key_value_heads
            ));
        }
        if self.vocabulary_size != GEMMA4_12B_UNIFIED_VOCABULARY_SIZE {
            return Err(format!(
                "vocabulary_size mismatch: expected {}, got {}",
                GEMMA4_12B_UNIFIED_VOCABULARY_SIZE, self.vocabulary_size
            ));
        }
        if !self.supports_image && !self.supports_audio {
            return Err(
                "schema must support at least one non-text modality for Gemma4Unified".into(),
            );
        }
        Ok(())
    }

    /// Reject any checkpoint that exposes a separate vision tower.
    pub fn reject_legacy_vision_tower(&self, tensor_names: &[String]) -> Result<(), String> {
        for name in tensor_names {
            let lower = name.to_lowercase();
            if lower.contains("vision_tower")
                || lower.contains("siglip")
                || lower.contains("vit")
                || (lower.contains("clip") && lower.contains("vision"))
            {
                return Err(format!(
                    "legacy vision tower tensor detected in Gemma4Unified checkpoint: {}",
                    name
                ));
            }
        }
        Ok(())
    }
}
