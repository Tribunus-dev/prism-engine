use serde::{Deserialize, Serialize};
use strum::{Display, EnumIter, EnumString};

/// Classified matrix role for per-family profile selection.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString, EnumIter,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum MatrixRole {
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    FfnGate,
    FfnUp,
    FfnDown,
    Embedding,
    LmHead,
    MultimodalProjection,
    TtsTalker,
    TtsCodePredictor,
    TtsCodec,
    UnknownLinear,
}

impl MatrixRole {
    /// Return true if this role is part of the attention family.
    pub fn is_attention(self) -> bool {
        matches!(
            self,
            Self::AttentionQ | Self::AttentionK | Self::AttentionV | Self::AttentionO
        )
    }

    /// Return true if this role is part of the FFN family.
    pub fn is_ffn(self) -> bool {
        matches!(self, Self::FfnGate | Self::FfnUp | Self::FfnDown)
    }

    /// Return true if this role is a boundary/high-sensitivity tensor.
    pub fn is_boundary(self) -> bool {
        matches!(
            self,
            Self::Embedding | Self::LmHead | Self::MultimodalProjection
        )
    }

    /// Return true if this role is TTS/codec related.
    pub fn is_tts(self) -> bool {
        matches!(
            self,
            Self::TtsTalker | Self::TtsCodePredictor | Self::TtsCodec
        )
    }

    /// Default profile ID for this role.
    pub fn default_profile_id(self) -> super::profile::ProfileId {
        use super::profile::{
            PROFILE_ID_CANONICAL_NF4_V1, PROFILE_ID_GEMMA_ATTENTION_V1,
            PROFILE_ID_GEMMA_BOUNDARY_V1, PROFILE_ID_GEMMA_FFN_V1, PROFILE_ID_TTS_CODEC_V1,
        };
        if self.is_attention() {
            PROFILE_ID_GEMMA_ATTENTION_V1
        } else if self.is_ffn() {
            PROFILE_ID_GEMMA_FFN_V1
        } else if self.is_boundary() {
            PROFILE_ID_GEMMA_BOUNDARY_V1
        } else if self.is_tts() {
            PROFILE_ID_TTS_CODEC_V1
        } else {
            PROFILE_ID_CANONICAL_NF4_V1
        }
    }
}

/// Classify a tensor name (from source checkpoint) into a MatrixRole.
pub fn classify_matrix_role(tensor_name: &str) -> MatrixRole {
    let lower = tensor_name.to_lowercase();
    // Check multimodal/vision BEFORE generic "embed" so vision_embedder is not
    // misclassified as a language model Embedding.
    if lower.contains("multimodal")
        || lower.contains("vision")
        || lower.contains("audio_")
        || lower.contains("embed_audio")
        || lower.contains("embed_vision")
        || lower.contains("mm_")
    {
        MatrixRole::MultimodalProjection
    } else if lower.contains("embed_tokens")
        || lower.contains("embedding")
        || lower.contains("embed") && !lower.contains("rotary")
    {
        MatrixRole::Embedding
    } else if lower.contains("lm_head") || lower.contains("output") {
        MatrixRole::LmHead
    } else if lower.contains("q_proj") || lower.contains("qkv") || lower.ends_with("q.") {
        MatrixRole::AttentionQ
    } else if lower.contains("k_proj") || lower.ends_with("k.") {
        MatrixRole::AttentionK
    } else if lower.contains("v_proj") || lower.ends_with("v.") {
        MatrixRole::AttentionV
    } else if lower.contains("o_proj") || lower.contains("out_proj") || lower.ends_with("o.") {
        MatrixRole::AttentionO
    } else if lower.contains("gate_proj") || lower.contains("w1") || lower.contains("gate") {
        MatrixRole::FfnGate
    } else if lower.contains("up_proj") || lower.contains("w3") {
        MatrixRole::FfnUp
    } else if lower.contains("down_proj") || lower.contains("w2") {
        MatrixRole::FfnDown
    } else if lower.contains("tts_talker") || lower.contains("talker") {
        MatrixRole::TtsTalker
    } else if lower.contains("tts_code_predictor")
        || lower.contains("code_predictor")
        || lower.contains("cp_")
    {
        MatrixRole::TtsCodePredictor
    } else if lower.contains("codec") || lower.contains("mimi") {
        MatrixRole::TtsCodec
    } else {
        MatrixRole::UnknownLinear
    }
}
