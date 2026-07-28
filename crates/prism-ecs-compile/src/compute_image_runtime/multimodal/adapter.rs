//! Multimodal adapter — pure data types for projection adapters.

use serde::{Deserialize, Serialize};

use super::descriptor::InputModality;

/// A prepared modality (ready for embedding lookup).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedModality {
    /// Modality kind.
    pub modality: InputModality,
    /// Soft token count.
    pub soft_token_count: u32,
    /// Embedding content hash.
    pub embedding_digest: String,
}

/// A modality that has been embedded (post projection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedModality {
    /// Modality kind.
    pub modality: InputModality,
    /// Embedded sequence length.
    pub sequence_len: u32,
    /// Hidden state content hash.
    pub hidden_state_digest: String,
}

/// Input to a modality adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityInput {
    /// Text input.
    Text {
        /// Text bytes.
        text: String,
    },
    /// Image input.
    Image {
        /// Image content hash.
        content_hash: String,
    },
    /// Audio input.
    Audio {
        /// Audio content hash.
        content_hash: String,
        /// Sample rate.
        sample_rate: u32,
    },
}

/// Token embedding adapter.
#[derive(Debug, Clone, Default)]
pub struct TokenEmbeddingAdapter;

impl TokenEmbeddingAdapter {
    /// Create a new token embedding adapter.
    pub fn new() -> Self {
        Self
    }
}

/// Legacy vision encoder projector adapter.
#[derive(Debug, Clone, Default)]
pub struct LegacyVisionEncoderProjectorAdapter;

impl LegacyVisionEncoderProjectorAdapter {
    /// Create a new legacy vision encoder projector adapter.
    pub fn new() -> Self {
        Self
    }
}

/// Gemma4 direct image projection adapter.
#[derive(Debug, Clone, Default)]
pub struct Gemma4DirectImageProjectionAdapter;

impl Gemma4DirectImageProjectionAdapter {
    /// Create a new Gemma4 direct image projection adapter.
    pub fn new() -> Self {
        Self
    }
}

/// Gemma4 direct audio projection adapter.
#[derive(Debug, Clone, Default)]
pub struct Gemma4DirectAudioProjectionAdapter;

impl Gemma4DirectAudioProjectionAdapter {
    /// Create a new Gemma4 direct audio projection adapter.
    pub fn new() -> Self {
        Self
    }
}
