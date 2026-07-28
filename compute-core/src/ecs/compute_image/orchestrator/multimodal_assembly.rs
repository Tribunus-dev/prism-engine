//! Mixed embedding assembly for multimodal prefill.
//!
//! Assembles text token embeddings and projected image/audio embeddings
//! into a single decoder-width input sequence.

use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::InputModality;

/// One part of a multimodal prompt.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum PromptPart {
    /// Text tokens to be looked up via the vocabulary embedding table.
    Text(Vec<u32>),
    /// Image reference — raw pixels with dimensions.
    Image(ImageInputRef),
    /// Audio reference — raw samples with sample rate.
    Audio(AudioInputRef),
}

/// Reference to image input data (not owned, for zero-copy).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImageInputRef {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    /// Byte-offset into a shared pixel buffer.
    pub pixel_offset: u64,
    /// Number of bytes in this image's pixel data.
    pub pixel_len: u64,
}

/// Reference to audio input data.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AudioInputRef {
    pub sample_rate: u32,
    pub num_samples: u32,
    /// Byte-offset into a shared sample buffer.
    pub sample_offset: u64,
    /// Number of bytes in this audio segment.
    pub sample_len: u64,
}

/// Describes where a modality's embeddings live in the decoder input.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecoderEmbeddingSpan {
    /// Absolute position in the assembled sequence.
    pub position_start: u32,
    /// Number of positions this span occupies.
    pub position_len: u32,
    /// Which modality produced these embeddings.
    pub modality: InputModality,
    /// Byte offset into the shared embedding arena.
    pub embedding_offset: u64,
    /// Byte stride between consecutive embedding vectors.
    pub embedding_stride_bytes: u32,
}

/// Metadata for image positional encoding in attention.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImagePositionRange {
    /// Start position in the assembled decoder sequence.
    pub sequence_start: u32,
    /// Number of positions occupied by this image.
    pub sequence_len: u32,
    /// Width of the image in patches.
    pub image_width_patches: u32,
    /// Height of the image in patches.
    pub image_height_patches: u32,
    /// X offset in the global image position grid.
    pub x_position_offset: u32,
    /// Y offset in the global image position grid.
    pub y_position_offset: u32,
    /// Number of soft tokens produced by pooling.
    pub soft_token_count: u32,
}

/// Layout hint for the attention kernel.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionLayout {
    /// Text-only sequence — standard causal attention.
    TextOnly,
    /// Text + image spans — image tokens attend to prefix text, not each other
    /// (unless the model contract specifies image-image attention).
    MultimodalPrefill,
    /// Multimodal prefill with cross-modal attention.
    CrossModal,
}

/// Complete plan for assembling a multimodal prompt into decoder input.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MultimodalPromptPlan {
    /// Total number of embedding vectors in the assembled sequence.
    pub sequence_len: u32,
    /// Ordered spans describing where each modality's embeddings live.
    pub embedding_spans: Vec<DecoderEmbeddingSpan>,
    /// Attention layout for this prompt.
    pub attention_layout: AttentionLayout,
    /// Per-image position metadata for 2D RoPE.
    pub image_positions: Vec<ImagePositionRange>,
}

impl MultimodalPromptPlan {
    #[allow(dead_code)]
    /// Create a text-only prompt plan.
    pub fn text_only(num_tokens: u32) -> Self {
        let span = DecoderEmbeddingSpan {
            position_start: 0,
            position_len: num_tokens,
            modality: InputModality::Text,
            embedding_offset: 0,
            embedding_stride_bytes: 0,
        };
        Self {
            sequence_len: num_tokens,
            embedding_spans: vec![span],
            attention_layout: AttentionLayout::TextOnly,
            image_positions: Vec::new(),
        }
    }

    /// Validate that all spans are contiguous and cover the full sequence.
    #[allow(dead_code)]
    pub fn validate(&self) -> Result<(), String> {
        if self.embedding_spans.is_empty() {
            return Err("empty embedding spans".into());
        }

        let mut expected_pos = 0u32;
        for span in &self.embedding_spans {
            if span.position_start != expected_pos {
                return Err(format!(
                    "gap at position {}: expected {}, got {}",
                    expected_pos, expected_pos, span.position_start
                ));
            }
            expected_pos += span.position_len;
        }

        if expected_pos != self.sequence_len {
            return Err(format!(
                "sequence_len mismatch: spans cover {} positions, plan declares {}",
                expected_pos, self.sequence_len
            ));
        }

        Ok(())
    }

    /// Check whether this plan requires multimodal capabilities.
    #[allow(dead_code)]
    pub fn is_multimodal(&self) -> bool {
        self.image_positions.iter().any(|_| true)
            || self
                .embedding_spans
                .iter()
                .any(|s| s.modality != InputModality::Text)
    }
}
