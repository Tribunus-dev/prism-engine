//! Multimodal processing pipeline.
//!
//! # Architecture
//!
//! The Legacy path (`VisionEncoderConfig` -> `ProjectorConfig` -> `multimodal_forward()`)
//! serves PaliGemma/LLaVA/Pixtral-class encoder-decoder models.
//!
//! Gemma 4 Unified uses a direct modality-to-decoder embedding adapter path defined
//! in `tribunus_compute_core::compute_image::multimodal::adapter`:
//! - `Gemma4DirectImageProjectionAdapter`: encoder-free image → decoder embedding
//! - `Gemma4DirectAudioProjectionAdapter`: encoder-free audio → decoder embedding
//!
//! The model-family-specific adapters replace the need for a separate vision encoder
//! artifact for encoder-free models.

pub mod dynamic_tiling;
pub mod projector;
pub mod vision_encoder;

use crate::lut::engine::PrismEngine;
use anyhow::Result;

pub struct ImageInput {
    // Basic representation of an image for the vision encoder
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

pub struct MultimodalPipeline {
    pub vision_encoder: vision_encoder::VisionEncoderConfig,
    pub projector: projector::ProjectorConfig,
    pub llm: PrismEngine,
    pub image_token_placement: ImageTokenStrategy,
}

pub enum ImageTokenStrategy {
    Inline { placeholder: String },
    CrossAttention { num_queries: u32 },
    DeepFusion,
}

pub fn build_embedding_sequence(
    _text_tokens: &[u32],
    image_embeds: &[Vec<u16>],
    _strategy: &ImageTokenStrategy,
) -> Vec<u16> {
    // Dummy implementation for building embedding sequence
    // In a real implementation, we would interleave these based on the strategy
    let mut combined = Vec::new();
    for embed in image_embeds {
        combined.extend_from_slice(embed);
    }
    // Note: This dummy ignores text_tokens to compile, a real implementation
    // would look up text_tokens in the LLM's embedding matrix.
    combined
}

pub fn multimodal_forward(
    text_tokens: &[u32],
    images: &[ImageInput],
    pipeline: &mut MultimodalPipeline,
) -> Result<Vec<u16>> {
    let image_embeds: Vec<Vec<u16>> = images
        .iter()
        .map(|img| pipeline.vision_encoder.encode(img))
        .collect();

    let projected: Vec<Vec<u16>> = image_embeds
        .into_iter()
        .map(|e| pipeline.projector.forward(&e))
        .collect();

    let combined =
        build_embedding_sequence(text_tokens, &projected, &pipeline.image_token_placement);

    // Using dummy dummy values for forward since the original forward in CImage has a different signature.
    // For this stub, we just return the combined embeddings.
    // pipeline.llm.forward(&combined)
    Ok(combined)
}
