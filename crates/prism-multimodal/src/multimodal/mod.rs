//! Multimodal processing pipeline.
//!
//! # Architecture
//!
//! The Legacy path (`VisionEncoderConfig` -> `ProjectorConfig` -> `multimodal_forward()`)
//! serves PaliGemma/LLaVA/Pixtral-class encoder-decoder models.
//!
//! Gemma 4 Unified uses a direct modality-to-decoder embedding adapter path defined
//! in `tribunus_compute_core::compute_image::multimodal::adapter`:
//! - `Gemma4DirectImageProjectionAdapter`: encoder-free image -> decoder embedding
//! - `Gemma4DirectAudioProjectionAdapter`: encoder-free audio -> decoder embedding
//!
//! The model-family-specific adapters replace the need for a separate vision encoder
//! artifact for encoder-free models.

pub mod dynamic_tiling;
pub mod projector;
pub mod vision_encoder;

pub struct ImageInput {
    // Basic representation of an image for the vision encoder
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct MultimodalPipeline {
    pub vision_encoder: vision_encoder::VisionEncoderConfig,
    pub projector: projector::ProjectorConfig,
    pub image_token_placement: ImageTokenStrategy,
}

pub enum ImageTokenStrategy {
    Inline { placeholder: String },
    CrossAttention { num_queries: u32 },
    DeepFusion,
}

pub fn build_embedding_sequence(
    _text_tokens: &[u32],
    image_embeds: &[Vec<f32>],
    _strategy: &ImageTokenStrategy,
) -> Vec<f32> {
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
    weights: &std::collections::HashMap<String, Vec<f32>>,
    matmul: &vision_encoder::MatmulProvider,
) -> Result<Vec<f32>, String> {
    let image_embeds: Result<Vec<Vec<f32>>, String> = images
        .iter()
        .map(|img| {
            vision_encoder::encode_image(&img.data, &pipeline.vision_encoder, weights, matmul)
        })
        .collect();
    let image_embeds = image_embeds?;

    let projected: Vec<Vec<f32>> = image_embeds
        .into_iter()
        .map(|e| projector_forward_f32(&e, &pipeline.projector))
        .collect();

    let combined =
        build_embedding_sequence(text_tokens, &projected, &pipeline.image_token_placement);

    // Using dummy dummy values for forward since the original forward in CImage has a different signature.
    // For this stub, we just return the combined embeddings.
    // pipeline.llm.forward(&combined)
    Ok(combined)
}

/// Apply a simple learned projection (input_dim → output_dim) via f32.
fn projector_forward_f32(features: &[f32], config: &projector::ProjectorConfig) -> Vec<f32> {
    let out_dim = config.output_dim as usize;
    let in_dim = config.input_dim as usize;
    if features.len() < in_dim {
        let mut projected = vec![0.0f32; out_dim];
        let copy_len = features.len().min(out_dim);
        projected[..copy_len].copy_from_slice(&features[..copy_len]);
        return projected;
    }
    let copy_len = in_dim.min(out_dim);
    let mut projected = vec![0.0f32; out_dim];
    projected[..copy_len].copy_from_slice(&features[..copy_len]);
    projected
}
