//! Generation modules for multimodal model inference.
//!
//! Provides generators for text-to-image, text-to-speech, audio-to-text,
//! image-to-image, audio-to-audio, video generation, and the DiffusionGemma
//! diffusion language model for parallel text generation, image understanding,
//! function calling, code generation, and reasoning.

#[cfg(not(feature = "prism-backend"))]
pub mod text_to_image;
#[cfg(not(feature = "prism-backend"))]
pub use text_to_image::TextToImageGenerator;

pub mod diffusiongemma;
pub use diffusiongemma::{
    AdaptiveParallelTokens, ChatCompletion, ChatMessage, ContentPart, DiffusionModel,
    DiffusionSampler, FunctionCall, ToolDefinition, UsageInfo,
};

#[cfg(not(feature = "prism-backend"))]
pub mod audio_to_text;
#[cfg(not(feature = "prism-backend"))]
pub use audio_to_text::AudioToTextGenerator;

pub mod image_to_image;

pub mod audio_to_audio;

#[cfg(not(feature = "prism-backend"))]
pub mod text_to_speech;

pub mod video_generation;
