use serde::Serialize;

// ── Prism LLM Inference — Multimodal Modality Provider ─────────────────────
//
// Defines a [`ModalityProvider`] trait and implements it on
// [`PrismInferenceServer`], delegating each modality to the appropriate
// Prism facade:
//
//   - `generate_image`     → stub (TODO: crate::image when ported)
//   - `generate_audio`     → stub (TODO: crate::audio when ported)
//   - `generate_video`     → stub (TODO: crate::video when ported)
//   - `generate_embeddings`→ (placeholder — delegates to compute-core)
//
// Every generation method is gated behind its respective feature flag.
// When the feature is disabled the method returns a structured error.

// TODO: import from prism-image when ported
// use crate::image::{ImageGenerationError, ImageGenerationRequest, ImageGenerationResult};

/// Stub types for image generation — TODO: replace with prism-image types.
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub batch_size: u32,
}

impl ImageGenerationRequest {
    pub fn new(prompt: String, width: u32, height: u32) -> Self {
        Self {
            prompt,
            width,
            height,
            batch_size: 1,
        }
    }
}

pub struct ImageGenerationResult {
    pub image: ImageData,
    pub receipt: ImageGenerationReceipt,
}

pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub digest: super::server_types::ArtifactDigest,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationReceipt {
    pub compute_ms: f64,
    pub peak_memory_bytes: u64,
}

#[derive(Debug)]
pub struct ImageGenerationError(pub String);

impl std::fmt::Display for ImageGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ImageGenerationError {}

/// Describes modalities the current build supports.
#[derive(Debug, Clone)]
pub struct ModalityCapabilities {
    /// Whether image generation (`generation-image` feature) is available.
    pub image: bool,
    /// Whether audio/speech generation (`generation-audio` feature) is available.
    pub audio: bool,
    /// Whether video generation (`generation-video` feature) is available.
    pub video: bool,
    /// Whether embedding generation is available.
    pub embeddings: bool,
    /// Whether multimodal (combined vision+text) inference is available.
    pub multimodal: bool,
}

impl ModalityCapabilities {
    /// Probe the active feature flags to determine which modalities are compiled in.
    pub fn current() -> Self {
        Self {
            #[cfg(feature = "generation-image")]
            image: true,
            #[cfg(not(feature = "generation-image"))]
            image: false,

            #[cfg(feature = "generation-audio")]
            audio: true,
            #[cfg(not(feature = "generation-audio"))]
            audio: false,

            #[cfg(feature = "generation-video")]
            video: true,
            #[cfg(not(feature = "generation-video"))]
            video: false,

            embeddings: false,
            multimodal: cfg!(feature = "prism-backend"),
        }
    }

    /// Return the modality capability names as a list of strings.
    pub fn active_capabilities(&self) -> Vec<&'static str> {
        let mut caps = Vec::new();
        caps.push("llm-inference");
        if self.image {
            caps.push("image-generation");
        }
        if self.audio {
            caps.push("audio-speech");
        }
        if self.video {
            caps.push("video-generation");
        }
        if self.embeddings {
            caps.push("embeddings");
        }
        if self.multimodal {
            caps.push("multimodal-inference");
        }
        caps
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trait
// ═══════════════════════════════════════════════════════════════════════════

/// Provider interface for non-text modalities.
///
/// Each method is unconditionally available at compile time.  When the
/// corresponding generation feature is not enabled the method returns
/// a structured error indicating the missing capability.
pub trait ModalityProvider {
    /// Generate an image from a text prompt.
    fn generate_image(
        &self,
        model_path: &str,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError>;

    /// Generate speech from text.
    fn generate_audio(
        &self,
        model_path: &str,
        text: &str,
        params: AudioParams,
    ) -> Result<AudioGenerationReceipt, PrismAudioError>;

    /// Generate a video from a text prompt.
    fn generate_video(
        &self,
        model_path: &str,
        prompt: &str,
        params: VideoParams,
    ) -> Result<VideoGenerationReceipt, PrismVideoError>;

    /// Generate text embeddings.
    fn generate_embeddings(&self, model_path: &str, text: &str) -> Result<Vec<f32>, String>;

    /// Report which modalities are available at compile time.
    fn capabilities(&self) -> ModalityCapabilities;
}

// ═══════════════════════════════════════════════════════════════════════════
// Audio / Video stub types
// ═══════════════════════════════════════════════════════════════════════════

// TODO: import from prism-audio when ported

/// Parameters for audio generation.
#[derive(Debug, Clone)]
pub struct AudioParams {
    pub voice: Option<String>,
}

/// Receipt for a completed audio generation.
#[derive(Debug, Clone, Serialize)]
pub struct AudioGenerationReceipt {
    pub sample_rate: u32,
    pub pcm_samples: u64,
    pub compute_ms: f64,
    pub output_digest: String,
}

/// Audio generation error.
#[derive(Debug)]
pub struct PrismAudioError(pub String);

impl std::fmt::Display for PrismAudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PrismAudioError {}

// TODO: import from prism-video when ported

/// Parameters for video generation.
#[derive(Debug, Clone)]
pub struct VideoParams {
    pub num_frames: u32,
    pub fps: u32,
    pub seed: u64,
}

/// Receipt for a completed video generation.
#[derive(Debug, Clone, Serialize)]
pub struct VideoGenerationReceipt {
    pub frames: u32,
    pub compute_ms: f64,
}

/// Video generation error.
#[derive(Debug)]
pub struct PrismVideoError(pub String);

impl std::fmt::Display for PrismVideoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PrismVideoError {}

// ═══════════════════════════════════════════════════════════════════════════
// PrismInferenceServer implementation
// ═══════════════════════════════════════════════════════════════════════════

use super::PrismInferenceServer;

impl ModalityProvider for PrismInferenceServer {
    fn generate_image(
        &self,
        _model_path: &str,
        _request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError> {
        // TODO: delegate to crate::image::generate_image when ported
        Err(ImageGenerationError(
            "image generation requires prism-image crate".into(),
        ))
    }

    fn generate_audio(
        &self,
        _model_path: &str,
        _text: &str,
        _params: AudioParams,
    ) -> Result<AudioGenerationReceipt, PrismAudioError> {
        // TODO: delegate to crate::audio::generate_speech when ported
        Err(PrismAudioError(
            "audio generation requires prism-audio crate".into(),
        ))
    }

    fn generate_video(
        &self,
        _model_path: &str,
        _prompt: &str,
        _params: VideoParams,
    ) -> Result<VideoGenerationReceipt, PrismVideoError> {
        // TODO: delegate to crate::video::generate_video when ported
        Err(PrismVideoError(
            "video generation requires prism-video crate".into(),
        ))
    }

    fn generate_embeddings(&self, _model_path: &str, _text: &str) -> Result<Vec<f32>, String> {
        #[cfg(feature = "prism-backend")]
        {
            // Delegate to compute-core embedding generation.
            Err("embedding generation requires a loaded embedding model".to_string())
        }
        #[cfg(not(feature = "prism-backend"))]
        {
            Err("embedding generation requires the `prism-backend` feature".to_string())
        }
    }

    fn capabilities(&self) -> ModalityCapabilities {
        ModalityCapabilities::current()
    }
}
