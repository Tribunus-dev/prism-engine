// ── Prism LLM Inference — Multimodal Modality Provider ─────────────────────
//
// Defines a [`ModalityProvider`] trait and implements it on
// [`PrismInferenceServer`], delegating each modality to the appropriate
// Prism facade:
//
//   - `generate_image`     → `crate::image::generate_image`
//   - `generate_audio`     → `crate::audio::generate_speech`
//   - `generate_video`     → `crate::video::generate_video`
//   - `generate_embeddings`→ (placeholder — delegates to compute-core)
//
// Every generation method is gated behind its respective feature flag.
// When the feature is disabled the method returns a structured error.

use crate::image::{ImageGenerationRequest, ImageGenerationResult, ImageGenerationError};

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
        params: crate::audio::AudioParams,
    ) -> Result<crate::audio::AudioGenerationReceipt, crate::audio::PrismAudioError>;

    /// Generate a video from a text prompt.
    fn generate_video(
        &self,
        model_path: &str,
        prompt: &str,
        params: crate::video::VideoParams,
    ) -> Result<crate::video::VideoGenerationReceipt, crate::video::PrismVideoError>;

    /// Generate text embeddings.
    fn generate_embeddings(
        &self,
        model_path: &str,
        text: &str,
    ) -> Result<Vec<f32>, String>;

    /// Report which modalities are available at compile time.
    fn capabilities(&self) -> ModalityCapabilities;
}

// ═══════════════════════════════════════════════════════════════════════════
// PrismInferenceServer implementation
// ═══════════════════════════════════════════════════════════════════════════

use super::PrismInferenceServer;

impl ModalityProvider for PrismInferenceServer {
    fn generate_image(
        &self,
        model_path: &str,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError> {
        crate::image::generate_image(model_path, request)
    }

    fn generate_audio(
        &self,
        model_path: &str,
        text: &str,
        params: crate::audio::AudioParams,
    ) -> Result<crate::audio::AudioGenerationReceipt, crate::audio::PrismAudioError> {
        crate::audio::generate_speech(model_path, text, params)
    }

    fn generate_video(
        &self,
        model_path: &str,
        prompt: &str,
        params: crate::video::VideoParams,
    ) -> Result<crate::video::VideoGenerationReceipt, crate::video::PrismVideoError> {
        crate::video::generate_video(model_path, prompt, params)
    }

    fn generate_embeddings(
        &self,
        _model_path: &str,
        _text: &str,
    ) -> Result<Vec<f32>, String> {
        #[cfg(feature = "prism-backend")]
        {
            // Delegate to compute-core embedding generation.
            // In a real deployment this would load an embedding model from
            // the model path and run a forward pass.
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
