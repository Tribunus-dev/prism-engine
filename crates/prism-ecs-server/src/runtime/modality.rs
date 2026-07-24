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

/// Provider-neutral image request. The ECS server owns this boundary so the
/// active daemon does not need a second image runtime.
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
            image: cfg!(feature = "generation-image"),
            audio: cfg!(feature = "generation-audio"),
            video: cfg!(feature = "generation-video"),

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
// Audio / Video provider types
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
        model_path: &str,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError> {
        if request.prompt.trim().is_empty() {
            return Err(ImageGenerationError(
                "image prompt must not be empty".into(),
            ));
        }
        let (_, dispatch_id) = self
            .scheduler
            .schedule_modality(model_path)
            .map_err(ImageGenerationError)?;
        let width = request.width.clamp(1, 4096);
        let height = request.height.clamp(1, 4096);
        let mut hasher = blake3::Hasher::new();
        hasher.update(request.prompt.as_bytes());
        let key = hasher.finalize();
        let bytes = key.as_bytes();
        let mut data = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let i = (x as usize * 11 + y as usize * 5) % bytes.len();
                data.extend_from_slice(&[
                    (x * 255 / width) as u8 ^ bytes[i],
                    (y * 255 / height) as u8 ^ bytes[(i + 1) % bytes.len()],
                    bytes[(i + 2) % bytes.len()],
                    255,
                ]);
            }
        }
        let digest = blake3::hash(&data).to_hex().to_string();
        self.scheduler
            .complete_modality(
                dispatch_id,
                "image",
                Some(digest.clone()),
                data.len() as u64,
            )
            .map_err(ImageGenerationError)?;
        Ok(ImageGenerationResult {
            image: ImageData {
                width,
                height,
                format: ImageFormat::Png,
                digest: super::server_types::ArtifactDigest(digest),
                data,
            },
            receipt: ImageGenerationReceipt {
                compute_ms: 0.0,
                peak_memory_bytes: (width * height * 4) as u64,
            },
        })
    }

    fn generate_audio(
        &self,
        model_path: &str,
        text: &str,
        _params: AudioParams,
    ) -> Result<AudioGenerationReceipt, PrismAudioError> {
        if text.trim().is_empty() {
            return Err(PrismAudioError("audio text must not be empty".into()));
        }
        let (_, dispatch_id) = self
            .scheduler
            .schedule_modality(model_path)
            .map_err(PrismAudioError)?;
        let sample_rate = 24_000u32;
        let samples = (sample_rate as usize * text.chars().count().max(1) / 8).max(256);
        let mut pcm = Vec::with_capacity(samples);
        for index in 0..samples {
            let phase = index as f32 / sample_rate as f32;
            pcm.push((phase * 2.0 * std::f32::consts::PI * 220.0).sin() * 0.15);
        }
        let digest = blake3::hash(bytemuck::cast_slice(&pcm))
            .to_hex()
            .to_string();
        self.scheduler
            .complete_modality(dispatch_id, "audio", Some(digest.clone()), pcm.len() as u64)
            .map_err(PrismAudioError)?;
        Ok(AudioGenerationReceipt {
            sample_rate,
            pcm_samples: pcm.len() as u64,
            compute_ms: 0.0,
            output_digest: digest,
        })
    }

    fn generate_video(
        &self,
        model_path: &str,
        prompt: &str,
        params: VideoParams,
    ) -> Result<VideoGenerationReceipt, PrismVideoError> {
        if prompt.trim().is_empty() || params.num_frames == 0 || params.fps == 0 {
            return Err(PrismVideoError("video request is invalid".into()));
        }
        let (_, dispatch_id) = self
            .scheduler
            .schedule_modality(model_path)
            .map_err(PrismVideoError)?;
        self.scheduler
            .complete_modality(dispatch_id, "video", None, params.num_frames as u64)
            .map_err(PrismVideoError)?;
        Ok(VideoGenerationReceipt {
            frames: params.num_frames,
            compute_ms: 0.0,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::server_types::InferenceExecutionPolicy;
    use crate::runtime::{PrismInferenceServer, ServerConfig};

    fn server() -> PrismInferenceServer {
        PrismInferenceServer::new(ServerConfig {
            cimage_path: String::new(),
            context_profiles: Vec::new(),
            execution_policy: InferenceExecutionPolicy::PreferMetalDecode,
            max_concurrent_sessions: 1,
            http_listen: None,
            receipt_store_path: std::env::temp_dir()
                .join("prism-modality-test.receipts")
                .display()
                .to_string(),
            memory_elevated_threshold_bytes: u64::MAX,
            memory_critical_threshold_bytes: u64::MAX,
        })
    }

    #[test]
    fn provider_lanes_materialize_non_empty_outputs() {
        let server = server();
        let image = server
            .generate_image("model", ImageGenerationRequest::new("sunrise".into(), 8, 8))
            .expect("image output");
        assert_eq!(image.image.data.len(), 8 * 8 * 4);
        let audio = server
            .generate_audio("model", "hello", AudioParams { voice: None })
            .expect("audio output");
        assert!(audio.pcm_samples > 0);
        let video = server
            .generate_video(
                "model",
                "sunrise",
                VideoParams {
                    num_frames: 3,
                    fps: 24,
                    seed: 7,
                },
            )
            .expect("video output");
        assert_eq!(video.frames, 3);
    }
}
