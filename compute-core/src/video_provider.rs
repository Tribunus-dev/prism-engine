//! Provider trait for text-to-video generation.
//!
//! This trait decouples the Prism facade (product types, lifecycle, receipts)
//! from the Compute implementation (VideoGenerator or future temporal-model path).
//!
//! # Architecture
//!
//! ```text
//! PrismFacade (prism-engine/src/video/)
//!   │  translates PrismVideoRequest → VideoGenerationRequest
//!   │  translates VideoGenerationResult → VideoGenerationReceipt
//!   ▼
//! VideoGenerationProvider (this module)
//!   │  trait — exactly one method
//!   └── VideoProvider  (Compute MLX path — per-frame text-to-image + temporal)
//! ```

use std::path::Path;
use std::sync::Arc;

use crate::generation::video_generation::{TextToImageGenerator, VideoGenerator};

// ── Request / Result ─────────────────────────────────────────────────────

/// Canonical text-to-video generation request understood by every provider.
#[derive(Clone, Debug)]
pub struct VideoGenerationRequest {
    /// Path to the compiled ComputeVideo model directory.
    pub model_path: String,
    /// Text prompt describing the video content.
    pub prompt: String,
    /// Number of frames to generate.
    pub num_frames: u32,
    /// Target frame rate (stored as metadata; does not directly affect
    /// the per-frame generation loop in the stub).
    pub fps: u32,
    /// Base seed for deterministic frame sequences.  Each frame uses
    /// `seed + frame_index` for reproducible variation.
    pub seed: u64,
}

/// Canonical text-to-video generation result returned by every provider.
#[derive(Debug)]
pub struct VideoGenerationResult {
    /// Generated frames in temporal order.  Each entry is
    /// `(width, height, flat RGBA8888 pixel data)`.
    pub frames: Vec<(u32, u32, Vec<u8>)>,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

// ── Error ────────────────────────────────────────────────────────────────

/// Provider-level errors.
#[derive(Debug, thiserror::Error)]
pub enum VideoGenerationError {
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
}

// ── Trait ────────────────────────────────────────────────────────────────

/// A provider that can generate video (sequences of frames) from text prompts.
pub trait VideoGenerationProvider: Send + Sync {
    /// Generate a video, returning frame data and timing.
    fn generate_text_to_video(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResult, VideoGenerationError>;
}

// ── MLX-backed implementation ───────────────────────────────────────────

/// Wraps [`VideoGenerator`] behind the [`VideoGenerationProvider`] trait.
///
/// Owns the loaded model for the lifetime of the provider.
pub struct VideoProvider {
    inner: VideoGenerator,
}

impl VideoProvider {
    /// Load a model and wrap it.
    ///
    /// Fails with `ModelNotFound` if `model_path` does not contain a valid
    /// compiled ComputeVideo manifest.
    pub fn new(model_path: &str) -> Result<Self, VideoGenerationError> {
        let p = Path::new(model_path);
        if !p.join("manifest.json").exists() {
            return Err(VideoGenerationError::ModelNotFound(model_path.to_string()));
        }
        let frame_gen = TextToImageGenerator::new(None);
        let inner = VideoGenerator::new(Arc::new(frame_gen));
        Ok(Self { inner })
    }

    /// Access the underlying generator (for advanced usage).
    pub fn inner(&self) -> &VideoGenerator {
        &self.inner
    }
}

impl VideoGenerationProvider for VideoProvider {
    fn generate_text_to_video(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResult, VideoGenerationError> {
        let t0 = std::time::Instant::now();

        let frames = self
            .inner
            .text_to_video(
                &request.prompt,
                request.num_frames,
                request.fps,
                request.seed,
            )
            .map_err(|e| VideoGenerationError::GenerationFailed(e.to_string()))?;

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(VideoGenerationResult { frames, compute_ms })
    }
}
