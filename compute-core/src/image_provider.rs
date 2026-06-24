//! Provider trait for text-to-image generation.
//!
//! This trait decouples the Prism facade (product types, lifecycle, receipts)
//! from the Compute implementation (TextToImageGenerator or future LUT path).
//!
//! # Architecture
//!
//! ```text
//! PrismFacade (prism-engine/src/image/)
//!   │  translates PrismImageRequest → ImageGenerationRequest
//!   │  translates ImageGenerationResult → ImageGenerationReceipt
//!   ▼
//! ImageGenerationProvider (this module)
//!   │  trait — exactly one method
//!   ├── TextToImageProvider  (Compute MLX path)
//!   └── LutImageProvider     (future — palettized LUT path)
//! ```

use std::path::Path;

use crate::generation::text_to_image::TextToImageGenerator;

// ── Request / Result ─────────────────────────────────────────────────────

/// Canonical generation request understood by every provider.
#[derive(Clone, Debug)]
pub struct ImageGenerationRequest {
    /// Path to the compiled ComputeImage directory.
    pub model_path: String,
    /// Text prompt.
    pub prompt: String,
    /// Denoising steps. `None` = provider default (typically 4 for schnell).
    pub steps: Option<u32>,
    /// Output resolution. `None` = provider default (typically 1024×1024).
    pub size: Option<(u32, u32)>,
}

/// Canonical generation result returned by every provider.
#[derive(Debug)]
pub struct ImageGenerationResult {
    pub width: u32,
    pub height: u32,
    /// Flat RGBA8888 pixel buffer (width × height × 4 bytes).
    pub rgba: Vec<u8>,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

// ── Error ────────────────────────────────────────────────────────────────

/// Provider-level errors.
#[derive(Debug, thiserror::Error)]
pub enum ImageGenerationError {
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("unsupported model type for text-to-image: {0}")]
    UnsupportedModelType(String),
}

// ── Trait ────────────────────────────────────────────────────────────────

/// A provider that can generate images from text prompts.
pub trait ImageGenerationProvider: Send + Sync {
    /// Generate an image, returning pixel data and timing.
    fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError>;
}

// ── MLX-backed implementation ───────────────────────────────────────────

/// Wraps [`TextToImageGenerator`] behind the [`ImageGenerationProvider`] trait.
///
/// Owns the loaded model for the lifetime of the provider.
pub struct TextToImageProvider {
    inner: TextToImageGenerator,
}

impl TextToImageProvider {
    /// Load a model and wrap it.
    ///
    /// Fails with `ModelNotFound` if `model_path` does not contain a valid
    /// compiled ComputeImage.
    pub fn new(model_path: &str) -> Result<Self, ImageGenerationError> {
        let p = Path::new(model_path);
        if !p.join("manifest.json").exists() {
            return Err(ImageGenerationError::ModelNotFound(model_path.to_string()));
        }
        let inner = TextToImageGenerator::load(model_path)
            .map_err(|e| ImageGenerationError::GenerationFailed(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Access the underlying generator (for advanced usage).
    pub fn inner(&self) -> &TextToImageGenerator {
        &self.inner
    }
}

impl ImageGenerationProvider for TextToImageProvider {
    fn generate(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResult, ImageGenerationError> {
        let t0 = std::time::Instant::now();

        let (width, height, rgba) = self
            .inner
            .generate(&request.prompt, request.steps, request.size)
            .map_err(|e| ImageGenerationError::GenerationFailed(e.to_string()))?;

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(ImageGenerationResult {
            width,
            height,
            rgba,
            compute_ms,
        })
    }
}
