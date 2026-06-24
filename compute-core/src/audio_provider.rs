//! Provider trait for text-to-speech generation.
//!
//! This trait decouples the Prism facade (product types, lifecycle, receipts)
//! from the Compute implementation (TextToSpeechGenerator or future path).
//!
//! # Architecture
//!
//! ```text
//! PrismFacade (prism-engine/src/audio/)
//!   │  translates AudioParams + text → AudioGenerationRequest
//!   │  translates AudioGenerationResult → AudioGenerationReceipt
//!   ▼
//! AudioGenerationProvider (this module)
//!   │  trait — exactly one method
//!   ├── TextToSpeechProvider  (Compute MLX / qwen3-tts-mlx path)
//!   └── (future providers)
//! ```

use std::path::Path;

use crate::generation::text_to_speech::TextToSpeechGenerator;

// ── Request / Result ─────────────────────────────────────────────────────

/// Canonical generation request understood by every audio provider.
#[derive(Clone, Debug)]
pub struct AudioGenerationRequest {
    /// Path to the compiled model directory.
    pub model_path: String,
    /// Text to synthesize.
    pub text: String,
    /// Optional voice preset (e.g. "vivian", "bella").
    pub voice: Option<String>,
}

/// Canonical generation result returned by every audio provider.
#[derive(Debug)]
pub struct AudioGenerationResult {
    /// Sample rate of the generated audio (e.g. 24000 Hz).
    pub sample_rate: u32,
    /// Mono PCM f32 samples in [-1.0, 1.0].
    pub pcm_samples: Vec<f32>,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

// ── Error ────────────────────────────────────────────────────────────────

/// Provider-level errors.
#[derive(Debug, thiserror::Error)]
pub enum AudioGenerationError {
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("unsupported model type for text-to-speech: {0}")]
    UnsupportedModelType(String),
}

// ── Trait ────────────────────────────────────────────────────────────────

/// A provider that can generate speech from text.
pub trait AudioGenerationProvider: Send + Sync {
    /// Generate speech, returning PCM samples and timing.
    fn generate_speech(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResult, AudioGenerationError>;
}

// ── MLX-backed implementation ───────────────────────────────────────────

/// Wraps [`TextToSpeechGenerator`] behind the [`AudioGenerationProvider`] trait.
///
/// Owns the loaded model and a tokio runtime for the lifetime of the provider.
pub struct TextToSpeechProvider {
    inner: TextToSpeechGenerator,
    rt: tokio::runtime::Runtime,
}

impl TextToSpeechProvider {
    /// Load a model and wrap it.
    ///
    /// Fails with `ModelNotFound` if `model_path` does not contain a valid
    /// qwen3-tts-mlx model directory.
    pub fn new(model_path: &str) -> Result<Self, AudioGenerationError> {
        let p = Path::new(model_path);
        if !p.join("config.json").exists() {
            return Err(AudioGenerationError::ModelNotFound(model_path.to_string()));
        }
        let inner = TextToSpeechGenerator::load(model_path)
            .map_err(|e| AudioGenerationError::GenerationFailed(e.to_string()))?;
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| AudioGenerationError::GenerationFailed(e.to_string()))?;
        Ok(Self { inner, rt })
    }

    /// Access the underlying generator (for advanced usage).
    pub fn inner(&self) -> &TextToSpeechGenerator {
        &self.inner
    }
}

impl AudioGenerationProvider for TextToSpeechProvider {
    fn generate_speech(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResult, AudioGenerationError> {
        let t0 = std::time::Instant::now();

        let (sample_rate, pcm_samples) = self
            .rt
            .block_on(
                self.inner
                    .synthesize(&request.text, request.voice.as_deref()),
            )
            .map_err(|e| AudioGenerationError::GenerationFailed(e.to_string()))?;

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(AudioGenerationResult {
            sample_rate,
            pcm_samples,
            compute_ms,
        })
    }
}
