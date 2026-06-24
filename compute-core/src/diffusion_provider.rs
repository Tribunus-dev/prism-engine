//! Provider trait for diffusion text generation.
//!
//! This trait decouples the Prism facade (product types, lifecycle, receipts)
//! from the Compute implementation (DiffusionSampler).
//!
//! # Architecture
//!
//! ```text
//! PrismFacade (prism-engine/src/diffusion/)
//!   │  translates PrismDiffusionParams → DiffusionGenerationRequest
//!   │  translates DiffusionGenerationResult → DiffusionGenerationReceipt
//!   ▼
//! DiffusionGenerationProvider (this module)
//!   │  trait — exactly one method
//!   └── DiffusionProvider  (Compute MLX path)
//! ```

use crate::generation::diffusiongemma::DiffusionSampler;

// ── Request / Result ─────────────────────────────────────────────────────

/// Canonical diffusion generation request understood by every provider.
#[derive(Clone, Debug)]
pub struct DiffusionGenerationRequest {
    /// Path to the compiled ComputeImage directory.
    pub model_path: String,
    /// Text prompt.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Number of denoising steps. `None` = provider default.
    pub steps: Option<u32>,
}

/// Canonical generation result returned by every provider.
#[derive(Debug)]
pub struct DiffusionGenerationResult {
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

// ── Error ────────────────────────────────────────────────────────────────

/// Provider-level errors.
#[derive(Debug, thiserror::Error)]
pub enum DiffusionGenerationError {
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("unsupported model type for diffusion generation: {0}")]
    UnsupportedModelType(String),
}

// ── Trait ────────────────────────────────────────────────────────────────

/// A provider that can generate text via diffusion sampling.
pub trait DiffusionGenerationProvider: Send + Sync {
    /// Generate text tokens from a prompt, returning token ids and timing.
    fn generate_text(
        &self,
        request: DiffusionGenerationRequest,
    ) -> Result<DiffusionGenerationResult, DiffusionGenerationError>;
}

// ── MLX-backed implementation ───────────────────────────────────────────

/// Wraps [`DiffusionSampler`] behind the [`DiffusionGenerationProvider`] trait.
///
/// Owns the loaded model for the lifetime of the provider.
pub struct DiffusionProvider {
    model: crate::generation::diffusiongemma::DiffusionModel,
}

impl DiffusionProvider {
    /// Load a model and wrap it.
    ///
    /// Fails with `ModelNotFound` if `model_path` does not contain a valid
    /// compiled ComputeImage.
    pub fn new(model_path: &str) -> Result<Self, DiffusionGenerationError> {
        use std::path::Path;
        let p = Path::new(model_path);
        if !p.join("manifest.json").exists() {
            return Err(DiffusionGenerationError::ModelNotFound(
                model_path.to_string(),
            ));
        }
        let model = crate::generation::diffusiongemma::DiffusionModel::load(model_path)
            .map_err(|e| DiffusionGenerationError::GenerationFailed(e.to_string()))?;
        Ok(Self { model })
    }

    /// Access the underlying sampler (for advanced usage).
    pub fn sampler(&self) -> &DiffusionSampler {
        &self.model.sampler
    }
}

impl DiffusionGenerationProvider for DiffusionProvider {
    fn generate_text(
        &self,
        request: DiffusionGenerationRequest,
    ) -> Result<DiffusionGenerationResult, DiffusionGenerationError> {
        let t0 = std::time::Instant::now();

        // Tokenize prompt as simple byte tokens (matching DiffusionModel::chat).
        let prompt_tokens: Vec<u32> = request.prompt.bytes().map(|b| b as u32).collect();

        let tokens = self
            .model
            .sampler
            .generate(
                &self.model,
                &prompt_tokens,
                request.max_tokens,
                request.steps,
            )
            .map_err(|e| DiffusionGenerationError::GenerationFailed(e))?;

        let compute_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(DiffusionGenerationResult { tokens, compute_ms })
    }
}
