// ═══════════════════════════════════════════════════════════════════════════
// Prism Diffusion Generation Facade
// ═══════════════════════════════════════════════════════════════════════════
//
// Stable public API for diffusion text generation.  Translates Prism-level
// request types into provider implementations and wraps results in Prism
// receipts with full provenance.

use std::time::Duration;

/// Parameters for a diffusion generation request.
#[derive(Debug, Clone)]
pub struct DiffusionParams {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Number of denoising steps. `None` = provider default.
    pub steps: Option<u32>,
}

impl Default for DiffusionParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            steps: None,
        }
    }
}

/// Full provenance receipt for a diffusion generation.
#[derive(Debug, Clone)]
pub struct DiffusionGenerationReceipt {
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// Provider that served the generation.
    pub provider: &'static str,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
    /// Actual device used.
    pub actual_device: String,
    /// Duration of the generation.
    pub duration: Duration,
    /// Whether a fallback was used.
    pub fallback_used: bool,
}

/// Diffusion generation errors.
#[derive(Debug, thiserror::Error)]
pub enum PrismDiffusionError {
    #[error("diffusion text generation requires the `generation-diffusion` feature")]
    MissingFeature,
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("CImage does not contain diffusion generation capability")]
    UnsupportedCImage,
}

/// Generate text via diffusion sampling.
///
/// Entry point for the Prism diffusion generation facade.  Always available at
/// compile time; returns `MissingFeature` when the `generation-diffusion` feature
/// is not enabled.
pub fn generate_text(
    model_path: &str,
    prompt: &str,
    params: DiffusionParams,
) -> Result<DiffusionGenerationReceipt, PrismDiffusionError> {
    #[cfg(feature = "generation-diffusion")]
    {
        generate_via_compute_core(model_path, prompt, params)
    }
    #[cfg(not(feature = "generation-diffusion"))]
    {
        let _ = (model_path, prompt, params);
        Err(PrismDiffusionError::MissingFeature)
    }
}

#[cfg(feature = "generation-diffusion")]
fn generate_via_compute_core(
    model_path: &str,
    prompt: &str,
    params: DiffusionParams,
) -> Result<DiffusionGenerationReceipt, PrismDiffusionError> {
    use tribunus_compute_core::diffusion_provider::{
        DiffusionGenerationProvider, DiffusionGenerationRequest, DiffusionProvider,
    };

    let provider = DiffusionProvider::new(model_path)
        .map_err(|e| PrismDiffusionError::ModelNotFound(e.to_string()))?;

    let request = DiffusionGenerationRequest {
        model_path: model_path.to_string(),
        prompt: prompt.to_string(),
        max_tokens: params.max_tokens,
        steps: params.steps,
    };

    let t0 = std::time::Instant::now();
    let result = provider
        .generate_text(request)
        .map_err(|e| PrismDiffusionError::GenerationFailed(e.to_string()))?;
    let elapsed = t0.elapsed();

    Ok(DiffusionGenerationReceipt {
        tokens: result.tokens,
        provider: "compute-mlx",
        compute_ms: result.compute_ms,
        actual_device: "apple-gpu".to_string(),
        duration: elapsed,
        fallback_used: false,
    })
}
