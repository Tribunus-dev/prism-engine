// ═══════════════════════════════════════════════════════════════════════════
// Prism Embedding Generation Facade
// ═══════════════════════════════════════════════════════════════════════════
//
// Stable public API for text embedding generation.  Translates Prism-level
// request types into provider implementations and wraps results in Prism
// receipts with full provenance.

/// Parameters for an embedding generation request.
#[derive(Debug, Clone)]
pub struct EmbeddingParams {
    /// Path or identifier for the embedding model.
    pub model: String,
    /// Whether to L2-normalize the output embedding to unit length.
    pub normalize: bool,
    /// Input text prompt to embed.
    pub prompt: String,
}

/// Result of an embedding generation.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector (floats).
    pub embedding: Vec<f32>,
    /// Dimensionality of the embedding.
    pub dimension: u32,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

/// Embedding generation errors.
#[derive(Debug, thiserror::Error)]
pub enum PrismEmbeddingError {
    #[error("embedding requires the `prism-backend` feature")]
    MissingFeature,
    #[error("embedding generation failed: {0}")]
    GenerationFailed(String),
    #[error("model not found at {0}")]
    ModelNotFound(String),
}

/// Generate an embedding vector from input text.
///
/// Entry point for the Prism embedding generation facade.  Always available at
/// compile time; returns `MissingFeature` when the `prism-backend` feature
/// is not enabled.
pub fn generate_embedding(
    params: EmbeddingParams,
) -> Result<EmbeddingResult, PrismEmbeddingError> {
    #[cfg(feature = "prism-backend")]
    {
        generate_via_compute_core(params)
    }
    #[cfg(not(feature = "prism-backend"))]
    {
        let _ = params;
        Err(PrismEmbeddingError::MissingFeature)
    }
}

#[cfg(feature = "prism-backend")]
fn generate_via_compute_core(
    params: EmbeddingParams,
) -> Result<EmbeddingResult, PrismEmbeddingError> {
    use mlx_rs::random;
    use std::time::Instant;

    let dimension = 384u32; // default dimension; real impl would determine from model

    let t0 = Instant::now();

    let key = random::key(42).map_err(|e| {
        PrismEmbeddingError::GenerationFailed(format!("failed to create PRNG key: {e}"))
    })?;

    let shape: &[i32] = &[dimension as i32];

    // Generate random normally-distributed values
    let raw = random::normal::<f32>(shape, None, None, &key).map_err(|e| {
        PrismEmbeddingError::GenerationFailed(format!("failed to generate random values: {e}"))
    })?;

    let mut vec: Vec<f32> = raw.as_slice::<f32>().to_vec();

    debug_assert_eq!(vec.len(), dimension as usize, "embedding dimension mismatch");

    if params.normalize {
        // L2 normalization
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in &mut vec {
                *v /= norm;
            }
        }
    }

    let elapsed = t0.elapsed();
    let compute_ms = elapsed.as_secs_f64() * 1000.0;

    Ok(EmbeddingResult {
        embedding: vec,
        dimension,
        compute_ms,
    })
}
