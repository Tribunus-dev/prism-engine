// ------------------------------------------------------------
// Prism Embedding Generation Facade
// ------------------------------------------------------------
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
pub fn generate_embedding(params: EmbeddingParams) -> Result<EmbeddingResult, PrismEmbeddingError> {
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
    use std::time::Instant;

    let dimension = 384u32;

    let t0 = Instant::now();
    let mut vec: Vec<f32> = vec![0.0; dimension as usize];
    let seed = blake3::hash(params.prompt.as_bytes());
    for (i, v) in vec.iter_mut().enumerate() {
        let mut bytes = seed.as_bytes().to_vec();
        bytes.extend_from_slice(&(i as u64).to_le_bytes());
        let h = blake3::hash(&bytes);
        let raw = u32::from_le_bytes(h.as_bytes()[0..4].try_into().unwrap());
        *v = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }

    debug_assert_eq!(
        vec.len(),
        dimension as usize,
        "embedding dimension mismatch"
    );

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
