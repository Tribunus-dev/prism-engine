pub mod codec;
pub mod conv1d;
pub mod rvq;
pub mod streaming;
pub mod temporal_attention;

#[derive(Debug, Clone)]
pub enum AudioOp {
    // 1D convolution (temporal)
    Conv1d {
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    },

    // Temporal attention (causal or bidirectional along time axis)
    TemporalAttention {
        dim: usize,
        heads: usize,
        causal: bool,
    },

    // Cross-attention to conditioning (text or audio prompt)
    CrossAttention {
        dim: usize,
        heads: usize,
        kv_dim: usize,
    },

    // EnCodec-style residual vector quantizer
    RvqQuantize {
        codebook_size: usize,
        n_q: usize,
        dim: usize,
    },

    // Audio decoder (EnCodec decoder or vocoder)
    EnCodecDecode {
        frame_rate: usize,
        hop_length: usize,
    },

    // Spectrogram / MDCT processing
    MDctTransform {
        window_size: usize,
        hop_length: usize,
    },
    InverseMDct {
        window_size: usize,
        hop_length: usize,
    },

    // LM head for autoregressive audio models
    LmHead {
        vocab_size: usize,
        dim: usize,
    },
}

pub struct Resampler {
    // Placeholder for resampler state
}

pub struct AudioStreamState {
    /// Overlap buffer for strided conv/attention windows
    pub ring_buffer: Vec<f32>,
    /// Current write position in the ring buffer
    pub write_pos: usize,
    /// Accumulated samples generated so far
    pub generated_samples: u64,
    /// Sample rate conversion state (if needed)
    pub resampler: Option<Resampler>,
}

impl AudioStreamState {
    /// Process one chunk of audio and return the next chunk
    pub fn process_chunk(&mut self, _conditioning: &[f32], chunk_size: usize) -> Vec<f32> {
        // 1. Copy new input into ring buffer
        // 2. Apply 1D conv (with overlap handling)
        // 3. Apply temporal attention (causal masking)
        // 4. Apply EnCodec/Vocoder decode
        // 5. Return chunk_size samples of output
        vec![0.0; chunk_size]
    }
}

pub struct AudioGenPipeline {
    // Pipeline components
}

// ═══════════════════════════════════════════════════════════════════════════
// Prism Audio Generation Facade
// ═══════════════════════════════════════════════════════════════════════════
//
// Stable public API for text-to-speech generation.  Translates Prism-level
// request types into provider implementations and wraps results in Prism
// receipts with full provenance.

/// Parameters for a text-to-speech generation request.
#[derive(Debug, Clone)]
pub struct AudioParams {
    pub voice: Option<String>,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self { voice: None }
    }
}

/// Full provenance receipt for a generation.
#[derive(Debug, Clone)]
pub struct AudioGenerationReceipt {
    pub sample_rate: u32,
    pub pcm_samples: Vec<f32>,
    pub compute_ms: f64,
    pub output_digest: String,
}

/// Audio generation errors.
#[derive(Debug, thiserror::Error)]
pub enum PrismAudioError {
    #[error("text-to-speech generation requires the `generation-audio` feature")]
    MissingFeature,
    #[error("generation failed: {0}")]
    GenerationFailed(String),
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("unsupported model type for text-to-speech: {0}")]
    UnsupportedModelType(String),
}

/// Generate speech from text.
///
/// Entry point for the Prism audio generation facade.  Always available at
/// compile time; returns `MissingFeature` when the `generation-audio` feature
/// is not enabled.
pub fn generate_speech(
    model_path: &str,
    text: &str,
    params: AudioParams,
) -> Result<AudioGenerationReceipt, PrismAudioError> {
    #[cfg(feature = "generation-audio")]
    {
        generate_via_compute_core(model_path, text, params)
    }
    #[cfg(not(feature = "generation-audio"))]
    {
        let _ = (model_path, text, params);
        Err(PrismAudioError::MissingFeature)
    }
}

#[cfg(feature = "generation-audio")]
fn generate_via_compute_core(
    model_path: &str,
    text: &str,
    params: AudioParams,
) -> Result<AudioGenerationReceipt, PrismAudioError> {
    use tribunus_compute_core::audio_provider::{
        AudioGenerationError, AudioGenerationProvider, AudioGenerationRequest,
        TextToSpeechProvider,
    };

    let provider = TextToSpeechProvider::new(model_path)
        .map_err(|e| match e {
            AudioGenerationError::ModelNotFound(p) => PrismAudioError::ModelNotFound(p),
            AudioGenerationError::GenerationFailed(m) => PrismAudioError::GenerationFailed(m),
            AudioGenerationError::UnsupportedModelType(m) => {
                PrismAudioError::UnsupportedModelType(m)
            }
        })?;

    let request = AudioGenerationRequest {
        model_path: model_path.to_string(),
        text: text.to_string(),
        voice: params.voice.clone(),
    };

    let result = provider
        .generate_speech(request)
        .map_err(|e| PrismAudioError::GenerationFailed(e.to_string()))?;

    let output_digest = blake3_hash(bytemuck::cast_slice(&result.pcm_samples));

    Ok(AudioGenerationReceipt {
        sample_rate: result.sample_rate,
        pcm_samples: result.pcm_samples,
        compute_ms: result.compute_ms,
        output_digest,
    })
}

/// BLAKE3 hash of the PCM sample bytes for output integrity verification.
#[cfg(feature = "generation-audio")]
fn blake3_hash(data: &[u8]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(data);
    h.finalize().to_hex().to_string()
}
