//! Audio ASR pipeline (canonical constitutional surface).
//!
//! This module owns the canonical authority for the audio automatic
//! speech recognition (ASR) front-end: the static configuration
//! describing an audio encoder model, the preprocessing step that
//! converts a raw waveform into a log-mel spectrogram, the audio
//! encoder placeholder that turns a mel spectrogram into per-frame
//! feature embeddings, and the injection step that prepends audio
//! features to a text hidden state.
//!
//! # Backend-not-yet-wired status
//!
//! The audio encoder in `compute-core/src/ecs/audio/encoder.rs` was
//! tightly coupled to the engine's `mlx_rs` tensors, its
//! `LoadedProfiledModel` weight loader, and its
//! `QuantizedLinearBinding` linear path. The constitutional surface
//! in this file ships the type-level authority (config, function
//! signatures, error taxonomy) using backend-agnostic
//! representations (`Vec<f32>`, `u32`, owned `String` errors) so
//! that callers do not gain a second source of truth. The
//! MLX-coupled implementation moves to a backend crate when its
//! dependents migrate; until then the constructors return
//! [`AudioPreprocessError::BackendNotWired`] /
//! [`AudioEncodeError::BackendNotWired`] /
//! [`AudioInjectError::BackendNotWired`] so any caller that
//! accidentally reaches the placeholder at runtime learns the
//! boundary instead of silently producing zero audio.
//!
//! The engine's `compute-core/src/ecs/audio/` directory is the
//! legacy duplicate scheduled for deletion (see
//! `changelogs/2026-07-27-engine-subsystem-deletion-audio.md`).

/// Audio encoder model configuration.
///
/// This struct owns the canonical authority for the static
/// hyperparameters that describe an audio encoder's shape and the
/// preprocessing parameters needed to convert a waveform into a
/// log-mel spectrogram. The fields are pure data; no I/O, no
/// backend handles, no process-local state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioArchitecture {
    /// Encoder hidden dimension.
    pub hidden_size: u32,
    /// Number of attention heads in each encoder layer.
    pub num_attention_heads: u32,
    /// Number of transformer/conformer encoder layers.
    pub num_hidden_layers: u32,
    /// Intermediate (FFN) dimension.
    pub intermediate_size: u32,
    /// Target sample rate in Hz (e.g. 16000).
    pub sample_rate: u32,
    /// Number of mel filterbank bins (e.g. 80).
    pub num_mel_bins: u32,
    /// STFT hop length in samples (e.g. 160 for 10 ms at 16 kHz).
    pub hop_length: u32,
    /// Maximum audio length in seconds (longer inputs are truncated).
    pub max_audio_length_s: u32,
    /// Projection dimension from audio features into the text
    /// model's hidden size.
    pub projection_dim: u32,
}

impl Default for AudioArchitecture {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            intermediate_size: 0,
            sample_rate: 16_000,
            num_mel_bins: 80,
            hop_length: 160,
            max_audio_length_s: 30,
            projection_dim: 0,
        }
    }
}

/// Errors raised by [`preprocess_audio`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AudioPreprocessError {
    /// The constitutional surface is the contract; the
    /// MLX-coupled implementation has not yet been ported to a
    /// backend crate.
    #[error("audio preprocessing backend not yet wired")]
    BackendNotWired,
    /// The audio file does not exist on disk.
    #[error("audio file not found: {0}")]
    FileNotFound(String),
    /// The audio file extension is not supported (only WAV today).
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
    /// The WAV file header or chunks are malformed.
    #[error("invalid WAV: {0}")]
    InvalidWav(String),
    /// An HTTP fetch or local read failed.
    #[error("io error: {0}")]
    Io(String),
}

/// Preprocess a raw audio waveform into a log-mel spectrogram.
///
/// `path_or_url` is either a local file path or an `http(s)://`
/// URL pointing at a WAV audio sample. `config` describes the
/// target sample rate, mel filterbank, hop length, and maximum
/// audio length.
///
/// The returned `Vec<f32>` is a flat row-major
/// `[1, num_mel_bins, num_frames]` log-mel spectrogram — the
/// engine's `mlx_rs::Array` representation collapses to this
/// when the backend port lands.
///
/// Until the backend is wired, this function returns
/// [`AudioPreprocessError::BackendNotWired`].
pub fn preprocess_audio(
    _path_or_url: &str,
    _config: &AudioArchitecture,
) -> Result<Vec<f32>, AudioPreprocessError> {
    Err(AudioPreprocessError::BackendNotWired)
}

/// Errors raised by [`AudioEncoder::load`] and
/// [`AudioEncoder::encode`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AudioEncodeError {
    /// The constitutional surface is the contract; the
    /// MLX-coupled implementation has not yet been ported to a
    /// backend crate.
    #[error("audio encoder backend not yet wired")]
    BackendNotWired,
    /// A weight tensor referenced by the encoder was not found in
    /// the loaded model.
    #[error("audio tensor not found: {0}")]
    TensorNotFound(String),
    /// An underlying linear / matmul / softmax op failed.
    #[error("audio encoder op failed: {0}")]
    OpFailed(String),
}

/// One layer of the audio encoder.
///
/// The engine's `AudioEncoderLayer` carried a per-layer
/// `QuantizedLinearBinding` set and the same `hidden_size` field.
/// The constitutional placeholder keeps the field shape so the
/// engine caller can be migrated mechanically; the binding type
/// collapses to `u32` (the layer's hidden dimension) because the
/// quantized-binding abstraction is a backend detail that lives
/// behind a backend port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioEncoderLayer {
    /// Hidden dimension for this layer.
    pub hidden_size: u32,
}

impl AudioEncoderLayer {
    /// Run one encoder layer forward pass.
    ///
    /// The constitutional surface is a placeholder until the
    /// backend lands. The signature is preserved so the engine
    /// caller's import path can migrate now and the call body
    /// can be re-pointed at the backend implementation later.
    pub fn forward(&self, _x: &[f32]) -> Result<Vec<f32>, AudioEncodeError> {
        Err(AudioEncodeError::BackendNotWired)
    }
}

/// Audio encoder — processes mel spectrograms into feature
/// embeddings.
///
/// The engine's `AudioEncoder` carried input/output projections,
/// a `Vec<AudioEncoderLayer>`, and a `LoadedProfiledModel`-coupled
/// `load` constructor. The constitutional placeholder keeps the
/// configuration field (the only field the engine caller actually
/// reads at the boundary) and a minimal `encode` method so the
/// type surface matches the engine's API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioEncoder {
    /// Static configuration for this encoder.
    pub config: AudioArchitecture,
    /// Encoder transformer/conformer layers.
    pub encoder_layers: Vec<AudioEncoderLayer>,
}

impl AudioEncoder {
    /// Construct an empty audio encoder from configuration.
    ///
    /// The constitutional surface ships this constructor; the
    /// engine's `AudioEncoder::load(&LoadedProfiledModel)` is the
    /// legacy constructor that pulls weights from a profiled
    /// model. The migration target is a backend-port
    /// `AudioEncoder::load(...)` that takes a backend-neutral
    /// weight source — until then this placeholder returns
    /// [`AudioEncodeError::BackendNotWired`].
    pub fn new(config: AudioArchitecture) -> Self {
        Self {
            config,
            encoder_layers: Vec::new(),
        }
    }

    /// Load an audio encoder from a backend-specific model.
    ///
    /// The engine caller passes `&LoadedProfiledModel`; the
    /// constitutional placeholder is backend-agnostic and
    /// returns [`AudioEncodeError::BackendNotWired`]. The type
    /// parameter is `?Sized` so the engine caller's
    /// `AudioEncoder::load(&LoadedProfiledModel)` resolves at
    /// the call site without requiring a trait bound that the
    /// engine's `LoadedProfiledModel` does not yet implement.
    /// The engine caller is updated to the backend-port
    /// signature when the migration of the encoder backend
    /// lands.
    pub fn load<M: ?Sized>(_model: &M) -> Result<Self, AudioEncodeError> {
        Err(AudioEncodeError::BackendNotWired)
    }

    /// Encode a mel spectrogram into per-frame audio features.
    ///
    /// `mel_spec` is a flat row-major
    /// `[1, num_mel_bins, num_frames]` log-mel spectrogram (the
    /// same layout the engine's `preprocess_audio` returns).
    /// The returned `Vec<f32>` is a flat
    /// `[num_frames, projection_dim]` feature tensor.
    pub fn encode(&self, _mel_spec: &[f32]) -> Result<Vec<f32>, AudioEncodeError> {
        Err(AudioEncodeError::BackendNotWired)
    }
}

/// Errors raised by [`inject_audio_features`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AudioInjectError {
    /// The constitutional surface is the contract; the
    /// MLX-coupled implementation has not yet been ported to a
    /// backend crate.
    #[error("audio injection backend not yet wired")]
    BackendNotWired,
    /// The audio feature tensor was not rank 2.
    #[error("audio_features must be rank 2, got shape len {0}")]
    InvalidRank(usize),
    /// An underlying concatenate op failed.
    #[error("concatenate audio features: {0}")]
    ConcatenateFailed(String),
}

/// Inject audio features into a text hidden state by
/// concatenating the audio feature tokens at the start of the
/// sequence.
///
/// `hidden` — flat row-major `[text_tokens, hidden_size]` text
/// hidden state. `audio_features` — flat row-major
/// `[num_frames, projection_dim]` audio encoder output.
///
/// Returns the concatenated hidden state as a flat
/// `[text_tokens + num_frames, hidden_size]` tensor.
///
/// Until the backend is wired, this function returns
/// [`AudioInjectError::BackendNotWired`].
pub fn inject_audio_features(
    _hidden: &[f32],
    _audio_features: &[f32],
) -> Result<Vec<f32>, AudioInjectError> {
    Err(AudioInjectError::BackendNotWired)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_architecture_default_uses_speech_typical_values() {
        // Architectural invariant: the default configuration
        // matches the speech-typical front-end used by
        // Whisper/Conformer encoders (16 kHz, 80 mel bins, 10 ms
        // hop, 30 s max). A caller can override any field
        // before passing the config into the pipeline.
        let cfg = AudioArchitecture::default();
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.num_mel_bins, 80);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.max_audio_length_s, 30);
    }

    #[test]
    fn preprocess_audio_returns_backend_not_wired() {
        // Architectural invariant: until the audio preprocessing
        // backend is ported to the constitutional surface, the
        // function refuses to produce a silent spectrogram. A
        // caller that reaches this branch learns the boundary
        // instead of getting a meaningless result.
        let cfg = AudioArchitecture::default();
        let result = preprocess_audio("/nonexistent.wav", &cfg);
        assert_eq!(result, Err(AudioPreprocessError::BackendNotWired));
    }

    #[test]
    fn audio_encoder_new_preserves_config_and_starts_empty() {
        // Architectural invariant: the constitutional
        // constructor preserves the configuration and starts
        // with no encoder layers. The engine caller (which
        // constructs encoders from a loaded model) is updated
        // to the backend-port `load` once the backend lands.
        let cfg = AudioArchitecture {
            hidden_size: 512,
            num_attention_heads: 8,
            num_hidden_layers: 6,
            intermediate_size: 2048,
            sample_rate: 16_000,
            num_mel_bins: 80,
            hop_length: 160,
            max_audio_length_s: 30,
            projection_dim: 1024,
        };
        let enc = AudioEncoder::new(cfg.clone());
        assert_eq!(enc.config, cfg);
        assert!(enc.encoder_layers.is_empty());
    }

    #[test]
    fn audio_encoder_load_returns_backend_not_wired() {
        // Architectural invariant: the backend-port `load` is
        // not yet wired. Callers learn the boundary instead of
        // getting a half-constructed encoder. The test passes
        // any sized reference (`?Sized` bound) so the engine
        // caller's `&LoadedProfiledModel` shape resolves at the
        // call site without a trait implementation.
        let cfg = AudioArchitecture::default();
        let result = AudioEncoder::load(&cfg);
        assert_eq!(result, Err(AudioEncodeError::BackendNotWired));
    }

    #[test]
    fn audio_encoder_encode_returns_backend_not_wired() {
        // Architectural invariant: the encode step refuses to
        // run until the backend lands, even when the encoder
        // has layers.
        let cfg = AudioArchitecture::default();
        let enc = AudioEncoder::new(cfg);
        let result = enc.encode(&[0.0; 80 * 100]);
        assert_eq!(result, Err(AudioEncodeError::BackendNotWired));
    }

    #[test]
    fn audio_encoder_layer_forward_returns_backend_not_wired() {
        // Architectural invariant: the per-layer forward pass
        // also refuses to run until the backend lands.
        let layer = AudioEncoderLayer { hidden_size: 64 };
        let result = layer.forward(&[0.0; 64]);
        assert_eq!(result, Err(AudioEncodeError::BackendNotWired));
    }

    #[test]
    fn inject_audio_features_returns_backend_not_wired() {
        // Architectural invariant: the injection step refuses
        // to concatenate until the backend lands.
        let result = inject_audio_features(&[0.0; 64], &[0.0; 64]);
        assert_eq!(result, Err(AudioInjectError::BackendNotWired));
    }
}
