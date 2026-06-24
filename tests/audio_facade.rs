//! Tests for the Prism audio generation facade.
//!
//! # Hermetic contract test
//!
//! Proves the request/receipt types construct correctly and the facade
//! routes to the correct error variant.  No real model required.
//!
//! # Real provider test (opt-in)
//!
//! Requires a compiled TTS model at the path in the
//! `PRISM_AUDIO_FIXTURE` env var.  Not run in normal CI.

use prism_engine::audio::{generate_speech, AudioGenerationReceipt, AudioParams, PrismAudioError};

// ── Type construction tests (always available) ──────────────────────────

#[test]
fn audio_params_defaults() {
    let p = AudioParams::default();
    assert!(p.voice.is_none());
}

#[test]
fn receipt_fields_round_trip() {
    let r = AudioGenerationReceipt {
        sample_rate: 24000,
        pcm_samples: vec![0.5f32; 48000],
        compute_ms: 123.4,
        output_digest: "def456".into(),
    };
    assert_eq!(r.sample_rate, 24000);
    assert_eq!(r.pcm_samples.len(), 48000);
    assert!((r.compute_ms - 123.4).abs() < 1e-9);
    assert_eq!(r.output_digest, "def456");
}

// ── Facade routing test ────────────────────────────────────────────────

#[test]
fn generate_speech_without_feature_returns_error() {
    // When the generation-audio feature is off, generate_speech
    // returns MissingFeature.  When it's on, it tries to load a real
    // model and fails with ModelNotFound (no fixture in CI).
    let result = generate_speech("/nonexistent/model", "hello world", AudioParams::default());

    #[cfg(not(feature = "generation-audio"))]
    {
        assert!(matches!(result, Err(PrismAudioError::MissingFeature)));
    }

    #[cfg(feature = "generation-audio")]
    {
        // Without a real model, we expect ModelNotFound, not a panic.
        let err = result.unwrap_err();
        assert!(
            matches!(&err, PrismAudioError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
    }
}

// ── Opt-in real provider test ───────────────────────────────────────────

/// Run with: PRISM_AUDIO_FIXTURE=/path/to/tts-model cargo test --features generation-audio -- --ignored
#[test]
#[ignore]
#[cfg(feature = "generation-audio")]
fn real_model_generates_speech() {
    let model_path = std::env::var("PRISM_AUDIO_FIXTURE")
        .expect("PRISM_AUDIO_FIXTURE must be set to a compiled TTS model path");

    let params = AudioParams {
        voice: Some("default".into()),
    };

    let receipt = generate_speech(&model_path, "test synthesis", params)
        .expect("real model generation should succeed");

    assert!(receipt.sample_rate > 0);
    assert!(!receipt.pcm_samples.is_empty());
    assert!(receipt.compute_ms >= 0.0);
    assert!(!receipt.output_digest.is_empty());
}
