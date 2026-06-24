//! Tests for the Prism diffusion generation facade.
//!
//! # Hermetic contract test
//!
//! Proves the request/receipt types construct correctly and the facade
//! routes to the correct error variant.  No real model required.
//!
//! # Real provider test (opt-in)
//!
//! Requires a compiled ComputeImage at the path in the
//! `PRISM_DIFFUSION_FIXTURE` env var.  Not run in normal CI.

use prism_engine::diffusion::{
    generate_text, DiffusionGenerationReceipt, DiffusionParams, PrismDiffusionError,
};

// ── Type construction tests (always available) ──────────────────────────

#[test]
fn diffusion_params_defaults() {
    let p = DiffusionParams::default();
    assert_eq!(p.max_tokens, 256);
    assert!(p.steps.is_none());
}

#[test]
fn receipt_fields_round_trip() {
    use std::time::Duration;
    let r = DiffusionGenerationReceipt {
        tokens: vec![1, 2, 3],
        provider: "compute-mlx",
        compute_ms: 42.0,
        actual_device: "apple-gpu".into(),
        duration: Duration::from_millis(42),
        fallback_used: false,
    };
    assert_eq!(r.tokens, vec![1, 2, 3]);
    assert_eq!(r.provider, "compute-mlx");
    assert!(r.compute_ms > 0.0);
    assert!(!r.fallback_used);
}

// ── Facade routing test ────────────────────────────────────────────────

#[test]
fn generate_text_without_feature_returns_error() {
    // When the generation-diffusion feature is off, generate_text
    // returns MissingFeature.  When it's on, it tries to load a real
    // model and fails with ModelNotFound (no fixture in CI).
    let result = generate_text(
        "/nonexistent/model",
        "test prompt",
        DiffusionParams::default(),
    );

    #[cfg(not(feature = "generation-diffusion"))]
    {
        assert!(matches!(result, Err(PrismDiffusionError::MissingFeature)));
    }

    #[cfg(feature = "generation-diffusion")]
    {
        // Without a real model, we expect ModelNotFound, not a panic.
        let err = result.unwrap_err();
        assert!(
            matches!(&err, PrismDiffusionError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
    }
}

// ── Opt-in real provider test ───────────────────────────────────────────

/// Run with:
/// PRISM_DIFFUSION_FIXTURE=/path/to/model cargo test --features generation-diffusion -- --ignored
#[test]
#[ignore]
#[cfg(feature = "generation-diffusion")]
fn real_model_generates_text() {
    let model_path = std::env::var("PRISM_DIFFUSION_FIXTURE")
        .expect("PRISM_DIFFUSION_FIXTURE must be set to a compiled ComputeImage path");

    let params = DiffusionParams {
        max_tokens: 64,
        steps: Some(4),
    };

    let receipt = generate_text(&model_path, "Hello, world!", params)
        .expect("real model generation should succeed");

    assert!(!receipt.tokens.is_empty(), "must produce tokens");
    assert_eq!(receipt.provider, "compute-mlx");
    assert!(receipt.compute_ms > 0.0, "compute time must be measured");
    assert!(!receipt.fallback_used);
}
