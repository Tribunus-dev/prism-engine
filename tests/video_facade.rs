//! Tests for the video generation facade (`prism-engine::video`).
//!
//! These tests exercise the public API surface: type construction,
//! facade routing (error paths when no feature is active), and an
//! opt-in provider integration test guarded by `#[ignore]`.

// ── Type construction ───────────────────────────────────────────────────

#[test]
fn video_params_default_construction() {
    let params = prism_engine::video::VideoParams {
        num_frames: 16,
        fps: 24,
        seed: 42,
    };
    assert_eq!(params.num_frames, 16);
    assert_eq!(params.fps, 24);
    assert_eq!(params.seed, 42);
}

#[test]
fn video_generation_receipt_construction() {
    let receipt = prism_engine::video::VideoGenerationReceipt {
        frames: vec![
            (512, 512, vec![0u8; 512 * 512 * 4]),
            (512, 512, vec![128u8; 512 * 512 * 4]),
        ],
        compute_ms: 123.45,
    };
    assert_eq!(receipt.frames.len(), 2);
    assert!((receipt.compute_ms - 123.45).abs() < f64::EPSILON);
}

#[test]
fn prism_video_error_construction() {
    let missing = prism_engine::video::PrismVideoError::MissingFeature;
    let not_found = prism_engine::video::PrismVideoError::ModelNotFound("/bad/path".into());
    let failed = prism_engine::video::PrismVideoError::GenerationFailed("oops".into());

    assert!(matches!(
        missing,
        prism_engine::video::PrismVideoError::MissingFeature
    ));
    assert!(matches!(
        not_found,
        prism_engine::video::PrismVideoError::ModelNotFound(_)
    ));
    assert!(matches!(
        failed,
        prism_engine::video::PrismVideoError::GenerationFailed(_)
    ));
}

#[test]
fn prism_video_error_display() {
    let missing = prism_engine::video::PrismVideoError::MissingFeature;
    assert!(format!("{}", missing).contains("generation-video"));

    let not_found = prism_engine::video::PrismVideoError::ModelNotFound("/null".into());
    assert!(format!("{}", not_found).contains("/null"));

    let failed = prism_engine::video::PrismVideoError::GenerationFailed("err".into());
    assert!(format!("{}", failed).contains("err"));
}

// ── Facade routing ──────────────────────────────────────────────────────

#[test]
fn generate_video_missing_feature() {
    // When the `generation-video` feature is disabled, the facade should
    // return MissingFeature.
    let params = prism_engine::video::VideoParams {
        num_frames: 8,
        fps: 30,
        seed: 0,
    };
    let result = prism_engine::video::generate_video("/nonexistent", "test prompt", params);
    match result {
        Err(prism_engine::video::PrismVideoError::MissingFeature) => { /* expected */ }
        // If the feature IS enabled, the model won't exist — that's also OK.
        Err(prism_engine::video::PrismVideoError::ModelNotFound(_)) => { /* feature enabled, model missing */
        }
        other => panic!("unexpected result: {other:?}"),
    }
}

// ── Opt-in real provider test ───────────────────────────────────────────

/// Integration test that exercises the full provider path.
///
/// Requires the `generation-video` feature and a real model at the given
/// path. Run with:
///
/// ```text
/// cargo test test_video_provider_integration --features generation-video -- --ignored
/// ```
#[test]
#[ignore]
fn test_video_provider_integration() {
    let model_path =
        std::env::var("VIDEO_MODEL_PATH").unwrap_or_else(|_| "/tmp/video_model".into());
    let params = prism_engine::video::VideoParams {
        num_frames: 4,
        fps: 8,
        seed: 12345,
    };
    let result =
        prism_engine::video::generate_video(&model_path, "a cat walking on a beach", params);
    let receipt = result.expect("video generation should succeed with a valid model");
    assert!(!receipt.frames.is_empty(), "expected at least one frame");
    assert!(receipt.compute_ms >= 0.0, "compute_ms must be non-negative");
    for (i, (w, h, pixels)) in receipt.frames.iter().enumerate() {
        assert_eq!(*w, 512, "frame {i} width mismatch");
        assert_eq!(*h, 512, "frame {i} height mismatch");
        let expected_len = (*w * *h * 4) as usize;
        assert_eq!(pixels.len(), expected_len, "frame {i} pixel buffer size");
    }
}
