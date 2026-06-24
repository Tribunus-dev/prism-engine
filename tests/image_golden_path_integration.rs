//! Integration test: full image-generation golden path through ComputeCoreMlx.
//!
//! This test exercises the real pipeline end to end: fixture resolution,
//! admission, routing, provider execution (ComputeCoreMlx), output
//! validation, and receipt verification.
//!
//! # Prerequisites
//!
//! - `PRISM_IMAGE_FIXTURE_PATH` must point to an installed image-generation
//!   CImage artifact (a compiled flux-klein-mlx bundle).
//!
//! # Execution
//!
//! ```sh
//! PRISM_IMAGE_FIXTURE_PATH=/path/to/cimage \
//!   cargo test -p prism-engine --test image_golden_path_integration \
//!   --features generation-image -- --ignored
//! ```

#![cfg(feature = "generation-image")]

use prism_engine::image::{
    generate_image, DevicePreference, GenerationExecutionPolicy, ImageGenerationRequest,
    ImageOutputFormat, ImageProviderKind, RouteOrigin,
};

#[test]
#[ignore]
fn prism_image_golden_path_executes_compute_mlx() {
    // ── Fixture path ────────────────────────────────────────────────────
    let fixture_path = match std::env::var("PRISM_IMAGE_FIXTURE_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("SKIP: PRISM_IMAGE_FIXTURE_PATH is not set — no CImage artifact available");
            return;
        }
    };

    // ── Build request ──────────────────────────────────────────────────
    let request = ImageGenerationRequest {
        prompt: "A minimal deterministic integration-test image".into(),
        negative_prompt: None,
        width: 256,
        height: 256,
        steps: 2,
        seed: Some(42),
        guidance_scale: Some(1.0),
        output_format: ImageOutputFormat::Rgba8,
        device_preference: DevicePreference::Auto,
        execution_policy: GenerationExecutionPolicy::AllowQualifiedFallback,
    };

    // ── Execute pipeline ───────────────────────────────────────────────
    let result = generate_image(&fixture_path, request);

    let result = match result {
        Ok(r) => r,
        Err(e) => panic!("generate_image failed: {e}"),
    };

    // ── Receipt assertions ─────────────────────────────────────────────
    assert_eq!(
        result.receipt.selected_provider,
        ImageProviderKind::ComputeCoreMlx,
        "expected ComputeCoreMlx provider"
    );
    assert!(
        !result.receipt.fallback_used,
        "fallback should not have been triggered"
    );
    assert!(
        result.receipt.route_origin != RouteOrigin::DryRun,
        "route_origin must not be DryRun"
    );
    assert!(
        result.receipt.provider_latency_ms > 0.0,
        "provider_latency_ms should be positive, got {}",
        result.receipt.provider_latency_ms
    );
    assert!(
        result.receipt.total_latency_ms >= result.receipt.provider_latency_ms,
        "total_latency_ms ({}) must be >= provider_latency_ms ({})",
        result.receipt.total_latency_ms,
        result.receipt.provider_latency_ms
    );

    // ── Output digest assertions ────────────────────────────────────────
    let digest_str = result.receipt.output_digest.to_string();
    assert!(!digest_str.is_empty(), "output_digest must not be empty");

    // ── Image assertions ───────────────────────────────────────────────
    assert_eq!(result.image.width, 256, "image width must be 256");
    assert_eq!(result.image.height, 256, "image height must be 256");
    assert!(
        !result.image.bytes.is_empty(),
        "image bytes must not be empty"
    );
    assert_eq!(
        result.image.bytes.len(),
        256 * 256 * 4,
        "image byte count must be width * height * 4 (RGBA8)"
    );

    // ── Integrity digest assertions ─────────────────────────────────────
    let image_digest = result.image.digest.to_string();
    assert!(!image_digest.is_empty(), "image digest must not be empty");

    // ── Alpha channel validation ────────────────────────────────────────
    // Sample a few pixels spread across the image; all must have non-zero
    // alpha (the provider must produce opaque output for a full run).
    let bytes = &result.image.bytes;
    let stride = 256 * 4;
    let samples = [
        (0usize, 0usize), // top-left
        (127, 127),       // center
        (255, 255),       // bottom-right
        (0, 255),         // bottom-left
        (255, 0),         // top-right
        (64, 128),        // interior
        (192, 64),        // interior
    ];
    for &(x, y) in &samples {
        let idx = y as usize * stride + x as usize * 4 + 3;
        assert!(
            bytes[idx] > 0,
            "alpha channel must be non-zero at pixel ({x}, {y}), got {}",
            bytes[idx]
        );
    }

    // ── Materialization assertion ───────────────────────────────────────
    let mat = &result.receipt.materialization;
    assert!(
        mat.bytes_materialized > 0,
        "materialization must record bytes transferred"
    );
}
