//! ComputeImage generation adapter types.
//!
//! These types form the adapter boundary between the Prism facade (product
//! types, lifecycle, receipts) and the Compute image provider
//! (`ComputeCoreMlxImageProvider`). They live on the Compute side of the
//! boundary and are consumed by both the Prism integration and the provider
//! implementation.

use std::fmt;

// ── Artifact handle ──────────────────────────────────────────────────────

/// A handle referencing a ComputeImage generation artifact (model) on disk.
#[derive(Debug, Clone)]
pub struct ComputeImageArtifactHandle {
    /// Filesystem path to the model artifact.
    pub path: String,
    /// SHA-256 digest of the artifact contents.
    pub digest: [u8; 32],
    /// Optional path to a sidecar fixture manifest for qualification.
    pub manifest_path: Option<String>,
}

// ── Output format ────────────────────────────────────────────────────────

/// Pixel encoding for generated image output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeImageOutputFormat {
    /// Uncompressed 8-bit RGBA pixel data (4 bytes per pixel).
    Rgba8,
    /// PNG-compressed byte stream.
    Png,
}

// ── Generation request ───────────────────────────────────────────────────

/// A request to generate an image from a text prompt.
#[derive(Debug, Clone)]
pub struct ComputeImageGenerationRequest {
    /// The positive text prompt describing the desired image.
    pub prompt: String,
    /// An optional negative prompt describing what to avoid.
    pub negative_prompt: Option<String>,
    /// Desired output width in pixels.
    pub width: u32,
    /// Desired output height in pixels.
    pub height: u32,
    /// Number of diffusion / solver steps to run.
    pub steps: u32,
    /// Optional random seed for reproducible generation.
    pub seed: Option<u64>,
    /// Optional classifier-free guidance scale (CFG).
    pub guidance_scale: Option<f32>,
    /// Desired output pixel format.
    pub output_format: ComputeImageOutputFormat,
}

// ── Generation result ────────────────────────────────────────────────────

/// The result of a single image generation.
#[derive(Debug, Clone)]
pub struct ComputeImageGenerationResult {
    /// Flat RGBA8888 pixel buffer (width × height × 4 bytes).
    pub rgba_bytes: Vec<u8>,
    /// Width of the generated image in pixels.
    pub width: u32,
    /// Height of the generated image in pixels.
    pub height: u32,
    /// Measured provider latency in milliseconds.
    pub provider_latency_ms: f64,
    /// Version string of the provider that produced this result.
    pub provider_version: String,
    /// Actual number of diffusion / solver steps completed.
    pub steps_completed: u32,
}

// ── Generation error ─────────────────────────────────────────────────────

/// Errors that can occur during ComputeImage generation.
#[derive(Debug, Clone)]
pub enum ComputeImageGenerationError {
    /// The requested model was not found at the expected location.
    ModelNotFound(String),
    /// The generation pipeline failed during execution.
    GenerationFailed(String),
    /// The generation request contains unsupported parameters.
    UnsupportedRequest(String),
    /// The generated output failed validation.
    InvalidOutput(String),
}

impl fmt::Display for ComputeImageGenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(detail) => {
                write!(f, "ComputeImage model not found: {detail}")
            }
            Self::GenerationFailed(detail) => {
                write!(f, "ComputeImage generation failed: {detail}")
            }
            Self::UnsupportedRequest(detail) => {
                write!(f, "ComputeImage unsupported request: {detail}")
            }
            Self::InvalidOutput(detail) => {
                write!(f, "ComputeImage invalid output: {detail}")
            }
        }
    }
}

impl std::error::Error for ComputeImageGenerationError {}

// ── Generation adapter trait ─────────────────────────────────────────────

/// Trait for executing image generation against a ComputeImage model
/// artifact.
///
/// Implementations wrap a concrete runtime (MLX, Core ML, etc.) and provide
/// the standard load-and-execute lifecycle.
pub trait ComputeImageGenerationAdapter: Send + Sync {
    /// Execute generation for the given artifact and request.
    ///
    /// # Errors
    ///
    /// Returns [`ComputeImageGenerationError`] if the artifact cannot be
    /// loaded, the request is invalid, or generation fails.
    fn execute(
        &self,
        artifact: &ComputeImageArtifactHandle,
        request: &ComputeImageGenerationRequest,
    ) -> Result<ComputeImageGenerationResult, ComputeImageGenerationError>;
}

// ── Fixture types ────────────────────────────────────────────────────────

/// Describes the kind of model fixture for image generation testing.
#[derive(Debug, Clone)]
pub enum ImageFixtureKind {
    /// A reduced-size test model used for quick iteration.
    ReducedTestModel,
    /// Production Flux model (Black Forest Labs).
    ProductionFlux,
    /// Production Stable Diffusion 3 model.
    ProductionSd3,
    /// A custom model identified by name.
    Custom(String),
}

/// Policy that governs how generated fixture output is validated.
#[derive(Debug, Clone)]
pub enum ImageFixtureOutputPolicy {
    /// Validate only structural properties (dimensions, format).
    StructuralOnly,
    /// Exact-match assertion against an allowlist of acceptable digests.
    DigestAllowlist(Vec<String>),
    /// Perceptual-similarity check against a reference digest.
    PerceptualThreshold {
        /// SHA-256 hex digest of the reference image.
        reference_digest: String,
        /// Maximum perceptual distance from the reference.
        max_distance: f32,
    },
}

/// Describes a known-good image generation fixture for qualification.
#[derive(Debug, Clone)]
pub struct ImageFixtureManifest {
    /// Unique fixture identifier (e.g. UUID or descriptive slug).
    pub fixture_id: String,
    /// The kind of model this fixture represents.
    pub fixture_kind: ImageFixtureKind,
    /// SHA-256 hex digest of the compiled ComputeImage artifact.
    pub cimage_digest: String,
    /// Name of the model family (e.g. `"flux.1-dev"`, `"sd3.5-medium"`).
    pub model_family: String,
    /// Supported generation width in pixels.
    pub supported_width: u32,
    /// Supported generation height in pixels.
    pub supported_height: u32,
    /// Supported range of (min, max) diffusion steps.
    pub supported_steps: (u32, u32),
    /// Required feature profile string (e.g. `"generation-image"`).
    pub required_feature_profile: String,
    /// Provider name (e.g. `"mlx"`, `"coreml"`).
    pub provider: String,
    /// Policy describing how to validate generated output.
    pub expected_output_policy: ImageFixtureOutputPolicy,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ComputeImageGenerationError::ModelNotFound("test.mlx".into());
        assert_eq!(err.to_string(), "ComputeImage model not found: test.mlx");

        let err = ComputeImageGenerationError::GenerationFailed("OOM".into());
        assert_eq!(err.to_string(), "ComputeImage generation failed: OOM");

        let err = ComputeImageGenerationError::UnsupportedRequest("bad format".into());
        assert_eq!(
            err.to_string(),
            "ComputeImage unsupported request: bad format"
        );

        let err = ComputeImageGenerationError::InvalidOutput("bad pixels".into());
        assert_eq!(err.to_string(), "ComputeImage invalid output: bad pixels");
    }

    #[test]
    fn test_generation_request_defaults() {
        let req = ComputeImageGenerationRequest {
            prompt: "a cat".into(),
            negative_prompt: None,
            width: 512,
            height: 512,
            steps: 28,
            seed: None,
            guidance_scale: None,
            output_format: ComputeImageOutputFormat::Rgba8,
        };
        assert_eq!(req.prompt, "a cat");
        assert!(req.negative_prompt.is_none());
        assert_eq!(req.output_format, ComputeImageOutputFormat::Rgba8);
    }

    #[test]
    fn test_output_format_equality() {
        assert_eq!(
            ComputeImageOutputFormat::Rgba8,
            ComputeImageOutputFormat::Rgba8
        );
        assert_ne!(
            ComputeImageOutputFormat::Rgba8,
            ComputeImageOutputFormat::Png
        );
    }

    #[test]
    fn test_fixture_kind_variants() {
        let variants = [
            ImageFixtureKind::ReducedTestModel,
            ImageFixtureKind::ProductionFlux,
            ImageFixtureKind::ProductionSd3,
            ImageFixtureKind::Custom("my-model".into()),
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn test_image_fixture_manifest() {
        let manifest = ImageFixtureManifest {
            fixture_id: "flux-dev-512".into(),
            fixture_kind: ImageFixtureKind::ProductionFlux,
            cimage_digest: "abcdef0123456789".into(),
            model_family: "flux.1-dev".into(),
            supported_width: 512,
            supported_height: 512,
            supported_steps: (4, 50),
            required_feature_profile: "generation-image".into(),
            provider: "mlx".into(),
            expected_output_policy: ImageFixtureOutputPolicy::StructuralOnly,
        };
        assert_eq!(manifest.fixture_id, "flux-dev-512");
        assert_eq!(manifest.supported_steps, (4, 50));
    }

    #[test]
    fn test_output_policy_digest_allowlist() {
        let policy =
            ImageFixtureOutputPolicy::DigestAllowlist(vec!["digest1".into(), "digest2".into()]);
        match policy {
            ImageFixtureOutputPolicy::DigestAllowlist(digests) => {
                assert_eq!(digests.len(), 2);
            }
            _ => panic!("expected DigestAllowlist"),
        }
    }

    #[test]
    fn test_output_policy_perceptual_threshold() {
        let policy = ImageFixtureOutputPolicy::PerceptualThreshold {
            reference_digest: "ref123".into(),
            max_distance: 0.05,
        };
        match policy {
            ImageFixtureOutputPolicy::PerceptualThreshold {
                reference_digest,
                max_distance,
            } => {
                assert_eq!(reference_digest, "ref123");
                assert!((max_distance - 0.05).abs() < f32::EPSILON);
            }
            _ => panic!("expected PerceptualThreshold"),
        }
    }

    #[test]
    fn test_artifact_handle() {
        let handle = ComputeImageArtifactHandle {
            path: "/models/flux.mlx".into(),
            digest: [0u8; 32],
            manifest_path: Some("/models/flux.manifest.json".into()),
        };
        assert_eq!(handle.path, "/models/flux.mlx");
        assert_eq!(handle.digest, [0u8; 32]);
        assert!(handle.manifest_path.is_some());
    }

    #[test]
    fn test_generation_result() {
        let result = ComputeImageGenerationResult {
            rgba_bytes: vec![255u8; 512 * 512 * 4],
            width: 512,
            height: 512,
            provider_latency_ms: 1234.5,
            provider_version: "1.0.0".into(),
            steps_completed: 28,
        };
        assert_eq!(result.width, 512);
        assert_eq!(result.height, 512);
        assert_eq!(result.rgba_bytes.len(), 512 * 512 * 4);
        assert!((result.provider_latency_ms - 1234.5).abs() < f64::EPSILON);
    }
}
