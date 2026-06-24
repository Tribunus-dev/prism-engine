// ── Prism Image Generation — Provider Trait & Implementations ────────
//
// The ImageGenerationProvider trait and concrete implementations for
// text-to-image generation.  This module owns the Prism-level provider
// contract — it is NOT a re-export of compute-core types.

use super::manifest::*;
use super::reliability::ImageGenerationCancellationToken;
use super::types::*;

// ── Provider request / result ───────────────────────────────────────────

/// Request passed to [`ImageGenerationProvider::generate`].
///
/// Carries references to the installed CImage, the user's generation
/// request, the machine profile, and a unique execution id so the
/// provider never needs to reach outside its own method boundary.
#[derive(Debug, Clone)]
pub struct ImageGenerationProviderRequest<'a> {
    /// Installed CImage artifact metadata and manifest.
    pub installed_image: &'a InstalledCImage,
    /// User-facing generation parameters.
    pub request: &'a ImageGenerationRequest,
    /// Hardware description of the execution machine.
    pub machine: &'a MachineProfile,
    /// Unique identifier for this execution.
    pub execution_id: ExecutionId,
}

/// Result returned by [`ImageGenerationProvider::generate`].
#[derive(Debug, Clone)]
pub struct ImageGenerationProviderResult {
    /// Flat RGBA8888 pixel buffer (width × height × 4 bytes).
    pub rgba_bytes: Vec<u8>,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Provider-internal wall-clock compute time in milliseconds.
    pub provider_latency_ms: f64,
    /// Provider-specific execution metadata.
    pub provider_metadata: ProviderExecutionMetadata,
    /// Materialization provenance for the output.
    pub materialization: MaterializationReceipt,
}

// ── Capability report ───────────────────────────────────────────────────

/// Whether a provider is prepared to serve a request for a specific
/// CImage + machine combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageProviderCapability {
    /// The provider is qualified and ready.
    ComputeCoreMlxQualified,
    /// MLX compute path exists but is not qualified for this combination.
    ComputeCoreMlxAvailableButUnqualified,
    /// Core ML ANE route is available and qualified.
    CoreMlAneQualified,
    /// Core ML ANE route exists but artifact+model combo not qualified.
    CoreMlAneAvailableButUnqualified,
    /// Core ML ANE route not available on this machine.
    CoreMlAneUnavailable,
    /// Core ML ANE exists but artifact policy explicitly refuses it.
    CoreMlAneRefusedByArtifactPolicy,
    /// Prism LUT provider is qualified and ready.
    PrismLutQualified,
    /// Prism LUT provider exists but is not qualified.
    PrismLutAvailableButUnqualified,
    /// The provider is not available at all on this machine.
    ProviderUnavailable,
}

impl ImageProviderCapability {
    /// Returns `true` when this capability represents a route that can serve
    /// the request.
    ///
    /// Qualified providers include general-purpose routes (e.g. Fake, LUT) and
    /// Core ML ANE-aware routes that are either qualified or available-but-
    /// waiting-on-qualification.  The simple availability of an ANE without a
    /// Core ML artifact does not count.
    pub fn is_qualified(&self) -> bool {
        matches!(
            self,
            Self::ComputeCoreMlxQualified
                | Self::CoreMlAneQualified
                | Self::CoreMlAneAvailableButUnqualified
        )
    }
}

// ── Provider metadata ──────────────────────────────────────────────────

/// Execution-time metadata emitted by a provider after generation.
#[derive(Debug, Clone)]
pub struct ProviderExecutionMetadata {
    /// Human-readable provider version string.
    pub provider_version: String,
    /// Number of denoising steps that were actually completed.
    pub steps_completed: u32,
}

// ── Machine profile ────────────────────────────────────────────────────

/// Snapshot of the execution machine's relevant hardware properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineProfile {
    /// Operating system version string (e.g. "macOS 15.2").
    pub os_version: String,
    /// Whether the Apple Neural Engine is available.
    pub has_ane: bool,
    /// Total unified memory in gigabytes.
    pub unified_memory_gb: u64,
}

// ── Execution identifier ───────────────────────────────────────────────

/// Opaque execution identifier (typically a UUID v4 string).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutionId(pub String);

impl ExecutionId {
    /// Create a new execution id from a UUID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for ExecutionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// ── Provider error ──────────────────────────────────────────────────────

/// Errors that can occur during provider selection or execution.
#[derive(Debug, thiserror::Error)]
pub enum ImageProviderError {
    /// The model artifact could not be found or loaded.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// The provider failed during generation.
    #[error("generation failed: {0}")]
    GenerationFailed(String),

    /// The request is not supported by this provider.
    #[error("unsupported request: {0}")]
    UnsupportedRequest(String),

    /// The provider is unavailable and cannot serve any request.
    #[error("provider unavailable")]
    ProviderUnavailable,
}

// ── Provider trait ──────────────────────────────────────────────────────

/// A provider that can generate images from text prompts.
///
/// Every provider reports its identity, qualification state for a
/// given CImage + machine pair, and can execute generation.
///
/// # Contract
///
/// * `kind()` — always returns the same discriminant for a given impl.
/// * `capability_report()` — must be cheap (no model loading); returns
///   the provider's best estimate of whether it could serve this
///   request given the current hardware and artifact.
/// * `generate()` — must produce valid RGBA8888 pixel data or return
///   an `ImageProviderError`.  The provider is responsible for setting
///   `provider_latency_ms` and `provider_metadata`.
pub trait ImageGenerationProvider: Send + Sync {
    /// Which provider variant this is.
    fn kind(&self) -> ImageProviderKind;

    /// Report capability for a specific CImage + machine combination.
    fn capability_report(
        &self,
        model: &InstalledCImage,
        machine: &MachineProfile,
    ) -> ImageProviderCapability;

    /// Execute a generation.
    fn generate(
        &self,
        request: &ImageGenerationProviderRequest,
        cancellation: &ImageGenerationCancellationToken,
    ) -> Result<ImageGenerationProviderResult, ImageProviderError>;
}

// ── Compute-core MLX provider ───────────────────────────────────────────

#[cfg(feature = "generation-image")]
pub(crate) mod compute_provider_adapter {
    use tribunus_compute_core::compute_image::adapter::{
        ComputeImageArtifactHandle, ComputeImageGenerationAdapter, ComputeImageGenerationError,
        ComputeImageGenerationRequest, ComputeImageGenerationResult,
    };
    use tribunus_compute_core::image_provider::{ImageGenerationProvider, TextToImageProvider};

    /// Adapter that wraps [`TextToImageProvider`] behind the
    /// [`ComputeImageGenerationAdapter`] trait.
    pub(crate) struct ComputeProviderAdapter {
        provider: TextToImageProvider,
    }

    impl ComputeProviderAdapter {
        /// Create a new adapter, loading the model at `model_path`.
        pub(crate) fn new(model_path: &str) -> Result<Self, ComputeImageGenerationError> {
            TextToImageProvider::new(model_path)
                .map(|provider| Self { provider })
                .map_err(|e| ComputeImageGenerationError::ModelNotFound(e.to_string()))
        }
    }

    impl ComputeImageGenerationAdapter for ComputeProviderAdapter {
        fn execute(
            &self,
            artifact: &ComputeImageArtifactHandle,
            request: &ComputeImageGenerationRequest,
        ) -> Result<ComputeImageGenerationResult, ComputeImageGenerationError> {
            let inner_request = tribunus_compute_core::image_provider::ImageGenerationRequest {
                model_path: artifact.path.clone(),
                prompt: request.prompt.clone(),
                steps: Some(request.steps),
                size: Some((request.width, request.height)),
            };

            let result = self
                .provider
                .generate(inner_request)
                .map_err(|e| ComputeImageGenerationError::GenerationFailed(e.to_string()))?;

            Ok(ComputeImageGenerationResult {
                rgba_bytes: result.rgba,
                width: result.width,
                height: result.height,
                provider_latency_ms: result.compute_ms,
                provider_version: env!("CARGO_PKG_VERSION").to_string(),
                steps_completed: request.steps,
            })
        }
    }
}

#[cfg(feature = "generation-image")]
mod compute_core_provider {
    use super::super::reliability::ImageGenerationCancellationToken;
    use super::*;
    use std::sync::Arc;
    use tribunus_compute_core::compute_image::adapter::{
        ComputeImageArtifactHandle, ComputeImageGenerationAdapter, ComputeImageGenerationRequest,
        ComputeImageOutputFormat,
    };

    /// Prism wrapper around [`ComputeImageGenerationAdapter`] from compute-core.
    ///
    /// Loads the model once at construction and delegates generation to
    /// the adapter trait, which wraps the MLX-backed compute-core engine.
    pub struct ComputeCoreMlxImageProvider {
        adapter: Arc<dyn ComputeImageGenerationAdapter>,
        model_path: String,
    }

    impl ComputeCoreMlxImageProvider {
        /// Load a model and wrap it.
        ///
        /// Fails with `ImageProviderError::ModelNotFound` if `model_path`
        /// does not contain a valid compiled ComputeImage.
        pub fn new(model_path: &str) -> Result<Self, ImageProviderError> {
            let adapter = super::compute_provider_adapter::ComputeProviderAdapter::new(model_path)
                .map_err(|e| ImageProviderError::ModelNotFound(e.to_string()))
                .map(Arc::new)?;
            Ok(Self {
                adapter,
                model_path: model_path.to_string(),
            })
        }
    }

    impl ImageGenerationProvider for ComputeCoreMlxImageProvider {
        fn kind(&self) -> ImageProviderKind {
            ImageProviderKind::ComputeCoreMlx
        }

        fn capability_report(
            &self,
            model: &InstalledCImage,
            machine: &MachineProfile,
        ) -> ImageProviderCapability {
            if !machine.has_ane {
                return ImageProviderCapability::ComputeCoreMlxQualified;
            }

            // Check whether the installed CImage manifest declares a Core ML
            // provider artifact (one that requires the ANE).
            let has_coreml_artifact = model
                .manifest
                .provider_artifacts
                .iter()
                .any(|a| a.required_hardware.iter().any(|hw| hw == "ane"));

            if has_coreml_artifact {
                ImageProviderCapability::CoreMlAneQualified
            } else {
                ImageProviderCapability::ComputeCoreMlxAvailableButUnqualified
            }
        }

        fn generate(
            &self,
            request: &ImageGenerationProviderRequest,
            cancellation: &ImageGenerationCancellationToken,
        ) -> Result<ImageGenerationProviderResult, ImageProviderError> {
            let t0 = std::time::Instant::now();

            if cancellation.is_cancelled() {
                return Err(ImageProviderError::GenerationFailed(
                    "cancelled before execution".into(),
                ));
            }

            let compute_request = ComputeImageGenerationRequest {
                prompt: request.request.prompt.clone(),
                negative_prompt: None,
                width: request.request.width,
                height: request.request.height,
                steps: request.request.steps,
                seed: None,
                guidance_scale: None,
                output_format: ComputeImageOutputFormat::Rgba8,
            };

            let artifact = ComputeImageArtifactHandle {
                path: self.model_path.clone(),
                digest: [0u8; 32],
                manifest_path: None,
            };

            let compute_result = self
                .adapter
                .execute(&artifact, &compute_request)
                .map_err(|e| ImageProviderError::GenerationFailed(e.to_string()))?;

            let provider_latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let bytes = compute_result.rgba_bytes.len() as u64;

            Ok(ImageGenerationProviderResult {
                rgba_bytes: compute_result.rgba_bytes,
                width: compute_result.width,
                height: compute_result.height,
                provider_latency_ms,
                provider_metadata: ProviderExecutionMetadata {
                    provider_version: compute_result.provider_version,
                    steps_completed: compute_result.steps_completed,
                },
                materialization: MaterializationReceipt::new_copied(bytes),
            })
        }
    }
}

#[cfg(feature = "generation-image")]
pub use compute_core_provider::ComputeCoreMlxImageProvider;

// ── DiffusionGemma provider (prism-backend only) ──────────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-diffusion"))]
mod diffusion_gemma_provider {
    use super::*;
    use std::sync::Arc;
    use tribunus_compute_core::diffusion_provider::{
        DiffusionGenerationProvider, DiffusionGenerationRequest, DiffusionProvider,
    };

    /// Prism wrapper around [`DiffusionProvider`] from compute-core.
    ///
    /// Loads a DiffusionGemma model once at construction and delegates
    /// generation to the compute-core diffusion provider.
    pub struct DiffusionGemmaImageProvider {
        inner: Arc<DiffusionProvider>,
        model_path: String,
    }

    impl DiffusionGemmaImageProvider {
        /// Load a DiffusionGemma model and wrap it.
        ///
        /// Fails with `ImageProviderError::ModelNotFound` if `model_path`
        /// does not contain a valid compiled DiffusionGemma ComputeImage.
        pub fn new(model_path: &str) -> Result<Self, ImageProviderError> {
            let provider = DiffusionProvider::new(model_path)
                .map_err(|e| ImageProviderError::ModelNotFound(e.to_string()))?;
            Ok(Self {
                inner: Arc::new(provider),
                model_path: model_path.to_string(),
            })
        }
    }

    impl ImageGenerationProvider for DiffusionGemmaImageProvider {
        fn kind(&self) -> ImageProviderKind {
            ImageProviderKind::ComputeCoreMlx
        }

        fn capability_report(
            &self,
            model: &InstalledCImage,
            _machine: &MachineProfile,
        ) -> ImageProviderCapability {
            if model.manifest.model_family == ImageModelFamily::DiffusionGemma {
                ImageProviderCapability::ComputeCoreMlxQualified
            } else {
                ImageProviderCapability::ComputeCoreMlxAvailableButUnqualified
            }
        }

        fn generate(
            &self,
            request: &ImageGenerationProviderRequest,
            cancellation: &ImageGenerationCancellationToken,
        ) -> Result<ImageGenerationProviderResult, ImageProviderError> {
            let t0 = std::time::Instant::now();

            if cancellation.is_cancelled() {
                return Err(ImageProviderError::GenerationFailed(
                    "cancelled before execution".into(),
                ));
            }

            let width = request.request.width;
            let height = request.request.height;
            // Each token maps to one RGBA pixel via to_le_bytes().
            let max_tokens = width * height;

            let diffusion_request = DiffusionGenerationRequest {
                model_path: self.model_path.clone(),
                prompt: request.request.prompt.clone(),
                max_tokens,
                steps: Some(request.request.steps),
            };

            let result = self
                .inner
                .generate_text(diffusion_request)
                .map_err(|e| ImageProviderError::GenerationFailed(e.to_string()))?;

            let expected = (width * height * 4) as usize;
            let rgba_bytes = token_vec_to_rgba(&result.tokens, expected);

            let provider_latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

            Ok(ImageGenerationProviderResult {
                rgba_bytes,
                width,
                height,
                provider_latency_ms,
                provider_metadata: ProviderExecutionMetadata {
                    provider_version: env!("CARGO_PKG_VERSION").to_string(),
                    steps_completed: request.request.steps,
                },
                materialization: MaterializationReceipt::new_copied(expected as u64),
            })
        }
    }

    /// Convert a vector of u32 tokens into an RGBA pixel buffer of exactly
    /// `expected_len` bytes.  Tokens are mapped to 4 bytes each (little-endian
    /// u32 representation).  If the token stream is short the buffer is
    /// zero-padded; if long it is truncated.
    fn token_vec_to_rgba(tokens: &[u32], expected_len: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(expected_len);
        for t in tokens {
            buf.extend_from_slice(&t.to_le_bytes());
            if buf.len() >= expected_len {
                buf.truncate(expected_len);
                return buf;
            }
        }
        buf.resize(expected_len, 0u8);
        buf
    }
}

#[cfg(feature = "prism-backend")]
#[cfg(all(feature = "prism-backend", feature = "generation-diffusion"))]
pub use diffusion_gemma_provider::DiffusionGemmaImageProvider;

/// Re-export the fake provider for hermetic tests.
#[cfg(test)]
pub use fake_provider::FakeImageProvider;

// ── Prism LUT provider ──────────────────────────────────────────────────

/// PrismLut image generation provider.
///
/// This provider does not yet have a concrete implementation.  It
/// always reports `Unqualified` from capability checks and returns
/// `ProviderUnavailable` from generate.
pub struct PrismLutImageProvider;

impl PrismLutImageProvider {
    /// Create a new PrismLut provider.
    pub fn new() -> Self {
        Self
    }
}

impl ImageGenerationProvider for PrismLutImageProvider {
    fn kind(&self) -> ImageProviderKind {
        ImageProviderKind::PrismLut
    }

    fn capability_report(
        &self,
        _model: &InstalledCImage,
        _machine: &MachineProfile,
    ) -> ImageProviderCapability {
        ImageProviderCapability::PrismLutAvailableButUnqualified
    }

    fn generate(
        &self,
        _request: &ImageGenerationProviderRequest,
        _cancellation: &ImageGenerationCancellationToken,
    ) -> Result<ImageGenerationProviderResult, ImageProviderError> {
        Err(ImageProviderError::ProviderUnavailable)
    }
}

// ── Fake provider (testing only) ────────────────────────────────────────

/// Fake provider for hermetic tests — not part of the stable API.
#[doc(hidden)]
#[cfg(test)]
mod fake_provider {
    use super::*;

    /// Deterministic fake provider for unit tests.
    ///
    /// Always reports `Qualified` and returns a 2×2 RGBA image with
    /// the pattern (255,0,0,255), (0,255,0,255) repeated across rows.
    pub struct FakeImageProvider;

    impl FakeImageProvider {
        pub fn new() -> Self {
            Self
        }
    }

    impl ImageGenerationProvider for FakeImageProvider {
        fn kind(&self) -> ImageProviderKind {
            ImageProviderKind::ComputeCoreMlx
        }

        fn capability_report(
            &self,
            _model: &InstalledCImage,
            _machine: &MachineProfile,
        ) -> ImageProviderCapability {
            ImageProviderCapability::ComputeCoreMlxQualified
        }

        fn generate(
            &self,
            _request: &ImageGenerationProviderRequest,
            _cancellation: &ImageGenerationCancellationToken,
        ) -> Result<ImageGenerationProviderResult, ImageProviderError> {
            let width = 2u32;
            let height = 2u32;
            // 2×2 RGBA: repeat (255,0,0,255), (0,255,0,255)
            let rgba_bytes = vec![
                255, 0, 0, 255, // Red
                0, 255, 0, 255, // Green
                255, 0, 0, 255, // Red
                0, 255, 0, 255, // Green
            ];

            Ok(ImageGenerationProviderResult {
                rgba_bytes,
                width,
                height,
                provider_latency_ms: 1.0,
                provider_metadata: ProviderExecutionMetadata {
                    provider_version: "fake-0.1.0".to_string(),
                    steps_completed: 4,
                },
                materialization: MaterializationReceipt::new_copied((width * height * 4) as u64),
            })
        }
    }
}
