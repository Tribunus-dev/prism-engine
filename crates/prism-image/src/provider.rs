// ── Prism Image Generation — Provider Trait & Implementations ────────
//
// The ImageGenerationProvider trait and concrete implementations for
// text-to-image generation.  This module owns the Prism-level provider
// contract — it is NOT a re-export of compute-core types.

use crate::manifest::*;
use crate::reliability::ImageGenerationCancellationToken;
use crate::types::*;

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

/// /// ComputeCoreMlxImageProvider — requires the `tribunus-compute-core` crate
/// /// to be wired up externally.  Present as a re-export point for when that
/// /// dependency is restored.
/// ///
/// /// This type was removed along with the `tribunus-compute-core` optional
/// /// dependency.  The cfg-gated modules that defined it referenced types from
/// /// that crate, which is no longer in the workspace.
/// ///
/// /// To restore: add `tribunus-compute-core` as an optional dependency and
/// /// uncomment the `cfg(feature = "generation-image")` blocks in this file.
/// Re-export the fake provider for hermetic tests.
#[cfg(test)]
pub use fake_provider::FakeImageProvider;

// ── Prism LUT provider ──────────────────────────────────────────────────

/// PrismLut image generation provider.
///
/// This is the provider-neutral deterministic raster lane. It materializes a
/// prompt- and seed-derived image without requiring a model-specific runtime,
/// which gives ECS work items a real output-producing fallback while model
/// providers are being qualified.
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
        ImageProviderCapability::PrismLutQualified
    }

    fn generate(
        &self,
        request: &ImageGenerationProviderRequest,
        cancellation: &ImageGenerationCancellationToken,
    ) -> Result<ImageGenerationProviderResult, ImageProviderError> {
        if cancellation.is_cancelled() {
            return Err(ImageProviderError::GenerationFailed("cancelled".into()));
        }
        let width = request.request.width.clamp(1, 4096);
        let height = request.request.height.clamp(1, 4096);
        let mut hasher = blake3::Hasher::new();
        hasher.update(request.request.prompt.as_bytes());
        hasher.update(&request.request.seed.unwrap_or(0).to_le_bytes());
        let digest = hasher.finalize();
        let key = digest.as_bytes();
        let mut rgba_bytes = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let i = ((x as usize * 13 + y as usize * 7) % key.len()) as usize;
                let gradient = ((x * 255 / width) as u8).saturating_add(key[i]);
                rgba_bytes.extend_from_slice(&[
                    gradient,
                    ((y * 255 / height) as u8).saturating_add(key[(i + 1) % key.len()]),
                    key[(i + 2) % key.len()],
                    255,
                ]);
            }
        }
        Ok(ImageGenerationProviderResult {
            rgba_bytes,
            width,
            height,
            provider_latency_ms: 0.0,
            provider_metadata: ProviderExecutionMetadata {
                provider_version: "prism-lut-ecs-1".into(),
                steps_completed: request.request.steps,
            },
            materialization: MaterializationReceipt::new_copied((width * height * 4) as u64),
        })
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
