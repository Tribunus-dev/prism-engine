// ── Prism Image Generation — Foundation Types ──────────────────────────
//
// Stable public API types for the Prism text-to-image generation facade.
// Every type defined here is Prism-owned and does not expose Compute or MLX
// internals through the public surface.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Artifact identity ─────────────────────────────────────────────────

/// Opaque request identifier (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub uuid::Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// BLAKE3 hex digest of an artifact (model, CImage, or output).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactDigest(pub String);

impl fmt::Display for ArtifactDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Type alias for output image digest.
pub type OutputDigest = ArtifactDigest;

// ── Output format and hardware preference ─────────────────────────────

/// Supported output image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageOutputFormat {
    /// Raw 8-bit RGBA pixel data (4 bytes per pixel).
    Rgba8,
    /// Encoded PNG bytes.
    Png,
}

impl fmt::Display for ImageOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgba8 => write!(f, "rgba8"),
            Self::Png => write!(f, "png"),
        }
    }
}

/// Device / provider preference for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Let Prism select the best available qualified provider.
    Auto,
    /// Require the Compute-core MLX image provider.
    ComputeCoreMlx,
    /// Require the Prism LUT image provider.
    PrismLut,
}

impl fmt::Display for DevicePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::ComputeCoreMlx => write!(f, "compute-core-mlx"),
            Self::PrismLut => write!(f, "prism-lut"),
        }
    }
}

/// Execution policy controls behaviour when the requested provider is
/// unavailable or unqualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationExecutionPolicy {
    /// Fail if the requested provider cannot serve the request.
    RequireRequestedProvider,
    /// Fall back to an alternative qualified provider.  The receipt records
    /// `fallback_used = true`.
    AllowQualifiedFallback,
    /// Run admission and routing only — never invoke generation.
    DryRunAdmission,
}

impl fmt::Display for GenerationExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequireRequestedProvider => write!(f, "require-requested"),
            Self::AllowQualifiedFallback => write!(f, "allow-fallback"),
            Self::DryRunAdmission => write!(f, "dry-run"),
        }
    }
}

// ── Provider identity ─────────────────────────────────────────────────

/// Identifies a concrete image-generation provider route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageProviderKind {
    /// Compute-core MLX-backed provider (flux-klein-mlx).
    ComputeCoreMlx,
    /// Palettised LUT engine (future).
    PrismLut,
    /// No provider is available for the requested capability.
    Unavailable,
}

impl fmt::Display for ImageProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComputeCoreMlx => write!(f, "compute-core-mlx"),
            Self::PrismLut => write!(f, "prism-lut"),
            Self::Unavailable => write!(f, "unavailable"),
        }
    }
}

/// How the selected provider was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteOrigin {
    /// The caller explicitly requested this provider.
    ExplicitRequest,
    /// Prism automatically selected this provider.
    AutoSelection,
    /// Fallback from the requested provider to an alternative qualified route.
    QualifiedFallback,
    /// Dry-run — admission and routing only, no execution.
    DryRun,
    /// Experimental override — bypasses normal qualification gates.
    ExperimentalOverride,
}

impl fmt::Display for RouteOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplicitRequest => write!(f, "explicit"),
            Self::AutoSelection => write!(f, "auto"),
            Self::QualifiedFallback => write!(f, "fallback"),
            Self::DryRun => write!(f, "dry-run"),
            Self::ExperimentalOverride => write!(f, "experimental-override"),
        }
    }
}

// ── Qualification ─────────────────────────────────────────────────────

/// Qualification status of an artifact or provider route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationStatus {
    /// Artifact is qualified and may be used for generation.
    Accepted,
    /// Artifact has not been qualified yet.
    Unqualified,
    /// Qualification failed with a reason.
    Declined(String),
}

impl fmt::Display for QualificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => write!(f, "accepted"),
            Self::Unqualified => write!(f, "unqualified"),
            Self::Declined(reason) => write!(f, "declined: {reason}"),
        }
    }
}

// ── Memory residency ─────────────────────────────────────────────────

/// Where output data resides in the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryResidency {
    /// CPU-accessible system memory.
    Cpu,
    /// Apple Unified Memory (GPU + CPU accessible).
    UnifiedGpu,
    /// Discrete GPU memory (requires copy to CPU).
    DiscreteGpu,
    /// Unknown or unrecorded residency.
    Unknown,
}

impl fmt::Display for MemoryResidency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::UnifiedGpu => write!(f, "unified-gpu"),
            Self::DiscreteGpu => write!(f, "discrete-gpu"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ── Materialization ───────────────────────────────────────────────────

/// Records how provider output became a user-visible image.
#[derive(Debug, Clone, Serialize)]
pub struct MaterializationReceipt {
    /// Residency of the output after provider execution.
    pub provider_output_residency: MemoryResidency,
    /// Residency after Prism materialization.
    pub prism_output_residency: MemoryResidency,
    /// Number of distinct copy operations recorded.
    pub copies_recorded: u32,
    /// Total bytes transferred during materialization.
    pub bytes_materialized: u64,
    /// Whether a temporary output buffer was discarded (e.g. after cancellation or failed validation).
    pub temporary_output_discarded: bool,
    /// Whether post-execution cleanup completed successfully.
    pub cleanup_completed: bool,
    /// Whether Prism can substantiate zero-copy output.
    pub zero_copy_claimed: bool,
    /// Human-readable notes about the materialization path.
    pub notes: Vec<String>,
}

impl MaterializationReceipt {
    pub fn new_copied(bytes: u64) -> Self {
        Self {
            provider_output_residency: MemoryResidency::UnifiedGpu,
            prism_output_residency: MemoryResidency::Cpu,
            copies_recorded: 1,
            bytes_materialized: bytes,
            temporary_output_discarded: false,
            cleanup_completed: true,
            zero_copy_claimed: false,
            notes: vec!["standard cpu copy".into()],
        }
    }
}

// ── Warnings ──────────────────────────────────────────────────────────

/// Non-fatal warning emitted during generation.
pub type GenerationWarning = String;

// ── Public request types ──────────────────────────────────────────────

/// User-facing request parameters for text-to-image generation.
#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    /// Text prompt describing the desired image.
    pub prompt: String,
    /// Optional negative prompt for guidance.
    pub negative_prompt: Option<String>,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of denoising steps.
    pub steps: u32,
    /// Optional deterministic seed.
    pub seed: Option<u64>,
    /// Classifier-free guidance scale (None → provider default).
    pub guidance_scale: Option<f32>,
    /// Desired output image format.
    pub output_format: ImageOutputFormat,
    /// Preferred device / provider.
    pub device_preference: DevicePreference,
    /// Execution policy when the requested provider is unavailable.
    pub execution_policy: GenerationExecutionPolicy,
}

impl ImageGenerationRequest {
    /// Create a minimal valid request.  All other fields use defaults.
    pub fn new(prompt: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            width,
            height,
            steps: 4,
            seed: None,
            guidance_scale: None,
            output_format: ImageOutputFormat::Rgba8,
            device_preference: DevicePreference::Auto,
            execution_policy: GenerationExecutionPolicy::AllowQualifiedFallback,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────

/// Generated image output with integrity digest.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Output format of the `bytes` field.
    pub format: ImageOutputFormat,
    /// Pixel data (RGBA8 or encoded PNG).
    pub bytes: Vec<u8>,
    /// BLAKE3 digest of `bytes` for integrity verification.
    pub digest: OutputDigest,
}

impl GeneratedImage {
    /// Returns `true` when the image has valid dimensions and non-empty bytes.
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && !self.bytes.is_empty()
            && (self.format == ImageOutputFormat::Rgba8
                && self.bytes.len() as u64 == self.width as u64 * self.height as u64 * 4
                || self.format == ImageOutputFormat::Png)
    }
}

/// Full provenance receipt for an image generation.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationReceipt {
    /// Unique request identifier.
    pub request_id: RequestId,
    /// Digest of the model artifact used for generation.
    pub model_digest: ArtifactDigest,
    /// Provider originally requested by the caller.
    pub requested_provider: DevicePreference,
    /// Provider that actually executed.
    pub selected_provider: ImageProviderKind,
    /// How the selected provider was determined.
    pub route_origin: RouteOrigin,
    /// Human-readable provider version string.
    pub provider_version: String,
    /// Qualification status of the selected provider for this artifact.
    pub qualification_status: QualificationStatus,
    /// Whether fallback occurred from the requested provider.
    pub fallback_used: bool,
    /// Number of denoising steps requested.
    pub denoising_steps_requested: u32,
    /// Number of denoising steps actually completed.
    pub denoising_steps_completed: u32,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Output image format.
    pub output_format: ImageOutputFormat,
    /// Integrity digest of the output image bytes.
    pub output_digest: OutputDigest,
    /// Total end-to-end latency including Prism overhead.
    pub total_latency_ms: f64,
    /// Provider-internal execution latency (excludes materialization).
    pub provider_latency_ms: f64,
    /// Materialization provenance.
    pub materialization: MaterializationReceipt,
    /// Non-fatal warnings emitted during generation.
    pub warnings: Vec<GenerationWarning>,
}

/// Top-level result returned by `generate_image()`.
#[derive(Debug, Clone)]
pub struct ImageGenerationResult {
    /// Generated image with integrity digest.
    pub image: GeneratedImage,
    /// Full provenance receipt.
    pub receipt: ImageGenerationReceipt,
}

// ── Error taxonomy ────────────────────────────────────────────────────

/// Typed errors for the Prism image-generation facade.
#[derive(Debug, thiserror::Error)]
pub enum ImageGenerationError {
    /// The required compile-time capability feature is not enabled.
    #[error("feature `{capability}` is not enabled")]
    FeatureUnavailable { capability: &'static str },

    /// The provided CImage does not declare image generation capability.
    #[error("CImage {artifact} is not image-generation capable")]
    ArtifactNotImageCapable {
        /// Digest of the model artifact that was checked.
        artifact: ArtifactDigest,
    },

    /// A required component is missing from the installed artifact.
    #[error("missing required component `{component}`")]
    MissingComponent {
        /// Name of the missing component.
        component: String,
    },

    /// The artifact is unqualified for the selected provider.
    #[error("{provider} is not qualified for this artifact: {reason}")]
    ArtifactUnqualified {
        /// Provider that was checked.
        provider: ImageProviderKind,
        /// Why qualification failed.
        reason: String,
    },

    /// The caller requested a provider that is unavailable.
    #[error("requested provider {requested} is unavailable; available: {available:?}")]
    RequestedProviderUnavailable {
        /// What the caller asked for.
        requested: DevicePreference,
        /// What was available at the time.
        available: Vec<ImageProviderKind>,
    },

    /// The selected provider failed during execution.
    #[error("{provider} execution failed: {source}")]
    ProviderExecutionFailed {
        /// Which provider failed.
        provider: ImageProviderKind,
        /// Error source message.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The provider returned an invalid or malformed output.
    #[error("{provider} returned invalid output: {reason}")]
    InvalidOutput {
        /// Provider that produced the output.
        provider: ImageProviderKind,
        /// Why the output is considered invalid.
        reason: String,
    },

    /// The request itself is invalid.
    #[error("unsupported request: {reason}")]
    UnsupportedRequest {
        /// Why the request was rejected.
        reason: String,
    },

    /// Admission gate refused the request.
    #[error("admission refused: {reason:?}")]
    AdmissionRefused {
        /// Structured refusal reason from the admission gate.
        reason: ImageGenerationRefusalReason,
    },
}

/// Structured refusal reasons returned by the admission gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageGenerationRefusalReason {
    /// Dimensions exceed supported range.
    DimensionsUnsupported { width: u32, height: u32 },
    /// Step count outside supported range.
    StepsOutOfRange { steps: u32, min: u32, max: u32 },
    /// Output format not supported by the provider.
    FormatUnsupported(ImageOutputFormat),
    /// A required provider component is absent.
    ComponentAbsent(String),
    /// Artifact has not been qualified.
    NotQualified,
    /// Dry-run only — admission succeeded but execution was skipped.
    DryRun,
    /// Catch-all for custom refusal reasons.
    Other(String),
}

impl fmt::Display for ImageGenerationRefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionsUnsupported { width, height } => {
                write!(f, "dimensions {width}x{height} not supported")
            }
            Self::StepsOutOfRange { steps, min, max } => {
                write!(f, "steps {steps} not in range {min}..={max}")
            }
            Self::FormatUnsupported(fmt) => write!(f, "output format {fmt} not supported"),
            Self::ComponentAbsent(name) => write!(f, "required component `{name}` is absent"),
            Self::NotQualified => write!(f, "artifact not qualified"),
            Self::DryRun => write!(f, "dry-run (admission only)"),
            Self::Other(reason) => write!(f, "{reason}"),
        }
    }
}
