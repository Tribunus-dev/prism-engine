// ── Prism Image Generation — CImage Modality Manifest ───────────────────
//
// Declares the image-generation capability of an installed CImage artifact.
// Prism uses the manifest for admission — it must not infer capability from
// a generic CImage container or checkpoint filename alone.

use super::types::*;
use std::fmt;
use std::time::SystemTime;

/// Schema version for the manifest format.  Bump on incompatible changes.
pub const IMAGE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Identifies the image model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModelFamily {
    Flux,
    StableDiffusion3,
    Sdxl,
    DiffusionGemma,
    Custom,
}

impl fmt::Display for ImageModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flux => write!(f, "flux"),
            Self::StableDiffusion3 => write!(f, "sd3"),
            Self::Sdxl => write!(f, "sdxl"),
            Self::DiffusionGemma => write!(f, "diffusion-gemma"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Whether a model component is available and qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentAvailability {
    /// The component is absent from the artifact.
    Absent,
    /// The component is declared in the manifest but not yet verified.
    PresentUnverified,
    /// The component is present and has passed qualification.
    PresentQualified,
    /// The component is explicitly unsupported.
    Unsupported,
    /// The component is present but qualification was refused.
    RefusedByPolicy,
}

impl fmt::Display for ComponentAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => write!(f, "absent"),
            Self::PresentUnverified => write!(f, "present-unverified"),
            Self::PresentQualified => write!(f, "qualified"),
            Self::Unsupported => write!(f, "unsupported"),
            Self::RefusedByPolicy => write!(f, "refused-by-policy"),
        }
    }
}

/// Constraint on image dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionConstraint {
    /// Exactly this size is supported.
    Exact(u32),
    /// Any value in the inclusive range [min, max].
    Range(u32, u32),
    /// Any value that is a multiple of `alignment`.
    Aligned { min: u32, max: u32, alignment: u32 },
    /// No constraint (any positive value accepted).
    Any,
}

impl DimensionConstraint {
    /// Returns `true` when `value` satisfies this constraint.
    pub fn accepts(&self, value: u32) -> bool {
        match self {
            Self::Exact(v) => *v == value,
            Self::Range(lo, hi) => *lo <= value && value <= *hi,
            Self::Aligned {
                min,
                max,
                alignment,
            } => *min <= value && value <= *max && value % alignment == 0,
            Self::Any => value > 0,
        }
    }
}

/// Supported step range for the denoising scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepRange {
    pub min: u32,
    pub max: u32,
}

impl StepRange {
    pub fn contains(&self, steps: u32) -> bool {
        self.min <= steps && steps <= self.max
    }
}

/// Binds a provider route to its artifact identity and hardware requirements.
#[derive(Debug, Clone)]
pub struct ImageProviderArtifact {
    /// Which provider this artifact targets.
    pub provider: ImageProviderKind,
    /// Provider-specific artifact identifier.
    pub artifact_id: String,
    /// Identifier of the compiler that produced the artifact.
    pub compiler_id: String,
    /// Minimum ABI version required by this artifact.
    pub abi_version: u32,
    /// Hardware features required by this artifact (e.g. "fp16", "ane").
    pub required_hardware: Vec<String>,
    /// Expected tensor layout format.
    pub tensor_layout: String,
    /// Qualification record for this artifact + provider pair.
    pub qualification_record: ImageQualificationRecord,
}

/// Qualification evidence for a specific artifact + provider combination.
#[derive(Debug, Clone)]
pub struct ImageQualificationRecord {
    pub status: QualificationStatus,
    pub fixture_id: String,
    pub compiler_version: String,
    pub runtime_version: String,
    pub machine_fingerprint: String,
    pub request_digest: [u8; 32],
    pub output_digest: Option<OutputDigest>,
    pub observed_width: Option<u32>,
    pub observed_height: Option<u32>,
    pub latency_ms: Option<f64>,
    pub verified_at: SystemTime,
    pub failure_reason: Option<String>,
}

/// The authoritative image-generation capability declaration for a CImage.
#[derive(Debug, Clone)]
pub struct ImageGenerationCapabilityManifest {
    pub schema_version: u32,
    pub model_family: ImageModelFamily,
    pub text_encoder: ComponentAvailability,
    pub denoiser: ComponentAvailability,
    pub vae_decoder: ComponentAvailability,
    pub tokenizer: ComponentAvailability,
    pub scheduler: ComponentAvailability,
    pub supported_dimensions: DimensionConstraint,
    pub supported_steps: StepRange,
    pub provider_artifacts: Vec<ImageProviderArtifact>,
    pub qualification: ImageQualificationRecord,
}

impl Default for ImageGenerationCapabilityManifest {
    /// Returns a minimal manifest that declares all components absent.
    fn default() -> Self {
        Self {
            schema_version: IMAGE_MANIFEST_SCHEMA_VERSION,
            model_family: ImageModelFamily::Custom,
            text_encoder: ComponentAvailability::Absent,
            denoiser: ComponentAvailability::Absent,
            vae_decoder: ComponentAvailability::Absent,
            tokenizer: ComponentAvailability::Absent,
            scheduler: ComponentAvailability::Absent,
            supported_dimensions: DimensionConstraint::Any,
            supported_steps: StepRange { min: 1, max: 50 },
            provider_artifacts: vec![],
            qualification: ImageQualificationRecord {
                status: QualificationStatus::Unqualified,
                fixture_id: String::new(),
                compiler_version: String::new(),
                runtime_version: String::new(),
                machine_fingerprint: String::new(),
                request_digest: [0u8; 32],
                output_digest: None,
                observed_width: None,
                observed_height: None,
                latency_ms: None,
                verified_at: std::time::UNIX_EPOCH,
                failure_reason: None,
            },
        }
    }
}

impl ImageGenerationCapabilityManifest {
    /// Returns `true` when all required image-generation components are
    /// present and qualified.
    pub fn is_admittable(&self) -> bool {
        self.text_encoder == ComponentAvailability::PresentQualified
            && self.denoiser == ComponentAvailability::PresentQualified
            && self.vae_decoder == ComponentAvailability::PresentQualified
            && self.tokenizer == ComponentAvailability::PresentQualified
            && self.scheduler == ComponentAvailability::PresentQualified
    }

    /// Returns the first provider artifact matching `kind`, if any.
    pub fn find_provider_artifact(
        &self,
        kind: ImageProviderKind,
    ) -> Option<&ImageProviderArtifact> {
        self.provider_artifacts.iter().find(|a| a.provider == kind)
    }
}

// ── InstalledCImage ────────────────────────────────────────────────────

/// A CImage artifact that has been discovered, loaded, and has its manifest
/// parsed.  This is the authoritative representation used for admission.
#[derive(Debug, Clone)]
pub struct InstalledCImage {
    /// Path to the installed artifact directory.
    pub path: String,
    /// Digest of the full artifact.
    pub digest: ArtifactDigest,
    /// Image-generation capability manifest.
    pub manifest: ImageGenerationCapabilityManifest,
    /// Provider-specific runtime loaders (lazily initialised).
    pub provider_handles: Vec<ImageProviderKind>,
}

impl InstalledCImage {
    /// Returns `true` when this CImage declares image-generation capability
    /// and at least one provider can serve it.
    pub fn is_image_capable(&self) -> bool {
        self.manifest.is_admittable() && !self.provider_handles.is_empty()
    }
}
