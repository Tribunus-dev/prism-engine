// ── Prism Image Generation — Compatibility Matrix ───────────────────────
//
// Artifact × machine profile × request profile × provider route
// compatibility qualification, receipts, and manifests.
//
// Every type defined here is Prism-owned and does not expose Compute or MLX
// internals through the public surface.
//
// Types are serializable for CI artifact storage and machine-readable
// release manifests.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::reliability::{ImageGenerationAdmissionEvidence, ImageGenerationTerminalReceipt};
use super::types::*;

// ═══════════════════════════════════════════════════════════════════════════
// Supporting types
// ═══════════════════════════════════════════════════════════════════════════

/// ISO 8601 / RFC 3339 timestamp string.
pub type Timestamp = String;

/// Opaque machine fingerprint (derived from hardware + OS + runtime versions).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MachineFingerprint(pub String);

/// Unique receipt identifier (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReceiptId(pub uuid::Uuid);

impl ReceiptId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Identifies a request profile within a compatibility matrix.
pub type RequestProfileId = String;

/// Family of an image model artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageModelFamily {
    /// Stable Diffusion 3 / 3.5 family.
    StableDiffusion3,
    /// Flux family (dev / schnell / pro).
    Flux,
    /// SDXL family (base / refiner).
    Sdxl,
    /// Diffusion Gemma family.
    DiffusionGemma,
    /// Custom / unknown family.
    Custom,
}

impl std::fmt::Display for ImageModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StableDiffusion3 => write!(f, "sd3"),
            Self::Flux => write!(f, "flux"),
            Self::Sdxl => write!(f, "sdxl"),
            Self::DiffusionGemma => write!(f, "diffusion-gemma"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Kind of scheduler used during denoising.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulerKind {
    /// Discrete-time flow-matching scheduler (Flux family).
    FlowMatch,
    /// Continuous-time flow-matching (Diffusion Gemma).
    FlowMatchContinuous,
    /// DDPM / DDIM scheduler.
    Ddpm,
    /// PNDM scheduler.
    Pndm,
    /// DPM-Solver++ scheduler.
    DpmSolverPP,
    /// Euler ancestral scheduler.
    EulerAncestral,
    /// Custom / unknown scheduler.
    Custom,
}

impl std::fmt::Display for SchedulerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlowMatch => write!(f, "flow-match"),
            Self::FlowMatchContinuous => write!(f, "flow-match-continuous"),
            Self::Ddpm => write!(f, "ddpm"),
            Self::Pndm => write!(f, "pndm"),
            Self::DpmSolverPP => write!(f, "dpm-solver++"),
            Self::EulerAncestral => write!(f, "euler-ancestral"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Tensor dtype profile summarizing layouts used in an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorDtypeProfile {
    /// All weights in fp32.
    Fp32,
    /// All weights in fp16 / half.
    Fp16,
    /// Mixed precision (fp16 weights, fp32 accumulation).
    MixedFp16,
    /// BFloat16.
    Bf16,
    /// 4-bit NormalFloat (NF4) with block size.
    Nf4 { block_size: u32 },
    /// Custom dtype profile.
    Custom(String),
}

impl std::fmt::Display for TensorDtypeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp32 => write!(f, "fp32"),
            Self::Fp16 => write!(f, "fp16"),
            Self::MixedFp16 => write!(f, "mixed-fp16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Nf4 { block_size } => write!(f, "nf4-b{block_size}"),
            Self::Custom(s) => write!(f, "custom-{s}"),
        }
    }
}

/// A provider-level requirement that an artifact declares.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderRequirement {
    /// Requires MLX runtime at or above the given version.
    MlxRuntime(String),
    /// Requires Core ML runtime at or above the given version.
    CoreMlRuntime(String),
    /// Requires ANE hardware.
    AneHardware,
    /// Requires minimum unified memory (bytes).
    MinimumMemory(u64),
    /// Requires minimum GPU core count.
    MinimumGpuCores(u16),
    /// Custom requirement.
    Custom(String),
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix types
// ═══════════════════════════════════════════════════════════════════════════

/// Immutable description of an image artifact used as a matrix dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCompatibilityArtifact {
    /// Human-readable identifier for this artifact.
    pub artifact_id: String,
    /// Immutable BLAKE3 digest of the artifact.
    pub artifact_digest: ArtifactDigest,
    /// Model family classification.
    pub model_family: ImageModelFamily,
    /// CImage bundle schema version.
    pub cimage_schema_version: u32,
    /// Digest of the tokenizer component.
    pub tokenizer_digest: ArtifactDigest,
    /// Scheduler kind used by this artifact.
    pub scheduler_kind: SchedulerKind,
    /// Tensor dtype profile of the artifact.
    pub tensor_dtype_profile: TensorDtypeProfile,
    /// Provider-level requirements for execution.
    pub provider_requirements: Vec<ProviderRequirement>,
    /// Request profile IDs that this artifact declares as supported.
    pub supported_request_profiles: Vec<RequestProfileId>,
}

/// Captures enough machine information to make route and performance
/// evidence meaningful.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQualificationMachineProfile {
    /// Hardware fingerprint derived from chip, memory, and runtime versions.
    pub machine_fingerprint: MachineFingerprint,
    /// Human-readable product name (e.g. "MacBook Pro (M1, 2020)").
    pub product_name: String,
    /// Chip family (e.g. "Apple M1", "Apple M3 Pro").
    pub chip_family: String,
    /// Number of performance + efficiency CPU cores.
    pub cpu_core_count: u16,
    /// Number of GPU cores.
    pub gpu_core_count: u16,
    /// Total unified memory in bytes.
    pub unified_memory_bytes: u64,
    /// macOS version (e.g. "15.5").
    pub macos_version: String,
    /// Core ML runtime version string.
    pub coreml_runtime_version: String,
    /// MLX runtime version string.
    pub mlx_runtime_version: String,
    /// Prism engine version string.
    pub prism_version: String,
    /// Compute-core library version string.
    pub compute_core_version: String,
}

/// A single request profile within the matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequestProfile {
    /// Unique identifier for this profile.
    pub id: RequestProfileId,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of denoising steps.
    pub steps: u32,
    /// Deterministic seed for reproducibility.
    pub seed: u64,
    /// Optional CFG guidance scale.
    pub guidance_scale: Option<f32>,
    /// Identifier for the prompt fixture used with this profile.
    pub prompt_fixture_id: String,
    /// Requested output image format.
    pub output_format: ImageOutputFormat,
}

/// How repeatability is verified for a matrix cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageRepeatabilityPolicy {
    /// Every run must produce an identical digest.
    ExactDigest,
    /// Digest must match one entry in an allowlist.
    DigestAllowlist(Vec<ArtifactDigest>),
    /// Compare via structural/perceptual metrics (no exact match required).
    StructuralAndPerceptual,
}

/// Evidence from repeatability qualification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRepeatabilityEvidence {
    /// Number of runs completed.
    pub run_count: u32,
    /// Digests from each run, in order.
    pub output_digests: Vec<ArtifactDigest>,
    /// Number of runs whose digest matched the first run exactly.
    pub exact_matches: u32,
    /// Perceptual distance scores (one per run after the first).
    pub perceptual_distances: Vec<f32>,
    /// Which repeatability policy was applied.
    pub policy: ImageRepeatabilityPolicy,
    /// Whether the evidence satisfies the policy.
    pub passed: bool,
}

/// Performance baseline for a single matrix cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePerformanceBaseline {
    /// Artifact identifier.
    pub artifact_id: String,
    /// Machine fingerprint.
    pub machine_fingerprint: MachineFingerprint,
    /// Request profile identifier.
    pub request_profile_id: RequestProfileId,
    /// Provider that executed.
    pub provider: ImageProviderKind,
    /// Total end-to-end latency in milliseconds.
    pub total_latency_ms: f64,
    /// Provider-internal execution latency in milliseconds.
    pub provider_latency_ms: f64,
    /// Text encoding phase latency (if instrumented).
    pub text_encoding_latency_ms: Option<f64>,
    /// Denoising phase latency (if instrumented).
    pub denoising_latency_ms: Option<f64>,
    /// VAE decode phase latency (if instrumented).
    pub vae_decode_latency_ms: Option<f64>,
    /// Peak estimated memory usage in bytes (source labelled in metadata).
    pub peak_estimated_memory_bytes: Option<u64>,
    /// Output image bytes.
    pub output_bytes: u64,
    /// Number of denoising steps actually completed.
    pub completed_steps: u32,
    /// When this baseline was recorded.
    pub timestamp: Timestamp,
}

/// Tolerance for performance regression detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePerformanceTolerance {
    /// Maximum allowable median latency regression (fraction, e.g. 0.20 = 20%).
    pub max_median_latency_regression_pct: f32,
    /// Maximum allowable p95 latency regression (fraction).
    pub max_p95_latency_regression_pct: f32,
    /// Maximum allowable peak memory regression (fraction).
    pub max_peak_memory_regression_pct: f32,
}

impl Default for ImagePerformanceTolerance {
    fn default() -> Self {
        Self {
            max_median_latency_regression_pct: 0.20,
            max_p95_latency_regression_pct: 0.30,
            max_peak_memory_regression_pct: 0.20,
        }
    }
}

/// Compatibility status for a single matrix cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompatibilityStatus {
    /// Cell has not been evaluated.
    Untried,
    /// Fixture is not available for qualification.
    FixtureUnavailable,
    /// Admission gate refused the request (recorded in receipt).
    AdmissionRefused,
    /// No provider is available for the artifact.
    ProviderUnavailable,
    /// Provider is available but not qualified.
    ProviderUnqualified,
    /// Provider executed successfully and output validated.
    FunctionallyQualified,
    /// Multiple runs produce consistent output within policy.
    RepeatabilityQualified,
    /// All prior criteria met plus performance baselines recorded.
    PerformanceQualified,
    /// Performance regression detected since last baseline.
    PerformanceRegressed,
    /// Reliability probes failed for this cell.
    ReliabilityFailed,
    /// Artifact is incompatible with this machine/provider.
    Incompatible,
}

impl std::fmt::Display for CompatibilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Untried => write!(f, "untried"),
            Self::FixtureUnavailable => write!(f, "fixture-unavailable"),
            Self::AdmissionRefused => write!(f, "admission-refused"),
            Self::ProviderUnavailable => write!(f, "provider-unavailable"),
            Self::ProviderUnqualified => write!(f, "provider-unqualified"),
            Self::FunctionallyQualified => write!(f, "functionally-qualified"),
            Self::RepeatabilityQualified => write!(f, "repeatability-qualified"),
            Self::PerformanceQualified => write!(f, "performance-qualified"),
            Self::PerformanceRegressed => write!(f, "performance-regressed"),
            Self::ReliabilityFailed => write!(f, "reliability-failed"),
            Self::Incompatible => write!(f, "incompatible"),
        }
    }
}

impl CompatibilityStatus {
    /// Returns `true` if this status is sufficient for automatic production routing.
    pub fn is_route_eligible(&self) -> bool {
        matches!(
            self,
            Self::PerformanceQualified | Self::RepeatabilityQualified
        )
    }

    /// Returns `true` if this status is sufficient under development policy.
    pub fn is_development_eligible(&self) -> bool {
        matches!(
            self,
            Self::PerformanceQualified | Self::RepeatabilityQualified | Self::FunctionallyQualified
        )
    }

    /// Returns `true` if the status represents a terminal qualification failure.
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            Self::AdmissionRefused
                | Self::ProviderUnavailable
                | Self::ProviderUnqualified
                | Self::ReliabilityFailed
                | Self::Incompatible
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compatibility receipt & manifest
// ═══════════════════════════════════════════════════════════════════════════

/// Persistent receipt for one artifact–machine–request–provider cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageCompatibilityReceipt {
    /// Unique receipt identifier.
    pub receipt_id: ReceiptId,
    /// Artifact under qualification.
    pub artifact: ImageCompatibilityArtifact,
    /// Machine profile used for qualification.
    pub machine: ImageQualificationMachineProfile,
    /// Request profile executed.
    pub request_profile: ImageRequestProfile,
    /// Provider that executed.
    pub provider: ImageProviderKind,
    /// Final qualification status.
    pub qualification_status: CompatibilityStatus,
    /// Admission evidence from each run.
    #[serde(skip)]
    pub admission_receipts: Vec<ImageGenerationAdmissionEvidence>,
    /// Terminal receipts from each execution run.
    #[serde(skip)]
    pub terminal_receipts: Vec<ImageGenerationTerminalReceipt>,
    /// Repeatability evidence across runs.
    pub repeatability: Option<ImageRepeatabilityEvidence>,
    /// Performance baselines across runs.
    pub performance: Vec<ImagePerformanceBaseline>,
    /// Performance tolerance used for regression detection.
    pub performance_tolerance: Option<ImagePerformanceTolerance>,
    /// Human-readable failure summary (set when status is a failure).
    pub failure_summary: Option<String>,
    /// When this receipt was generated.
    pub generated_at: Timestamp,
}

/// Machine-readable compatibility manifest for release artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismImageCompatibilityManifest {
    /// Schema version of this manifest format (currently 1).
    pub schema_version: u32,
    /// When this manifest was generated.
    pub generated_at: Timestamp,
    /// Prism engine version string.
    pub prism_version: String,
    /// Compute-core library version string.
    pub compute_core_version: String,
    /// All qualified and attempted matrix cells.
    pub cells: Vec<ImageCompatibilityReceipt>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Compatibility runner
// ═══════════════════════════════════════════════════════════════════════════

/// Error type for the compatibility runner.
#[derive(Debug, thiserror::Error)]
pub enum CompatibilityRunnerError {
    /// The requested artifact fixture is not available at the expected path.
    #[error("fixture not available: {0}")]
    FixtureUnavailable(String),
    /// Artifact digest does not match the declared digest.
    #[error("artifact digest mismatch: declared {declared}, actual {actual}")]
    ArtifactDigestMismatch {
        declared: ArtifactDigest,
        actual: ArtifactDigest,
    },
    /// Admission gate refused the request.
    #[error("admission refused: {0}")]
    AdmissionRefused(String),
    /// Provider is not available for this artifact.
    #[error("provider {0} not available")]
    ProviderUnavailable(ImageProviderKind),
    /// Execution failed during qualification.
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    /// Repeatability check failed.
    #[error("repeatability check failed: {0}")]
    RepeatabilityFailed(String),
    /// Resilience probe failed.
    #[error("resilience check failed: {0}")]
    ResilienceFailed(String),
    /// Receipt persistence failed.
    #[error("receipt persistence failed: {0}")]
    ReceiptPersistenceFailed(String),
    /// Internal error during qualification.
    #[error("internal error: {0}")]
    Internal(String),
}

/// A dedicated runner that qualifies one artifact–machine–request–provider
/// cell through the full qualification sequence.
pub trait ImageCompatibilityRunner {
    /// Run the full qualification sequence for one matrix cell.
    ///
    /// The runner:
    /// 1. Resolves the artifact fixture
    /// 2. Validates artifact digest
    /// 3. Resolves the machine profile
    /// 4. Dry-runs admission
    /// 5. Executes repeated image requests
    /// 6. Validates terminal receipts
    /// 7. Calculates repeatability evidence
    /// 8. Aggregates performance baselines
    /// 9. Executes resilience checks
    /// 10. Persists the compatibility receipt
    fn qualify(
        &self,
        artifact: &ImageCompatibilityArtifact,
        machine: &ImageQualificationMachineProfile,
        profile: &ImageRequestProfile,
        provider: ImageProviderKind,
    ) -> Result<ImageCompatibilityReceipt, CompatibilityRunnerError>;
}

/// Dry-run runner that simulates qualification for offline testing.
///
/// Never touches real artifacts, providers, or hardware.  Used for
/// schema validation and test coverage of receipt serialization.
#[derive(Default)]
pub struct DryRunCompatibilityRunner;

impl ImageCompatibilityRunner for DryRunCompatibilityRunner {
    fn qualify(
        &self,
        artifact: &ImageCompatibilityArtifact,
        machine: &ImageQualificationMachineProfile,
        profile: &ImageRequestProfile,
        provider: ImageProviderKind,
    ) -> Result<ImageCompatibilityReceipt, CompatibilityRunnerError> {
        let now = iso_now();
        let status = CompatibilityStatus::FunctionallyQualified;

        Ok(ImageCompatibilityReceipt {
            receipt_id: ReceiptId::new(),
            artifact: artifact.clone(),
            machine: machine.clone(),
            request_profile: profile.clone(),
            provider,
            qualification_status: status,
            admission_receipts: vec![],
            terminal_receipts: vec![],
            repeatability: None,
            performance: vec![],
            performance_tolerance: None,
            failure_summary: None,
            generated_at: now,
        })
    }
}

/// Build an ISO 8601 timestamp string from the current system time.
pub fn iso_now() -> String {
    // Simple format without pulling in chrono: YYYY-MM-DDTHH:MM:SS.ssssssZ
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let secs = dur.as_secs();
    let subsec = dur.subsec_micros();

    // Split into date/time components
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let secs_rem = time_secs % 60;

    // Days since epoch -> year/month/day (Graham-Toomey algorithm)
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
        y, m, d, hours, mins, secs_rem, subsec
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: build a MachineFingerprint from runtime constants
// ═══════════════════════════════════════════════════════════════════════════

/// Given the machine profile fields (minus fingerprint), derive a
/// fingerprint string from the most stable identifiers.
pub fn build_machine_fingerprint(
    chip_family: &str,
    cpu_core_count: u16,
    gpu_core_count: u16,
    unified_memory_bytes: u64,
    macos_version: &str,
    prism_version: &str,
) -> MachineFingerprint {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    chip_family.hash(&mut hasher);
    cpu_core_count.hash(&mut hasher);
    gpu_core_count.hash(&mut hasher);
    unified_memory_bytes.hash(&mut hasher);
    macos_version.hash(&mut hasher);
    prism_version.hash(&mut hasher);
    MachineFingerprint(format!("{:016x}", hasher.finish()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "generation-image")]
mod tests {
    use super::*;

    fn sample_artifact() -> ImageCompatibilityArtifact {
        ImageCompatibilityArtifact {
            artifact_id: "test-flux-schnell-v1".into(),
            artifact_digest: ArtifactDigest("aabbccdd".into()),
            model_family: ImageModelFamily::Flux,
            cimage_schema_version: 1,
            tokenizer_digest: ArtifactDigest("token123".into()),
            scheduler_kind: SchedulerKind::FlowMatch,
            tensor_dtype_profile: TensorDtypeProfile::MixedFp16,
            provider_requirements: vec![ProviderRequirement::MlxRuntime("0.20".into())],
            supported_request_profiles: vec!["smoke".into(), "nominal".into(), "boundary".into()],
        }
    }

    fn sample_machine() -> ImageQualificationMachineProfile {
        ImageQualificationMachineProfile {
            machine_fingerprint: MachineFingerprint("deadbeef".into()),
            product_name: "MacBook Pro".into(),
            chip_family: "Apple M1".into(),
            cpu_core_count: 8,
            gpu_core_count: 8,
            unified_memory_bytes: 16_000_000_000,
            macos_version: "15.5".into(),
            coreml_runtime_version: "4.0".into(),
            mlx_runtime_version: "0.20.0".into(),
            prism_version: "0.1.0".into(),
            compute_core_version: "0.1.0".into(),
        }
    }

    fn sample_profile(id: &str) -> ImageRequestProfile {
        ImageRequestProfile {
            id: id.into(),
            width: 256,
            height: 256,
            steps: 4,
            seed: 42,
            guidance_scale: None,
            prompt_fixture_id: "synthetic-prompt-v1".into(),
            output_format: ImageOutputFormat::Rgba8,
        }
    }

    #[test]
    fn compatibility_artifact_constructs() {
        let a = sample_artifact();
        assert_eq!(a.artifact_id, "test-flux-schnell-v1");
        assert_eq!(a.provider_requirements.len(), 1);
    }

    #[test]
    fn compatibility_machine_profile_constructs() {
        let m = sample_machine();
        assert_eq!(m.chip_family, "Apple M1");
        assert_eq!(m.gpu_core_count, 8);
    }

    #[test]
    fn compatibility_request_profile_constructs() {
        let p = sample_profile("smoke");
        assert_eq!(p.id, "smoke");
        assert_eq!(p.width, 256);
        assert_eq!(p.seed, 42);
    }

    #[test]
    fn compatibility_status_display() {
        assert_eq!(CompatibilityStatus::Untried.to_string(), "untried");
        assert_eq!(
            CompatibilityStatus::FixtureUnavailable.to_string(),
            "fixture-unavailable"
        );
        assert_eq!(
            CompatibilityStatus::FunctionallyQualified.to_string(),
            "functionally-qualified"
        );
        assert_eq!(
            CompatibilityStatus::RepeatabilityQualified.to_string(),
            "repeatability-qualified"
        );
        assert_eq!(
            CompatibilityStatus::PerformanceQualified.to_string(),
            "performance-qualified"
        );
        assert_eq!(
            CompatibilityStatus::PerformanceRegressed.to_string(),
            "performance-regressed"
        );
    }

    #[test]
    fn compatibility_status_route_eligibility() {
        assert!(CompatibilityStatus::PerformanceQualified.is_route_eligible());
        assert!(CompatibilityStatus::RepeatabilityQualified.is_route_eligible());
        assert!(!CompatibilityStatus::FunctionallyQualified.is_route_eligible());
        assert!(!CompatibilityStatus::Untried.is_route_eligible());
        assert!(!CompatibilityStatus::FixtureUnavailable.is_route_eligible());
        assert!(!CompatibilityStatus::AdmissionRefused.is_route_eligible());
        assert!(!CompatibilityStatus::Incompatible.is_route_eligible());
    }

    #[test]
    fn compatibility_status_development_eligibility() {
        assert!(CompatibilityStatus::PerformanceQualified.is_development_eligible());
        assert!(CompatibilityStatus::RepeatabilityQualified.is_development_eligible());
        assert!(CompatibilityStatus::FunctionallyQualified.is_development_eligible());
        assert!(!CompatibilityStatus::Untried.is_development_eligible());
        assert!(!CompatibilityStatus::FixtureUnavailable.is_development_eligible());
        assert!(!CompatibilityStatus::Incompatible.is_development_eligible());
    }

    #[test]
    fn compatibility_status_failure() {
        assert!(CompatibilityStatus::AdmissionRefused.is_failure());
        assert!(CompatibilityStatus::ProviderUnavailable.is_failure());
        assert!(CompatibilityStatus::ProviderUnqualified.is_failure());
        assert!(CompatibilityStatus::ReliabilityFailed.is_failure());
        assert!(CompatibilityStatus::Incompatible.is_failure());
        assert!(!CompatibilityStatus::FunctionallyQualified.is_failure());
        assert!(!CompatibilityStatus::RepeatabilityQualified.is_failure());
        assert!(!CompatibilityStatus::Untried.is_failure());
    }

    #[test]
    fn compatibility_artifact_serde_roundtrip() {
        let a = sample_artifact();
        let json = serde_json::to_string(&a).unwrap();
        let back: ImageCompatibilityArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(a.artifact_id, back.artifact_id);
        assert_eq!(a.artifact_digest, back.artifact_digest);
        assert_eq!(a.model_family, back.model_family);
    }

    #[test]
    fn compatibility_machine_serde_roundtrip() {
        let m = sample_machine();
        let json = serde_json::to_string(&m).unwrap();
        let back: ImageQualificationMachineProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(m.machine_fingerprint, back.machine_fingerprint);
        assert_eq!(m.chip_family, back.chip_family);
    }

    #[test]
    fn compatibility_receipt_serde_roundtrip() {
        let artifact = sample_artifact();
        let machine = sample_machine();
        let profile = sample_profile("smoke");
        let receipt = ImageCompatibilityReceipt {
            receipt_id: ReceiptId::new(),
            artifact,
            machine,
            request_profile: profile,
            provider: ImageProviderKind::ComputeCoreMlx,
            qualification_status: CompatibilityStatus::FunctionallyQualified,
            admission_receipts: vec![],
            terminal_receipts: vec![],
            repeatability: None,
            performance: vec![],
            performance_tolerance: None,
            failure_summary: None,
            generated_at: iso_now(),
        };
        let json = serde_json::to_string(&receipt).unwrap();
        let back: ImageCompatibilityReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.receipt_id, back.receipt_id);
        assert_eq!(receipt.qualification_status, back.qualification_status);
        assert_eq!(receipt.provider, back.provider);
    }

    #[test]
    fn compatibility_manifest_serde_roundtrip() {
        let manifest = PrismImageCompatibilityManifest {
            schema_version: 1,
            generated_at: iso_now(),
            prism_version: "0.1.0".into(),
            compute_core_version: "0.1.0".into(),
            cells: vec![],
        };
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        assert!(json.contains("prism_version"));
        assert!(json.contains("compute_core_version"));
        let back: PrismImageCompatibilityManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.schema_version, 1);
        assert_eq!(back.cells.len(), 0);
    }

    #[test]
    fn dry_run_runner_produces_receipt() {
        let runner = DryRunCompatibilityRunner;
        let artifact = sample_artifact();
        let machine = sample_machine();
        let profile = sample_profile("smoke");

        let receipt = runner
            .qualify(
                &artifact,
                &machine,
                &profile,
                ImageProviderKind::ComputeCoreMlx,
            )
            .unwrap();

        assert_eq!(receipt.artifact.artifact_id, "test-flux-schnell-v1");
        assert_eq!(
            receipt.qualification_status,
            CompatibilityStatus::FunctionallyQualified
        );
        assert_eq!(receipt.provider, ImageProviderKind::ComputeCoreMlx);
    }

    #[test]
    fn machine_fingerprint_is_deterministic() {
        let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
        let fp2 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn machine_fingerprint_changes_on_input() {
        let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
        let fp2 = build_machine_fingerprint("Apple M2", 10, 10, 16_000_000_000, "15.5", "0.1.0");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn performance_tolerance_default() {
        let t = ImagePerformanceTolerance::default();
        assert!((t.max_median_latency_regression_pct - 0.20).abs() < 1e-6);
        assert!((t.max_p95_latency_regression_pct - 0.30).abs() < 1e-6);
        assert!((t.max_peak_memory_regression_pct - 0.20).abs() < 1e-6);
    }

    #[test]
    fn iso_now_format() {
        let ts = iso_now();
        // ISO 8601: YYYY-MM-DDTHH:MM:SS.ffffffZ
        assert_eq!(ts.len(), 27, "expected 27-char ISO 8601: got {ts}");
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
        assert_eq!(&ts[13..14], ":");
        assert_eq!(&ts[16..17], ":");
        assert_eq!(&ts[19..20], ".");
        assert_eq!(&ts[26..27], "Z");
    }
}
