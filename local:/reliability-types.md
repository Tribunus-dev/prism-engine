All reliability hardening types for PRISM-RELIABILITY-AND-RECOVERY-0001.
Append this entire block to the end of types.rs (after ImageGenerationRefusalReason).

```rust
// ── Terminal state model ──────────────────────────────────────────────

/// Terminal state of an image-generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageGenerationTerminalState {
    Succeeded,
    RefusedBeforeExecution,
    FailedDuringExecution,
    SucceededViaQualifiedFallback,
    Cancelled,
}

impl fmt::Display for ImageGenerationTerminalState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Succeeded => write!(f, "succeeded"),
            Self::RefusedBeforeExecution => write!(f, "refused-before-execution"),
            Self::FailedDuringExecution => write!(f, "failed-during-execution"),
            Self::SucceededViaQualifiedFallback => write!(f, "succeeded-via-qualified-fallback"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Terminal receipt produced for every generate_image request.
#[derive(Debug, Clone)]
pub struct ImageGenerationTerminalReceipt {
    pub request_id: RequestId,
    pub terminal_state: ImageGenerationTerminalState,
    pub admission: ImageGenerationAdmissionEvidence,
    pub route: ImageGenerationRouteEvidence,
    pub execution: Option<ImageGenerationExecutionEvidence>,
    pub output: Option<ImageGenerationOutputEvidence>,
    pub failure: Option<ImageGenerationFailureEvidence>,
    pub cancellation: Option<ImageGenerationCancellationEvidence>,
    pub created_at: String,
    pub completed_at: String,
}

/// Public outcome of an image generation request.
#[derive(Debug, Clone)]
pub enum ImageGenerationOutcome {
    Success(ImageGenerationResult),
    Refused(ImageGenerationRefusal),
    Failed(ImageGenerationFailure),
    Cancelled(ImageGenerationCancellation),
}

#[derive(Debug, Clone)]
pub struct ImageGenerationRefusal {
    pub reason: ImageGenerationRefusalReason,
    pub admission_evidence: ImageGenerationAdmissionEvidence,
}

#[derive(Debug, Clone)]
pub struct ImageGenerationFailure {
    pub evidence: ImageGenerationFailureEvidence,
    pub route_evidence: ImageGenerationRouteEvidence,
}

#[derive(Debug, Clone)]
pub struct ImageGenerationCancellation {
    pub evidence: ImageGenerationCancellationEvidence,
    pub route_evidence: ImageGenerationRouteEvidence,
}

/// Full response including terminal receipt.
#[derive(Debug, Clone)]
pub struct ImageGenerationResponse {
    pub outcome: ImageGenerationOutcome,
    pub receipt: ImageGenerationTerminalReceipt,
}

// ── Admission evidence ────────────────────────────────────────────────

/// Evidence produced by the admission gate for every request.
#[derive(Debug, Clone)]
pub struct ImageGenerationAdmissionEvidence {
    pub artifact_digest: ArtifactDigest,
    pub machine_fingerprint: String,
    pub request_digest: [u8; 32],
    pub image_capability_declared: bool,
    pub required_components: Vec<ComponentRequirement>,
    pub present_components: Vec<ComponentRequirement>,
    pub missing_components: Vec<ComponentRequirement>,
    pub requested_dimensions: (u32, u32),
    pub supported_dimensions: DimensionConstraint,
    pub requested_steps: u32,
    pub supported_steps: StepRange,
    pub qualification_status: QualificationStatus,
    pub admitted: bool,
    pub refusal_reason: Option<ImageGenerationRefusalReason>,
}

// ── Route evidence ────────────────────────────────────────────────────

/// Reason why a candidate provider was ineligible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderIneligibilityReason {
    Unavailable,
    Unqualified,
    QualificationStale,
    ArtifactIncompatible,
    MachineIncompatible,
    PolicyProhibited,
}

/// Reason for fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    RequestedProviderUnavailable,
    RequestedProviderUnqualified,
    RequestedProviderFailed,
    RequestedProviderCancelled,
}

/// Evidence for a single provider candidate considered during routing.
#[derive(Debug, Clone)]
pub struct ImageProviderCandidateEvidence {
    pub provider: ImageProviderKind,
    pub capability: ImageProviderCapability,
    pub eligible: bool,
    pub ineligibility_reason: Option<ProviderIneligibilityReason>,
}

/// Evidence produced by the route selector for every request.
#[derive(Debug, Clone)]
pub struct ImageGenerationRouteEvidence {
    pub requested_provider: DevicePreference,
    pub route_origin: RouteOrigin,
    pub candidates: Vec<ImageProviderCandidateEvidence>,
    pub selected_provider: Option<ImageProviderKind>,
    pub attempted_provider: Option<ImageProviderKind>,
    pub fallback_considered: bool,
    pub fallback_attempted: bool,
    pub fallback_provider: Option<ImageProviderKind>,
    pub fallback_reason: Option<FallbackReason>,
    pub selected_provider_qualified: bool,
}

// ── Qualification freshness ──────────────────────────────────────────

/// Key identifying a specific qualification evidence instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualificationFreshnessKey {
    pub artifact_digest: ArtifactDigest,
    pub provider_kind: ImageProviderKind,
    pub provider_version: String,
    pub bridge_version: String,
    pub machine_fingerprint: String,
    pub os_version: String,
    pub runtime_version: String,
    pub execution_policy: GenerationExecutionPolicy,
    pub input_contract_digest: [u8; 32],
    pub output_contract_digest: [u8; 32],
}

/// Whether a qualification record is still valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationFreshness {
    Fresh,
    StaleArtifact,
    StaleProvider,
    StaleMachine,
    StaleRuntime,
    StaleContract,
    Missing,
}

// ── Failure taxonomy ──────────────────────────────────────────────────

/// Stage at which a failure occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImageGenerationFailureStage {
    Admission,
    QualificationResolution,
    ProviderLoad,
    ProviderInitialization,
    PromptPreparation,
    Tokenization,
    ModelWeightResolution,
    TextEncoding,
    Denoising,
    Scheduler,
    VaeDecode,
    OutputMaterialization,
    OutputValidation,
    ReceiptPersistence,
    Cleanup,
}

/// Classification of a failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageGenerationFailureClass {
    ArtifactMalformed,
    ComponentMissing,
    ArtifactIncompatible,
    QualificationStale,
    ProviderUnavailable,
    ProviderUnqualified,
    ProviderInitializationFailed,
    ModelLoadFailed,
    TensorContractMismatch,
    UnsupportedDtype,
    MemoryAllocationFailed,
    ExecutionInterrupted,
    ProviderRuntimeError,
    OutputInvalid,
    ReceiptStoreFailed,
    Unknown,
}

/// Whether a failed request can be retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryability {
    Never,
    RetrySameProvider,
    RetryAfterArtifactRepair,
    RetryAfterMemoryPressureRelief,
    RetryWithQualifiedFallback,
    RetryAfterQualification,
}

/// Whether fallback is permitted after a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackEligibility {
    Forbidden,
    AllowedIfQualified,
    AllowedIfQualifiedAndCompatible,
    AllowedOnlyByExplicitOverride,
}

/// Evidence for a failure.
#[derive(Debug, Clone)]
pub struct ImageGenerationFailureEvidence {
    pub stage: ImageGenerationFailureStage,
    pub class: ImageGenerationFailureClass,
    pub attempted_provider: Option<ImageProviderKind>,
    pub source: String,
    pub retryability: Retryability,
    pub fallback_eligibility: FallbackEligibility,
    pub partial_output_detected: bool,
}

// ── Cancellation ──────────────────────────────────────────────────────

/// Token passed to providers for cooperative cancellation.
#[derive(Debug, Clone)]
pub struct ImageGenerationCancellationToken {
    pub request_id: RequestId,
    pub cancelled: bool,
}

impl ImageGenerationCancellationToken {
    pub fn new(request_id: RequestId) -> Self {
        Self { request_id, cancelled: false }
    }
    pub fn cancel(&mut self) { self.cancelled = true; }
    pub fn is_cancelled(&self) -> bool { self.cancelled }
}

/// Evidence for a cancellation event.
#[derive(Debug, Clone)]
pub struct ImageGenerationCancellationEvidence {
    pub requested_at_stage: ImageGenerationFailureStage,
    pub provider: Option<ImageProviderKind>,
    pub completed_denoising_steps: Option<u32>,
    pub partial_output_discarded: bool,
    pub cleanup_completed: bool,
}

// ── Execution evidence ────────────────────────────────────────────────

/// Evidence produced during provider execution.
#[derive(Debug, Clone)]
pub struct ImageGenerationExecutionEvidence {
    pub provider: ImageProviderKind,
    pub provider_version: String,
    pub denoising_steps_requested: u32,
    pub denoising_steps_completed: u32,
    pub provider_latency_ms: f64,
    pub materialization: MaterializationReceipt,
}

// ── Output evidence ───────────────────────────────────────────────────

/// Evidence produced during output generation and validation.
#[derive(Debug, Clone)]
pub struct ImageGenerationOutputEvidence {
    pub width: u32,
    pub height: u32,
    pub output_format: ImageOutputFormat,
    pub output_digest: OutputDigest,
    pub lifecycle: ImageOutputLifecycle,
    pub bytes_produced: u64,
    pub validation_passed: bool,
}

/// Lifecycle stage of generated output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageOutputLifecycle {
    Uninitialized,
    ProviderProduced,
    Materialized,
    Validated,
    Published,
    Discarded,
}

// ── Memory pressure ───────────────────────────────────────────────────

/// Level of memory pressure for admission decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressureLevel {
    Normal,
    Elevated,
    Critical,
}

// ── Receipt persistence ───────────────────────────────────────────────

/// Whether a terminal receipt was persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptPersistenceState {
    Persisted,
    PersistedVolatileFallback,
    Failed,
}

// ── Reliability metrics ───────────────────────────────────────────────

/// Local observability counters for image generation reliability.
#[derive(Debug, Clone, Default)]
pub struct ImageReliabilityMetrics {
    pub requests_total: u64,
    pub successes_total: u64,
    pub refused_total: u64,
    pub failures_total: u64,
    pub cancellations_total: u64,
    pub qualified_fallbacks_total: u64,
    pub stale_qualification_refusals_total: u64,
    pub invalid_output_rejections_total: u64,
    pub receipt_persistence_failures_total: u64,
    pub provider_failures_by_stage: std::collections::BTreeMap<ImageGenerationFailureStage, u64>,
}
```
