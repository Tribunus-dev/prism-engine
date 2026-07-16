// ── Prism LLM Inference — Server Types ─────────────────────────────────
//
// Foundation types for the Prism LLM inference server: session lifecycle,
// KV-cache epochs, lane dispatch, island allocation, and end-to-end
// inference receipts.
//
// Ported from src/llm/server.rs. Dependent types from manifest.rs and
// image/types.rs are defined inline until a shared types crate exists.

use serde::{Deserialize, Serialize};
use std::fmt;

// ========================================================================
// Dependent types — ported inline from src/llm/manifest.rs
// ========================================================================

/// BLAKE3 hex digest of an artifact (model, CImage, or output).
/// Ported from src/image/types.rs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactDigest(pub String);

impl fmt::Display for ArtifactDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Qualification status of an artifact or provider route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestQualificationStatus {
    Accepted,
    Unqualified,
    Declined(String),
}

impl fmt::Display for ManifestQualificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => write!(f, "accepted"),
            Self::Unqualified => write!(f, "unqualified"),
            Self::Declined(reason) => write!(f, "declined: {reason}"),
        }
    }
}

/// Step within the LLM inference lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManifestInferencePhase {
    SessionAdmission,
    CImageLoad,
    WeightResidency,
    Tokenization,
    PromptPrefill,
    KvAllocation,
    KvEpochPublication,
    Decode,
    Sampling,
    AuxiliaryInference,
    KvCompression,
    KvRefreshPrefill,
    OutputStreaming,
    Cancellation,
    Recovery,
    Cleanup,
}

impl fmt::Display for ManifestInferencePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionAdmission => write!(f, "session-admission"),
            Self::CImageLoad => write!(f, "cimage-load"),
            Self::WeightResidency => write!(f, "weight-residency"),
            Self::Tokenization => write!(f, "tokenization"),
            Self::PromptPrefill => write!(f, "prompt-prefill"),
            Self::KvAllocation => write!(f, "kv-allocation"),
            Self::KvEpochPublication => write!(f, "kv-epoch-publication"),
            Self::Decode => write!(f, "decode"),
            Self::Sampling => write!(f, "sampling"),
            Self::AuxiliaryInference => write!(f, "auxiliary-inference"),
            Self::KvCompression => write!(f, "kv-compression"),
            Self::KvRefreshPrefill => write!(f, "kv-refresh-prefill"),
            Self::OutputStreaming => write!(f, "output-streaming"),
            Self::Cancellation => write!(f, "cancellation"),
            Self::Recovery => write!(f, "recovery"),
            Self::Cleanup => write!(f, "cleanup"),
        }
    }
}

/// Execution lane that processes tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManifestExecutionLane {
    CpuControlPlane,
    Accelerate,
    Metal,
    CoreMlAne,
    UnifiedMemoryIsland,
}

impl fmt::Display for ManifestExecutionLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuControlPlane => write!(f, "cpu-control-plane"),
            Self::Accelerate => write!(f, "accelerate"),
            Self::Metal => write!(f, "metal"),
            Self::CoreMlAne => write!(f, "coreml-ane"),
            Self::UnifiedMemoryIsland => write!(f, "unified-memory-island"),
        }
    }
}

/// How tensor data moves between lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManifestTransferKind {
    ZeroCopyRetained,
    SharedMemoryMapped,
    ExplicitCopy,
    CpuReadback,
    TensorLayoutConversion,
    DtypeConversion,
    ProviderOpaqueMaterialization,
    Serialization,
    Unknown,
}

impl fmt::Display for ManifestTransferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCopyRetained => write!(f, "zero-copy-retained"),
            Self::SharedMemoryMapped => write!(f, "shared-memory-mapped"),
            Self::ExplicitCopy => write!(f, "explicit-copy"),
            Self::CpuReadback => write!(f, "cpu-readback"),
            Self::TensorLayoutConversion => write!(f, "tensor-layout-conversion"),
            Self::DtypeConversion => write!(f, "dtype-conversion"),
            Self::ProviderOpaqueMaterialization => {
                write!(f, "provider-opaque-materialization")
            }
            Self::Serialization => write!(f, "serialization"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Why a materialization event occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ManifestMaterializationReason {
    WeightLoad,
    LaneTransition,
    LaneOutputConsumption,
    KvPageAllocation,
    SamplingBuffer,
    AuxiliaryIslandIo,
    StreamingStaging,
    ProviderOpaque,
    Unknown,
}

impl fmt::Display for ManifestMaterializationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightLoad => write!(f, "weight-load"),
            Self::LaneTransition => write!(f, "lane-transition"),
            Self::LaneOutputConsumption => write!(f, "lane-output-consumption"),
            Self::KvPageAllocation => write!(f, "kv-page-allocation"),
            Self::SamplingBuffer => write!(f, "sampling-buffer"),
            Self::AuxiliaryIslandIo => write!(f, "auxiliary-island-io"),
            Self::StreamingStaging => write!(f, "streaming-staging"),
            Self::ProviderOpaque => write!(f, "provider-opaque"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Monotonically increasing materialization event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ManifestMaterializationEventId(pub u64);

/// Unique session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ManifestSessionId(pub uuid::Uuid);

/// Identifier for a single allocation within an execution island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ManifestIslandAllocationId(pub u64);

/// Identity of a tensor within a materialization event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestTensorIdentity {
    /// Provider-level weight key, if the tensor is a loaded weight.
    pub weight_key: Option<String>,
    /// Role the tensor plays (e.g. "query", "key", "value", "output").
    pub tensor_role: Option<String>,
    /// Allocation slot within the execution island.
    pub allocation_id: ManifestIslandAllocationId,
}

/// Shape, dtype, and layout of a tensor at a materialization boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestTensorRepresentation {
    /// Data type string (e.g. "fp16", "bf16", "int8").
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Layout string (e.g. "nchw", "nhwc", "blocked-2x4").
    pub layout: String,
}

/// Records a single materialization event between execution lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestMaterializationEvent {
    /// Unique event identifier.
    pub event_id: ManifestMaterializationEventId,
    /// Session in which this event occurred.
    pub session_id: ManifestSessionId,
    /// Lane that produced the data.
    pub source_lane: ManifestExecutionLane,
    /// Lane that consumed the data.
    pub destination_lane: ManifestExecutionLane,
    /// Island allocation identifier.
    pub allocation_id: ManifestIslandAllocationId,
    /// Identity of the tensor being transferred.
    pub tensor_identity: ManifestTensorIdentity,
    /// Method of transfer.
    pub transfer_kind: ManifestTransferKind,
    /// Number of bytes transferred, if known.
    pub byte_count: Option<u64>,
    /// Representation at the source lane boundary.
    pub source_representation: ManifestTensorRepresentation,
    /// Representation at the destination lane boundary.
    pub destination_representation: ManifestTensorRepresentation,
    /// Why the materialization was performed.
    pub reason: ManifestMaterializationReason,
    /// ISO 8601 timestamp of the event.
    pub timestamp: String,
}

// ========================================================================
// Server type aliases
// ========================================================================

/// Opaque request identifier (UUID v4).
pub type RequestId = uuid::Uuid;

/// Opaque dispatch identifier for a lane execution unit.
pub type DispatchId = u64;

/// Identifier for a KV-cache epoch.
pub type KvEpochId = u64;

/// Identifier for a single KV-cache page.
pub type KvPageId = u64;

/// Identifier for a completion fence synchronising lane dispatches.
pub type CompletionFenceId = u64;

/// Identifier for a compiled CImage artifact.
pub type CImageId = String;

/// Identifier for a context profile configuration.
pub type ContextProfileId = String;

/// Opaque signature identifying a device's capability profile.
pub type DeviceCapabilitySignature = String;

/// Identifier for a multi-island inference receipt.
pub type ReceptionId = uuid::Uuid;

// ========================================================================
// Key structs
// ========================================================================

/// Key identifying a unique weight residency on device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightResidencyKey {
    pub cimage_digest: ArtifactDigest,
    pub tensor_manifest_digest: ArtifactDigest,
    pub provider_kind: String,
    pub dtype_profile: String,
}

/// Describes the action to take when recovering an inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRecoveryAction {
    pub action: InferenceRecoveryActionKind,
    pub retry_count: u32,
    pub max_retries: u32,
}

// ── Enums ────────────────────────────────────────────────────────────

/// Lease mode for an island allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationLeaseMode {
    Read,
    Write,
    ExclusiveWrite,
}

/// Lifecycle state of an island allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IslandAllocationState {
    Allocated,
    Resident,
    Shared,
    InFlight,
    PendingRelease,
    Reclaimed,
    Invalidated,
}

/// Owner responsible for an island allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllocationOwner {
    WeightResidency,
    KvCache,
    TokenBuffer,
    SamplingBuffer,
    AuxiliaryIslandInput,
    AuxiliaryIslandOutput,
    StreamingStaging,
    Unknown,
}

/// Visibility set of hardware lanes an allocation is accessible from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LaneVisibilitySet {
    MetalOnly,
    AccelerateOnly,
    CpuOnly,
    MetalAndAccelerate,
    MetalAndCoreMl,
    All,
    Unknown,
}

/// Lifecycle state of an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceSessionState {
    Created,
    Admitting,
    LoadingCImage,
    EstablishingResidency,
    Resident,
    Prefilling,
    PublishingKvEpoch,
    Ready,
    Decoding,
    CompressingKv,
    RefreshingContext,
    Cancelling,
    Recovering,
    Completed,
    Cancelled,
    Failed,
    Closed,
}

/// Terminal outcome of an inference session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceTerminalState {
    Succeeded,
    RefusedBeforeExecution,
    FailedDuringPrefill,
    FailedDuringDecode,
    FailedDuringAuxiliaryWork,
    Cancelled,
    RecoveredAndSucceeded,
}

/// Kind of recovery action to take on inference failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceRecoveryActionKind {
    RetryAuxiliaryLane,
    SkipOptionalAuxiliaryLane,
    RebuildKvFromCheckpoint,
    ContextRefresh,
    FailSession,
}

/// Class of inference failure for diagnostics and recovery routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceFailureClass {
    CImageAdmissionFailed,
    WeightResidencyFailed,
    UnifiedMemoryAllocationFailed,
    MetalPrefillFailed,
    MetalDecodeFailed,
    AccelerateStageFailed,
    CoreMlAuxiliaryFailed,
    KvEpochPublicationFailed,
    KvCompressionFailed,
    ContextRefreshFailed,
    StreamBackpressureExceeded,
    ReceiptPersistenceFailed,
    CleanupFailed,
    Unknown,
}

/// Action taken when a consumer cannot keep up with the token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlowConsumerAction {
    PauseGeneration,
    CancelGeneration,
    DropNonTerminalStatusEvents,
}

/// Policy governing which execution lanes are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceExecutionPolicy {
    RequireMetalDecode,
    AllowQualifiedFallback,
    AllowExperimentalLanes,
}

/// Whether an auxiliary lane is optional, required, or disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuxiliaryLanePolicy {
    Optional,
    Required,
    Disabled,
}

/// Eviction status of a weight residency on device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightEvictionStatus {
    Retained,
    Evicted,
    Ineligible,
}

/// Visibility state of CoreML on the compute graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreMlVisibilityState {
    NotVisible,
    CpuVisible,
    GpuVisible,
    AneVisible,
    Full,
}

/// Lifecycle state of a single KV-cache page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvPageState {
    Allocated,
    Writing,
    Sealed,
    Active,
    RetainedSparse,
    PendingReclaim,
    Reclaimed,
    Invalidated,
}

/// Lifecycle state of a KV-cache epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvEpochState {
    Building,
    Active,
    Superseded,
    Draining,
    Reclaimable,
    Invalidated,
}

/// Contract for position encoding within a KV dispatch view.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopePositionContract {
    Absolute { start: u32 },
    Relative { delta: i32 },
    Custom(String),
}

/// Pressure level on unified memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Normal,
    Elevated,
    Critical,
}

// ── Server structs ───────────────────────────────────────────────────

/// Top-level inference server descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismInferenceServer {
    pub admission_gate: String,
    pub cimage_registry: String,
    pub residency_manager: String,
    pub kv_manager: String,
    pub scheduler: String,
    pub lane_router: String,
    pub receipt_store: String,
    pub session_registry: String,
    pub memory_pressure_monitor: String,
}

/// Request to create a new inference session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub cimage_id: CImageId,
    pub context_profile: ContextProfileId,
    pub execution_policy: InferenceExecutionPolicy,
    pub auxiliary_lane_policy: AuxiliaryLanePolicy,
}

/// Request to generate tokens from a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub session_id: ManifestSessionId,
    pub prompt: String,
    pub max_new_tokens: u32,
    pub sampling: SamplingConfig,
    pub stream: bool,
}

/// Sampling parameters for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: Option<f32>,
}

/// Handle used to cancel an in-flight inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationHandle {
    pub session_id: ManifestSessionId,
    pub request_id: RequestId,
}

/// Describes an allocation of unified memory for a specific island.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandAllocation {
    pub allocation_id: ManifestIslandAllocationId,
    pub bytes: u64,
    pub residency: String,
    pub owner: AllocationOwner,
    pub lane_visibility: LaneVisibilitySet,
    pub lease_count: u32,
    pub epoch: Option<KvEpochId>,
    pub state: IslandAllocationState,
}

/// Describes a single lane dispatch unit within an inference execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneDispatch {
    pub dispatch_id: DispatchId,
    pub lane: ManifestExecutionLane,
    pub phase: ManifestInferencePhase,
    pub input_allocations: Vec<ManifestIslandAllocationId>,
    pub output_allocations: Vec<ManifestIslandAllocationId>,
    pub required_epoch: Option<KvEpochId>,
    pub dependencies: Vec<DispatchId>,
    pub completion_fence: CompletionFenceId,
}

/// A single page in the KV-cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvPage {
    pub page_id: KvPageId,
    pub layer_range: (u32, u32),
    pub token_range: (u32, u32),
    pub original_position_range: (u32, u32),
    pub allocation_id: ManifestIslandAllocationId,
    pub residency: String,
    pub state: KvPageState,
}

/// A complete KV-cache epoch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEpoch {
    pub epoch_id: KvEpochId,
    pub parent_epoch: Option<KvEpochId>,
    pub generation_token_index: u64,
    pub logical_context_length: u32,
    pub retained_token_count: u32,
    pub state: KvEpochState,
}

/// View into a KV epoch as seen by a dispatch, including position encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvDispatchView {
    pub epoch_id: KvEpochId,
    pub absolute_decode_position: u32,
    pub rope_position_contract: RopePositionContract,
}

/// Plan for sparse retention of KV-cache pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseRetentionPlan {
    pub source_epoch: KvEpochId,
    pub retained_pages: Vec<KvPageId>,
    pub removed_pages: Vec<KvPageId>,
    pub preserves_absolute_positions: bool,
    pub target_epoch: KvEpochId,
}

/// Plan for refreshing context via a new prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRefreshPlan {
    pub source_epoch: KvEpochId,
    pub retained_source_ranges: Vec<(u32, u32)>,
    pub new_prompt_digest: ArtifactDigest,
    pub target_epoch: KvEpochId,
}

/// Policy governing behaviour when a streaming consumer is slow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamBackpressurePolicy {
    pub max_buffered_events: usize,
    pub max_buffered_tokens: usize,
    pub slow_consumer_timeout_secs: f64,
    pub action_on_overflow: SlowConsumerAction,
}

/// Receipt confirming weight residency on device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightResidencyReceipt {
    pub cimage_digest: ArtifactDigest,
    pub cache_hit: bool,
    pub initial_load_bytes: u64,
    pub decode_step_reload_count: u32,
    pub active_weight_leases: u32,
    pub metal_visible: bool,
    pub accelerate_visible: bool,
    pub coreml_auxiliary_visibility: CoreMlVisibilityState,
    pub materialization_events: Vec<ManifestMaterializationEvent>,
    pub eviction_status: WeightEvictionStatus,
}

/// Receipt for a Metal compute dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalExecutionReceipt {
    pub dispatch_id: DispatchId,
    pub phase: ManifestInferencePhase,
    pub kv_epoch: Option<KvEpochId>,
    pub command_submission_time: String,
    pub completion_time: String,
    pub input_allocation_ids: Vec<ManifestIslandAllocationId>,
    pub output_allocation_ids: Vec<ManifestIslandAllocationId>,
    pub authoritative_result_committed: bool,
}

/// Receipt for an Accelerate framework dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateExecutionReceipt {
    pub dispatch_id: DispatchId,
    pub operations: Vec<String>,
    pub shared_memory_mapped: bool,
    pub cpu_readback: bool,
    pub fallback_used: bool,
}

/// Receipt for a CoreML auxiliary island execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlAuxiliaryReceipt {
    pub auxiliary_island_id: String,
    pub artifact_digest: ArtifactDigest,
    pub source_epoch: Option<KvEpochId>,
    pub qualification_status: ManifestQualificationStatus,
    pub input_contract_verified: bool,
    pub output_contract_verified: bool,
    pub provider_opaque_materialization: bool,
}

/// Receipt for a single lane execution (Metal, Accelerate, or CoreML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneExecutionReceipt {
    pub lane: ManifestExecutionLane,
    pub metal: Option<MetalExecutionReceipt>,
    pub accelerate: Option<AccelerateExecutionReceipt>,
    pub coreml: Option<CoreMlAuxiliaryReceipt>,
}

/// Receipt produced when an inference session is cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCancelledReceipt {
    pub session_id: ManifestSessionId,
    pub request_id: RequestId,
    pub state_at_cancellation: InferenceSessionState,
    pub active_epoch: Option<KvEpochId>,
    pub completed_decode_tokens: u32,
    pub cleanup_completed: bool,
}

/// Receipt produced when an inference session fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceFailureReceipt {
    pub class: InferenceFailureClass,
    pub phase: ManifestInferencePhase,
    pub lane: Option<ManifestExecutionLane>,
    pub retryable: bool,
    pub recovery_action: Option<InferenceRecoveryActionKind>,
}

/// Receipt recording a memory-pressure event and the action taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureReceipt {
    pub level: MemoryPressureLevel,
    pub timestamp: String,
    pub action_taken: String,
}

/// Receipt for the outcome of session admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceAdmissionReceipt {
    pub cimage_id: CImageId,
    pub context_profile: ContextProfileId,
    pub execution_policy: InferenceExecutionPolicy,
    pub admitted: bool,
    pub refusal_reason: Option<String>,
}

/// Receipt summarising the output of an inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutputReceipt {
    pub total_tokens: u32,
    pub tokens_per_second: f64,
    pub total_latency_ms: f64,
    pub metal_decode_latency_ms: f64,
}

/// Receipt for a single KV epoch in the inference execution history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvEpochReceipt {
    pub epoch_id: KvEpochId,
    pub parent_epoch: Option<KvEpochId>,
    pub logical_context_length: u32,
    pub state: KvEpochState,
}

/// Complete end-to-end receipt for a multi-island inference execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiIslandInferenceReceipt {
    pub receipt_id: ReceptionId,
    pub session_id: ManifestSessionId,
    pub request_id: RequestId,
    pub terminal_state: InferenceTerminalState,
    pub cimage_digest: ArtifactDigest,
    pub context_profile: ContextProfileId,
    pub admission: InferenceAdmissionReceipt,
    pub weight_residency: WeightResidencyReceipt,
    pub lane_receipts: Vec<LaneExecutionReceipt>,
    pub kv_history: Vec<KvEpochReceipt>,
    pub materialization_events: Vec<ManifestMaterializationEvent>,
    pub output: Option<InferenceOutputReceipt>,
    pub failure: Option<InferenceFailureReceipt>,
    pub cancellation: Option<InferenceCancelledReceipt>,
    pub memory_pressure_history: Vec<MemoryPressureReceipt>,
    pub started_at: String,
    pub completed_at: String,
}

/// Qualification record for long-context capabilities of a CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongContextQualificationRecord {
    pub cimage_digest: ArtifactDigest,
    pub context_profile: ContextProfileId,
    pub metal_decode_qualified: bool,
    pub accelerate_stage_qualified: bool,
    pub coreml_auxiliary_qualified: bool,
    pub sparse_retention_qualified: bool,
    pub context_refresh_qualified: bool,
    pub zero_copy_weight_residency_qualified: bool,
    pub soak_profile_qualified: bool,
    pub qualification_status: ManifestQualificationStatus,
}
