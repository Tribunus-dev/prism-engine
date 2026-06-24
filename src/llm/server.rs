// ── Prism LLM Inference — Server Core Types ──────────────────────────────
//
// Foundation types for the Prism LLM inference server: session lifecycle,
// KV-cache epochs, lane dispatch, island allocation, and end-to-end
// inference receipts.

use super::manifest::{
    ExecutionLane, InferencePhase, IslandAllocationId, MaterializationEvent,
    QualificationStatus, SessionId,
};
use crate::image::types::ArtifactDigest;
use serde::{Deserialize, Serialize};

// ── Identifiers ──────────────────────────────────────────────────────

/// Opaque request identifier (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub uuid::Uuid);

/// Opaque dispatch identifier for a lane execution unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DispatchId(pub u64);

/// Identifier for a KV-cache epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvEpochId(pub u64);

/// Identifier for a single KV-cache page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvPageId(pub u64);

/// Identifier for a completion fence synchronising lane dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompletionFenceId(pub u64);

/// Identifier for a compiled CImage artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CImageId(pub String);

/// Identifier for a context profile configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContextProfileId(pub String);

/// Key identifying a unique weight residency on device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WeightResidencyKey {
    pub cimage_digest: ArtifactDigest,
    pub tensor_manifest_digest: ArtifactDigest,
    pub provider_kind: String,
    pub dtype_profile: String,
}

/// Opaque signature identifying a device's capability profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceCapabilitySignature(pub String);

/// Identifier for a multi-island inference receipt.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReceptionId(pub uuid::Uuid);

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
    pub session_id: SessionId,
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
    pub session_id: SessionId,
    pub request_id: RequestId,
}

/// Describes an allocation of unified memory for a specific island.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IslandAllocation {
    pub allocation_id: IslandAllocationId,
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
    pub lane: ExecutionLane,
    pub phase: InferencePhase,
    pub input_allocations: Vec<IslandAllocationId>,
    pub output_allocations: Vec<IslandAllocationId>,
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
    pub allocation_id: IslandAllocationId,
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
    pub materialization_events: Vec<MaterializationEvent>,
    pub eviction_status: WeightEvictionStatus,
}

/// Receipt for a Metal compute dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalExecutionReceipt {
    pub dispatch_id: DispatchId,
    pub phase: InferencePhase,
    pub kv_epoch: Option<KvEpochId>,
    pub command_submission_time: String,
    pub completion_time: String,
    pub input_allocation_ids: Vec<IslandAllocationId>,
    pub output_allocation_ids: Vec<IslandAllocationId>,
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
    pub qualification_status: QualificationStatus,
    pub input_contract_verified: bool,
    pub output_contract_verified: bool,
    pub provider_opaque_materialization: bool,
}

/// Receipt for a single lane execution (Metal, Accelerate, or CoreML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneExecutionReceipt {
    pub lane: ExecutionLane,
    pub metal: Option<MetalExecutionReceipt>,
    pub accelerate: Option<AccelerateExecutionReceipt>,
    pub coreml: Option<CoreMlAuxiliaryReceipt>,
}

/// Receipt produced when an inference session is cancelled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCancelledReceipt {
    pub session_id: SessionId,
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
    pub phase: InferencePhase,
    pub lane: Option<ExecutionLane>,
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
    pub session_id: SessionId,
    pub request_id: RequestId,
    pub terminal_state: InferenceTerminalState,
    pub cimage_digest: ArtifactDigest,
    pub context_profile: ContextProfileId,
    pub admission: InferenceAdmissionReceipt,
    pub weight_residency: WeightResidencyReceipt,
    pub lane_receipts: Vec<LaneExecutionReceipt>,
    pub kv_history: Vec<KvEpochReceipt>,
    pub materialization_events: Vec<MaterializationEvent>,
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
    pub qualification_status: QualificationStatus,
}
