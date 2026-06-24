// ── Prism LLM — CImage LLM Capability Manifest ─────────────────────────
//
// Declares the LLM-generation capability of an installed CImage artifact.
// Prism uses the manifest for admission — it must not infer capability from
// a generic CImage container or checkpoint filename alone.

use crate::image::types::ArtifactDigest;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Shared qualification types (serde-enabled for manifest persistence) ─

/// Whether a model component is available and qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentAvailability {
    Absent,
    PresentUnverified,
    PresentQualified,
    Unsupported,
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

/// Qualification status of an artifact or provider route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationStatus {
    Accepted,
    Unqualified,
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

// ── Model family ──────────────────────────────────────────────────────

/// Identifies the LLM model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmModelFamily {
    Qwen2,
    Qwen3,
    Qwen3_5,
    Gemma4,
    Mistral,
    Llama,
    Custom,
}

impl fmt::Display for LlmModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen2 => write!(f, "qwen2"),
            Self::Qwen3 => write!(f, "qwen3"),
            Self::Qwen3_5 => write!(f, "qwen3.5"),
            Self::Gemma4 => write!(f, "gemma-4"),
            Self::Mistral => write!(f, "mistral"),
            Self::Llama => write!(f, "llama"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// ── KV cache data types ───────────────────────────────────────────────

/// Data type used for KV-cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvDtype {
    Fp16,
    Bf16,
    Fp8,
    Int8,
}

impl fmt::Display for KvDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp16 => write!(f, "fp16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Int8 => write!(f, "int8"),
        }
    }
}

/// Rotary position embedding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopeMode {
    Standard,
    Mrope,
    LinearScaling,
    YaRNScaling,
    Custom,
}

impl fmt::Display for RopeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::Mrope => write!(f, "mrope"),
            Self::LinearScaling => write!(f, "linear-scaling"),
            Self::YaRNScaling => write!(f, "yarn-scaling"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// ── Auxiliary islands ─────────────────────────────────────────────────

/// Function performed by an auxiliary inference island.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuxiliaryIslandFunction {
    TokenClassification,
    ConfidenceScoring,
    RoutingSignal,
    RetrievalScoring,
    SessionDiagnostics,
    Custom(String),
}

impl fmt::Display for AuxiliaryIslandFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenClassification => write!(f, "token-classification"),
            Self::ConfidenceScoring => write!(f, "confidence-scoring"),
            Self::RoutingSignal => write!(f, "routing-signal"),
            Self::RetrievalScoring => write!(f, "retrieval-scoring"),
            Self::SessionDiagnostics => write!(f, "session-diagnostics"),
            Self::Custom(label) => write!(f, "custom({label})"),
        }
    }
}

/// Which KV-cache epochs this auxiliary island can consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvEpochCompatibility {
    /// Compatible with any sealed epoch.
    AnySealedEpoch,
    /// Only epochs whose digest appears in this allowlist.
    EpochDigestAllowlist(Vec<ArtifactDigest>),
}

impl fmt::Display for KvEpochCompatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnySealedEpoch => write!(f, "any-sealed-epoch"),
            Self::EpochDigestAllowlist(list) => {
                write!(f, "epoch-digest-allowlist({} entries)", list.len())
            }
        }
    }
}

/// Execution policy for Core ML / ANE inference islands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreMlExecutionPolicy {
    /// Prefer ANE, fall back to GPU/CPU if unavailable.
    Preferred,
    /// Require ANE execution; fail if unavailable.
    RequireAne,
    /// Execute on CPU and GPU only (excludes ANE).
    CpuAndGpu,
    /// CPU-only execution.
    CpuOnly,
}

impl fmt::Display for CoreMlExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preferred => write!(f, "preferred"),
            Self::RequireAne => write!(f, "require-ane"),
            Self::CpuAndGpu => write!(f, "cpu-and-gpu"),
            Self::CpuOnly => write!(f, "cpu-only"),
        }
    }
}

// ── Inference phase & execution lane ──────────────────────────────────

/// Step within the LLM inference lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferencePhase {
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

impl fmt::Display for InferencePhase {
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
pub enum ExecutionLane {
    CpuControlPlane,
    Accelerate,
    Metal,
    CoreMlAne,
    UnifiedMemoryIsland,
}

impl fmt::Display for ExecutionLane {
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

// ── Transfer kinds ────────────────────────────────────────────────────

/// How tensor data moves between lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferKind {
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

impl fmt::Display for TransferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCopyRetained => write!(f, "zero-copy-retained"),
            Self::SharedMemoryMapped => write!(f, "shared-memory-mapped"),
            Self::ExplicitCopy => write!(f, "explicit-copy"),
            Self::CpuReadback => write!(f, "cpu-readback"),
            Self::TensorLayoutConversion => write!(f, "tensor-layout-conversion"),
            Self::DtypeConversion => write!(f, "dtype-conversion"),
            Self::ProviderOpaqueMaterialization => write!(f, "provider-opaque-materialization"),
            Self::Serialization => write!(f, "serialization"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Why a materialization event occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterializationReason {
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

impl fmt::Display for MaterializationReason {
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

// ── Value identifiers ─────────────────────────────────────────────────

/// Monotonically increasing materialization event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaterializationEventId(pub u64);

/// Unique session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub uuid::Uuid);

/// Identifier for a single allocation within an execution island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IslandAllocationId(pub u64);

// ── Tensor metadata ───────────────────────────────────────────────────

/// Identity of a tensor within a materialization event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIdentity {
    /// Provider-level weight key, if the tensor is a loaded weight.
    pub weight_key: Option<String>,
    /// Role the tensor plays (e.g. "query", "key", "value", "output").
    pub tensor_role: Option<String>,
    /// Allocation slot within the execution island.
    pub allocation_id: IslandAllocationId,
}

/// Shape, dtype, and layout of a tensor at a materialization boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRepresentation {
    /// Data type string (e.g. "fp16", "bf16", "int8").
    pub dtype: String,
    /// Shape dimensions.
    pub shape: Vec<u64>,
    /// Layout string (e.g. "nchw", "nhwc", "blocked-2x4").
    pub layout: String,
}

/// Records a single materialization event between execution lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationEvent {
    /// Unique event identifier.
    pub event_id: MaterializationEventId,
    /// Session in which this event occurred.
    pub session_id: SessionId,
    /// Lane that produced the data.
    pub source_lane: ExecutionLane,
    /// Lane that consumed the data.
    pub destination_lane: ExecutionLane,
    /// Island allocation identifier.
    pub allocation_id: IslandAllocationId,
    /// Identity of the tensor being transferred.
    pub tensor_identity: TensorIdentity,
    /// Method of transfer.
    pub transfer_kind: TransferKind,
    /// Number of bytes transferred, if known.
    pub byte_count: Option<u64>,
    /// Representation at the source lane boundary.
    pub source_representation: TensorRepresentation,
    /// Representation at the destination lane boundary.
    pub destination_representation: TensorRepresentation,
    /// Why the materialization was performed.
    pub reason: MaterializationReason,
    /// ISO 8601 timestamp of the event.
    pub timestamp: String,
}

// ── KV cache contract ─────────────────────────────────────────────────

/// Declares the KV-cache contract for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheContract {
    /// Number of transformer layers.
    pub layer_count: u32,
    /// Number of attention heads.
    pub attention_head_count: u32,
    /// Number of key-value heads (may differ for GQA/MQA).
    pub kv_head_count: u32,
    /// Dimension of each attention head.
    pub head_dimension: u32,
    /// Data type of cache entries.
    pub dtype: KvDtype,
    /// Rotary position embedding mode.
    pub rope_mode: RopeMode,
    /// Whether sparse retention is supported.
    pub supports_sparse_retention: bool,
    /// Whether context refresh (partial re-prefill) is supported.
    pub supports_context_refresh: bool,
    /// Whether position renumbering is supported.
    pub supports_position_renumbering: bool,
    /// Maximum number of context tokens declared by the model.
    pub max_declared_context_tokens: u32,
    /// Number of tokens per KV cache page.
    pub page_token_capacity: u32,
}

// ── Auxiliary island manifest ─────────────────────────────────────────

/// Describes an auxiliary inference island attached to an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryIslandManifest {
    /// Provider-unique identifier for this island.
    pub island_id: String,
    /// Function this auxiliary island performs.
    pub function: AuxiliaryIslandFunction,
    /// Digest of the island's compiled artifact.
    pub artifact_digest: ArtifactDigest,
    /// Digest of the input contract used by this island.
    pub input_contract_digest: ArtifactDigest,
    /// Digest of the output contract produced by this island.
    pub output_contract_digest: ArtifactDigest,
    /// Qualification status of this island.
    pub qualification_status: QualificationStatus,
    /// KV-cache epochs this island can consume.
    pub allowed_source_epochs: Vec<KvEpochCompatibility>,
    /// Execution policy for Core ML / ANE.
    pub execution_policy: CoreMlExecutionPolicy,
}

// ── Context profile ───────────────────────────────────────────────────

/// A supported inference context profile for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProfile {
    /// Unique profile identifier (e.g. "default", "long-context", "low-mem").
    pub id: String,
    /// Maximum prompt tokens allowed.
    pub max_prompt_tokens: u32,
    /// Maximum newly generated tokens.
    pub max_new_tokens: u32,
    /// Number of tokens per KV cache page for this profile.
    pub kv_page_capacity_tokens: u32,
    /// Token count at which KV compression is triggered, if configured.
    pub compression_threshold_tokens: Option<u32>,
    /// Token count at which context refresh is triggered, if configured.
    pub refresh_threshold_tokens: Option<u32>,
    /// Memory reservation in bytes for this profile.
    pub memory_reservation_bytes: u64,
}

// ── Provider artifact ─────────────────────────────────────────────────

/// Links a provider route to its artifact identity and hardware requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderArtifact {
    /// Provider name (e.g. "coreml", "mlx", "lut").
    pub provider: String,
    /// Provider-specific artifact identifier.
    pub artifact_id: String,
    /// Identifier of the compiler that produced the artifact.
    pub compiler_id: String,
    /// Minimum ABI version required by this artifact.
    pub abi_version: u32,
    /// Hardware features required by this artifact.
    pub required_hardware: Vec<String>,
    /// Expected tensor layout format.
    pub tensor_layout: String,
}

// ── Residency requirements ────────────────────────────────────────────

/// Memory residency requirements for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyRequirements {
    /// Minimum usable unified memory bytes required.
    pub min_unified_memory_bytes: u64,
    /// Persistent weight storage in bytes.
    pub persistent_weight_bytes: u64,
    /// Scratch memory in bytes needed at peak.
    pub scratch_bytes: u64,
    /// KV-cache reservation per token in bytes.
    pub kv_reservation_per_token: u64,
}

// ── Qualification record ──────────────────────────────────────────────

/// Qualification evidence for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmQualificationRecord {
    /// Qualification result.
    pub status: QualificationStatus,
    /// Identifier of the fixture used for qualification.
    pub fixture_id: String,
    /// ISO 8601 timestamp of the qualification run.
    pub verified_at: String,
    /// Human-readable failure reason, if declined.
    pub failure_reason: Option<String>,
}

// ── Capability manifest ───────────────────────────────────────────────

/// The authoritative LLM-generation capability declaration for a CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCapabilityManifest {
    /// Manifest schema version.  Bump on incompatible changes.
    pub schema_version: u32,
    /// LLM model family.
    pub model_family: LlmModelFamily,
    /// Tokenizer component availability.
    pub tokenizer: ComponentAvailability,
    /// Embedding component availability.
    pub embedding: ComponentAvailability,
    /// Transformer block component availability.
    pub transformer_blocks: ComponentAvailability,
    /// LM head component availability.
    pub lm_head: ComponentAvailability,
    /// KV-cache contract for the model.
    pub kv_cache_contract: KvCacheContract,
    /// Supported context profiles.
    pub supported_context_profiles: Vec<ContextProfile>,
    /// Provider-specific artifact bindings.
    pub provider_artifacts: Vec<LlmProviderArtifact>,
    /// Auxiliary inference islands.
    pub auxiliary_islands: Vec<AuxiliaryIslandManifest>,
    /// Memory residency requirements.
    pub residency_requirements: ResidencyRequirements,
    /// Overall qualification record.
    pub qualification: LlmQualificationRecord,
}
