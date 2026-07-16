// ── Manifest type stubs ──────────────────────────────────────────────
//
// Re-exports / stubs for LLM manifest types originally from
// `crate::llm::manifest`. These should eventually be ported to
// `prism-ecs-ir::manifest`.
//
// # TODO: import from prism-ecs-ir::manifest when ported

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Shared qualification types ───────────────────────────────────────

/// Whether a model component is available and qualified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentAvailability {
    NotPresent,
    PresentNotQualified,
    PresentQualified,
}

impl fmt::Display for ComponentAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPresent => write!(f, "not present"),
            Self::PresentNotQualified => write!(f, "present, not qualified"),
            Self::PresentQualified => write!(f, "present, qualified"),
        }
    }
}

/// Qualification status of an artifact or provider route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationStatus {
    Accepted,
    Rejected(String),
    RequiresReview,
}

impl fmt::Display for QualificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => write!(f, "accepted"),
            Self::Rejected(reason) => write!(f, "rejected: {reason}"),
            Self::RequiresReview => write!(f, "requires review"),
        }
    }
}

// ── Model family ─────────────────────────────────────────────────────

/// Identifies the LLM model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmModelFamily {
    Llama,
    Qwen3,
    Deepseek,
    Gemma4,
    Mistral,
    Phi,
    Custom(u64),
}

impl fmt::Display for LlmModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llama => write!(f, "Llama"),
            Self::Qwen3 => write!(f, "Qwen-3"),
            Self::Deepseek => write!(f, "DeepSeek"),
            Self::Gemma4 => write!(f, "Gemma-4"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Phi => write!(f, "Phi"),
            Self::Custom(id) => write!(f, "custom-{id}"),
        }
    }
}

// ── KV cache data types ──────────────────────────────────────────────

/// Data type used for KV-cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvDtype {
    Fp16,
    Fp8,
    Int8,
    Nf4,
}

impl fmt::Display for KvDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp16 => write!(f, "fp16"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Int8 => write!(f, "int8"),
            Self::Nf4 => write!(f, "nf4"),
        }
    }
}

/// Rotary position embedding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopeMode {
    Standard,
    None,
    #[allow(dead_code)]
    Interleaved,
    #[allow(dead_code)]
    Su,
    #[allow(dead_code)]
    YaRn,
}

impl fmt::Display for RopeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::None => write!(f, "none"),
            Self::Interleaved => write!(f, "interleaved"),
            Self::Su => write!(f, "su"),
            Self::YaRn => write!(f, "yarn"),
        }
    }
}

// ── Auxiliary islands ────────────────────────────────────────────────

/// Function performed by an auxiliary inference island.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuxiliaryIslandFunction {
    VisionEncoder,
    AudioEncoder,
    CrossAttention,
    Classifier,
    MoERouter,
    Custom(String),
}

impl fmt::Display for AuxiliaryIslandFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VisionEncoder => write!(f, "vision encoder"),
            Self::AudioEncoder => write!(f, "audio encoder"),
            Self::CrossAttention => write!(f, "cross-attention"),
            Self::Classifier => write!(f, "classifier"),
            Self::MoERouter => write!(f, "MoE router"),
            Self::Custom(s) => write!(f, "custom: {s}"),
        }
    }
}

/// Which KV-cache epochs this auxiliary island can consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvEpochCompatibility {
    CurrentOnly,
    Any,
    Specific(u64),
}

impl fmt::Display for KvEpochCompatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CurrentOnly => write!(f, "current only"),
            Self::Any => write!(f, "any"),
            Self::Specific(id) => write!(f, "epoch {id}"),
        }
    }
}

/// Execution policy for Core ML / ANE inference islands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreMlExecutionPolicy {
    Required,
    Optional,
    Disabled,
    FallbackToAccelerate,
    FallbackToMetal,
}

impl fmt::Display for CoreMlExecutionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Required => write!(f, "required"),
            Self::Optional => write!(f, "optional"),
            Self::Disabled => write!(f, "disabled"),
            Self::FallbackToAccelerate => write!(f, "fallback to accelerate"),
            Self::FallbackToMetal => write!(f, "fallback to metal"),
        }
    }
}

// ── Inference phase & execution lane ─────────────────────────────────

/// Step within the LLM inference lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferencePhase {
    PromptPrefill,
    Decode,
    AuxiliaryInference,
    Embedding,
    PostProcessing,
}

impl fmt::Display for InferencePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PromptPrefill => write!(f, "prompt prefill"),
            Self::Decode => write!(f, "decode"),
            Self::AuxiliaryInference => write!(f, "auxiliary inference"),
            Self::Embedding => write!(f, "embedding"),
            Self::PostProcessing => write!(f, "post-processing"),
        }
    }
}

/// Execution lane that processes tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionLane {
    Metal,
    Accelerate,
    CoreMlAne,
}

impl fmt::Display for ExecutionLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal => write!(f, "Metal"),
            Self::Accelerate => write!(f, "Accelerate"),
            Self::CoreMlAne => write!(f, "Core ML / ANE"),
        }
    }
}

// ── Transfer kinds ───────────────────────────────────────────────────

/// How tensor data moves between lanes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferKind {
    UnifiedMemoryShare,
    PcieCopy,
    AxiStream,
    RingBus,
    Custom(String),
}

impl fmt::Display for TransferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnifiedMemoryShare => write!(f, "unified memory share"),
            Self::PcieCopy => write!(f, "PCIe copy"),
            Self::AxiStream => write!(f, "AXI stream"),
            Self::RingBus => write!(f, "ring bus"),
            Self::Custom(s) => write!(f, "custom: {s}"),
        }
    }
}

/// Why a materialization event occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterializationReason {
    LaneDispatch,
    SparseRetention,
    ContextRefresh,
    ProviderPrefetch,
    WeightReload,
    ModelWarm,
}

impl fmt::Display for MaterializationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LaneDispatch => write!(f, "lane dispatch"),
            Self::SparseRetention => write!(f, "sparse retention"),
            Self::ContextRefresh => write!(f, "context refresh"),
            Self::ProviderPrefetch => write!(f, "provider prefetch"),
            Self::WeightReload => write!(f, "weight reload"),
            Self::ModelWarm => write!(f, "model warm"),
        }
    }
}

// ── Value identifiers ────────────────────────────────────────────────

/// Monotonically increasing materialization event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaterializationEventId(pub u64);

/// Unique session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub uuid::Uuid);

/// Identifier for a single allocation within an execution island.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IslandAllocationId(pub u64);

// ── Tensor metadata ──────────────────────────────────────────────────

/// Identity of a tensor within a materialization event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorIdentity {
    pub name: String,
}

/// Shape, dtype, and layout of a tensor at a materialization boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRepresentation {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub layout: String,
}

/// Records a single materialization event between execution lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationEvent {
    pub event_id: MaterializationEventId,
    pub reason: MaterializationReason,
    pub transfer_kind: TransferKind,
    pub source_lane: ExecutionLane,
    pub target_lane: ExecutionLane,
    pub tensors: Vec<TensorIdentity>,
    pub representation: Option<TensorRepresentation>,
    pub timestamp: String,
}

// ── KV cache contract ────────────────────────────────────────────────

/// Declares the KV-cache contract for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheContract {
    pub layer_count: u32,
    pub attention_head_count: u32,
    pub kv_head_count: u32,
    pub head_dimension: u32,
    pub dtype: KvDtype,
    pub rope_mode: RopeMode,
    pub supports_sparse_retention: bool,
    pub supports_context_refresh: bool,
    pub supports_position_renumbering: bool,
    pub max_declared_context_tokens: u32,
    pub page_token_capacity: u32,
}

// ── Auxiliary island manifest ────────────────────────────────────────

/// Describes an auxiliary inference island attached to an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryIslandManifest {
    pub island_id: String,
    pub function: AuxiliaryIslandFunction,
    pub kv_compatibility: KvEpochCompatibility,
    pub execution_policy: CoreMlExecutionPolicy,
    pub memory_requirement_bytes: u64,
    pub input_schema: Vec<TensorRepresentation>,
    pub output_schema: Vec<TensorRepresentation>,
}

// ── Context profile ──────────────────────────────────────────────────

/// A supported inference context profile for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProfile {
    pub name: String,
    pub max_tokens: u32,
    pub supports_sparse_retention: bool,
    pub supports_context_refresh: bool,
}

// ── Provider artifact ────────────────────────────────────────────────

/// Links a provider route to its artifact identity and hardware requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmProviderArtifact {
    pub provider_kind: String,
    pub artifact_digest: String,
    pub memory_requirement_bytes: u64,
}

// ── Residency requirements ───────────────────────────────────────────

/// Memory residency requirements for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidencyRequirements {
    pub min_unified_memory_bytes: u64,
    pub persistent_weight_bytes: u64,
    pub scratch_bytes: u64,
    pub kv_reservation_per_token: u32,
}

// ── Qualification record ─────────────────────────────────────────────

/// Qualification evidence for an LLM CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmQualificationRecord {
    pub status: QualificationStatus,
    pub fixture_id: String,
    pub verified_at: String,
    pub failure_reason: Option<String>,
}

// ── Capability manifest ──────────────────────────────────────────────

/// The authoritative LLM-generation capability declaration for a CImage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCapabilityManifest {
    pub schema_version: u32,
    pub model_family: LlmModelFamily,
    pub tokenizer: ComponentAvailability,
    pub embedding: ComponentAvailability,
    pub transformer_blocks: ComponentAvailability,
    pub lm_head: ComponentAvailability,
    pub kv_cache_contract: KvCacheContract,
    pub supported_context_profiles: Vec<ContextProfile>,
    pub provider_artifacts: Vec<LlmProviderArtifact>,
    pub auxiliary_islands: Vec<AuxiliaryIslandManifest>,
    pub residency_requirements: ResidencyRequirements,
    pub qualification: LlmQualificationRecord,
}
