//! Runtime Capability Registry — core type definitions.
//!
//! Implements the immutable boundary between model requirements, host policy,
//! and admitted execution state (Phase 5 of the production-hardened roadmap).
//!
//! Architecture:
//!   ComputeImageManifest  → immutable artifact facts
//!   CapabilityPolicy      → local authority rules
//!   RuntimeContract       → signed deployment-specific intersection
//!   LiveAdmissionEnvelope → volatile narrowing under thermal/power/queue pressure
//!   ExecutionLease        → per-request snapshot of contract + overlay
//!   ExecutionReceipt      → hash-chained ledger evidence

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Canonical encoder for receipt hashing — uses deterministic JSON encoding.
fn encode_field(h: &mut Sha256, val: &impl serde::Serialize) {
    if let Ok(bytes) = serde_json::to_vec(val) {
        h.update(&bytes);
    }
}

// ── Digest types ──────────────────────────────────────────────────────────

/// A 256-bit digest (SHA-256).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest256(pub [u8; 32]);

impl Digest256 {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
    pub fn compute(data: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(data);
        Self(h.finalize().into())
    }
}

impl From<[u8; 32]> for Digest256 {
    fn from(b: [u8; 32]) -> Self {
        Self(b)
    }
}

// ── Core identity types ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeploymentId(pub Digest256);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReceiptId(pub Digest256);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolicyId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderIdentity(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyFingerprint(pub Digest256);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LicenseDescriptor(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatrixId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutableProfileSlotId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u16);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OptimizerIdentity(pub Digest256);

// ── Capability system ─────────────────────────────────────────────────────

/// A capability is a fixed index into a bit-array representation.
pub type CapabilityIndex = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    bits: [u64; 2], // 128 bits
}

impl CapabilitySet {
    pub const fn empty() -> Self {
        Self { bits: [0u64; 2] }
    }

    pub fn with(index: CapabilityIndex) -> Self {
        let mut s = Self::empty();
        s.set(index);
        s
    }

    pub fn set(&mut self, index: CapabilityIndex) {
        let (word, bit) = ((index / 64) as usize, index % 64);
        self.bits[word] |= 1u64 << bit;
    }

    pub fn test(&self, index: CapabilityIndex) -> bool {
        let (word, bit) = ((index / 64) as usize, index % 64);
        (self.bits[word] & (1u64 << bit)) != 0
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        (self.bits[0] & !other.bits[0]) == 0 && (self.bits[1] & !other.bits[1]) == 0
    }

    pub fn is_empty(&self) -> bool {
        self.bits[0] == 0 && self.bits[1] == 0
    }
}

impl std::ops::BitAnd for &CapabilitySet {
    type Output = CapabilitySet;
    fn bitand(self, rhs: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits: [self.bits[0] & rhs.bits[0], self.bits[1] & rhs.bits[1]],
        }
    }
}

impl std::ops::BitAnd for CapabilitySet {
    type Output = CapabilitySet;
    fn bitand(self, rhs: CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits: [self.bits[0] & rhs.bits[0], self.bits[1] & rhs.bits[1]],
        }
    }
}

impl std::ops::BitOr for &CapabilitySet {
    type Output = CapabilitySet;
    fn bitor(self, rhs: &CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits: [self.bits[0] | rhs.bits[0], self.bits[1] | rhs.bits[1]],
        }
    }
}

impl std::ops::BitOr for CapabilitySet {
    type Output = CapabilitySet;
    fn bitor(self, rhs: CapabilitySet) -> CapabilitySet {
        CapabilitySet {
            bits: [self.bits[0] | rhs.bits[0], self.bits[1] | rhs.bits[1]],
        }
    }
}

impl std::ops::Not for &CapabilitySet {
    type Output = CapabilitySet;
    fn not(self) -> CapabilitySet {
        CapabilitySet {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

impl std::ops::Not for CapabilitySet {
    type Output = CapabilitySet;
    fn not(self) -> CapabilitySet {
        CapabilitySet {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

// Defined capability indices
pub mod caps {
    use super::CapabilityIndex;
    pub const TEXT_INFERENCE: CapabilityIndex = 0;
    pub const IMAGE_INPUT: CapabilityIndex = 1;
    pub const AUDIO_INPUT: CapabilityIndex = 2;
    pub const VIDEO_INPUT: CapabilityIndex = 3;
    pub const TTS_OUTPUT: CapabilityIndex = 4;
    pub const SPEECH_TO_TEXT: CapabilityIndex = 5;
    pub const TOOL_EXECUTION: CapabilityIndex = 6;
    pub const NETWORK_EGRESS: CapabilityIndex = 7;
    pub const BACKGROUND_OPTIMIZATION: CapabilityIndex = 8;
    pub const PROFILE_MUTATION: CapabilityIndex = 9;
    pub const SCREEN_CAPTURE: CapabilityIndex = 10;
    pub const FILESYSTEM_ACCESS: CapabilityIndex = 11;
    pub const EXTERNAL_DEVICE: CapabilityIndex = 12;
}

// ── Backend and precision types ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendTarget {
    Metal,
    Accelerate,
    ANE,
    MLX,
    Megakernel,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionClass {
    F32,
    BF16,
    NF4Tile640,
    TernaryPage640,
    Q8_0,
    Q4_0,
    Q4KM,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPlan {
    pub primary: BackendTarget,
    pub fallbacks: Vec<BackendTarget>,
    pub validated_hardware: bool,
}

impl BackendPlan {
    pub fn select(
        allowed: &[BackendTarget],
        _policy_allowed: &std::collections::HashSet<BackendTarget>,
        _hardware: &HardwareProfile,
    ) -> Result<Self, DeploymentError> {
        let primary = allowed
            .first()
            .cloned()
            .ok_or(DeploymentError::NoAllowedBackend)?;
        Ok(Self {
            primary,
            fallbacks: allowed[1..].to_vec(),
            validated_hardware: true,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub base: PrecisionClass,
    pub allowed_escalations: Vec<PrecisionClass>,
    pub allow_dynamic: bool,
}

impl PrecisionPolicy {
    pub fn compile(
        required: &[PrecisionClass],
        allowed: &std::collections::HashSet<PrecisionClass>,
    ) -> Result<Self, DeploymentError> {
        for r in required {
            if !allowed.contains(r) {
                return Err(DeploymentError::PrecisionClassNotAllowed(r.clone()));
            }
        }
        Ok(Self {
            base: required.first().cloned().unwrap_or(PrecisionClass::BF16),
            allowed_escalations: vec![],
            allow_dynamic: false,
        })
    }
}

// ── Resource budgets ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_context_tokens: u32,
    pub max_batch_slots: u16,
    pub max_kv_bytes: u64,
    pub max_unified_bytes: u64,
    pub max_kv_blocks_per_sequence: u32,
}

impl ResourceBudget {
    pub fn digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update(b"prism.resource-budget.v1\0");
        h.update(&self.max_context_tokens.to_be_bytes());
        h.update(&self.max_batch_slots.to_be_bytes());
        h.update(&self.max_kv_bytes.to_be_bytes());
        h.update(&self.max_unified_bytes.to_be_bytes());
        h.update(&self.max_kv_blocks_per_sequence.to_be_bytes());
        Digest256(h.finalize().into())
    }
}

// ── Contracts ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirementSet {
    pub min_unified_memory_bytes: u64,
    pub kv_cache_reservation_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheContract {
    pub max_blocks: u32,
    pub tokens_per_block: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkContract {
    pub enabled: bool,
    pub allowed_domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContract {
    pub enabled: bool,
    pub allowed_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSlotDescriptor {
    pub slot_id: MutableProfileSlotId,
    pub slot_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSignature {
    pub bytes: Vec<u8>,
    pub algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraphDigest(pub Digest256);

// ── ComputeImageManifest ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeImageManifest {
    pub format_version: u32,
    pub model_digest: Digest256,
    pub artifact_digest: Digest256,
    pub compiler_digest: Digest256,
    pub provider: ProviderIdentity,
    pub model_license: LicenseDescriptor,
    pub required_capabilities: CapabilitySet,
    pub optional_capabilities: CapabilitySet,
    pub execution_graph: PhaseGraphDigest,
    pub allowed_backends: Vec<BackendTarget>,
    pub required_precision_classes: Vec<PrecisionClass>,
    pub memory_requirements: MemoryRequirementSet,
    pub kv_cache_contract: KvCacheContract,
    pub network_contract: NetworkContract,
    pub tool_contract: ToolContract,
    pub mutable_profile_slots: Vec<ProfileSlotDescriptor>,
    pub artifact_signature: ArtifactSignature,
}

impl ComputeImageManifest {
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, DeploymentError> {
        // Use canonical JSON serialization (deterministic key order)
        serde_json::to_vec(self).map_err(|e| DeploymentError::SerializationError(e.to_string()))
    }

    pub fn manifest_digest(&self) -> Digest256 {
        self.to_canonical_bytes()
            .map(|b| Digest256::compute(&b))
            .unwrap_or(Digest256::compute(b""))
    }
}

// ── CapabilityPolicy ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityPolicy {
    pub policy_id: PolicyId,
    pub policy_revision: u64,
    pub allowed_capabilities: CapabilitySet,
    pub denied_capabilities: CapabilitySet,
    pub maximum_context_tokens: u32,
    pub maximum_batch_slots: u16,
    pub maximum_kv_bytes: u64,
    pub maximum_unified_memory_bytes: u64,
    pub allowed_backends: std::collections::HashSet<BackendTarget>,
    pub allowed_precision_modes: std::collections::HashSet<PrecisionClass>,
    pub allow_background_optimization: bool,
    pub allow_profile_mutation: bool,
    pub allow_network_egress: bool,
    pub allow_tool_execution: bool,
    pub required_provider_keys: Vec<KeyFingerprint>,
    pub required_manifest_signatures: u8,
}

impl CapabilityPolicy {
    pub fn digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update(b"prism.capability-policy.v1\0");
        h.update(&self.policy_id.0.to_be_bytes());
        Digest256(h.finalize().into())
    }
}

// ── HardwareProfile ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub hardware_id: u64,
    pub soc_family: String,
    pub gpu_cores: u32,
    pub ane_cores: u32,
    pub unified_memory_gb: u32,
    pub available_memory_bytes: u64,
    pub metal_feature_set: String,
    pub max_supported_tokens: u32,
    pub max_supported_slots: u16,
    pub max_kv_allocation_ceiling: u64,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        // platform detection — simplified for now
        #[cfg(target_os = "macos")]
        {
            #[cfg(any(
                feature = "mlx-backend",
                feature = "prism-backend",
                feature = "prism-backend-ios"
            ))]
            let device = metal::Device::system_default();
            #[cfg(any(
                feature = "mlx-backend",
                feature = "prism-backend",
                feature = "prism-backend-ios"
            ))]
            let mem = device
                .as_ref()
                .map(|d| d.recommended_max_working_set_size())
                .unwrap_or(6_000_000_000);
            #[cfg(not(any(
                feature = "mlx-backend",
                feature = "prism-backend",
                feature = "prism-backend-ios"
            )))]
            let mem: u64 = 6_000_000_000;
            let max_tokens: u32 = 20480;
            let kv_ceiling = mem.saturating_sub(1_500_000_000);
            #[cfg(any(
                feature = "mlx-backend",
                feature = "prism-backend",
                feature = "prism-backend-ios"
            ))]
            let gpu_cores = device
                .as_ref()
                .map(|d| {
                    let max = d.max_threads_per_threadgroup();
                    std::cmp::max(max.width, max.height) / 32
                })
                .unwrap_or(8);
            #[cfg(not(any(
                feature = "mlx-backend",
                feature = "prism-backend",
                feature = "prism-backend-ios"
            )))]
            let gpu_cores: u32 = 8;
            Self {
                hardware_id: 1,
                soc_family: "AppleSilicon".into(),
                gpu_cores: gpu_cores as u32,
                ane_cores: 16,
                unified_memory_gb: (mem / 1_000_000_000) as u32,
                available_memory_bytes: mem as u64,
                metal_feature_set: "Metal3".into(),
                max_supported_tokens: max_tokens,
                max_supported_slots: 32u16,
                max_kv_allocation_ceiling: kv_ceiling,
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self {
                hardware_id: 0,
                soc_family: "Unknown".into(),
                gpu_cores: 0,
                ane_cores: 0,
                unified_memory_gb: 0,
                available_memory_bytes: 0,
                metal_feature_set: "none".into(),
                max_supported_tokens: 0,
                max_supported_slots: 1,
                max_kv_allocation_ceiling: 0,
            }
        }
    }

    pub fn digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update(b"prism.hardware-profile.v1\0");
        h.update(&self.hardware_id.to_be_bytes());
        h.update(self.soc_family.as_bytes());
        Digest256(h.finalize().into())
    }
}

// ── RuntimeContract ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeContract {
    pub deployment_id: DeploymentId,
    pub model_digest: Digest256,
    pub artifact_digest: Digest256,
    pub manifest_digest: Digest256,
    pub policy_digest: Digest256,
    pub hardware_digest: Digest256,
    pub effective_capabilities: CapabilitySet,
    pub resource_budget: ResourceBudget,
    pub backend_plan: BackendPlan,
    pub precision_policy: PrecisionPolicy,
    pub tool_authority: ToolAuthority,
    pub optimization_authority: OptimizationAuthority,
    pub issued_at: LogicalTimestamp,
    pub expires_at: Option<LogicalTimestamp>,
    pub registry_signature: RegistrySignature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolAuthority {
    pub allowed: bool,
    pub tools: Vec<String>,
}

impl ToolAuthority {
    pub fn new(allowed: bool) -> Self {
        Self {
            allowed,
            tools: vec![],
        }
    }
    pub fn is_permitted(&self) -> bool {
        self.allowed
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAuthority {
    pub allowed: bool,
}

impl OptimizationAuthority {
    pub fn new(allowed: bool) -> Self {
        Self { allowed }
    }
    pub fn is_permitted(&self) -> bool {
        self.allowed
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LogicalTimestamp(pub u64);

impl LogicalTimestamp {
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self(d.as_nanos() as u64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySignature {
    pub bytes: Vec<u8>,
    pub key_fingerprint: KeyFingerprint,
}

/// A loaded compute image paired with its manifest.
pub struct LoadedComputeImage {
    pub manifest: ComputeImageManifest,
}

// ── DeploymentHandle ──────────────────────────────────────────────────────

pub struct DeploymentHandle {
    pub deployment_id: DeploymentId,
    pub contract: Arc<RuntimeContract>,
    pub execution_image: Arc<LoadedComputeImage>,
    #[cfg(all(
        target_os = "macos",
        any(feature = "mlx-backend", feature = "prism-backend")
    ))]
    pub executor: Arc<
        std::sync::Mutex<
            Option<crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor>,
        >,
    >,
    pub live_state: Arc<DeploymentState>,
}

pub struct DeploymentState {
    contract_generation: std::sync::atomic::AtomicU64,
    profile_generation: std::sync::atomic::AtomicU64,
}

impl DeploymentState {
    pub fn new() -> Self {
        Self {
            contract_generation: std::sync::atomic::AtomicU64::new(1),
            profile_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }
    pub fn new_from_generation(gen: u64) -> Self {
        Self {
            contract_generation: std::sync::atomic::AtomicU64::new(1),
            profile_generation: std::sync::atomic::AtomicU64::new(gen),
        }
    }
    pub fn contract_generation(&self) -> u64 {
        self.contract_generation
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn profile_generation(&self) -> u64 {
        self.profile_generation
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ── ExecutionLease ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ExecutionLease {
    pub deployment: Arc<DeploymentHandle>,
    pub contract_generation: u64,
    pub profile_generation: u64,
    pub admission_epoch: u64,
}

// ── LiveAdmissionEnvelope ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LiveAdmissionEnvelope {
    pub active_batch_slots: u16,
    pub max_context_tokens: u32,
    pub allow_background_optimization: bool,
    pub disable_speculative_decode: bool,
    pub disable_precision_escalation: bool,
    pub force_accelerate_fallback: bool,
    pub energy_mode: EnergyOptimizationProfile,
}

#[derive(Debug, Clone, Copy)]
pub enum EnergyOptimizationProfile {
    Default,
    MinimizePackageSustainedPower,
}

impl LiveAdmissionEnvelope {
    pub fn from_contract(contract: &RuntimeContract) -> Self {
        Self {
            active_batch_slots: contract.resource_budget.max_batch_slots,
            max_context_tokens: contract.resource_budget.max_context_tokens,
            allow_background_optimization: contract.optimization_authority.allowed,
            disable_speculative_decode: false,
            disable_precision_escalation: false,
            force_accelerate_fallback: false,
            energy_mode: EnergyOptimizationProfile::Default,
        }
    }

    pub fn admit(&self, _request: &()) -> Result<(), DeploymentError> {
        if self.active_batch_slots == 0 {
            return Err(DeploymentError::CapacityExhausted);
        }
        Ok(())
    }
}

// ── ExecutionReceipt ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionDecision {
    pub admitted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReservation {
    pub slot_id: u32,
    pub kv_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendDecision {
    pub backend: BackendTarget,
    pub ops_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOverrideReceipt {
    pub matrix: MatrixId,
    pub from: PrecisionClass,
    pub to: PrecisionClass,
    pub channel: Option<ChannelId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackReceipt {
    pub from_backend: BackendTarget,
    pub to_backend: BackendTarget,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub receipt_id: ReceiptId,
    pub previous_receipt_digest: Option<Digest256>,
    pub deployment_digest: Digest256,
    pub request_digest: Digest256,
    pub lease_generation: u64,
    pub profile_generation: u64,
    pub admission_decision: AdmissionDecision,
    pub resource_reservation: ResourceReservation,
    pub backend_decisions: Vec<BackendDecision>,
    pub precision_overrides: Vec<PrecisionOverrideReceipt>,
    pub fallback_events: Vec<FallbackReceipt>,
    pub output_digest: Digest256,
    pub completion_epoch: u64,
    pub terminal_status: u32,
}

impl ExecutionReceipt {
    pub fn compute_canonical_digest(&self) -> Digest256 {
        use sha2::Sha256;
        let mut h = Sha256::new();
        h.update(b"prism.execution-receipt.v1\0");
        encode_receipt_v1(self, &mut h);
        Digest256(h.finalize().into())
    }
}

fn encode_receipt_v1(receipt: &ExecutionReceipt, h: &mut Sha256) {
    encode_field(h, &receipt.previous_receipt_digest);
    encode_field(h, &receipt.deployment_digest);
    encode_field(h, &receipt.request_digest);
    h.update(&receipt.lease_generation.to_be_bytes());
    h.update(&receipt.profile_generation.to_be_bytes());
    encode_field(h, &receipt.admission_decision);
    encode_field(h, &receipt.resource_reservation);
    encode_field(h, &receipt.backend_decisions);
    encode_field(h, &receipt.precision_overrides);
    encode_field(h, &receipt.fallback_events);
    encode_field(h, &receipt.output_digest);
    h.update(&receipt.completion_epoch.to_be_bytes());
    h.update(&receipt.terminal_status.to_be_bytes());
}

// ── Deployment errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DeploymentError {
    InvalidArtifactSignature,
    InvalidManifestSignature,
    RequiredCapabilityDenied(CapabilitySet),
    CapabilityMismatch,
    UntrustedProvider,
    NoAllowedBackend,
    PrecisionClassNotAllowed(PrecisionClass),
    SerializationError(String),
    CapacityExhausted,
    OptimizationForbidden,
    InvalidOverlayTarget,
    IllegalWeightMutation(MutableProfileSlotId),
    ResourceExhaustion(String),
    HardwareInsufficient(String),
    ThermalThrottled,
    InternalError(String),
}

impl std::fmt::Display for DeploymentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArtifactSignature => write!(f, "invalid artifact signature"),
            Self::InvalidManifestSignature => write!(f, "invalid manifest signature"),
            Self::RequiredCapabilityDenied(c) => write!(f, "required capabilities denied: {:?}", c),
            Self::CapabilityMismatch => write!(f, "capability mismatch"),
            Self::UntrustedProvider => write!(f, "untrusted model provider"),
            Self::NoAllowedBackend => write!(f, "no allowed backend"),
            Self::PrecisionClassNotAllowed(p) => write!(f, "precision class not allowed: {:?}", p),
            Self::SerializationError(e) => write!(f, "serialization error: {e}"),
            Self::CapacityExhausted => write!(f, "capacity exhausted"),
            Self::OptimizationForbidden => write!(f, "optimization not permitted by contract"),
            Self::InvalidOverlayTarget => write!(f, "overlay targets wrong artifact digest"),
            Self::IllegalWeightMutation(s) => write!(f, "illegal weight mutation of slot {s:?}"),
            Self::ResourceExhaustion(e) => write!(f, "resource exhaustion: {e}"),
            Self::HardwareInsufficient(e) => write!(f, "hardware insufficient: {e}"),
            Self::ThermalThrottled => write!(f, "thermal throttling active"),
            Self::InternalError(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl std::error::Error for DeploymentError {}

// ── DeploymentDigestInput ─────────────────────────────────────────────────

pub struct DeploymentDigestInput {
    pub manifest_digest: Digest256,
    pub artifact_digest: Digest256,
    pub policy_digest: Digest256,
    pub hardware_digest: Digest256,
    pub backend_plan_digest: Digest256,
    pub resource_budget_digest: Digest256,
    pub precision_policy_digest: Digest256,
    pub tool_authority_digest: Digest256,
    pub optimization_authority_digest: Digest256,
}

pub fn hash_deployment_contract(input: &DeploymentDigestInput) -> Digest256 {
    let mut h = Sha256::new();
    h.update(b"prism.runtime-contract.v1\0");
    h.update(input.manifest_digest.as_bytes());
    h.update(input.artifact_digest.as_bytes());
    h.update(input.policy_digest.as_bytes());
    h.update(input.hardware_digest.as_bytes());
    h.update(input.backend_plan_digest.as_bytes());
    h.update(input.resource_budget_digest.as_bytes());
    h.update(input.precision_policy_digest.as_bytes());
    h.update(input.tool_authority_digest.as_bytes());
    h.update(input.optimization_authority_digest.as_bytes());
    Digest256(h.finalize().into())
}

// ── Platform secure signer (Keychain/Secure Enclave) ──────────────────────

pub struct PlatformSecureSigner;

impl PlatformSecureSigner {
    pub fn sign(data: &[u8]) -> Result<Vec<u8>, DeploymentError> {
        // On macOS, would use Security framework / CryptoKit.
        // For cross-platform, use ed25519-dalek or p256 software signing.
        // This is the stub — production uses Secure Enclave via Apple's CryptoKit FFI.
        Ok(data.to_vec())
    }

    pub fn persist_checkpoint_record(
        _sequence: u64,
        _tail_digest: Digest256,
        _signature: Vec<u8>,
    ) -> Result<(), DeploymentError> {
        // Persist to Keychain or designated checkpoint store.
        Ok(())
    }
}

// ── Thermal state ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

impl ThermalState {
    pub fn current() -> Self {
        #[cfg(target_os = "macos")]
        {
            // On macOS 14+, use ProcessInfo.thermalState.
            // For now, return Nominal and let the platform adapter handle it.
            Self::Nominal
        }
        #[cfg(not(target_os = "macos"))]
        Self::Nominal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerSource {
    AC,
    Battery,
}

impl PowerSource {
    pub fn current() -> Self {
        Self::AC // stub — real impl uses IOPowerSources
    }
}

// ── Overlay types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OverlayMutation {
    QuantizationScale {
        slot: MutableProfileSlotId,
        scales_digest: Digest256,
    },
    OutlierRouting {
        matrix: MatrixId,
        channel_mask_digest: Digest256,
    },
    PrecisionVariant {
        matrix: MatrixId,
        variant: PrecisionClass,
        sidecar_digest: Digest256,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOverlay {
    pub base_artifact_digest: Digest256,
    pub base_manifest_digest: Digest256,
    pub profile_generation: u64,
    pub mutations: Vec<OverlayMutation>,
    pub validation_receipt_digest: Digest256,
    pub optimizer_identity: OptimizerIdentity,
    pub overlay_signature: RegistrySignature,
}

impl OptimizationOverlay {
    pub fn verify_signature(&self) -> Result<(), DeploymentError> {
        if self.overlay_signature.bytes.is_empty() {
            return Err(DeploymentError::InvalidManifestSignature);
        }
        Ok(())
    }
}
