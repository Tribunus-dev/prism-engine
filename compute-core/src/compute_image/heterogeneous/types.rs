//! PRISM-CIMAGE-HETEROGENEOUS-COMPILATION-0001
//!
//! Heterogeneous execution image types — the compiler-emitted primary artifact
//! for tri-lane (Metal GPU / Core ML ANE / Accelerate CPU) execution.
//!
//! These types form the new top-level cimage section. Every cimage intended
//! for Prism Engine serving must contain a [`HeterogeneousExecutionImage`].
//! A backend-only (Metal-only) image is represented as a degenerate one-lane
//! heterogeneous graph.
//!
//! ── Design invariants ────────────────────────────────────────────────────
//!
//! * All types are `Serialize + Deserialize` via serde for embedding in the
//!   cimage as a dedicated JSON section.
//! * Types reference existing shared vocabulary (`ExecutionLane`, `ActivationAbi`,
//!   `ContentHash`) where appropriate.
//! * The image is immutable after sealing — no mutable runtime state here.
//! * The graph is guaranteed acyclic at emission time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export shared type identities.
pub use crate::backend::placement::ExecutionLane;
pub use crate::compilation::activation_abi::ActivationAbi;
pub use crate::integration::ContentHash;

#[cfg(test)]
use crate::compilation::activation_abi::{DecodeActivationV1Params, PhysicalLayout};
#[cfg(test)]
use crate::compilation::phase_ir::TensorDtype;

// ═══════════════════════════════════════════════════════════════════════════
// Section 0: Top-level image
// ═══════════════════════════════════════════════════════════════════════════

/// Primary top-level cimage section for heterogeneous execution.
///
/// Bundles the compiler-emitted phase graph, resource plan, lane programs,
/// concurrency plan, admission rules, fallback topology, execution policies,
/// and evidence contract into one sealed artifact.
///
/// The runtime consumes this image directly via [`HeterogeneousRuntime`] —
/// it does not reconstruct backend placement, resource ownership, or
/// concurrency semantics from disconnected manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionImage {
    pub image_version: u32,
    pub model_identity: ModelIdentity,
    pub graph_digest: ContentHash,
    pub phase_graph: CompiledPhaseGraph,
    pub resources: CompiledResourcePlan,
    pub lane_programs: CompiledLanePrograms,
    pub concurrency: CompiledConcurrencyPlan,
    pub admission: CompiledAdmissionPlan,
    pub fallback: CompiledFallbackPlan,
    pub execution_policy: CompiledExecutionPolicies,
    pub evidence_contract: CompiledEvidenceContract,
}

/// Identity and provenance of the imported model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_name: String,
    pub model_family: String,
    pub model_variant: String,
    pub canonical_graph_hash: ContentHash,
    pub compile_timestamp: String,
    pub compiler_version: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: Canonical PhaseIR
// ═══════════════════════════════════════════════════════════════════════════

/// The compiler lowers all frontends into one canonical graph before
/// backend decisions.  A [`PhaseNode`] represents an executable semantic
/// region, not necessarily one operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraph {
    pub phases: Vec<PhaseNode>,
    pub edges: Vec<PhaseEdge>,
    pub values: Vec<PhaseValue>,
}

/// A single executable semantic region in the canonical PhaseIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseNode {
    pub phase_id: PhaseId,
    pub kind: PhaseKind,
    pub operators: Vec<OperatorId>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub shape_contract: ShapeContract,
    pub numerical_contract: NumericalContract,
    pub dependency_class: DependencyClass,
}

/// A dependency edge between two PhaseIR nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEdge {
    pub from: PhaseId,
    pub to: PhaseId,
    pub value: Option<ValueId>,
    pub kind: PhaseEdgeKind,
}

/// A value (tensor / activation) flowing between phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseValue {
    pub value_id: ValueId,
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub producer: Option<PhaseId>,
    pub consumers: Vec<PhaseId>,
}

/// Identifies a phase within the compilation session.
pub type PhaseId = u64;

/// Identifies a value in the phase graph.
pub type ValueId = u64;

/// Identifies an operator within a phase.
pub type OperatorId = String;

/// Kind of executable semantic region.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhaseKind {
    Attention,
    MlpGate,
    MlpUp,
    MlpDown,
    MlpActivation,
    RmsNorm,
    RoPE,
    ResidualAdd,
    LogitsProjection,
    Sampling,
    Softmax,
    KvUpdate,
    KvCacheLookup,
    Prologue,
    Epilogue,
    Fusion,
    DataTransfer,
}

/// How a dependency class affects concurrency.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DependencyClass {
    /// Strict token-autoregressive dependency — must serialize.
    StrictTokenDependency,
    /// Intra-layer dependency (e.g., attention → MLP within one layer).
    IntraLayerDependency,
    /// Cross-sequence independent — phases from different sequences can overlap.
    CrossSequenceIndependent,
    /// Prefill batch independent — phases within a prefill batch are independent.
    PrefillBatchIndependent,
    /// Background or speculative work — can overlap with decode.
    BackgroundSpeculativeIndependent,
    /// Host-only dependency (e.g., tokenization, metadata).
    HostOnlyDependency,
}

/// Shape contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeContract {
    pub batch_dim: Option<u64>,
    pub seq_len: Option<u64>,
    pub hidden_dim: u64,
    pub num_heads: u64,
    pub head_dim: u64,
}

/// Numerical contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalContract {
    pub accumulation_dtype: String,
    pub activation_dtype: String,
    pub requires_determinism: bool,
}

/// Kind of edge between PhaseIR nodes.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhaseEdgeKind {
    Data,
    Control,
    State,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: Backend capability analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Every phase receives a capability record for all three lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCapabilityMatrix {
    pub phase_id: PhaseId,
    pub metal: LaneCapability,
    pub ane: LaneCapability,
    pub accelerate: LaneCapability,
}

/// Lane capability for a single phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaneCapability {
    /// The lane supports this phase directly with the given cost and ABI.
    Supported {
        estimated_cost: CompileCostEstimate,
        required_abi: ActivationAbi,
        required_artifacts: Vec<ArtifactRequirement>,
    },
    /// Supported but requires materialization (data transfer) at boundaries.
    SupportedWithMaterialization {
        estimated_cost: CompileCostEstimate,
        materialization: MaterializationPlan,
        required_abi: ActivationAbi,
    },
    /// Not supported — records the reason explicitly.
    Unsupported { reason: UnsupportedReason },
}

/// Why a lane cannot execute a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnsupportedReason {
    OperatorNotImplemented(String),
    ShapeOutOfRange(String),
    NumericalContractUnsatisfied(String),
    DynamicShapeUnsupported(String),
    ResourceConstraint(String),
    QualificationFailed(String),
    Other(String),
}

/// Compile-time cost estimate for a phase on a lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileCostEstimate {
    pub expected_ns: u64,
    pub memory_bytes: u64,
    pub compute_intensity: f64,
    pub confidence: CostConfidence,
}

/// Confidence level of a cost estimate.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CostConfidence {
    Measured,
    Profiled,
    Estimated,
    Speculative,
}

/// An artifact requirement (e.g., a compiled .mlmodelc or .metallib).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRequirement {
    pub artifact_kind: ArtifactKind,
    pub artifact_id: String,
    pub content_hash: ContentHash,
}

/// Kind of compiled artifact.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactKind {
    CoreMlModel,
    MetalLibrary,
    MetalKernel,
    AccelerateRoutine,
    WeightPack,
    ArenaPlan,
}

/// How tensor data crosses device boundaries.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum MaterializationPlan {
    /// Zero-copy IOSurface — the preferred mode.
    IOSurfaceShared,
    /// IOSurface-backed pointer binding through MLMultiArray.
    IOSurfacePointerBackedMultiArray,
    /// Explicit host-side copy.
    HostCopy,
    /// IOSurface pixel buffer (CVPixelBuffer) binding.
    IOSurfacePixelBuffer,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 3: Region formation decisions
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a region formation decision — whether to merge adjacent phases
/// into a fused region or keep them separate for concurrency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionFormationDecision {
    pub region_id: RegionId,
    pub merged_phases: Vec<PhaseId>,
    pub selected_lane_candidates: Vec<ExecutionLane>,
    pub fusion_gain_ns: u64,
    pub lost_overlap_ns: u64,
    pub decision: RegionDecision,
}

/// Whether a region formation was accepted or rejected.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum RegionDecision {
    Fused,
    KeptSeparate,
}

pub type RegionId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 4: Resource and ABI planning
// ═══════════════════════════════════════════════════════════════════════════

/// The compiler-owned activation resource plan.
///
/// Describes every arena, slot, alias, materialization node, and resource
/// lifetime interval.  This is the source of truth — the runtime does not
/// guess resource ownership.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledResourcePlan {
    pub arenas: Vec<ArenaPlan>,
    pub slots: Vec<CompiledSlot>,
    pub aliases: Vec<SlotAlias>,
    pub materializations: Vec<MaterializationNode>,
    pub lifetime_intervals: Vec<ResourceLifetime>,
}

/// Describes one activation arena (IOSurface pool or host heap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaPlan {
    pub arena_id: ArenaId,
    pub byte_size: u64,
    pub alignment: u64,
    pub backing: ArenaBacking,
    pub ring_depth: u32,
}

/// How an arena is backed at runtime.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArenaBacking {
    /// IOSurface — the standard shared activation backing where Metal and
    /// Core ML interoperate.
    IOSurface,
    /// Host heap allocation (CPU-pointer accessible).
    HostHeap,
    /// Metal buffer allocation.
    MetalBuffer,
}

/// A compiled slot — describes a single activation or tensor binding
/// within an arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledSlot {
    pub slot_id: SlotId,
    pub arena_id: ArenaId,
    pub activation_abi: ActivationAbi,
    pub byte_length: u64,
    pub alignment: u64,
    pub backing: SlotBacking,
    pub producer_phase: PhaseId,
    pub consumer_phases: Vec<PhaseId>,
    pub concurrency_class: ConcurrencyClass,
}

/// How a slot's backing memory is provisioned.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SlotBacking {
    /// IOSurface — shared between Metal and Core ML.
    IOSurface,
    /// Host pointer — CPU-accessible, usable by Accelerate.
    HostPointer,
    /// Metal private buffer — GPU-only.
    MetalPrivate,
}

/// ABI binding mode for the slot.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConcurrencyClass {
    Exclusive,
    SharedRead,
    ProducerConsumer,
}

/// A slot alias — two slot ids that share the same backing memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotAlias {
    pub alias_id: SlotAliasId,
    pub primary_slot: SlotId,
    pub secondary_slot: SlotId,
    pub offset_bytes: u64,
}

/// A materialization node — inserted where data must cross device boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationNode {
    pub materialization_id: MaterializationId,
    pub from_slot: SlotId,
    pub to_slot: SlotId,
    pub plan: MaterializationPlan,
    pub estimated_cost_ns: u64,
}

/// A resource lifetime interval — when a slot is live.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLifetime {
    pub slot_id: SlotId,
    pub first_phase: PhaseId,
    pub last_phase: PhaseId,
}

pub type ArenaId = u64;
pub type SlotId = u64;
pub type SlotAliasId = u64;
pub type MaterializationId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 5: Concurrency analysis
// ═══════════════════════════════════════════════════════════════════════════

/// The compiler-emitted concurrency plan.
///
/// Describes which phases are independently ready, which groups may run in
/// parallel, which edges require serialization, what lane capacity is
/// required, and hints about expected overlap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledConcurrencyPlan {
    pub ready_sets: Vec<ReadySetTemplate>,
    pub parallel_groups: Vec<ParallelGroup>,
    pub serialization_edges: Vec<SerializationEdge>,
    pub lane_caps: LaneCapacityRequirements,
    pub overlap_hints: Vec<OverlapHint>,
}

/// Template for a ready set — phases that are independently dispatchable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadySetTemplate {
    pub ready_set_id: ReadySetId,
    pub phases: Vec<PhaseId>,
}

/// A parallel group — phases that may be dispatched before awaiting any member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    pub group_id: ParallelGroupId,
    pub phases: Vec<PhaseId>,
    pub required_distinct_slots: Vec<SlotId>,
    pub allowed_lanes: Vec<ExecutionLane>,
    pub expected_overlap_kind: OverlapKind,
}

/// How overlap is expected to manifest.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum OverlapKind {
    /// True concurrent execution across lanes.
    ConcurrentLanes,
    /// Pipelined execution within a single lane.
    PipelineWithinLane,
    /// Interleaved execution across sequences.
    InterleavedSequences,
    /// Sequential — no overlap possible.
    Sequential,
}

/// A serialization constraint between phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationEdge {
    pub from: PhaseId,
    pub to: PhaseId,
    pub reason: SerializationReason,
}

/// Why two phases must be serialized.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SerializationReason {
    DataDependency,
    MutableSlot,
    LaneCapacity,
    Barrier,
    AdmissionGate,
    NumericalConstraint,
}

/// Compiler-emitted lane capacity requirements.
///
/// The runtime may provide more capacity, but may not run a concurrency-
/// required image below its declared safe minimum without downgrading to
/// a serial plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneCapacityRequirements {
    pub metal_in_flight_min: u32,
    pub ane_in_flight_min: u32,
    pub accelerate_workers_min: u32,
    pub iosurface_ring_depth_min: u32,
    pub completion_queue_min: u32,
}

/// A hint about expected overlap between specific phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapHint {
    pub phase_a: PhaseId,
    pub phase_b: PhaseId,
    pub expected_overlap_kind: OverlapKind,
    pub confidence: CostConfidence,
}

pub type ReadySetId = u64;
pub type ParallelGroupId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 6: Backend lowering — lane programs
// ═══════════════════════════════════════════════════════════════════════════

/// All compiled lane programs for the three execution lanes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledLanePrograms {
    pub metal: Vec<MetalProgram>,
    pub ane: Vec<AneProgram>,
    pub accelerate: Vec<AccelerateProgram>,
}

/// A compiled Metal GPU program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalProgram {
    pub program_id: ProgramId,
    pub pipeline_identifier: String,
    pub threadgroup_size: (u32, u32, u32),
    pub grid_size: (u32, u32, u32),
    pub buffer_bindings: Vec<SlotId>,
    pub texture_bindings: Vec<SlotId>,
    pub specialization_constants: HashMap<String, u32>,
    pub estimated_occupancy: f64,
    pub memory_cost_bytes: u64,
    pub synchronization_requirements: Vec<String>,
    pub binding: ProgramBinding,
}

/// A compiled Core ML / ANE program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneProgram {
    pub program_id: ProgramId,
    pub package_identity: String,
    pub compiled_model_key: String,
    pub function_name: String,
    pub shape_bucket: String,
    pub compute_policy: String,
    pub input_bindings: Vec<FeatureBinding>,
    pub output_bindings: Vec<FeatureBinding>,
    pub warmup_contract: WarmupContract,
    pub qualification_key: String,
    pub binding_mode: AneBindingMode,
    pub binding: ProgramBinding,
}

/// How a Core ML feature is bound to a slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBinding {
    pub feature_name: String,
    pub slot_id: SlotId,
    pub dtype: String,
    pub shape: Vec<u64>,
}

/// Warmup contract for a Core ML artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupContract {
    pub min_warmup_predictions: u32,
    pub max_warmup_predictions: u32,
    pub warmup_batch_size: u32,
}

/// Binding representation for ANE programs.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum AneBindingMode {
    /// IOSurface pixel buffer binding.
    IOSurfacePixelBuffer,
    /// IOSurface-backed MLMultiArray pointer binding.
    IOSurfacePointerBackedMultiArray,
    /// CPU host materialized — explicit host-side buffer.
    HostMaterialized,
}

/// A compiled Accelerate / CPU program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerateProgram {
    pub program_id: ProgramId,
    pub routine_identity: String,
    pub input_slot_bindings: Vec<SlotId>,
    pub output_slot_bindings: Vec<SlotId>,
    pub vector_shape: Vec<u64>,
    pub matrix_shape: (u64, u64),
    pub worker_class: String,
    pub determinism_contract: String,
    pub expected_duration_ns: u64,
    pub binding: ProgramBinding,
}

/// Common binding wrapper for all lane program types.
///
/// The executor sees one interface regardless of lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramBinding {
    pub program_id: ProgramId,
    pub phase_id: PhaseId,
    pub lane: ExecutionLane,
    pub input_slots: Vec<SlotId>,
    pub output_slots: Vec<SlotId>,
    pub input_abi: Vec<ActivationAbi>,
    pub output_abi: Vec<ActivationAbi>,
    pub resource_accesses: Vec<ResourceAccess>,
    pub execution_constraints: ExecutionConstraints,
}

/// Description of a resource access (slot read/write).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAccess {
    pub slot_id: SlotId,
    pub access: SlotAccess,
}

/// Slot access kind.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SlotAccess {
    Read,
    Write,
    ReadWrite,
}

/// Execution constraints for a program binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    pub max_concurrent_invocations: u32,
    pub requires_determinism: bool,
    pub priority: PriorityClass,
    pub required_capabilities: Vec<String>,
}

/// Priority class for scheduling.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PriorityClass {
    Critical,
    Interactive,
    Batch,
    Background,
}

pub type ProgramId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 7: Variant graph synthesis
// ═══════════════════════════════════════════════════════════════════════════

/// For each phase, the compiler emits all semantically valid variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseVariantSet {
    pub phase_id: PhaseId,
    pub variant_set_id: VariantSetId,
    pub variants: Vec<CompiledPhaseVariant>,
}

/// A single compiled variant for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseVariant {
    pub variant_id: VariantId,
    pub lane: ExecutionLane,
    pub program_id: ProgramId,
    pub input_slots: Vec<SlotId>,
    pub output_slots: Vec<SlotId>,
    pub predicted_cost: CompileCostEstimate,
    pub admission_requirements: Vec<AdmissionRequirement>,
    pub fallback_target: Option<VariantId>,
    pub overlap_profile: OverlapProfile,
}

/// An admission requirement for a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdmissionRequirement {
    ArtifactQualified { artifact_id: String },
    HardwareCapability { capability: String },
    ThermalState { max_thermal_state: u32 },
    MemoryPressure { max_memory_pressure: u32 },
    Warm { artifact_id: String },
}

/// Overlap profile for a variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapProfile {
    pub lane_occupancy_ns: u64,
    pub slot_read_set: Vec<SlotId>,
    pub slot_write_set: Vec<SlotId>,
    pub can_overlap_with: Vec<ExecutionLane>,
}

pub type VariantSetId = u64;
pub type VariantId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 8: Compiled dependency graph (runtime-ready)
// ═══════════════════════════════════════════════════════════════════════════

/// The final runtime-ready graph.  This is what the executor consumes
/// directly — no runtime dependency reconstruction is required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseGraph {
    pub nodes: Vec<CompiledPhaseNode>,
    pub edges: Vec<CompiledPhaseEdge>,
    pub entrypoints: Vec<PhaseId>,
    pub terminal_nodes: Vec<PhaseId>,
}

/// A node in the compiled phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseNode {
    pub phase_id: PhaseId,
    pub variant_set_id: VariantSetId,
    pub ready_condition: ReadyCondition,
    pub parallel_group: Option<ParallelGroupId>,
    pub priority_class: PriorityClass,
}

/// When a compiled phase node becomes ready for dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadyCondition {
    /// All statically declared dependencies must be satisfied.
    AllDependenciesSatisfied,
    /// At least one dependency group satisfied (OR semantics).
    AnyDependencyGroupSatisfied,
    /// Always ready (e.g., entrypoint).
    AlwaysReady,
    /// Conditional on an admission gate result.
    AdmissionGate { gate_id: String },
}

/// A directed edge in the compiled phase graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledPhaseEdge {
    pub from: PhaseId,
    pub to: PhaseId,
    pub dependency: CompiledDependency,
}

/// The kind of dependency between two compiled phase nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompiledDependency {
    /// Data dependency on a specific slot with required access.
    Data {
        slot: SlotId,
        required_access: SlotAccess,
    },
    /// Control dependency — ordering constraint only.
    Control,
    /// Barrier — a synchronization point.
    Barrier { reason: BarrierReason },
    /// Admission gate — a dependency on runtime qualification.
    Admission { requirement: AdmissionRequirement },
}

/// Why a barrier edge exists.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum BarrierReason {
    EpochBoundary,
    MaterializationComplete,
    CacheFlush,
    ThermalThrottleRecovery,
    WarmupComplete,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 9: Qualification and admission
// ═══════════════════════════════════════════════════════════════════════════

/// The compiler-emitted admission plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledAdmissionPlan {
    pub hardware_signature_requirements: HardwareRequirements,
    pub artifact_qualification: Vec<ArtifactQualificationPlan>,
    pub route_admission_rules: Vec<RouteAdmissionRule>,
}

/// Required hardware signature for the image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    pub min_soc_family: String,
    pub min_macos_version: String,
    pub min_coreml_version: String,
    pub min_ane_count: u32,
    pub min_gpu_core_count: u32,
    pub required_features: Vec<String>,
}

/// Plan for qualifying an artifact at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactQualificationPlan {
    pub artifact_id: String,
    pub artifact_kind: ArtifactKind,
    pub qualification_key: String,
    pub required_digest: ContentHash,
    pub warmup_required: bool,
    pub min_warmup_runs: u32,
}

/// A route admission rule — which conditions must hold for a route to be used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteAdmissionRule {
    pub route_id: String,
    pub lane: ExecutionLane,
    pub required_qualifications: Vec<String>,
    pub required_hardware_features: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 10: Fallback plan
// ═══════════════════════════════════════════════════════════════════════════

/// The compiler-emitted fallback plan — describes fallback transitions
/// that preserve semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledFallbackPlan {
    pub fallback_chains: Vec<FallbackChain>,
    pub transition_rules: Vec<FallbackTransitionRule>,
}

/// A fallback chain — ordered list of variant alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    pub chain_id: FallbackChainId,
    pub phase_id: PhaseId,
    pub primary_variant: VariantId,
    pub fallback_variants: Vec<FallbackStep>,
}

/// A single fallback step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackStep {
    pub variant_id: VariantId,
    pub reason: FallbackReason,
    pub estimated_cost_delta_ns: i64,
}

/// Why a fallback was taken.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackReason {
    ArtifactUnavailable,
    ArtifactNotWarm,
    AdmissionDenied,
    LaneOverloaded,
    NumericalMismatch,
    ThermalThrottle,
    MemoryPressure,
    Timeout,
    ExplicitDowngrade,
}

/// A transition rule for fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackTransitionRule {
    pub from_variant: VariantId,
    pub to_variant: VariantId,
    pub preserves_semantics: bool,
    pub requires_rematerialization: bool,
    pub transition_cost_ns: u64,
}

pub type FallbackChainId = u64;

// ═══════════════════════════════════════════════════════════════════════════
// Section 11: Execution policies
// ═══════════════════════════════════════════════════════════════════════════

/// Multiple execution policies for different serving modes.
///
/// These policies share code and artifacts but differ in scheduling hints,
/// queue limits, ring depth requirements, and variant preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledExecutionPolicies {
    pub latency_single_sequence: ExecutionPolicyId,
    pub throughput_multi_sequence: ExecutionPolicyId,
    pub prefill_batch: ExecutionPolicyId,
    pub degraded_metal_only: ExecutionPolicyId,
    pub policies: HashMap<ExecutionPolicyId, CompiledExecutionPolicy>,
}

/// A single execution policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledExecutionPolicy {
    pub policy_id: ExecutionPolicyId,
    pub scheduling_hints: SchedulingHints,
    pub queue_limits: QueueLimits,
    pub variant_preferences: Vec<LanePreference>,
    pub lane_capacity_overrides: Option<LaneCapacityRequirements>,
}

/// Scheduling hints for a policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingHints {
    pub prefer_concurrent_lanes: bool,
    pub max_in_flight_sequences: u32,
    pub prefill_priority: PriorityClass,
    pub decode_priority: PriorityClass,
}

/// Queue limits for a policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueLimits {
    pub metal_queue_depth: u32,
    pub ane_queue_depth: u32,
    pub accelerate_queue_depth: u32,
    pub completion_queue_depth: u32,
}

/// Lane preference for variant selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanePreference {
    pub lane: ExecutionLane,
    pub preference: LanePreferenceKind,
}

/// Preference kind for a lane.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum LanePreferenceKind {
    Preferred,
    Allowed,
    FallbackOnly,
    Prohibited,
}

pub type ExecutionPolicyId = String;

// ═══════════════════════════════════════════════════════════════════════════
// Section 12: Evidence contract
// ═══════════════════════════════════════════════════════════════════════════

/// Contract for runtime evidence collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledEvidenceContract {
    pub collect_latency: bool,
    pub collect_power: bool,
    pub collect_slot_usage: bool,
    pub emit_traces: bool,
    pub sampling_rate: f64,
    pub max_entries: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 13: Compilation receipt
// ═══════════════════════════════════════════════════════════════════════════

/// Receipt produced by the compilation process — evidence that the
/// heterogeneous image was correctly compiled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationReceipt {
    pub image_digest: ContentHash,
    pub phase_count: usize,
    pub variant_count: usize,
    pub metal_program_count: usize,
    pub ane_program_count: usize,
    pub accelerate_program_count: usize,
    pub parallel_group_count: usize,
    pub materialization_count: usize,
    pub rejected_variants: Vec<RejectedVariant>,
    pub emitted_fallback_count: usize,
}

/// A rejected variant — records why a phase did not receive an ANE or
/// Accelerate variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedVariant {
    pub phase_id: PhaseId,
    pub lane: ExecutionLane,
    pub reason: UnsupportedReason,
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 14: Compiler-to-executor contract
// ═══════════════════════════════════════════════════════════════════════════

/// Versioned contract between compiler and executor.
///
/// The executor refuses to instantiate an image with missing or incompatible
/// sections.  Each component digest is computed independently so partial
/// invalidation is detectable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionContract {
    pub contract_version: u32,
    pub image_digest: ContentHash,
    pub phase_graph_digest: ContentHash,
    pub resource_plan_digest: ContentHash,
    pub program_plan_digest: ContentHash,
    pub concurrency_plan_digest: ContentHash,
}

// ═══════════════════════════════════════════════════════════════════════════
// Default implementations
// ═══════════════════════════════════════════════════════════════════════════

impl Default for LaneCapacityRequirements {
    fn default() -> Self {
        Self {
            metal_in_flight_min: 1,
            ane_in_flight_min: 1,
            accelerate_workers_min: 1,
            iosurface_ring_depth_min: 2,
            completion_queue_min: 1,
        }
    }
}

impl Default for CompiledEvidenceContract {
    fn default() -> Self {
        Self {
            collect_latency: true,
            collect_power: false,
            collect_slot_usage: true,
            emit_traces: false,
            sampling_rate: 0.01,
            max_entries: 10000,
        }
    }
}

impl Default for CompiledExecutionPolicies {
    fn default() -> Self {
        let latency_policy = ExecutionPolicyId::from("latency_single");
        let throughput_policy = ExecutionPolicyId::from("throughput_multi");
        let prefill_policy = ExecutionPolicyId::from("prefill_batch");
        let degraded_policy = ExecutionPolicyId::from("degraded_metal");
        let mut policies = HashMap::new();
        policies.insert(
            latency_policy.clone(),
            CompiledExecutionPolicy {
                policy_id: latency_policy.clone(),
                scheduling_hints: SchedulingHints {
                    prefer_concurrent_lanes: true,
                    max_in_flight_sequences: 1,
                    prefill_priority: PriorityClass::Critical,
                    decode_priority: PriorityClass::Critical,
                },
                queue_limits: QueueLimits {
                    metal_queue_depth: 2,
                    ane_queue_depth: 1,
                    accelerate_queue_depth: 1,
                    completion_queue_depth: 4,
                },
                variant_preferences: vec![
                    LanePreference {
                        lane: ExecutionLane::MlxGpu,
                        preference: LanePreferenceKind::Preferred,
                    },
                    LanePreference {
                        lane: ExecutionLane::CoreMlAne,
                        preference: LanePreferenceKind::Allowed,
                    },
                    LanePreference {
                        lane: ExecutionLane::AccelerateCpu,
                        preference: LanePreferenceKind::FallbackOnly,
                    },
                ],
                lane_capacity_overrides: None,
            },
        );
        policies.insert(
            throughput_policy.clone(),
            CompiledExecutionPolicy {
                policy_id: throughput_policy.clone(),
                scheduling_hints: SchedulingHints {
                    prefer_concurrent_lanes: true,
                    max_in_flight_sequences: 4,
                    prefill_priority: PriorityClass::Interactive,
                    decode_priority: PriorityClass::Interactive,
                },
                queue_limits: QueueLimits {
                    metal_queue_depth: 4,
                    ane_queue_depth: 2,
                    accelerate_queue_depth: 2,
                    completion_queue_depth: 8,
                },
                variant_preferences: vec![
                    LanePreference {
                        lane: ExecutionLane::MlxGpu,
                        preference: LanePreferenceKind::Preferred,
                    },
                    LanePreference {
                        lane: ExecutionLane::CoreMlAne,
                        preference: LanePreferenceKind::Preferred,
                    },
                    LanePreference {
                        lane: ExecutionLane::AccelerateCpu,
                        preference: LanePreferenceKind::Allowed,
                    },
                ],
                lane_capacity_overrides: None,
            },
        );
        policies.insert(
            prefill_policy.clone(),
            CompiledExecutionPolicy {
                policy_id: prefill_policy.clone(),
                scheduling_hints: SchedulingHints {
                    prefer_concurrent_lanes: false,
                    max_in_flight_sequences: 1,
                    prefill_priority: PriorityClass::Critical,
                    decode_priority: PriorityClass::Background,
                },
                queue_limits: QueueLimits {
                    metal_queue_depth: 1,
                    ane_queue_depth: 1,
                    accelerate_queue_depth: 1,
                    completion_queue_depth: 2,
                },
                variant_preferences: vec![
                    LanePreference {
                        lane: ExecutionLane::MlxGpu,
                        preference: LanePreferenceKind::Preferred,
                    },
                    LanePreference {
                        lane: ExecutionLane::CoreMlAne,
                        preference: LanePreferenceKind::Allowed,
                    },
                ],
                lane_capacity_overrides: None,
            },
        );
        policies.insert(
            degraded_policy.clone(),
            CompiledExecutionPolicy {
                policy_id: degraded_policy.clone(),
                scheduling_hints: SchedulingHints {
                    prefer_concurrent_lanes: false,
                    max_in_flight_sequences: 1,
                    prefill_priority: PriorityClass::Interactive,
                    decode_priority: PriorityClass::Batch,
                },
                queue_limits: QueueLimits {
                    metal_queue_depth: 1,
                    ane_queue_depth: 0,
                    accelerate_queue_depth: 1,
                    completion_queue_depth: 2,
                },
                variant_preferences: vec![
                    LanePreference {
                        lane: ExecutionLane::MlxGpu,
                        preference: LanePreferenceKind::Preferred,
                    },
                    LanePreference {
                        lane: ExecutionLane::AccelerateCpu,
                        preference: LanePreferenceKind::Allowed,
                    },
                ],
                lane_capacity_overrides: None,
            },
        );
        Self {
            latency_single_sequence: latency_policy,
            throughput_multi_sequence: throughput_policy,
            prefill_batch: prefill_policy,
            degraded_metal_only: degraded_policy,
            policies,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all types serialize and deserialize correctly through a
    /// round-trip of the top-level image.
    #[test]
    fn test_heterogeneous_execution_image_roundtrip() {
        let image = sample_image();
        let json = serde_json::to_string_pretty(&image).expect("serialize");
        let deserialized: HeterogeneousExecutionImage =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(image.image_version, deserialized.image_version);
        assert_eq!(
            image.model_identity.model_name,
            deserialized.model_identity.model_name
        );
        assert_eq!(image.graph_digest, deserialized.graph_digest);
        assert_eq!(
            image.phase_graph.nodes.len(),
            deserialized.phase_graph.nodes.len()
        );
        assert_eq!(
            image.phase_graph.edges.len(),
            deserialized.phase_graph.edges.len()
        );
        assert_eq!(
            image.phase_graph.entrypoints,
            deserialized.phase_graph.entrypoints
        );
        assert_eq!(
            image.resources.slots.len(),
            deserialized.resources.slots.len()
        );
        assert_eq!(
            image.lane_programs.metal.len(),
            deserialized.lane_programs.metal.len()
        );
        assert_eq!(
            image.lane_programs.ane.len(),
            deserialized.lane_programs.ane.len()
        );
        assert_eq!(
            image.lane_programs.accelerate.len(),
            deserialized.lane_programs.accelerate.len()
        );
        assert_eq!(
            image.concurrency.parallel_groups.len(),
            deserialized.concurrency.parallel_groups.len()
        );
        assert!(deserialized
            .execution_policy
            .policies
            .contains_key(&deserialized.execution_policy.latency_single_sequence));
    }

    /// Verify that the image serializes to valid JSON.
    #[test]
    fn test_serializes_to_valid_json() {
        let image = sample_image();
        let json = serde_json::to_string_pretty(&image).expect("serialize");
        // Verify it's parseable JSON
        let _parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        // Verify key fields exist in the JSON
        assert!(json.contains("image_version"));
        assert!(json.contains("model_identity"));
        assert!(json.contains("phase_graph"));
        assert!(json.contains("resources"));
        assert!(json.contains("lane_programs"));
        assert!(json.contains("concurrency"));
        assert!(json.contains("admission"));
        assert!(json.contains("fallback"));
        assert!(json.contains("execution_policy"));
        assert!(json.contains("evidence_contract"));
    }

    /// Verify that the image can be sealed and unsealed correctly.
    #[test]
    fn test_compilation_receipt_roundtrip() {
        let receipt = sample_receipt();
        let json = serde_json::to_string_pretty(&receipt).expect("serialize");
        let deserialized: CompilationReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt.phase_count, deserialized.phase_count);
        assert_eq!(
            receipt.metal_program_count,
            deserialized.metal_program_count
        );
        assert_eq!(receipt.ane_program_count, deserialized.ane_program_count);
        assert_eq!(
            receipt.rejected_variants.len(),
            deserialized.rejected_variants.len()
        );
    }

    /// Verify the contract round-trips correctly.
    #[test]
    fn test_contract_roundtrip() {
        let contract = HeterogeneousExecutionContract {
            contract_version: 1,
            image_digest: ContentHash(42),
            phase_graph_digest: ContentHash(100),
            resource_plan_digest: ContentHash(200),
            program_plan_digest: ContentHash(300),
            concurrency_plan_digest: ContentHash(400),
        };
        let json = serde_json::to_string_pretty(&contract).expect("serialize");
        let deserialized: HeterogeneousExecutionContract =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(contract.contract_version, deserialized.contract_version);
        assert_eq!(contract.image_digest, deserialized.image_digest);
    }

    // ── Sample data helpers ────────────────────────────────────────────

    fn sample_image() -> HeterogeneousExecutionImage {
        let phase_graph = sample_phase_graph();
        let resources = sample_resource_plan();
        let lane_programs = sample_lane_programs();
        let concurrency = sample_concurrency_plan();
        let admission = sample_admission_plan();
        let fallback = sample_fallback_plan();
        let execution_policy = CompiledExecutionPolicies::default();
        let evidence_contract = CompiledEvidenceContract::default();

        HeterogeneousExecutionImage {
            image_version: 1,
            model_identity: ModelIdentity {
                model_name: "gemma-2b".into(),
                model_family: "gemma".into(),
                model_variant: "2b".into(),
                canonical_graph_hash: ContentHash(0xDEAD_BEEF),
                compile_timestamp: "2026-06-25T00:00:00Z".into(),
                compiler_version: "0.1.0".into(),
            },
            graph_digest: ContentHash(0xCAFE_BABE),
            phase_graph,
            resources,
            lane_programs,
            concurrency,
            admission,
            fallback,
            execution_policy,
            evidence_contract,
        }
    }

    fn sample_phase_graph() -> CompiledPhaseGraph {
        CompiledPhaseGraph {
            nodes: vec![
                CompiledPhaseNode {
                    phase_id: 0,
                    variant_set_id: 0,
                    ready_condition: ReadyCondition::AlwaysReady,
                    parallel_group: Some(0),
                    priority_class: PriorityClass::Critical,
                },
                CompiledPhaseNode {
                    phase_id: 1,
                    variant_set_id: 1,
                    ready_condition: ReadyCondition::AllDependenciesSatisfied,
                    parallel_group: Some(0),
                    priority_class: PriorityClass::Critical,
                },
                CompiledPhaseNode {
                    phase_id: 2,
                    variant_set_id: 2,
                    ready_condition: ReadyCondition::AllDependenciesSatisfied,
                    parallel_group: None,
                    priority_class: PriorityClass::Interactive,
                },
            ],
            edges: vec![
                CompiledPhaseEdge {
                    from: 0,
                    to: 1,
                    dependency: CompiledDependency::Data {
                        slot: 0,
                        required_access: SlotAccess::Read,
                    },
                },
                CompiledPhaseEdge {
                    from: 1,
                    to: 2,
                    dependency: CompiledDependency::Control,
                },
            ],
            entrypoints: vec![0],
            terminal_nodes: vec![2],
        }
    }

    fn sample_resource_plan() -> CompiledResourcePlan {
        CompiledResourcePlan {
            arenas: vec![ArenaPlan {
                arena_id: 0,
                byte_size: 64 * 1024 * 1024, // 64 MB
                alignment: 16384,
                backing: ArenaBacking::IOSurface,
                ring_depth: 2,
            }],
            slots: vec![
                CompiledSlot {
                    slot_id: 0,
                    arena_id: 0,
                    activation_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                        dtype: TensorDtype::Float16,
                        seq_bucket: 8192,
                        physical_layout: PhysicalLayout::ContiguousRowMajor,
                        hidden_dim: 2048,
                        alignment: 16384,
                        stride_constraint: None,
                    }),
                    byte_length: 8192 * 2048 * 2,
                    alignment: 16384,
                    backing: SlotBacking::IOSurface,
                    producer_phase: 0,
                    consumer_phases: vec![1, 2],
                    concurrency_class: ConcurrencyClass::ProducerConsumer,
                },
                CompiledSlot {
                    slot_id: 1,
                    arena_id: 0,
                    activation_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                        dtype: TensorDtype::Float16,
                        seq_bucket: 8192,
                        physical_layout: PhysicalLayout::ContiguousRowMajor,
                        hidden_dim: 2048,
                        alignment: 16384,
                        stride_constraint: None,
                    }),
                    byte_length: 8192 * 2048 * 2,
                    alignment: 16384,
                    backing: SlotBacking::IOSurface,
                    producer_phase: 1,
                    consumer_phases: vec![2],
                    concurrency_class: ConcurrencyClass::Exclusive,
                },
            ],
            aliases: vec![],
            materializations: vec![],
            lifetime_intervals: vec![
                ResourceLifetime {
                    slot_id: 0,
                    first_phase: 0,
                    last_phase: 2,
                },
                ResourceLifetime {
                    slot_id: 1,
                    first_phase: 1,
                    last_phase: 2,
                },
            ],
        }
    }

    fn sample_lane_programs() -> CompiledLanePrograms {
        CompiledLanePrograms {
            metal: vec![MetalProgram {
                program_id: 0,
                pipeline_identifier: "attention_kernel_v2".into(),
                threadgroup_size: (16, 1, 1),
                grid_size: (128, 1, 1),
                buffer_bindings: vec![0],
                texture_bindings: vec![],
                specialization_constants: HashMap::new(),
                estimated_occupancy: 0.75,
                memory_cost_bytes: 0,
                synchronization_requirements: vec!["memory_barrier".into()],
                binding: ProgramBinding {
                    program_id: 0,
                    phase_id: 1,
                    lane: ExecutionLane::MlxGpu,
                    input_slots: vec![0],
                    output_slots: vec![1],
                    input_abi: vec![],
                    output_abi: vec![],
                    resource_accesses: vec![],
                    execution_constraints: ExecutionConstraints {
                        max_concurrent_invocations: 2,
                        requires_determinism: true,
                        priority: PriorityClass::Critical,
                        required_capabilities: vec![],
                    },
                },
            }],
            ane: vec![AneProgram {
                program_id: 1,
                package_identity: "gemma_mlp_v1".into(),
                compiled_model_key: "gemma_2b_mlp".into(),
                function_name: "main".into(),
                shape_bucket: "static_small".into(),
                compute_policy: "cpuAndNeuralEngine".into(),
                input_bindings: vec![FeatureBinding {
                    feature_name: "input".into(),
                    slot_id: 0,
                    dtype: "float16".into(),
                    shape: vec![1, 2048],
                }],
                output_bindings: vec![FeatureBinding {
                    feature_name: "output".into(),
                    slot_id: 1,
                    dtype: "float16".into(),
                    shape: vec![1, 2048],
                }],
                warmup_contract: WarmupContract {
                    min_warmup_predictions: 3,
                    max_warmup_predictions: 10,
                    warmup_batch_size: 1,
                },
                qualification_key: "gemma_2b_mlp_v1_qk".into(),
                binding_mode: AneBindingMode::IOSurfacePointerBackedMultiArray,
                binding: ProgramBinding {
                    program_id: 1,
                    phase_id: 2,
                    lane: ExecutionLane::CoreMlAne,
                    input_slots: vec![0],
                    output_slots: vec![1],
                    input_abi: vec![],
                    output_abi: vec![],
                    resource_accesses: vec![],
                    execution_constraints: ExecutionConstraints {
                        max_concurrent_invocations: 1,
                        requires_determinism: true,
                        priority: PriorityClass::Interactive,
                        required_capabilities: vec!["ane".into()],
                    },
                },
            }],
            accelerate: vec![],
        }
    }

    fn sample_concurrency_plan() -> CompiledConcurrencyPlan {
        CompiledConcurrencyPlan {
            ready_sets: vec![],
            parallel_groups: vec![ParallelGroup {
                group_id: 0,
                phases: vec![0, 1],
                required_distinct_slots: vec![0],
                allowed_lanes: vec![ExecutionLane::MlxGpu, ExecutionLane::CoreMlAne],
                expected_overlap_kind: OverlapKind::ConcurrentLanes,
            }],
            serialization_edges: vec![],
            lane_caps: LaneCapacityRequirements::default(),
            overlap_hints: vec![],
        }
    }

    fn sample_admission_plan() -> CompiledAdmissionPlan {
        CompiledAdmissionPlan {
            hardware_signature_requirements: HardwareRequirements {
                min_soc_family: "Apple M1".into(),
                min_macos_version: "14.0".into(),
                min_coreml_version: "7.0".into(),
                min_ane_count: 1,
                min_gpu_core_count: 8,
                required_features: vec!["fp16".into(), "iosurface".into()],
            },
            artifact_qualification: vec![],
            route_admission_rules: vec![],
        }
    }

    fn sample_fallback_plan() -> CompiledFallbackPlan {
        CompiledFallbackPlan {
            fallback_chains: vec![],
            transition_rules: vec![],
        }
    }

    fn sample_receipt() -> CompilationReceipt {
        CompilationReceipt {
            image_digest: ContentHash(0xDEAD_BEEF),
            phase_count: 3,
            variant_count: 4,
            metal_program_count: 2,
            ane_program_count: 1,
            accelerate_program_count: 0,
            parallel_group_count: 1,
            materialization_count: 0,
            rejected_variants: vec![RejectedVariant {
                phase_id: 0,
                lane: ExecutionLane::AccelerateCpu,
                reason: UnsupportedReason::OperatorNotImplemented("paged_attention".into()),
            }],
            emitted_fallback_count: 1,
        }
    }
}
