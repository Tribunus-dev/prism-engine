//! ANE-TRI-LANE-CIMAGE-0001: Apple tri-lane execution plan types.
//!
//! Compile-time plan embedding for three-lane (ANE/GPU/CPU) heterogeneous
//! execution on Apple Silicon.  The plan is sealed in the CImage manifest
//! and executed by the runtime scheduler as authoritative data — placement
//! is NOT rediscovered dynamically at decode time.
//!
//! Epoch schedule: GPU owns projection GEMMs, paged attention, KV-cache;
//! ANE owns large static-shape Core-ML-compatible fused islands; CPU owns
//! tokenization, sampling, metadata, and receipt assembly.

use serde::{Deserialize, Serialize};

// Re-export shared lane identity from the existing placement system.
pub use crate::backend::placement::ExecutionLane;

// ── Hardware identity ────────────────────────────────────────────────────

/// Apple hardware signature used for cost-model keying and admission
/// qualification.  Identifies the SoC family, macOS version, and Core ML
/// runtime version for which the plan was compiled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleHardwareSignature {
    /// Apple Silicon SoC family: "M1", "M2", "M3", "M4"
    pub soc_family: String,
    /// macOS version at compile time, e.g. "14.5"
    pub macos_version: String,
    /// Core ML runtime version, e.g. "7.2.0"
    pub coreml_version: String,
    /// Number of performance CPU cores
    pub p_core_count: u32,
    /// Number of GPU cores
    pub gpu_core_count: u32,
    /// ANE neural engine cores (16 on M1, 16 on M2, …)
    pub ane_core_count: u32,
    /// Unified memory size in GB
    pub unified_memory_gb: u32,
}

// ── Shape classification ─────────────────────────────────────────────────

/// Shape class for plan specialization.  The plan is compiled for a specific
/// shape class — dynamic shapes beyond the certified range are rejected by
/// the admission gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeClass {
    /// Batch dimension (1 for autoregressive decode)
    pub batch: u32,
    /// Sequence dimension (1 for decode, `N` for prefill)
    pub sequence: u32,
    /// Hidden dimension
    pub hidden: u32,
    /// Number of query heads
    pub num_heads: u32,
    /// Number of key-value heads
    pub num_kv_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Sliding window (0 if not applicable)
    pub sliding_window: u32,
    /// Maximum context length the plan was compiled for
    pub max_context: u32,
}

// ── Numerical policy ─────────────────────────────────────────────────────

/// Numerical policy for tri-lane execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalPolicy {
    /// Whether all lanes must produce bit-exact results
    pub require_bit_exact: bool,
    /// Maximum tolerated RMS error per layer (when not bit-exact)
    pub max_relative_error: f64,
    /// Whether mixed-precision computation is permitted
    pub allow_mixed_precision: bool,
}

// ── Core ML compute-unit policy ─────────────────────────────────────────

/// Core ML compute-unit configuration for ANE lane programs.
///
/// Only `CpuAndNeuralEngineRequired` and `CpuAndNeuralEnginePreferred` are
/// exposed to the default planner.  `All` is intentionally excluded — it
/// would allow Core ML to consume GPU capacity and invalidate the three-lane
/// schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMlComputeUnitPolicy {
    /// Require CPU + Neural Engine (no GPU fallback for this lane).
    /// Corresponds to `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`
    /// with `setPrecompiledModelAtURLAndConfiguration`.
    CpuAndNeuralEngineRequired,
    /// Prefer CPU + Neural Engine but permit runtime CPU fallback.
    CpuAndNeuralEnginePreferred,
    /// Core ML lane disabled — no ANE program for this artifact.
    Disabled,
}

// ── ANE admission ────────────────────────────────────────────────────────

/// Result of the compile-time ANE admission gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AneAdmission {
    /// Region admitted for ANE compilation.
    Admitted,
    /// Region rejected — reason documented.
    Rejected(AneRejectionReason),
    /// Region admitted experimentally (requires runtime qualification).
    Experimental(AneExperimentalReason),
}

/// Reasons an ANE candidate region was rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AneRejectionReason {
    /// Required operator cannot be lowered to Core ML MIL.
    UnsupportedOperatorLowering(String),
    /// Quantization representation not supported by Core ML.
    UnsupportedQuantization(String),
    /// Dynamic shape outside certified range.
    DynamicShapeOutOfRange(String),
    /// Layout conversion cost exceeds budget.
    LayoutConversionExceedsBudget(u64),
    /// Core ML compilation failed during qualification.
    CoreMlCompilationFailure(String),
    /// Runtime Core ML model load failed.
    RuntimeLoadFailure(String),
    /// Output contract mismatch after prediction.
    OutputContractMismatch(String),
    /// Numerical divergence exceeds tolerance.
    NumericalDivergence(f64),
    /// ANE lane cannot overlap critical GPU path.
    CannotOverlapCriticalPath,
    /// Predicted speedup below minimum threshold.
    PredictedGainBelowThreshold { predicted_us: u64, threshold_us: u64 },
    /// GPU contention risk from this placement.
    GpuContentionRisk,
}

/// Reasons a region was admitted on an experimental basis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AneExperimentalReason {
    /// Only a subset of output ops have been qualified.
    PartialQualification,
    /// Production training data differs from compile-time profiles.
    TrainingDrift,
    /// Region accepted for telemetry gathering.
    TelemetryGathering,
}

// ── Materialization mode ─────────────────────────────────────────────────

/// How tensor data crosses a device boundary.  Should never be called
/// "zero-copy" unless the provider proves it for the exact buffer route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationMode {
    /// IO-arena buffer reused by the consumer lane (proven zero-copy).
    ReusedProviderBuffer,
    /// Explicit shared-memory binding (IOSurface page-table op).
    ExplicitSharedMemoryBinding,
    /// Format conversion required (e.g. FP16→FP32).
    ExplicitConversion,
    /// Memory-level copy required.
    ExplicitCopy,
    /// Managed by Core ML runtime — exact cost unknown.
    RuntimeManagedOpaqueTransfer,
}

// ── Buffer ownership ─────────────────────────────────────────────────────

/// Ownership model for a tensor buffer crossing a lane boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferOwnership {
    /// Producer retains ownership; consumer borrows.
    ProducerOwned,
    /// Consumer takes ownership after boundary.
    ConsumerOwned,
    /// Shared ownership via explicit reference counting.
    SharedRefCounted,
    /// Registered in the IO-arena ring; released after epoch completion.
    ArenaRingSlot,
}

// ── Tensor contracts ─────────────────────────────────────────────────────

/// Description of a single tensor at a Core ML program boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlTensorContract {
    /// Logical tensor name.
    pub name: String,
    /// Expected shape (with symbolic dimensions resolved).
    pub shape: Vec<u64>,
    /// Data type.
    pub dtype: String,
    /// Memory layout (NHWC, NCHW, etc.).
    pub layout: String,
}

/// State contract for stateful Core ML models that maintain RNN-like state
/// across invocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlStateContract {
    /// Number of state tensors.
    pub state_count: u32,
    /// Per-state tensor contracts.
    pub states: Vec<CoreMlTensorContract>,
    /// Whether state is initially zero or loaded from a checkpoint.
    pub initial_state: String,
}

/// Shape contract — static or certified dynamic range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlShapeContract {
    /// Fully static shape.
    pub static_shape: Option<Vec<u64>>,
    /// Dynamic range (min/max per dimension).
    pub dynamic_range: Option<Vec<(u64, u64)>>,
}

/// Warmup contract — how many predictions to run before the lane is healthy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlWarmupContract {
    /// Minimum warmup predictions required.
    pub min_warmup_predictions: u32,
    /// Maximum allowed warmup latency (per prediction).
    pub max_warmup_latency_ms: u64,
    /// Expected output tolerances during warmup.
    pub tolerance: f64,
}

/// Qualification record for an ANE artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneQualificationRecord {
    /// Whether compilation succeeded.
    pub compile_success: bool,
    /// Whether runtime load succeeded.
    pub load_success: bool,
    /// Whether warmup prediction succeeded.
    pub warmup_success: bool,
    /// Expected output present.
    pub output_present: bool,
    /// Numerical comparison result.
    pub numerical_match: bool,
    /// Steady-state latency (ns).
    pub steady_state_latency_ns: u64,
    /// Observed CPU contention.
    pub cpu_contention_ns: u64,
    /// Observed GPU contention.
    pub gpu_contention_ns: u64,
    /// Whether fallback produces correct results.
    pub fallback_correct: bool,
}

// ── Core ML program binding ──────────────────────────────────────────────

/// Sealed Core ML artifact — immutable after CImage compilation.
///
/// The artifact is compiled once during `cimage build`, loaded once during
/// `cimage install`, and executed repeatedly under the declared compute-unit
/// policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlProgramBinding {
    /// Artifact identifier (content-addressed).
    pub artifact_id: String,
    /// Digest of the full .mlpackage bundle.
    pub package_digest: String,
    /// Digest of the compiled .mlmodelc bundle.
    pub compiled_model_digest: String,
    /// Compute-unit policy for this artifact.
    pub compute_unit_policy: CoreMlComputeUnitPolicy,
    /// Input tensor contracts.
    pub input_contract: Vec<CoreMlTensorContract>,
    /// Output tensor contracts.
    pub output_contract: Vec<CoreMlTensorContract>,
    /// State contract (for stateful models).
    pub state_contract: Option<CoreMlStateContract>,
    /// Shape contract.
    pub shape_contract: CoreMlShapeContract,
    /// Warmup contract.
    pub warmup_contract: CoreMlWarmupContract,
    /// Qualification evidence.
    pub qualification: AneQualificationRecord,
}

// ── Core ML boundary contract ───────────────────────────────────────────

/// Contract for activation movement at every Core ML boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlBoundaryContract {
    /// Input tensor identifier.
    pub input_tensor: String,
    /// Output tensor identifier.
    pub output_tensor: String,
    /// Tensor layout.
    pub layout: String,
    /// Data type.
    pub dtype: String,
    /// Producer lane.
    pub producer: ExecutionLane,
    /// Consumer lane.
    pub consumer: ExecutionLane,
    /// How the buffer is materialised.
    pub materialization: MaterializationMode,
    /// Number of reuse slots in the activation ring.
    pub reuse_slots: u8,
    /// Buffer ownership model.
    pub ownership: BufferOwnership,
    /// Measured boundary latency (ns), if profiled.
    pub measured_boundary_ns: Option<u64>,
}

// ── Dependency types ─────────────────────────────────────────────────────

/// Kind of dependency between lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyKind {
    /// Consumer needs producer's output tensor data.
    DataReady,
    /// Consumer may reuse producer's buffer after it's released.
    BufferReuse,
    /// KV-cache writes are visible to the consumer lane.
    KvGenerationVisible,
    /// Numerical validation pass required before consumer may proceed.
    NumericalValidation,
    /// Host-level decision (e.g. sampling result) required.
    HostDecision,
    /// Artifact health check required (e.g. ANE lane health probe).
    ArtifactHealth,
}

/// Completion contract for a dependency edge — what proves the dependency
/// is satisfied and what happens if it isn't.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionContract {
    /// Signal type: fence, timestamp, receipt field.
    pub signal: String,
    /// Timeout for this dependency to resolve.
    pub timeout_ns: u64,
    /// Fallback action on timeout.
    pub on_timeout: String,
    /// Whether the dependency is optional (best-effort).
    pub optional: bool,
}

/// A single dependency edge between two lanes in the epoch schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneDependency {
    /// Producer lane (provides data or signal).
    pub producer: ExecutionLane,
    /// Consumer lane (needs data or signal).
    pub consumer: ExecutionLane,
    /// Dependency kind.
    pub kind: DependencyKind,
    /// Resource identifier (tensor name, buffer slot, etc.).
    pub resource: String,
    /// Source epoch.
    pub from_epoch: u64,
    /// Destination epoch.
    pub to_epoch: u64,
    /// How this edge is satisfied on timeout.
    pub completion: CompletionContract,
}

// ── GPU program binding ──────────────────────────────────────────────────

/// Metal/GPU program binding — the GPU lane plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalProgramBinding {
    /// Metal function name.
    pub function_name: String,
    /// Pipeline state digest.
    pub pipeline_digest: String,
    /// Expected threadgroup size.
    pub threadgroup_size: (u32, u32, u32),
    /// Expected grid size.
    pub grid_size: (u32, u32, u32),
}

// ── CPU program binding ──────────────────────────────────────────────────

/// CPU/Accelerate program binding — the CPU lane plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProgramBinding {
    /// Accelerate function selector.
    pub function_selector: String,
    /// vDSP/vForce routine name.
    pub routine: String,
    /// Expected element count.
    pub element_count: u64,
}

// ── Epoch schedule ───────────────────────────────────────────────────────

/// A single execution epoch — a unit of scheduled work across all three lanes.
///
/// Example epoch structure during decode:
/// - GPU starts projection or attention work for token N.
/// - CPU prepares token N+1 metadata, sampling state, dispatch bindings.
/// - ANE executes an admitted independent or pipelined region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEpoch {
    /// Epoch index (monotonically increasing).
    pub epoch_index: u64,
    /// GPU work descriptor.
    pub gpu_work: Option<String>,
    /// ANE work descriptor.
    pub ane_work: Option<String>,
    /// CPU work descriptor.
    pub cpu_work: Option<String>,
    /// Dependencies that must be satisfied before this epoch begins.
    pub dependencies: Vec<LaneDependency>,
}

// ── Fallback plan ────────────────────────────────────────────────────────

/// Fallback topology — what the scheduler falls back to when ANE is unhealthy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleFallbackPlan {
    /// ANE→GPU fallback for each ANE region.
    pub ane_to_gpu: Vec<String>,
    /// ANE→CPU fallback for each ANE region.
    pub ane_to_cpu: Vec<String>,
    /// Whether the GPU-only fallback is valid.
    pub gpu_only_valid: bool,
    /// Whether the CPU-only fallback is valid.
    pub cpu_only_valid: bool,
}

// ── Cost model ───────────────────────────────────────────────────────────

/// Cost estimate for running a region on a given lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneCostEstimate {
    /// Compute time estimate (ns).
    pub compute_ns: u64,
    /// Memory access time estimate (ns).
    pub memory_ns: u64,
    /// Boundary transfer time estimate (ns).
    pub boundary_ns: u64,
    /// Synchronisation overhead (ns).
    pub sync_ns: u64,
}

/// Tri-lane cost model — predicted cost of a region on each lane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriLaneCostModel {
    /// Estimated cost on GPU (MLX/Metal).
    pub gpu: LaneCostEstimate,
    /// Estimated cost on ANE (Core ML).
    pub ane: LaneCostEstimate,
    /// Estimated cost on CPU (Accelerate/vDSP).
    pub cpu: LaneCostEstimate,
    /// Critical-path latency estimate (minimum possible with overlap).
    pub critical_path_ns: u64,
    /// GPU contention penalty estimate (ns).
    pub gpu_contention_penalty_ns: u64,
    /// CPU contention penalty estimate (ns).
    pub cpu_contention_penalty_ns: u64,
    /// Numerical risk penalty (0.0–1.0, multiplies cost).
    pub numerical_risk_penalty: f64,
    /// Fallback risk penalty (0.0–1.0, multiplies cost).
    pub fallback_risk_penalty: f64,
}

impl TriLaneCostModel {
    /// Construct a TriLaneCostModel from per-lane cost estimates and
    /// contention penalties.
    pub fn new(
        gpu: LaneCostEstimate,
        ane: LaneCostEstimate,
        cpu: LaneCostEstimate,
        critical_path_ns: u64,
        gpu_contention_penalty_ns: u64,
        cpu_contention_penalty_ns: u64,
    ) -> Self {
        Self {
            gpu,
            ane,
            cpu,
            critical_path_ns,
            gpu_contention_penalty_ns,
            cpu_contention_penalty_ns,
            numerical_risk_penalty: 0.0,
            fallback_risk_penalty: 0.0,
        }
    }
}

// ── Evidence and calibration types ──────────────────────────────────────────

/// Evidence about how Core ML configuration was applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlConfigurationEvidence {
    /// Whether the model was loaded with cpuAndNeuralEngine compute policy
    pub loaded_with_cpu_and_neural_engine: bool,
    /// The actual compute policy string used
    pub compute_policy: String,
    /// Timestamp of configuration
    pub configured_at: String,
}

/// Level of ANE execution evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AneExecutionEvidence {
    /// ANE execution not observed
    NotObserved,
    /// Model configured for ANE but execution on ANE not directly confirmed
    ConfiguredOnly,
    /// IOSurface-backed prediction completed and outputs validated
    IOSurfacePredictionValidated,
    /// Trace/profiler verified ANE execution with trace identifier
    TraceVerified { trace_id: String },
}

impl Default for AneExecutionEvidence {
    fn default() -> Self {
        Self::NotObserved
    }
}

/// Slot event for per-epoch receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotEvent {
    pub slot_id: u32,
    pub tensor_id: String,
    pub epoch: u64,
    pub slot_generation: u64,
    pub state: String,
}

/// Fallback status for receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStatus {
    NotActivated,
    Activated { epoch: u64, reason: String },
    RecoveryInProgress { epoch: u64 },
    Permanent { epoch: u64 },
}

impl Default for FallbackStatus {
    fn default() -> Self {
        Self::NotActivated
    }
}

/// Calibration record keyed by device and artifact identity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleTriLaneCalibrationRecord {
    pub hardware_signature: String,
    pub os_build: String,
    pub coreml_runtime_identity: String,
    pub region_fingerprint: String,
    pub artifact_digest: String,
    pub shape_class: ShapeClass,
    pub ring_depth: u8,
    pub measured_ane_ns: u64,
    pub measured_metal_ns: u64,
    pub measured_cpu_ns: u64,
    pub measured_epoch_wall_ns: u64,
    pub measured_overlap_ns: u64,
    pub slot_wait_ns: u64,
    pub fallback_metal_ns: u64,
    pub numerical_error: f64,
}

// ── Evidence requirements ────────────────────────────────────────────────

/// Evidence requirements for tri-lane execution qualification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriLaneEvidenceRequirements {
    /// Whether each ANE region must pass numerical validation.
    pub validate_numerics: bool,
    /// Minimum steady-state predictions required.
    pub min_steady_state_predictions: u32,
    /// Whether to collect boundary materialisation costs.
    pub collect_boundary_costs: bool,
    /// Whether to profile GPU contention.
    pub profile_gpu_contention: bool,
    /// Whether to profile CPU contention.
    pub profile_cpu_contention: bool,
    /// Whether to verify fallback correctness.
    pub verify_fallback: bool,
}

// ── Execution receipt ────────────────────────────────────────────────────

/// Receipt produced after executing one epoch of the tri-lane plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneExecutionEvent {
    /// Which lane.
    pub lane: ExecutionLane,
    /// Whether execution succeeded.
    pub success: bool,
    /// Compute time (ns).
    pub compute_ns: u64,
    /// Memory time (ns).
    pub memory_ns: u64,
    /// Synchronisation time (ns).
    pub sync_ns: u64,
}

/// Boundary materialisation receipt — costs of crossing a lane boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryMaterializationReceipt {
    pub input_tensor: String,
    pub output_tensor: String,
    pub producer: ExecutionLane,
    pub consumer: ExecutionLane,
    pub materialization: MaterializationMode,
    pub actual_ns: u64,
}

/// Overlap metrics for an epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlapMetrics {
    /// Wall-clock time for the epoch (ns).
    pub epoch_wall_ns: u64,
    /// Total compute time across all lanes (ns).
    pub total_compute_ns: u64,
    /// Total synchronisation overhead (ns).
    pub total_sync_ns: u64,
    /// Useful overlap time (compute in parallel, ns).
    pub overlap_ns: u64,
    /// Useful overlap time (compute in parallel, ns).
    pub overlap_fraction: f64,
}

/// Numerical status for an epoch.

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NumericalStatus {
    /// All outputs matched the reference within tolerance.
    Pass,
    /// Some outputs exceeded tolerance (logged).
    Warning(String),
    /// Numerical divergence detected.
    Fail(String),
}

/// Per-epoch execution receipt for Apple tri-lane plans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleTriLaneExecutionReceipt {
    /// CImage identifier that this receipt refers to.
    pub cimage_id: String,
    /// Digest of the plan that was executed.
    pub plan_digest: String,
    /// Epoch index.
    pub epoch: u64,
    /// Per-lane execution events.
    pub lane_events: Vec<LaneExecutionEvent>,
    /// ANE artifact identifier (if ANE was used).
    pub ane_artifact_id: Option<String>,
    /// ANE admission status for this epoch.
    pub ane_admission: AneAdmission,
    /// Boundary materialisation costs.
    pub boundary_events: Vec<BoundaryMaterializationReceipt>,
    /// Overlap metrics.
    pub overlap_ns: OverlapMetrics,
    /// Whether fallback was activated.
    pub fallback_used: bool,
    /// Per-slot IO-arena events for this epoch.
    pub slot_events: Vec<SlotEvent>,
    /// Detailed fallback status beyond the boolean.
    pub fallback_status: FallbackStatus,
    /// Evidence about how Core ML configuration was applied (if available).
    pub coreml_configuration: Option<CoreMlConfigurationEvidence>,
    /// Level of evidence confirming ANE execution.
    pub ane_execution_evidence: AneExecutionEvidence,
    /// Numerical validation status.
    pub numerical_status: NumericalStatus,
    /// Whether the plan was configured for cpuAndNeuralEngine.
    pub configured_cpu_and_neural_engine: bool,
    /// Whether ANE execution was actually observed (not just configured).
    pub observed_ane_execution: bool,
}

// ── The main execution plan ──────────────────────────────────────────────

/// Apple tri-lane execution plan — sealed in the CImage manifest.
///
/// Contains three lane programs, their resource contracts, epoch
/// dependencies, and a fallback topology.  The scheduler executes this
/// plan as authoritative data — it does NOT rediscover placement
/// dynamically at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleTriLaneExecutionPlan {
    /// Plan version (increment on breaking changes).
    pub plan_version: u32,
    /// Hardware signature for which this plan was compiled.
    pub hardware_signature: AppleHardwareSignature,
    /// Shape class (batch=1, seq=1 for decode).
    pub shape_class: ShapeClass,
    /// Numerical policy.
    pub numerical_policy: NumericalPolicy,
    /// ANE lane program (None when no region is ANE-eligible).
    pub ane_program: Option<CoreMlProgramBinding>,
    /// GPU lane program.
    pub gpu_program: MetalProgramBinding,
    /// CPU lane program.
    pub cpu_program: CpuProgramBinding,
    /// All tensor bindings shared across lanes.
    pub tensors: Vec<CoreMlTensorContract>,
    /// Lane dependency graph.
    pub dependencies: Vec<LaneDependency>,
    /// Epoch schedule.
    pub epochs: Vec<ExecutionEpoch>,
    /// Fallback topology.
    pub fallback_plan: AppleFallbackPlan,
    /// Predicted cost model.
    pub predicted_cost: TriLaneCostModel,
    /// Evidence requirements for qualification.
    pub evidence_requirements: TriLaneEvidenceRequirements,
}

// ── ANE lane lifecycle ───────────────────────────────────────────────────

/// States of the ANE lane lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AneLaneLifecycle {
    /// Lane not yet initialised.
    Unavailable,
    /// Artifact digest verified.
    ArtifactVerified,
    /// Core ML model compiled.
    Compiled,
    /// Model loaded into runtime.
    Loaded,
    /// Warmup completed.
    Warmed,
    /// Lane ready for inference.
    Healthy,
    /// Lane throttled (high temperature, power limit, etc.).
    Throttled,
    /// Lane degraded (partial functionality).
    Degraded,
    /// Lane failed — fallback active.
    Failed,
    /// Fallback plan is executing.
    FallbackActive,
    /// Lane released (resources freed).
    Released,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apple_tri_lane_plan_roundtrip() {
        let plan = AppleTriLaneExecutionPlan {
            plan_version: 1,
            hardware_signature: AppleHardwareSignature {
                soc_family: "M1".into(),
                macos_version: "14.5".into(),
                coreml_version: "7.2.0".into(),
                p_core_count: 4,
                gpu_core_count: 8,
                ane_core_count: 16,
                unified_memory_gb: 16,
            },
            shape_class: ShapeClass {
                batch: 1,
                sequence: 1,
                hidden: 2048,
                num_heads: 8,
                num_kv_heads: 8,
                head_dim: 128,
                sliding_window: 0,
                max_context: 8192,
            },
            numerical_policy: NumericalPolicy {
                require_bit_exact: false,
                max_relative_error: 1e-4,
                allow_mixed_precision: true,
            },
            ane_program: None,
            gpu_program: MetalProgramBinding {
                function_name: "rms_norm_fused".into(),
                pipeline_digest: "abcd1234".into(),
                threadgroup_size: (32, 1, 1),
                grid_size: (64, 1, 1),
            },
            cpu_program: CpuProgramBinding {
                function_selector: "vDSP_vadd".into(),
                routine: "vDSP".into(),
                element_count: 2048,
            },
            tensors: vec![CoreMlTensorContract {
                name: "hidden_states".into(),
                shape: vec![1, 2048],
                dtype: "float16".into(),
                layout: "NHWC".into(),
            }],
            dependencies: vec![],
            epochs: vec![],
            fallback_plan: AppleFallbackPlan {
                ane_to_gpu: vec![],
                ane_to_cpu: vec![],
                gpu_only_valid: true,
                cpu_only_valid: false,
            },
            predicted_cost: TriLaneCostModel {
                gpu: LaneCostEstimate { compute_ns: 100_000, memory_ns: 50_000, boundary_ns: 0, sync_ns: 5_000 },
                ane: LaneCostEstimate { compute_ns: 80_000, memory_ns: 40_000, boundary_ns: 20_000, sync_ns: 10_000 },
                cpu: LaneCostEstimate { compute_ns: 300_000, memory_ns: 100_000, boundary_ns: 0, sync_ns: 2_000 },
                critical_path_ns: 120_000,
                gpu_contention_penalty_ns: 0,
                cpu_contention_penalty_ns: 0,
                numerical_risk_penalty: 0.0,
                fallback_risk_penalty: 0.0,
            },
            evidence_requirements: TriLaneEvidenceRequirements {
                validate_numerics: true,
                min_steady_state_predictions: 100,
                collect_boundary_costs: true,
                profile_gpu_contention: true,
                profile_cpu_contention: false,
                verify_fallback: true,
            },
        };

        let json = serde_json::to_string_pretty(&plan).unwrap();
        let deserialized: AppleTriLaneExecutionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.plan_version, 1);
        assert_eq!(deserialized.hardware_signature.soc_family, "M1");
        assert!(deserialized.ane_program.is_none());
        assert_eq!(deserialized.gpu_program.function_name, "rms_norm_fused");
        assert!(deserialized.fallback_plan.gpu_only_valid);
    }

    #[test]
    fn test_ane_admission_serialization() {
        let admitted = AneAdmission::Admitted;
        let rejected = AneAdmission::Rejected(AneRejectionReason::DynamicShapeOutOfRange("batch > 1".into()));
        let experimental = AneAdmission::Experimental(AneExperimentalReason::PartialQualification);

        let j1 = serde_json::to_string(&admitted).unwrap();
        let j2 = serde_json::to_string(&rejected).unwrap();
        let j3 = serde_json::to_string(&experimental).unwrap();

        assert!(matches!(serde_json::from_str::<AneAdmission>(&j1).unwrap(), AneAdmission::Admitted));
        let back: AneAdmission = serde_json::from_str(&j2).unwrap();
        match back {
            AneAdmission::Rejected(ref r) => match r {
                AneRejectionReason::DynamicShapeOutOfRange(s) => assert_eq!(s, "batch > 1"),
                _ => panic!("wrong rejection variant"),
            },
            _ => panic!("wrong admission variant"),
        }
        assert!(matches!(serde_json::from_str::<AneAdmission>(&j3).unwrap(), AneAdmission::Experimental(_)));
    }

    #[test]
    fn test_core_ml_program_binding_serde() {
        let binding = CoreMlProgramBinding {
            artifact_id: "test-artifact".into(),
            package_digest: "pkg123".into(),
            compiled_model_digest: "cmp456".into(),
            compute_unit_policy: CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired,
            input_contract: vec![CoreMlTensorContract {
                name: "input".into(),
                shape: vec![1, 2048],
                dtype: "float16".into(),
                layout: "NHWC".into(),
            }],
            output_contract: vec![],
            state_contract: None,
            shape_contract: CoreMlShapeContract {
                static_shape: Some(vec![1, 2048]),
                dynamic_range: None,
            },
            warmup_contract: CoreMlWarmupContract {
                min_warmup_predictions: 3,
                max_warmup_latency_ms: 100,
                tolerance: 0.01,
            },
            qualification: AneQualificationRecord {
                compile_success: true,
                load_success: true,
                warmup_success: true,
                output_present: true,
                numerical_match: true,
                steady_state_latency_ns: 80_000,
                cpu_contention_ns: 1_000,
                gpu_contention_ns: 0,
                fallback_correct: true,
            },
        };

        let json = serde_json::to_string_pretty(&binding).unwrap();
        let back: CoreMlProgramBinding = serde_json::from_str(&json).unwrap();
        assert_eq!(back.artifact_id, "test-artifact");
        assert_eq!(back.compute_unit_policy, CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired);
        assert_eq!(back.input_contract[0].shape, vec![1, 2048]);
    }

    #[test]
    fn test_lifecycle_transitions() {
        let states = vec![
            AneLaneLifecycle::Unavailable,
            AneLaneLifecycle::ArtifactVerified,
            AneLaneLifecycle::Compiled,
            AneLaneLifecycle::Loaded,
            AneLaneLifecycle::Warmed,
            AneLaneLifecycle::Healthy,
            AneLaneLifecycle::Throttled,
            AneLaneLifecycle::Degraded,
            AneLaneLifecycle::Failed,
            AneLaneLifecycle::FallbackActive,
            AneLaneLifecycle::Released,
        ];
        for state in &states {
            let json = serde_json::to_string(state).unwrap();
            let back: AneLaneLifecycle = serde_json::from_str(&json).unwrap();
            assert_eq!(*state, back);
        }
    }

    #[test]
    fn test_core_ml_compute_unit_policy_serde() {
        let policies = vec![
            CoreMlComputeUnitPolicy::CpuAndNeuralEngineRequired,
            CoreMlComputeUnitPolicy::CpuAndNeuralEnginePreferred,
            CoreMlComputeUnitPolicy::Disabled,
        ];
        for p in &policies {
            let json = serde_json::to_string(p).unwrap();
            let back: CoreMlComputeUnitPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(*p, back);
        }
    }

    #[test]
    fn test_apple_tri_lane_execution_receipt() {
        let receipt = AppleTriLaneExecutionReceipt {
            cimage_id: "test-cimage".into(),
            plan_digest: "plan123".into(),
            epoch: 0,
            lane_events: vec![],
            ane_artifact_id: None,
            ane_admission: AneAdmission::Rejected(
                AneRejectionReason::PredictedGainBelowThreshold { predicted_us: 90, threshold_us: 100 },
            ),
            boundary_events: vec![],
            overlap_ns: OverlapMetrics {
                epoch_wall_ns: 150_000,
                total_compute_ns: 300_000,
                total_sync_ns: 10_000,
                overlap_ns: 50_000,
                overlap_fraction: 0.33,
            },
            fallback_used: false,
            slot_events: vec![],
            fallback_status: FallbackStatus::NotActivated,
            coreml_configuration: None,
            ane_execution_evidence: AneExecutionEvidence::NotObserved,
            numerical_status: NumericalStatus::Pass,
            configured_cpu_and_neural_engine: false,
            observed_ane_execution: false,
        };

        let json = serde_json::to_string_pretty(&receipt).unwrap();
        let back: AppleTriLaneExecutionReceipt = serde_json::from_str(&json).unwrap();
        match back.ane_admission {
            AneAdmission::Rejected(AneRejectionReason::PredictedGainBelowThreshold { predicted_us, threshold_us }) => {
                assert_eq!(predicted_us, 90);
                assert_eq!(threshold_us, 100);
            }
            _ => panic!("wrong admission variant"),
        }
        assert!(matches!(back.numerical_status, NumericalStatus::Pass));
        assert!(back.slot_events.is_empty());
        assert!(matches!(back.fallback_status, FallbackStatus::NotActivated));
        assert!(back.coreml_configuration.is_none());
        assert!(matches!(back.ane_execution_evidence, AneExecutionEvidence::NotObserved));
    }
}
