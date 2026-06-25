//! Heterogeneous compiler pipeline phase IR — intermediate representation
//! types shared across compilation stages.
//!
//! These types describe what a compile phase is, where it runs, what
//! evidence it produced, and the deterministic contracts the pipeline
//! must uphold — forming the shared vocabulary between palette
//! compilation, ANE admission, staging rings, and calibration lanes.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
// Runtime residency contract reused for compile-phase tensor tracking.
pub use crate::backend::residency::{
    TensorResidency, TransferDecision, BackendId, MemoryDomain,
};

// ── Core identities ───────────────────────────────────────────────────────

/// Unique compilation session identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationId(pub u64);

impl CompilationId {
    /// Allocate the next globally-unique session id.
    pub fn next() -> Self {
        static NEXT_COMPILATION: AtomicU64 = AtomicU64::new(1);
        CompilationId(NEXT_COMPILATION.fetch_add(1, Ordering::Relaxed))
    }
}

/// Unique numeric identifier for a compile phase within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhaseId(pub u64);

impl PhaseId {
    /// Allocate the next phase id for this session.
    pub fn next() -> Self {
        static NEXT_PHASE: AtomicUsize = AtomicUsize::new(1);
        PhaseId(NEXT_PHASE.fetch_add(1, Ordering::Relaxed) as u64)
    }
}

// ── Device identity ───────────────────────────────────────────────────────

/// Describes a device for admission and profiling purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSignature {
    /// Unique device or profile identifier.
    pub device_id: String,
    /// Chip model string (e.g. "Apple M1", "arm64").
    pub chip: String,
    /// Total memory available on this device (bytes).
    pub max_memory_bytes: u64,
}

// ── Tensor contracts ──────────────────────────────────────────────────────

/// Describes a single tensor's name, type, shape, and materialisation mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorContract {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub materialization: MaterializationContract,
}

/// How a tensor is materialised in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterializationContract {
    CpuOwned,
    MetalShared,
    AneIoSurface,
}

// ── Shape / intensity / mutation ──────────────────────────────────────────

/// Shape classification for a phase's primary tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeClass {
    /// Fully known at plan time.
    Static(Vec<u64>),
    /// Resolved at runtime.
    Dynamic,
}

/// Arithmetic intensity regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArithmeticIntensity {
    MemoryBound,
    ComputeBound,
}

/// How a phase mutates its tensor state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationClass {
    ReadOnly,
    ProducesNew,
    MutatesInPlace,
}

// ── Determinism ───────────────────────────────────────────────────────────

/// Numerical determinism requirement for a compile phase.
///
/// `Unknown` phases are ineligible for ANE placement — the admission gate
/// requires either `BitExact` or `NumericallyBounded`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompileDeterminism {
    BitExact,
    NumericallyBounded {
        abs_error: f32,
        rel_error: f32,
    },
    Unknown,
}

// ── Placement / routing ───────────────────────────────────────────────────

/// Target (or actual) backend placement for a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompilePlacement {
    MetalGpu,
    Ane,
    CpuAccelerate,
    Cpu,
}

/// Effective execution route taken at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectiveRoute {
    AccelerateCpu,
    MetalGpu,
    Ane,
    Cpu,
}

// ── Bridge / data transfer ────────────────────────────────────────────────

/// Classification of how tensor data crosses device boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeKind {
    CpuOwned,
    MetalShared,
    AneIoSurface,
}

// ── Validation / fallback ─────────────────────────────────────────────────

/// Result of a numerical validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    Passed,
    Failed(String),
    Skipped,
}

/// Reason a phase fell back from its requested placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackReason {
    Performance,
    Memory,
    Capability,
    Numerical,
    BridgeBudget,
}

// ── Phase descriptor ──────────────────────────────────────────────────────

/// Full descriptor for a single compilation phase — everything the
/// admission gate, scheduler, and profiling infrastructure need.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilePhaseDescriptor {
    pub phase_id: PhaseId,
    pub inputs: Vec<TensorContract>,
    pub outputs: Vec<TensorContract>,
    pub shape_class: ShapeClass,
    pub arithmetic_intensity: ArithmeticIntensity,
    pub mutation: MutationClass,
    pub determinism: CompileDeterminism,
    pub allowed_placements: Vec<CompilePlacement>,
    pub minimum_profitable_elements: u64,
    pub fallback: CompilePlacement,
    /// Estimated ANE execution duration (ns), for admission-gate
    /// performance comparison against the GPU baseline.
    pub estimated_ane_duration_ns: u64,
    /// Number of bytes expected to cross the ANE bridge.
    pub bridge_copy_bytes: u64,
}

// ── Execution receipt ─────────────────────────────────────────────────────

/// Evidence produced when a compilation phase completes on a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileExecutionReceipt {
    pub compilation_id: CompilationId,
    pub phase_id: PhaseId,
    pub requested_placement: CompilePlacement,
    pub effective_route: EffectiveRoute,
    pub artifact_key: Option<String>,
    pub device_signature: DeviceSignature,
    pub input_elements: u64,
    pub output_elements: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub submit_ns: u64,
    pub queue_wait_ns: u64,
    pub execution_ns: u64,
    pub materialization_ns: u64,
    pub dependency_wait_ns: u64,
    pub total_ns: u64,
    pub bridge_kind: BridgeKind,
    pub copy_count: u64,
    pub copied_bytes: u64,
    pub numerical_validation: ValidationResult,
    pub fallback_reason: Option<FallbackReason>,
    pub coreml_compute_units: Option<String>,
}

// ── ANE artifact key ──────────────────────────────────────────────────────

/// Content-addressed key identifying a compilable ANE artifact.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ANEArtifactKey {
    pub program_hash: [u8; 32],
}

// ── Boundary tensor contracts ──────────────────────────────────────────────

/// Canonical dtype for phase boundary contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TensorDtype {
    Float16,
    Float32,
    BFloat16,
    Int8,
    UInt8,
    UInt16,
    Int32,
    Unknown,
}

impl TensorDtype {
    pub fn is_fp16(self) -> bool {
        matches!(self, Self::Float16)
    }
}

/// Boundary tensor contract for a phase that crosses the ANE-to-Metal boundary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoundaryTensorContract {
    pub tensor_id: String,
    pub dtype: TensorDtype,
    pub logical_shape: Vec<u64>,
    pub physical_shape: Vec<u64>,
    pub strides_bytes: Vec<u64>,
    pub static_shape: bool,
    pub layout_digest: String,
}

// ── Region and edge types ────────────────────────────────────────────────────

/// A region of contiguous PhaseIR operations with a shared ANE eligibility assessment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhaseRegion {
    pub region_id: RegionId,
    pub operations: Vec<CompilePhaseDescriptor>,
    pub placement_candidates: Vec<crate::compilation::phase_ir::CompilePlacement>,
    pub ane_eligibility: crate::compilation::ane_eligibility::AneEligibility,
    pub input_contract: Option<crate::compilation::activation_abi::ActivationContract>,
    pub output_contract: Option<crate::compilation::activation_abi::ActivationContract>,
}

/// Edge connecting two PhaseIR phases with ABI contract and materialization plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhaseEdge {
    pub producer: PhaseId,
    pub consumer: PhaseId,
    pub logical_tensor: LogicalTensorId,
    pub producer_output_abi: crate::compilation::activation_abi::ActivationAbi,
    pub consumer_input_abi: crate::compilation::activation_abi::ActivationAbi,
    pub materialization_plan: MaterializationPlan,
}

/// How tensor data crosses device boundaries between phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MaterializationPlan {
    /// Direct shared memory (IOSurface handoff).
    DirectShared,
    /// Metal epilogue repacks tensor layout for ANE consumption.
    MetalEpilogueRepack,
    /// Explicit Metal-side repack.
    ExplicitMetalRepack,
    /// CPU-side materialization (slow path).
    CpuMaterialization,
    /// Not permitted for production ANE edges.
    Forbidden,
}

/// Unique region identifier.
pub type RegionId = u64;

/// Unique logical tensor identifier within a session.
pub type LogicalTensorId = u64;

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compilation_id_unique() {
        let a = CompilationId::next();
        let b = CompilationId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn phase_id_unique() {
        let a = PhaseId::next();
        let b = PhaseId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn compile_determinism_serde() {
        for variant in &[
            CompileDeterminism::BitExact,
            CompileDeterminism::NumericallyBounded {
                abs_error: 0.001,
                rel_error: 0.01,
            },
            CompileDeterminism::Unknown,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let back: CompileDeterminism = serde_json::from_str(&json).unwrap();
            match (variant, &back) {
                (CompileDeterminism::BitExact, CompileDeterminism::BitExact) => {}
                (CompileDeterminism::Unknown, CompileDeterminism::Unknown) => {}
                (
                    CompileDeterminism::NumericallyBounded {
                        abs_error: a,
                        rel_error: b,
                    },
                    CompileDeterminism::NumericallyBounded {
                        abs_error: c,
                        rel_error: d,
                    },
                ) => {
                    assert!((a - c).abs() < 1e-6);
                    assert!((b - d).abs() < 1e-6);
                }
                _ => panic!("variant mismatch"),
            }
        }
    }

    #[test]
    fn compile_placement_serde() {
        for variant in &[
            CompilePlacement::MetalGpu,
            CompilePlacement::Ane,
            CompilePlacement::CpuAccelerate,
            CompilePlacement::Cpu,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let back: CompilePlacement = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, back);
        }
    }

    #[test]
    fn ane_artifact_key_serde() {
        let key = ANEArtifactKey {
            program_hash: [0xab; 32],
        };
        let json = serde_json::to_string(&key).unwrap();
        let restored: ANEArtifactKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, restored);
    }
}
