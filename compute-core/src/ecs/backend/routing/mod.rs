//! Deterministic heterogeneous routing types.

pub mod lanes;
pub mod policy;

pub use lanes::*;
pub use policy::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use super::{DType, TensorHandle};

// ── Identity types ────────────────────────────────────────────────────────

/// Identifies a logical tensor across backend boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

/// Identifies a logical operation in the Tribunus-owned execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub u64);

/// Identifies a specific backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub u32);

/// Canonical backend identity values — stable across all builds and artifacts.
pub const BACKEND_METAL: BackendId = BackendId(0);
pub const BACKEND_ACCELERATE: BackendId = BackendId(1);
pub const BACKEND_ANE: BackendId = BackendId(2);
pub const BACKEND_MLX: BackendId = BackendId(3);
/// Megakernel fused Metal GPU decode — the production autoregressive decode path.
pub const BACKEND_MEGAKERNEL: BackendId = BackendId(4);

/// Identifies a sealed route profile (deterministic backend assignment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteProfileId(pub u64);

/// Identifies a compiled backend artifact (e.g. Core ML model, packed layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendArtifactId(pub u64);

/// Identifies a specific materialization of a tensor on a particular backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorMaterializationId(pub u64);

/// Identifies a compiled graph region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompiledRegionHandle {
    /// Slot index into the backend's compiled region array.
    pub slot: u32,
    /// Generation counter, bumped on eviction/replacement.
    pub generation: u32,
}

/// Identifies an evaluation group (synchronization fence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvaluationGroupId(pub u64);

/// Machine profile identity (model + hardware + thermal state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MachineProfileId(pub u64);

/// Evidence digest — content-addressed proof of a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EvidenceDigest(pub String);

// ── Substrate ─────────────────────────────────────────────────────────────

/// Requested compute substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestedSubstrate {
    Cpu,
    Gpu,
    NeuralEngine,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

/// Observed compute substrate — `Unknown` until native instrumentation
/// provides defensible placement evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Substrate {
    Cpu,
    Gpu,
    NeuralEngine,
    Unknown,
}

// ── Operation descriptor ──────────────────────────────────────────────────

/// Logical shape before any physical layout is applied.
#[derive(Debug, Clone)]
pub struct LogicalShape {
    pub dims: Vec<u32>,
}

/// Physical layout (row-major, column-major, packed, etc.).
#[derive(Debug, Clone)]
pub enum PhysicalLayout {
    RowMajor,
    ColumnMajor,
    PackedU32 { group_size: u32, bits: u8 },
    Custom(String),
}

/// Quantization contract carried through the operation.
#[derive(Debug, Clone)]
pub struct QuantizationContract {
    pub bits: u8,
    pub group_size: u32,
    pub symmetric: bool,
}

/// Tensor shape descriptor.
#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dims: Vec<u32>,
}

/// Execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
    Conditioning,
    Qualification,
}

/// Operation family for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationFamily {
    QuantizedMatmul,
    Matmul,
    RmsNorm,
    RoPE,
    Silu,
    Add,
    Multiply,
    Softmax,
    Transpose,
    Reshape,
    IndexSelect,
    Sampling,
    Reduction,
    LayoutTransform,
    Checksum,
    MlpBlock,
    AttentionBlock,
    DecoderLayer,
    PrefillFragment,
    /// Vision encoder (image → embeddings)
    VisionEncode,
    /// Audio encoder (audio → embeddings)
    AudioEncode,
    /// Multimodal projection (encoder embeddings → hidden space)
    MultimodalProject,
}

pub type OperationContractDigest = EvidenceDigest;

/// Policy for correctness checkpointing.
#[derive(Debug, Clone)]
pub enum CorrectnessCheckpointPolicy {
    None,
    CompareAgainstAuthority { tolerance: f64 },
    Checksum { digest: EvidenceDigest },
}

/// Complete descriptor for a single logical operation.
#[derive(Debug, Clone)]
pub struct OperationDescriptor {
    pub operation_id: OperationId,
    pub family: OperationFamily,
    pub layer_index: Option<u32>,
    pub phase: Phase,
    pub logical_shape: LogicalShape,
    pub physical_layout: PhysicalLayout,
    pub input_dtypes: Vec<DType>,
    pub output_dtype: DType,
    pub quantization: Option<QuantizationContract>,
    pub expected_output_shape: TensorShape,
    pub correctness_checkpoint: CorrectnessCheckpointPolicy,
}

// ── Tensor version ────────────────────────────────────────────────────────

/// Version counter for a logical tensor (incremented on mutation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorVersion(pub u64);

// ── Route profile ──────────────────────────────────────────────────────────

/// One routed operation in a deterministic profile.
#[derive(Debug, Clone)]
pub struct RoutedOperation {
    pub operation_id: OperationId,
    pub operation_contract: OperationContractDigest,
    pub backend: BackendId,
    pub requested_substrate: RequestedSubstrate,
    pub backend_artifact: Option<BackendArtifactId>,
    pub input_materializations: Vec<TensorMaterializationId>,
    pub output_materialization: TensorMaterializationId,
    pub evaluation_group: EvaluationGroupId,
    pub fallback_policy: FallbackPolicy,
}

/// What to do when the routed backend cannot execute.
#[derive(Debug, Clone)]
pub enum FallbackPolicy {
    FailClosed,
    FallbackTo(BackendId),
    RetryOnce(BackendId),
}

/// Manifest of backend-specific artifacts referenced by a route profile.
#[derive(Debug, Clone)]
pub struct BackendArtifactManifest {
    pub coreai: Vec<BackendArtifactId>,
    pub accelerate: Vec<BackendArtifactId>,
    pub mlx: Vec<BackendArtifactId>,
}

/// A sealed, deterministic route profile — compiled, not improvised.
#[derive(Debug, Clone)]
pub struct ComputeRouteProfile {
    pub profile_id: RouteProfileId,
    pub logical_image_hash: EvidenceDigest,
    pub artifact_root_hash: EvidenceDigest,
    pub machine_profile: MachineProfileId,
    pub operations: Vec<RoutedOperation>,
    pub transfers: Vec<TensorTransferPlan>,
    pub backend_artifacts: BackendArtifactManifest,
    /// Single source of truth for evaluation boundaries — supersedes
    /// both SynchronizationGroup and EvaluationGroupPlan.
    pub execution_boundaries: Vec<SealedExecutionBoundaryPlan>,
    pub evidence_basis: Vec<EvidenceDigest>,
}

// ── Graph region descriptor ───────────────────────────────────────────────

/// A stable subgraph region (e.g. MLP block, attention block, decoder layer).
#[derive(Debug, Clone)]
pub struct GraphRegion {
    pub region_id: u64,
    pub family: OperationFamily,
    pub operations: Vec<OperationId>,
    pub input_tensors: Vec<TensorId>,
    pub output_tensors: Vec<TensorId>,
    pub shape_constraints: Vec<TensorShape>,
    /// Named input bindings (name → TensorHandle) for ANE/Metal compiled
    /// regions. Populated by the executor at dispatch time from the tensor
    /// registry. The ANE backend reads these to bind IOSurface-backed
    /// activation tensors as MIL graph inputs.
    pub inputs: HashMap<String, TensorHandle>,
    /// Named output bindings (name → TensorHandle). Allocated by the
    /// executor and registered in the tensor registry for future lookups.
    pub outputs: HashMap<String, TensorHandle>,
    /// Named tensor bindings for weights, KV cache buffers, etc.
    /// Populated by the executor from registered weight bindings.
    pub tensor_bindings: HashMap<String, TensorHandle>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::backend::residency;

    // ── 1. All BACKEND_* constants are distinct ──────────────────────────

    #[test]
    fn backend_constants_are_distinct() {
        let ids = [
            BACKEND_METAL.0,
            BACKEND_ACCELERATE.0,
            BACKEND_ANE.0,
            BACKEND_MLX.0,
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(
                    ids[i], ids[j],
                    "BACKEND_* constants must have distinct values; found duplicate {}",
                    ids[i]
                );
            }
        }
    }

    // ── 2. Canonical values ──────────────────────────────────────────────

    #[test]
    fn backend_constant_values() {
        assert_eq!(BACKEND_METAL.0, 0, "BACKEND_METAL must be 0");
        assert_eq!(BACKEND_ACCELERATE.0, 1, "BACKEND_ACCELERATE must be 1");
        assert_eq!(BACKEND_ANE.0, 2, "BACKEND_ANE must be 2");
        assert_eq!(BACKEND_MLX.0, 3, "BACKEND_MLX must be 3");
    }

    // ── 3. Fixtures using canonical constants ────────────────────────────

    fn make_descriptor(operation_id: u64, family: OperationFamily) -> OperationDescriptor {
        OperationDescriptor {
            operation_id: OperationId(operation_id),
            family,
            layer_index: Some(0),
            phase: Phase::Decode,
            logical_shape: LogicalShape {
                dims: vec![1, 32, 128],
            },
            physical_layout: PhysicalLayout::RowMajor,
            input_dtypes: vec![DType::F16],
            output_dtype: DType::F16,
            quantization: None,
            expected_output_shape: TensorShape {
                dims: vec![1, 32, 128],
            },
            correctness_checkpoint: CorrectnessCheckpointPolicy::None,
        }
    }

    fn make_receipt(operation_id: u64, backend_id: BackendId) -> BackendExecutionReceipt {
        BackendExecutionReceipt {
            operation_id: OperationId(operation_id),
            backend_id,
            backend_version: BackendVersion {
                backend_name: "test".into(),
                version: "0.1.0".into(),
                git_commit: None,
            },
            requested_substrate: None,
            observed_substrate: None,
            graph_build_ns: None,
            compile_ns: None,
            queue_wait_ns: None,
            submit_ns: Some(100),
            execution_ns: Some(1000),
            synchronization_ns: None,
            total_wall_ns: 1200,
            bytes_read: None,
            bytes_written: None,
            temporary_bytes: None,
            active_memory_before: None,
            active_memory_after: None,
            cache_memory_before: None,
            cache_memory_after: None,
            transfer_in_ns: None,
            transfer_out_ns: None,
            fallback_occurred: false,
        }
    }

    #[test]
    fn fixtures_use_canonical_backend_ids() {
        // OperationDescriptor fixtures — just ensure they construct without
        // panicking (BackendId is not a field on this type).
        let _metal_op = make_descriptor(1, OperationFamily::Matmul);
        let _accel_op = make_descriptor(2, OperationFamily::QuantizedMatmul);
        let _ane_op = make_descriptor(3, OperationFamily::AttentionBlock);

        // BackendExecutionReceipt fixtures — assert backend_id is canonical.
        let metal_receipt = make_receipt(1, BACKEND_METAL);
        let accel_receipt = make_receipt(2, BACKEND_ACCELERATE);
        let ane_receipt = make_receipt(3, BACKEND_ANE);
        let mlx_receipt = make_receipt(4, BACKEND_MLX);

        assert_eq!(metal_receipt.backend_id, BACKEND_METAL);
        assert_eq!(accel_receipt.backend_id, BACKEND_ACCELERATE);
        assert_eq!(ane_receipt.backend_id, BACKEND_ANE);
        assert_eq!(mlx_receipt.backend_id, BACKEND_MLX);

        // Guard: ensure we aren't using arbitrary raw integers.
        assert_ne!(metal_receipt.backend_id.0, 99);
        assert_ne!(accel_receipt.backend_id.0, 99);
    }

    // ── 4. residency::BackendId → routing::BackendId mapping ─────────────

    #[test]
    fn residency_to_routing_id_mapping() {
        // MlxMetal → BACKEND_MLX
        assert_eq!(
            residency::BackendId::MlxMetal.to_routing_id(),
            Some(BACKEND_MLX)
        );
        // Accelerate → BACKEND_ACCELERATE
        assert_eq!(
            residency::BackendId::Accelerate.to_routing_id(),
            Some(BACKEND_ACCELERATE)
        );
        // CoreAi → BACKEND_ANE
        assert_eq!(
            residency::BackendId::CoreAi.to_routing_id(),
            Some(BACKEND_ANE)
        );
        // Ane → BACKEND_ANE
        assert_eq!(residency::BackendId::Ane.to_routing_id(), Some(BACKEND_ANE));

        // Variants that map to None
        assert_eq!(residency::BackendId::CandleCpu.to_routing_id(), None);
        assert_eq!(residency::BackendId::TensixTensix.to_routing_id(), None);
        assert_eq!(residency::BackendId::IntelLevelZero.to_routing_id(), None);
        assert_eq!(residency::BackendId::IntelOpenCl.to_routing_id(), None);
        assert_eq!(residency::BackendId::HostCpu.to_routing_id(), None);
        assert_eq!(residency::BackendId::Unknown.to_routing_id(), None);
    }
}
