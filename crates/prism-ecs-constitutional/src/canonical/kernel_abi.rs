//! KernelAbi, KernelPlan — backend-neutral kernel grouping and
//! ABI contracts. Authority: the kernel interface contract.
//!
//! Every kernel implementation registers against a semantic
//! contract. The ABI defines buffer bindings, constants,
//! threadgroup geometry, and dispatch policy. Handwritten and
//! generated kernels use the same semantic catalogue and the
//! same ABI.
//!
//! The `KernelSemanticId` primitive is re-exported from
//! `prism_ecs_core::canonical::kernel_abi` (the source of truth).
//! Everything else in this module is the constitutional surface
//! for kernel ABI types: the implementation identity newtype,
//! the implementation class enum, the buffer / constant /
//! threadgroup binding structs, the dispatch geometry policy,
//! the kernel plan and group types, the compiled-kernel artifact
//! and provenance types, and the helper functions for digest
//! computation, Metal code generation, and ABI validation.

use super::execution_graph::{ExecutionLane, ExecutionOp};
use super::identity::{RegionId, TargetIdentity, ToolchainIdentity};
use super::model_ir::ArchitectureId;
use super::representation::TensorRepresentation;
use serde::{Deserialize, Serialize};

/// Semantic identifier for a kernel purpose (e.g. "prism.linear.nf4.v1").
pub use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;

/// Implementation identifier for a specific kernel variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct KernelImplementationId(pub String);

/// How a kernel group is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelImplementationClass {
    /// A single persistent transformer kernel handling all layers.
    PersistentTransformer,
    /// A fused kernel handling a group of layers.
    FusedLayerGroup,
    /// One kernel per layer.
    PerLayer,
    /// A specialized projection kernel (e.g. QKV projection).
    SpecializedProjection,
    /// A primitive operation kernel (e.g. RMSNorm, RoPE).
    Primitive,
    /// CPU reference kernel (for differential testing).
    CpuReference,
    /// ANE subgraph (MIL program).
    AneSubgraph,
}

/// A buffer binding slot in a kernel's ABI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferBinding {
    /// Metal buffer index ([[buffer(N)]]).
    pub slot: u32,
    /// Logical name of the buffer.
    pub name: String,
    /// Byte size of the binding.
    pub byte_size: u64,
    /// Whether this binding is optional.
    pub optional: bool,
}

/// A function constant binding in a kernel's ABI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstantBinding {
    /// Metal constant index.
    pub index: u32,
    /// Logical name.
    pub name: String,
    /// The constant value, if fixed at compile time.
    pub default_value: Option<u32>,
}

/// A threadgroup memory allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreadgroupAllocation {
    /// Byte size of threadgroup memory.
    pub byte_size: u32,
}

/// How dispatch geometry is determined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DispatchGeometryPolicy {
    /// Fixed grid dimensions (width, height, depth).
    Fixed(u32, u32, u32),
    /// Derived from buffer sizes (typically output dimension).
    FromOutputBuffer,
    /// Dynamic via function constant.
    FromConstant,
}

/// KernelAbi — the complete interface contract for a compiled kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelAbi {
    pub version: u32,
    pub buffers: Vec<BufferBinding>,
    pub constants: Vec<ConstantBinding>,
    pub threadgroup_memory: Vec<ThreadgroupAllocation>,
    pub dispatch_geometry: DispatchGeometryPolicy,
    pub threads_per_threadgroup: (u32, u32, u32),
}

/// Specialization parameters for a kernel instance.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecializationParameters {
    pub tile_m: Option<u32>,
    pub tile_k: Option<u32>,
    pub tile_n: Option<u32>,
    pub group_size: Option<u32>,
    pub metadata_layout: Option<String>,
}

/// A semantic kernel registration (what it does, contractually).
#[derive(Debug, Clone)]
pub struct KernelSemanticRegistration {
    pub semantic_id: KernelSemanticId,
    pub version: String,
    pub description: String,
}

/// A Metal-specific kernel implementation registration.
#[derive(Debug, Clone)]
pub struct MetalImplementationRegistration {
    pub semantic_id: KernelSemanticId,
    pub implementation_id: KernelImplementationId,
    pub supported_architectures: Vec<ArchitectureId>,
    pub supported_representations: Vec<TensorRepresentation>,
    pub abi: KernelAbi,
    /// Path to the .metal source file (relative to crate root). None for generated kernels.
    pub source_path: Option<String>,
    /// Entry point function name in the .metal file.
    pub source_entry_point: Option<String>,
}

/// A group of execution operations that are compiled together into one kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelGroup {
    pub semantic_id: KernelSemanticId,
    pub implementation_class: KernelImplementationClass,
    pub operations: Vec<ExecutionOp>,
    pub specialization: SpecializationParameters,
    pub abi: KernelAbi,
    pub source_region: RegionId,
    pub target_lane: ExecutionLane,
}

/// KernelPlan — the complete plan for all kernels needed to execute a model.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelPlan {
    pub groups: Vec<KernelGroup>,
}

impl KernelPlan {
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn groups_for_lane(&self, lane: ExecutionLane) -> Vec<&KernelGroup> {
        self.groups
            .iter()
            .filter(|g| g.target_lane == lane)
            .collect()
    }
}

/// A compiled kernel artifact (e.g. a .metallib with metadata).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledKernelArtifact {
    pub implementation_id: KernelImplementationId,
    pub semantic_id: KernelSemanticId,
    pub compiled_bytes: Vec<u8>,
    pub sha256: String,
    pub entry_point: String,
    pub abi: KernelAbi,
}

/// Provenance chain for a compiled artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactProvenance {
    pub semantic_id: KernelSemanticId,
    pub implementation_id: KernelImplementationId,
    pub source_digest: Option<String>,
    pub mlir_digest: Option<String>,
    pub abi_digest: [u8; 32],
    pub toolchain: ToolchainIdentity,
    pub target: TargetIdentity,
    pub compiled_byte_digest: String,
}

impl ArtifactProvenance {
    pub fn new(
        artifact: &CompiledKernelArtifact,
        source_digest: Option<String>,
        mlir_digest: Option<String>,
        toolchain: ToolchainIdentity,
        target: TargetIdentity,
    ) -> Self {
        let abi_digest = compute_abi_digest(&artifact.abi);
        Self {
            semantic_id: artifact.semantic_id.clone(),
            implementation_id: artifact.implementation_id.clone(),
            source_digest,
            mlir_digest,
            abi_digest,
            toolchain,
            target,
            compiled_byte_digest: artifact.sha256.clone(),
        }
    }
}

/// Generate Metal `#define` constants for buffer indices from a KernelAbi.
///
/// Produces lines like:
/// ```metal
/// #define SLOT_INPUT 0
/// #define SLOT_WEIGHTS 1
/// #define SLOT_OUTPUT 4
/// ```
pub fn generate_buffer_constants(abi: &KernelAbi) -> String {
    let mut out = String::new();
    out.push_str("// Buffer slot constants (generated from KernelAbi)\n");
    for binding in &abi.buffers {
        let name = binding.name.to_uppercase().replace(' ', "_");
        out.push_str(&format!("#define SLOT_{} {}\n", name, binding.slot));
    }
    out
}

/// Generate Metal function constant indices from a KernelAbi.
pub fn generate_constant_indices(abi: &KernelAbi) -> String {
    let mut out = String::new();
    out.push_str("// Function constant indices (generated from KernelAbi)\n");
    for constant in &abi.constants {
        let name = constant.name.to_uppercase().replace(' ', "_");
        out.push_str(&format!("#define CONSTANT_{} {}\n", name, constant.index));
    }
    out
}

/// Compute dispatch geometry from a KernelAbi and output size.
///
/// Returns (grid_x, grid_y, grid_z) appropriate for the ABI's
/// dispatch geometry policy.
pub fn compute_dispatch_geometry(abi: &KernelAbi, output_elements: u64) -> (u32, u32, u32) {
    match abi.dispatch_geometry {
        DispatchGeometryPolicy::Fixed(w, h, d) => (w, h, d),
        DispatchGeometryPolicy::FromOutputBuffer => {
            let tgs = abi.threads_per_threadgroup.0.max(1) as u64;
            let grid_x = ((output_elements + tgs - 1) / tgs) as u32;
            (grid_x.max(1), 1, 1)
        }
        DispatchGeometryPolicy::FromConstant => (1, 1, 1),
    }
}

/// Validate that actual buffer bindings match the declared ABI.
///
/// Returns Ok(()) if every required buffer slot in the ABI has a
/// corresponding entry in `actual_slots`, and no extra slots exist.
pub fn validate_bindings(abi: &KernelAbi, actual_slots: &[u32]) -> Result<(), String> {
    use std::collections::BTreeSet;
    let required: BTreeSet<u32> = abi
        .buffers
        .iter()
        .filter(|b| !b.optional)
        .map(|b| b.slot)
        .collect();
    let provided: BTreeSet<u32> = actual_slots.iter().copied().collect();

    for slot in &required {
        if !provided.contains(slot) {
            return Err(format!(
                "ABI requires buffer slot {} but it is not provided",
                slot
            ));
        }
    }
    for slot in &provided {
        if !required.contains(slot) {
            return Err(format!(
                "provided buffer slot {} is not declared in ABI",
                slot
            ));
        }
    }
    Ok(())
}

/// Compute a deterministic ABI digest (for caching and equality checks).
pub fn compute_abi_digest(abi: &KernelAbi) -> [u8; 32] {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(b"prism.kernel.abi.v1");
    h.update(&abi.version.to_le_bytes());
    for buf in &abi.buffers {
        h.update(&buf.slot.to_le_bytes());
        h.update(buf.name.as_bytes());
        h.update(&buf.byte_size.to_le_bytes());
    }
    for c in &abi.constants {
        h.update(&c.index.to_le_bytes());
        h.update(c.name.as_bytes());
    }
    h.update(&[abi.threads_per_threadgroup.0 as u8]);
    h.finalize().into()
}

/// Validate that a compiled kernel artifact matches its declared ABI.
pub fn validate_artifact_abi(artifact: &CompiledKernelArtifact) -> Result<(), String> {
    if artifact.abi.buffers.is_empty() && !artifact.compiled_bytes.is_empty() {
        return Err("compiled artifact with no buffer bindings declared in ABI".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::identity::RegionId;

    fn sample_abi() -> KernelAbi {
        KernelAbi {
            version: 1,
            buffers: vec![
                BufferBinding {
                    slot: 0,
                    name: "input".into(),
                    byte_size: 1024,
                    optional: false,
                },
                BufferBinding {
                    slot: 1,
                    name: "weights".into(),
                    byte_size: 4096,
                    optional: false,
                },
                BufferBinding {
                    slot: 2,
                    name: "scratch".into(),
                    byte_size: 0,
                    optional: true,
                },
            ],
            constants: vec![ConstantBinding {
                index: 0,
                name: "tile_m".into(),
                default_value: Some(64),
            }],
            threadgroup_memory: vec![ThreadgroupAllocation { byte_size: 256 }],
            dispatch_geometry: DispatchGeometryPolicy::FromOutputBuffer,
            threads_per_threadgroup: (64, 1, 1),
        }
    }

    #[test]
    fn buffer_constants_match_slot_index() {
        let s = generate_buffer_constants(&sample_abi());
        assert!(s.contains("#define SLOT_INPUT 0"), "got:\n{}", s);
        assert!(s.contains("#define SLOT_WEIGHTS 1"), "got:\n{}", s);
        assert!(s.contains("#define SLOT_SCRATCH 2"), "got:\n{}", s);
    }

    #[test]
    fn constant_indices_match_const_index() {
        let s = generate_constant_indices(&sample_abi());
        assert!(s.contains("#define CONSTANT_TILE_M 0"), "got:\n{}", s);
    }

    #[test]
    fn dispatch_geometry_from_output_uses_threadgroup_size() {
        let abi = sample_abi();
        // threads_per_threadgroup.0 = 64
        // 1024 elements / 64 = 16 grid_x
        let (x, _y, _z) = compute_dispatch_geometry(&abi, 1024);
        assert_eq!(x, 16);
    }

    #[test]
    fn dispatch_geometry_fixed_returns_inline_dims() {
        let mut abi = sample_abi();
        abi.dispatch_geometry = DispatchGeometryPolicy::Fixed(7, 3, 2);
        let (x, y, z) = compute_dispatch_geometry(&abi, 1024);
        assert_eq!((x, y, z), (7, 3, 2));
    }

    #[test]
    fn validate_bindings_rejects_missing_required_slot() {
        let abi = sample_abi();
        // Only slot 0 provided; slot 1 (weights) is required.
        let err = validate_bindings(&abi, &[0]).expect_err("should fail");
        assert!(err.contains("slot 1"), "got: {}", err);
    }

    #[test]
    fn validate_bindings_rejects_extra_slot() {
        let abi = sample_abi();
        // Provide slot 99 which is not in the ABI.
        let err = validate_bindings(&abi, &[0, 1, 99]).expect_err("should fail");
        assert!(err.contains("99"), "got: {}", err);
    }

    #[test]
    fn validate_bindings_accepts_full_required_set() {
        let abi = sample_abi();
        // slot 2 is optional; required is [0, 1]. Provide just [0, 1].
        assert!(validate_bindings(&abi, &[0, 1]).is_ok());
    }

    #[test]
    fn abi_digest_is_deterministic_and_distinct() {
        let abi_a = sample_abi();
        let mut abi_b = sample_abi();
        abi_b.version = 2;
        assert_eq!(compute_abi_digest(&abi_a), compute_abi_digest(&abi_a));
        assert_ne!(compute_abi_digest(&abi_a), compute_abi_digest(&abi_b));
    }

    #[test]
    fn kernel_plan_groups_for_lane_filters_correctly() {
        let plan = KernelPlan {
            groups: vec![KernelGroup {
                semantic_id: KernelSemanticId("a".into()),
                implementation_class: KernelImplementationClass::PerLayer,
                operations: vec![],
                specialization: SpecializationParameters {
                    tile_m: None,
                    tile_k: None,
                    tile_n: None,
                    group_size: None,
                    metadata_layout: None,
                },
                abi: sample_abi(),
                source_region: RegionId("r".into()),
                target_lane: ExecutionLane::Cpu,
            }],
        };
        assert_eq!(plan.group_count(), 1);
        assert_eq!(plan.groups_for_lane(ExecutionLane::Cpu).len(), 1);
        assert_eq!(plan.groups_for_lane(ExecutionLane::MetalGpu).len(), 0);
    }

    #[test]
    fn validate_artifact_abi_rejects_empty_buffers_with_bytes() {
        let mut abi = sample_abi();
        abi.buffers.clear();
        let artifact = CompiledKernelArtifact {
            implementation_id: KernelImplementationId("impl-1".into()),
            semantic_id: KernelSemanticId("k-1".into()),
            compiled_bytes: vec![0u8; 8],
            sha256: "deadbeef".into(),
            entry_point: "main".into(),
            abi,
        };
        assert!(validate_artifact_abi(&artifact).is_err());
    }
}
