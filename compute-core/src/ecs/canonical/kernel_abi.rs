//! KernelAbi, KernelPlan — backend-neutral kernel grouping and ABI contracts.
//!
//! Every kernel implementation registers against a semantic contract.
//! The ABI defines buffer bindings, constants, threadgroup geometry, and
//! dispatch policy. Handwritten and generated kernels use the same
//! semantic catalogue and the same ABI.

use super::execution_graph::{ExecutionLane, ExecutionOp, RegionId};
use super::model_ir::ArchitectureId;
use super::representation::TensorRepresentation;
use serde::{Deserialize, Serialize};

/// Semantic identifier for a kernel purpose (e.g. "prism.linear.nf4.v1").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct KernelSemanticId(pub String);

/// Implementation identifier for a specific kernel variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct KernelImplementationId(pub String);

/// How a kernel group is implemented.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantBinding {
    /// Metal constant index.
    pub index: u32,
    /// Logical name.
    pub name: String,
    /// The constant value, if fixed at compile time.
    pub default_value: Option<u32>,
}

/// A threadgroup memory allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadgroupAllocation {
    /// Byte size of threadgroup memory.
    pub byte_size: u32,
}

/// How dispatch geometry is determined.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchGeometryPolicy {
    /// Fixed grid dimensions (width, height, depth).
    Fixed(u32, u32, u32),
    /// Derived from buffer sizes (typically output dimension).
    FromOutputBuffer,
    /// Dynamic via function constant.
    FromConstant,
}

/// KernelAbi — the complete interface contract for a compiled kernel.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledKernelArtifact {
    pub implementation_id: KernelImplementationId,
    pub semantic_id: KernelSemanticId,
    pub compiled_bytes: Vec<u8>,
    pub sha256: String,
    pub entry_point: String,
    pub abi: KernelAbi,
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
    let required: std::collections::HashSet<u32> = abi
        .buffers
        .iter()
        .filter(|b| !b.optional)
        .map(|b| b.slot)
        .collect();
    let provided: std::collections::HashSet<u32> = actual_slots.iter().copied().collect();

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
