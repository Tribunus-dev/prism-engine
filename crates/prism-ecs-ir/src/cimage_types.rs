//! CImage format types — execution graph, generation, header, and plan types.
//!
//! These types define the execution-oriented representation of a compiled
//! model image. They are shared between the compiler pipeline, the cimage
//! packer, the admission pipeline, and the serving runtime.
//!
//! Origin: migrated from compute-core (prism-engine monolith) into this
//! canonical crate so that prism-ecs-server can break its dependency on
//! compute-core.

use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;
use prism_ecs_core::identity::{
    CompilerIdentity, GenerationId, HardwareProfileId, ModelSourceId, ReceiptId, Timestamp,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

// =============================================================================
// CImage constants
// =============================================================================

/// Magic bytes for cimage V0 files.
pub const CIMAGE_MAGIC: [u8; 8] = *b"PRISMCIM";

/// Supported format version.
pub const CIMAGE_FORMAT_VERSION: u32 = 0;

// =============================================================================
// CImage header / footer types
// =============================================================================

/// Fixed-size V0 header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageHeaderV0 {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub header_len: u64,
    pub manifest_offset: u64,
    pub manifest_len: u64,
    pub payload_directory_offset: u64,
    pub payload_directory_len: u64,
    pub receipt_directory_offset: u64,
    pub receipt_directory_len: u64,
    pub payload_blob_offset: u64,
    pub payload_blob_len: u64,
    pub footer_offset: u64,
}

impl CImageHeaderV0 {
    pub fn new() -> Self {
        Self {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: std::mem::size_of::<CImageHeaderV0>() as u64,
            manifest_offset: 0,
            manifest_len: 0,
            payload_directory_offset: 0,
            payload_directory_len: 0,
            receipt_directory_offset: 0,
            receipt_directory_len: 0,
            payload_blob_offset: 0,
            payload_blob_len: 0,
            footer_offset: 0,
        }
    }

    pub fn validate_magic(&self) -> Result<(), [u8; 8]> {
        if self.magic == CIMAGE_MAGIC {
            Ok(())
        } else {
            Err(self.magic)
        }
    }

    pub fn supports_format(&self) -> bool {
        self.format_version == CIMAGE_FORMAT_VERSION
    }
}

impl Default for CImageHeaderV0 {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size V0 footer.
///
/// Binds the whole file together with a recursive digest:
/// `cimage_sha256_without_footer` covers bytes [0, footer_offset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageFooterV0 {
    pub manifest_sha256: String,
    pub payload_directory_sha256: String,
    pub receipt_directory_sha256: String,
    pub payload_blob_sha256: String,
    pub cimage_sha256_without_footer: String,
}

impl CImageFooterV0 {
    pub fn new() -> Self {
        Self {
            manifest_sha256: String::new(),
            payload_directory_sha256: String::new(),
            receipt_directory_sha256: String::new(),
            payload_blob_sha256: String::new(),
            cimage_sha256_without_footer: String::new(),
        }
    }
}

impl Default for CImageFooterV0 {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ExecutionGraph
// =============================================================================

/// Unique identifier for a region within an execution graph (numeric).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphRegionId(pub usize);

/// Identifies which execution lane (backend) a region targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionLane {
    /// Apple GPU via Metal.
    MetalGpu,
    /// Apple Neural Engine.
    Ane,
    /// CPU fallback.
    Cpu,
    /// AMD GPU via ROCm.
    Rocm,
    /// Intel GPU via Level Zero.
    LevelZero,
    /// MLX framework (Apple GPU).
    Mlx,
}

/// Unique identifier for a tensor within a ModelIr.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorId(pub usize);

/// A value that flows between execution operations (a buffer reference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferValue {
    pub name: String,
    pub byte_size: u64,
    pub tensor_id: Option<TensorId>,
}

/// A single executable operation within a region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOp {
    pub name: String,
    pub kind: ExecutionOpKind,
    pub inputs: Vec<BufferValue>,
    pub outputs: Vec<BufferValue>,
    pub attributes: HashMap<String, String>,
}

/// Kinds of executable operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionOpKind {
    RmsNorm,
    LayerNorm,
    Linear,
    QuantizedLinear,
    Attention,
    RoPE,
    SiLU,
    Mul,
    Add,
    Softmax,
    RotaryEmbedding,
    Gather,
    ScalarAdd,
    Scale,
    Fp32Dequant,
    Nf4Dequant,
    Int8Dequant,
    TernaryDequant,
    Other(String),
}

/// Constraints that guide fusion decisions for a region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionConstraints {
    /// Maximum operations that can be fused into one kernel.
    pub max_fused_ops: Option<usize>,
    /// Whether the region must be a single fused kernel.
    pub force_fused: bool,
    /// Whether the region must remain unfused (individual kernels).
    pub force_unfused: bool,
}

/// A single execution region — a group of operations that execute together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionRegion {
    pub id: GraphRegionId,
    pub name: String,
    pub operations: Vec<ExecutionOp>,
    pub target_lane: ExecutionLane,
    pub fusion_constraints: FusionConstraints,
    pub inputs: Vec<BufferValue>,
    pub outputs: Vec<BufferValue>,
}

/// A directed edge between execution regions (data dependency).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionEdge {
    pub source_region: GraphRegionId,
    pub source_output: String,
    pub target_region: GraphRegionId,
    pub target_input: String,
}

/// Plan for runtime state (KV cache, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeStatePlan {
    pub max_context_tokens: usize,
    pub kv_cache_bytes_per_token: u64,
    pub total_kv_cache_bytes: u64,
}

/// Plan for memory allocation across regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub total_activation_bytes: u64,
    pub total_weight_bytes: u64,
    pub arena_region_count: usize,
}

/// ExecutionGraph — the complete execution-oriented representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub regions: Vec<ExecutionRegion>,
    pub edges: Vec<ExecutionEdge>,
    pub state: RuntimeStatePlan,
    pub memory: MemoryPlan,
}

impl ExecutionGraph {
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}

// =============================================================================
// Identity types (not yet migrated to prism-ecs-core)
// =============================================================================

/// Stable semantic identity independent of physical layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalTensorId(pub String);

/// Codec, grouping, scale structure, residual policy, and generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RepresentationId(pub String);

/// Content digest of packed tensor bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalSegmentId(pub String);

/// Exact source, parameters, toolchain, and target-hardware implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct KernelImplementationId(pub String);

/// Stable logical engram identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngramId(pub String);

/// Digest of canonical executable engram bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngramArtifactId(pub String);

/// ISO 8601 timestamp wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

/// Region identifier (string-based) for generation identity bindings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub String);

/// Tensor shape — dimensions vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

// =============================================================================
// Codec family
// =============================================================================

/// Family of quantization codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum CodecFamily {
    Nf4,
    Int8,
    Fp16,
    RawF32,
    SymInt4,
    Ternary,
    Ternary1_58,
    Mixed,
    Q8_0,
    #[allow(non_camel_case_types)]
    Q4_K,
    #[allow(non_camel_case_types)]
    Q2_K,
    #[allow(non_camel_case_types)]
    IQ2_XXS,
}

impl Default for CodecFamily {
    fn default() -> Self {
        Self::RawF32
    }
}

// =============================================================================
// Execution profile / tile layout types
// =============================================================================

/// Logical shape of a single tile in rows and columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileShape {
    pub rows: u32,
    pub cols: u32,
}

impl TileShape {
    pub const fn tile640() -> Self {
        Self {
            rows: 640,
            cols: 640,
        }
    }

    pub const fn elements(&self) -> u32 {
        self.rows * self.cols
    }
}

/// A family of tiles sharing a shape and default group sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileFamily {
    pub name: String,
    pub tile_shape: TileShape,
    pub default_group_sizes: Vec<u32>,
}

impl TileFamily {
    pub fn tile640() -> Self {
        Self {
            name: "Tile640".into(),
            tile_shape: TileShape::tile640(),
            default_group_sizes: vec![32, 64, 128],
        }
    }
}

/// Whether the storage is row-major or column-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageOrder {
    RowMajor,
    ColumnMajor,
}

/// How groups of values are arranged within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupAxis {
    /// Groups are contiguous in storage order.
    PackedContiguous,
    /// Groups span output-index space.
    OutputAxis,
    /// Groups span input-index space.
    InputAxis,
    /// Groups do not cross tile boundaries.
    TileLocal,
}

/// How metadata (scales, offsets) is laid out relative to tile data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataLayout {
    AdjacentTile,
    SeparatedManifest,
    Interleaved,
}

/// Concrete tile layout describing how a tensor is physically stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTileLayout {
    pub format: String,
    pub tile_family: TileFamily,
    pub logical_shape: [u32; 2],
    pub storage_order: StorageOrder,
    pub tile_shape: TileShape,
    pub group_size: u32,
    pub group_axis: GroupAxis,
    pub metadata_layout: MetadataLayout,
    pub padding_policy: String,
    pub alignment_bytes: u32,
    pub interleave: String,
}

impl Default for PhysicalTileLayout {
    fn default() -> Self {
        Self {
            format: "NF4".into(),
            tile_family: TileFamily::tile640(),
            logical_shape: [0, 0],
            storage_order: StorageOrder::RowMajor,
            tile_shape: TileShape::tile640(),
            group_size: 32,
            group_axis: GroupAxis::PackedContiguous,
            metadata_layout: MetadataLayout::AdjacentTile,
            padding_policy: "ZeroPadToTile".into(),
            alignment_bytes: 256,
            interleave: "None".into(),
        }
    }
}

// =============================================================================
// CimageGeneration — the canonical output of one compiler invocation
// =============================================================================

/// A tensor representation binding within a generation.
///
/// Describes exactly how a logical tensor is quantized, tiled, and stored
/// in the compiled artifact, with optional provenance from a source
/// representation and an acceptance receipt from the admission pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentationBinding {
    /// The representation identifier for this binding.
    pub representation_id: RepresentationId,
    /// The codec family used for this tensor.
    pub codec: CodecFamily,
    /// The physical tile layout describing how the tensor is stored.
    pub layout: PhysicalTileLayout,
    /// Primary segment containing the main quantized data.
    pub primary_segment: PhysicalSegmentId,
    /// Scale factor segments.
    pub scale_segments: Vec<PhysicalSegmentId>,
    /// Residual/remainder segments.
    pub residual_segments: Vec<PhysicalSegmentId>,
    /// Optional source representation this was derived from.
    pub source_representation: Option<RepresentationId>,
    /// Acceptance receipt from the admission/calibration pipeline.
    pub acceptance_receipt: ReceiptId,
}

/// An engram binding within a generation.
///
/// Links a logical engram to its artifact bytes and defines where in
/// the execution graph it should be inserted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramBinding {
    /// The logical engram identifier.
    pub engram_id: EngramId,
    /// The artifact identifier pointing to the canonical executable bytes.
    pub artifact_id: EngramArtifactId,
    /// Whether this engram is enabled for insertion.
    pub enabled: bool,
    /// The region in the execution graph where this engram is inserted.
    pub insertion_region: RegionId,
}

/// A resolved execution image generation.
///
/// Every binding is fully resolved — no late-stage ambiguity about which
/// representation, kernel, or engram goes where. The receipt root chains
/// all downstream evidence back to this generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CimageGeneration {
    /// Unique generation identifier (digest of parent + change set).
    pub generation_id: GenerationId,
    /// Optional parent generation that this one evolved from.
    pub parent_generation: Option<GenerationId>,
    /// Base model this generation was compiled from.
    pub base_model: ModelSourceId,
    /// Identity of the compiler that produced this generation.
    pub compiler_identity: CompilerIdentity,
    /// Target hardware profile.
    pub hardware_profile: HardwareProfileId,
    /// Tensor → representation binding map.
    pub tensor_bindings: BTreeMap<LogicalTensorId, RepresentationBinding>,
    /// Kernel semantic → implementation binding map.
    pub kernel_bindings: BTreeMap<KernelSemanticId, KernelImplementationId>,
    /// Engram → engram binding map.
    pub engram_bindings: BTreeMap<EngramId, EngramBinding>,
    /// The complete execution graph describing execution order and regions.
    pub execution_graph: ExecutionGraph,
    /// Root receipt chaining all downstream evidence.
    pub receipt_root: ReceiptId,
    /// When this generation was created.
    pub created_at: Timestamp,
}
