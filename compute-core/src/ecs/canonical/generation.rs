//! Generation types — what a compiler run produces.
//!
//! A `CimageGeneration` is the canonical output of one compiler invocation:
//! a resolved execution image with fully-specified tensor, kernel, and engram
//! bindings. Every binding carries its own receipt chain.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::ecs::canonical::identity::{
    CompilerIdentity, EngramArtifactId, EngramId, GenerationId, HardwareProfileId, LogicalTensorId,
    ModelSourceId, PhysicalSegmentId, ReceiptId, RegionId, RepresentationId, Timestamp,
};
use crate::ecs::canonical::kernel_abi::{KernelImplementationId, KernelSemanticId};
use crate::ecs::canonical::ExecutionGraph;
use crate::ecs::execution_profile::PhysicalTileLayout;
use crate::ecs::plan::CodecFamily;

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
