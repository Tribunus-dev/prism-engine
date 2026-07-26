//! Identity types for the canonical domain model.
//!
//! Every tensor, kernel, engram, candidate, and compiler artifact in the
//! system is uniquely identified by one of these newtype wrappers. The
//! identity hierarchy mirrors the provenance chain:
//!
//!   Source (ModelSourceId)
//!     → logical tensors (LogicalTensorId)
//!       → quantized representations (RepresentationId)
//!         → packed segments (PhysicalSegmentId)
//!           → kernel semantics (KernelSemanticId)
//!             → concrete implementations (KernelImplementationId)
//!               → engrams (EngramId → EngramArtifactId)
//!                 → generations (GenerationId)
//!                   → receipts (ReceiptId)
//!
//! CorpusId, CompilerIdentity, HardwareProfileId, and CandidateId provide
//! cross-cutting identity for corpora, compilers, hardware targets, and
//! evolutionary candidates respectively.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Re-exported from prism-ecs-core (phase 1 of compute-core dependency removal)
// ---------------------------------------------------------------------------
pub use prism_ecs_core::identity::{
    CompilerIdentity, GenerationId, HardwareProfileId, ModelSourceId, ReceiptId, Timestamp,
};

// ---------------------------------------------------------------------------
// Provenance-chain identity types
// ---------------------------------------------------------------------------

/// Stable semantic identity independent of physical layout.
pub use prism_ecs_ir::cimage_types::LogicalTensorId;

/// Codec, grouping, scale structure, residual policy, and generation.
pub use prism_ecs_ir::cimage_types::RepresentationId;

/// Content digest of packed tensor bytes.
pub use prism_ecs_ir::cimage_types::PhysicalSegmentId;

/// Stable operation contract such as NF4 Tile640 GEMV.
/// (Also re-exported via prism_ecs_core::canonical::kernel_abi.)
pub use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;

/// Exact source, parameters, toolchain, and target-hardware implementation.
pub use prism_ecs_ir::cimage_types::KernelImplementationId;

/// Stable logical engram identity.
pub use prism_ecs_ir::cimage_types::EngramId;

/// Digest of canonical executable engram bytes.
pub use prism_ecs_ir::cimage_types::EngramArtifactId;

/// Toolchain identity — name, version, and target triple.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ToolchainIdentity {
    pub name: String,
    pub version: String,
    pub target_triple: String,
}

/// Target hardware identity — arch and feature flags.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TargetIdentity {
    pub name: String,
    pub arch: String,
    pub features: Vec<String>,
}

/// Digest of the ordered training, calibration, and holdout manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CorpusId(pub String);

// ---------------------------------------------------------------------------
// Cross-cutting identity types
// ---------------------------------------------------------------------------

/// Candidate identifier for evolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

// ---------------------------------------------------------------------------
// Auxiliary types referenced by EngramInsertionContract et al.
// ---------------------------------------------------------------------------

/// Region identifier in the execution graph (string-based for identity).
pub use prism_ecs_ir::cimage_types::RegionId;

/// Tensor shape — dimensions vector.
pub use prism_ecs_ir::cimage_types::TensorShape;
