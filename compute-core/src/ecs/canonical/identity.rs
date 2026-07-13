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
// Provenance-chain identity types
// ---------------------------------------------------------------------------

/// Digest of the original model source and relevant sidecars.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelSourceId(pub String);

/// Stable semantic identity independent of physical layout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalTensorId(pub String);

/// Codec, grouping, scale structure, residual policy, and generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RepresentationId(pub String);

/// Content digest of packed tensor bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PhysicalSegmentId(pub String);

/// Stable operation contract such as NF4 Tile640 GEMV.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct KernelSemanticId(pub String);

/// Exact source, parameters, toolchain, and target-hardware implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct KernelImplementationId(pub String);

/// Stable logical engram identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngramId(pub String);

/// Digest of canonical executable engram bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EngramArtifactId(pub String);

/// Digest of parent generation plus the complete promoted change set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenerationId(pub String);

/// Digest of canonical receipt content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ReceiptId(pub String);

/// Digest of the ordered training, calibration, and holdout manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CorpusId(pub String);

// ---------------------------------------------------------------------------
// Cross-cutting identity types
// ---------------------------------------------------------------------------

/// Compiler identity — name, version, and build metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompilerIdentity {
    pub name: String,
    pub version: String,
    pub build_hash: Option<String>,
    pub build_timestamp: Option<String>,
}

/// Hardware profile identifier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HardwareProfileId(pub String);

/// ISO 8601 timestamp wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub String);

/// Candidate identifier for evolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

// ---------------------------------------------------------------------------
// Auxiliary types referenced by EngramInsertionContract et al.
// ---------------------------------------------------------------------------

/// Region identifier in the execution graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RegionId(pub String);

/// Tensor shape — dimensions vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}
