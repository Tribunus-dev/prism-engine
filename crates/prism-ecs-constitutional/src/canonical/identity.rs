//! Canonical identity primitives — generation, candidate, engram,
//! hardware, model, compiler, corpus, and region. Authority: the
//! type system.
//!
//! Every tensor, kernel, engram, candidate, and compiler artifact
//! in the system is uniquely identified by one of these newtype
//! wrappers. The identity hierarchy mirrors the provenance chain:
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
//! `CorpusId`, `CompilerIdentity`, `HardwareProfileId`, and
//! `CandidateId` provide cross-cutting identity for corpora,
//! compilers, hardware targets, and evolutionary candidates
//! respectively.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Re-exported from prism-ecs-core — the source-of-truth identity
// types live in the core crate. The constitutional canonical
// surface re-exports them so the compiler pipeline has a single,
// stable import path: `prism_ecs_constitutional::canonical::*`.
// ---------------------------------------------------------------------------
pub use prism_ecs_core::identity::{
    CompilerIdentity, GenerationId, HardwareProfileId, ModelSourceId, ReceiptId, Timestamp,
};

// ---------------------------------------------------------------------------
// Provenance-chain identity types (re-exported from prism-ecs-ir).
// These are the identity primitives that thread through the
// `Source → ModelIr → RepresentationPlan → ExecutionGraph →
//  KernelPlan → CompiledKernelArtifact → CimageBuildInput`
// ownership chain. They live in `prism_ecs_ir::cimage_types` and
// are re-exported here so the compiler pipeline never has to
// import from the IR crate directly.
// ---------------------------------------------------------------------------

/// Stable semantic identity independent of physical layout.
pub use prism_ecs_ir::cimage_types::LogicalTensorId;

/// Codec, grouping, scale structure, residual policy, and generation.
pub use prism_ecs_ir::cimage_types::RepresentationId;

/// Content digest of packed tensor bytes.
pub use prism_ecs_ir::cimage_types::PhysicalSegmentId;

/// Stable operation contract such as NF4 Tile640 GEMV.
pub use prism_ecs_core::canonical::kernel_abi::KernelSemanticId;

/// Exact source, parameters, toolchain, and target-hardware implementation.
pub use prism_ecs_ir::cimage_types::KernelImplementationId;

/// Stable logical engram identity.
pub use prism_ecs_ir::cimage_types::EngramId;

/// Digest of canonical executable engram bytes.
pub use prism_ecs_ir::cimage_types::EngramArtifactId;

/// Region identifier in the execution graph (string-based for identity).
pub use prism_ecs_ir::cimage_types::RegionId;

/// Tensor shape — dimensions vector.
pub use prism_ecs_ir::cimage_types::TensorShape;

// ---------------------------------------------------------------------------
// Constitutional identity types — these are introduced by the
// canonical surface itself. They are authority-bearing newtypes,
// so the underlying `String` payloads are not exposed in the
// public API beyond the construction/destructure boundary.
// ---------------------------------------------------------------------------

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

/// Candidate identifier for evolution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_chain_re_exports_compile() {
        // Touch every re-exported identity type so the constitutional
        // surface is statically known to expose the full authority
        // chain. A missing `pub use` in the surface would fail this
        // test.
        let _src = ModelSourceId(String::from("src-1"));
        let _tensor = LogicalTensorId(String::from("tensor-1"));
        let _repr = RepresentationId(String::from("repr-1"));
        let _seg = PhysicalSegmentId(String::from("seg-1"));
        let _kernel = KernelSemanticId(String::from("kernel-1"));
        let _impl = KernelImplementationId(String::from("impl-1"));
        let _engram = EngramId(String::from("engram-1"));
        let _engram_artifact = EngramArtifactId(String::from("engram-art-1"));
        let _gen = GenerationId(String::from("gen-1"));
        let _receipt = ReceiptId(String::from("receipt-1"));
        let _region = RegionId(String::from("region-1"));
        let _corpus = CorpusId(String::from("corpus-1"));
        let _cand = CandidateId(String::from("cand-1"));
        let _hw = HardwareProfileId(String::from("hw-1"));
        let _ts = Timestamp(String::from("2026-07-28T00:00:00Z"));
        let _compiler = CompilerIdentity {
            name: "prism".into(),
            version: "0.0.1".into(),
            build_hash: None,
            build_timestamp: None,
        };
        let _tool = ToolchainIdentity {
            name: "metal".into(),
            version: "15.0".into(),
            target_triple: "arm64-apple-macos".into(),
        };
        let _target = TargetIdentity {
            name: "m2".into(),
            arch: "arm64".into(),
            features: vec!["neon".into()],
        };
        let _shape = TensorShape { dims: vec![2, 4] };
    }

    #[test]
    fn corpus_id_ordering_is_string_order() {
        // CorpusId is `Ord` so the constitutional surface can use it
        // as a BTreeMap key in the receipt store and replay manifest.
        let mut ids = vec![
            CorpusId("gamma".into()),
            CorpusId("alpha".into()),
            CorpusId("beta".into()),
        ];
        ids.sort();
        assert_eq!(ids[0].0, "alpha");
        assert_eq!(ids[2].0, "gamma");
    }
}
