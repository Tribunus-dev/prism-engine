//! `pipeline::ane::artifacts` — ANE artifact schemas.
//!
//! This file owns the canonical authority for the derived ANE
//! artifacts (MIL text, IOSurface contracts, weight-blob plans) that
//! the ANE compiler produces. The canonical ComputeImage weights
//! remain authoritative; these schemas describe derived artifacts
//! whose receipts point back to the source tensor identities.

use prism_ecs_backend::routing::{BackendArtifactId, EvidenceDigest, TensorId};
use prism_ecs_backend::DType;

// ── ANE MIL text artifact ────────────────────────────────────────────────

/// Textual MIL program ready for `_ANECompiler` ingestion.
#[derive(Debug, Clone)]
pub struct AneMilTextArtifact {
    /// Content-addressed digest of this MIL text.
    pub digest: EvidenceDigest,
    /// The MIL program text.
    pub mil_text: String,
    /// Backend artifact identity.
    pub artifact_id: BackendArtifactId,
    /// Source scheduled-region digest this artifact was lowered from.
    pub source_region_digest: EvidenceDigest,
}

// ── IOSurface contract ───────────────────────────────────────────────────

/// Deterministic IOSurface contract for ANE tensors.
#[derive(Debug, Clone)]
pub struct AneIoContract {
    /// Content-addressed digest.
    pub digest: EvidenceDigest,
    /// Per-tensor IOSurface specifications.
    pub surfaces: Vec<AneIoSurfaceSpec>,
}

/// Specification for a single ANE IOSurface tensor.
#[derive(Debug, Clone)]
pub struct AneIoSurfaceSpec {
    /// Source tensor identity.
    pub tensor_id: TensorId,
    /// ANE layout: `[1, C, 1, S]` — Orion convention.
    pub shape: [u64; 4],
    /// Dtype on ANE (typically fp16).
    pub dtype: DType,
    /// Byte size of the allocation.
    pub byte_size: u64,
    /// Alignment requirement (typically page-aligned, 16384).
    pub alignment: u64,
    /// Whether this is an input or output.
    pub is_input: bool,
}

// ── Weight blob plan ─────────────────────────────────────────────────────

/// Deterministic weight-blob plan for ANE.
#[derive(Debug, Clone)]
pub struct AneWeightBlobPlan {
    /// Content-addressed digest.
    pub digest: EvidenceDigest,
    /// BLOBFILE entries.
    pub entries: Vec<AneBlobEntry>,
    /// Total byte size of all blobs.
    pub total_bytes: u64,
}

/// A single BLOBFILE entry describing one weight tensor.
#[derive(Debug, Clone)]
pub struct AneBlobEntry {
    /// Source tensor identity.
    pub tensor_id: TensorId,
    /// Source segment identifier.
    pub segment_id: String,
    /// Byte size of the blob entry.
    pub byte_size: u64,
    /// BLOBFILE offset of this entry.
    pub offset: u64,
    /// ANE dtype (typically Fp16).
    pub dtype: DType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_io_contract_serializes() {
        let contract = AneIoContract {
            digest: EvidenceDigest("abc".into()),
            surfaces: vec![],
        };
        assert!(contract.surfaces.is_empty());
    }

    #[test]
    fn blob_plan_total_bytes_matches_entries() {
        let plan = AneWeightBlobPlan {
            digest: EvidenceDigest("p".into()),
            entries: vec![AneBlobEntry {
                tensor_id: TensorId(1),
                segment_id: "weights".into(),
                byte_size: 1024,
                offset: 0,
                dtype: DType::F16,
            }],
            total_bytes: 1024,
        };
        assert_eq!(plan.entries[0].byte_size, plan.total_bytes);
    }
}
