//! `pipeline::lowering::receipts` — Core ML lowering pipeline receipts.
//!
//! This file owns the canonical authority for the immutable stage
//! receipts produced by the Core ML lowering pipeline: the per-stage
//! [`MilLoweringReceipt`] and [`PackageReceipt`], the per-op
//! [`OpLegalityEntry`], and the aggregate [`CoreAiGateReport`].

use prism_ecs_backend::routing::{BackendId, EvidenceDigest, OperationId, TensorId};

use super::params::{CoreAiTarget, LoweringDiagnostic, Opcode, PrecisionPolicy, ShapePolicy};

// ── MIL lowering receipt ──────────────────────────────────────────────────

/// Receipt from stage 1 (lowering): MIL production.
#[derive(Debug, Clone)]
pub struct MilLoweringReceipt {
    /// Digest of the produced MIL program.
    pub program_digest: EvidenceDigest,
    /// Number of operations emitted.
    pub op_count: usize,
    /// Number of constants registered (after dedup).
    pub constant_count: usize,
    /// Per-op legality results.
    pub op_legality: Vec<OpLegalityEntry>,
    /// Warnings accumulated during lowering.
    pub warnings: Vec<LoweringDiagnostic>,
    /// Opset used.
    pub opset: String,
}

/// Legality result for one scheduled operation during lowering.
#[derive(Debug, Clone)]
pub struct OpLegalityEntry {
    /// Operation identifier.
    pub op_id: OperationId,
    /// Operation opcode.
    pub opcode: Opcode,
    /// Whether the op is legal for this lowering target.
    pub legal: bool,
    /// Per-op diagnostics.
    pub diagnostics: Vec<LoweringDiagnostic>,
}

// ── Package receipt ──────────────────────────────────────────────────────

/// Receipt from stage 2 (packaging): deterministic materialization.
#[derive(Debug, Clone)]
pub struct PackageReceipt {
    /// SHA-256 of the source .mlpackage directory.
    pub source_package_sha256: String,
    /// SHA-256 of the manifest file.
    pub manifest_sha256: String,
    /// Number of weight files written.
    pub weight_file_count: usize,
    /// Hash of each weight file (for dedup verification).
    pub weight_file_hashes: Vec<String>,
    /// Total weight bytes written.
    pub total_weight_bytes: u64,
}

// ── Qualification report ──────────────────────────────────────────────────

/// Aggregate qualification report for the Core ML lowering gate.
#[derive(Debug, Clone)]
pub struct CoreAiGateReport {
    /// Target Core ML version.
    pub target: CoreAiTarget,
    /// Backend identifier (e.g. Core AI / ANE).
    pub backend_id: BackendId,
    /// Precision policy.
    pub precision: PrecisionPolicy,
    /// Shape policy.
    pub shape_policy: ShapePolicy,
    /// Whether the gate passed (all checks).
    pub passed: bool,
    /// MIL lowering receipt.
    pub lowering: MilLoweringReceipt,
    /// Package receipt.
    pub package: PackageReceipt,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_legality_entry_default_construction() {
        let e = OpLegalityEntry {
            op_id: OperationId(1),
            opcode: Opcode::Matmul,
            legal: true,
            diagnostics: vec![],
        };
        assert!(e.legal);
        assert_eq!(e.opcode, Opcode::Matmul);
    }

    #[test]
    fn mil_lowering_receipt_default_construction() {
        let r = MilLoweringReceipt {
            program_digest: EvidenceDigest("abc".into()),
            op_count: 0,
            constant_count: 0,
            op_legality: vec![],
            warnings: vec![],
            opset: "CoreML6".into(),
        };
        assert_eq!(r.opset, "CoreML6");
        assert_eq!(r.op_count, 0);
    }
}
