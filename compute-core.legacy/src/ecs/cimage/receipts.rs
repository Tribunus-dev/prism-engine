//! CImage receipt types.
//!
//! Receipts are emitted during loading, validation, and numerical comparison.
//! They are stored in the cimage receipt directory and/or returned to the caller.

use serde::{Deserialize, Serialize};

/// Receipt emitted by CImageLoader::load_v0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageLoadReceipt {
    pub cimage_path: String,
    pub cimage_digest: String,
    pub schema_version: u32,
    pub tensor_count: usize,
    pub payload_count: usize,
    pub receipt_count: usize,
    pub total_payload_bytes: u64,
    pub validation_status: CImageValidationStatus,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validation status for a cimage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CImageValidationStatus {
    Valid,
    ValidWithWarnings,
    Invalid,
}

/// Receipt directory stored inside the cimage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptDirectoryV0 {
    pub receipts: Vec<CImageReceiptEntry>,
}

/// One receipt entry in the directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptEntry {
    pub receipt_id: String,
    pub receipt_kind: String,
    pub offset: u64,
    pub len: u64,
    pub sha256: String,
}

/// Shard validation receipt — emitted after numerical comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageShardValidationReceipt {
    pub shard_id: String,
    pub cimage_digest: String,
    pub input_digest: String,
    pub raw_output_digest: String,
    pub packed_output_digest: String,
    pub output_nrmse: f64,
    pub output_cosine: f64,
    pub max_abs_error: f64,
    pub passed: bool,
    pub evidence_kind: ReceiptEvidenceKind,
}

/// Kind of evidence supporting a validation claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptEvidenceKind {
    SyntheticNumericalProof,
    RealTensorNumericalProof,
    MeasuredRuntimeProof,
}

/// An evidence receipt in the receipt directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceReceiptV0 {
    pub receipt_id: String,
    pub receipt_kind: String,
    pub manifest_digest: String,
    pub shard_validation: Option<CImageShardValidationReceipt>,
    pub load_receipt: Option<CImageLoadReceipt>,
}

/// Write receipt emitted by CImageWriter::write_v0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageWriteReceipt {
    pub path: String,
    pub file_size_bytes: u64,
    pub cimage_digest: String,
    pub tensor_count: usize,
    pub payload_count: usize,
    pub receipt_count: usize,
}

/// Evidence kind used in numerical comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CImageProofKind {
    SyntheticNumericalProof,
    RealTensorNumericalProof,
    MeasuredRuntimeProof,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_receipt_serde_roundtrip() {
        let r = CImageLoadReceipt {
            cimage_path: "/tmp/test.cimage".into(),
            cimage_digest: "deadbeef".into(),
            schema_version: 0,
            tensor_count: 4,
            payload_count: 8,
            receipt_count: 2,
            total_payload_bytes: 4096,
            validation_status: CImageValidationStatus::Valid,
            errors: vec![],
            warnings: vec![],
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageLoadReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tensor_count, 4);
        assert_eq!(back.validation_status, CImageValidationStatus::Valid);
    }

    #[test]
    fn test_shard_validation_receipt_serde() {
        let r = CImageShardValidationReceipt {
            shard_id: "synth_mlp_000".into(),
            cimage_digest: "abc".into(),
            input_digest: "def".into(),
            raw_output_digest: "ghi".into(),
            packed_output_digest: "jkl".into(),
            output_nrmse: 0.05,
            output_cosine: 0.999,
            max_abs_error: 0.01,
            passed: true,
            evidence_kind: ReceiptEvidenceKind::SyntheticNumericalProof,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageShardValidationReceipt = serde_json::from_str(&json).unwrap();
        assert!(back.passed);
        assert!((back.output_nrmse - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_validation_status_serde() {
        for status in &[
            CImageValidationStatus::Valid,
            CImageValidationStatus::ValidWithWarnings,
            CImageValidationStatus::Invalid,
        ] {
            let json = serde_json::to_string(status).unwrap();
            let back: CImageValidationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, back);
        }
    }

    #[test]
    fn test_evidence_kind_serde() {
        for kind in &[
            ReceiptEvidenceKind::SyntheticNumericalProof,
            ReceiptEvidenceKind::RealTensorNumericalProof,
            ReceiptEvidenceKind::MeasuredRuntimeProof,
        ] {
            let json = serde_json::to_string(kind).unwrap();
            let back: ReceiptEvidenceKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    fn test_receipt_directory_serde() {
        let dir = CImageReceiptDirectoryV0 {
            receipts: vec![CImageReceiptEntry {
                receipt_id: "r0".into(),
                receipt_kind: "LoadReceipt".into(),
                offset: 100,
                len: 200,
                sha256: "abc".into(),
            }],
        };
        let json = serde_json::to_string_pretty(&dir).unwrap();
        let back: CImageReceiptDirectoryV0 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.receipts.len(), 1);
        assert_eq!(back.receipts[0].receipt_id, "r0");
    }

    #[test]
    fn test_write_receipt_serde() {
        let r = CImageWriteReceipt {
            path: "/tmp/t.cimage".into(),
            file_size_bytes: 8192,
            cimage_digest: "abc".into(),
            tensor_count: 4,
            payload_count: 8,
            receipt_count: 1,
        };
        let json = serde_json::to_string_pretty(&r).unwrap();
        let back: CImageWriteReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.file_size_bytes, 8192);
    }
}
