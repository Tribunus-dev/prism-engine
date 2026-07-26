//! CImage payload directory — maps payload IDs to byte ranges in the payload blob.
//!
//! Offsets are relative to `payload_blob_offset` (not absolute file offsets),
//! making validation simpler.

use serde::{Deserialize, Serialize};

/// V0 payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImagePayloadDirectoryV0 {
    pub payloads: Vec<CImagePayloadEntry>,
}

/// One entry in the payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImagePayloadEntry {
    pub payload_id: String,
    pub payload_kind: CImagePayloadKind,
    pub codec: Option<String>,
    pub offset: u64,
    pub len: u64,
    pub alignment_bytes: u32,
    pub sha256: String,
}

/// Kind of payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CImagePayloadKind {
    PackedTensorCodes,
    TensorMetadata,
    RawF32Reference,
    MixedPrecisionOverrideTable,
    MixedPrecisionSidecar,
    ExecutionPlanJson,
    ReceiptJson,
    AssistantGraphJson,
    StateStoreSchemaJson,
    TernaryPackedCodes,
    TernaryScales,
    TernaryCalibrationMetadata,
    TernaryAdmissionReceiptJson,
}

/// A pending payload ready to be written.
#[derive(Debug, Clone)]
pub struct PendingPayload {
    pub payload_id: String,
    pub payload_kind: CImagePayloadKind,
    pub codec: Option<String>,
    pub alignment_bytes: u32,
    pub bytes: Vec<u8>,
}

/// A receipt ready to be written into the receipt directory.
#[derive(Debug, Clone)]
pub struct PendingReceipt {
    pub receipt_id: String,
    pub receipt_kind: String,
    pub bytes: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_directory_serde_roundtrip() {
        let dir = CImagePayloadDirectoryV0 {
            payloads: vec![
                CImagePayloadEntry {
                    payload_id: "p_t0_codes".into(),
                    payload_kind: CImagePayloadKind::PackedTensorCodes,
                    codec: Some("NF4".into()),
                    offset: 0,
                    len: 320,
                    alignment_bytes: 1,
                    sha256: "deadbeef".into(),
                },
                CImagePayloadEntry {
                    payload_id: "p_t0_rawf32".into(),
                    payload_kind: CImagePayloadKind::RawF32Reference,
                    codec: None,
                    offset: 320,
                    len: 256,
                    alignment_bytes: 1,
                    sha256: "cafebabe".into(),
                },
            ],
        };
        let json = serde_json::to_string_pretty(&dir).unwrap();
        let back: CImagePayloadDirectoryV0 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.payloads.len(), 2);
        assert_eq!(back.payloads[0].payload_id, "p_t0_codes");
        assert_eq!(
            back.payloads[0].payload_kind,
            CImagePayloadKind::PackedTensorCodes
        );
    }

    #[test]
    fn test_payload_kind_serde() {
        for kind in &[
            CImagePayloadKind::PackedTensorCodes,
            CImagePayloadKind::TensorMetadata,
            CImagePayloadKind::RawF32Reference,
            CImagePayloadKind::MixedPrecisionOverrideTable,
            CImagePayloadKind::MixedPrecisionSidecar,
            CImagePayloadKind::ExecutionPlanJson,
            CImagePayloadKind::ReceiptJson,
            CImagePayloadKind::AssistantGraphJson,
            CImagePayloadKind::StateStoreSchemaJson,
        ] {
            let json = serde_json::to_string(kind).unwrap();
            let back: CImagePayloadKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }
}
