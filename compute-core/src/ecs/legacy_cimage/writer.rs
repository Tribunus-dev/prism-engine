//! CImage V0 file writer.
//!
//! Assembles a cimage artifact from a manifest, payloads, and receipts,
//! serializes all sections canonically, and writes the result atomically
//! to disk via `tempfile::NamedTempFile`.
//!
//! Layout (in file order):
//!   1. Header (binary bincode)
//!   2. Manifest JSON (canonical)
//!   3. Payload directory JSON (canonical)
//!   4. Receipt directory JSON (canonical)
//!   5. Payload blob (concatenated raw bytes)
//!   6. Footer  (binary bincode)
//!
//! Each JSON section and the payload blob are individually SHA-256 hashed.
//! The footer records all five digests, plus a recursive digest of the file
//! up to (but not including) the footer itself.

use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ecs::legacy_cimage::{
    canonical::canonical_json_bytes, CImageError, CImageFooterV0, CImageHeaderV0, CImageManifestV0,
    CImagePayloadDirectoryV0, CImagePayloadEntry, CImagePayloadKind, CImageReceiptDirectoryV0,
    CImageReceiptEntry, CImageResult, CImageWriteReceipt, PendingPayload, PendingReceipt,
    CIMAGE_FORMAT_VERSION, CIMAGE_MAGIC,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Stateless writer for cimage V0 artifacts.
pub struct CImageWriter;

impl CImageWriter {
    /// Assemble and write a cimage V0 file at `path`.
    ///
    /// Receipt bytes are embedded in the payload blob as
    /// [`CImagePayloadKind::ReceiptJson`] entries. The returned
    /// [`CImageWriteReceipt`] records the final file size and digest.
    ///
    /// # Errors
    ///
    /// Returns [`CImageError::Io`] when the file cannot be written or renamed,
    /// [`CImageError::JsonSerialize`] when a JSON section cannot be serialized.
    pub fn write_v0(
        path: &Path,
        manifest: CImageManifestV0,
        payloads: Vec<PendingPayload>,
        receipts: Vec<PendingReceipt>,
    ) -> CImageResult<CImageWriteReceipt> {
        // ------------------------------------------------------------------
        // 1. Build receipt payload entries
        // ------------------------------------------------------------------
        // Each PendingReceipt becomes a CImagePayloadEntry with kind ReceiptJson
        // whose bytes are appended to the payload blob. We build a temporary
        // list that includes both original payloads and receipt payloads.
        // Offsets are computed later once we have the concatenated payload blob.

        let receipt_payloads: Vec<PendingPayload> = receipts
            .iter()
            .map(|r| PendingPayload {
                payload_id: r.receipt_id.clone(),
                payload_kind: CImagePayloadKind::ReceiptJson,
                codec: None,
                alignment_bytes: 1,
                bytes: r.bytes.clone(),
            })
            .collect();

        // ------------------------------------------------------------------
        // 2. Serialize JSON sections (canonical)
        // ------------------------------------------------------------------
        let manifest_bytes = canonical_json_bytes(&manifest)?;

        // Concatenated payload blob = original payloads + receipt payloads.
        let all_payloads: Vec<&PendingPayload> =
            payloads.iter().chain(receipt_payloads.iter()).collect();

        // Build payload directory entries: offsets are relative to blob start.
        let mut payload_entries: Vec<CImagePayloadEntry> = Vec::with_capacity(all_payloads.len());
        let mut payload_blob = Vec::new();
        for pp in &all_payloads {
            let offset = payload_blob.len() as u64;
            let len = pp.bytes.len() as u64;
            payload_blob.extend_from_slice(&pp.bytes);
            payload_entries.push(CImagePayloadEntry {
                payload_id: pp.payload_id.clone(),
                payload_kind: pp.payload_kind,
                codec: pp.codec.clone(),
                offset,
                len,
                alignment_bytes: pp.alignment_bytes,
                sha256: sha256_hex(&pp.bytes),
            });
        }

        let payload_directory = CImagePayloadDirectoryV0 {
            payloads: payload_entries.clone(),
        };
        let payload_directory_bytes = canonical_json_bytes(&payload_directory)?;

        // Build receipt directory entries referencing receipt payload entries.
        // A receipt entry has its own offset/len/sha256 pointing into the
        // payload blob (the same range covered by the corresponding
        // CImagePayloadEntry with kind ReceiptJson).
        let receipt_entries: Vec<CImageReceiptEntry> = receipts
            .iter()
            .map(|r| {
                // Find the corresponding payload entry.
                let entry = payload_entries
                    .iter()
                    .find(|pe| pe.payload_id == r.receipt_id)
                    .expect("receipt payload entry must exist");
                CImageReceiptEntry {
                    receipt_id: r.receipt_id.clone(),
                    receipt_kind: r.receipt_kind.clone(),
                    offset: entry.offset,
                    len: entry.len,
                    sha256: entry.sha256.clone(),
                }
            })
            .collect();

        let receipt_directory = CImageReceiptDirectoryV0 {
            receipts: receipt_entries,
        };
        let receipt_directory_bytes = canonical_json_bytes(&receipt_directory)?;

        // ------------------------------------------------------------------
        // 3. Compute section offsets
        // ------------------------------------------------------------------
        let header_size = std::mem::size_of::<CImageHeaderV0>() as u64;
        let manifest_len = manifest_bytes.len() as u64;
        let payload_directory_len = payload_directory_bytes.len() as u64;
        let receipt_directory_len = receipt_directory_bytes.len() as u64;
        let payload_blob_len = payload_blob.len() as u64;

        let manifest_offset = header_size;
        let payload_directory_offset = manifest_offset + manifest_len;
        let receipt_directory_offset = payload_directory_offset + payload_directory_len;
        let payload_blob_offset = receipt_directory_offset + receipt_directory_len;
        let footer_offset = payload_blob_offset + payload_blob_len;

        // ------------------------------------------------------------------
        // 4. Build header
        // ------------------------------------------------------------------
        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: header_size,
            manifest_offset,
            manifest_len,
            payload_directory_offset,
            payload_directory_len,
            receipt_directory_offset,
            receipt_directory_len,
            payload_blob_offset,
            payload_blob_len,
            footer_offset,
        };

        // ------------------------------------------------------------------
        // 5. Write everything to a NamedTempFile
        // ------------------------------------------------------------------
        let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(dir)
            .map_err(|e| CImageError::Io(format!("create tempfile: {e}")))?;

        // 5a. Write header
        let header_bytes = bincode::serialize(&header)
            .map_err(|e| CImageError::Io(format!("bincode header: {e}")))?;
        tmp.write_all(&header_bytes)
            .map_err(|e| CImageError::Io(format!("write header: {e}")))?;
        // bincode may not use all bytes of the repr — pad with zeros to
        // guarantee the fixed-size header occupies sizeof(CImageHeaderV0) bytes.
        if header_bytes.len() < header_size as usize {
            let pad = header_size as usize - header_bytes.len();
            tmp.write_all(&vec![0u8; pad])
                .map_err(|e| CImageError::Io(format!("pad header: {e}")))?;
        }

        // 5b. Write manifest
        tmp.write_all(&manifest_bytes)
            .map_err(|e| CImageError::Io(format!("write manifest: {e}")))?;

        // 5c. Write payload directory
        tmp.write_all(&payload_directory_bytes)
            .map_err(|e| CImageError::Io(format!("write payload directory: {e}")))?;

        // 5d. Write receipt directory
        tmp.write_all(&receipt_directory_bytes)
            .map_err(|e| CImageError::Io(format!("write receipt directory: {e}")))?;

        // 5e. Write payload blob
        tmp.write_all(&payload_blob)
            .map_err(|e| CImageError::Io(format!("write payload blob: {e}")))?;

        // ------------------------------------------------------------------
        // 6. Compute digests for the footer
        // ------------------------------------------------------------------
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let payload_directory_sha256 = sha256_hex(&payload_directory_bytes);
        let receipt_directory_sha256 = sha256_hex(&receipt_directory_bytes);
        let payload_blob_sha256 = sha256_hex(&payload_blob);

        // cimage_sha256_without_footer covers everything written so far.
        let cimage_sha256_without_footer = {
            // The simplest approach is to read back what we wrote.
            // We avoid seeking on NamedTempFile and instead hash in-memory
            // copies of what was written — but those are already available as
            // serialized byte slices.
            let mut hasher = Sha256::new();
            hasher.update(&header_bytes);
            // Include header padding if any
            if header_bytes.len() < header_size as usize {
                let pad = header_size as usize - header_bytes.len();
                hasher.update(&vec![0u8; pad]);
            }
            hasher.update(&manifest_bytes);
            hasher.update(&payload_directory_bytes);
            hasher.update(&receipt_directory_bytes);
            hasher.update(&payload_blob);
            format!("{:x}", hasher.finalize())
        };

        let footer = CImageFooterV0 {
            manifest_sha256,
            payload_directory_sha256,
            receipt_directory_sha256,
            payload_blob_sha256,
            cimage_sha256_without_footer,
        };

        // ------------------------------------------------------------------
        // 7. Write footer
        // ------------------------------------------------------------------
        let footer_bytes = bincode::serialize(&footer)
            .map_err(|e| CImageError::Io(format!("bincode footer: {e}")))?;
        tmp.write_all(&footer_bytes)
            .map_err(|e| CImageError::Io(format!("write footer: {e}")))?;
        // Pad footer to sizeof(CImageFooterV0) if bincode is shorter.
        let footer_size = std::mem::size_of::<CImageFooterV0>() as u64;
        if footer_bytes.len() < footer_size as usize {
            let pad = footer_size as usize - footer_bytes.len();
            tmp.write_all(&vec![0u8; pad])
                .map_err(|e| CImageError::Io(format!("pad footer: {e}")))?;
        }

        // ------------------------------------------------------------------
        // 8. Atomic rename and compute final digest
        // ------------------------------------------------------------------
        tmp.flush()
            .map_err(|e| CImageError::Io(format!("flush tempfile: {e}")))?;

        tmp.persist(path)
            .map_err(|e| CImageError::Io(format!("persist tempfile: {e}")))?;

        // Read the final file to compute its full digest.
        let final_bytes =
            std::fs::read(path).map_err(|e| CImageError::Io(format!("read final file: {e}")))?;
        let file_size_bytes = final_bytes.len() as u64;
        let cimage_digest = sha256_hex(&final_bytes);

        Ok(CImageWriteReceipt {
            path: path.to_string_lossy().to_string(),
            file_size_bytes,
            cimage_digest,
            tensor_count: manifest.tensors.len(),
            payload_count: payload_entries.len(),
            receipt_count: receipts.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_cimage::{
        CImageArtifactKind, CImageLoader, CImageManifestV0, ModelExecutionPlanSummary,
    };
    use crate::execution_plan::HardwareProfileId;

    /// A small, self-contained manifest for roundtrip testing.
    fn make_test_manifest() -> CImageManifestV0 {
        CImageManifestV0 {
            schema_version: 1,
            model_family: "writer-test".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "test-policy-0001".into(),
            layout_profile: HardwareProfileId::AppleA18Tiny,
            tensors: vec![],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "plan-writer-test".into(),
                region_count: 0,
                total_kernel_ops: 0,
                total_input_bytes: 0,
                total_output_bytes: 0,
                tensor_refs: vec![],
            },
            receipts: vec![],
            assistant_graph: None,
            state_store_schema: None,
        }
    }

    #[test]
    fn roundtrip_v0_with_payloads_and_receipts() {
        let manifest = make_test_manifest();

        let payloads = vec![
            PendingPayload {
                payload_id: "p1".into(),
                payload_kind: CImagePayloadKind::PackedTensorCodes,
                codec: Some("nf4".into()),
                alignment_bytes: 1,
                bytes: vec![0x10, 0x20, 0x30],
            },
            PendingPayload {
                payload_id: "p2".into(),
                payload_kind: CImagePayloadKind::ExecutionPlanJson,
                codec: None,
                alignment_bytes: 1,
                bytes: br#"{"op":"matmul"}"#.to_vec(),
            },
        ];

        let receipts = vec![PendingReceipt {
            receipt_id: "r1".into(),
            receipt_kind: "test-receipt".into(),
            bytes: br#"{"result":"ok"}"#.to_vec(),
        }];

        let tmp_dir = std::env::temp_dir();
        let output_path = tmp_dir.join("cimage_writer_roundtrip.cim");

        // Write.
        let write_receipt =
            CImageWriter::write_v0(&output_path, manifest, payloads, receipts).expect("write_v0");

        // Load back.
        let loaded = CImageLoader::load_v0(&output_path).expect("load_v0");

        // Verify basic structure.
        assert_eq!(loaded.header.magic, CIMAGE_MAGIC);
        assert_eq!(loaded.header.format_version, CIMAGE_FORMAT_VERSION);
        assert_eq!(loaded.manifest.model_family, "writer-test");

        // Verify payload directory.
        assert_eq!(loaded.payload_directory.payloads.len(), 3);
        let p1 = &loaded.payload_directory.payloads[0];
        assert_eq!(p1.payload_id, "p1");
        assert_eq!(p1.payload_kind, CImagePayloadKind::PackedTensorCodes);
        assert_eq!(p1.codec.as_deref(), Some("nf4"));
        assert_eq!(p1.len, 3);
        assert_eq!(p1.offset, 0);

        let p2 = &loaded.payload_directory.payloads[1];
        assert_eq!(p2.payload_id, "p2");
        assert_eq!(p2.payload_kind, CImagePayloadKind::ExecutionPlanJson);
        assert_eq!(p2.len, 15);
        assert_eq!(p2.offset, 3);

        let r1_pe = &loaded.payload_directory.payloads[2];
        assert_eq!(r1_pe.payload_id, "r1");
        assert_eq!(r1_pe.payload_kind, CImagePayloadKind::ReceiptJson);
        assert_eq!(r1_pe.offset, 18);

        // Verify receipt directory.
        assert_eq!(loaded.receipt_directory.receipts.len(), 1);
        let r1_re = &loaded.receipt_directory.receipts[0];
        assert_eq!(r1_re.receipt_id, "r1");
        assert_eq!(r1_re.receipt_kind, "test-receipt");
        assert_eq!(r1_re.offset, r1_pe.offset);
        assert_eq!(r1_re.len, r1_pe.len);
        assert_eq!(r1_re.sha256, r1_pe.sha256);

        // Verify payload blob contents.
        assert_eq!(loaded.payload_blob.len(), 33);
        assert_eq!(&loaded.payload_blob[0..3], &[0x10, 0x20, 0x30]);
        assert_eq!(&loaded.payload_blob[3..18], br#"{"op":"matmul"}"#);
        assert_eq!(&loaded.payload_blob[18..33], br#"{"result":"ok"}"#);

        // Verify write receipt fields.
        assert_eq!(write_receipt.tensor_count, 0);
        assert_eq!(write_receipt.payload_count, 3);
        assert_eq!(write_receipt.receipt_count, 1);
        assert!(!write_receipt.cimage_digest.is_empty());
        assert!(write_receipt.file_size_bytes > 0);

        // Clean up.
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn roundtrip_v0_empty_payloads() {
        let manifest = make_test_manifest();

        let tmp = std::env::temp_dir().join("cimage_empty.cim");
        let write_receipt = CImageWriter::write_v0(&tmp, manifest, vec![], vec![])
            .expect("write_v0 with no payloads or receipts");

        assert_eq!(write_receipt.payload_count, 0);
        assert_eq!(write_receipt.receipt_count, 0);
        assert_eq!(write_receipt.tensor_count, 0);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn roundtrip_v0_only_receipts() {
        let manifest = make_test_manifest();

        let receipts = vec![PendingReceipt {
            receipt_id: "r1".into(),
            receipt_kind: "self-test".into(),
            bytes: b"receipt-data".to_vec(),
        }];

        let tmp = std::env::temp_dir().join("cimage_only_receipts.cim");
        let wr = CImageWriter::write_v0(&tmp, manifest, vec![], receipts).expect("write_v0");
        assert_eq!(wr.payload_count, 1);
        assert_eq!(wr.receipt_count, 1);

        let loaded = CImageLoader::load_v0(&tmp).expect("load_v0");
        assert_eq!(loaded.payload_directory.payloads.len(), 1);
        assert_eq!(
            loaded.payload_directory.payloads[0].payload_kind,
            CImagePayloadKind::ReceiptJson
        );
        assert_eq!(loaded.payload_blob, b"receipt-data");
        assert_eq!(loaded.receipt_directory.receipts.len(), 1);
        assert_eq!(
            loaded.receipt_directory.receipts[0].sha256,
            loaded.payload_directory.payloads[0].sha256
        );

        let _ = std::fs::remove_file(&tmp);
    }
}
