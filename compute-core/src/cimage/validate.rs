//! CImage validation — checks all 14 gates to verify cimage integrity.
//!
//! The validator operates on a [`LoadedCImageV0`] and returns a
//! [`CImageLoadReceipt`] summarising all found errors and warnings.
//! Validation never short-circuits: all checks run so the caller sees the
//! full picture in a single pass.

use std::collections::HashSet;

use sha2::{Digest, Sha256};

use crate::cimage::*;
use crate::execution_plan::CodecFamily;

/// Hex SHA-256 digest helper.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// CImage validator — stateless, callable once per loaded image.
pub struct CImageValidator;

impl CImageValidator {
    /// Validate every gate on a loaded cimage.
    ///
    /// Errors are collected (never short-circuit) so the caller sees the full
    /// picture.
    pub fn validate_loaded(image: &LoadedCImageV0) -> CImageResult<CImageLoadReceipt> {
        let mut errors: Vec<String> = Vec::new();
        let warnings: Vec<String> = Vec::new();

        let file_size = image.raw_file_bytes.len() as u64;
        let header = &image.header;

        // ── Gate 1: Magic check ────────────────────────────────────────
        if header.magic != CIMAGE_MAGIC {
            errors.push("invalid magic".to_string());
        }

        // ── Gate 2: Format version ─────────────────────────────────────
        if header.format_version != CIMAGE_FORMAT_VERSION {
            errors.push(format!(
                "unsupported format version {}",
                header.format_version
            ));
        }

        // ── Gate 3: Range bounds for every section ─────────────────────
        let sections: [(&str, u64, u64); 4] = [
            ("manifest", header.manifest_offset, header.manifest_len),
            (
                "payload_directory",
                header.payload_directory_offset,
                header.payload_directory_len,
            ),
            (
                "receipt_directory",
                header.receipt_directory_offset,
                header.receipt_directory_len,
            ),
            (
                "payload_blob",
                header.payload_blob_offset,
                header.payload_blob_len,
            ),
        ];
        let footer_offset = header.footer_offset;
        let _footer_len = file_size.saturating_sub(footer_offset);

        for (name, offset, len) in &sections {
            // offset + len may overflow; checked_add avoids wrap-around.
            let end = match offset.checked_add(*len) {
                Some(v) => v,
                None => {
                    errors.push(format!(
                        "offset {offset} + len {len} overflows for section {name}"
                    ));
                    continue;
                }
            };
            if end > file_size {
                errors.push(format!(
                    "offset {offset} + len {len} exceeds file size {file_size} for section {name}"
                ));
            }
        }
        // Footer range check (footer extends to end of file).
        if footer_offset > file_size {
            errors.push(format!(
                "footer offset {footer_offset} exceeds file size {file_size}"
            ));
        }

        // ── Parse footer (needed for digest gates 4–8) ─────────────────
        // If the footer range is invalid, push an error and skip digest
        // gates that depend on it.
        let footer: Option<CImageFooterV0> = if footer_offset <= file_size {
            let footer_bytes = &image.raw_file_bytes[footer_offset as usize..];
            match bincode::deserialize::<CImageFooterV0>(footer_bytes) {
                Ok(f) => Some(f),
                Err(e) => {
                    errors.push(format!("failed to deserialize footer: {e}"));
                    None
                }
            }
        } else {
            None
        };

        if let Some(footer) = &footer {
            // ── Gate 4: Manifest digest ────────────────────────────────
            Self::check_digest_slice(
                &image.raw_file_bytes,
                header.manifest_offset,
                header.manifest_len,
                file_size,
                "manifest",
                &footer.manifest_sha256,
                &mut errors,
            );

            // ── Gate 5: Payload directory digest ───────────────────────
            Self::check_digest_slice(
                &image.raw_file_bytes,
                header.payload_directory_offset,
                header.payload_directory_len,
                file_size,
                "payload_directory",
                &footer.payload_directory_sha256,
                &mut errors,
            );

            // ── Gate 6: Receipt directory digest ───────────────────────
            Self::check_digest_slice(
                &image.raw_file_bytes,
                header.receipt_directory_offset,
                header.receipt_directory_len,
                file_size,
                "receipt_directory",
                &footer.receipt_directory_sha256,
                &mut errors,
            );

            // ── Gate 7: Payload blob digest ────────────────────────────
            Self::check_digest_slice(
                &image.raw_file_bytes,
                header.payload_blob_offset,
                header.payload_blob_len,
                file_size,
                "payload_blob",
                &footer.payload_blob_sha256,
                &mut errors,
            );

            // ── Gate 8: File-wide digest (without footer) ──────────────
            let prefix_end = footer_offset as usize;
            if prefix_end <= image.raw_file_bytes.len() {
                let computed = sha256_hex(&image.raw_file_bytes[..prefix_end]);
                if computed != footer.cimage_sha256_without_footer {
                    errors.push(format!(
                        "digest mismatch for cimage (without footer): expected {} got {}",
                        footer.cimage_sha256_without_footer, computed
                    ));
                }
            }
        }

        // ── Gate 9: Payload entry digests ──────────────────────────────
        for entry in &image.payload_directory.payloads {
            let start = entry.offset as usize;
            let end = match start.checked_add(entry.len as usize) {
                Some(v) => v,
                None => {
                    errors.push(format!(
                        "payload {} offset {} + len {} overflows usize",
                        entry.payload_id, entry.offset, entry.len
                    ));
                    continue;
                }
            };
            if end <= image.payload_blob.len() {
                let computed = sha256_hex(&image.payload_blob[start..end]);
                if computed != entry.sha256 {
                    errors.push(format!(
                        "payload digest mismatch for {}: expected {} got {}",
                        entry.payload_id, entry.sha256, computed
                    ));
                }
            } else {
                errors.push(format!(
                    "payload {} range {}+{} out of bounds in payload blob (len {})",
                    entry.payload_id,
                    entry.offset,
                    entry.len,
                    image.payload_blob.len()
                ));
            }
        }

        // ── Build payload ID set for ref resolution ────────────────────
        let payload_ids: HashSet<&str> = image
            .payload_directory
            .payloads
            .iter()
            .map(|e| e.payload_id.as_str())
            .collect();

        // ── Gate 10: Tensor payload refs resolve ───────────────────────
        for tensor in &image.manifest.tensors {
            Self::check_payload_ref_resolves(
                &tensor.payload_ref,
                &payload_ids,
                &tensor.tensor_id,
                &mut errors,
            );
            if let Some(raw_ref) = &tensor.raw_f32_reference_ref {
                Self::check_payload_ref_resolves(
                    raw_ref,
                    &payload_ids,
                    &tensor.tensor_id,
                    &mut errors,
                );
            }
        }

        // ── Gate 11: Physical layout validity ──────────────────────────
        for tensor in &image.manifest.tensors {
            if !tensor.physical_layout.is_valid() {
                errors.push(format!(
                    "invalid physical layout for tensor {}: tile_m={} tile_n={} group_size={} groups_per_tile={}",
                    tensor.tensor_id,
                    tensor.physical_layout.tile_m,
                    tensor.physical_layout.tile_n,
                    tensor.physical_layout.group_size,
                    tensor.physical_layout.groups_per_tile,
                ));
            }
        }

        // ── Gate 12: Mixed precision invariants ────────────────────────
        for tensor in &image.manifest.tensors {
            match tensor.codec {
                CodecFamily::Mixed => {
                    if tensor.precision_plan.is_none() {
                        errors.push(format!(
                            "mixed codec without precision plan for tensor {}",
                            tensor.tensor_id
                        ));
                    }
                }
                _ => {
                    if matches!(tensor.payload_ref, CImagePayloadRef::MixedPrecision { .. }) {
                        errors.push(format!(
                            "non-mixed tensor {} carries mixed payload ref",
                            tensor.tensor_id
                        ));
                    }
                }
            }
        }

        // ── Gate 13: Execution plan tensor refs resolve ────────────────
        let tensor_ids: HashSet<&str> = image
            .manifest
            .tensors
            .iter()
            .map(|t| t.tensor_id.as_str())
            .collect();
        for tensor_ref in &image.manifest.execution_plan.tensor_refs {
            if !tensor_ids.contains(tensor_ref.as_str()) {
                errors.push(format!("unresolved tensor ref: {tensor_ref}"));
            }
        }

        // ── Gate 14: Receipt refs resolve ──────────────────────────────
        let receipt_ids: HashSet<&str> = image
            .receipt_directory
            .receipts
            .iter()
            .map(|r| r.receipt_id.as_str())
            .collect();
        for receipt_ref in &image.manifest.receipts {
            if !receipt_ids.contains(receipt_ref.receipt_id.as_str()) {
                errors.push(format!(
                    "unresolved receipt ref: {}",
                    receipt_ref.receipt_id
                ));
            }
        }

        // ── Build receipt ──────────────────────────────────────────────
        let cimage_digest = sha256_hex(&image.raw_file_bytes);
        let validation_status = if !errors.is_empty() {
            CImageValidationStatus::Invalid
        } else if !warnings.is_empty() {
            CImageValidationStatus::ValidWithWarnings
        } else {
            CImageValidationStatus::Valid
        };

        Ok(CImageLoadReceipt {
            cimage_path: image.path.display().to_string(),
            cimage_digest,
            schema_version: image.manifest.schema_version,
            tensor_count: image.manifest.tensors.len(),
            payload_count: image.payload_directory.payloads.len(),
            receipt_count: image.manifest.receipts.len(),
            total_payload_bytes: image.payload_blob.len() as u64,
            validation_status,
            errors,
            warnings,
        })
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Check that a byte-range's digest matches the expected value.
    fn check_digest_slice(
        raw: &[u8],
        offset: u64,
        len: u64,
        file_size: u64,
        section: &str,
        expected: &str,
        errors: &mut Vec<String>,
    ) {
        if len == 0 {
            errors.push(format!("{section} has zero length, cannot verify digest"));
            return;
        }
        let end = match offset.checked_add(len) {
            Some(v) => v,
            None => {
                errors.push(format!("{section} offset {offset} + len {len} overflows"));
                return;
            }
        };
        if end > file_size {
            errors.push(format!(
                "{section} offset {offset} + len {len} exceeds file size {file_size}"
            ));
            return;
        }
        let start_usize = offset as usize;
        let end_usize = end as usize;
        let computed = sha256_hex(&raw[start_usize..end_usize]);
        if computed != expected {
            errors.push(format!(
                "digest mismatch for {section}: expected {expected} got {computed}"
            ));
        }
    }

    /// Verify that all payload IDs referenced in a [`CImagePayloadRef`] exist
    /// in the payload directory.
    fn check_payload_ref_resolves(
        payload_ref: &CImagePayloadRef,
        payload_ids: &HashSet<&str>,
        tensor_id: &str,
        errors: &mut Vec<String>,
    ) {
        match payload_ref {
            CImagePayloadRef::Single { payload_id } => {
                if !payload_ids.contains(payload_id.as_str()) {
                    errors.push(format!(
                        "unresolved payload ref {payload_id} in tensor {tensor_id}"
                    ));
                }
            }
            CImagePayloadRef::MixedPrecision {
                base_payload_id,
                override_table_payload_id,
                sidecar_payload_ids,
            } => {
                if !payload_ids.contains(base_payload_id.as_str()) {
                    errors.push(format!(
                        "unresolved payload ref {base_payload_id} (base) in tensor {tensor_id}"
                    ));
                }
                if !payload_ids.contains(override_table_payload_id.as_str()) {
                    errors.push(format!(
                        "unresolved payload ref {override_table_payload_id} (override_table) in tensor {tensor_id}"
                    ));
                }
                for sid in sidecar_payload_ids {
                    if !payload_ids.contains(sid.as_str()) {
                        errors.push(format!(
                            "unresolved payload ref {sid} (sidecar) in tensor {tensor_id}"
                        ));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cimage::payload::CImagePayloadKind;

    use crate::execution_plan::{DType, HardwareProfileId};

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Build a minimal valid `LoadedCImageV0` for testing.
    ///
    /// The caller must construct and pass the raw file bytes consistent with
    /// the header, manifest, etc.  This helper wraps them into a
    /// `LoadedCImageV0` whose internal references are all self-consistent.
    fn make_loaded(
        raw_file_bytes: Vec<u8>,
        header: CImageHeaderV0,
        manifest: CImageManifestV0,
        payload_directory: CImagePayloadDirectoryV0,
        receipt_directory: CImageReceiptDirectoryV0,
        payload_blob: Vec<u8>,
    ) -> LoadedCImageV0 {
        LoadedCImageV0 {
            path: std::path::PathBuf::from("test.cimage"),
            raw_file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        }
    }

    /// Build a valid roundtrip cimage where every gate passes.
    ///
    /// Returns the bytes, the header, the footer, and all deserialized
    /// sections.
    fn build_valid_cimage() -> (
        Vec<u8>,
        CImageHeaderV0,
        CImageFooterV0,
        CImageManifestV0,
        CImagePayloadDirectoryV0,
        CImageReceiptDirectoryV0,
        Vec<u8>,
    ) {
        let payload_blob = vec![0xABu8; 64];

        // ── Payload directory ──────────────────────────────────────────
        let payload_entry = CImagePayloadEntry {
            payload_id: "p1".to_string(),
            payload_kind: CImagePayloadKind::PackedTensorCodes,
            codec: Some("nf4".to_string()),
            offset: 0,
            len: 64,
            alignment_bytes: 1,
            sha256: sha256_hex(&payload_blob[0..64]),
        };
        let payload_directory = CImagePayloadDirectoryV0 {
            payloads: vec![payload_entry],
        };

        // ── Receipt directory ──────────────────────────────────────────
        let receipt_entry = CImageReceiptEntry {
            receipt_id: "r1".to_string(),
            receipt_kind: "validation".to_string(),
            offset: 0,
            len: 0,
            sha256: String::new(),
        };
        let receipt_directory = CImageReceiptDirectoryV0 {
            receipts: vec![receipt_entry],
        };

        // ── Manifest ───────────────────────────────────────────────────
        let tensor_entry = CImageTensorEntry {
            tensor_id: "t1".to_string(),
            tensor_key: "model.layers.0.self_attn.q_proj.weight".to_string(),
            tensor_class: "q_proj".to_string(),
            logical_shape: vec![4096, 4096],
            source_dtype: DType::F16,
            codec: CodecFamily::Nf4,
            precision_plan: None,
            physical_layout: PhysicalTileLayout {
                tile_m: 128,
                tile_n: 128,
                tiles_per_row: 32,
                total_tiles: 1024,
                padded_cols: 128,
                group_size: 32,
                groups_per_tile: 4,
                packed_bytes_per_tile: 4096,
                metadata_f32_per_tile: 2,
            },
            payload_ref: CImagePayloadRef::Single {
                payload_id: "p1".to_string(),
            },
            raw_f32_reference_ref: None,
            tensor_sha256: String::new(),
            validation_digest: None,
        };
        let manifest = CImageManifestV0 {
            schema_version: 0,
            model_family: "gemma-4-test".to_string(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "abc123".to_string(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            tensors: vec![tensor_entry],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "plan-1".to_string(),
                region_count: 1,
                total_kernel_ops: 1,
                total_input_bytes: 4096,
                total_output_bytes: 4096,
                tensor_refs: vec!["t1".to_string()],
            },
            receipts: vec![CImageReceiptRef {
                receipt_id: "r1".to_string(),
                receipt_kind: "validation".to_string(),
            }],
        };

        // ── Serialize sections ─────────────────────────────────────────
        let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        let payload_dir_bytes = serde_json::to_vec(&payload_directory).unwrap();
        let receipt_dir_bytes = serde_json::to_vec(&receipt_directory).unwrap();

        // ── Footer ─────────────────────────────────────────────────────
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let payload_directory_sha256 = sha256_hex(&payload_dir_bytes);
        let receipt_directory_sha256 = sha256_hex(&receipt_dir_bytes);
        let payload_blob_sha256 = sha256_hex(&payload_blob);

        // ── Compute offsets ────────────────────────────────────────────
        // bincode serializes without struct padding, so we must use the
        // actual bincode size for accurate offset computation.
        let header_len = bincode::serialize(&CImageHeaderV0::new()).unwrap().len() as u64;
        let manifest_offset = header_len;
        let manifest_len = manifest_bytes.len() as u64;
        let payload_directory_offset = manifest_offset + manifest_len;
        let payload_directory_len = payload_dir_bytes.len() as u64;
        let receipt_directory_offset = payload_directory_offset + payload_directory_len;
        let receipt_directory_len = receipt_dir_bytes.len() as u64;
        let payload_blob_offset = receipt_directory_offset + receipt_directory_len;
        let payload_blob_len = payload_blob.len() as u64;

        // Build footer and compute its bytes.
        let prefix_end = payload_blob_offset + payload_blob_len;
        let full_file_without_footer: Vec<u8> = {
            let header = CImageHeaderV0 {
                header_len,
                manifest_offset,
                manifest_len,
                payload_directory_offset,
                payload_directory_len,
                receipt_directory_offset,
                receipt_directory_len,
                payload_blob_offset,
                payload_blob_len,
                footer_offset: prefix_end, // will be patched
                ..CImageHeaderV0::new()
            };
            // Re-serialize with correct offsets.
            let hb = bincode::serialize(&header).unwrap();
            let mut all = Vec::new();
            all.extend_from_slice(&hb);
            all.extend_from_slice(&manifest_bytes);
            all.extend_from_slice(&payload_dir_bytes);
            all.extend_from_slice(&receipt_dir_bytes);
            all.extend_from_slice(&payload_blob);
            all
        };

        let cimage_sha256_without_footer = sha256_hex(&full_file_without_footer);

        let footer = CImageFooterV0 {
            manifest_sha256,
            payload_directory_sha256,
            receipt_directory_sha256,
            payload_blob_sha256,
            cimage_sha256_without_footer,
        };
        let footer_bytes = bincode::serialize(&footer).unwrap();
        let footer_offset = prefix_end;

        let header = CImageHeaderV0 {
            header_len,
            manifest_offset,
            manifest_len,
            payload_directory_offset,
            payload_directory_len,
            receipt_directory_offset,
            receipt_directory_len,
            payload_blob_offset,
            payload_blob_len,
            footer_offset,
            ..CImageHeaderV0::new()
        };

        // ── Assemble final file ────────────────────────────────────────
        let header_bytes = bincode::serialize(&header).unwrap();
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_bytes);
        file_bytes.extend_from_slice(&manifest_bytes);
        file_bytes.extend_from_slice(&payload_dir_bytes);
        file_bytes.extend_from_slice(&receipt_dir_bytes);
        file_bytes.extend_from_slice(&payload_blob);
        file_bytes.extend_from_slice(&footer_bytes);

        (
            file_bytes,
            header,
            footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        )
    }

    // ── Test 1: Roundtrip — valid → Valid ──────────────────────────────
    #[test]
    fn test_valid_cimage_roundtrip() {
        let (
            file_bytes,
            header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Valid,
            "expected Valid, got errors: {:?} warnings: {:?}",
            receipt.errors,
            receipt.warnings,
        );
    }

    // ── Test 2: Bad magic → Invalid ────────────────────────────────────
    #[test]
    fn test_bad_magic() {
        let (
            mut file_bytes,
            mut header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Corrupt the magic bytes in raw_file_bytes (first 8 bytes).
        file_bytes[..8].copy_from_slice(b"BADMAGIC");
        // Also corrupt the header struct so the validator sees invalid magic.
        header.magic = *b"BADMAGIC";

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for bad magic"
        );
        assert!(
            receipt.errors.iter().any(|e| e.contains("invalid magic")),
            "expected 'invalid magic' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 3: Corrupted payload blob → Invalid ───────────────────────
    #[test]
    fn test_bad_payload_digest() {
        let (
            file_bytes,
            header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            mut payload_blob,
        ) = build_valid_cimage();

        // Corrupt one byte in the payload blob.
        payload_blob[0] ^= 0xFF;

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for corrupted payload blob"
        );
        // The payload blob digest (gate 7) and the payload entry digest
        // (gate 9) should both fire.  Check at least one.
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("digest mismatch") && e.contains("payload")),
            "expected payload digest mismatch, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 4: Out-of-range payload directory → Invalid ───────────────
    #[test]
    fn test_out_of_range_payload() {
        let (
            file_bytes,
            mut header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Set the payload directory to point beyond the file end.
        header.payload_directory_offset = file_bytes.len() as u64 + 1000;
        header.payload_directory_len = 50;

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for out-of-range payload directory"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("exceeds file size") && e.contains("payload_directory")),
            "expected 'exceeds file size' error for payload_directory, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 5: Missing tensor payload ref → Invalid ───────────────────
    #[test]
    fn test_missing_tensor_payload_ref() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Point tensor t1 at a payload_id that doesn't exist in the directory.
        manifest.tensors[0].payload_ref = CImagePayloadRef::Single {
            payload_id: "nonexistent_payload".to_string(),
        };

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for missing payload ref"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("unresolved payload ref")),
            "expected 'unresolved payload ref' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 6: Mixed codec without precision plan → Invalid ───────────
    #[test]
    fn test_mixed_codec_without_precision_plan() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        manifest.tensors[0].codec = CodecFamily::Mixed;
        manifest.tensors[0].precision_plan = None;
        // MixedPrecision payload ref is fine for the ref-resolve check,
        // but we also need to make sure the payload entry exists.
        // Override the payload ref to a MixedPrecision that resolves.
        manifest.tensors[0].payload_ref = CImagePayloadRef::MixedPrecision {
            base_payload_id: "p1".to_string(),
            override_table_payload_id: "p1".to_string(),
            sidecar_payload_ids: vec![],
        };

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for mixed codec without precision plan"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("mixed codec without precision plan")),
            "expected 'mixed codec without precision plan' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 7: Non-mixed tensor with MixedPrecision ref → Invalid ─────
    #[test]
    fn test_non_mixed_with_mixed_payload_ref() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Keep codec as Nf4 (non-Mixed) but use a MixedPrecision ref.
        manifest.tensors[0].payload_ref = CImagePayloadRef::MixedPrecision {
            base_payload_id: "p1".to_string(),
            override_table_payload_id: "p1".to_string(),
            sidecar_payload_ids: vec![],
        };

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for non-mixed tensor with mixed payload ref"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("non-mixed tensor") && e.contains("mixed payload ref")),
            "expected 'non-mixed tensor carries mixed payload ref' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 8: Unresolved execution plan tensor ref → Invalid ─────────
    #[test]
    fn test_unresolved_execution_plan_tensor_ref() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        manifest.execution_plan.tensor_refs = vec!["nonexistent_tensor".to_string()];

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for unresolved execution plan tensor ref"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("unresolved tensor ref")),
            "expected 'unresolved tensor ref' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 9: Unresolved receipt ref → Invalid ───────────────────────
    #[test]
    fn test_unresolved_receipt_ref() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        manifest.receipts[0].receipt_id = "nonexistent_receipt".to_string();

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for unresolved receipt ref"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("unresolved receipt ref")),
            "expected 'unresolved receipt ref' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 10: Invalid physical layout → Invalid ─────────────────────
    #[test]
    fn test_invalid_physical_layout() {
        let (
            file_bytes,
            header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // tile_m == 0 makes the layout invalid.
        manifest.tensors[0].physical_layout.tile_m = 0;

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for invalid physical layout"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("invalid physical layout")),
            "expected 'invalid physical layout' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 11: Footer digest mismatch → Invalid ──────────────────────
    #[test]
    fn test_footer_digest_mismatch() {
        let (
            mut file_bytes,
            header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Corrupt the manifest bytes in the raw file so the digest won't
        // match the footer's manifest_sha256.
        let m_start = header.manifest_offset as usize;
        let m_end = m_start + header.manifest_len as usize;
        if m_end <= file_bytes.len() {
            file_bytes[m_start] ^= 0xFF;
        }

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for digest mismatch"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("digest mismatch for manifest")),
            "expected 'digest mismatch for manifest' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 12: Invalid format version → Invalid ──────────────────────
    #[test]
    fn test_unsupported_format_version() {
        let (
            file_bytes,
            mut header,
            _footer,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        header.format_version = 99;

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid for unsupported format version"
        );
        assert!(
            receipt
                .errors
                .iter()
                .any(|e| e.contains("unsupported format version")),
            "expected 'unsupported format version' error, got: {:?}",
            receipt.errors
        );
    }

    // ── Test 13: Many errors collected (no short-circuit) ───────────────
    #[test]
    fn test_multiple_errors_collected() {
        let (
            file_bytes,
            mut header,
            _footer,
            mut manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        ) = build_valid_cimage();

        // Introduce multiple independent errors.
        header.magic = *b"BADMAGIC";
        header.format_version = 99;
        manifest.execution_plan.tensor_refs = vec!["nonexistent".to_string()];

        let loaded = make_loaded(
            file_bytes,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        );

        let receipt = CImageValidator::validate_loaded(&loaded).unwrap();
        assert_eq!(
            receipt.validation_status,
            CImageValidationStatus::Invalid,
            "expected Invalid"
        );
        // Should have at least 3 errors (magic, version, unresolved ref).
        assert!(
            receipt.errors.len() >= 3,
            "expected at least 3 errors, got {}: {:?}",
            receipt.errors.len(),
            receipt.errors
        );
    }
}
