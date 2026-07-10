//! CImage V0 file loader.
//!
//! Reads a cimage V0 file from disk, validates structure, and returns a
//! [`LoadedCImageV0`] with all sections deserialized and in-memory.

use crate::cimage::{
    CImageError, CImageHeaderV0, CImageManifestV0, CImagePayloadDirectoryV0,
    CImageReceiptDirectoryV0, CImageResult,
};
use std::fs::File;
use std::path::Path;

#[derive(Debug)]
pub struct LoadedCImageV0 {
    /// Original filesystem path.
    pub path: std::path::PathBuf,
    /// Memory-mapped backing of the entire file, retained for zero-copy
    /// access to payload data when creating Metal buffers.
    pub _mmap: Option<memmap2::Mmap>,
    /// Offset within the mmap to the start of the payload blob (for
    /// no-copy GPU buffer creation).
    pub payload_mmap_offset: usize,
    /// Length of the payload blob within the mmap.
    pub payload_mmap_len: usize,
    /// The raw bytes of the entire file, kept for re-validation or digesting.
    pub raw_file_bytes: Vec<u8>,
    /// Deserialized fixed-size header (bincode).
    pub header: CImageHeaderV0,
    /// Deserialized manifest (JSON).
    pub manifest: CImageManifestV0,
    /// Deserialized payload directory (JSON).
    pub payload_directory: CImagePayloadDirectoryV0,
    /// Deserialized receipt directory (JSON).
    pub receipt_directory: CImageReceiptDirectoryV0,
    /// Contiguous payload blob (raw bytes).
    pub payload_blob: Vec<u8>,
}

impl Clone for LoadedCImageV0 {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            _mmap: None, // mmap is not Cloneable; reset to None
            payload_mmap_offset: self.payload_mmap_offset,
            payload_mmap_len: self.payload_mmap_len,
            raw_file_bytes: self.raw_file_bytes.clone(),
            header: self.header.clone(),
            manifest: self.manifest.clone(),
            payload_directory: self.payload_directory.clone(),
            receipt_directory: self.receipt_directory.clone(),
            payload_blob: self.payload_blob.clone(),
        }
    }
}

/// Stateless loader for cimage V0 artifacts.
pub struct CImageLoader;

impl CImageLoader {
    /// Load and deserialize a cimage V0 file.
    ///
    /// # Errors
    ///
    /// Returns [`CImageError::Io`] when the file cannot be opened or read,
    /// [`CImageError::InvalidMagic`] when the magic bytes are wrong,
    /// [`CImageError::UnsupportedFormatVersion`] when the format version is not 0,
    /// [`CImageError::RangeOutOfBounds`] when any offset + len exceeds the file,
    /// [`CImageError::JsonDeserialize`] when a JSON section cannot be parsed.
    pub fn load_v0(path: &Path) -> CImageResult<LoadedCImageV0> {
        let file = File::open(path).map_err(|e| CImageError::Io(e.to_string()))?;
        let mmap =
            unsafe { memmap2::Mmap::map(&file).map_err(|e| CImageError::Io(e.to_string()))? };
        drop(file);
        let raw = mmap[..].to_vec();
        let file_size = raw.len() as u64;
        let path = path.to_path_buf();

        // ── Header ─────────────────────────────────────────────────────
        let header_size = std::mem::size_of::<CImageHeaderV0>();
        if raw.len() < header_size {
            return Err(CImageError::RangeOutOfBounds {
                offset: 0,
                len: header_size as u64,
                file_size,
            });
        }
        let header: CImageHeaderV0 = bincode::deserialize(&raw[..header_size])
            .map_err(|e| CImageError::Io(format!("bincode header deserialize: {e}")))?;

        // Validate magic and format version.
        if header.magic != crate::cimage::CIMAGE_MAGIC {
            return Err(CImageError::InvalidMagic {
                expected: crate::cimage::CIMAGE_MAGIC,
                got: header.magic.to_vec(),
            });
        }
        if !header.supports_format() {
            return Err(CImageError::UnsupportedFormatVersion(header.format_version));
        }

        // ── Helper: read a JSON section ────────────────────────────────
        fn read_json_section<T>(
            data: &[u8],
            offset: u64,
            len: u64,
            file_size: u64,
            label: &str,
        ) -> CImageResult<T>
        where
            T: serde::de::DeserializeOwned,
        {
            // Range sanity check.
            if len == 0 {
                return Err(CImageError::RangeOutOfBounds {
                    offset,
                    len,
                    file_size,
                });
            }
            let end = offset
                .checked_add(len)
                .ok_or(CImageError::RangeOutOfBounds {
                    offset,
                    len,
                    file_size,
                })?;
            if end > file_size {
                return Err(CImageError::RangeOutOfBounds {
                    offset,
                    len,
                    file_size,
                });
            }
            let start_usize = offset as usize;
            let end_usize = end as usize;
            let bytes = &data[start_usize..end_usize];
            serde_json::from_slice(bytes)
                .map_err(|e| CImageError::JsonDeserialize(format!("{label}: {e}")))
        }

        // ── Manifest ───────────────────────────────────────────────────
        let manifest: CImageManifestV0 = read_json_section(
            &raw,
            header.manifest_offset,
            header.manifest_len,
            file_size,
            "manifest",
        )?;

        // ── Payload directory ──────────────────────────────────────────
        let payload_directory: CImagePayloadDirectoryV0 = read_json_section(
            &raw,
            header.payload_directory_offset,
            header.payload_directory_len,
            file_size,
            "payload_directory",
        )?;

        // ── Receipt directory ──────────────────────────────────────────
        let receipt_directory: CImageReceiptDirectoryV0 = read_json_section(
            &raw,
            header.receipt_directory_offset,
            header.receipt_directory_len,
            file_size,
            "receipt_directory",
        )?;

        // ── Payload blob ───────────────────────────────────────────────
        // Payload blob MAY be empty (len == 0 is allowed for degenerate files).
        let payload_blob = if header.payload_blob_len == 0 {
            Vec::new()
        } else {
            let end = header
                .payload_blob_offset
                .checked_add(header.payload_blob_len)
                .ok_or(CImageError::RangeOutOfBounds {
                    offset: header.payload_blob_offset,
                    len: header.payload_blob_len,
                    file_size,
                })?;
            if end > file_size {
                return Err(CImageError::RangeOutOfBounds {
                    offset: header.payload_blob_offset,
                    len: header.payload_blob_len,
                    file_size,
                });
            }
            let start_usize = header.payload_blob_offset as usize;
            let end_usize = end as usize;
            raw[start_usize..end_usize].to_vec()
        };

        Ok(LoadedCImageV0 {
            path,
            _mmap: Some(mmap),
            payload_mmap_offset: header.payload_blob_offset as usize,
            payload_mmap_len: header.payload_blob_len as usize,
            raw_file_bytes: raw,
            header,
            manifest,
            payload_directory,
            receipt_directory,
            payload_blob,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cimage::{
        CImageArtifactKind, CImageHeaderV0, CImageManifestV0, CImagePayloadDirectoryV0,
        CImagePayloadEntry, CImagePayloadKind, CImageReceiptDirectoryV0, CImageReceiptEntry,
        ModelExecutionPlanSummary, CIMAGE_FORMAT_VERSION, CIMAGE_MAGIC,
    };
    use crate::execution_plan::HardwareProfileId;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal valid cimage V0 in memory bytes, covering all sections.
    fn build_minimal_cimage_bytes() -> Vec<u8> {
        // Sections laid out sequentially:
        //  [header]  [manifest JSON]  [payload_dir JSON]  [receipt_dir JSON]  [payload blob]

        let manifest = CImageManifestV0 {
            schema_version: 1,
            model_family: "test".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "abc123".into(),
            layout_profile: HardwareProfileId::AppleA18Tiny,
            tensors: vec![],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "plan-0".into(),
                region_count: 0,
                total_kernel_ops: 0,
                total_input_bytes: 0,
                total_output_bytes: 0,
                tensor_refs: vec![],
            },
            receipts: vec![],
            assistant_graph: None,
            state_store_schema: None,
        };

        let payload_dir = CImagePayloadDirectoryV0 {
            payloads: vec![CImagePayloadEntry {
                payload_id: "p1".into(),
                payload_kind: CImagePayloadKind::PackedTensorCodes,
                codec: None,
                offset: 0,
                len: 4,
                alignment_bytes: 1,
                sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
            }],
        };

        let receipt_dir = CImageReceiptDirectoryV0 {
            receipts: vec![CImageReceiptEntry {
                receipt_id: "r1".into(),
                receipt_kind: "test".into(),
                offset: 0,
                len: 0,
                sha256: "".into(),
            }],
        };

        let payload_blob: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];

        let manifest_bytes = serde_json::to_vec(&manifest).expect("json manifest");
        let payload_dir_bytes = serde_json::to_vec(&payload_dir).expect("json payload_dir");
        let receipt_dir_bytes = serde_json::to_vec(&receipt_dir).expect("json receipt_dir");

        let header_size = std::mem::size_of::<CImageHeaderV0>();

        // Compute offsets (after header).
        let manifest_offset = header_size as u64;
        let manifest_len = manifest_bytes.len() as u64;

        let payload_directory_offset = manifest_offset + manifest_len;
        let payload_directory_len = payload_dir_bytes.len() as u64;

        let receipt_directory_offset = payload_directory_offset + payload_directory_len;
        let receipt_directory_len = receipt_dir_bytes.len() as u64;

        let payload_blob_offset = receipt_directory_offset + receipt_directory_len;
        let payload_blob_len = payload_blob.len() as u64;

        let footer_offset = payload_blob_offset + payload_blob_len;

        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: header_size as u64,
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

        let header_bytes = bincode::serialize(&header).expect("bincode header");

        let mut buf = Vec::new();
        // Pad header bytes to size_of — bincode omits trailing alignment padding.
        buf.extend_from_slice(&header_bytes);
        buf.extend(std::iter::repeat(0u8).take(header_size as usize - header_bytes.len()));
        buf.extend_from_slice(&manifest_bytes);
        buf.extend_from_slice(&payload_dir_bytes);
        buf.extend_from_slice(&receipt_dir_bytes);
        buf.extend_from_slice(&payload_blob);
        buf
    }

    #[test]
    fn roundtrip_v0() {
        let bytes = build_minimal_cimage_bytes();

        // Write to temp file.
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&bytes).expect("write");
        let path = tmp.path().to_path_buf();

        // Load.
        let loaded = CImageLoader::load_v0(&path).expect("load_v0");

        // Verify all sections.
        assert_eq!(loaded.raw_file_bytes, bytes);
        assert_eq!(loaded.header.magic, CIMAGE_MAGIC);
        assert_eq!(loaded.header.format_version, CIMAGE_FORMAT_VERSION);
        assert_eq!(loaded.manifest.schema_version, 1);
        assert_eq!(loaded.manifest.model_family, "test");
        assert_eq!(loaded.payload_directory.payloads.len(), 1);
        assert_eq!(loaded.payload_directory.payloads[0].payload_id, "p1");
        assert_eq!(loaded.receipt_directory.receipts.len(), 1);
        assert_eq!(loaded.receipt_directory.receipts[0].receipt_id, "r1");
        assert_eq!(loaded.payload_blob, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut bytes = build_minimal_cimage_bytes();
        // Corrupt magic.
        bytes[0..4].copy_from_slice(b"BAD!");
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&bytes).expect("write");
        let err = CImageLoader::load_v0(tmp.path()).unwrap_err();
        assert!(
            matches!(&err, CImageError::InvalidMagic { .. }),
            "expected InvalidMagic, got {err}"
        );
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut bytes = build_minimal_cimage_bytes();
        // Patch format_version field (starts at byte 8 in the header).
        let header_size = std::mem::size_of::<CImageHeaderV0>();
        // CImageHeaderV0: magic[0..8], format_version[8..12] (u32 LE)
        let version_bytes = 42u32.to_le_bytes();
        bytes[8..12].copy_from_slice(&version_bytes);
        // Recompute bincode won't work trivially; instead rebuild with changed version.

        // Rebuild header with different version.
        let manifest_offset = header_size as u64;
        let manifest = CImageManifestV0 {
            schema_version: 1,
            model_family: "test".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "abc".into(),
            layout_profile: HardwareProfileId::AppleA18Tiny,
            tensors: vec![],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "p".into(),
                region_count: 0,
                total_kernel_ops: 0,
                total_input_bytes: 0,
                total_output_bytes: 0,
                tensor_refs: vec![],
            },
            receipts: vec![],
            assistant_graph: None,
            state_store_schema: None,
        };
        let payload_dir = CImagePayloadDirectoryV0 { payloads: vec![] };
        let receipt_dir = CImageReceiptDirectoryV0 { receipts: vec![] };

        let mb = serde_json::to_vec(&manifest).unwrap();
        let pb = serde_json::to_vec(&payload_dir).unwrap();
        let rb = serde_json::to_vec(&receipt_dir).unwrap();

        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: 42, // unsupported
            header_len: header_size as u64,
            manifest_offset,
            manifest_len: mb.len() as u64,
            payload_directory_offset: manifest_offset + mb.len() as u64,
            payload_directory_len: pb.len() as u64,
            receipt_directory_offset: manifest_offset + mb.len() as u64 + pb.len() as u64,
            receipt_directory_len: rb.len() as u64,
            payload_blob_offset: manifest_offset
                + mb.len() as u64
                + pb.len() as u64
                + rb.len() as u64,
            payload_blob_len: 0,
            footer_offset: 0,
        };

        let hb = bincode::serialize(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&hb);
        buf.extend(std::iter::repeat(0u8).take(header_size - hb.len()));
        buf.extend_from_slice(&mb);
        buf.extend_from_slice(&pb);
        buf.extend_from_slice(&rb);

        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&buf).expect("write");
        let err = CImageLoader::load_v0(tmp.path()).unwrap_err();
        assert!(
            matches!(&err, CImageError::UnsupportedFormatVersion(v) if *v == 42),
            "expected UnsupportedFormatVersion(42), got {err}"
        );
    }

    #[test]
    fn range_out_of_bounds_rejected() {
        // Truncated file: only header, no manifest.
        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: std::mem::size_of::<CImageHeaderV0>() as u64,
            manifest_offset: std::mem::size_of::<CImageHeaderV0>() as u64,
            manifest_len: 100, // non-zero but file doesn't contain this
            ..Default::default()
        };
        let hb = bincode::serialize(&header).unwrap();
        let header_size = std::mem::size_of::<CImageHeaderV0>();
        let mut tmp = NamedTempFile::new().expect("tempfile");
        let mut padded = hb.clone();
        padded.extend(std::iter::repeat(0u8).take(header_size - hb.len()));
        tmp.write_all(&padded).expect("write");

        let err = CImageLoader::load_v0(tmp.path()).unwrap_err();
        assert!(
            matches!(&err, CImageError::RangeOutOfBounds { .. }),
            "expected RangeOutOfBounds, got {err}"
        );
    }

    #[test]
    fn empty_payload_blob_allowed() {
        // Build a cimage with zero-length payload blob (degenerate but valid).
        let header_size = std::mem::size_of::<CImageHeaderV0>();
        let manifest = CImageManifestV0 {
            schema_version: 1,
            model_family: "empty".into(),
            artifact_kind: CImageArtifactKind::SyntheticShard,
            source_model_digest: None,
            compiler_policy_digest: "x".into(),
            layout_profile: HardwareProfileId::AppleA18Tiny,
            tensors: vec![],
            execution_plan: ModelExecutionPlanSummary {
                plan_id: "p".into(),
                region_count: 0,
                total_kernel_ops: 0,
                total_input_bytes: 0,
                total_output_bytes: 0,
                tensor_refs: vec![],
            },
            receipts: vec![],
            assistant_graph: None,
            state_store_schema: None,
        };
        let payload_dir = CImagePayloadDirectoryV0 { payloads: vec![] };
        let receipt_dir = CImageReceiptDirectoryV0 { receipts: vec![] };

        let mb = serde_json::to_vec(&manifest).unwrap();
        let pb = serde_json::to_vec(&payload_dir).unwrap();
        let rb = serde_json::to_vec(&receipt_dir).unwrap();

        let mo = header_size as u64;
        let po = mo + mb.len() as u64;
        let ro = po + pb.len() as u64;
        let bo = ro + rb.len() as u64; // payload_blob_offset, but payload_blob_len = 0

        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: header_size as u64,
            manifest_offset: mo,
            manifest_len: mb.len() as u64,
            payload_directory_offset: po,
            payload_directory_len: pb.len() as u64,
            receipt_directory_offset: ro,
            receipt_directory_len: rb.len() as u64,
            payload_blob_offset: bo,
            payload_blob_len: 0,
            footer_offset: bo,
        };

        let hb = bincode::serialize(&header).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&hb);
        buf.extend(std::iter::repeat(0u8).take(header_size - hb.len()));
        buf.extend_from_slice(&mb);
        buf.extend_from_slice(&pb);
        buf.extend_from_slice(&rb);

        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(&buf).expect("write");
        let loaded = CImageLoader::load_v0(tmp.path()).expect("load empty payload");
        assert!(loaded.payload_blob.is_empty());
    }
}
