//! CImage types re-defined for the BitNet module's self-contained surface.
//!
//! The bitnet module emits cimage artifacts as its output format, and the
//! existing engine-side cimage types live behind the engine's
//! `compute-core/src/ecs/cimage/` wall. To keep the bitnet module
//! self-contained — i.e. not coupled to engine-internal types that
//! another migration has yet to absorb — the bitnet module ships its
//! own copies of the cimage manifest / payload / writer types it
//! needs.
//!
//! These types are STRUCTURALLY IDENTICAL to the engine's
//! `compute-core/src/ecs/cimage/manifest.rs`,
//! `payload.rs`, `shard_builder.rs`, and `streaming_writer.rs` types.
//! The bitnet module is the canonical authority for BitNet-native
//! cimage emission, so its types are the source of truth for the
//! surface it produces. Engine code that wants to consume bitnet's
//! output (e.g. `CImageWriter::write_v0`) must convert via
//! field-by-field copying or `From` impls in the engine caller.
//!
//! # Migration status
//!
//! This shim is part of the bitnet engine-deletion migration
//! (2026-07-27). When the cimage types themselves are absorbed into
//! the constitutional surface, this shim is expected to be replaced
//! by a `pub use` of the canonical cimage types and the engine
//! becomes a thin consumer of the constitutional cimage surface.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use prism_ecs_ir::cimage_types::CodecFamily;

// ── Error / Result ─────────────────────────────────────────────────────

/// Errors that can occur during cimage operations in the bitnet module.
#[derive(Debug, Clone, Error)]
pub enum CImageError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("json serialization error: {0}")]
    JsonSerialize(String),

    #[error("json deserialization error: {0}")]
    JsonDeserialize(String),

    #[error("sha256 error: {0}")]
    Sha256(String),

    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for CImageError {
    fn from(e: std::io::Error) -> Self {
        CImageError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for CImageError {
    fn from(e: serde_json::Error) -> Self {
        CImageError::JsonSerialize(e.to_string())
    }
}

/// Convenience result alias for bitnet cimage operations.
pub type CImageResult<T> = Result<T, CImageError>;

// ── DType / HardwareProfileId (re-defined for the bitnet shim) ────────

/// Numeric data type for a tensor (subset used by the bitnet module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I8,
    U8,
    I32,
    U32,
}

/// Hardware profile identifier for layout selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareProfileId {
    AppleA18Tiny,
    AppleMBaseMemoryBound,
    AppleMProBalanced,
    AppleMMaxBandwidth,
    AppleMUltraSharded,
}

// ── Manifest / payload / receipt types ─────────────────────────────────

/// Classification of a cimage artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CImageArtifactKind {
    SyntheticShard,
    ModelShard,
    FullModel,
    AssistantGraphProof,
}

/// One tensor entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageTensorEntry {
    pub tensor_id: String,
    pub tensor_key: String,
    pub tensor_class: String,
    pub logical_shape: Vec<u32>,
    pub source_dtype: DType,
    pub codec: CodecFamily,
    /// Precision plan is unused by the bitnet module's emission; kept
    /// here for struct compatibility with the engine's manifest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision_plan: Option<()>,
    pub physical_layout: PhysicalTileLayout,
    pub payload_ref: CImagePayloadRef,
    pub raw_f32_reference_ref: Option<CImagePayloadRef>,
    pub tensor_sha256: String,
    pub validation_digest: Option<String>,
}

/// Reference into the payload directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CImagePayloadRef {
    Single {
        payload_id: String,
    },
    MixedPrecision {
        base_payload_id: String,
        override_table_payload_id: String,
        sidecar_payload_ids: Vec<String>,
    },
}

/// Physical tile layout description for a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTileLayout {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tiles_per_row: u32,
    pub total_tiles: u32,
    pub padded_cols: u32,
    pub group_size: u32,
    pub groups_per_tile: u32,
    pub packed_bytes_per_tile: u32,
    pub metadata_f32_per_tile: u32,
}

impl PhysicalTileLayout {
    /// Validate that the layout is self-consistent.
    pub fn is_valid(&self) -> bool {
        if self.tile_m == 0 {
            return false;
        }
        if self.group_size > 0 {
            if self.tile_n % self.group_size != 0 {
                return false;
            }
            if self.groups_per_tile != self.tile_n / self.group_size {
                return false;
            }
        } else if self.groups_per_tile != 0 {
            return false;
        }
        true
    }
}

/// Summary of the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionPlanSummary {
    pub plan_id: String,
    pub region_count: u32,
    pub total_kernel_ops: u32,
    pub total_input_bytes: u64,
    pub total_output_bytes: u64,
    pub tensor_refs: Vec<String>,
}

/// Reference to a receipt stored in the receipt directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptRef {
    pub receipt_id: String,
    pub receipt_kind: String,
}

/// Reference to an assistant graph JSON payload in the payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantGraphPayloadRef {
    pub graph_json_payload_id: String,
}

/// Reference to a state-store schema JSON payload in the payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateStoreSchemaPayloadRef {
    pub schema_json_payload_id: String,
}

/// V0 manifest: semantic contract for one cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageManifestV0 {
    pub schema_version: u32,
    pub model_family: String,
    pub artifact_kind: CImageArtifactKind,
    pub source_model_digest: Option<String>,
    pub compiler_policy_digest: String,
    pub layout_profile: HardwareProfileId,
    pub tensors: Vec<CImageTensorEntry>,
    pub execution_plan: ModelExecutionPlanSummary,
    pub receipts: Vec<CImageReceiptRef>,
    pub assistant_graph: Option<AssistantGraphPayloadRef>,
    pub state_store_schema: Option<StateStoreSchemaPayloadRef>,
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

/// A pending (unwritten) cimage shard, ready for `CImageWriter`.
#[derive(Debug, Clone)]
pub struct PendingCImageShard {
    pub manifest: CImageManifestV0,
    pub payloads: Vec<PendingPayload>,
    pub receipts: Vec<PendingReceipt>,
}

/// Write receipt emitted by `CImageWriter::write_v0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageWriteReceipt {
    pub path: String,
    pub file_size_bytes: u64,
    pub cimage_digest: String,
    pub tensor_count: usize,
    pub payload_count: usize,
    pub receipt_count: usize,
}

// ── CImage header (small, used by the streaming writer) ───────────────

/// V0 file header (mirrors the engine's `CImageHeaderV0`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CImageHeaderV0 {
    pub magic: [u8; 8],
    pub format_version: u32,
    pub header_len: u64,
    pub manifest_offset: u64,
    pub manifest_len: u64,
    pub payload_directory_offset: u64,
    pub payload_directory_len: u64,
    pub receipt_directory_offset: u64,
    pub receipt_directory_len: u64,
    pub payload_blob_offset: u64,
    pub payload_blob_len: u64,
    pub footer_offset: u64,
}

/// File-wide magic bytes for the cimage V0 format.
pub const CIMAGE_MAGIC: [u8; 8] = *b"CIMG0000";
/// Current cimage format version.
pub const CIMAGE_FORMAT_VERSION: u32 = 0;

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

/// V0 payload directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImagePayloadDirectoryV0 {
    pub payloads: Vec<CImagePayloadEntry>,
}

/// One entry in the receipt directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptEntry {
    pub receipt_id: String,
    pub receipt_kind: String,
    pub offset: u64,
    pub len: u64,
    pub sha256: String,
}

/// V0 receipt directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceiptDirectoryV0 {
    pub receipts: Vec<CImageReceiptEntry>,
}

// ── Streaming writer ──────────────────────────────────────────────────

/// Streaming cimage V0 builder that writes payloads to disk immediately.
pub struct StreamingCImageWriter {
    path: std::path::PathBuf,
    tmp: tempfile::NamedTempFile,
    payload_entries: Vec<CImagePayloadEntry>,
    receipt_entries: Vec<CImageReceiptEntry>,
    payload_hasher: Sha256,
    blob_cursor: u64,
    header_size: u64,
    write_cursor: u64,
}

impl StreamingCImageWriter {
    /// Create a new streaming writer. Reserves space for the header at offset 0.
    pub fn new(path: &Path) -> CImageResult<Self> {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(dir)
            .map_err(|e| CImageError::Io(format!("create tempfile: {e}")))?;

        let header_size = std::mem::size_of::<CImageHeaderV0>() as u64;
        let buf = vec![0u8; header_size as usize];
        tmp.write_all(&buf)
            .map_err(|e| CImageError::Io(format!("reserve header: {e}")))?;

        Ok(Self {
            path: path.to_path_buf(),
            tmp,
            payload_entries: Vec::new(),
            receipt_entries: Vec::new(),
            payload_hasher: Sha256::new(),
            blob_cursor: 0,
            header_size,
            write_cursor: header_size,
        })
    }

    /// Append a raw binary payload to the cimage. Computes offset + sha256.
    pub fn append_payload(
        &mut self,
        payload_id: String,
        payload_kind: CImagePayloadKind,
        codec: Option<String>,
        alignment_bytes: u32,
        bytes: &[u8],
    ) -> CImageResult<()> {
        let offset = self.blob_cursor;
        let len = bytes.len() as u64;

        self.tmp
            .write_all(bytes)
            .map_err(|e| CImageError::Io(format!("write payload {payload_id}: {e}")))?;

        self.payload_hasher.update(bytes);

        self.payload_entries.push(CImagePayloadEntry {
            payload_id,
            payload_kind,
            codec,
            offset,
            len,
            alignment_bytes,
            sha256: sha256_hex(bytes),
        });

        self.write_cursor += len;
        self.blob_cursor += len;
        Ok(())
    }

    /// Add a receipt entry referencing an already-written payload.
    pub fn append_receipt(
        &mut self,
        receipt_id: String,
        receipt_kind: String,
    ) -> CImageResult<()> {
        let entry = self
            .payload_entries
            .iter()
            .find(|pe| pe.payload_id == receipt_id)
            .ok_or_else(|| {
                CImageError::Other(format!("receipt payload {receipt_id} not found"))
            })?;
        self.receipt_entries.push(CImageReceiptEntry {
            receipt_id,
            receipt_kind,
            offset: entry.offset,
            len: entry.len,
            sha256: entry.sha256.clone(),
        });
        Ok(())
    }

    /// Finalize the cimage: write manifest, directories, footer, and persist.
    pub fn finalize(mut self, manifest: CImageManifestV0) -> CImageResult<CImageWriteReceipt> {
        // 1. Serialize manifest
        let manifest_bytes = canonical_json_bytes(&manifest)?;
        let manifest_offset = self.write_cursor;
        let manifest_len = manifest_bytes.len() as u64;
        self.tmp
            .write_all(&manifest_bytes)
            .map_err(|e| CImageError::Io(format!("write manifest: {e}")))?;
        self.write_cursor += manifest_len;

        // 2. Serialize payload directory
        let payload_directory = CImagePayloadDirectoryV0 {
            payloads: self.payload_entries.clone(),
        };
        let payload_directory_bytes = canonical_json_bytes(&payload_directory)?;
        let payload_directory_offset = self.write_cursor;
        let payload_directory_len = payload_directory_bytes.len() as u64;
        self.tmp
            .write_all(&payload_directory_bytes)
            .map_err(|e| CImageError::Io(format!("write payload dir: {e}")))?;
        self.write_cursor += payload_directory_len;

        // 3. Serialize receipt directory
        let receipt_directory = CImageReceiptDirectoryV0 {
            receipts: self.receipt_entries.clone(),
        };
        let receipt_directory_bytes = canonical_json_bytes(&receipt_directory)?;
        let receipt_directory_offset = self.write_cursor;
        let receipt_directory_len = receipt_directory_bytes.len() as u64;
        self.tmp
            .write_all(&receipt_directory_bytes)
            .map_err(|e| CImageError::Io(format!("write receipt dir: {e}")))?;
        self.write_cursor += receipt_directory_len;

        // 4. Build header
        let payload_blob_offset = std::mem::size_of::<CImageHeaderV0>() as u64;
        let payload_blob_len = manifest_offset - payload_blob_offset;
        let payload_blob_sha256 = format!("{:x}", self.payload_hasher.finalize());

        let footer_offset = self.write_cursor;
        let header = CImageHeaderV0 {
            magic: CIMAGE_MAGIC,
            format_version: CIMAGE_FORMAT_VERSION,
            header_len: std::mem::size_of::<CImageHeaderV0>() as u64,
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

        let header_bytes = bincode::serialize(&header)
            .map_err(|e| CImageError::Io(format!("bincode header (final): {e}")))?;

        // 5. Seek back and write the real header.
        self.tmp
            .seek(SeekFrom::Start(0))
            .map_err(|e| CImageError::Io(format!("seek to header: {e}")))?;
        self.tmp
            .write_all(&header_bytes)
            .map_err(|e| CImageError::Io(format!("write final header: {e}")))?;
        if header_bytes.len() < self.header_size as usize {
            let pad = self.header_size as usize - header_bytes.len();
            self.tmp
                .write_all(&vec![0u8; pad])
                .map_err(|e| CImageError::Io(format!("pad header: {e}")))?;
        }

        // 6. Compute file-wide hash from the actual prefix.
        self.tmp
            .seek(SeekFrom::Start(0))
            .map_err(|e| CImageError::Io(format!("seek to start for hash: {e}")))?;
        let mut prefix_bytes = vec![0u8; footer_offset as usize];
        self.tmp
            .read_exact(&mut prefix_bytes)
            .map_err(|e| CImageError::Io(format!("read prefix: {e}")))?;
        let mut prefix_hasher = Sha256::new();
        prefix_hasher.update(&prefix_bytes);
        let cimage_digest = format!("{:x}", prefix_hasher.finalize());

        // 7. Append receipt-directory-style footer with the cimage digest.
        let footer = serde_json::json!({
            "cimage_digest": cimage_digest,
            "payload_blob_sha256": payload_blob_sha256,
            "tensor_count": manifest.tensors.len(),
            "payload_count": self.payload_entries.len(),
            "receipt_count": self.receipt_entries.len(),
        });
        let footer_bytes = serde_json::to_vec(&footer)
            .map_err(|e| CImageError::JsonSerialize(e.to_string()))?;
        let footer_len = footer_bytes.len() as u64;
        self.tmp
            .write_all(&footer_bytes)
            .map_err(|e| CImageError::Io(format!("write footer: {e}")))?;

        let file_size = self.write_cursor + footer_len;

        // 8. Persist to final path.
        self.tmp
            .flush()
            .map_err(|e| CImageError::Io(format!("flush: {e}")))?;
        self.tmp
            .persist(&self.path)
            .map_err(|e| CImageError::Io(format!("persist: {e}")))?;

        Ok(CImageWriteReceipt {
            path: self.path.to_string_lossy().to_string(),
            file_size_bytes: file_size,
            cimage_digest,
            tensor_count: manifest.tensors.len(),
            payload_count: self.payload_entries.len(),
            receipt_count: self.receipt_entries.len(),
        })
    }
}

// ── Helpers (ported from engine) ──────────────────────────────────────

/// Compute SHA-256 hex digest of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Compute canonical JSON bytes for serialization (sorts object keys).
pub fn canonical_json_bytes<T: Serialize>(value: &T) -> CImageResult<Vec<u8>> {
    let json = serde_json::to_string(value)?;
    let value: serde_json::Value = serde_json::from_str(&json)?;
    let canonical = serde_json::to_vec(&canonicalize_value(&value))?;
    Ok(canonical)
}

fn canonicalize_value(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize_value(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut sorted = serde_json::Map::new();
            for (k, v) in entries {
                sorted.insert(k, v);
            }
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize_value).collect()),
        other => other.clone(),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_tile_layout_validates_grouped() {
        let ok = PhysicalTileLayout {
            tile_m: 1,
            tile_n: 64,
            tiles_per_row: 1,
            total_tiles: 1,
            padded_cols: 64,
            group_size: 32,
            groups_per_tile: 2,
            packed_bytes_per_tile: 16,
            metadata_f32_per_tile: 0,
        };
        assert!(ok.is_valid());

        // tile_n not divisible by group_size
        let bad = PhysicalTileLayout {
            tile_n: 30,
            group_size: 32,
            groups_per_tile: 2,
            ..ok.clone()
        };
        assert!(!bad.is_valid());

        // groups_per_tile doesn't match tile_n / group_size
        let bad2 = PhysicalTileLayout {
            tile_n: 64,
            group_size: 32,
            groups_per_tile: 3,
            ..ok
        };
        assert!(!bad2.is_valid());
    }

    #[test]
    fn payload_ref_serde_roundtrip() {
        let r = CImagePayloadRef::Single {
            payload_id: "p_t0".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: CImagePayloadRef = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn sha256_hex_known_value() {
        // SHA-256 of empty input
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn canonical_json_sorts_keys() {
        let v = serde_json::json!({"b": 1, "a": 2});
        let bytes = canonical_json_bytes(&v).unwrap();
        let s = String::from_utf8(bytes).unwrap();
        // 'a' must come before 'b' in canonical output
        assert!(s.find("\"a\"").unwrap() < s.find("\"b\"").unwrap());
    }
}
