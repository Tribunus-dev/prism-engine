//! Streaming CImage V0 file writer — writes payloads one at a time to disk,
//! avoiding the OOM trap of accumulating all tensors in memory.
//!
//! Layout:
//!   1. Header placeholder (padded, address known)
//!   2. Payload stream: each chunk written directly to disk
//!   3. Manifest JSON (written after all payloads — we know the offsets)
//!   4. Payload directory JSON
//!   5. Receipt directory JSON (if any)
//!   6. Footer
//!
//! On disk the file layout is identical to CImageWriter::write_v0's output,
//! just assembled in a different order.

use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom};

use crate::cimage::*;

/// Streaming cimage V0 builder that writes payloads to disk immediately.
pub struct StreamingCImageWriter {
    path: std::path::PathBuf,
    tmp: tempfile::NamedTempFile,
    payload_entries: Vec<CImagePayloadEntry>,
    receipt_entries: Vec<CImageReceiptEntry>,
    /// Running sha256 of all payload bytes written so far.
    payload_hasher: Sha256,
    /// Blob-relative cursor; starts at 0, increments per payload.
    blob_cursor: u64,
    /// Size of the reserved header (blob starts at this offset in the file).
    header_size: u64,
    /// Current write offset in the file (starts at reserved header).
    write_cursor: u64,
}

impl StreamingCImageWriter {
    /// Create a new streaming writer. Reserves space for the header at offset 0.
    pub fn new(path: &Path) -> CImageResult<Self> {
        let dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let mut tmp = tempfile::NamedTempFile::new_in(dir)
            .map_err(|e| CImageError::Io(format!("create tempfile: {e}")))?;

        // Reserve space for header + padding (will be filled in finalize)
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
        // Offset is blob-relative (not file-relative).
        let offset = self.blob_cursor;
        let len = bytes.len() as u64;

        // Write payload bytes
        self.tmp
            .write_all(bytes)
            .map_err(|e| CImageError::Io(format!("write payload {payload_id}: {e}")))?;

        // Update hashers
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
    pub fn append_receipt(&mut self, receipt_id: String, receipt_kind: String) -> CImageResult<()> {
        let entry = self
            .payload_entries
            .iter()
            .find(|pe| pe.payload_id == receipt_id)
            .ok_or_else(|| CImageError::Other(format!("receipt payload {receipt_id} not found")))?;
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

        // 4. Build header structure (all fields known except file-wide hash).
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

        // 5. Seek back to 0 and write the real header (replaces placeholder).
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

        // 6. Compute file-wide hash from the actual prefix (header now correct).
        self.tmp
            .seek(SeekFrom::Start(0))
            .map_err(|e| CImageError::Io(format!("seek to start for hash: {e}")))?;
        let mut prefix_bytes = vec![0u8; footer_offset as usize];
        self.tmp
            .read_exact(&mut prefix_bytes)
            .map_err(|e| CImageError::Io(format!("read prefix: {e}")))?;
        let cimage_sha256_without_footer = sha256_hex(&prefix_bytes);

        // 7. Write footer with correct file-wide hash.
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let payload_directory_sha256 = sha256_hex(&payload_directory_bytes);
        let receipt_directory_sha256 = sha256_hex(&receipt_directory_bytes);

        let footer = CImageFooterV0 {
            manifest_sha256,
            payload_directory_sha256,
            receipt_directory_sha256,
            payload_blob_sha256,
            cimage_sha256_without_footer,
        };

        let footer_bytes = bincode::serialize(&footer)
            .map_err(|e| CImageError::Io(format!("bincode footer: {e}")))?;
        let footer_size = std::mem::size_of::<CImageFooterV0>() as u64;
        self.tmp
            .write_all(&footer_bytes)
            .map_err(|e| CImageError::Io(format!("write footer: {e}")))?;
        if footer_bytes.len() < footer_size as usize {
            let pad = footer_size as usize - footer_bytes.len();
            self.tmp
                .write_all(&vec![0u8; pad])
                .map_err(|e| CImageError::Io(format!("pad footer: {e}")))?;
        }

        self.tmp
            .flush()
            .map_err(|e| CImageError::Io(format!("flush: {e}")))?;

        // 8. Persist (atomic rename)
        self.tmp
            .persist(&self.path)
            .map_err(|e| CImageError::Io(format!("persist: {e}")))?;

        let final_bytes =
            std::fs::read(&self.path).map_err(|e| CImageError::Io(format!("read final: {e}")))?;
        let file_size_bytes = final_bytes.len() as u64;
        let cimage_digest = sha256_hex(&final_bytes);

        Ok(CImageWriteReceipt {
            path: self.path.to_string_lossy().to_string(),
            file_size_bytes,
            cimage_digest,
            tensor_count: manifest.tensors.len(),
            payload_count: self.payload_entries.len(),
            receipt_count: self.receipt_entries.len(),
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_bytes<T: serde::Serialize>(value: &T) -> CImageResult<Vec<u8>> {
    let mut serializer =
        serde_json::Serializer::with_formatter(Vec::new(), serde_json::ser::PrettyFormatter::new());
    value
        .serialize(&mut serializer)
        .map_err(|e| CImageError::JsonSerialize(e.to_string()))?;
    Ok(serializer.into_inner())
}
