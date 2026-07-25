//! .cimage: Hardware-native memory dump format for Apple Silicon.
//!
//! Every tensor payload starts on a 16 KB page boundary, enabling
//! zero-copy `mmap` directly into an IOSurface arena.  No parsing or
//! deserialization of the payload body at load time.
//!
//! Layout per file:
//! ┌─ Magic: "TRB_CIMG" (8B)
//! ├─ Header size: u64 LE (8B)
//! ├─ JSON header (variable, padded to 16 KB)
//! ├─ Padding to 16 KB boundary
//! ├─ Tensor 0 payload (16 KB aligned)
//! ├─ Padding to next 16 KB boundary
//! ├─ Tensor 1 payload
//! └─ ...

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// Apple Silicon page size — required for zero-copy IOSurface mmap.
const PAGE_SIZE: u64 = 16384;
/// Header reservation: enough for ~500 tensor entries (typical 12B model).
const HEADER_PAGES: u64 = 8; // 128 KB

/// Magic identifier for .cimage files.
const MAGIC: &[u8; 8] = b"TRB_CIMG";

// ── Header types ────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum TensorType {
    Bf16,
    Int8,
    Nf8,
    StandardFP16,
    Palettized4Bit,
    Blob,
    /// Ternary (Ternary158) — 2-bit ternary encoding per value.
    Ternary158,
    /// Binary — 1-bit binary encoding per value.
    Binary1,
    /// NF4 — 4-bit NormalFloat with per-group scale/bias.
    NF4,
    /// Int4 — symmetric 4-bit quantization with per-group scale.
    Int4,
    TernaryTile640,
    /// FP8 — 8-bit floating point.
    FP8,
}

impl TensorType {
    /// Stable single-byte discriminant for content addressing.
    ///
    /// The order of variants in this enum is part of the public ABI of
    /// the CImage format. Do not reorder existing variants — append new
    /// variants at the end with a fresh discriminant byte.
    pub fn discriminant_byte(&self) -> [u8; 1] {
        let n: u8 = match self {
            TensorType::Bf16 => 0,
            TensorType::Int8 => 1,
            TensorType::Nf8 => 2,
            TensorType::StandardFP16 => 3,
            TensorType::Palettized4Bit => 4,
            TensorType::Blob => 5,
            TensorType::Ternary158 => 6,
            TensorType::Binary1 => 7,
            TensorType::NF4 => 8,
            TensorType::Int4 => 9,
            TensorType::TernaryTile640 => 10,
            TensorType::FP8 => 11,
        };
        [n]
    }

    /// Reverse mapping for `discriminant_byte`. Returns `None` for an
    /// unknown byte so callers can detect stale or corrupted schema
    /// versions without panicking.
    pub fn from_discriminant_byte(byte: u8) -> Option<TensorType> {
        match byte {
            0 => Some(TensorType::Bf16),
            1 => Some(TensorType::Int8),
            2 => Some(TensorType::Nf8),
            3 => Some(TensorType::StandardFP16),
            4 => Some(TensorType::Palettized4Bit),
            5 => Some(TensorType::Blob),
            6 => Some(TensorType::Ternary158),
            7 => Some(TensorType::Binary1),
            8 => Some(TensorType::NF4),
            9 => Some(TensorType::Int4),
            10 => Some(TensorType::TernaryTile640),
            11 => Some(TensorType::FP8),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorRecord {
    pub tensor_type: TensorType,
    /// Byte offset from start of file (always 16 KB aligned).
    pub offset: u64,
    /// Total payload byte size.
    pub size: u64,
    /// Output dimension (number of rows).
    pub dim_m: u32,
    /// Input dimension (number of columns).
    pub dim_n: u32,
}

impl TensorRecord {
    pub fn ternary_tile640_layout(&self) -> Result<(usize, usize, usize, usize), String> {
        let elements = (self.dim_m as usize)
            .checked_mul(self.dim_n as usize)
            .ok_or("tensor dimensions overflow")?;
        let pages = elements.div_ceil(640);
        Ok((
            elements.div_ceil(4),
            pages * 2,
            elements.div_ceil(16) * 2,
            pages,
        ))
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CImageHeader {
    pub tensors: HashMap<String, TensorRecord>,
    /// Optional execution plan (serialized JSON) for heterogeneous routing.
    /// Contains per-layer OperationRoute assignments and ANE fused islands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<String>,
    #[serde(default)]
    pub kernels: HashMap<String, TensorRecord>,
}

// ── Writer ──────────────────────────────────────────────────────────────

/// Streaming writer for .cimage files.
///
/// Each `append_*` call:
/// 1. Pads the file to the next 16 KB boundary.
/// 2. Writes the payload at that aligned offset.
/// 3. Records the offset + size in the header.
///
/// `finalize()` seeks back to offset 0 and writes the header.
pub struct CImageWriter {
    file: File,
    header: CImageHeader,
}

impl CImageWriter {
    /// Create a new .cimage file at `path`.
    ///
    /// Reserves the first 128 KB for the header (configurable via HEADER_PAGES).
    pub fn new(path: &Path) -> Result<Self, String> {
        let mut file = File::create(path).map_err(|e| format!("create .cimage: {e}"))?;
        // Reserve first HEADER_PAGES × PAGE_SIZE for the header
        let header_bytes = (HEADER_PAGES * PAGE_SIZE) as usize;
        let zeros = vec![0u8; header_bytes];
        file.write_all(&zeros)
            .map_err(|e| format!("reserve header: {e}"))?;
        // Seek to end of header block so append starts at the next page
        file.seek(SeekFrom::Start(header_bytes as u64))
            .map_err(|e| format!("seek: {e}"))?;
        Ok(CImageWriter {
            file,
            header: CImageHeader::default(),
        })
    }

    /// Write a palettized split-block payload.
    ///
    /// Payload layout expected:
    ///   [codebook_block: dim_m × 16 × 2 bytes]
    ///   [indices_block:  dim_m × dim_n/2 bytes]
    pub fn append_palettized(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
    ) -> Result<(), String> {
        self.append(name, payload, dim_m, dim_n, TensorType::Palettized4Bit)
    }

    /// Write a tensor payload with the given tensor type.
    ///
    /// The payload format is specific to the tensor type and must be
    /// understood by the runtime loader.
    pub fn append(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
        tensor_type: TensorType,
    ) -> Result<(), String> {
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(payload)
            .map_err(|e| format!("write payload: {e}"))?;
        self.header.tensors.insert(
            name.to_string(),
            TensorRecord {
                tensor_type,
                offset,
                size: payload.len() as u64,
                dim_m,
                dim_n,
            },
        );
        Ok(())
    }

    /// Write a standard FP16 tensor payload.
    pub fn append_fp16(
        &mut self,
        name: &str,
        payload: &[u8],
        dim_m: u32,
        dim_n: u32,
    ) -> Result<(), String> {
        self.align_to_page()?;
        let offset = self.current_pos()?;
        self.file
            .write_all(payload)
            .map_err(|e| format!("write payload: {e}"))?;
        self.header.tensors.insert(
            name.to_string(),
            TensorRecord {
                tensor_type: TensorType::StandardFP16,
                offset,
                size: payload.len() as u64,
                dim_m,
                dim_n,
            },
        );
        Ok(())
    }

    /// Finalize: write magic + header to the first 16 KB block.
    /// Set the execution plan JSON to embed in the CImage header.
    pub fn set_execution_plan(&mut self, plan_json: String) {
        self.header.execution_plan = Some(plan_json);
    }

    /// Finalize: write magic + header to the first 16 KB block.
    pub fn finalize(mut self) -> Result<(), String> {
        let header_json =
            serde_json::to_string(&self.header).map_err(|e| format!("serialize header: {e}"))?;
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;

        // Must fit in the reserved header block
        let reserved = HEADER_PAGES * PAGE_SIZE;
        assert!(
            16 + header_size <= reserved,
            "Header ({} B) exceeds reserved {} B",
            16 + header_size,
            reserved
        );

        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| format!("seek to start: {e}"))?;
        self.file
            .write_all(MAGIC)
            .map_err(|e| format!("write magic: {e}"))?;
        self.file
            .write_all(&header_size.to_le_bytes())
            .map_err(|e| format!("write header size: {e}"))?;
        self.file
            .write_all(header_bytes)
            .map_err(|e| format!("write header: {e}"))?;
        self.file.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    /// Pad file to the next 16 KB boundary.
    fn align_to_page(&mut self) -> Result<(), String> {
        let pos = self.current_pos()?;
        let remainder = pos % PAGE_SIZE;
        if remainder != 0 {
            let pad = (PAGE_SIZE - remainder) as usize;
            let zeros = vec![0u8; pad];
            self.file
                .write_all(&zeros)
                .map_err(|e| format!("align padding: {e}"))?;
        }
        Ok(())
    }

    fn current_pos(&mut self) -> Result<u64, String> {
        self.file
            .stream_position()
            .map_err(|e| format!("stream position: {e}"))
    }
}

// ── Reader (runtime loader) ─────────────────────────────────────────────

/// Loaded .cimage header (disk metadata read without payload).
pub struct CImageReader {
    pub header: CImageHeader,
    pub(crate) _file: File,
}

impl CImageReader {
    pub fn load_kernel(&self, _name: &str) -> Result<Vec<u8>, String> {
        Err("embedded kernels are not present in this cimage format".into())
    }
    /// Open a .cimage file and parse the header.
    pub fn open(path: &Path) -> Result<Self, String> {
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| format!("open .cimage: {e}"))?;

        // Read magic
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if &magic != MAGIC {
            return Err(format!("Invalid magic: expected TRB_CIMG, got {:?}", magic));
        }

        // Read header size
        let mut hdr_size_bytes = [0u8; 8];
        file.read_exact(&mut hdr_size_bytes)
            .map_err(|e| format!("read header size: {e}"))?;
        let hdr_size = u64::from_le_bytes(hdr_size_bytes) as usize;

        // Read JSON header
        let mut hdr_buf = vec![0u8; hdr_size];
        file.read_exact(&mut hdr_buf)
            .map_err(|e| format!("read header: {e}"))?;
        let header: CImageHeader =
            serde_json::from_slice(&hdr_buf).map_err(|e| format!("parse header: {e}"))?;

        Ok(CImageReader {
            header,
            _file: file,
        })
    }

    /// Return the offset + size for a named tensor.
    pub fn tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.header.tensors.get(name)
    }
}

/// Append a blob payload to an already-finalized .cimage file.
pub fn cimage_append_blob(
    path: &std::path::Path,
    name: &str,
    payload: &[u8],
) -> Result<(), String> {
    use std::io::{Seek, Write};
    let reader = CImageReader::open(path)?;
    let end_offset = reader
        .header
        .tensors
        .values()
        .map(|r| r.offset + r.size)
        .max()
        .unwrap_or(HEADER_PAGES * PAGE_SIZE);
    let aligned = end_offset.div_ceil(PAGE_SIZE) * PAGE_SIZE;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("open: {e}"))?;
    file.seek(SeekFrom::Start(aligned))
        .map_err(|e| format!("seek: {e}"))?;
    file.write_all(payload)
        .map_err(|e| format!("write blob: {e}"))?;
    let mut header = reader.header;
    header.tensors.insert(
        name.to_string(),
        TensorRecord {
            tensor_type: TensorType::Blob,
            offset: aligned,
            size: payload.len() as u64,
            dim_m: 0,
            dim_n: 0,
        },
    );
    let hdr_json = serde_json::to_string(&header).map_err(|e| format!("serialize: {e}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| format!("seek: {e}"))?;
    file.write_all(MAGIC)
        .map_err(|e| format!("write magic: {e}"))?;
    let hdr_size = hdr_json.len() as u64;
    file.write_all(&hdr_size.to_le_bytes())
        .map_err(|e| format!("write size: {e}"))?;
    file.write_all(hdr_json.as_bytes())
        .map_err(|e| format!("write json: {e}"))?;
    file.flush().map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_roundtrip.cimage");
        let _path_str = path.to_str().unwrap().to_string();

        // Write
        {
            let mut writer = CImageWriter::new(&path).unwrap();
            // Palettized payload: 2 rows × (16 codebook × 2B + 4 indices)
            let payload = (0..(2 * 32 + 2 * 4) as u8).collect::<Vec<_>>();
            writer
                .append_palettized("layer.0.wq", &payload, 2, 8)
                .unwrap();
            writer.finalize().unwrap();
        }

        // Read back
        let reader = CImageReader::open(&path).unwrap();
        let rec = reader.tensor("layer.0.wq").unwrap();
        assert_eq!(rec.dim_m, 2);
        assert_eq!(rec.dim_n, 8);
        assert!(matches!(rec.tensor_type, TensorType::Palettized4Bit));
        assert!(
            rec.offset.is_multiple_of(PAGE_SIZE),
            "offset {} not page-aligned",
            rec.offset
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_header_within_first_page() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_header_page.cimage");
        {
            let mut writer = CImageWriter::new(&path).unwrap();
            // Many tensors to fill the header
            for i in 0..50 {
                let name = format!("layer.{i}.wq");
                let payload = vec![0u8; 64];
                writer.append_fp16(&name, &payload, 1, 1).unwrap();
            }
            writer.finalize().unwrap();
        }

        let reader = CImageReader::open(&path).unwrap();
        assert!(reader.header.tensors.len() == 50);

        std::fs::remove_file(&path).ok();
    }
}
