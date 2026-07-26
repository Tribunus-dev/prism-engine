//! Standalone GGUF binary parser and safetensors converter.
//!
//! Parses GGUF (GGML Universal Format) files, extracts metadata and tensor
//! inventory, reads tensor data with dequantization to f32, and converts
//! to safetensors format for the cimage compilation pipeline.

use half::{bf16, f16};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
/// Writer for GGUF v3 format (test fixtures and synthetic weights).
#[doc(hidden)]
pub mod writer;
/// Manifest extraction — typed model architecture from GGUF metadata.
pub mod manifest;

/// Minimum GGUF version supported for import.
pub const MIN_GGUF_VERSION: u32 = 3;

// ── Error type ──────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF file: {0}")]
    InvalidFormat(String),

    #[error("Unsupported GGUF version: {0} (supported: 1–3)")]
    UnsupportedVersion(u32),

    #[error("Unsupported tensor dtype code {0}")]
    UnsupportedDtype(u32),

    #[error("Tensor data truncated: {0}")]
    TruncatedData(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("No such metadata key: {0}")]
    MissingMetadata(String),
}

// ── Public types ────────────────────────────────────────────────────────

/// Metadata for a single tensor in a GGUF file.
#[derive(Clone, Debug)]
pub struct GgufTensorMeta {
    pub name: String,
    /// Human-readable dtype name, e.g. "f32", "f16", "bf16", "q4_0", "q4_K_M", etc.
    pub dtype: String,
    pub shape: Vec<u32>,
    /// Byte offset of the tensor's raw data within the GGUF file.
    pub byte_offset: u64,
    /// Total byte size of the tensor's stored (possibly quantized) data.
    pub byte_size: u64,
}

/// Complete result of parsing a GGUF file's header.
pub struct GgufImportResult {
    pub metadata: Vec<(String, String)>,
    pub tensor_inventory: Vec<GgufTensorMeta>,
    pub source_path: PathBuf,
}

// ── GGUF binary value-type constants ────────────────────────────────────

const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;
const GGUF_TYPE_BF16: u32 = 13;

// ── GGML tensor dtype constants ─────────────────────────────────────────

/// GGML tensor type codes used in GGUF tensor info entries.
pub mod ggml_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    // 4, 5 = removed types
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q8_1: u32 = 9;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const Q8_K: u32 = 15;
    pub const IQ1_S: u32 = 16;
    pub const IQ1_M: u32 = 17;
    pub const IQ2_XXS: u32 = 18;
    pub const IQ2_XS: u32 = 19;
    pub const IQ2_S: u32 = 20;
    pub const IQ2_M: u32 = 21;
    pub const IQ3_XXS: u32 = 22;
    pub const IQ3_XS: u32 = 23;
    pub const IQ3_S: u32 = 24;
    pub const IQ4_NL: u32 = 26;
    pub const IQ4_XS: u32 = 27;
    pub const BF16: u32 = 28;
    pub const Q2_0: u32 = 29;
    /// Non-standard Q2_0 alias used by some GGUF exporters.
    pub const Q2_0_ALT: u32 = 42;
}

/// Return a human-readable dtype name for a GGML type code.
fn ggml_dtype_name(typ: u32) -> &'static str {
    match typ {
        ggml_type::Q2_0_ALT => "q2_0",
        ggml_type::Q2_0 => "q2_0",
        ggml_type::F32 => "f32",
        ggml_type::F16 => "f16",
        ggml_type::Q4_0 => "q4_0",
        ggml_type::Q4_1 => "q4_1",
        ggml_type::Q5_0 => "q5_0",
        ggml_type::Q5_1 => "q5_1",
        ggml_type::Q8_0 => "q8_0",
        ggml_type::Q8_1 => "q8_1",
        ggml_type::Q2_K => "q2_K",
        ggml_type::Q3_K => "q3_K",
        ggml_type::Q4_K => "q4_K",
        ggml_type::Q5_K => "q5_K",
        ggml_type::Q6_K => "q6_K",
        ggml_type::Q8_K => "q8_K",
        ggml_type::IQ1_S => "iq1_s",
        ggml_type::IQ1_M => "iq1_m",
        ggml_type::IQ2_XXS => "iq2_xxs",
        ggml_type::IQ2_XS => "iq2_xs",
        ggml_type::IQ2_S => "iq2_s",
        ggml_type::IQ2_M => "iq2_m",
        ggml_type::IQ3_XXS => "iq3_xxs",
        ggml_type::IQ3_XS => "iq3_xs",
        ggml_type::IQ3_S => "iq3_s",
        ggml_type::IQ4_NL => "iq4_nl",
        ggml_type::IQ4_XS => "iq4_xs",
        ggml_type::BF16 => "bf16",
        _ => "unknown",
    }
}

/// Return (block_size, type_size_in_bytes) for a GGML type.
fn ggml_block_info(typ: u32) -> (u64, u64) {
    match typ {
        ggml_type::Q2_0_ALT => (256, 66),
        ggml_type::Q2_0 => (256, 66),
        ggml_type::F32 => (1, 4),
        ggml_type::F16 | ggml_type::BF16 => (1, 2),
        ggml_type::Q4_0 => (32, 18),
        ggml_type::Q4_1 => (32, 20),
        ggml_type::Q5_0 => (32, 22),
        ggml_type::Q5_1 => (32, 24),
        ggml_type::Q8_0 => (32, 34),
        ggml_type::Q8_1 => (32, 40),
        ggml_type::Q2_K => (256, 72),
        ggml_type::Q3_K => (256, 104),
        ggml_type::Q4_K => (256, 144),
        ggml_type::Q5_K => (256, 176),
        ggml_type::Q6_K => (256, 208),
        ggml_type::Q8_K => (256, 272),
        // IQ types: approximate; consult ggml.h for exact layouts.
        ggml_type::IQ1_S => (256, 36),
        ggml_type::IQ1_M => (256, 52),
        ggml_type::IQ2_XXS => (256, 36),
        ggml_type::IQ2_XS => (256, 52),
        ggml_type::IQ2_S => (256, 70),
        ggml_type::IQ2_M => (256, 86),
        ggml_type::IQ3_XXS => (256, 78),
        ggml_type::IQ3_XS => (256, 94),
        ggml_type::IQ3_S => (256, 110),
        ggml_type::IQ4_NL => (32, 18),
        ggml_type::IQ4_XS => (256, 150),
        _ => (1, 1),
    }
}

/// Compute the byte size of a tensor given its dtype and shape.
fn ggml_tensor_byte_size(dtype: u32, shape: &[u32]) -> u64 {
    let total_elems: u64 = shape.iter().map(|&d| d as u64).product();
    if total_elems == 0 {
        return 0;
    }
    let (block_size, type_size) = ggml_block_info(dtype);
    let num_blocks = total_elems.div_ceil(block_size);
    num_blocks * type_size
}

// ── Binary reader helpers ───────────────────────────────────────────────

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, GgufError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, GgufError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read + Seek>(r: &mut R, version: u32, pos: u64) -> Result<String, GgufError> {
    if version >= 2 {
        let mut buf8 = [0u8; 8];
        r.seek(SeekFrom::Start(pos))?;
        r.read_exact(&mut buf8)?;
        let len64 = u64::from_le_bytes(buf8);
        if len64 > 0 && len64 < 500_000 {
            let mut name_buf = vec![0u8; len64 as usize];
            r.read_exact(&mut name_buf)?;
            return Ok(String::from_utf8_lossy(&name_buf).into_owned());
        }
        // Too large — try u32 (some GGUF v3 writers still use u32 lengths)
        r.seek(SeekFrom::Start(pos + 4))?;
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let len32 = u32::from_le_bytes(buf4) as u64;
        if len32 > 0 && len32 < 500_000 {
            let mut name_buf = vec![0u8; len32 as usize];
            r.read_exact(&mut name_buf)?;
            return Ok(String::from_utf8_lossy(&name_buf).into_owned());
        }
        return Err(GgufError::InvalidFormat(format!(
            "invalid string length at offset {}: u64={} u32={}",
            pos, len64, len32
        )));
    }
    // v1 — string length is always u32
    let len = read_u32_le(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Parse a single GGUF typed value and return its string representation.
fn read_typed_value<R: Read + Seek>(
    r: &mut R,
    typ: u32,
    version: u32,
    _pos: u64,
) -> Result<String, GgufError> {
    match typ {
        GGUF_TYPE_UINT8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", buf[0]))
        }
        GGUF_TYPE_INT8 => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", buf[0] as i8))
        }
        GGUF_TYPE_UINT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", u16::from_le_bytes(buf)))
        }
        GGUF_TYPE_INT16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", i16::from_le_bytes(buf)))
        }
        GGUF_TYPE_UINT32 => Ok(format!("{}", read_u32_le(r)?)),
        GGUF_TYPE_INT32 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", i32::from_le_bytes(buf)))
        }
        GGUF_TYPE_FLOAT32 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", f32::from_le_bytes(buf)))
        }
        GGUF_TYPE_BOOL => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
            Ok(if buf[0] == 0 {
                "false".into()
            } else {
                "true".into()
            })
        }
        GGUF_TYPE_STRING => {
            let s_pos = r.stream_position()?;
            read_string(r, version, s_pos)
        }
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            if count > 10000 {
                // Skip large arrays (e.g. tokenizer vocab) without materialising.
                for _ in 0..count {
                    let pos = r.stream_position()?;
                    read_typed_value(r, elem_type, version, pos)?;
                }
                return Ok(format!("[<{} elements>]", count));
            }
            let mut elems = Vec::with_capacity(count as usize);
            for _ in 0..count {
                let pos = r.stream_position()?;
                elems.push(read_typed_value(r, elem_type, version, pos)?);
            }
            Ok(format!("[{}]", elems.join(", ")))
        }
        GGUF_TYPE_UINT64 => Ok(format!("{}", read_u64_le(r)?)),
        GGUF_TYPE_INT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", i64::from_le_bytes(buf)))
        }
        GGUF_TYPE_FLOAT64 => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(format!("{}", f64::from_le_bytes(buf)))
        }
        GGUF_TYPE_BF16 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            let bits = u16::from_le_bytes(buf) as u32;
            let approx = f32::from_bits(bits << 16);
            Ok(format!("{}", approx))
        }
        _ => {
            // Unknown type — skip 4 bytes for recovery.
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            Ok(format!("<unknown_type_{}>", typ))
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Parse the GGUF header and return metadata KV pairs plus a tensor inventory.
///
/// This reads only the header section of the file (metadata + tensor
/// descriptions) without loading the weight data. The returned
/// `GgufTensorMeta` entries contain `byte_offset` and `byte_size` fields
/// for on-demand tensor data access.
/// Result of parsing a GGUF header: (metadata KV pairs, tensor inventory).
type GgufHeaderResult = Result<(Vec<(String, String)>, Vec<GgufTensorMeta>), GgufError>;

pub fn parse_gguf_header(path: &Path) -> GgufHeaderResult {
    let mut f = File::open(path)?;

    // ── Magic ───────────────────────────────────────────────────────────────
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufError::InvalidFormat(format!(
            "Not a GGUF file: magic={magic:?} expected b\"GGUF\""
        )));
    }

    // ── Version ─────────────────────────────────────────────────────────────
    let version = read_u32_le(&mut f)?;
    if !(1..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    // ── Tensor count ────────────────────────────────────────────────────────
    let tensor_count = if version >= 2 {
        read_u64_le(&mut f)?
    } else {
        read_u32_le(&mut f)? as u64
    };

    // ── Metadata KV count ───────────────────────────────────────────────────
    let metadata_kv_count = if version >= 2 {
        read_u64_le(&mut f)?
    } else {
        read_u32_le(&mut f)? as u64
    };

    // ── Metadata KV pairs ───────────────────────────────────────────────────
    let mut metadata: Vec<(String, String)> = Vec::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key_pos = f.stream_position()?;
        let key = read_string(&mut f, version, key_pos)?;
        let value_type = read_u32_le(&mut f)?;
        let pos = f.stream_position()?;
        let value = read_typed_value(&mut f, value_type, version, pos)?;
        metadata.push((key, value));
    }

    // ── Tensor infos ────────────────────────────────────────────────────────
    let mut tensors: Vec<GgufTensorMeta> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        // Some GGUF files insert stray bytes (alignment padding) between
        // tensor info entries. Scan byte-by-byte for a valid name_len.
        if !tensors.is_empty() {
            loop {
                match read_u64_le(&mut f) {
                    Ok(len) if len > 0 && len < 500 => {
                        let mut name_buf = vec![0u8; len as usize];
                        if let Ok(()) = f.read_exact(&mut name_buf) {
                            let name_str = String::from_utf8_lossy(&name_buf);
                            if name_str.chars().all(|c| {
                                c.is_ascii_graphic()
                                    || c.is_ascii_whitespace()
                                    || c == '.'
                                    || c == '_'
                                    || c == '/'
                            }) {
                                // Valid name — seek back to re-read properly.
                                f.seek(SeekFrom::Current(-(len as i64 + 8)))?;
                                break;
                            }
                        }
                    }
                    _ => {}
                }
                f.seek(SeekFrom::Current(-7))?;
            }
        }

        let name_pos = f.stream_position()?;
        let name = read_string(&mut f, version, name_pos)?;
        let n_dims = read_u32_le(&mut f)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = if version >= 3 {
                read_u64_le(&mut f)? as u32
            } else {
                read_u32_le(&mut f)?
            };
            dims.push(dim);
        }
        let dtype_code = read_u32_le(&mut f)?;
        let byte_offset = read_u64_le(&mut f)?;

        let byte_size = ggml_tensor_byte_size(dtype_code, &dims);
        let dtype_name = ggml_dtype_name(dtype_code).to_string();

        tensors.push(GgufTensorMeta {
            name,
            dtype: dtype_name,
            shape: dims,
            byte_offset,
            byte_size,
        });
    }

    Ok((metadata, tensors))
}

// ── Tensor data reading ─────────────────────────────────────────────────

/// Dequantize a GGML/GGUF tensor to f32 from a raw byte slice.
///
/// Supports: f32, f16, bf16, q8_0, q4_0, q2_0.
pub fn dequantize_ggml_tensor(
    data: &[u8],
    dtype: &str,
    num_elements: usize,
) -> Result<Vec<f32>, GgufError> {
    match dtype {
        "f32" => {
            if data.len() < num_elements * 4 {
                return Err(GgufError::TruncatedData(format!(
                    "f32: {} bytes for {} elements",
                    data.len(),
                    num_elements
                )));
            }
            Ok(data[..num_elements * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        "f16" => {
            if data.len() < num_elements * 2 {
                return Err(GgufError::TruncatedData(format!(
                    "f16: {} bytes for {} elements",
                    data.len(),
                    num_elements
                )));
            }
            Ok(data[..num_elements * 2]
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        "bf16" => {
            if data.len() < num_elements * 2 {
                return Err(GgufError::TruncatedData(format!(
                    "bf16: {} bytes for {} elements",
                    data.len(),
                    num_elements
                )));
            }
            Ok(data[..num_elements * 2]
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        "q8_0" => Ok(dequantize_ggml_q8_0(data, num_elements)),
        "q4_0" => Ok(dequantize_ggml_q4_0(data, num_elements)),
        "q2_0" => Ok(dequantize_ggml_q2_0(data, num_elements)),
        _ => Err(GgufError::UnsupportedDtype(0)), // dtype string, not code
    }
}

const Q8_0_BLOCK_SIZE: usize = 32;
const Q8_0_BYTES_PER_BLOCK: usize = 34;

/// Dequantize a GGML Q8_0 tensor to f32.
fn dequantize_ggml_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_count = num_elements.div_ceil(Q8_0_BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..block_count {
        let offset = b * Q8_0_BYTES_PER_BLOCK;
        if offset + Q8_0_BYTES_PER_BLOCK > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let vals_start = offset + 2;
        let remaining = num_elements
            .saturating_sub(b * Q8_0_BLOCK_SIZE)
            .min(Q8_0_BLOCK_SIZE);
        for i in 0..remaining {
            out.push((data[vals_start + i] as i8 as f32) * scale);
        }
    }
    out
}

const Q4_0_BLOCK_SIZE: usize = 32;
const Q4_0_BYTES_PER_BLOCK: usize = 18;

/// Dequantize a GGML Q4_0 tensor to f32.
fn dequantize_ggml_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_count = num_elements.div_ceil(Q4_0_BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..block_count {
        let offset = b * Q4_0_BYTES_PER_BLOCK;
        if offset + Q4_0_BYTES_PER_BLOCK > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let nibbles_start = offset + 2;
        let remaining = num_elements
            .saturating_sub(b * Q4_0_BLOCK_SIZE)
            .min(Q4_0_BLOCK_SIZE);
        for i in 0..remaining {
            let byte = data[nibbles_start + i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out.push(((nibble as i8 - 8) as f32) * scale);
        }
    }
    out
}

/// Read a single tensor from a GGUF file and dequantize to f32.
///
/// Seeks to `meta.byte_offset`, reads `meta.byte_size` raw bytes, then
const Q2_0_BLOCK_SIZE: usize = 256;
const Q2_0_BYTES_PER_BLOCK: usize = 66;

/// Dequantize a GGML Q2_0 tensor to f32.
///
/// Q2_0 block layout (66 bytes per 256 values):
/// - 2 bytes: fp16 scale
/// - 64 bytes: 256 × 2-bit values (4 values per byte)
/// Dequant: val = (packed_val - 1) * scale
fn dequantize_ggml_q2_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    let block_count = num_elements.div_ceil(Q2_0_BLOCK_SIZE);
    let mut out = Vec::with_capacity(num_elements);
    for b in 0..block_count {
        let offset = b * Q2_0_BYTES_PER_BLOCK;
        if offset + Q2_0_BYTES_PER_BLOCK > data.len() {
            break;
        }
        let scale_bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        let scale = f16::from_bits(scale_bits).to_f32();
        let bits_start = offset + 2;
        let remaining = num_elements
            .saturating_sub(b * Q2_0_BLOCK_SIZE)
            .min(Q2_0_BLOCK_SIZE);
        for i in 0..remaining {
            let byte = data[bits_start + i / 4];
            let bit_offset = (i % 4) * 2;
            let packed_val = (byte >> bit_offset) & 0x03;
            // Q2_0: val = (packed_val - 1) * scale  (packed_val: 2-bit unsigned 0-3)
            out.push(((packed_val as i32 - 1) as f32) * scale);
        }
    }
    out
}

/// Read a single tensor from a GGUF file and dequantize to f32.
///
/// Seeks to `meta.byte_offset`, reads `meta.byte_size` raw bytes, then
/// dequantizes according to `meta.dtype`.
pub fn read_tensor_f32(file: &mut File, meta: &GgufTensorMeta) -> Result<Vec<f32>, GgufError> {
    file.seek(SeekFrom::Start(meta.byte_offset))?;
    let mut raw = vec![0u8; meta.byte_size as usize];
    file.read_exact(&mut raw)?;
    let num_elements: usize = meta.shape.iter().map(|&d| d as usize).product();

    // Fast path for f32/f16/bf16 — use dequantize_ggml_tensor.
    dequantize_ggml_tensor(&raw, &meta.dtype, num_elements)
}

/// Map a GGUF dtype string to a safetensors dtype string.
fn safetensors_dtype(gguf_dtype: &str) -> &'static str {
    match gguf_dtype {
        "f32" => "F32",
        "f16" => "F16",
        "bf16" => "BF16",
        _ => "U8", // quantized → store raw block bytes as U8
    }
}

// ── gguf_to_safetensors_dir ─────────────────────────────────────────────

/// Convert every tensor in a GGUF file to its own safetensors file in
/// `output_dir`. Each output file is named `<tensor_name>.safetensors`.
///
/// For float dtypes (f32, f16, bf16) the data is written in its native
/// format. Quantized tensors are written as raw U8 bytes (the original
/// GGUF dtype is recorded in the safetensors metadata).
pub fn gguf_to_safetensors_dir(gguf_path: &Path, output_dir: &Path) -> Result<(), GgufError> {
    std::fs::create_dir_all(output_dir)?;
    let (_metadata, tensors) = parse_gguf_header(gguf_path)?;
    let mut file = File::open(gguf_path)?;

    for tensor in &tensors {
        let st_dtype = safetensors_dtype(&tensor.dtype);
        let num_elems: u64 = tensor.shape.iter().map(|&d| d as u64).product();

        // Read raw tensor data.
        file.seek(SeekFrom::Start(tensor.byte_offset))?;
        let mut raw = vec![0u8; tensor.byte_size as usize];
        file.read_exact(&mut raw)?;

        // Convert to payload (dequantize if needed).
        let payload: Vec<u8> =
            if st_dtype == "U8" || st_dtype == "F32" || st_dtype == "F16" || st_dtype == "BF16" {
                raw
            } else {
                let dequantized = dequantize_ggml_tensor(&raw, &tensor.dtype, num_elems as usize)?;
                let mut buf = Vec::with_capacity(dequantized.len() * 4);
                for v in &dequantized {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                buf
            };

        // For U8 (packed quantized), use packed shape to match safetensors validation.
        let (out_dtype, out_shape) = if st_dtype == "U8" {
            ("U8", serde_json::json!([payload.len()]))
        } else {
            (st_dtype, serde_json::json!(tensor.shape))
        };

        // Build full header with padding included as JSON (trailing spaces are valid JSON).
        let full_json = {
            let mut root = serde_json::Map::new();
            let mut meta = serde_json::Map::new();
            meta.insert(
                "gguf_dtype".to_string(),
                serde_json::Value::String(tensor.dtype.clone()),
            );
            root.insert("__metadata__".to_string(), serde_json::Value::Object(meta));
            root.insert(
                tensor.name.clone(),
                serde_json::json!({
                    "dtype": out_dtype,
                    "shape": out_shape,
                    "data_offsets": [0, payload.len()],
                }),
            );
            let hb = serde_json::to_vec(&serde_json::Value::Object(root))?;
            // Pad header to 8-byte boundary using spaces (valid JSON whitespace)
            let padding = (8 - ((hb.len()) % 8)) % 8;
            let mut padded = hb;
            for _ in 0..padding {
                padded.push(b' ');
            }
            padded
        };

        let header_len = full_json.len() as u64;

        // Write safetensors file.
        let mut out_path = output_dir.to_path_buf();
        let safe_name = tensor.name.replace("/", "--");
        out_path.push(format!("{}.safetensors", safe_name));

        let mut out = File::create(&out_path)?;
        out.write_all(&header_len.to_le_bytes())?;
        out.write_all(&full_json)?;
        out.write_all(&payload)?;
    }

    Ok(())
}

// ── extract_model_config ───────────────────────────────────────────────

/// Extract model configuration from GGUF metadata as a JSON object.
///
/// Looks for keys that start with `name` (a model architecture prefix such
/// as `"llama"`, `"gemma"`, etc.) and collects their values. The prefix is
/// stripped from the key in the returned JSON object. The tokenizer
/// vocabulary size and architecture name from `general.*` are also included.
///
/// Returns `Value::Null` if no matching keys are found.
pub fn extract_model_config(metadata: &[(String, String)], name: &str) -> Value {
    let prefix = format!("{}.", name);
    let mut config = serde_json::Map::new();

    // Collect architecture-prefixed keys (strip the prefix).
    for (key, val) in metadata {
        if let Some(rest) = key.strip_prefix(&prefix) {
            if let Ok(n) = val.parse::<i64>() {
                config.insert(rest.to_string(), json!(n));
            } else if let Ok(f) = val.parse::<f64>() {
                config.insert(rest.to_string(), json!(f));
            } else if val == "true" {
                config.insert(rest.to_string(), json!(true));
            } else if val == "false" {
                config.insert(rest.to_string(), json!(false));
            } else {
                config.insert(rest.to_string(), json!(val));
            }
        }
    }

    // Also include general.architecture to identify the arch.
    if let Some(arch) = metadata.iter().find(|(k, _)| k == "general.architecture") {
        config.insert("architecture".to_string(), json!(arch.1));
    }

    Value::Object(config)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspect_bonsai_gguf_tensors() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf");
        if !path.exists() {
            eprintln!("GGUF not found at {:?}", path);
            return;
        }
        eprintln!("=== GGUF TENSOR DUMP ===");
        let (metadata, tensors) = parse_gguf_header(&path).unwrap();
        for (k, v) in &metadata {
            if k.contains("arch")
                || k.contains("block")
                || k.contains("layer")
                || k.contains("head")
            {
                println!("meta: {k} = {v}");
            }
        }
        println!("\n=== blk.0 tensors ===");
        for t in &tensors {
            if t.name.starts_with("blk.0") {
                println!("tensor: {}  dtype={}  shape={:?}", t.name, t.dtype, t.shape);
            }
        }
        println!("\n=== Global tensors ===");
        for t in &tensors {
            if !t.name.starts_with("blk.") {
                println!("tensor: {}  dtype={}  shape={:?}", t.name, t.dtype, t.shape);
            }
        }
        println!("\nTotal: {}", tensors.len());
    }

    #[test]
    fn test_dtype_names() {
        assert_eq!(ggml_dtype_name(0), "f32");
        assert_eq!(ggml_dtype_name(1), "f16");
        assert_eq!(ggml_dtype_name(2), "q4_0");
        assert_eq!(ggml_dtype_name(8), "q8_0");
        assert_eq!(ggml_dtype_name(14), "q6_K");
        assert_eq!(ggml_dtype_name(28), "bf16");
        assert_eq!(ggml_dtype_name(29), "q2_0");
        assert_eq!(ggml_dtype_name(99), "unknown");
    }

    #[test]
    fn test_block_info() {
        assert_eq!(ggml_block_info(0), (1, 4));
        assert_eq!(ggml_block_info(1), (1, 2));
        assert_eq!(ggml_block_info(28), (1, 2));
        assert_eq!(ggml_block_info(2), (32, 18));
        assert_eq!(ggml_block_info(8), (32, 34));
        assert_eq!(ggml_block_info(10), (256, 72));
        assert_eq!(ggml_block_info(29), (256, 66));
    }

    #[test]
    fn test_tensor_byte_size() {
        // f32 [1, 64, 64, 3] = 1*64*64*3*4 = 49152
        assert_eq!(ggml_tensor_byte_size(0, &[1, 64, 64, 3]), 49152);
        // f16 [1, 64] = 1*64*2 = 128
        assert_eq!(ggml_tensor_byte_size(1, &[1, 64]), 128);
        // q8_0 [64] = 2 blocks * 34 = 68
        assert_eq!(ggml_tensor_byte_size(8, &[64]), 68);
        // empty shape
        assert_eq!(ggml_tensor_byte_size(0, &[0, 64]), 0);
    }

    #[test]
    fn test_extract_model_config_empty() {
        let meta = vec![("general.architecture".into(), "llama".into())];
        let cfg = extract_model_config(&meta, "nonexistent");
        // Should still include general.architecture
        assert!(cfg.is_object());
        assert_eq!(cfg["architecture"], "llama");
    }

    #[test]
    fn test_extract_model_config_prefixed() {
        let meta = vec![
            ("llama.block_count".into(), "32".into()),
            ("llama.embedding_length".into(), "4096".into()),
            ("llama.attention.head_count".into(), "32".into()),
            ("general.architecture".into(), "llama".into()),
        ];
        let cfg = extract_model_config(&meta, "llama");
        assert_eq!(cfg["block_count"], 32);
        assert_eq!(cfg["embedding_length"], 4096);
        assert_eq!(cfg["attention.head_count"], 32);
        assert_eq!(cfg["architecture"], "llama");
    }

    #[test]
    fn test_dequantize_f32() {
        // Two f32 values: 1.0, 2.0
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        let result = dequantize_ggml_tensor(&data, "f32", 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_f16() {
        let a = f16::from_f32(3.5);
        let b = f16::from_f32(-1.25);
        let mut data = Vec::new();
        data.extend_from_slice(&a.to_bits().to_le_bytes());
        data.extend_from_slice(&b.to_bits().to_le_bytes());
        let result = dequantize_ggml_tensor(&data, "f16", 2).unwrap();
        assert!((result[0] - 3.5).abs() < 0.01);
        assert!((result[1] - (-1.25)).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_bf16() {
        let a = bf16::from_f32(42.0);
        let b = bf16::from_f32(0.5);
        let mut data = Vec::new();
        data.extend_from_slice(&a.to_bits().to_le_bytes());
        data.extend_from_slice(&b.to_bits().to_le_bytes());
        let result = dequantize_ggml_tensor(&data, "bf16", 2).unwrap();
        assert!((result[0] - 42.0).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_safetensors_dtype_mapping() {
        assert_eq!(safetensors_dtype("f32"), "F32");
        assert_eq!(safetensors_dtype("f16"), "F16");
        assert_eq!(safetensors_dtype("bf16"), "BF16");
        assert_eq!(safetensors_dtype("q4_0"), "U8");
        assert_eq!(safetensors_dtype("q8_0"), "U8");
        assert_eq!(safetensors_dtype("q2_0"), "U8");
    }

    #[test]
    fn test_dequantize_q2_0() {
        // Construct 2 Q2_0 blocks (256 values each = 512 values)
        // Block 0: scale = fp16(1.0), data = all values = packed 2'b01 (value = 1)
        // val = (1 - 1) * 1.0 = 0.0 for all values
        // Block 1: scale = fp16(2.0), packed values = packed 2'b10 (value = 2)
        // val = (2 - 1) * 2.0 = 2.0 for all values
        let mut data = Vec::with_capacity(2 * 66);

        // Block 0: scale = 1.0, all zeros
        let scale0 = f16::from_f32(1.0);
        data.extend_from_slice(&scale0.to_bits().to_le_bytes());
        // 64 bytes of 2-bit values all set to 0 (packed_2bit = 0, val = -1)
        data.extend_from_slice(&[0u8; 64]);

        // Block 1: scale = 2.0, all twos (packed 2'b10)
        let scale1 = f16::from_f32(2.0);
        data.extend_from_slice(&scale1.to_bits().to_le_bytes());
        // Each byte: (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6) = 0xAA
        data.extend_from_slice(&[0xAAu8; 64]);

        let result = dequantize_ggml_q2_0(&data, 512);
        assert_eq!(result.len(), 512);

        // Check block 0: unpacked 0 -> val = (0 - 1) * 1.0 = -1.0
        for i in 0..256 {
            assert!(
                (result[i] - (-1.0)).abs() < 1e-6,
                "idx {i}: expected -1.0, got {}",
                result[i]
            );
        }

        // Check block 1: unpacked 2 -> val = (2 - 1) * 2.0 = 2.0
        for i in 256..512 {
            assert!(
                (result[i] - 2.0).abs() < 1e-6,
                "idx {i}: expected 2.0, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q2_0_via_tensor() {
        // Test dispatching through dequantize_ggml_tensor
        let mut data = Vec::with_capacity(66);
        let scale = f16::from_f32(3.0);
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        // Mixed 2-bit values: 0, 1, 2, 3 repeated
        // Each byte: 0xE4 = 0b11100100 = val[0]=0, val[1]=1, val[2]=2, val[3]=3
        data.extend_from_slice(&[0xE4u8; 64]);

        let result = dequantize_ggml_tensor(&data, "q2_0", 256).unwrap();
        assert_eq!(result.len(), 256);
        // val = (0 - 1) * 3.0 = -3.0, (1 - 1) * 3.0 = 0.0, (2 - 1) * 3.0 = 3.0, (3 - 1) * 3.0 = 6.0
        assert!((result[0] - (-3.0)).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 6.0).abs() < 1e-6);
        // Verify the pattern repeats every 4
        for i in 4..256 {
            assert!(
                (result[i] - result[i % 4]).abs() < 1e-6,
                "idx {i}: value mismatch"
            );
        }
    }
    #[test]
    fn test_error_display() {
        let err = GgufError::InvalidFormat("bad magic".into());
        assert!(format!("{err}").contains("bad magic"));

        let err = GgufError::UnsupportedVersion(5);
        assert!(format!("{err}").contains("5"));
    }
}
