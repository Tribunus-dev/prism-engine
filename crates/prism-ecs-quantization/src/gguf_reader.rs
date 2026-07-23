//! Generic GGUF tensor reading utilities — format agnostic.
//!
//! Provides low-level tensor byte reading and metadata extraction from GGUF
//! files for compilation pipelines. Callers are responsible for any dtype-
//! specific dequantization they need.
//!
//! # Feature gate
//!
//! All functions are behind `#[cfg(feature = "gguf-compile")]` because they
//! depend on the optional `prism-gguf` crate.

#[cfg(feature = "gguf-compile")]
use std::collections::HashMap;
#[cfg(feature = "gguf-compile")]
use std::io::{Read, Seek, SeekFrom};
#[cfg(feature = "gguf-compile")]
use std::path::Path;

#[cfg(feature = "gguf-compile")]
use prism_gguf::GgufTensorMeta;

/// Read raw bytes of a tensor from a GGUF file given its metadata.
///
/// Reads `meta.byte_size` bytes starting at `meta.byte_offset` without any
/// conversion or dequantization.
#[cfg(feature = "gguf-compile")]
pub fn read_raw_gguf_tensor(gguf_path: &Path, meta: &GgufTensorMeta) -> Result<Vec<u8>, String> {
    let mut file =
        std::fs::File::open(gguf_path).map_err(|e| format!("open {}: {e}", gguf_path.display()))?;

    let offset = meta.byte_offset;
    let size = meta.byte_size as usize;

    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek to tensor: {e}"))?;

    let mut buf = vec![0u8; size];
    file.read_exact(&mut buf)
        .map_err(|e| format!("read tensor data: {e}"))?;
    Ok(buf)
}

/// Read a single tensor from a GGUF file and dequantize to f32.
///
/// This is a convenience wrapper around `prism_gguf::read_tensor_f32` that
/// opens the file internally, so callers don't need to manage the file handle.
#[cfg(feature = "gguf-compile")]
pub fn read_gguf_tensor_f32(gguf_path: &Path, meta: &GgufTensorMeta) -> Result<Vec<f32>, String> {
    let mut file =
        std::fs::File::open(gguf_path).map_err(|e| format!("open {}: {e}", gguf_path.display()))?;
    prism_gguf::read_tensor_f32(&mut file, meta)
        .map_err(|e| format!("read tensor {}: {e}", meta.name))
}

/// Parse a GGUF header and return the full tensor metadata list.
///
/// Each entry contains the name, dtype, shape, byte offset, and byte size.
/// Use this when you need to look up individual tensor metadata by name.
#[cfg(feature = "gguf-compile")]
pub fn parse_gguf_tensor_meta(gguf_path: &Path) -> Result<Vec<GgufTensorMeta>, String> {
    let (_metadata, tensors) =
        prism_gguf::parse_gguf_header(gguf_path).map_err(|e| format!("parse GGUF header: {e}"))?;
    Ok(tensors)
}

/// Return a map of tensor name → `(rows, cols)` shapes for every tensor in a
/// GGUF file.
///
/// For 1-D tensors (e.g. biases, RMS norms) the shape is `(size, 1)`. For
/// scalars it is `(1, 1)`.
#[cfg(feature = "gguf-compile")]
pub fn get_gguf_tensor_shapes(gguf_path: &Path) -> Result<HashMap<String, (u32, u32)>, String> {
    let tensors = parse_gguf_tensor_meta(gguf_path)?;
    let mut shapes = HashMap::with_capacity(tensors.len());
    for meta in &tensors {
        let (rows, cols) = match meta.shape.len() {
            0 => (1, 1),
            1 => (meta.shape[0], 1),
            _ => (meta.shape[0], meta.shape[1]),
        };
        shapes.insert(meta.name.clone(), (rows, cols));
    }
    Ok(shapes)
}

#[cfg(test)]
#[cfg(feature = "gguf-compile")]
mod tests {
    use super::*;

    #[test]
    fn test_shape_extraction_2d() {
        let meta = GgufTensorMeta {
            name: "w1".into(),
            dtype: "f16".into(),
            shape: vec![64, 128],
            byte_offset: 0,
            byte_size: 64 * 128 * 2,
        };
        let shapes = vec![meta];
        let mut map = HashMap::new();
        for m in shapes {
            let (r, c) = if m.shape.len() >= 2 {
                (m.shape[0], m.shape[1])
            } else if m.shape.len() == 1 {
                (m.shape[0], 1)
            } else {
                (1, 1)
            };
            map.insert(m.name.clone(), (r, c));
        }
        assert_eq!(map.get("w1"), Some(&(64, 128)));
    }

    #[test]
    fn test_shape_extraction_1d() {
        let meta = GgufTensorMeta {
            name: "bias".into(),
            dtype: "f32".into(),
            shape: vec![768],
            byte_offset: 0,
            byte_size: 768 * 4,
        };
        let (r, c) = if meta.shape.len() >= 2 {
            (meta.shape[0], meta.shape[1])
        } else if meta.shape.len() == 1 {
            (meta.shape[0], 1)
        } else {
            (1, 1)
        };
        assert_eq!(r, 768);
        assert_eq!(c, 1);
    }
}
