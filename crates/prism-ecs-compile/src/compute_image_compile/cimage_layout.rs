//! CImage layout metadata — `CimageLayoutMeta`, `TensorRecord`, and the
//! `verify_cimage` function that pairs a header with its layout segment.
//!
//! Authority: per-cimage tensor layout descriptors. This module owns the
//! wire format of the layout segment (kind 3) and the pure parsing
//! function that reads it back from a cimage byte slice.

use crate::compute_image_compile::cimage_format::{
    read_cimage_header_le, CimageHeader, PrismCimageHeader, SegmentKind, CIMAGE_HEADER_WIRE_SIZE,
};

/// Wire size of the canonical `CimageLayoutMeta`.
pub const CIMAGE_LAYOUT_META_WIRE_SIZE: usize = 6 * std::mem::size_of::<TensorRecord>() + 32;

/// One (offset, length) record describing where a logical tensor lives
/// inside a cimage segment.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct TensorRecord {
    /// Byte offset from start of the cimage.
    pub offset: u64,
    /// Byte length of the tensor payload.
    pub length: u64,
}

impl TensorRecord {
    /// Construct a new tensor record from offset and length.
    pub fn new(offset: u64, length: u64) -> Self {
        Self { offset, length }
    }
}

/// Canonical layout metadata — fixed slots for the well-known tensors
/// embedded in every cimage.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CimageLayoutMeta {
    /// Clustered embedding table (ternary + FP16 scales).
    pub embed_clustered: TensorRecord,
    /// Centroid table for the embedding.
    pub centroid_table: TensorRecord,
    /// Cluster assignment map (vocab_size entries).
    pub cluster_map: TensorRecord,
    /// Primary weight payload (ternary or NF4, depending on schema).
    pub ternary_weights: TensorRecord,
    /// FP16 block scales for the primary weight payload.
    pub block_scales: TensorRecord,
    /// Auxiliary tensors (norms, biases, RoPE).
    pub aux: TensorRecord,
    /// Reserved trailing padding.
    pub _pad: [u8; 32],
}

/// Backward-compatible type alias for the v1 layout metadata.
pub type PrismCimageLayoutMeta = CimageLayoutMeta;

/// Parse a `CimageHeader` and (optionally) the `LayoutMeta` segment from
/// a cimage byte slice.
///
/// Returns `(header, layout)`. If the layout segment is missing or
/// malformed, the default `CimageLayoutMeta` is returned.
pub fn verify_cimage(bytes: &[u8]) -> Result<(CimageHeader, CimageLayoutMeta), String> {
    if bytes.len() < CIMAGE_HEADER_WIRE_SIZE {
        return Err(format!(
            "cimage too small for header: {} < {}",
            bytes.len(),
            CIMAGE_HEADER_WIRE_SIZE
        ));
    }
    let header = read_cimage_header_le(bytes)?;
    let layout_meta_size = std::mem::size_of::<CimageLayoutMeta>();
    let layout = header
        .segment(SegmentKind::LayoutMeta)
        .and_then(|entry| {
            let end = (entry.offset as usize).checked_add(entry.length as usize)?;
            if end > bytes.len() || entry.length as usize != layout_meta_size {
                return None;
            }
            let bytes_at = &bytes[entry.offset as usize..end];
            // SAFETY: the slice is exactly `size_of::<CimageLayoutMeta>()` bytes
            // and we just bounds-checked it. The struct is `#[repr(C)]` with
            // POD fields, so the read is sound.
            let value = unsafe {
                std::ptr::read_unaligned(bytes_at.as_ptr() as *const CimageLayoutMeta)
            };
            Some(value)
        })
        .unwrap_or_default();
    Ok((header, layout))
}

/// Backward-compatible wrapper used by legacy callers.
pub fn verify_prism_cimage(
    bytes: &[u8],
) -> Result<(PrismCimageHeader, PrismCimageLayoutMeta), String> {
    verify_cimage(bytes)
}

/// Sanity check that the layout-metadata wire size matches the struct.
const _: () = {
    // 6 TensorRecords (16 bytes each) + 32 bytes padding = 128 bytes.
    assert!(CIMAGE_LAYOUT_META_WIRE_SIZE == 6 * 16 + 32);
};
