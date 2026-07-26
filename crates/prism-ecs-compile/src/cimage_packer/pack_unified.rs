//! Unified 5-segment packer (`pack_unified_cimage`).
//!
//! This module owns the canonical authority for the legacy unified
//! packer: a 5-segment `.cimage` consisting of the Metal library, the
//! main ternary weights, the MTP (multi-token prediction) ternary
//! weights, the main execution graph, and the MTP execution graph.
//! The packer is used by the speculative-compile path; the canonical
//! `pack_cimage_from_dir` is the general-purpose packer for every
//! other compile.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};

use super::segment_writer::align_to_page;
use super::{CimageHeader, CImagePackerResult, SegmentEntry, SegmentKind};

/// Pack five pre-built segment buffers into a V4 unified `.cimage`.
///
/// The function reserves `header_size` bytes for the header at the
/// start of the file, then writes each segment at the next
/// 16 KB-aligned offset. The MTP graph segment is optional: if
/// `mtp_graph_bytes` is empty, the segment table records three
/// segments and the rest of the file is reserved for future growth.
///
/// The packer is the *write* half of the format; the reader is in
/// `super::super::cimage::reader`. The packer does not own canonical
/// state — it is an effect that produces an immutable binary. The
/// runtime observes the binary through the canonical artifact store
/// after the pipeline's [`super::super::cimage_pipeline::publish`]
/// step.
pub fn pack_unified_cimage(
    output_path: &str,
    metal_lib_bytes: &[u8],
    main_graph_bytes: &[u8],
    main_weights_bytes: &[u8],
    mtp_graph_bytes: &[u8],
    mtp_weights_bytes: &[u8],
) -> CImagePackerResult<()> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let header_size = std::mem::size_of::<CimageHeaderSerialized>() as u64;

    // Reserve the header space.
    let zeros = vec![0u8; header_size as usize];
    writer.write_all(&zeros)?;

    let (metal_lib_offset, metal_lib_len) =
        write_segment(&mut writer, metal_lib_bytes)?;
    let (main_weights_offset, main_weights_len) =
        write_segment(&mut writer, main_weights_bytes)?;
    let (mtp_weights_offset, mtp_weights_len) =
        write_segment(&mut writer, mtp_weights_bytes)?;
    let (main_graph_offset, main_graph_len) =
        write_segment(&mut writer, main_graph_bytes)?;
    let (mtp_graph_offset, mtp_graph_len) =
        write_segment(&mut writer, mtp_graph_bytes)?;

    let segment_count = if mtp_graph_bytes.is_empty() { 3 } else { 5 };

    let header = CimageHeaderSerialized {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count,
        payload_hash: [0u8; 32],
        num_layers: 0,
        num_heads: 0,
        head_dim: 0,
        hidden_dim: 0,
        intermediate_dim: 0,
        vocab_size: 0,
        quantization_schema: 0,
        draft_num_layers: 0,
        metal_lib_offset,
        metal_lib_len,
        main_weights_offset,
        main_weights_len,
        mtp_weights_offset,
        mtp_weights_len,
        main_graph_offset,
        main_graph_len,
        mtp_graph_offset,
        mtp_graph_len,
    };

    writer.seek(SeekFrom::Start(0))?;
    let header_bytes = header.to_bytes();
    writer.write_all(&header_bytes)?;
    writer.flush()?;

    // Touch the unused SegmentEntry list to keep the public types
    // exported (the engine's `SegmentEntry` lives in this module's
    // public surface; the unified packer doesn't need it but downstream
    // consumers do).
    let _ = SegmentEntry::new(SegmentKind::MetalLib, metal_lib_offset, metal_lib_len);
    Ok(())
}

/// Write one segment at the next 16 KB-aligned offset and return the
/// (offset, length) recorded in the header.
fn write_segment(
    writer: &mut BufWriter<File>,
    data: &[u8],
) -> Result<(u64, u64), std::io::Error> {
    let offset = align_to_page(writer)?;
    writer.write_all(data)?;
    Ok((offset, data.len() as u64))
}

// ── On-disk header ──────────────────────────────────────────────────────

/// On-disk header for the unified 5-segment packer.
///
/// The 5-segment header is a fixed-size record with explicit
/// (offset, length) pairs for each segment, rather than the
/// flexible `SegmentEntry` array used by the from-dir packer. The
/// serialization is byte-stable: the order of fields is the on-disk
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct CimageHeaderSerialized {
    /// Magic identifier (8 bytes).
    pub magic: [u8; 8],
    /// Version field.
    pub version: u32,
    /// Number of segments in the segment table.
    pub segment_count: u32,
    /// 32-byte payload hash.
    pub payload_hash: [u8; 32],
    /// Number of layers.
    pub num_layers: u32,
    /// Number of heads.
    pub num_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// Intermediate dimension.
    pub intermediate_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Quantization schema.
    pub quantization_schema: u32,
    /// Number of draft layers.
    pub draft_num_layers: u32,
    /// Metal library offset.
    pub metal_lib_offset: u64,
    /// Metal library length.
    pub metal_lib_len: u64,
    /// Main weights offset.
    pub main_weights_offset: u64,
    /// Main weights length.
    pub main_weights_len: u64,
    /// MTP weights offset.
    pub mtp_weights_offset: u64,
    /// MTP weights length.
    pub mtp_weights_len: u64,
    /// Main graph offset.
    pub main_graph_offset: u64,
    /// Main graph length.
    pub main_graph_len: u64,
    /// MTP graph offset.
    pub mtp_graph_offset: u64,
    /// MTP graph length.
    pub mtp_graph_len: u64,
}

impl CimageHeaderSerialized {
    /// Serialize the header to its on-disk byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(std::mem::size_of::<Self>());
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.segment_count.to_le_bytes());
        out.extend_from_slice(&self.payload_hash);
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.num_heads.to_le_bytes());
        out.extend_from_slice(&self.head_dim.to_le_bytes());
        out.extend_from_slice(&self.hidden_dim.to_le_bytes());
        out.extend_from_slice(&self.intermediate_dim.to_le_bytes());
        out.extend_from_slice(&self.vocab_size.to_le_bytes());
        out.extend_from_slice(&self.quantization_schema.to_le_bytes());
        out.extend_from_slice(&self.draft_num_layers.to_le_bytes());
        out.extend_from_slice(&self.metal_lib_offset.to_le_bytes());
        out.extend_from_slice(&self.metal_lib_len.to_le_bytes());
        out.extend_from_slice(&self.main_weights_offset.to_le_bytes());
        out.extend_from_slice(&self.main_weights_len.to_le_bytes());
        out.extend_from_slice(&self.mtp_weights_offset.to_le_bytes());
        out.extend_from_slice(&self.mtp_weights_len.to_le_bytes());
        out.extend_from_slice(&self.main_graph_offset.to_le_bytes());
        out.extend_from_slice(&self.main_graph_len.to_le_bytes());
        out.extend_from_slice(&self.mtp_graph_offset.to_le_bytes());
        out.extend_from_slice(&self.mtp_graph_len.to_le_bytes());
        out
    }
}
