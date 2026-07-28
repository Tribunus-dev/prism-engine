//! CImage build entry point — `ModelConfig`, `CompiledTensor`, and
//! `build_cimage` (the pure-algo seal-into-vec path that takes a list of
//! compiled tensors and produces a single `.cimage` byte vector).
//!
//! Authority: end-to-end `.cimage` byte vector builder. Pure data + pure
//! I/O. No engine-coupled dependencies.
//!
//! Engine-coupled variants (file writer, MLX capture, ANE-side `mlmodelc`
//! packaging) live in the engine's `legacy_compute_image_compile/`
//! directory.

use crate::compute_image_compile::cimage_format::{
    write_cimage_header_le, CimageHeader, SegmentEntry, SegmentKind, CIMAGE_HEADER_WIRE_SIZE,
    CIMAGE_PAGE_SIZE, CIMAGE_SEGMENT_CAPACITY, PRISM_MAGIC,
};
use crate::compute_image_compile::matrix_binding::{
    write_matrix_weight_binding_v1_le, MatrixWeightBindingV1, MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH,
};
use std::collections::BTreeMap;

/// Model hyper-parameters needed to build the cimage header.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    /// Number of decoder layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Hidden dimension.
    pub hidden_dim: u32,
    /// FFN intermediate dimension.
    pub intermediate_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Quantization schema (see [`crate::compute_image_compile::cimage_format`]).
    pub quantization_schema: u32,
    /// Number of layers in the MTP draft decoder (0 = no draft model).
    pub draft_num_layers: u32,
    /// Number of experts in MoE layers (0 = dense model).
    pub num_experts: u32,
    /// Number of shared experts (DeepSeek-style, unused = 0).
    pub num_shared_experts: u32,
    /// Top-K active experts per token.
    pub top_k: u32,
    /// Hidden dimension of the expert MLP intermediate.
    pub expert_intermediate_dim: u32,
}

/// Input: one compiled tensor ready for sealing into a cimage.
#[derive(Debug, Clone)]
pub struct CompiledTensor {
    /// The per-tensor format contract.
    pub binding: MatrixWeightBindingV1,
    /// Serialised codes payload (length == `binding.code_length`).
    pub codes: Vec<u8>,
    /// Serialised metadata payload (length == `binding.metadata_length`).
    pub metadata: Vec<u8>,
}

/// Build a sealed cimage binary from compiled tensors and a model config.
///
/// The output contains:
///   - `CimageHeader` at offset 0 with segment directory entries
///   - A `MatrixContract` segment (kind 41)
///   - Code payload segments per representation
///   - Metadata payload segments per representation
///
/// Returns the complete cimage as `Vec<u8>` on success, or a descriptive error.
pub fn build_cimage(
    tensors: Vec<CompiledTensor>,
    config: ModelConfig,
) -> Result<Vec<u8>, String> {
    // ── Compute MatrixContract segment ──────────────────────────────
    let mut contract_buf =
        Vec::with_capacity(4 + tensors.len() * MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH);
    contract_buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for tensor in &tensors {
        write_matrix_weight_binding_v1_le(&mut contract_buf, &tensor.binding)
            .map_err(|e| format!("write binding: {e}"))?;
    }
    let contract_len = contract_buf.len() as u64;

    // PAD to page boundary
    let pad_to_page = |n: u64| -> u64 {
        ((n + CIMAGE_PAGE_SIZE - 1) / CIMAGE_PAGE_SIZE) * CIMAGE_PAGE_SIZE
    };

    // ── Compute code and metadata payloads per tensor ───────────────
    // Segment kind mapping per representation (code, metadata).
    let segment_kinds = |rep: u8| -> (u32, u32) {
        match rep {
            0 => (
                SegmentKind::TernaryWeights as u32,
                SegmentKind::BlockScales as u32,
            ),
            1 => (
                SegmentKind::Nf4Tile640Weights as u32,
                SegmentKind::BlockBiases as u32,
            ),
            2 => (
                SegmentKind::Int8Tile640Weights as u32,
                SegmentKind::BlockBiases as u32,
            ),
            3 => (SegmentKind::RawF16Weights as u32, 0xFFFFFFFF),
            _ => (0xFFFFFFFF, 0xFFFFFFFF),
        }
    };

    // Collect unique segment kinds and their payloads.
    let mut code_segments: Vec<(u32, Vec<u8>)> = Vec::with_capacity(tensors.len());
    let mut meta_segments: Vec<(u32, Vec<u8>)> = Vec::with_capacity(tensors.len());

    for tensor in &tensors {
        let (code_kind, meta_kind) = segment_kinds(tensor.binding.representation);
        if code_kind != 0xFFFFFFFF {
            code_segments.push((code_kind, tensor.codes.clone()));
        }
        if meta_kind != 0xFFFFFFFF && !tensor.metadata.is_empty() {
            meta_segments.push((meta_kind, tensor.metadata.clone()));
        }
    }

    // Concatenate payloads by segment kind (BTreeMap gives deterministic order).
    fn concat_by_kind(segments: &[(u32, Vec<u8>)]) -> Vec<(u32, Vec<u8>)> {
        let mut map: BTreeMap<u32, Vec<u8>> = BTreeMap::new();
        for (kind, data) in segments {
            map.entry(*kind).or_default().extend_from_slice(data);
        }
        map.into_iter().collect()
    }

    let code_blobs = concat_by_kind(&code_segments);
    let meta_blobs = concat_by_kind(&meta_segments);

    // ── Compute segment offsets ─────────────────────────────────────
    let header_size = CIMAGE_HEADER_WIRE_SIZE as u64;
    let header_page = pad_to_page(header_size);

    // Contract segment starts right after header page
    let contract_offset = header_page;
    let contract_page_end = pad_to_page(contract_offset + contract_len);

    // Code segments start after contract
    let mut code_offset = contract_page_end;
    let mut code_sizes: Vec<(u32, u64, u64)> = Vec::with_capacity(code_blobs.len());
    for (kind, blob) in &code_blobs {
        let len = blob.len() as u64;
        code_sizes.push((*kind, code_offset, len));
        code_offset = pad_to_page(code_offset + len);
    }

    // Metadata segments start after codes
    let mut meta_offset = code_offset.max(contract_page_end + 1); // ensure past contract
    let mut meta_sizes: Vec<(u32, u64, u64)> = Vec::with_capacity(meta_blobs.len());
    for (kind, blob) in &meta_blobs {
        let len = blob.len() as u64;
        meta_sizes.push((*kind, meta_offset, len));
        meta_offset = pad_to_page(meta_offset + len);
    }

    let total_size = meta_offset;

    // ── Build segment directory ─────────────────────────────────────
    let mut segment_count: u32 = 1; // always have MatrixContract
    let mut segments = [SegmentEntry {
        kind: 0,
        offset: 0,
        length: 0,
    }; CIMAGE_SEGMENT_CAPACITY];

    segments[0] = SegmentEntry {
        kind: SegmentKind::MatrixContract as u32,
        offset: contract_offset,
        length: contract_len,
    };

    let mut seg_index = 1;
    for (kind, offset, len) in &code_sizes {
        if *len > 0 && seg_index < CIMAGE_SEGMENT_CAPACITY {
            segments[seg_index] = SegmentEntry {
                kind: *kind,
                offset: *offset,
                length: *len,
            };
            seg_index += 1;
            segment_count += 1;
        }
    }
    for (kind, offset, len) in &meta_sizes {
        if *len > 0 && seg_index < CIMAGE_SEGMENT_CAPACITY {
            segments[seg_index] = SegmentEntry {
                kind: *kind,
                offset: *offset,
                length: *len,
            };
            seg_index += 1;
            segment_count += 1;
        }
    }

    // ── Build header ────────────────────────────────────────────────
    let header = CimageHeader {
        magic: PRISM_MAGIC,
        version: 1,
        segment_count,
        payload_hash: [0u8; 32],
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        hidden_dim: config.hidden_dim,
        intermediate_dim: config.intermediate_dim,
        vocab_size: config.vocab_size,
        quantization_schema: config.quantization_schema,
        draft_num_layers: config.draft_num_layers,
        segments,
        _pad: [0u8; 8],
    };
    // ── Write the cimage to a Vec<u8> buffer ────────────────────────
    let mut out = vec![0u8; total_size as usize];

    // Write header
    let mut cursor = std::io::Cursor::new(&mut out);
    use std::io::{Seek, Write};
    cursor
        .seek(std::io::SeekFrom::Start(0))
        .map_err(|e| format!("seek header: {e}"))?;
    write_cimage_header_le(&mut cursor, &header).map_err(|e| format!("write header: {e}"))?;

    // Write MatrixContract segment
    cursor
        .seek(std::io::SeekFrom::Start(contract_offset))
        .map_err(|e| format!("seek contract: {e}"))?;
    cursor
        .write_all(&contract_buf)
        .map_err(|e| format!("write contract: {e}"))?;

    // Write code segments
    for (kind, blob) in &code_blobs {
        let offset = code_sizes
            .iter()
            .find(|(k, _, _)| *k == *kind)
            .map(|(_, o, _)| *o)
            .ok_or_else(|| format!("missing code segment offset for kind {kind}"))?;
        cursor
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| format!("seek code seg {kind}: {e}"))?;
        cursor
            .write_all(blob)
            .map_err(|e| format!("write code seg {kind}: {e}"))?;
    }

    // Write metadata segments
    for (kind, blob) in &meta_blobs {
        let offset = meta_sizes
            .iter()
            .find(|(k, _, _)| *k == *kind)
            .map(|(_, o, _)| *o)
            .ok_or_else(|| format!("missing meta segment offset for kind {kind}"))?;
        cursor
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| format!("seek meta seg {kind}: {e}"))?;
        cursor
            .write_all(blob)
            .map_err(|e| format!("write meta seg {kind}: {e}"))?;
    }

    Ok(out)
}
