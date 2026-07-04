//! AOT layout calculator — determines all offsets before touching disk.
//!
//! `predict_tar_size` walks a .mlmodelc directory tree and computes the
//! exact uncompressed tar size.  `CImageLayoutPlan::calculate` uses that
//! along with known weight sizes to lay out all 7 segments at 16 KB
//! boundaries.

use std::path::Path;

const APPLE_PAGE_SIZE: u64 = super::APPLE_PAGE_SIZE as u64;

// ── Tar size predictor ──────────────────────────────────────────────

/// Walk a directory tree and compute the exact byte size an uncompressed
/// tar archive of it will occupy.  Tar is deterministic:
///   - 512 bytes per file/directory header
///   - File payloads padded to 512 bytes
///   - Two 512-byte zero-block EOF markers
pub fn predict_tar_size<P: AsRef<Path>>(path: P) -> std::io::Result<u64> {
    fn walk(dir: &Path, size: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            *size += 512; // tar header
            if meta.is_dir() {
                walk(&entry.path(), size)?;
            } else {
                let len = meta.len();
                *size += if len % 512 == 0 {
                    len
                } else {
                    len + (512 - len % 512)
                };
            }
        }
        Ok(())
    }
    let mut total = 512u64; // root directory header
    walk(path.as_ref(), &mut total)?;
    total += 1024; // two EOF blocks
    Ok(total)
}

// ── Segment descriptor ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct SegmentDescriptor {
    pub offset: u64,
    pub length: u64,
}

// ── Stride descriptor ───────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StrideDescriptor {
    pub chunk_size_bytes: u32,
    pub prefetch_stride_elements: u32,
    pub alignment_padding_bytes: u32,
    pub tensor_shape_quad: [u32; 4],
}

// ── Topology table ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CImageTopologyTable {
    pub slice_4: StrideDescriptor,
    pub slice_8: StrideDescriptor,
    pub slice_16: StrideDescriptor,
    pub slice_32: StrideDescriptor,
}

impl CImageTopologyTable {
    /// Precompute AOT stride/prefetch parameters for each slice width.
    /// Each slice processes `slice_count` FP16 elements per chunk.
    /// The prefetch stride advances along the intermediate (inner) dimension.
    pub fn compute(
        hidden_size: u32,
        intermediate_size: u32,
        n_layers: u32,
        n_heads: u32,
        head_dim: u32,
    ) -> Self {
        let bytes_per_element = 2u32; // FP16 weight storage
        let make_slice = |slice_count: u32| -> StrideDescriptor {
            let chunk_size = slice_count * bytes_per_element;
            let prefetch_stride = intermediate_size / slice_count;
            let align_pad = if chunk_size % 64 == 0 {
                0
            } else {
                64 - chunk_size % 64
            };
            StrideDescriptor {
                chunk_size_bytes: chunk_size,
                prefetch_stride_elements: prefetch_stride,
                alignment_padding_bytes: align_pad,
                tensor_shape_quad: [n_layers, n_heads, hidden_size, head_dim],
            }
        };
        Self {
            slice_4: make_slice(4),
            slice_8: make_slice(8),
            slice_16: make_slice(16),
            slice_32: make_slice(32),
        }
    }
}

// ── Layout plan ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CImageLayoutPlan {
    pub total_file_size: u64,
    pub header: SegmentDescriptor,
    pub metal_lib: SegmentDescriptor,
    pub main_graph: SegmentDescriptor,
    pub main_weights: SegmentDescriptor,
    pub mtp_graph: SegmentDescriptor,
    pub mtp_weights: SegmentDescriptor,
    pub topology_table: SegmentDescriptor,
    /// TernaryTile640-packed `embed_tokens.weight` (shared vocab for both models).
    pub vocabulary: SegmentDescriptor,
    /// Per-layer weight/scale offset table (array of LayerDirectoryEntry).
    /// Present when `num_layers > 0`; enables ANE/GPU interleaved scheduling.
    pub layer_directory: SegmentDescriptor,
    /// Multimodal projection weights segment. `None` for text-only cimages.
    pub multimodal_projection_weights: Option<SegmentDescriptor>,
    /// Multimodal projection scales segment. `None` for text-only cimages.
    pub multimodal_projection_scales: Option<SegmentDescriptor>,
    /// Multimodal input descriptor segment. `None` for text-only cimages.
    pub multimodal_input_descriptor: Option<SegmentDescriptor>,
    /// Multimodal position embeddings segment. `None` for text-only cimages.
    pub multimodal_position_embeddings: Option<SegmentDescriptor>,
    /// Multimodal auxiliary weights segment. `None` for text-only cimages.
    pub multimodal_auxiliary_weights: Option<SegmentDescriptor>,
}

impl CImageLayoutPlan {
    /// Compute the entire file layout given the known sizes of each
    /// segment.  Every segment starts on a 16 KB boundary.
    pub fn calculate(
        header_size: u64,
        metal_lib_len: u64,
        main_graph_len: u64,
        main_weights_total_elements: u64,
        mtp_graph_len: u64,
        mtp_weights_total_elements: u64,
        // Number of elements in the embed_tokens.weight tensor (vocab_size × hidden_dim).
        vocab_weight_total_elements: u64,
        // Number of transformer layers (for LayerDirectory sizing).
        num_layers: u32,
        multimodal_projection_weights_elements: Option<u64>,
        multimodal_position_embeddings_elements: Option<u64>,
        multimodal_auxiliary_bytes: Option<u64>,
    ) -> Self {
        let mut cursor = 0u64;
        let mut next = |size: u64| -> SegmentDescriptor {
            let desc = SegmentDescriptor {
                offset: cursor,
                length: size,
            };
            let raw_end = cursor + size;
            cursor = if raw_end % APPLE_PAGE_SIZE == 0 {
                raw_end
            } else {
                raw_end + (APPLE_PAGE_SIZE - raw_end % APPLE_PAGE_SIZE)
            };
            desc
        };

        // TernaryTile640: 640 weights → 32 u32 lanes × 4 bytes = 128 bytes per tile
        let main_weights_len = (main_weights_total_elements / 640) * 128;
        let mtp_weights_len = (mtp_weights_total_elements / 640) * 128;
        // Vocabulary segment: same TernaryTile640 packing for embed_tokens.weight.
        // Also append one F32 scale per tile (4 bytes × num_tiles, one per row-tile group).
        let vocab_num_tiles = (vocab_weight_total_elements + 639) / 640;
        // packed u32 payload: num_tiles tiles, each 32 u32s × 4 bytes
        let vocab_packed_len = vocab_num_tiles * 32 * 4;
        // scales payload: one f32 per (row, tile) pair already amortised into vocab_num_tiles
        let vocab_scales_len = vocab_num_tiles * 4;
        let vocab_len = vocab_packed_len + vocab_scales_len;

        let header = next(header_size);
        let metal_lib = next(metal_lib_len);
        let main_graph = next(main_graph_len);
        let main_weights = next(main_weights_len);
        let mtp_graph = next(mtp_graph_len);
        let mtp_weights = next(mtp_weights_len);
        let topology_table_size = std::mem::size_of::<CImageTopologyTable>() as u64;
        let topology_table = next(topology_table_size);
        let vocabulary = next(vocab_len);

        // LayerDirectory: num_layers × sizeof(LayerDirectoryEntry)
        // LayerDirectoryEntry is 6 × u64 = 48 bytes.
        let layer_dir_len = (num_layers as u64) * 48;
        let layer_directory = next(layer_dir_len);

        let multimodal_input_descriptor = {
            let size = std::mem::size_of::<
                crate::compute_image::multimodal::MultimodalInputDescriptorV1,
            >() as u64;
            Some(next(size))
        };

        let multimodal_projection_weights = multimodal_projection_weights_elements.map(|elems| {
            let len = (elems / 640) * 128; // TernaryTile640 packing
            next(len)
        });

        let multimodal_projection_scales = multimodal_projection_weights_elements.map(|elems| {
            let num_tiles = (elems + 639) / 640;
            let len = num_tiles * 4; // one F32 scale per tile
            next(len)
        });

        let multimodal_position_embeddings = multimodal_position_embeddings_elements.map(|elems| {
            let len = elems * 2; // FP16
            next(len)
        });

        let multimodal_auxiliary_weights = multimodal_auxiliary_bytes.map(|bytes| next(bytes));

        Self {
            total_file_size: cursor,
            header,
            metal_lib,
            main_graph,
            main_weights,
            mtp_graph,
            mtp_weights,
            topology_table,
            vocabulary,
            layer_directory,
            multimodal_projection_weights,
            multimodal_projection_scales,
            multimodal_input_descriptor,
            multimodal_position_embeddings,
            multimodal_auxiliary_weights,
        }
    }
}
