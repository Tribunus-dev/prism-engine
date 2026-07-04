//! AOT layout calculator — determines all offsets before touching disk.
//!
//! `predict_tar_size` walks a .mlmodelc directory tree and computes the
//! exact uncompressed tar size.  `CImageLayoutPlan::calculate` uses that
//! along with known weight sizes to lay out all 7 segments at 16 KB
//! boundaries.

use crate::config::CompileQuantMode;
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
    pub main_scales: SegmentDescriptor,
    pub main_biases: SegmentDescriptor,
    pub mtp_graph: SegmentDescriptor,
    pub mtp_weights: SegmentDescriptor,
    pub mtp_scales: SegmentDescriptor,
    pub mtp_biases: SegmentDescriptor,
    pub topology_table: SegmentDescriptor,
    /// TernaryTile640-packed `embed_tokens.weight` (shared vocab for both models).
    pub vocabulary: SegmentDescriptor,
    /// Per-layer weight/scale offset table (array of LayerDirectoryEntry).
    /// Present when `num_layers > 0`; enables ANE/GPU interleaved scheduling.
    pub layer_directory: SegmentDescriptor,
    /// Execution graph descriptor for graph-driven runtime dispatch.
    pub execution_graph: SegmentDescriptor,
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
        main_weights_len: u64,
        main_scales_len: u64,
        main_biases_len: u64,
        mtp_graph_len: u64,
        mtp_weights_len: u64,
        mtp_scales_len: u64,
        mtp_biases_len: u64,
        vocab_len: u64,
        // Number of transformer layers (for LayerDirectory sizing).
        num_layers: u32,
        execution_graph_len: u64,
        multimodal_projection_weights_len: Option<u64>,
        multimodal_projection_scales_len: Option<u64>,
        multimodal_input_descriptor_len: Option<u64>,
        multimodal_position_embeddings_len: Option<u64>,
        multimodal_auxiliary_bytes: Option<u64>,
        _qmode: CompileQuantMode,
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

        let header = next(header_size);
        let metal_lib = next(metal_lib_len);
        let main_graph = next(main_graph_len);
        let main_weights = next(main_weights_len);
        let main_scales = next(main_scales_len);
        let main_biases = next(main_biases_len);
        let mtp_graph = next(mtp_graph_len);
        let mtp_weights = next(mtp_weights_len);
        let mtp_scales = next(mtp_scales_len);
        let mtp_biases = next(mtp_biases_len);
        let topology_table_size = std::mem::size_of::<CImageTopologyTable>() as u64;
        let topology_table = next(topology_table_size);
        let vocabulary = next(vocab_len);

        // LayerDirectory: num_layers × sizeof(LayerDirectoryEntry)
        // LayerDirectoryEntry is 6 × u64 = 48 bytes.
        let layer_dir_len = (num_layers as u64) * 48;
        let layer_directory = next(layer_dir_len);
        let execution_graph = next(execution_graph_len);

        let multimodal_input_descriptor =
            multimodal_input_descriptor_len.map(|len| next(len));

        let multimodal_projection_weights =
            multimodal_projection_weights_len.map(|len| next(len));

        let multimodal_projection_scales =
            multimodal_projection_scales_len.map(|len| next(len));

        let multimodal_position_embeddings =
            multimodal_position_embeddings_len.map(|len| next(len));

        let multimodal_auxiliary_weights = multimodal_auxiliary_bytes.map(|bytes| next(bytes));

        Self {
            total_file_size: cursor,
            header,
            metal_lib,
            main_graph,
            main_weights,
            main_scales,
            main_biases,
            mtp_graph,
            mtp_weights,
            mtp_scales,
            mtp_biases,
            topology_table,
            vocabulary,
            layer_directory,
            execution_graph,
            multimodal_projection_weights,
            multimodal_projection_scales,
            multimodal_input_descriptor,
            multimodal_position_embeddings,
            multimodal_auxiliary_weights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_tile640_layout_reserves_triplet_segments() {
        let plan = CImageLayoutPlan::calculate(
            256,
            1024,
            4096,
            3200,
            640,
            128,
            512,
            2048,
            1600,
            320,
            320,
            2,
            256,
            None,
            None,
            None,
            None,
            None,
            CompileQuantMode::Nf4Tile640 { group_size: 128 },
        );

        assert_eq!(plan.main_weights.length, 3200);
        assert_eq!(plan.main_scales.length, 640);
        assert_eq!(plan.main_biases.length, 128);
        assert_eq!(plan.mtp_weights.length, 2048);
        assert_eq!(plan.mtp_scales.length, 1600);
        assert_eq!(plan.mtp_biases.length, 320);
        assert_eq!(plan.vocabulary.length, 320);
        assert!(plan.main_scales.offset > plan.main_weights.offset);
        assert!(plan.main_biases.offset > plan.main_scales.offset);
        assert_eq!(plan.execution_graph.length, 256);
        assert!(plan.execution_graph.offset > plan.layer_directory.offset);
    }
}
