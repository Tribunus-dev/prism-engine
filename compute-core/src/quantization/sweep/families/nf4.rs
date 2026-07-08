//! NF4 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, codebooks, affine modes, clipping policies, and
//! group optimizers, producing a `FamilyCandidate` per combination.

use serde_json::json;

use crate::nf4tile640::{pack_nf4_tile_with_group_size, unpack_nf4_weights};
use crate::quantization::contract::NF4_TILE640_CODE_BYTES;
use crate::quantization::sweep::spec::Nf4SweepGrid;
use crate::quantization::sweep::families::FamilyCandidate;
use crate::quantization::sweep::spec::{AffineMode, ClippingPolicy, GroupOptimizer, Nf4CodebookId};

const TILE_ELEMENTS: usize = 640;

/// Create the default NF4 sweep grid with standard parameter ranges.
pub fn create_nf4_grid() -> Nf4SweepGrid {
    Nf4SweepGrid {
        codebooks: vec![
            Nf4CodebookId::PrismCurrent,
            Nf4CodebookId::BitsAndBytesNf4,
            Nf4CodebookId::SymmetricNormalFloat,
        ],
        group_sizes: vec![32, 64, 128, 256],
        affine_modes: vec![AffineMode::ScaleOnly, AffineMode::ScaleBias],
        clip_policies: vec![
            ClippingPolicy::None,
            ClippingPolicy::Percentile(99.9),
            ClippingPolicy::StddevMultiple(2.5),
        ],
        optimizers: vec![
            GroupOptimizer::None,
            GroupOptimizer::AffineAlternating { max_iters: 3 },
        ],
    }
}

// ── Byte-count estimators (fn pointers, no capture) ──────────────────────

/// Estimate NF4 code bytes for a matrix of shape (in_features, out_features).
/// Each tile stores 320 bytes of 4-bit codes regardless of group_size.
fn nf4_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
}

/// Estimate NF4 metadata bytes for a matrix.
/// Default estimate: group_size=128 → 5 groups × 8 bytes (scale+bias) = 40 bytes/tile.
/// Exact metadata depends on group_size and is recorded in the packer output.
fn nf4_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // 5 groups × 2 f32 values (scale + bias) × 4 bytes = 40 bytes per tile
    (total_tiles as u64) * 40
}

// ── Tile-based matrix packer ─────────────────────────────────────────────

/// Pack a weight matrix using NF4 with per-tile `pack_nf4_tile_with_group_size`.
///
/// Tiles across the `out_features` axis: each row (in_features) of `out_features`
/// elements is split into ceil(out_features / 640) tiles. Non-multiple trailing
/// elements are zero-padded.
fn pack_nf4_matrix(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;

    let total_tiles = in_features * tiles_per_row;
    let mut codes = Vec::with_capacity(total_tiles * codes_per_tile);
    let mut scales = Vec::with_capacity(total_tiles * groups_per_tile);
    let mut biases = Vec::with_capacity(total_tiles * groups_per_tile);

    for row in 0..in_features {
        let row_base = row * out_features;
        for t in 0..tiles_per_row {
            let col_start = t * TILE_ELEMENTS;
            let mut tile_buf = [0.0f32; TILE_ELEMENTS];
            let remaining = out_features.saturating_sub(col_start);
            let copy_len = remaining.min(TILE_ELEMENTS);
            for i in 0..copy_len {
                tile_buf[i] = weights[row_base + col_start + i];
            }
            // Zero-padding for trailing elements is already in place (array init).

            let (t_codes, t_scales, t_biases) =
                pack_nf4_tile_with_group_size(&tile_buf, group_size);

            codes.extend(t_codes);
            scales.extend(t_scales);
            biases.extend(t_biases);
        }
    }

    (codes, scales, biases, Vec::new())
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all NF4 family candidates from the sweep grid.
pub fn generate_nf4_candidates(grid: &Nf4SweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &group_size in &grid.group_sizes {
        for codebook in &grid.codebooks {
            for affine in &grid.affine_modes {
                for clip in &grid.clip_policies {
                    for opt in &grid.optimizers {
                        let params = json!({
                            "family": "NF4",
                            "group_size": group_size,
                            "codebook": format!("{:?}", codebook),
                            "affine_mode": format!("{:?}", affine),
                            "clip_policy": format!("{:?}", clip),
                            "optimizer": format!("{:?}", opt),
                        });

                        let gs = group_size;
                        // Closure captures group_size by value; zero-overhead for Copy types.
                        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                            pack_nf4_matrix(w, r, c, gs)
                        });

                        let unpacker = Box::new(|codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                            unpack_nf4_weights(codes, scales, biases, rows, cols)
                        });

                        candidates.push(FamilyCandidate {
                            label: format!(
                                "NF4_g{}_cb{:?}_aff{:?}_clip{:?}_opt{:?}",
                                group_size, codebook, affine, clip, opt,
                            ),
                            parameters: params,
                            packer,
                            unpacker,
                            code_bytes_fn: nf4_code_bytes,
                            metadata_bytes_fn: nf4_metadata_bytes,
                        });
                    }
                }
            }
        }
    }

    candidates
}
