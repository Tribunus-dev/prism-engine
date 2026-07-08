//! NF4 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, codebooks, affine modes, clipping policies, and
//! group optimizers, producing a `FamilyCandidate` per combination.

use serde_json::json;

use crate::nf4tile640::{pack_nf4_tile_with_group_size, unpack_nf4_weights_with_group_size, validate_tile_group_size};
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
        group_sizes: vec![32, 64, 128],
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
pub(crate) fn nf4_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
}

/// Estimate NF4 metadata bytes for a matrix.
/// Default estimate: group_size=128 → 5 groups × 8 bytes (scale+bias) = 40 bytes/tile.
fn nf4_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // 5 groups × 2 f32 values (scale+bias) × 4 bytes each
    (total_tiles as u64) * 5 * 8
}

/// Estimate NF4 metadata bytes with an explicit group_size.
pub fn nf4_metadata_bytes_with_group_size(in_features: usize, out_features: usize, group_size: usize) -> u64 {
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (groups_per_tile as u64) * 8  // scale + bias, both f32
}

// ── Tile-based matrix packer ─────────────────────────────────────────────

/// Pack a weight matrix using NF4 with per-tile `pack_nf4_tile_with_group_size`.
///
/// Tiles across the `out_features` axis: each row (in_features) of `out_features`
/// elements is split into ceil(out_features / 640) tiles. Non-multiple trailing
/// elements are zero-padded.
pub(crate) fn pack_nf4_matrix(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    validate_tile_group_size(group_size)
        .expect("invalid NF4 tile group_size in pack_nf4_matrix");
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
        // Skip group sizes that don't divide 640 — they cannot tile.
        if TILE_ELEMENTS % group_size != 0 {
            continue;
        }
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

                        // TODO: codebook, affine_mode, clipping, optimizer are
                        // serialized into JSON but currently do NOT affect the packer.
                        // The packer always uses the default NF4 quantizer with
                        // group_size only. Wire these up once sweep supports per-family
                        // quantization variant dispatch.
                        let gs = group_size;
                        // Closure captures group_size by value; zero-overhead for Copy types.
                        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                            pack_nf4_matrix(w, r, c, gs)
                        });

                        let unpacker = Box::new(move |codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                            unpack_nf4_weights_with_group_size(codes, scales, biases, rows, cols, gs)
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
                            metadata_bytes_fn: Box::new(nf4_metadata_bytes),
                        });
                    }
                }
            }
        }
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nf4tile640::{
        pack_nf4_tile_with_group_size, unpack_nf4_weights_with_group_size,
        validate_tile_group_size, TILE_ELEMENTS,
    };
    use crate::quantization::contract::NF4_TILE640_CODE_BYTES;
    use crate::quantization::sweep::spec::Nf4SweepGrid;

    fn make_test_tile() -> [f32; 640] {
        // Deterministic linspace from -1 to 1
        let mut tile = [0.0f32; 640];
        for i in 0..640 {
            tile[i] = -1.0 + (i as f32) * 2.0 / 639.0;
        }
        tile
    }

    // ── 1. NF4 group-size pack/unpack produces correct shapes ───────────────

    #[test]
    fn nf4_pack_unpack_produces_correct_shapes() {
        let tile = make_test_tile();

        for &group_size in &[32, 64, 128] {
            let (codes, scales, biases) =
                pack_nf4_tile_with_group_size(&tile, group_size);
            let reconstructed = unpack_nf4_weights_with_group_size(
                &codes, &scales, &biases, 1, 640, group_size,
            );
            assert_eq!(
                reconstructed.len(),
                tile.len(),
                "output length must match input for group_size={group_size}"
            );
            let all_zero = reconstructed.iter().all(|&x| x == 0.0);
            assert!(
                !all_zero,
                "reconstruction not all zeros for group_size={group_size}"
            );
        }

        let (_, scales_32, _) = pack_nf4_tile_with_group_size(&tile, 32);
        let (_, scales_128, _) = pack_nf4_tile_with_group_size(&tile, 128);
        assert!(
            scales_32.len() > scales_128.len(),
            "group_size=32 scales ({}) should exceed group_size=128 scales ({})",
            scales_32.len(),
            scales_128.len(),
        );
    }

    // ── 2. NF4 rejects invalid group sizes ──────────────────────────────────

    #[test]
    fn nf4_rejects_invalid_group_sizes() {
        assert!(validate_tile_group_size(256).is_err());
        assert!(validate_tile_group_size(0).is_err());
        assert!(validate_tile_group_size(7).is_err());

        assert!(validate_tile_group_size(32).is_ok());
        assert!(validate_tile_group_size(64).is_ok());
        assert!(validate_tile_group_size(128).is_ok());
    }

    #[test]
    #[should_panic(expected = "invalid NF4 tile group_size")]
    fn nf4_pack_matrix_panics_on_invalid_group_size() {
        let weights = vec![0.0f32; 640];
        pack_nf4_matrix(&weights, 1, 640, 256);
    }

    // ── 3. NF4 byte accounting ─────────────────────────────────────────────

    #[test]
    fn nf4_byte_accounting() {
        let rows: usize = 10;
        let cols: usize = 640;

        let meta_32 = nf4_metadata_bytes_with_group_size(rows, cols, 32);
        let meta_128 = nf4_metadata_bytes_with_group_size(rows, cols, 128);

        assert_eq!(
            meta_32,
            4 * meta_128,
            "metadata for group_size=32 should be 4× metadata for group_size=128"
        );

        // group_size=128: tiles × 5 groups × 2 f32 × 4 bytes
        let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
        let total_tiles = rows * tiles_per_row;
        let expected_meta_128 = (total_tiles as u64) * 5 * 2 * 4;
        assert_eq!(meta_128, expected_meta_128);
    }

    // ── 4. NF4 candidate generation skips invalid group sizes ───────────────

    #[test]
    fn nf4_candidates_skip_invalid_group_sizes() {
        use crate::quantization::sweep::spec::{
            AffineMode, ClippingPolicy, GroupOptimizer, Nf4CodebookId,
        };

        let grid = Nf4SweepGrid {
            codebooks: vec![Nf4CodebookId::PrismCurrent],
            group_sizes: vec![32, 128, 256],
            affine_modes: vec![AffineMode::ScaleOnly],
            clip_policies: vec![ClippingPolicy::None],
            optimizers: vec![GroupOptimizer::None],
        };
        let candidates = generate_nf4_candidates(&grid);
        assert_eq!(
            candidates.len(),
            2,
            "should produce exactly 2 candidates (32 and 128), got {}",
            candidates.len()
        );
    }

    // ── 5. NF4 code_bytes is always 320 per tile ───────────────────────────

    #[test]
    fn nf4_code_bytes_per_tile() {
        let in_features: usize = 3;
        let out_features: usize = 640;
        let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
        let total_tiles = in_features * tiles_per_row;

        let bytes = nf4_code_bytes(in_features, out_features);
        assert_eq!(
            bytes,
            (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
        );

        // non-multiple out_features
        let out_features: usize = 700;
        let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
        let total_tiles = in_features * tiles_per_row;
        let bytes = nf4_code_bytes(in_features, out_features);
        assert_eq!(
            bytes,
            (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
        );
    }
}
