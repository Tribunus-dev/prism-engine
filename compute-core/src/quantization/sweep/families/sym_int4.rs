//! Symmetric INT4 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, signed ranges, affine modes, clipping policies,
//! and scale policies, producing a `FamilyCandidate` per combination.

use serde_json::json;

use crate::nf4tile640::pack_symmetric_int4_tile;
use crate::nf4tile640::TILE_ELEMENTS;
use crate::quantization::contract::NF4_TILE640_CODE_BYTES;
use crate::quantization::sweep::spec::{SignedInt4Range, SymInt4SweepGrid};
use crate::quantization::sweep::families::FamilyCandidate;

// ── Byte-count estimators ────────────────────────────────────────────────

/// SymInt4 uses the same 4-bit packing as NF4: 320 bytes of codes per tile.
fn sym_int4_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
}

/// Estimate SymInt4 metadata bytes.
/// Uses default group_size=128 → 5 groups × 8 bytes (scale+bias) = 40 bytes/tile.
fn sym_int4_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * 40
}

// ── Tile-based matrix packer ─────────────────────────────────────────────

/// Pack a weight matrix using symmetric INT4 with per-tile packing.
///
/// Tiles across the `out_features` axis; each tile is 640 elements.
/// Non-multiple trailing columns are zero-padded.
fn pack_sym_int4_matrix(
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

            let (t_codes, t_scales, t_biases) =
                pack_symmetric_int4_tile(&tile_buf, group_size);

            codes.extend(t_codes);
            scales.extend(t_scales);
            biases.extend(t_biases);
        }
    }

    (codes, scales, biases, Vec::new())
}

// ── Multi-tile unpacker for SymInt4 ──────────────────────────────────────

/// Unpack symmetric INT4 tiles back to f32.
///
/// Layout matches the output of `pack_symmetric_int4_tile`:
/// - Codes: 4-bit nibbles packed per u8, same layout as NF4.
/// - Scales: one f32 per group.
/// - Biases: one f32 per group (always 0.0 for symmetric mode).
///
/// Reconstruction: `output[i] = (signed_code as f32) * scale[group] + bias[group]`
/// where `signed_code` maps the 4-bit nibble (0..15) to signed value:
///   - Neg7ToPos7: 0..14 → -7..7 (15 is clamped to 7)
///   - Neg8ToPos7: 0..15 → -8..7
fn unpack_sym_int4(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    signed_range: SignedInt4Range,
) -> Vec<f32> {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;

    let mut output = vec![0.0f32; in_features * out_features];

    for tile_idx in 0..total_tiles {
        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;

        for g in 0..groups_per_tile {
            let scale = scales[tile_idx * groups_per_tile + g];
            let bias = biases[tile_idx * groups_per_tile + g];
            let codes_base = tile_idx * (groups_per_tile * bytes_per_group) + g * bytes_per_group;
            let out_base = row * out_features + col_base + g * group_size;

            for i in 0..bytes_per_group {
                let packed = codes[codes_base + i];
                let code0 = packed & 0x0F;
                let code1 = (packed >> 4) & 0x0F;

                let pos = out_base + 2 * i;
                if pos < row * out_features + out_features && pos + 1 <= in_features * out_features {
                    output[pos] = decode_sym_int4(code0, signed_range) * scale + bias;
                    if pos + 1 < in_features * out_features {
                        output[pos + 1] = decode_sym_int4(code1, signed_range) * scale + bias;
                    }
                }
            }
        }
    }

    output
}

/// Decode a 4-bit unsigned nibble to a signed int4 value.
#[inline(always)]
fn decode_sym_int4(nibble: u8, range: SignedInt4Range) -> f32 {
    match range {
        SignedInt4Range::Neg7ToPos7 => {
            if nibble > 14 {
                7.0 // clamp
            } else {
                (nibble as i8 - 7) as f32
            }
        }
        SignedInt4Range::Neg8ToPos7 => (nibble as i8 - 8) as f32,
    }
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all SymInt4 family candidates from the sweep grid.

/// Default SymInt4 sweep grid for initial exploration.
pub fn create_sym_int4_grid() -> SymInt4SweepGrid {
    use crate::quantization::sweep::spec::{
        AffineMode, ClippingPolicy, ScalePolicy, SignedInt4Range,
    };
    SymInt4SweepGrid {
        group_sizes: vec![16, 32, 64, 128],
        signed_ranges: vec![SignedInt4Range::Neg7ToPos7, SignedInt4Range::Neg8ToPos7],
        affine_modes: vec![AffineMode::ScaleOnly],
        clip_policies: vec![ClippingPolicy::None],
        scale_policies: vec![ScalePolicy::MaxAbs],
    }
}
pub fn generate_sym_int4_candidates(grid: &SymInt4SweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &group_size in &grid.group_sizes {
        for signed_range in &grid.signed_ranges {
            for affine in &grid.affine_modes {
                for clip in &grid.clip_policies {
                    for scale_policy in &grid.scale_policies {
                        let params = json!({
                            "family": "SymInt4",
                            "group_size": group_size,
                            "signed_range": format!("{:?}", signed_range),
                            "affine_mode": format!("{:?}", affine),
                            "clip_policy": format!("{:?}", clip),
                            "scale_policy": format!("{:?}", scale_policy),
                        });

                        let gs = group_size;
                        let sr = *signed_range;
                        let sr2 = sr;

                        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                            pack_sym_int4_matrix(w, r, c, gs)
                        });

                        let unpacker = Box::new(move |codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                            unpack_sym_int4(codes, scales, biases, rows, cols, gs, sr2)
                        });

                        candidates.push(FamilyCandidate {
                            label: format!(
                                "SymInt4_g{}_sr{:?}_aff{:?}_clip{:?}_sp{:?}",
                                group_size, signed_range, affine, clip, scale_policy,
                            ),
                            parameters: params,
                            packer,
                            unpacker,
                            code_bytes_fn: sym_int4_code_bytes,
                            metadata_bytes_fn: sym_int4_metadata_bytes,
                        });
                    }
                }
            }
        }
    }

    candidates
}
