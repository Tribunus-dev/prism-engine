//! Symmetric INT4 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, signed ranges, affine modes, clipping policies,
//! and scale policies, producing a `FamilyCandidate` per combination.


use serde_json::json;

use crate::nf4tile640::validate_tile_group_size;
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

/// SymInt4 metadata bytes with explicit group_size: 8 bytes (scale+bias) per group.
fn sym_int4_metadata_bytes_with_group_size(
    in_features: usize, out_features: usize, group_size: usize
) -> u64 {
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (groups_per_tile as u64) * 8
}

// ── Tile-based matrix packer ─────────────────────────────────────────────

/// Pack a single tile of 640 f32 values using symmetric INT4 with range awareness.
///
/// Same logic as `pack_symmetric_int4_tile` from nf4tile640, but with configurable
/// signed range for clamping and nibble mapping.
///
/// For Neg7ToPos7: clamp to -7..7, map to 0..14 (values >7 become 7 → nibble 14)
/// For Neg8ToPos7: clamp to -8..7, map to 0..15
fn pack_sym_int4_tile_with_range(
    values: &[f32; TILE_ELEMENTS],
    group_size: usize,
    range: SignedInt4Range,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;
    let packed_codes_len = num_groups * bytes_per_group;
    let mut packed_codes = vec![0u8; packed_codes_len];
    let mut scales = vec![0.0f32; num_groups];
    let mut biases = vec![0.0f32; num_groups];
    // max_code is always 7.0 — both ranges have max abs code value of 7
    // (Neg8ToPos7 has -8 but it's the same magnitude for scale denominator)
    let max_code = 7.0f32;

    let (clamp_min, clamp_max, offset) = match range {
        SignedInt4Range::Neg7ToPos7 => (-7.0f32, 7.0f32, 7i8),
        SignedInt4Range::Neg8ToPos7 => (-8.0f32, 7.0f32, 8i8),
    };

    for group in 0..num_groups {
        let base = group * group_size;
        let max_abs = values[base..base + group_size]
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let scale = if max_abs < 1e-30 { 1.0f32 } else { max_abs / max_code };
        scales[group] = scale;
        biases[group] = 0.0;

        for i in 0..(group_size / 2) {
            let bit_idx = group * bytes_per_group + i;
            let val0 = (values[base + 2 * i] / scale).round().clamp(clamp_min, clamp_max) as i8;
            let val1 = (values[base + 2 * i + 1] / scale).round().clamp(clamp_min, clamp_max) as i8;
            // Map signed value to unsigned 4-bit nibble
            let code0 = (val0 + offset) as u8;
            let code1 = (val1 + offset) as u8;
            packed_codes[bit_idx] = code0 | (code1 << 4);
        }
    }

    (packed_codes, scales, biases)
}

/// Pack a weight matrix using symmetric INT4 with per-tile packing and range awareness.
///
/// Tiles across the `out_features` axis; each tile is 640 elements.
/// Non-multiple trailing columns are zero-padded.
fn pack_sym_int4_matrix_with_range(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    signed_range: SignedInt4Range,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) {
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
                pack_sym_int4_tile_with_range(&tile_buf, group_size, signed_range);

            codes.extend(t_codes);
            scales.extend(t_scales);
            biases.extend(t_biases);
        }
    }

    (codes, scales, biases, Vec::<u8>::new())
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
        // Reject group sizes that don't divide 640.
        if validate_tile_group_size(group_size).is_err() {
            continue;
        }
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

                        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                            pack_sym_int4_matrix_with_range(w, r, c, gs, sr)
                        });

                        let sr_unpack = *signed_range;
                        let unpacker = Box::new(move |codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                            unpack_sym_int4(codes, scales, biases, rows, cols, gs, sr_unpack)
                        });

                        let meta_fn = move |r: usize, c: usize| sym_int4_metadata_bytes_with_group_size(r, c, gs);

                        candidates.push(FamilyCandidate {
                            label: format!(
                                "SymInt4_g{}_sr{:?}_aff{:?}_clip{:?}_sp{:?}",
                                group_size, signed_range, affine, clip, scale_policy,
                            ),
                            parameters: params,
                            packer,
                            unpacker,
                            code_bytes_fn: sym_int4_code_bytes,
                            metadata_bytes_fn: Box::new(meta_fn),
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
    use crate::nf4tile640::TILE_ELEMENTS;

    /// Verify that the two signed ranges produce different packed codes for values
    /// that cannot be represented in Neg7ToPos7 but can in Neg8ToPos7.
    ///
    /// A tile of all -8.0 values: Neg7ToPos7 clamps to -7 (code=0),
    /// Neg8ToPos7 allows -8 (code=1). Both use the same scale (=8/7≈1.143)
    /// so the packed bytes differ at every nibble position.
    #[test]
    fn sym_int4_signed_range_produces_different_codes() {
        let tile = [-8.0f32; TILE_ELEMENTS];
        let group_size = 128;

        let (neg7_codes, _, _) =
            pack_sym_int4_tile_with_range(&tile, group_size, SignedInt4Range::Neg7ToPos7);
        let (neg8_codes, _, _) =
            pack_sym_int4_tile_with_range(&tile, group_size, SignedInt4Range::Neg8ToPos7);

        assert_ne!(
            neg7_codes, neg8_codes,
            "Neg7ToPos7 and Neg8ToPos7 must produce different packed codes for tile of all -8.0"
        );
    }

    /// Round-trip pack-then-unpack for both signed ranges at group_size=128.
    ///
    /// * Unpacked length always equals input length (640).
    /// * For Neg7ToPos7, each reconstructed value is within 1.0 of the original,
    ///   confirming no clamping wraparound or off-by-one in the nibble mapping.
    #[test]
    fn sym_int4_pack_unpack_round_trip() {
        // Tile with values spanning roughly [-5, 5] — all representable in both ranges.
        let mut tile = [0.0f32; TILE_ELEMENTS];
        for (i, v) in tile.iter_mut().enumerate() {
            *v = ((i as f32) - 320.0) / 64.0;
        }

        let group_size = 128;

        for range in [SignedInt4Range::Neg7ToPos7, SignedInt4Range::Neg8ToPos7] {
            let (codes, scales, biases) =
                pack_sym_int4_tile_with_range(&tile, group_size, range);
            let unpacked = unpack_sym_int4(
                &codes, &scales, &biases,
                1, TILE_ELEMENTS, group_size, range,
            );

            assert_eq!(
                unpacked.len(),
                TILE_ELEMENTS,
                "Unpacked length must match TILE_ELEMENTS for {:?}",
                range
            );
        }

        // For Neg7ToPos7, verify reconstruction error is bounded.
        let (codes, scales, biases) =
            pack_sym_int4_tile_with_range(&tile, group_size, SignedInt4Range::Neg7ToPos7);
        let unpacked = unpack_sym_int4(
            &codes, &scales, &biases,
            1, TILE_ELEMENTS, group_size, SignedInt4Range::Neg7ToPos7,
        );

        for i in 0..TILE_ELEMENTS {
            let err = (unpacked[i] - tile[i]).abs();
            assert!(
                err < 1.0,
                "Reconstruction error at index {}: {} (expected < 1.0)",
                i, err
            );
        }
    }

    /// Verify that `sym_int4_metadata_bytes_with_group_size` uses the actual
    /// number of groups derived from `TILE_ELEMENTS / group_size`, not a fixed
    /// value.
    ///
    /// * group_size=64  → 10 groups/tile → 1 × 10 × 8 = 80 bytes
    /// * group_size=128 →  5 groups/tile → 2 ×  5 × 8 = 80 bytes  (2 tiles)
    /// * group_size=32  → 20 groups/tile → 3 × 20 × 8 = 480 bytes
    #[test]
    fn sym_int4_metadata_bytes_uses_group_size() {
        // 1 tile (in_features=1, out_features=640), group_size=64: 10 groups.
        assert_eq!(
            sym_int4_metadata_bytes_with_group_size(1, TILE_ELEMENTS, 64),
            80,
            "1 tile, group_size=64 => 10 groups => 80 bytes"
        );

        // 2 tiles (in_features=2, out_features=640): 1 tile/row, 2 total.
        assert_eq!(
            sym_int4_metadata_bytes_with_group_size(2, TILE_ELEMENTS, 128),
            80,
            "2 tiles, group_size=128 => 5 groups/tile => 80 bytes"
        );

        // 3 tiles (in_features=3, out_features=640), group_size=32: 20 groups/tile.
        assert_eq!(
            sym_int4_metadata_bytes_with_group_size(3, TILE_ELEMENTS, 32),
            480,
            "3 tiles, group_size=32 => 20 groups/tile => 480 bytes"
        );
    }
}
