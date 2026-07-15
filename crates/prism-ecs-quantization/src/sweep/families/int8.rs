//! INT8 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, clipping policies, and scale policies.
//! INT8 uses direct byte-per-element quantization (640 bytes per tile)
//! with one f32 scale per group (groups_per_tile = 640/group_size).

use serde_json::json;

use crate::sweep::families::FamilyCandidate;
use crate::sweep::spec::Int8SweepGrid;
use crate::nf4tile640::{unpack_int8_weights_with_group_size, TILE_ELEMENTS};

// ── Byte-count estimators ────────────────────────────────────────────────

/// INT8 code bytes: each element is one byte, padded to tile boundary (640).
fn int8_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let tile_cols = tiles_per_row * TILE_ELEMENTS;
    (in_features * tile_cols) as u64
}

fn int8_metadata_bytes_with_group_size(
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> u64 {
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (groups_per_tile as u64) * 4 // one f32 per group
}

// ── Group-size-aware tile packer ─────────────────────────────────────────

/// Pack INT8 weights with configurable group size.
/// Each group of `group_size` elements gets its own f32 scale.
/// groups_per_tile = 640 / group_size.
pub(crate) fn pack_int8_matrix_with_group_size(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) {
    assert!(
        TILE_ELEMENTS % group_size == 0,
        "group_size must divide 640, got {}",
        group_size
    );
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let num_tiles = out_features.div_ceil(TILE_ELEMENTS);
    let tile_cols = num_tiles * TILE_ELEMENTS;

    let mut codes = vec![0u8; in_features * tile_cols];
    let total_groups = in_features * num_tiles * groups_per_tile;
    let mut scales = vec![0.0f32; total_groups];
    let mut biases = vec![0.0f32; total_groups];

    for i in 0..in_features {
        for t in 0..num_tiles {
            let col_start = t * TILE_ELEMENTS;
            let col_end = (col_start + TILE_ELEMENTS).min(out_features);

            for g in 0..groups_per_tile {
                let group_start = col_start + g * group_size;
                let group_end = (group_start + group_size).min(col_end);

                // Compute max_abs for this group
                let mut max_abs = 0.0f32;
                for j in group_start..group_end {
                    let v = weights[i * out_features + j].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                let scale = if max_abs > 1e-10 {
                    max_abs / 127.0
                } else {
                    1.0
                };
                let scale_idx = (i * num_tiles * groups_per_tile) + (t * groups_per_tile) + g;
                scales[scale_idx] = scale;
                biases[scale_idx] = 0.0;

                for j in group_start..group_end {
                    let code_idx = i * tile_cols + j;
                    let q = (weights[i * out_features + j] / scale)
                        .round()
                        .clamp(-127.0, 127.0) as i8;
                    codes[code_idx] = q as u8;
                }
            }
        }
    }
    (codes, scales, biases, Vec::<u8>::new())
}
// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all INT8 family candidates from the sweep grid.

/// Default INT8 sweep grid for initial exploration.
pub fn create_int8_grid() -> Int8SweepGrid {
    use crate::sweep::spec::{ClippingPolicy, ScalePolicy};
    Int8SweepGrid {
        group_sizes: vec![128, 640],
        clipping_policies: vec![ClippingPolicy::None],
        scale_policies: vec![ScalePolicy::MaxAbs],
        per_channel: false,
    }
}
pub fn generate_int8_candidates(grid: &Int8SweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &group_size in &grid.group_sizes {
        // Reject group sizes that don't divide 640.
        if TILE_ELEMENTS % group_size != 0 {
            continue;
        }
        for clip in &grid.clipping_policies {
            for scale_policy in &grid.scale_policies {
                let params = json!({
                    "family": "Int8",
                    "group_size": group_size,
                    "clipping_policy": format!("{:?}", clip),
                    "scale_policy": format!("{:?}", scale_policy),
                    "per_channel": grid.per_channel,
                });

                let gs = group_size;
                let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                    pack_int8_matrix_with_group_size(w, r, c, gs)
                });

                let unpacker = Box::new(
                    move |codes: &[u8],
                          scales: &[f32],
                          biases: &[f32],
                          _extra: &[u8],
                          rows: usize,
                          cols: usize| {
                        unpack_int8_weights_with_group_size(codes, scales, biases, rows, cols, gs)
                    },
                );

                let meta_fn =
                    move |r: usize, c: usize| int8_metadata_bytes_with_group_size(r, c, gs);

                candidates.push(FamilyCandidate {
                    label: format!("Int8_g{}_clip{:?}_sp{:?}", group_size, clip, scale_policy,),
                    parameters: params,
                    packer,
                    unpacker,
                    code_bytes_fn: int8_code_bytes,
                    metadata_bytes_fn: Box::new(meta_fn),
                });
            }
        }
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_group_size_determines_scale_count() {
        // Different group sizes produce proportionally different scale counts.
        // 640-wide tile: group_size=128 => 5 groups (5 scales), 640 => 1 group (1 scale).
        let in_features = 1;
        let out_features = 640;
        let n = in_features * out_features;
        let weights: Vec<f32> = (0..n)
            .map(|i| (i as f32 - (n as f32 / 2.0)) * 0.01)
            .collect();

        let (_, scales_128, _, _) =
            pack_int8_matrix_with_group_size(&weights, in_features, out_features, 128);
        let (_, scales_640, _, _) =
            pack_int8_matrix_with_group_size(&weights, in_features, out_features, 640);

        assert_eq!(
            scales_128.len(),
            5,
            "group_size=128: expected 5 scales (5 groups), got {}",
            scales_128.len()
        );
        assert_eq!(
            scales_640.len(),
            1,
            "group_size=640: expected 1 scale (1 group), got {}",
            scales_640.len()
        );
        // The counts MUST differ — different group sizes => different number of groups.
        assert_ne!(scales_128.len(), scales_640.len());
    }

    #[test]
    fn test_int8_pack_unpack_roundtrip() {
        // Round-trip pack then unpack: reconstruction length must match input,
        // and NRMSE must be within INT8 quantization error bounds.
        let in_features = 2;
        let out_features = 640;
        let n = in_features * out_features;
        let weights: Vec<f32> = (0..n)
            .map(|i| (i as f32 - (n as f32 / 2.0)) * 0.01)
            .collect();

        let group_size = 128;
        let (codes, scales, biases, _extra) =
            pack_int8_matrix_with_group_size(&weights, in_features, out_features, group_size);

        let reconstruction = unpack_int8_weights_with_group_size(
            &codes,
            &scales,
            &biases,
            in_features,
            out_features,
            group_size,
        );

        // Length must be preserved.
        assert_eq!(
            reconstruction.len(),
            weights.len(),
            "reconstruction length {} != input length {}",
            reconstruction.len(),
            weights.len()
        );

        // NRMSE: INT8 quantisation clips to [-127, 127] at 1/127 of max_abs per group;
        // worst-case error per element is ~scale/2, so NRMSE << 0.1 is guaranteed.
        let max_val = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let range = (max_val - min_val).max(1e-10);

        let mse: f64 = weights
            .iter()
            .zip(reconstruction.iter())
            .map(|(&w, &r)| {
                let d = w as f64 - r as f64;
                d * d
            })
            .sum::<f64>()
            / weights.len() as f64;

        let nrmse = (mse.sqrt() as f32) / range;
        assert!(nrmse < 0.1, "NRMSE {} >= 0.1 after INT8 round-trip", nrmse);
    }

    #[test]
    fn test_int8_metadata_bytes() {
        // Metadata = total_tiles * groups_per_tile * 4 (bytes per f32 scale).
        // 1 x 640 matrix => 1 tile.
        // group_size=640 => 1 group/tile => 1 * 1 * 4 = 4.
        assert_eq!(
            int8_metadata_bytes_with_group_size(1, 640, 640),
            4,
            "1 tile, 1 group/tile, 4 bytes/scale"
        );
        // group_size=128 => 5 groups/tile => 1 * 5 * 4 = 20.
        assert_eq!(
            int8_metadata_bytes_with_group_size(1, 640, 128),
            20,
            "1 tile, 5 groups/tile, 4 bytes/scale"
        );
    }
}
