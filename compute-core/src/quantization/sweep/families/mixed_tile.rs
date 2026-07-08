//! Mixed tile rescue family candidate generation for QuantSweep.
//!
//! Packs weight tiles using the base 4-bit codec, measures per-tile
//! reconstruction error, and rescues the worst tiles by storing their
//! raw f32 values in the extra payload. The rescue fraction controls
//! the number of tiles replaced.

use serde_json::json;

use crate::nf4tile640::{
    pack_nf4_tile_with_group_size, TILE_ELEMENTS,
};
use crate::nf4tile640::NF4_CODEBOOK;
use crate::nf4tile640::nf4_dequantize;
use crate::quantization::contract::NF4_TILE640_CODE_BYTES;
use crate::quantization::sweep::spec::{MixedTileSweepGrid, RescueSchedule};
use crate::quantization::sweep::families::FamilyCandidate;

// ── Byte-count estimators ────────────────────────────────────────────────

/// Mixed tile code bytes: base 4-bit codes plus rescued f32 tiles.
/// The estimate assumes standard 4-bit codes for non-rescued tiles.
fn mixed_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // Upper bound: all tiles packed as NF4 (320 bytes each), worst-case
    // the extra rescue f32 data is separate.
    (total_tiles as u64) * (NF4_TILE640_CODE_BYTES as u64)
}

/// Mixed tile metadata bytes: base NF4 metadata plus rescue metadata.
fn mixed_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // Base estimate: 5 groups × 8 bytes = 40 bytes per tile
    (total_tiles as u64) * 40
}

// ── Tile error computation ───────────────────────────────────────────────

/// Compute the MSE for a single tile after NF4 quantization.
fn tile_mse(original: &[f32; TILE_ELEMENTS], group_size: usize) -> f32 {
    let groups = TILE_ELEMENTS / group_size;
    let mut total_sq_err = 0.0f32;

    for g in 0..groups {
        let base = g * group_size;
        // Compute scale (max-abs)
        let max_abs = original[base..base + group_size]
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let scale = if max_abs < 1e-30 { 1.0f32 } else { max_abs };

        for i in 0..group_size {
            let orig = original[base + i];
            // Quantize and dequantize inline
            let norm = (orig / scale).clamp(-1.0, 1.0);
            let code = nf4_quantize_closest(norm);
            let decoded = nf4_dequantize(code) * scale;
            let err = orig - decoded;
            total_sq_err += err * err;
        }
    }

    total_sq_err / (TILE_ELEMENTS as f32)
}

/// Find the closest NF4 codebook index for a value in [-1, 1].
fn nf4_quantize_closest(value: f32) -> u8 {
    let mut best_idx = 7u8; // default: 0.0
    let mut best_dist = f32::MAX;
    for (i, &cb_val) in NF4_CODEBOOK.iter().enumerate() {
        let d = (value - cb_val).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

// ── Tile rescuing logic ──────────────────────────────────────────────────

/// Pack a weight matrix with mixed tile rescue.
///
/// 1. Pack all tiles using NF4 (standard tile packer).
/// 2. Compute reconstruction error per tile.
/// 3. Identify the worst `rescue_count` tiles by highest MSE.
/// 4. Store their raw f32 values in the extra `Vec<f32>`.
/// 5. Mark rescued tiles with a sentinel: zero-code block + zero scale.
///
/// The runner can reconstruct rescued tiles from the extra payload
/// and regular NF4 tiles from the code/scale/bias payloads.
fn pack_mixed_tile_matrix(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    rescue_fraction: f32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;

    // Step 1: Pack all tiles normally.
    let mut all_tiles: Vec<[f32; TILE_ELEMENTS]> = Vec::with_capacity(total_tiles);
    let mut all_codes: Vec<Vec<u8>> = Vec::with_capacity(total_tiles);
    let mut all_scales: Vec<Vec<f32>> = Vec::with_capacity(total_tiles);
    let mut all_biases: Vec<Vec<f32>> = Vec::with_capacity(total_tiles);
    let mut tile_errors: Vec<(usize, f32)> = Vec::with_capacity(total_tiles);

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
                pack_nf4_tile_with_group_size(&tile_buf, group_size);

            let mse = tile_mse(&tile_buf, group_size);
            let tile_idx = all_tiles.len();

            all_tiles.push(tile_buf);
            all_codes.push(t_codes);
            all_scales.push(t_scales);
            all_biases.push(t_biases);
            tile_errors.push((tile_idx, mse));
        }
    }

    // Step 2: Select worst tiles for rescue.
    tile_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rescue_count = ((total_tiles as f32) * rescue_fraction).ceil() as usize;
    let rescue_count = rescue_count.min(total_tiles);

    let mut rescued_set = std::collections::HashSet::new();
    for i in 0..rescue_count {
        rescued_set.insert(tile_errors[i].0);
    }

    // Step 3: Build output buffers.
    let mut codes = Vec::with_capacity(total_tiles * codes_per_tile);
    let mut scales = Vec::with_capacity(total_tiles * groups_per_tile);
    let mut biases = Vec::with_capacity(total_tiles * groups_per_tile);
    let mut extra: Vec<f32> = Vec::new();

    for tile_idx in 0..total_tiles {
        if rescued_set.contains(&tile_idx) {
            // Store the original tile data as raw f32 in extra.
            extra.extend_from_slice(&all_tiles[tile_idx]);
            // Store sentinel codes/scales/biases: all zeros.
            codes.extend(std::iter::repeat(0u8).take(codes_per_tile));
            scales.extend(std::iter::repeat(0.0f32).take(groups_per_tile));
            biases.extend(std::iter::repeat(0.0f32).take(groups_per_tile));
        } else {
            codes.extend(&all_codes[tile_idx]);
            scales.extend(&all_scales[tile_idx]);
            biases.extend(&all_biases[tile_idx]);
        }
    }

    (codes, scales, biases, extra)
}

// ── Mixed tile unpacker ──────────────────────────────────────────────────

fn unpack_mixed_tile(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    extra: &[u8],
    in_features: usize,
    out_features: usize,
    group_size: usize,
    rescue_fraction: f32,
) -> Vec<f32> {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;

    let rescue_count_raw = ((total_tiles as f32) * rescue_fraction).ceil() as usize;
    let rescue_count = rescue_count_raw.min(total_tiles);

    // Identify rescued tiles by their all-zero code blocks.
    let mut rescued_set = std::collections::HashSet::new();
    let mut rescue_extra_offset = 0usize;

    for tile_idx in 0..total_tiles {
        let code_start = tile_idx * codes_per_tile;
        let code_slice = &codes[code_start..code_start + codes_per_tile];
        if code_slice.iter().all(|&b| b == 0) {
            // Check: only count as rescued if we still have extras for it.
            if rescue_extra_offset < rescue_count {
                rescued_set.insert(tile_idx);
                rescue_extra_offset += 1;
            }
        }
    }

    // Decompose into normal NF4 tiles and rescued tiles.
    let mut output = vec![0.0f32; in_features * out_features];
    let mut extra_idx = 0usize;

    for tile_idx in 0..total_tiles {
        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;

        if rescued_set.contains(&tile_idx) {
            // Reconstruct from extra f32 payload.
            for i in 0..TILE_ELEMENTS {
                let out_pos = row * out_features + col_base + i;
                if out_pos < in_features * out_features && col_base + i < out_features {
                    output[out_pos] = f32::from_le_bytes(extra[extra_idx..extra_idx + 4].try_into().unwrap());
                }
                extra_idx += 4;
            }
        } else {
            // Regular NF4 tile — reuse standard unpacker logic but only for one row's worth.
            // Build single-tile slices.
            let code_start = tile_idx * codes_per_tile;
            let scale_start = tile_idx * groups_per_tile;
            let mut tile_out = [0.0f32; TILE_ELEMENTS];

            for g in 0..groups_per_tile {
                let scale = scales[scale_start + g];
                let bias = biases[scale_start + g];
                let cb_base = code_start + g * (group_size / 2);
                let out_base = g * group_size;

                for i in 0..(group_size / 2) {
                    let packed = codes[cb_base + i];
                    let code0 = packed & 0x0F;
                    let code1 = (packed >> 4) & 0x0F;
                    tile_out[out_base + 2 * i] = nf4_dequantize(code0) * scale + bias;
                    tile_out[out_base + 2 * i + 1] = nf4_dequantize(code1) * scale + bias;
                }
            }

            // Copy tile output into result, respecting original boundaries.
            for i in 0..TILE_ELEMENTS {
                let out_pos = row * out_features + col_base + i;
                if out_pos < in_features * out_features && col_base + i < out_features {
                    output[out_pos] = tile_out[i];
                }
            }
        }
    }

    output
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all MixedTile family candidates from the sweep grid.
pub fn generate_mixed_tile_candidates(grid: &MixedTileSweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    // Default group_size = 128 for the base 4-bit codec.
    let default_group_size: usize = 128;

    for base_policy in &grid.base_policies {
    for schedule in &grid.schedules {
        // Extract a scalar rescue fraction from the schedule (total across all rounds).
        let rf_total: f32 = rescue_fraction_total(schedule);

        let params = json!({
            "family": "MixedTile",
            "base_policy": base_policy,
            "rescue_fraction": rf_total,
            "group_size": default_group_size,
            "rescue_schedule": schedule,
        });

        let gs = default_group_size;
        let rf = rf_total;

        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
            pack_mixed_tile_matrix(w, r, c, gs, rf)
        });

        let unpacker = Box::new(move |codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
            unpack_mixed_tile(codes, scales, biases, _extra, rows, cols, gs, rf)
        });

        candidates.push(FamilyCandidate {
            label: format!("MixedTile_{}_rescue{:.2}", base_policy.family, rf_total),
            parameters: params,
            packer,
            unpacker,
            code_bytes_fn: mixed_code_bytes,
            metadata_bytes_fn: mixed_metadata_bytes,
        });
    }
    }

    candidates
}

/// Total rescue fraction from a schedule (sum of all rounds).
fn rescue_fraction_total(schedule: &RescueSchedule) -> f32 {
    match schedule {
        RescueSchedule::OneShot { fraction } => *fraction as f32,
        RescueSchedule::FixedPerRound { fraction_per_round, rounds } => (*fraction_per_round * *rounds as f64) as f32,
        RescueSchedule::Geometric { fractions } => fractions.iter().sum::<f64>() as f32,
    }
}
