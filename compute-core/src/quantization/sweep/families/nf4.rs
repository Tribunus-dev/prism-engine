//! NF4 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, codebooks, affine modes, clipping policies, and
//! group optimizers, producing a `FamilyCandidate` per combination.

use serde_json::json;
use rayon::prelude::*;

use crate::nf4tile640::{
    pack_nf4_tile_with_group_size, unpack_nf4_weights_with_group_size_and_codebook,
    validate_tile_group_size, nf4_codebook, nf4_quantize_with_codebook, TILE_ELEMENTS,
};
use crate::quantization::contract::NF4_TILE640_CODE_BYTES;
use crate::quantization::sweep::candidate::PackedTileLayout;
use crate::quantization::sweep::families::FamilyCandidate;
use crate::quantization::sweep::spec::{
    AffineMode, ClippingPolicy, GroupOptimizer, Nf4CodebookId, Nf4SweepGrid, ScalePolicy,
};

// ── Nf4Params ─────────────────────────────────────────────────────────────

/// Fully-resolved NF4 codec parameters. Every field must affect
/// pack, unpack, byte accounting, or validation.
#[derive(Debug, Clone)]
pub struct Nf4Params {
    pub group_size: usize,
    pub codebook: Nf4CodebookId,
    pub affine_mode: AffineMode,
    pub clip_policy: ClippingPolicy,
    pub scale_policy: ScalePolicy,
    pub optimizer: GroupOptimizer,
    pub packed_layout: PackedTileLayout,
}

// ── Clipping helper ──────────────────────────────────────────────────────

/// Apply clipping policy to a group of values.
/// Returns clipped values for quantization (originals are still used for validation).
fn apply_group_clipping(values: &[f32], policy: &ClippingPolicy) -> Vec<f32> {
    match policy {
        ClippingPolicy::None => values.to_vec(),
        ClippingPolicy::Percentile(pct) => {
            let mut abs_vals: Vec<f32> = values.iter().map(|v| v.abs()).collect();
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = if abs_vals.is_empty() {
                0.0
            } else {
                let idx = ((abs_vals.len() as f32) * pct / 100.0).ceil() as usize;
                abs_vals[idx.min(abs_vals.len() - 1)]
            };
            values.iter().map(|v| v.clamp(-threshold, threshold)).collect()
        }
        ClippingPolicy::StddevMultiple(mult) => {
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let stddev = variance.sqrt();
            let threshold = mult * stddev;
            values.iter().map(|v| v.clamp(-threshold, threshold)).collect()
        }
        ClippingPolicy::GridFractionOfMaxAbs(fractions) => {
            // Use the last fraction in the list
            let fraction = fractions.last().copied().unwrap_or(1.0);
            let max_abs = values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let threshold = max_abs * fraction;
            values.iter().map(|v| v.clamp(-threshold, threshold)).collect()
        }
    }
}

// ── Group packing ─────────────────────────────────────────────────────────

/// Result of packing a single NF4 quantization group.
pub struct Nf4GroupPack {
    pub codes: Vec<u8>, // unpacked code indices (0..15), one per element
    pub scale: f32,
    pub bias: Option<f32>, // None for ScaleOnly, Some for ScaleBias
    pub mse: f64,
}

/// Pack a single NF4 group with full parameter dispatch.
fn pack_nf4_group(source_group: &[f32], params: &Nf4Params) -> Nf4GroupPack {
    let codebook = nf4_codebook(params.codebook);
    let max_cb_abs = codebook.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

    // Apply clipping
    let fit_values = apply_group_clipping(source_group, &params.clip_policy);

    match (&params.affine_mode, &params.optimizer) {
        (AffineMode::ScaleOnly, GroupOptimizer::None) => {
            let max_abs = fit_values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let scale = if max_abs < 1e-30 {
                1.0
            } else {
                max_abs / max_cb_abs
            };
            let mut codes = Vec::with_capacity(source_group.len());
            let mut sq_err = 0.0f64;
            for &v in &fit_values {
                let norm = (v / scale).clamp(-1.0, 1.0);
                let idx = nf4_quantize_with_codebook(norm, codebook);
                let decoded = codebook[idx as usize] * scale;
                let err = v as f64 - decoded as f64;
                sq_err += err * err;
                codes.push(idx);
            }
            Nf4GroupPack {
                codes,
                scale,
                bias: None,
                mse: sq_err / source_group.len() as f64,
            }
        }
        (AffineMode::ScaleBias, GroupOptimizer::None) => {
            let max_abs = fit_values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let scale = if max_abs < 1e-30 {
                1.0
            } else {
                max_abs / max_cb_abs
            };
            let bias = 0.0f32;
            let mut codes = Vec::with_capacity(source_group.len());
            let mut sq_err = 0.0f64;
            for &v in &fit_values {
                let norm = ((v - bias) / scale).clamp(-1.0, 1.0);
                let idx = nf4_quantize_with_codebook(norm, codebook);
                let decoded = codebook[idx as usize] * scale + bias;
                let err = v as f64 - decoded as f64;
                sq_err += err * err;
                codes.push(idx);
            }
            Nf4GroupPack {
                codes,
                scale,
                bias: Some(bias),
                mse: sq_err / source_group.len() as f64,
            }
        }
        (AffineMode::ScaleBias, GroupOptimizer::AffineAlternating { max_iters }) => {
            // Initial max-abs fit
            let max_abs = fit_values
                .iter()
                .map(|v| v.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let mut scale = if max_abs < 1e-30 {
                1.0
            } else {
                max_abs / max_cb_abs
            };
            let mut bias = 0.0f32;
            let mut codes: Vec<u8> = fit_values
                .iter()
                .map(|&v| {
                    let norm = ((v - bias) / scale).clamp(-1.0, 1.0);
                    nf4_quantize_with_codebook(norm, codebook)
                })
                .collect();

            let mut best_mse = f64::MAX;
            let mut best_codes = codes.clone();
            let mut best_scale = scale;
            let mut best_bias = bias;

            for _iter in 0..*max_iters {
                // Step 1: Given codes, solve for optimal scale and bias via least squares
                let mut sum_xy = 0.0f64;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_xx = 0.0f64;
                let n = fit_values.len() as f64;

                for (i, &v) in fit_values.iter().enumerate() {
                    let x = codebook[codes[i] as usize] as f64;
                    let y = v as f64;
                    sum_xy += x * y;
                    sum_x += x;
                    sum_y += y;
                    sum_xx += x * x;
                }

                // scale = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
                let denom = n * sum_xx - sum_x * sum_x;
                if denom.abs() > 1e-30 {
                    scale = ((n * sum_xy - sum_x * sum_y) / denom) as f32;
                }
                // bias = (sum_y - scale*sum_x) / n
                bias = ((sum_y - scale as f64 * sum_x) / n) as f32;

                // Step 2: Given scale and bias, re-assign codes
                let mut sq_err = 0.0f64;
                for (i, &v) in fit_values.iter().enumerate() {
                    let norm = ((v - bias) / scale).clamp(-1.0, 1.0);
                    codes[i] = nf4_quantize_with_codebook(norm, codebook);
                    let decoded = codebook[codes[i] as usize] * scale + bias;
                    let err = v as f64 - decoded as f64;
                    sq_err += err * err;
                }
                let mse = sq_err / n;

                if mse < best_mse - 1e-12 {
                    best_mse = mse;
                    best_codes = codes.clone();
                    best_scale = scale;
                    best_bias = bias;
                } else {
                    break; // converged
                }
            }

            Nf4GroupPack {
                codes: best_codes,
                scale: best_scale,
                bias: Some(best_bias),
                mse: best_mse,
            }
        }
        (AffineMode::ScaleOnly, GroupOptimizer::AffineAlternating { .. }) => {
            // Rejected by candidate generation. This arm is unreachable.
            unreachable!("ScaleOnly + AffineAlternating is an invalid combination");
        }
        (_, GroupOptimizer::ActivationWeighted { .. }) => {
            // Without activation trace, this is unsupported. Candidate gen skips it.
            unreachable!("ActivationWeighted optimizer requires activation trace");
        }
    }
}

/// Pack a weight matrix using NF4 with full parameter set.
/// Returns (codes, scales, biases, extra_bytes).
pub fn pack_nf4_matrix_with_params(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    params: &Nf4Params,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) {
    validate_tile_group_size(params.group_size)
        .expect("invalid NF4 tile group_size");
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let groups_per_tile = TILE_ELEMENTS / params.group_size;
    let codes_per_tile = TILE_ELEMENTS / 2;
    let total_tiles = in_features * tiles_per_row;

    let mut codes = Vec::with_capacity(total_tiles * codes_per_tile);
    let mut scales = Vec::with_capacity(total_tiles * groups_per_tile);
    let mut biases = Vec::with_capacity(total_tiles * groups_per_tile);

    // Parallelize over rows — each row's tiles are independent
    let row_results: Vec<(Vec<u8>, Vec<f32>, Vec<f32>)> = (0..in_features)
        .into_par_iter()
        .map(|row| {
            let row_base = row * out_features;
            let mut row_codes = Vec::with_capacity(tiles_per_row * codes_per_tile);
            let mut row_scales = Vec::with_capacity(tiles_per_row * groups_per_tile);
            let mut row_biases = Vec::new();
            for t in 0..tiles_per_row {
                let col_start = t * TILE_ELEMENTS;
                for g in 0..groups_per_tile {
                    let group_start = col_start + g * params.group_size;
                    let end = (row_base + group_start + params.group_size).min(row_base + out_features);
                    let group = &weights[row_base + group_start..end];
                    let mut buf = vec![0.0f32; params.group_size];
                    for (i, &v) in group.iter().enumerate() { buf[i] = v; }
                    let result = pack_nf4_group(&buf, params);
                    row_scales.push(result.scale);
                    if params.affine_mode == AffineMode::ScaleBias {
                        row_biases.push(result.bias.unwrap_or(0.0));
                    }
                    for pair in result.codes.chunks(2) {
                        let code0 = pair[0];
                        let code1 = *pair.get(1).unwrap_or(&0);
                        row_codes.push(code0 | (code1 << 4));
                    }
                }
            }
            (row_codes, row_scales, row_biases)
        }).collect();

    // Serial; concatenate row results into contiguous vecs
    for (row_codes, row_scales, row_biases) in &row_results {
        codes.extend(row_codes);
        scales.extend(row_scales);
        biases.extend(row_biases);
    }

    (codes, scales, biases, Vec::new())
}

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
#[allow(dead_code)]
fn nf4_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    // 5 groups × 2 f32 values (scale+bias) × 4 bytes each
    (total_tiles as u64) * 5 * 8
}

/// Estimate NF4 metadata bytes with an explicit group_size.
pub fn nf4_metadata_bytes_with_group_size(
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> u64 {
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * (groups_per_tile as u64) * 8 // scale + bias, both f32
}

/// Estimate NF4 metadata bytes accounting for affine mode (ScaleOnly = 4 bytes/group).
pub fn nf4_metadata_bytes_with_params(
    in_features: usize,
    out_features: usize,
    params: &Nf4Params,
) -> u64 {
    let groups_per_tile = TILE_ELEMENTS / params.group_size;
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    let bytes_per_group = match params.affine_mode {
        AffineMode::ScaleOnly => 4, // scale only
        AffineMode::ScaleBias => 8, // scale + bias
    };
    (total_tiles as u64) * (groups_per_tile as u64) * (bytes_per_group as u64)
}

// ── Tile-based matrix packer ─────────────────────────────────────────────

/// Pack a weight matrix using NF4 with per-tile `pack_nf4_tile_with_group_size`.
///
/// Tiles across the `out_features` axis: each row (in_features) of `out_features`
/// elements is split into ceil(out_features / 640) tiles. Non-multiple trailing
/// elements are zero-padded.
#[allow(dead_code)]
pub(crate) fn pack_nf4_matrix(
    weights: &[f32],
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) {
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

    (codes, scales, biases, Vec::<u8>::new())
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all NF4 family candidates from the sweep grid.
pub fn generate_nf4_candidates(grid: &Nf4SweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &gs in &grid.group_sizes {
        if validate_tile_group_size(gs).is_err() { continue; }
        for codebook in &grid.codebooks {
            for affine in &grid.affine_modes {
                for clip in &grid.clip_policies {
                    for opt in &grid.optimizers {
                        // Validate combinations
                        match (&affine, &opt) {
                            (AffineMode::ScaleOnly, GroupOptimizer::AffineAlternating { .. }) => {
                                continue;
                            }
                            _ => {}
                        }

                        let params = Nf4Params {
                            group_size: gs,
                            codebook: *codebook,
                            affine_mode: *affine,
                            clip_policy: clip.clone(),
                            scale_policy: ScalePolicy::MaxAbs,
                            optimizer: *opt,
                            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
                        };

                        let params_json = json!({
                            "family": "NF4",
                            "group_size": gs,
                            "codebook": format!("{:?}", codebook),
                            "affine_mode": format!("{:?}", affine),
                            "clip_policy": format!("{:?}", clip),
                            "optimizer": format!("{:?}", opt),
                            "scale_policy": "MaxAbs",
                        });

                        let pack_params = params.clone();
                        let cb_array = nf4_codebook(pack_params.codebook);
                        let packer = Box::new(move |w: &[f32], r: usize, c: usize| {
                            pack_nf4_matrix_with_params(w, r, c, &pack_params)
                        });
                        let unpacker = Box::new(move |codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                            unpack_nf4_weights_with_group_size_and_codebook(codes, scales, biases, rows, cols, gs, cb_array)
                        });

                        let meta_params = params.clone();
                        let metadata_fn = Box::new(move |r: usize, c: usize| {
                            nf4_metadata_bytes_with_params(r, c, &meta_params)
                        });

                        candidates.push(FamilyCandidate {
                            label: format!(
                                "NF4_g{}_cb{:?}_aff{:?}_clip{:?}_opt{:?}",
                                gs, codebook, affine, clip, opt,
                            ),
                            parameters: params_json,
                            packer,
                            unpacker,
                            code_bytes_fn: nf4_code_bytes,
                            metadata_bytes_fn: metadata_fn,
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
        unpack_nf4_weights_with_group_size_and_codebook,
        validate_tile_group_size, PRISM_NF4_CODEBOOK, TILE_ELEMENTS,
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

    fn make_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = Vec::with_capacity(rows * cols);
        for i in 0..rows * cols {
            w.push(((i as f32) / (rows * cols) as f32) * 2.0 - 1.0);
        }
        w
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

    #[test]
    fn nf4_byte_accounting_with_params() {
        let rows: usize = 10;
        let cols: usize = 640;

        let params_scale_only = Nf4Params {
            group_size: 128,
            codebook: Nf4CodebookId::PrismCurrent,
            affine_mode: AffineMode::ScaleOnly,
            clip_policy: ClippingPolicy::None,
            scale_policy: ScalePolicy::MaxAbs,
            optimizer: GroupOptimizer::None,
            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        };
        let params_scale_bias = Nf4Params {
            affine_mode: AffineMode::ScaleBias,
            ..params_scale_only.clone()
        };

        let meta_so = nf4_metadata_bytes_with_params(rows, cols, &params_scale_only);
        let meta_sb = nf4_metadata_bytes_with_params(rows, cols, &params_scale_bias);

        // ScaleOnly: 4 bytes/group, ScaleBias: 8 bytes/group
        assert_eq!(
            meta_sb,
            2 * meta_so,
            "ScaleBias metadata should be 2× ScaleOnly"
        );
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

    #[test]
    fn nf4_no_label_only_duplicate_codebooks() {
        // Different codebooks MUST produce different packed results
        // for the same input. If two codebook variants produce
        // identical payloads, the sweep has label-only duplicates.
        let weights = make_test_matrix(1, 640);
        let pairs = [
            (Nf4CodebookId::PrismCurrent, Nf4CodebookId::BitsAndBytesNf4),
            (Nf4CodebookId::PrismCurrent, Nf4CodebookId::SymmetricNormalFloat),
            (Nf4CodebookId::BitsAndBytesNf4, Nf4CodebookId::SymmetricNormalFloat),
        ];
        for &(cb_a, cb_b) in &pairs {
            let make_params = |cb| Nf4Params {
                group_size: 128, codebook: cb,
                affine_mode: AffineMode::ScaleOnly,
                clip_policy: ClippingPolicy::None,
                scale_policy: ScalePolicy::MaxAbs,
                optimizer: GroupOptimizer::None,
                packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
            };
            let (codes_a, _, _, _) = pack_nf4_matrix_with_params(&weights, 1, 640, &make_params(cb_a));
            let (codes_b, _, _, _) = pack_nf4_matrix_with_params(&weights, 1, 640, &make_params(cb_b));
            assert_ne!(codes_a, codes_b,
                "codebook variants {:?} and {:?} produce identical payloads — label-only duplicate",
                cb_a, cb_b);
        }
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

    // ── 6. NF4 group clipping ──────────────────────────────────────────────

    #[test]
    fn nf4_clip_none_preserves_values() {
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let clipped = apply_group_clipping(&values, &ClippingPolicy::None);
        assert_eq!(clipped, values);
    }

    #[test]
    fn nf4_clip_percentile_clamps_extremes() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32) / 100.0 * 10.0 - 5.0).collect();
        let clipped = apply_group_clipping(&values, &ClippingPolicy::Percentile(90.0));
        let max_clipped = clipped.iter().map(|v| v.abs()).max_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(max_clipped.is_some());
        assert!(
            max_clipped.unwrap() < 5.0 || (max_clipped.unwrap() - 4.55).abs() < 1.1,
            "percentile clipping should reduce max absolute value"
        );
    }

    #[test]
    fn nf4_clip_stddev_reduces_extremes() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32) / 10.0).collect();
        let clipped = apply_group_clipping(&values, &ClippingPolicy::StddevMultiple(1.0));
        // All values should be within 1 stddev of mean
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let stddev = variance.sqrt();
        for &v in &clipped {
            assert!(v.abs() <= mean.abs() + stddev + 1e-6);
        }
    }

    #[test]
    fn nf4_clip_grid_fraction() {
        let values = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let clipped = apply_group_clipping(
            &values,
            &ClippingPolicy::GridFractionOfMaxAbs(vec![0.5]),
        );
        // max_abs = 10, threshold = 5
        for &v in &clipped {
            assert!(v.abs() <= 5.0 + 1e-6, "value {} should be <= 5.0 after 0.5 fraction clip", v);
        }
    }

    // ── 7. NF4 pack_nf4_group produces valid output ────────────────────────

    #[test]
    fn nf4_pack_group_scale_only() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0 * 2.0 - 1.0).collect();
        let params = Nf4Params {
            group_size: 128,
            codebook: Nf4CodebookId::PrismCurrent,
            affine_mode: AffineMode::ScaleOnly,
            clip_policy: ClippingPolicy::None,
            scale_policy: ScalePolicy::MaxAbs,
            optimizer: GroupOptimizer::None,
            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        };
        let result = pack_nf4_group(&values, &params);
        assert_eq!(result.codes.len(), values.len());
        assert!(result.scale > 0.0);
        assert!(result.bias.is_none());
        assert!(result.mse >= 0.0);
        // Verify all codes are in range
        for &code in &result.codes {
            assert!(code < 16, "NF4 code must be 0..15, got {}", code);
        }
    }

    #[test]
    fn nf4_pack_group_scale_bias() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0 * 2.0 - 1.0).collect();
        let params = Nf4Params {
            group_size: 128,
            codebook: Nf4CodebookId::PrismCurrent,
            affine_mode: AffineMode::ScaleBias,
            clip_policy: ClippingPolicy::None,
            scale_policy: ScalePolicy::MaxAbs,
            optimizer: GroupOptimizer::None,
            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        };
        let result = pack_nf4_group(&values, &params);
        assert_eq!(result.codes.len(), values.len());
        assert!(result.scale > 0.0);
        assert!(result.bias.is_some());
        assert!(result.mse >= 0.0);
        for &code in &result.codes {
            assert!(code < 16, "NF4 code must be 0..15, got {}", code);
        }
    }

    // ── 8. NF4 pack_nf4_matrix_with_params round-trips ─────────────────────

    #[test]
    fn nf4_matrix_with_params_round_trip_scale_only() {
        let rows = 2;
        let cols = 640;
        let weights = make_test_matrix(rows, cols);

        let params = Nf4Params {
            group_size: 128,
            codebook: Nf4CodebookId::PrismCurrent,
            affine_mode: AffineMode::ScaleOnly,
            clip_policy: ClippingPolicy::None,
            scale_policy: ScalePolicy::MaxAbs,
            optimizer: GroupOptimizer::None,
            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        };

        let (codes, scales, biases, _extra) =
            pack_nf4_matrix_with_params(&weights, rows, cols, &params);

        // ScaleOnly should produce zero biases (existing format always stores bias)
        assert!(biases.is_empty(), "ScaleOnly should produce no biases");

        let reconstructed =
            unpack_nf4_weights_with_group_size_and_codebook(&codes, &scales, &biases, rows, cols, 128, &PRISM_NF4_CODEBOOK);

        assert_eq!(reconstructed.len(), weights.len());
        // Should not be all zeros
        let max_diff: f32 = reconstructed
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        assert!(
            max_diff < 2.0,
            "max diff {} should be reasonable for NF4",
            max_diff
        );
    }

    #[test]
    fn nf4_matrix_with_params_round_trip_scale_bias() {
        let rows = 2;
        let cols = 640;
        let weights = make_test_matrix(rows, cols);

        let params = Nf4Params {
            group_size: 128,
            codebook: Nf4CodebookId::PrismCurrent,
            affine_mode: AffineMode::ScaleBias,
            clip_policy: ClippingPolicy::None,
            scale_policy: ScalePolicy::MaxAbs,
            optimizer: GroupOptimizer::None,
            packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
        };

        let (codes, scales, biases, _extra) =
            pack_nf4_matrix_with_params(&weights, rows, cols, &params);

        assert!(!biases.is_empty(), "ScaleBias should produce biases");

        let reconstructed =
            unpack_nf4_weights_with_group_size(&codes, &scales, &biases, rows, cols, 128);

        assert_eq!(reconstructed.len(), weights.len());
        let max_diff: f32 = reconstructed
            .iter()
            .zip(weights.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        assert!(
            max_diff < 2.0,
            "max diff {} should be reasonable for NF4",
            max_diff
        );
    }

    // ── 9. NF4 candidate generation skips ScaleOnly+AffineAlternating ──────

    #[test]
    fn nf4_candidates_skip_invalid_combos() {
        use crate::quantization::sweep::spec::{
            AffineMode, ClippingPolicy, GroupOptimizer, Nf4CodebookId,
        };

        let grid = Nf4SweepGrid {
            codebooks: vec![Nf4CodebookId::PrismCurrent],
            group_sizes: vec![128],
            affine_modes: vec![AffineMode::ScaleOnly, AffineMode::ScaleBias],
            clip_policies: vec![ClippingPolicy::None],
            optimizers: vec![
                GroupOptimizer::None,
                GroupOptimizer::AffineAlternating { max_iters: 3 },
            ],
        };
        let candidates = generate_nf4_candidates(&grid);
        // 1 group_size × 2 codebooks → 1 (only PrismCurrent) × 2 affine modes × 1 clip × 2 optimizers = 4
        // but ScaleOnly+AffineAlternating is skipped → 3
        assert_eq!(
            candidates.len(),
            3,
            "should skip ScaleOnly+AffineAlternating, expected 3, got {}",
            candidates.len()
        );
    }
}
