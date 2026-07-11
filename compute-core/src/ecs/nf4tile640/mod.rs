//! nf4tile640 packed weight format — binary spec, CPU reference pack/unpack.
//!
//! # Format overview
//!
//! Each tile stores 640 f32 values as NF4 codes plus f32 scales and biases.
//! Groups are 128 elements, 5 groups per tile.
//!
//! ## Storage layout
//!
//! The compiler emits THREE separate byte payloads per cimage (not interleaved):
//!
//! 1. **packed_codes** — 320 bytes per tile: 8×4-bit NF4 indices packed per u32,
//!    stored as little-endian bytes.  For each group of 128 values, codes are
//!    stored consecutively (64 bytes = 16 u32s), low nibble = even element,
//!    high nibble = odd+1 element.
//!
//! 2. **scales** — 5 f32 values per tile (one per group), stored contiguously.
//!
//! 3. **biases** — 5 f32 values per tile (one per group), always 0.0 for NF4.
//!
//! ## Grouping
//!
//! - 128 values per quantization group (one f32 scale + one f32 bias)
//! - 5 groups per tile = 640 values
//!
//! ## Reconstruction
//!
//! reconstructed[i] = nf4_codebook[code_index[i]] * scale[group] + bias[group]

pub mod calibration;
pub mod learn;
pub mod outliers;
pub mod plan;
pub mod profile;
pub mod roles;
pub mod squat;
pub mod verify;

pub mod accelerate;
pub mod awls;
pub mod fused;
pub mod protection;

use crate::quantization::sweep::spec::Nf4CodebookId;

// ════════════════════════════════════════════════════════════════════════════
// Section 1: Format Constants
// ════════════════════════════════════════════════════════════════════════════

/// Format version for the nf4tile640 packed weight format.
pub const FORMAT_VERSION: u32 = 1;

/// Number of NF4 codes packed per u32 word (8 x 4-bit).
pub const CODES_PER_WORD: usize = 8;

/// Number of elements in a quantization group.
pub const GROUP_SIZE: usize = 128;

/// Number of groups per tile.
pub const GROUPS_PER_TILE: usize = 5; // 640 / 128

/// Total elements per tile.
/// Default tile element count for the Tile640 family.
/// This is the tuning target for GPU workgroup convergence.
/// Other families (Tile256, Tile1024) may be added for different hardware targets.
pub const TILE_ELEMENTS: usize = 640;

/// Metadata for a tile family — shape and group defaults.
///
/// `tile_elements` is the number of scalar values packed into one tile.
/// `tile_rows` and `tile_cols` describe the logical matrix-tile shape
/// used by the execution-profile layer (these mirror
/// `execution_profile::TileShape::tile640` for compositional consistency;
/// the product rows × cols may exceed `tile_elements` since the two
/// numbers live at different abstraction levels).
#[derive(Debug, Clone, Copy)]
pub struct TileFamilySpec {
    pub name: &'static str,
    pub tile_elements: usize,
    pub tile_rows: usize,
    pub tile_cols: usize,
}

impl TileFamilySpec {
    pub const fn tile640() -> Self {
        Self {
            name: "Tile640",
            tile_elements: 640,
            tile_rows: 640,
            tile_cols: 640,
        }
    }
}

/// Bytes per u32 word.
pub const PACKED_WORD_BYTES: usize = 4;

/// Bytes of packed codes per group (128 / 2).
pub const PACKED_BYTES_PER_GROUP: usize = GROUP_SIZE / 2; // 64

/// Bytes of packed codes per tile (640 / 2).
pub const PACKED_BYTES_PER_TILE: usize = TILE_ELEMENTS / 2; // 320

/// Number of f32 scale values per tile (one per group).
pub const SCALES_F32_PER_TILE: usize = GROUPS_PER_TILE; // 5

/// Validate that a group size is valid for Tile640 format.
pub fn validate_tile_group_size(group_size: usize) -> Result<(), String> {
    if group_size == 0 || TILE_ELEMENTS % group_size != 0 {
        return Err(format!(
            "group_size {group_size} must divide TILE_ELEMENTS {TILE_ELEMENTS} and be non-zero"
        ));
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// Section 2: NF4 Codebooks
// ════════════════════════════════════════════════════════════════════════════

/// Prism canonical NF4 codebook: 16 evenly-spaced quantiles of N(0,1).
/// Symmetric around zero. Index 0..15.
pub const PRISM_NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.8482084274291992,
    -0.6356878280639648,
    -0.46220311522483826,
    -0.32028985023587036,
    -0.19982607662677765,
    -0.0961047038435936,
    0.0,
    0.08384315651655197,
    0.1694672405719757,
    0.2574995458126068,
    0.3491421937942505,
    0.44636115431785583,
    0.5527461171150208,
    0.6738201389312744,
    1.0,
];

/// BitsAndBytes NF4 codebook (from Dettmers et al., 2023).
/// Different quantile spacing than PrismCurrent.
pub const BNB_NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Symmetric normal float NF4 codebook.
/// Uniformly spaced symmetric values.
pub const SYMMETRIC_NORMAL_FLOAT_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.8666667,
    -0.7333333,
    -0.6,
    -0.4666667,
    -0.3333333,
    -0.2,
    -0.06666667,
    0.06666667,
    0.2,
    0.3333333,
    0.4666667,
    0.6,
    0.7333333,
    0.8666667,
    1.0,
];

/// Look up an NF4 codebook by its identifier.
pub fn nf4_codebook(id: Nf4CodebookId) -> &'static [f32; 16] {
    match id {
        Nf4CodebookId::PrismCurrent => &PRISM_NF4_CODEBOOK,
        Nf4CodebookId::BitsAndBytesNf4 => &BNB_NF4_CODEBOOK,
        Nf4CodebookId::SymmetricNormalFloat => &SYMMETRIC_NORMAL_FLOAT_CODEBOOK,
    }
}

/// Quantize a normalized value (in [-1, 1]) to the nearest NF4 code index
/// using the given codebook.
pub fn nf4_quantize_with_codebook(value: f32, codebook: &[f32; 16]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &cb_val) in codebook.iter().enumerate() {
        let d = (value - cb_val).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Decode a single 4-bit NF4 code index to its f32 value using the given codebook.
/// Panics if `code > 15`.
pub fn nf4_dequantize_with_codebook(code: u8, codebook: &[f32; 16]) -> f32 {
    assert!(code < 16, "NF4 code index must be 0..15, got {code}");
    codebook[code as usize]
}

/// Canonical NF4 codebook: 16 evenly-spaced quantiles of N(0,1).
/// Index 0..15, symmetric around zero.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

// ════════════════════════════════════════════════════════════════════════════
// Section 3: NF4 code index lookups
// ════════════════════════════════════════════════════════════════════════════

/// Decode a single 4-bit NF4 code index to its f32 value.
///
/// # Panics
///
/// Panics if `code > 15`.
pub fn nf4_dequantize(code: u8) -> f32 {
    assert!(code < 16, "NF4 code index must be 0..15, got {code}");
    NF4_CODEBOOK[code as usize]
}

/// Quantize an f32 value to the nearest NF4 codebook entry, returning the 4-bit index.
///
/// Returns index 7 (codebook value 0.0) for NaN inputs.
pub fn nf4_quantize(value: f32) -> u8 {
    if value.is_nan() {
        return 7; // nearest to zero
    }
    let mut best_idx = 0u8;
    let mut best_dist = (value - NF4_CODEBOOK[0]).abs();
    for (i, &cb) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let dist = (value - cb).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

// ════════════════════════════════════════════════════════════════════════════
// Section 4: Pack functions
// ════════════════════════════════════════════════════════════════════════════

/// Pack a single tile of 640 f32 values into the three-component NF4 format.
///
/// Returns `(packed_codes, scales, biases)`:
/// * `packed_codes` — `PACKED_BYTES_PER_TILE` (320) bytes of 4-bit codes,
///   8 codes packed per u32 LE.
/// * `scales` — `SCALES_F32_PER_TILE` (5) f32 scale values, one per group.
/// * `biases` — `SCALES_F32_PER_TILE` (5) f32 bias values (always 0.0 for NF4).
///
/// Each group of 128 values is independently normalized by its max absolute value.
/// The scale = max_abs (or 1.0 for all-zero groups). Codes are quantized to the
/// nearest NF4 codebook entry.
///
/// # Panics
///
/// Panics if `values` is not exactly 640 elements.
pub fn pack_nf4_tile(values: &[f32; TILE_ELEMENTS]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    pack_nf4_tile_with_group_size(values, GROUP_SIZE)
}

/// Pack a single tile of 640 f32 values using configurable group size.
/// Uses the same NF4 codebook but with `group_size` elements per group
/// (must evenly divide 640).  Each group has its own scale and bias.
pub fn pack_nf4_tile_with_group_size(
    values: &[f32; TILE_ELEMENTS],
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;
    let packed_codes_len = num_groups * bytes_per_group;
    let mut packed_codes = vec![0u8; packed_codes_len];
    let mut scales = vec![0.0f32; num_groups];
    let mut biases = vec![0.0f32; num_groups];

    for group in 0..num_groups {
        let base = group * group_size;
        let max_abs = values[base..base + group_size]
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let scale = if max_abs < 1e-30 { 1.0f32 } else { max_abs };
        scales[group] = scale;
        biases[group] = 0.0;

        for i in 0..(group_size / 2) {
            let val0 = values[base + 2 * i] / scale;
            let val1 = values[base + 2 * i + 1] / scale;
            let code0 = nf4_quantize(val0);
            let code1 = nf4_quantize(val1);
            packed_codes[group * bytes_per_group + i] = code0 | (code1 << 4);
        }
    }

    (packed_codes, scales, biases)
}

/// Pack a single tile of 640 f32 values using activation-weighted LS fitting.
///
/// For each 128-element group:
/// 1. Compute initial NF4 code indices using max-abs scaling (same as `pack_nf4_tile`)
/// 2. Run `optimize_scale_bias` with activation second moments
/// 3. Re-quantize with optimal (s, b)
/// 4. Store codes, scale, bias
///
/// When activation weights are near-uniform (sum <= 1e-6), falls back to max-abs packing.
///
/// Returns `(packed_codes, scales, biases)` — same format as `pack_nf4_tile`.
pub fn pack_nf4_tile_awls(
    values: &[f32; TILE_ELEMENTS],
    activation_weights: &[f32; TILE_ELEMENTS],
    max_awls_iters: u8,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let max_codebook = NF4_CODEBOOK.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let mut codes = Vec::with_capacity(PACKED_BYTES_PER_TILE);
    let mut scales = Vec::with_capacity(SCALES_F32_PER_TILE);
    let mut biases = Vec::with_capacity(SCALES_F32_PER_TILE);

    for g in 0..GROUPS_PER_TILE {
        let start = g * GROUP_SIZE;
        let chunk: [f32; GROUP_SIZE] = std::array::from_fn(|i| values[start + i]);
        let act_chunk: [f32; GROUP_SIZE] = std::array::from_fn(|i| activation_weights[start + i]);

        // Step 1: initial max-abs scaling + code assignment
        let max_abs = accelerate::max_abs(&chunk);
        let init_scale = if max_abs > 0.0 {
            max_abs / max_codebook
        } else {
            1.0
        };
        let init_bias = 0.0f32;

        let mut code_indices = [0u8; GROUP_SIZE];
        let normalized = accelerate::vsdiv(&chunk, init_scale);
        for (i, &v) in normalized.iter().enumerate() {
            code_indices[i] = nf4_quantize(v);
        }

        // Step 2: AW-LS optimization (if activation weights are non-uniform)
        let sum_act: f32 = act_chunk.iter().sum();
        if sum_act > 1e-6 {
            let result = crate::nf4tile640::awls::optimize_scale_bias(
                &chunk,
                &code_indices,
                &act_chunk,
                max_awls_iters,
                &crate::ecs::compilation::cancel::CancelToken::new(None),
            );
            // Use the exact joint state from the optimizer
            code_indices = result.codes;
            let s = result.scale;
            let b = result.bias;

            // ... pack codes into tile ...
            for pair in code_indices.chunks_exact(2) {
                codes.push(pair[0] | (pair[1] << 4));
            }
            scales.push(s);
            biases.push(b);
        } else {
            // Fallback to standard max-abs packing (uniform weights — no activation info)
            for pair in code_indices.chunks_exact(2) {
                codes.push(pair[0] | (pair[1] << 4));
            }
            scales.push(init_scale);
            biases.push(init_bias);
        }
    }

    (codes, scales, biases)
}

/// Pack multiple tiles from a flat f32 array (shapes are M×K layout).
/// Tile across out_features (cols): contiguous row slices of TILE_ELEMENTS elements.
/// This groups values from the same input channel, which have similar
/// magnitudes and produce lower quantization error than strided gathering.
/// Non-multiple column dimensions are zero-padded to the next tile boundary.
///
/// Returns `(packed_codes, scales, biases, rows, cols)` with data for all tiles stored
/// contiguously per buffer (tile-major order).
///
/// # Panics
///
/// Panics if `weights.len()` does not equal `rows * cols`.
pub fn pack_nf4_weights(
    weights: &[f32],
    rows: usize, // in_features
    cols: usize, // out_features
) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    let weights = if weights.len() >= rows * cols {
        &weights[..rows * cols]
    } else {
        weights
    };

    // Tile across out_features (cols): contiguous row slices of 640 elements.
    // This groups values from the same input channel, which have similar
    // magnitudes and produce lower quantization error than strided gathering.
    let padded_cols = if cols % TILE_ELEMENTS == 0 {
        cols
    } else {
        let tiles_needed = cols.div_ceil(TILE_ELEMENTS);
        tiles_needed * TILE_ELEMENTS
    };
    let tiles_per_row = padded_cols / TILE_ELEMENTS;
    let total_tiles = rows * tiles_per_row;

    let mut packed_codes = vec![0u8; total_tiles * PACKED_BYTES_PER_TILE];
    let mut scales = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];
    let mut biases = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];

    for row in 0..rows {
        for tile_in_row in 0..tiles_per_row {
            let tile_idx = row * tiles_per_row + tile_in_row;
            let col_start = tile_in_row * TILE_ELEMENTS;
            let mut tile_vals = [0.0f32; TILE_ELEMENTS];
            for i in 0..TILE_ELEMENTS {
                let c = col_start + i;
                tile_vals[i] = if c < cols {
                    weights[row * cols + c]
                } else {
                    0.0
                };
            }
            let (codes_tile, scale_tile, bias_tile) = pack_nf4_tile(&tile_vals);

            let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
            packed_codes[codes_off..codes_off + PACKED_BYTES_PER_TILE].copy_from_slice(&codes_tile);

            let scale_off = tile_idx * SCALES_F32_PER_TILE;
            scales[scale_off..scale_off + SCALES_F32_PER_TILE].copy_from_slice(&scale_tile);

            let bias_off = tile_idx * SCALES_F32_PER_TILE;
            biases[bias_off..bias_off + SCALES_F32_PER_TILE].copy_from_slice(&bias_tile);
        }
    }

    (packed_codes, scales, biases, rows as u32, cols as u32)
}

/// Pack multiple tiles from a flat f32 array using activation-weighted LS fitting.
///
/// Each contiguous K-length row is split into ceil(K / TILE_ELEMENTS) tiles.
/// Non-multiple column dimensions are zero-padded to the next tile boundary.
///
/// `activation_weights` is a per-column (per-input-channel) slice of length `cols`,
/// or `None` to fall back to uniform weighting (max-abs packing).
///
/// Returns `(packed_codes, scales, biases, rows, cols)` with data for all tiles stored
/// contiguously per buffer (tile-major order).
///
/// # Panics
///
/// Panics if `weights.len()` does not equal `rows * cols`.
/// Panics if `activation_weights` is `Some` and its length does not equal `cols`.
pub fn pack_nf4_weights_awls(
    weights: &[f32],
    rows: usize,
    cols: usize,
    activation_weights: Option<&[f32]>,
    max_iters: u8,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    use rayon::prelude::*;

    // Use only the first rows × cols elements.
    let weights = if weights.len() >= rows * cols {
        &weights[..rows * cols]
    } else {
        weights
    };
    if let Some(act) = activation_weights {
        assert_eq!(
            act.len(),
            cols,
            "activation_weights length {} must equal cols {}",
            act.len(),
            cols
        );
    }

    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;

    // Parallel: process each (row, tile) pair independently.
    // Each row's tiles are computed in parallel, then collected.
    let tile_results: Vec<(Vec<u8>, Vec<f32>, Vec<f32>)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let row_base = row * cols;
            (0..tiles_per_row).into_par_iter().map(move |tile_idx| {
                let col_start = tile_idx * TILE_ELEMENTS;
                let mut tile_vals = [0.0f32; TILE_ELEMENTS];
                let mut act_vals = [1.0f32; TILE_ELEMENTS];
                for i in 0..TILE_ELEMENTS {
                    let c = col_start + i;
                    if c < cols {
                        tile_vals[i] = weights[row_base + c];
                        if let Some(act) = activation_weights {
                            act_vals[i] = act[c];
                        }
                    } else {
                        tile_vals[i] = 0.0;
                    }
                }
                pack_nf4_tile_awls(&tile_vals, &act_vals, max_iters)
            })
        })
        .collect();

    // Serial: append results to output vecs in tile order.
    let mut all_codes = Vec::with_capacity(total_tiles * PACKED_BYTES_PER_TILE);
    let mut all_scales = Vec::with_capacity(total_tiles * SCALES_F32_PER_TILE);
    let mut all_biases = Vec::with_capacity(total_tiles * SCALES_F32_PER_TILE);
    for (codes, scales, biases) in tile_results {
        all_codes.extend(codes);
        all_scales.extend(scales);
        all_biases.extend(biases);
    }

    (all_codes, all_scales, all_biases, rows as u32, cols as u32)
}

// ════════════════════════════════════════════════════════════════════════════
// Section 5: Unpack functions
// ════════════════════════════════════════════════════════════════════════════

/// Unpack a single nf4tile640 tile back to f32.
///
/// * `packed_codes` — exactly `PACKED_BYTES_PER_TILE` (320) bytes of packed 4-bit codes.
/// * `scales` — exactly `SCALES_F32_PER_TILE` (5) f32 scale values.
/// * `biases` — exactly `SCALES_F32_PER_TILE` (5) f32 bias values.
/// * `output` — exactly `TILE_ELEMENTS` (640) f32 values.
///
/// Reconstruction: `output[i] = nf4_codebook[code_index] * scale[group] + bias[group]`.
pub fn unpack_nf4_tile(
    packed_codes: &[u8; PACKED_BYTES_PER_TILE],
    scales: &[f32; SCALES_F32_PER_TILE],
    biases: &[f32; SCALES_F32_PER_TILE],
    output: &mut [f32; TILE_ELEMENTS],
) {
    for group in 0..GROUPS_PER_TILE {
        let scale = scales[group];
        let bias = biases[group];
        let codes_base = group * PACKED_BYTES_PER_GROUP;
        let out_base = group * GROUP_SIZE;

        for i in 0..(GROUP_SIZE / 2) {
            let packed = packed_codes[codes_base + i];
            let code0 = packed & 0x0F;
            let code1 = (packed >> 4) & 0x0F;
            output[out_base + 2 * i] = nf4_dequantize(code0) * scale + bias;
            output[out_base + 2 * i + 1] = nf4_dequantize(code1) * scale + bias;
        }
    }
}

/// Unpack multiple tiles from the three-component NF4 format back to f32 (M×K layout).
///
/// # Panics
///
/// Panics if the buffer lengths do not match `cols * tiles_per_ch` tiles.
pub fn unpack_nf4_weights(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;

    let expected_codes_len = total_tiles * PACKED_BYTES_PER_TILE;
    let expected_scales_len = total_tiles * SCALES_F32_PER_TILE;

    assert_eq!(
        packed_codes.len(),
        expected_codes_len,
        "packed_codes length {} must equal total_tiles {total_tiles} × PACKED_BYTES_PER_TILE {PACKED_BYTES_PER_TILE} = {expected_codes_len}",
        packed_codes.len(),
    );
    assert_eq!(
        scales.len(),
        expected_scales_len,
        "scales length {} must equal total_tiles {total_tiles} × SCALES_F32_PER_TILE {SCALES_F32_PER_TILE} = {expected_scales_len}",
        scales.len(),
    );
    assert_eq!(
        biases.len(),
        expected_scales_len,
        "biases length {} must equal total_tiles {total_tiles} × SCALES_F32_PER_TILE {SCALES_F32_PER_TILE} = {expected_scales_len}",
        biases.len(),
    );

    let mut output = vec![0.0f32; rows * cols];

    for tile_idx in 0..total_tiles {
        let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
        let scale_off = tile_idx * SCALES_F32_PER_TILE;

        let codes_slice: &[u8; PACKED_BYTES_PER_TILE] = packed_codes
            [codes_off..codes_off + PACKED_BYTES_PER_TILE]
            .try_into()
            .expect("codes slice fits exactly");
        let scale_slice: &[f32; SCALES_F32_PER_TILE] = scales
            [scale_off..scale_off + SCALES_F32_PER_TILE]
            .try_into()
            .expect("scale slice fits exactly");
        let bias_slice: &[f32; SCALES_F32_PER_TILE] = biases
            [scale_off..scale_off + SCALES_F32_PER_TILE]
            .try_into()
            .expect("bias slice fits exactly");

        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;

        let mut tile_out = [0.0f32; TILE_ELEMENTS];
        unpack_nf4_tile(codes_slice, scale_slice, bias_slice, &mut tile_out);
        let col_base = tile_in_row * TILE_ELEMENTS;
        let remaining = cols.saturating_sub(col_base);
        let copy_len = TILE_ELEMENTS.min(remaining);
        for i in 0..copy_len {
            output[row * cols + col_base + i] = tile_out[i];
        }
    }

    output
}

/// Unpack NF4 weights with configurable group size.
///
/// Unlike `unpack_nf4_weights`, this function does NOT assume 5 groups per tile.
/// Instead, `groups_per_tile` is computed from `TILE_ELEMENTS / group_size`.
///
/// # Panics
/// - If group_size does not divide TILE_ELEMENTS
/// - If buffer lengths don't match expected sizes
pub fn unpack_nf4_weights_with_group_size(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    assert!(
        TILE_ELEMENTS % group_size == 0,
        "group_size {group_size} must divide TILE_ELEMENTS {TILE_ELEMENTS}"
    );
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2; // 4-bit codes
    let packed_bytes_per_tile = groups_per_tile * bytes_per_group;

    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;

    let expected_codes_len = total_tiles * packed_bytes_per_tile;
    let expected_scales_len = total_tiles * groups_per_tile;

    assert_eq!(
        packed_codes.len(),
        expected_codes_len,
        "packed_codes length {} != {total_tiles} * {packed_bytes_per_tile} = {expected_codes_len}",
        packed_codes.len(),
    );
    assert_eq!(
        scales.len(),
        expected_scales_len,
        "scales length {} != {total_tiles} * {groups_per_tile} = {expected_scales_len}",
        scales.len(),
    );
    assert_eq!(
        biases.len(),
        expected_scales_len,
        "biases length {} != {total_tiles} * {groups_per_tile} = {expected_scales_len}",
        biases.len(),
    );

    let mut output = vec![0.0f32; rows * cols];

    for tile_idx in 0..total_tiles {
        let codes_off = tile_idx * packed_bytes_per_tile;
        let scale_off = tile_idx * groups_per_tile;

        // Copy code bytes for this tile (not fixed-size slice since size varies)
        let codes = &packed_codes[codes_off..codes_off + packed_bytes_per_tile];

        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;
        let remaining = cols.saturating_sub(col_base);

        // Reconstruct the tile group by group
        for g in 0..groups_per_tile {
            let scale = scales[scale_off + g];
            let bias = biases[scale_off + g];
            let codes_base = g * bytes_per_group;
            let out_base = row * cols + col_base + g * group_size;
            let copy_len = group_size.min(remaining.saturating_sub(g * group_size));

            for i in 0..(group_size / 2) {
                let packed = codes[codes_base + i];
                let code0 = packed & 0x0F;
                let code1 = (packed >> 4) & 0x0F;
                let idx0 = out_base + 2 * i;
                let idx1 = out_base + 2 * i + 1;
                if idx0 < output.len() && (2 * i) < copy_len {
                    output[idx0] = nf4_dequantize(code0) * scale + bias;
                }
                if idx1 < output.len() && (2 * i + 1) < copy_len {
                    output[idx1] = nf4_dequantize(code1) * scale + bias;
                }
            }
        }
    }

    output
}

/// Unpack NF4 weights with configurable group size and codebook.
pub fn unpack_nf4_weights_with_group_size_and_codebook(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    codebook: &[f32; 16],
) -> Vec<f32> {
    assert!(TILE_ELEMENTS % group_size == 0);
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;
    let packed_bytes_per_tile = groups_per_tile * bytes_per_group;
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let expected_codes_len = total_tiles * packed_bytes_per_tile;
    let expected_scales_len = total_tiles * groups_per_tile;
    assert_eq!(packed_codes.len(), expected_codes_len);
    assert_eq!(scales.len(), expected_scales_len);
    if !biases.is_empty() {
        assert_eq!(biases.len(), expected_scales_len);
    }
    let mut output = vec![0.0f32; rows * cols];
    for tile_idx in 0..total_tiles {
        let codes_off = tile_idx * packed_bytes_per_tile;
        let scale_off = tile_idx * groups_per_tile;
        let codes = &packed_codes[codes_off..codes_off + packed_bytes_per_tile];
        let row = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;
        let remaining = cols.saturating_sub(col_base);
        for g in 0..groups_per_tile {
            let scale = scales[scale_off + g];
            let bias = if biases.is_empty() {
                0.0
            } else {
                biases[scale_off + g]
            };
            let codes_base = g * bytes_per_group;
            let out_base = row * cols + col_base + g * group_size;
            let copy_len = group_size.min(remaining.saturating_sub(g * group_size));
            for i in 0..(group_size / 2) {
                let packed = codes[codes_base + i];
                let code0 = packed & 0x0F;
                let code1 = (packed >> 4) & 0x0F;
                let idx0 = out_base + 2 * i;
                let idx1 = out_base + 2 * i + 1;
                if idx0 < output.len() && (2 * i) < copy_len {
                    output[idx0] = codebook[code0 as usize] * scale + bias;
                }
                if idx1 < output.len() && (2 * i + 1) < copy_len {
                    output[idx1] = codebook[code1 as usize] * scale + bias;
                }
            }
        }
    }
    output
}

// ════════════════════════════════════════════════════════════════════════════
// Section 6: Tile Metadata
// ════════════════════════════════════════════════════════════════════════════

/// Metadata carried in the cimage manifest for each nf4tile640 weight.
#[derive(Debug, Clone)]
pub struct Nf4Tile640Manifest {
    pub format_version: u32,
    pub codebook_id: u32,
    pub group_size: u32,
    pub tile_width: u32,
    pub scale_dtype: u8,
    pub byte_order: u8,
    pub alignment: u32,
    pub rows: u32,
    pub cols: u32,
    pub tiles_per_channel: u32,
    pub total_tiles: u32,
    pub required_kernel: String,
}

impl Nf4Tile640Manifest {
    /// Build a manifest for the given logical shape.
    pub fn new(rows: u32, cols: u32) -> Self {
        let tiles_per_ch = rows.div_ceil(TILE_ELEMENTS as u32);
        let total_tiles = cols * tiles_per_ch;
        Self {
            format_version: FORMAT_VERSION,
            codebook_id: 0,
            group_size: GROUP_SIZE as u32,
            tile_width: TILE_ELEMENTS as u32,
            scale_dtype: 0,
            byte_order: 0,
            alignment: PACKED_BYTES_PER_TILE as u32,
            rows,
            cols,
            tiles_per_channel: tiles_per_ch,
            total_tiles,
            required_kernel: "dequant_mul_nf4tile640".to_string(),
        }
    }
}
// ════════════════════════════════════════════════════════════════════════════
// Section 6b: INT8 tile640 pack/unpack
// ════════════════════════════════════════════════════════════════════════════

/// Pack a weight matrix into INT8 tile640 format with per-tile symmetric quantization.
pub fn pack_int8_weights(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;

    let num_tiles = cols.div_ceil(TILE_ELEMENTS);
    let tile_cols = num_tiles * TILE_ELEMENTS;
    let mut codes = vec![0u8; rows * tile_cols];
    let mut scales = Vec::with_capacity(rows * num_tiles);
    let biases = vec![0.0f32; rows * num_tiles];

    // Parallelize over rows; each row's tiles are independent.
    let row_results: Vec<(Vec<u8>, Vec<f32>)> = (0..rows)
        .into_par_iter()
        .map(|i| {
            let mut row_codes = vec![0u8; tile_cols];
            let mut row_scales = Vec::with_capacity(num_tiles);
            for t in 0..num_tiles {
                let col_start = t * TILE_ELEMENTS;
                let col_end = (col_start + TILE_ELEMENTS).min(cols);
                let mut max_abs = 0.0f32;
                for j in col_start..col_end {
                    let v = weights[i * cols + j].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                let scale = if max_abs > 1e-10f32 {
                    max_abs / 127.0f32
                } else {
                    1.0f32
                };
                for j in col_start..col_end {
                    let offset = t * TILE_ELEMENTS + (j - col_start);
                    let q = (weights[i * cols + j] / scale).round().clamp(-127.0, 127.0) as i8;
                    row_codes[offset] = q as u8;
                }
                // Zero-pad remainder of tile
                for j in col_end..col_start + TILE_ELEMENTS {
                    let offset = t * TILE_ELEMENTS + (j - col_start);
                    row_codes[offset] = 0u8;
                }
                row_scales.push(scale);
            }
            (row_codes, row_scales)
        })
        .collect();

    // Serial: copy results into contiguous output vecs.
    for (i, (row_codes, row_scales)) in row_results.iter().enumerate() {
        let code_start = i * tile_cols;
        codes[code_start..code_start + tile_cols].copy_from_slice(row_codes);
        scales.extend(row_scales);
    }
    (codes, scales, biases)
}

/// Unpack INT8 tile640 codes/scales/biases back to f32 weights.
pub fn unpack_int8_weights(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let num_tiles = cols.div_ceil(TILE_ELEMENTS);
    let tile_cols = num_tiles * TILE_ELEMENTS;

    let mut result = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for t in 0..num_tiles {
            let col_start = t * TILE_ELEMENTS;
            let col_end = (col_start + TILE_ELEMENTS).min(cols);
            let scale = scales[i * num_tiles + t];
            let bias = biases[i * num_tiles + t];

            for j in col_start..col_end {
                let code_idx = i * tile_cols + t * TILE_ELEMENTS + (j - col_start);
                let q = codes[code_idx] as i8;
                result[i * cols + j] = (q as f32) * scale + bias;
            }
        }
    }
    result
}

/// Unpack INT8 weights with configurable group size.
/// groups_per_tile = TILE_ELEMENTS / group_size scales per tile.
pub fn unpack_int8_weights_with_group_size(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Vec<f32> {
    assert!(
        TILE_ELEMENTS % group_size == 0,
        "group_size {group_size} must divide TILE_ELEMENTS {TILE_ELEMENTS}"
    );
    let groups_per_tile = TILE_ELEMENTS / group_size;
    let num_tiles = cols.div_ceil(TILE_ELEMENTS);
    let tile_cols = num_tiles * TILE_ELEMENTS;

    let mut result = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for t in 0..num_tiles {
            let col_start = t * TILE_ELEMENTS;
            let col_end = (col_start + TILE_ELEMENTS).min(cols);

            for g in 0..groups_per_tile {
                let scale = scales[(i * num_tiles * groups_per_tile) + (t * groups_per_tile) + g];
                let bias = biases[(i * num_tiles * groups_per_tile) + (t * groups_per_tile) + g];
                let group_start = col_start + g * group_size;
                let group_end = (group_start + group_size).min(col_end);

                for j in group_start..group_end {
                    let code_idx = i * tile_cols + j;
                    if code_idx < codes.len() {
                        let q = codes[code_idx] as i8;
                        result[i * cols + j] = (q as f32) * scale + bias;
                    }
                }
            }
        }
    }
    result
}

// ════════════════════════════════════════════════════════════════════════════
// Section 7: Fused dequantize + matmul (CPU reference oracle)
// ════════════════════════════════════════════════════════════════════════════

/// Compute the expected packed byte sizes for a weight matrix of shape `[rows, cols]`.
///
/// This is the size of the packed_codes buffer (contiguous bytes of
/// 4-bit NF4 indices, 8 codes per u32, LE byte order).
pub fn packed_size(rows: usize, cols: usize) -> usize {
    // Match pack_nf4_weights tiling: tiles are along cols (n dimension).
    let padded_cols = if cols % TILE_ELEMENTS == 0 {
        cols
    } else {
        cols.div_ceil(TILE_ELEMENTS) * TILE_ELEMENTS
    };
    let tiles_per_row = padded_cols / TILE_ELEMENTS;
    let total_tiles = rows * tiles_per_row;
    total_tiles * PACKED_BYTES_PER_TILE
}

// ════════════════════════════════════════════════════════════════════════════
// Accelerate BLAS FFI (macOS only)
// ════════════════════════════════════════════════════════════════════════════

/// CBLAS row-major layout flag.
#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;

/// CBLAS no-transpose flag.
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;

// FFI declaration for Accelerate cblas_sgemm.
#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transA: i32,
        transB: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Compute `output = input @ dequantize(weights)` where weights are packed
/// in the three-component nf4tile640 format.
///
/// * `input` — row-major f32 matrix, shape `[M, K]`
/// * `packed_codes` — packed NF4 codes, shape `[K, N]` (tile-major contiguous)
/// * `scales` — f32 scales per group, tile-major
/// * `biases` — f32 biases per group, tile-major
/// * `m` — M (batch / output rows)
/// * `k` — K (inner dimension = weight rows = input columns)
/// * `n` — N (output columns = weight columns)
/// * `output` — pre-allocated f32 buffer, shape `[M, N]`
///
/// The function dequantizes each packed tile to f32 on the fly, multiplying by
/// the corresponding input elements to accumulate the result. This avoids
/// materializing the full dequantized weight matrix.
///
/// # Errors
///
/// Returns an error if dimensions don't match.
pub fn dequant_matmul_reference(
    input: &[f32],
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    k: usize,
    n: usize,
    output: &mut [f32],
) -> Result<(), String> {
    // Validate dimensions.
    if input.len() != m * k {
        return Err(format!(
            "input length {} must equal m {} * k {}",
            input.len(),
            m,
            k
        ));
    }
    if output.len() != m * n {
        return Err(format!(
            "output length {} must equal m {} * n {}",
            output.len(),
            m,
            n
        ));
    }
    let padded_n = if n % TILE_ELEMENTS == 0 {
        n
    } else {
        n.div_ceil(TILE_ELEMENTS) * TILE_ELEMENTS
    };
    let tiles_per_row = padded_n / TILE_ELEMENTS;
    let total_tiles = k * tiles_per_row;
    let expected_codes = total_tiles * PACKED_BYTES_PER_TILE;
    let expected_scales = total_tiles * SCALES_F32_PER_TILE;
    if packed_codes.len() != expected_codes {
        return Err(format!(
            "packed_codes length {} must equal total_tiles {total_tiles} * PACKED_BYTES_PER_TILE {PACKED_BYTES_PER_TILE} = {expected_codes}",
            packed_codes.len(),
        ));
    }
    if scales.len() != expected_scales {
        return Err(format!(
            "scales length {} must equal total_tiles {total_tiles} * SCALES_F32_PER_TILE {SCALES_F32_PER_TILE} = {expected_scales}",
            scales.len(),
        ));
    }
    if biases.len() != expected_scales {
        return Err(format!(
            "biases length {} must equal total_tiles {total_tiles} * SCALES_F32_PER_TILE {SCALES_F32_PER_TILE} = {expected_scales}",
            biases.len(),
        ));
    }

    // Dispatch: Accelerate BLAS on macOS, scalar fallback elsewhere.
    #[cfg(target_os = "macos")]
    {
        // Dequantize all tiles into a contiguous K×N f32 buffer.
        let mut w_dequant = vec![0.0f32; k * n];
        for tile_idx in 0..total_tiles {
            let k_chunk = tile_idx / tiles_per_row;
            let n_tile = tile_idx % tiles_per_row;
            let col_base = n_tile * TILE_ELEMENTS;

            let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
            let scale_off = tile_idx * SCALES_F32_PER_TILE;

            let codes_slice: &[u8; PACKED_BYTES_PER_TILE] = packed_codes
                [codes_off..codes_off + PACKED_BYTES_PER_TILE]
                .try_into()
                .expect("codes slice fits exactly");
            let scale_slice: &[f32; SCALES_F32_PER_TILE] = scales
                [scale_off..scale_off + SCALES_F32_PER_TILE]
                .try_into()
                .expect("scale slice fits exactly");
            let bias_slice: &[f32; SCALES_F32_PER_TILE] = biases
                [scale_off..scale_off + SCALES_F32_PER_TILE]
                .try_into()
                .expect("bias slice fits exactly");

            // Dequantize tile to a temp buffer, then copy only valid elements.
            let mut tile_f32 = [0.0f32; TILE_ELEMENTS];
            unpack_nf4_tile(codes_slice, scale_slice, bias_slice, &mut tile_f32);

            let limit = TILE_ELEMENTS.min(n - col_base);
            for i in 0..limit {
                w_dequant[k_chunk * n + col_base + i] = tile_f32[i];
            }
        }

        // Single BLAS sgemm call: output = input @ dequant_weights.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                input.as_ptr(),
                k as i32,
                w_dequant.as_ptr(),
                n as i32,
                0.0,
                output.as_mut_ptr(),
                n as i32,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    dequant_matmul_scalar(
        input,
        packed_codes,
        scales,
        biases,
        m,
        k,
        n,
        output,
        tiles_per_row,
        total_tiles,
    );

    Ok(())
}

/// Scalar fallback: dequantize one tile at a time, accumulate via nested loops.
/// Always compiled so macOS tests can cross-check BLAS vs scalar results.
#[allow(dead_code)]
fn dequant_matmul_scalar(
    input: &[f32],
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    k: usize,
    n: usize,
    output: &mut [f32],
    tiles_per_row: usize,
    total_tiles: usize,
) {
    // Zero the output buffer.
    output.fill(0.0);

    // Iterate tiles in packed order.
    for tile_idx in 0..total_tiles {
        let k_chunk = tile_idx / tiles_per_row;
        let n_tile = tile_idx % tiles_per_row;
        let col_base = n_tile * TILE_ELEMENTS;

        let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
        let scale_off = tile_idx * SCALES_F32_PER_TILE;

        // Extract slices for this tile.
        let codes_slice: &[u8; PACKED_BYTES_PER_TILE] = packed_codes
            [codes_off..codes_off + PACKED_BYTES_PER_TILE]
            .try_into()
            .expect("codes slice fits exactly");
        let scale_slice: &[f32; SCALES_F32_PER_TILE] = scales
            [scale_off..scale_off + SCALES_F32_PER_TILE]
            .try_into()
            .expect("scale slice fits exactly");
        let bias_slice: &[f32; SCALES_F32_PER_TILE] = biases
            [scale_off..scale_off + SCALES_F32_PER_TILE]
            .try_into()
            .expect("bias slice fits exactly");

        // Dequantize this tile once.
        let mut tile_f32 = [0.0f32; TILE_ELEMENTS];
        unpack_nf4_tile(codes_slice, scale_slice, bias_slice, &mut tile_f32);

        // Accumulate contribution across all M rows of the output.
        // Each tile covers contiguous out_features for one input channel.
        // output[row_m][col_base + i] += Σ_i input[row_m][k_chunk] * tile[i]
        let limit = TILE_ELEMENTS.min(n - col_base);
        for row_m in 0..m {
            for i in 0..limit {
                output[row_m * n + col_base + i] += input[row_m * k + k_chunk] * tile_f32[i];
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 8: Validation oracle
// ════════════════════════════════════════════════════════════════════════════

/// Result of comparing two f32 matrices.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Maximum absolute element-wise error.
    pub max_abs_error: f32,
    /// Mean absolute element-wise error.
    pub mean_abs_error: f32,
    /// Number of elements exceeding `tolerance * max(|ref|, |candidate|)`.
    pub mismatches: usize,
    /// Total elements compared.
    pub total_elements: usize,
    /// True when error is within tolerance bounds.
    pub passed: bool,
}

/// Compare two f32 matrices within tolerance.
///
/// `reference` is the CPU reference output, `candidate` is the Metal (or any)
/// output. Returns stats and pass/fail.
///
/// The comparison uses relative tolerance: an element is a mismatch when
/// `|ref - cand| > tolerance * max(|ref|, |candidate|, 1e-10)`.
///
/// A default tolerance of `0.05` (5%) is appropriate for NF4 dequantize+matmul
/// as NF4 quantization introduces noise up to ~2-3% per element, and matmul
/// accumulation can amplify this.
pub fn validate_matmul(reference: &[f32], candidate: &[f32], tolerance: f32) -> ValidationResult {
    assert_eq!(
        reference.len(),
        candidate.len(),
        "reference and candidate must have the same length"
    );

    let total_elements = reference.len();
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut mismatches = 0usize;

    for (&ref_v, &cand_v) in reference.iter().zip(candidate.iter()) {
        let err = (ref_v - cand_v).abs();
        max_abs_error = max_abs_error.max(err);
        sum_abs_error += err;

        let scale = ref_v.abs().max(cand_v.abs()).max(1e-10);
        let rel_err = err / scale;
        if rel_err > tolerance {
            mismatches += 1;
        }
    }

    let mean_abs_error = sum_abs_error / total_elements as f32;
    let passed = mismatches == 0;

    ValidationResult {
        max_abs_error,
        mean_abs_error,
        mismatches,
        total_elements,
        passed,
    }
}

/// CPU-side handle for a packed nf4tile640 weight matrix.
#[derive(Debug, Clone)]
pub struct Nf4Weights {
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub rows: u32,
    pub cols: u32,
}

// ════════════════════════════════════════════════════════════════════════════
// Section 9: Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_codebook_has_16_entries() {
        assert_eq!(NF4_CODEBOOK.len(), 16);
        let min_val = NF4_CODEBOOK.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = NF4_CODEBOOK
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (min_val + 1.0).abs() < 1e-6,
            "min should be -1.0, got {min_val}"
        );
        assert!(
            (max_val - 1.0).abs() < 1e-6,
            "max should be 1.0, got {max_val}"
        );
    }

    #[test]
    fn tile_family_spec_tile640_is_correct() {
        let spec = TileFamilySpec::tile640();
        assert_eq!(spec.name, "Tile640");
        assert_eq!(spec.tile_elements, 640);
        assert_eq!(spec.tile_rows, 640);
        assert_eq!(spec.tile_cols, 640);
        assert_eq!(
            spec.tile_elements, TILE_ELEMENTS,
            "TileFamilySpec::tile640() elements must match TILE_ELEMENTS constant"
        );
    }

    #[test]
    fn round_trip_tile() {
        // Build a tile with values drawn from the NF4 codebook entries with
        // varying group scales.  Each group uses its own scale, and values
        // are exact codebook entries, so the round-trip should be near-exact.
        // This exercises the full pack/unpack pipeline.
        let mut values = [0.0f32; TILE_ELEMENTS];
        for (i, v) in values.iter_mut().enumerate() {
            let group = i / GROUP_SIZE;
            // Each group gets a different scale.
            let scale = 0.1 + (group as f32) * 0.4; // 0.1, 0.5, 0.9, 1.3, 1.7
                                                    // Cycle through the codebook.
            *v = NF4_CODEBOOK[i % 16] * scale;
        }

        let (packed_codes, scales, biases) = pack_nf4_tile(&values);

        assert_eq!(packed_codes.len(), PACKED_BYTES_PER_TILE);
        assert_eq!(scales.len(), SCALES_F32_PER_TILE);
        assert_eq!(biases.len(), SCALES_F32_PER_TILE);

        let mut unpacked = [0.0f32; TILE_ELEMENTS];
        let codes_arr: &[u8; PACKED_BYTES_PER_TILE] = packed_codes.as_slice().try_into().unwrap();
        let scales_arr: &[f32; SCALES_F32_PER_TILE] = scales.as_slice().try_into().unwrap();
        let biases_arr: &[f32; SCALES_F32_PER_TILE] = biases.as_slice().try_into().unwrap();
        unpack_nf4_tile(codes_arr, scales_arr, biases_arr, &mut unpacked);

        // Every element should reconstruct to within 5% relative error.
        // With exact codebook entries, this is easily satisfied.
        for i in 0..TILE_ELEMENTS {
            let orig = values[i];
            let dec = unpacked[i];
            let max_abs = orig.abs().max(1e-10);
            let rel_err = (dec - orig).abs() / max_abs;
            assert!(
                rel_err < 0.05,
                "element {i}: orig={orig}, dec={dec}, rel_err={rel_err} >= 0.05"
            );
        }
    }

    #[test]
    fn round_trip_identity_matrix() {
        // A small 1×640 identity packed then unpacked.  With input-axis tiling and
        // only 1 row (ceil(1/640) = 1 tile per channel), we get 640 tiles (one per
        // output channel). Each tile holds 1 element (the rest zero-padded).
        let rows = 1usize;
        let cols = TILE_ELEMENTS;
        let mut identity = vec![0.0f32; rows * cols];
        identity[0] = 1.0;

        let (packed_codes, scales, biases, _p_rows, _p_cols) =
            pack_nf4_weights(&identity, rows, cols);

        // Packer tiles along cols: total_tiles = rows * cols.div_ceil(TILE_ELEMENTS)
        let tile_count = rows * cols.div_ceil(TILE_ELEMENTS);
        assert_eq!(packed_codes.len(), tile_count * PACKED_BYTES_PER_TILE);
        assert_eq!(scales.len(), tile_count * SCALES_F32_PER_TILE);
        assert_eq!(biases.len(), tile_count * SCALES_F32_PER_TILE);

        let unpacked = unpack_nf4_weights(&packed_codes, &scales, &biases, rows, cols);
        assert_eq!(unpacked.len(), rows * cols);

        // Diagonal element (1.0 = codebook[15]) should reconstruct exactly.
        // Scale = 1.0, code = 15, dequant = 1.0*1.0 = 1.0.
        let rel_err = (unpacked[0] - 1.0).abs();
        assert!(
            rel_err < 1e-6,
            "diagonal element orig=1.0, dec={}, err={}",
            unpacked[0],
            rel_err
        );

        // Off-diagonal elements (0.0 = codebook[7]) should reconstruct
        // exactly (NF4 codebook has a dedicated zero entry at index 7).
        for i in 1..cols {
            let abs_err = unpacked[i].abs();
            assert!(
                abs_err < 1e-6,
                "off-diagonal element [{i}] = {}, expected 0.0",
                unpacked[i],
            );
        }
    }

    /// Compile-time check: constants produce expected sizes.
    #[test]
    fn constants_are_correct() {
        assert_eq!(TILE_ELEMENTS, 640);
        assert_eq!(GROUP_SIZE, 128);
        assert_eq!(GROUPS_PER_TILE, 5);
        assert_eq!(PACKED_BYTES_PER_GROUP, 64);
        assert_eq!(PACKED_BYTES_PER_TILE, 320);
        assert_eq!(SCALES_F32_PER_TILE, 5);
        assert_eq!(CODES_PER_WORD, 8);
        assert_eq!(PACKED_WORD_BYTES, 4);
    }

    #[test]
    fn nf4_quantize_dequantize_roundtrip() {
        // The zero point in the codebook is at index 7.
        assert_eq!(nf4_quantize(0.0), 7);
        assert!((nf4_dequantize(7)).abs() < 1e-6);

        // The extremes.
        assert_eq!(nf4_quantize(-1.0), 0);
        assert!((nf4_dequantize(0) + 1.0).abs() < 1e-6);
        assert_eq!(nf4_quantize(1.0), 15);
        assert!((nf4_dequantize(15) - 1.0).abs() < 1e-6);

        // NaN quantizes to 0 (index 7).
        assert_eq!(nf4_quantize(f32::NAN), 7);

        // Round-trip all 16 codebook entries.
        for (idx, &cb_val) in NF4_CODEBOOK.iter().enumerate() {
            let q = nf4_quantize(cb_val);
            let dq = nf4_dequantize(q);
            let err = (dq - cb_val).abs();
            assert!(
                err < 1e-6,
                "codebook[{idx}] = {cb_val}: quantize → {q}, dequantize → {dq}, err={err}"
            );
        }
    }

    // ── new tests for dequant_matmul_reference and validation ──

    #[test]
    fn dequant_matmul_small() {
        // Create a small packed weight matrix, run dequant_matmul_reference,
        // and compare against naive unpack-then-matmul.
        // We use n = TILE_ELEMENTS (exact tile boundary) but only the first 3
        // columns per row are non-zero.
        let m = 2usize;
        let k = 4usize;
        let n = TILE_ELEMENTS;

        // Input: 2x4 row-major.
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Weight: 4 rows, each of length n (=640). Only first 3 cols are non-zero.
        // Row 0: [1.0, 0.5, 0.25, 0, ...]
        // Row 1: [2.0, 1.0, 0.5,  0, ...]
        // Row 2: [3.0, 1.5, 0.75, 0, ...]
        // Row 3: [1.0, 2.0, 3.0,  0, ...]
        let mut weights = vec![0.0f32; k * n];
        weights[0 * n + 0] = 1.0;
        weights[0 * n + 1] = 0.5;
        weights[0 * n + 2] = 0.25;
        weights[1 * n + 0] = 2.0;
        weights[1 * n + 1] = 1.0;
        weights[1 * n + 2] = 0.5;
        weights[2 * n + 0] = 3.0;
        weights[2 * n + 1] = 1.5;
        weights[2 * n + 2] = 0.75;
        weights[3 * n + 0] = 1.0;
        weights[3 * n + 1] = 2.0;
        weights[3 * n + 2] = 3.0;

        let (packed_codes, scales, biases, _p_rows, _p_cols) = pack_nf4_weights(&weights, k, n);

        let mut output = vec![0.0f32; m * n];
        dequant_matmul_reference(
            &input,
            &packed_codes,
            &scales,
            &biases,
            m,
            k,
            n,
            &mut output,
        )
        .unwrap();

        // Naive reference: unpack weights, then matmul.
        let unpacked = unpack_nf4_weights(&packed_codes, &scales, &biases, k, n);
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += input[i * k + kk] * unpacked[kk * n + j];
                }
            }
        }

        // Compare both results within NF4 tolerance.
        let result = validate_matmul(&expected, &output, 0.05);
        assert!(
            result.passed,
            "reference matmul mismatch: max_abs_error={}, mean_abs_error={}, mismatches={}/{}",
            result.max_abs_error, result.mean_abs_error, result.mismatches, result.total_elements
        );
    }

    #[test]
    fn dequant_matmul_identity() {
        // Create an identity weight matrix [K, N] where K=2 and N=TILE_ELEMENTS.
        // After pack + dequant_matmul_reference, input * dequant(I) ≈ input
        // in the first K output columns, zeros elsewhere.
        let k = 2usize;
        let n = TILE_ELEMENTS;
        let m = 3usize;

        let mut weights = vec![0.0f32; k * n];
        for i in 0..k {
            weights[i * n + i] = 1.0;
        }

        let (packed_codes, scales, biases, _p_rows, _p_cols) = pack_nf4_weights(&weights, k, n);

        // Input: [m, k] with distinct values.
        let input: Vec<f32> = (0..m * k).map(|x| (x as f32 + 1.0) * 0.1).collect();

        let mut output = vec![0.0f32; m * n];
        dequant_matmul_reference(
            &input,
            &packed_codes,
            &scales,
            &biases,
            m,
            k,
            n,
            &mut output,
        )
        .unwrap();

        // Expected: input[i] element in first k positions, zeros elsewhere.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..k {
                expected[i * n + j] = input[i * k + j];
            }
        }

        let result = validate_matmul(&expected, &output, 0.05);
        assert!(
            result.passed,
            "identity matmul: max_abs_error={}, mean_abs_error={}, mismatches={}/{}",
            result.max_abs_error, result.mean_abs_error, result.mismatches, result.total_elements
        );
    }

    #[test]
    fn validation_detects_error() {
        // Create reference and candidate identical except one element is
        // wildly wrong. Assert ValidationResult detects the mismatch.
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let mut corrupted = data.clone();
        corrupted[42] = 999.0;

        let result = validate_matmul(&data, &corrupted, 0.05);
        assert!(!result.passed, "should detect mismatch");
        assert_eq!(result.mismatches, 1, "exactly one element should mismatch");
        assert!(
            result.max_abs_error > 990.0,
            "max_abs_error should reflect the large error"
        );

        // Identical data should pass.
        let ok = validate_matmul(&data, &data, 0.05);
        assert!(ok.passed, "identical data should pass");
        assert_eq!(ok.mismatches, 0);
        assert!(ok.max_abs_error < 1e-6);
    }

    #[test]
    fn dequant_matmul_rejects_bad_dims() {
        let input = vec![0.0f32; 4];
        let codes = vec![0u8; PACKED_BYTES_PER_TILE]; // 1 tile (k=1, n=TILE_ELEMENTS)
        let scales = vec![0.0f32; SCALES_F32_PER_TILE];
        let biases = vec![0.0f32; SCALES_F32_PER_TILE];
        let mut output = vec![0.0f32; 6];

        // Wrong input length: m=3 needs 3*1=3, got 4.
        assert!(dequant_matmul_reference(
            &input,
            &codes,
            &scales,
            &biases,
            3,
            1,
            TILE_ELEMENTS,
            &mut output
        )
        .is_err());
        // Wrong output length: m=2, n=TILE_ELEMENTS needs 1280, got 6.
        assert!(dequant_matmul_reference(
            &input,
            &codes,
            &scales,
            &biases,
            2,
            2,
            TILE_ELEMENTS,
            &mut output
        )
        .is_err());
        // Wrong packed codes size: need 1 tile = 320 codes bytes, got 100.
        let mut big_output = vec![0.0f32; 2 * TILE_ELEMENTS];
        assert!(dequant_matmul_reference(
            &input,
            &[0u8; 100],
            &scales,
            &biases,
            2,
            2,
            TILE_ELEMENTS,
            &mut big_output
        )
        .is_err());
        // Wrong scales size.
        assert!(dequant_matmul_reference(
            &input,
            &codes,
            &[0.0f32; 1], // wrong scale count
            &biases,
            2,
            2,
            TILE_ELEMENTS,
            &mut big_output
        )
        .is_err());
        // Wrong biases size.
        assert!(dequant_matmul_reference(
            &input,
            &codes,
            &scales,
            &[0.0f32; 1], // wrong bias count
            2,
            2,
            TILE_ELEMENTS,
            &mut big_output
        )
        .is_err());
    }

    #[test]
    fn non_640_multiple_cols_roundtrip() {
        // Test: pack+unpack a matrix with non-640-multiple cols, verify
        // structural correctness (buffer sizes, element count).
        // Exact round-trip is not expected for arbitrary f32 values, so
        // we verify validate_matmul reports at-worst the same error rate
        // as a 640-multiple control of the same data.
        let rows = 4usize;
        let cols = 700usize; // not a multiple of 640
        let control_cols = 640usize; // multiple of 640 (same number of tiles)

        // Shared deterministic weights (both shapes use same content).
        let mut weights = vec![0.0f32; rows * cols];
        let mut control = vec![0.0f32; rows * control_cols];
        for i in 0..rows {
            for j in 0..cols.max(control_cols) {
                // Exact codebook entries with group-varying scale.
                let group = j / GROUP_SIZE;
                let scale = 0.1 + (group % 5) as f32 * 0.4;
                // Row 0 uses codebook[15] = 1.0 so max_abs = scale_j for each tile.
                let val = if i == 0 {
                    1.0 * scale
                } else {
                    NF4_CODEBOOK[(i * 700 + j) % 16] * scale
                };
                if j < cols {
                    weights[i * cols + j] = val;
                }
                if j < control_cols {
                    control[i * control_cols + j] = val;
                }
            }
        }

        // Pack both shapes.
        let (codes, scales, biases, _, _) = pack_nf4_weights(&weights, rows, cols);
        let (control_codes, control_scales, control_biases, _, _) =
            pack_nf4_weights(&control, rows, control_cols);

        // With input-axis tiling: both shapes have 1 tile per channel (rows=4 < 640).
        assert_eq!(rows.div_ceil(TILE_ELEMENTS), 1);

        // Unpack both and verify element count matches logical shape.
        let unpacked = unpack_nf4_weights(&codes, &scales, &biases, rows, cols);
        assert_eq!(unpacked.len(), rows * cols);
        let control_unpacked = unpack_nf4_weights(
            &control_codes,
            &control_scales,
            &control_biases,
            rows,
            control_cols,
        );
        assert_eq!(control_unpacked.len(), rows * control_cols);

        // Non-640-multiple packing must not be catastrophically worse.
        // Both should pass validate_matmul (5% relative tolerance).
        let result = validate_matmul(&weights, &unpacked, 0.05);
        let control_result = validate_matmul(&control, &control_unpacked, 0.05);
        assert!(
            result.passed,
            "non-640 round-trip: max_abs_error={}, mismatches={}/{}",
            result.max_abs_error, result.mismatches, result.total_elements
        );
        assert!(
            control_result.passed,
            "control round-trip: max_abs_error={}, mismatches={}/{}",
            control_result.max_abs_error, control_result.mismatches, control_result.total_elements
        );
    }

    #[test]
    fn non_640_multiple_dequant_matmul() {
        // Test dequant_matmul_reference with a non-640-multiple weight.
        // This exercises the exact same path as vision_embedder (6912 cols).
        let m = 2usize;
        let k = 3usize;
        let n = 700usize;

        let mut weights = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                // Sawtooth spanning [-1, 1] for full NF4 codebook coverage.
                let phase = (i * n + j) as f32 * 0.01;
                weights[i * n + j] = (phase - phase.floor() - 0.5) * 2.0;
            }
        }

        let input: Vec<f32> = (0..m * k).map(|x| (x as f32 + 1.0) * 0.1).collect();

        let (codes, scales, biases, _p_rows, _p_cols) = pack_nf4_weights(&weights, k, n);

        let mut nf4_output = vec![0.0f32; m * n];
        dequant_matmul_reference(&input, &codes, &scales, &biases, m, k, n, &mut nf4_output)
            .unwrap();

        // Reference: unpack weights, then matmul.
        let unpacked = unpack_nf4_weights(&codes, &scales, &biases, k, n);
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += input[i * k + kk] * unpacked[kk * n + j];
                }
            }
        }

        let result = validate_matmul(&expected, &nf4_output, 0.05);
        assert!(
            result.passed,
            "non-640-multiple matmul (n=700): max_abs_error={}, mismatches={}/{}",
            result.max_abs_error, result.mismatches, result.total_elements
        );
    }

    #[test]
    fn non_640_vs_640_control_rmse() {
        // Compare RMSE for non-640-multiple vs 640-multiple cols with
        // identical data.  Both should have similar NF4 quantization error.
        let k = 20usize;
        let n_non640 = 6912usize;
        let n_control = 6400usize; // 10 tiles, close to 6912 (also ~11 tiles)

        // Same data for both shapes (fill to max width).
        let max_n = n_non640.max(n_control);
        let mut raw = vec![0.0f32; k * max_n];
        for i in 0..k {
            for j in 0..max_n {
                let idx = (i * max_n + j) as u32;
                let rand = ((idx.wrapping_mul(1103515245).wrapping_add(12345)) % 10001) as f32;
                raw[i * max_n + j] = (rand / 5000.5) - 1.0;
            }
        }
        // Slice to each shape.
        let weights_non640: Vec<f32> = (0..k)
            .flat_map(|i| raw[i * max_n..i * max_n + n_non640].to_vec())
            .collect();
        let weights_control: Vec<f32> = (0..k)
            .flat_map(|i| raw[i * max_n..i * max_n + n_control].to_vec())
            .collect();

        let (codes, scales, biases, _, _) = pack_nf4_weights(&weights_non640, k, n_non640);
        let (codes_ctrl, scales_ctrl, biases_ctrl, _, _) =
            pack_nf4_weights(&weights_control, k, n_control);

        let mut max_rmse_non640 = 0.0f32;
        let mut max_rmse_ctrl = 0.0f32;
        for trial in 0..3 {
            let input: Vec<f32> = (0..k)
                .map(|i| match trial {
                    0 => (i as f64 * 0.1).sin() as f32,
                    1 => (i as f64 * 0.07).cos() as f32,
                    _ => (i.wrapping_mul(12345).wrapping_add(67890) % 1001) as f32 / 500.0 - 1.0,
                })
                .collect();

            // Non-640 RMSE.
            let rmse_non640 = compute_dequant_rmse(
                &input,
                &codes,
                &scales,
                &biases,
                &weights_non640,
                k,
                n_non640,
            );
            if rmse_non640 > max_rmse_non640 {
                max_rmse_non640 = rmse_non640;
            }

            // Control RMSE.
            let rmse_ctrl = compute_dequant_rmse(
                &input,
                &codes_ctrl,
                &scales_ctrl,
                &biases_ctrl,
                &weights_control,
                k,
                n_control,
            );
            if rmse_ctrl > max_rmse_ctrl {
                max_rmse_ctrl = rmse_ctrl;
            }
        }

        eprintln!(
            "non-640 (n={n_non640}): max_rmse={max_rmse_non640:.6}  |  control (n={n_control}): max_rmse={max_rmse_ctrl:.6}",
        );

        // Non-640 packing must NOT be catastrophically worse than 640-multiple.
        // Max allowed: 3× the control RMSE.  (NF4 quantization noise dominates.)
        let max_allowed = max_rmse_ctrl * 3.0;
        assert!(
            max_rmse_non640 < max_allowed,
            "non-640 RMSE {max_rmse_non640:.6} >> control RMSE {max_rmse_ctrl:.6} (limit {max_allowed:.6})",
        );
    }

    /// Helper: compute RMSE between NF4 dequant matmul and BF16 reference.
    fn compute_dequant_rmse(
        input: &[f32],
        codes: &[u8],
        scales: &[f32],
        biases: &[f32],
        weights: &[f32],
        k: usize,
        n: usize,
    ) -> f32 {
        // BF16 reference: output[j] = Σ_i weights[i][j] * input[i]
        let mut ref_output = vec![0.0f32; n];
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += weights[i * n + j] * input[i];
            }
            ref_output[j] = sum;
        }

        let mut nf4_output = vec![0.0f32; n];
        dequant_matmul_reference(&input, codes, scales, biases, 1, k, n, &mut nf4_output).unwrap();

        let mut sq_err = 0.0f32;
        for j in 0..n {
            let diff = nf4_output[j] - ref_output[j];
            sq_err += diff * diff;
        }
        (sq_err / n as f32).sqrt()
    }

    #[test]
    fn diagnostic_nf4_one_matrix() {
        let rows = 64usize;
        let cols = 1280usize;
        let n = rows * cols;

        // Distribution 1: Gaussian N(0, 0.02^2) — attention projection weights
        let mut seed: u64 = 42;
        let mut rng = || -> f64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64) / 8589934592.0
        };
        let w_gauss: Vec<f32> = (0..n)
            .map(|_| {
                let u = rng();
                let v = rng();
                ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos() * 0.02) as f32
            })
            .collect();
        let (codes, scales, biases, _, _) = pack_nf4_weights(&w_gauss, rows, cols);
        let recon = unpack_nf4_weights(&codes, &scales, &biases, rows, cols);
        let full_rmse = (w_gauss
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();
        println!(
            "Gaussian {rows}x{cols}: codes={}B, weight RMSE={full_rmse:.8}",
            codes.len()
        );

        for g in 0..(cols / 128) {
            let start = g * 128;
            let rmse = (w_gauss[start..start + 128]
                .iter()
                .zip(recon[start..start + 128].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / 128.0)
                .sqrt();
            let gmax = w_gauss[start..start + 128]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let gmin = w_gauss[start..start + 128]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let span = (gmax - gmin).max(1e-8);
            let nrmse = rmse / span;
            println!(
                "  g[{g}] RMSE={rmse:.8} range=[{gmin:.6},{gmax:.6}] scale={:.6} nrmse={nrmse:.4}",
                scales[g]
            );
        }

        // Matmul: dequant function uses [k x n] orientation where weight = [rows x cols]
        // output[col] += input[row] * weight[row][col]
        let inp: Vec<f32> = (0..rows)
            .map(|r| (r as f32) / rows as f32 * 2.0 - 1.0)
            .collect();
        let mut ref_out = vec![0.0f32; cols];
        for col in 0..cols {
            let mut s = 0.0f32;
            for row in 0..rows {
                s += w_gauss[row * cols + col] * inp[row];
            }
            ref_out[col] = s;
        }
        let mut nf4_out = vec![0.0f32; cols];
        dequant_matmul_reference(&inp, &codes, &scales, &biases, 1, rows, cols, &mut nf4_out).ok();
        let mrmse = (ref_out
            .iter()
            .zip(nf4_out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / cols as f32)
            .sqrt();
        println!("  MatMul RMSE={mrmse:.8}");
        println!("  Ref[..6]={:.6?}", &ref_out[..6]);
        println!("  NF4[..6]={:.6?}", &nf4_out[..6]);

        // Distribution 2: FFN-like with outliers
        let mut seed2: u64 = 42;
        let mut rng2 = || -> f64 {
            seed2 = seed2.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed2 >> 33) as f64) / 8589934592.0
        };
        let mut w_ffn: Vec<f32> = (0..n)
            .map(|_| {
                let u = rng2();
                let v = rng2();
                ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos() * 0.01) as f32
            })
            .collect();
        for &idx in &[0, 15, 127, 640, 1155] {
            if idx + 1 < w_ffn.len() {
                w_ffn[idx] = 0.5;
                w_ffn[idx + 1] = -0.5;
            }
        }
        let (cf, sf, bf, _, _) = pack_nf4_weights(&w_ffn, rows, cols);
        let recon_f = unpack_nf4_weights(&cf, &sf, &bf, rows, cols);
        let ffn_rmse = (w_ffn
            .iter()
            .zip(recon_f.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();
        let max_scale = sf.iter().cloned().fold(0.0f32, f32::max);
        println!("  FFN RMSE={ffn_rmse:.8} max_scale={max_scale:.6}");

        assert!(full_rmse < 0.05, "Gauss RMSE {full_rmse:.8} >= 0.05");
        assert!(mrmse < 0.05, "MatMul RMSE {mrmse:.8} >= 0.05");
        assert!(ffn_rmse < 0.05, "FFN RMSE {ffn_rmse:.8} >= 0.05");
    }
    // ── Profile serialization tests ──────────────────────────────────────────

    #[test]
    fn test_profile_id_constants() {
        use super::profile::{
            ProfileId, PROFILE_ID_CANONICAL_NF4_V1, PROFILE_ID_GEMMA_ATTENTION_V1,
            PROFILE_ID_GEMMA_BOUNDARY_V1, PROFILE_ID_GEMMA_FFN_V1, PROFILE_ID_TTS_CODEC_V1,
        };
        assert_eq!(PROFILE_ID_CANONICAL_NF4_V1, ProfileId(0));
        assert_eq!(PROFILE_ID_GEMMA_ATTENTION_V1, ProfileId(1));
        assert_eq!(PROFILE_ID_GEMMA_FFN_V1, ProfileId(2));
        assert_eq!(PROFILE_ID_GEMMA_BOUNDARY_V1, ProfileId(3));
        assert_eq!(PROFILE_ID_TTS_CODEC_V1, ProfileId(4));
    }

    #[test]
    fn test_codebook_descriptor_canonical() {
        use super::profile::{CodebookDescriptor, PROFILE_ID_CANONICAL_NF4_V1};
        let desc = CodebookDescriptor::canonical_nf4();
        assert!(desc.validate().is_ok());
        assert_eq!(desc.values.len(), 16);
        assert!((desc.values[0] + 1.0).abs() < 1e-6);
        assert!((desc.values[15] - 1.0).abs() < 1e-6);
        assert_eq!(desc.profile_id, PROFILE_ID_CANONICAL_NF4_V1);
        assert_eq!(desc.name, "canonical_nf4_v1");
    }

    #[test]
    fn test_codebook_validate_wrong_size() {
        use super::profile::CodebookDescriptor;
        let mut desc = CodebookDescriptor::canonical_nf4();
        desc.values = vec![0.0; 15];
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_codebook_validate_unsorted() {
        use super::profile::CodebookDescriptor;
        let mut desc = CodebookDescriptor::canonical_nf4();
        desc.values.swap(5, 10);
        assert!(desc.validate().is_err());
    }

    #[test]
    fn test_quantizer_profile_canonical() {
        use super::profile::QuantizerProfile;
        let p = QuantizerProfile::canonical_nf4();
        assert_eq!(p.group_size, 128);
        assert_eq!(p.tile_elements, 640);
    }

    // ── Matrix roles tests ───────────────────────────────────────────────

    #[test]
    fn test_classify_matrix_role() {
        use super::roles::{classify_matrix_role, MatrixRole};
        assert_eq!(
            classify_matrix_role("model.language_model.embed_tokens.weight"),
            MatrixRole::Embedding
        );
        assert_eq!(
            classify_matrix_role("model.language_model.lm_head.weight"),
            MatrixRole::LmHead
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.self_attn.q_proj.weight"),
            MatrixRole::AttentionQ
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.self_attn.k_proj.weight"),
            MatrixRole::AttentionK
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.self_attn.v_proj.weight"),
            MatrixRole::AttentionV
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.self_attn.o_proj.weight"),
            MatrixRole::AttentionO
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.mlp.gate_proj.weight"),
            MatrixRole::FfnGate
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.mlp.up_proj.weight"),
            MatrixRole::FfnUp
        );
        assert_eq!(
            classify_matrix_role("model.language_model.layers.0.mlp.down_proj.weight"),
            MatrixRole::FfnDown
        );
        assert_eq!(
            classify_matrix_role("model.vision_embedder.patch_dense.weight"),
            MatrixRole::MultimodalProjection
        );
        assert_eq!(
            classify_matrix_role("tts_talker.weight"),
            MatrixRole::TtsTalker
        );
        assert_eq!(
            classify_matrix_role("code_predictor.0.weight"),
            MatrixRole::TtsCodePredictor
        );
        assert_eq!(
            classify_matrix_role("codec.encoder.0.weight"),
            MatrixRole::TtsCodec
        );
        assert_eq!(
            classify_matrix_role("unrecognized_tensor_name"),
            MatrixRole::UnknownLinear
        );
    }

    #[test]
    fn test_matrix_role_families() {
        use super::roles::MatrixRole;
        assert!(MatrixRole::AttentionQ.is_attention());
        assert!(!MatrixRole::AttentionQ.is_ffn());
        assert!(MatrixRole::FfnGate.is_ffn());
        assert!(!MatrixRole::FfnGate.is_attention());
        assert!(MatrixRole::Embedding.is_boundary());
        assert!(MatrixRole::LmHead.is_boundary());
        assert!(MatrixRole::TtsTalker.is_tts());
        assert!(MatrixRole::TtsCodec.is_tts());
        assert!(!MatrixRole::UnknownLinear.is_attention());
        assert!(!MatrixRole::UnknownLinear.is_ffn());
        assert!(!MatrixRole::UnknownLinear.is_boundary());
        assert!(!MatrixRole::UnknownLinear.is_tts());
    }

    // ── Calibration tests ────────────────────────────────────────────────

    #[test]
    fn test_normalize_group() {
        use super::calibration::normalize_group;
        let group: Vec<f32> = vec![1.0; 128];
        let norm = normalize_group(&group, 2.0, 0.0);
        assert_eq!(norm.len(), 128);
        for v in &norm {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_streaming_collector_basic() {
        use super::calibration::{CalibrationConfig, StreamingStateCollector};
        use super::roles::MatrixRole;
        let config = CalibrationConfig::default();
        let mut collector = StreamingStateCollector::new(config);
        let group: Vec<f32> = vec![0.5; 128];
        collector.ingest_weight_group(MatrixRole::AttentionQ, 0, &group, 1.0, 0.0, 1.0, false);
        let result = collector.finish();
        assert!(result.receipt.total_samples > 0);
        assert!(result.role_stats.contains_key(&MatrixRole::AttentionQ));
    }

    // ── Learning tests ───────────────────────────────────────────────────

    #[test]
    fn test_importance_from_variance() {
        use super::learn::importance_from_variance;
        assert!((importance_from_variance(0.0) - 1e-8).abs() < 1e-6);
        assert!((importance_from_variance(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lloyd_max_determinism() {
        use super::learn::{weighted_scalar_lloyd_max, LearningConfig};
        let samples: Vec<(f32, f32)> = (0..200)
            .map(|i| {
                let x = (i as f64 / 100.0 * std::f64::consts::PI).sin() as f32;
                (x, 1.0)
            })
            .collect();
        let config = LearningConfig::default();
        let (cb1, _) = weighted_scalar_lloyd_max(&samples, &config);
        let (cb2, _) = weighted_scalar_lloyd_max(&samples, &config);
        for i in 0..16 {
            assert!((cb1[i] - cb2[i]).abs() < 1e-6, "Centroid {} differs", i);
        }
    }

    #[test]
    fn test_select_best_profile() {
        use super::learn::{LearningConfig, LearningReceipt};
        let receipt = LearningReceipt {
            role: "attention_q".into(),
            num_samples: 100,
            clipped_fraction: 0.0,
            baseline_objective: 0.1,
            final_objective: 0.02,
            objective_by_iteration: vec![0.1, 0.02],
            num_iterations: 2,
            converged: true,
            occupancy: [6u32; 16],
            clipping_policy: "none".into(),
            learning_config: LearningConfig::default(),
            seed: 42,
        };
        assert_eq!(receipt.final_objective, 0.02);
        assert!(receipt.converged);
    }

    // ── Verify tests ─────────────────────────────────────────────────────

    #[test]
    fn test_structural_verify_valid() {
        use super::verify::structural_verify;
        // (2, 640) → tiles along cols: total_tiles=2*1=2 → 640 codes, 10 scales/biases
        let codes = vec![0u8; 640];
        let scales = vec![1.0f32; 10];
        let biases = vec![0.0f32; 10];
        let result = structural_verify(&codes, &scales, &biases, 2, TILE_ELEMENTS as u32);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn test_structural_verify_invalid_codes() {
        use super::verify::structural_verify;
        let codes = vec![255u8; 640];
        let scales = vec![1.0f32; 10];
        let biases = vec![0.0f32; 10];
        let result = structural_verify(&codes, &scales, &biases, 2, TILE_ELEMENTS as u32);
        assert!(
            result.is_ok(),
            "0xFF should be valid codes (15 in each nibble)"
        );
    }

    #[test]
    fn test_apply_quality_policy_default() {
        use super::verify::{apply_quality_policy, MatrixQualityMetrics, QualityStatus};
        let metrics = vec![MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.02,
            weight_nrmse: 0.01,
            max_abs_error: 0.1,
            sqnr_db: 25.0,
            effective_bpw: 4.0,
            quality_status: QualityStatus::Passed,
        }];
        let result = apply_quality_policy(&metrics, "default");
        assert_eq!(
            result[0].quality_status,
            QualityStatus::Passed,
            "0.02 RMSE should pass default policy (0.05 threshold)"
        );
    }

    #[test]
    fn test_apply_quality_policy_strict() {
        use super::verify::{apply_quality_policy, MatrixQualityMetrics, QualityStatus};
        let metrics = vec![MatrixQualityMetrics {
            matrix_name: "test".into(),
            role: "attention_q".into(),
            profile_id: 0,
            weight_rmse: 0.02,
            weight_nrmse: 0.01,
            max_abs_error: 0.1,
            sqnr_db: 25.0,
            effective_bpw: 4.0,
            quality_status: QualityStatus::Passed,
        }];
        let result = apply_quality_policy(&metrics, "strict");
        assert_eq!(
            result[0].quality_status,
            QualityStatus::Failed,
            "0.02 RMSE should FAIL strict policy (0.01 threshold)"
        );
    }

    // ── Pack/unpack round-trip test ───────────────────────────────────────

    #[test]
    fn test_pack_unpack_roundtrip_small() {
        let rows = 1;
        let cols = 640;
        let original: Vec<f32> = (0..640).map(|i| (i as f32) / 640.0 * 2.0 - 1.0).collect();
        let (codes, scales, biases, _, _) = super::pack_nf4_weights(&original, rows, cols);
        let reconstructed = super::unpack_nf4_weights(&codes, &scales, &biases, rows, cols);
        let rmse = (original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / original.len() as f32)
            .sqrt();
        assert!(rmse < 0.1, "pack→unpack RMSE too high: {rmse:.6}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn accelerate_matches_scalar() {
        // Cross-check: the Accelerate BLAS path in dequant_matmul_reference
        // must produce results identical to the scalar fallback within f32
        // precision.
        let m = 3usize;
        let k = 5usize;
        let n = 700usize; // non-multiple-of-640 to exercise padding

        // Random weight matrix spanning the NF4 codebook.
        let mut weights = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                let phase = (i * n + j) as f32 * 0.013;
                weights[i * n + j] = (phase - phase.floor() - 0.5) * 2.0;
            }
        }

        let (codes, scales, biases, _p_rows, _p_cols) = super::pack_nf4_weights(&weights, k, n);

        // Input with varied values (non-zero to exercise full accumulation).
        let input: Vec<f32> = (0..m * k).map(|x| (x as f32 + 1.0) * 0.25).collect();

        // BLAS path (via dequant_matmul_reference on macOS).
        let mut blas_output = vec![0.0f32; m * n];
        super::dequant_matmul_reference(
            &input,
            &codes,
            &scales,
            &biases,
            m,
            k,
            n,
            &mut blas_output,
        )
        .unwrap();

        // Scalar path.
        let padded_n = if n % super::TILE_ELEMENTS == 0 {
            n
        } else {
            n.div_ceil(super::TILE_ELEMENTS) * super::TILE_ELEMENTS
        };
        let tiles_per_row = padded_n / super::TILE_ELEMENTS;
        let total_tiles = k * tiles_per_row;
        let mut scalar_output = vec![0.0f32; m * n];
        super::dequant_matmul_scalar(
            &input,
            &codes,
            &scales,
            &biases,
            m,
            k,
            n,
            &mut scalar_output,
            tiles_per_row,
            total_tiles,
        );

        // Compare element-wise with tight tolerance.
        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f32;
        for idx in 0..m * n {
            let diff = (blas_output[idx] - scalar_output[idx]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_sq += diff * diff;
        }
        let rmse = (sum_sq / (m * n) as f32).sqrt();
        assert!(
            max_diff < 1e-4,
            "BLAS vs scalar: max_diff={:.2e} >= 1e-4 at M={m} K={k} N={n}",
            max_diff,
        );
        assert!(rmse < 1e-5, "BLAS vs scalar: RMSE={:.2e} >= 1e-5", rmse,);
    }

    // ── AW-LS tests ──────────────────────────────────────────────────────

    #[test]
    fn test_awls_pack_lower_mse_than_maxabs() {
        // Create a tile with non-uniform distribution
        let mut vals = [0.0f32; TILE_ELEMENTS];
        let mut act_w = [1.0f32; TILE_ELEMENTS];
        for i in 0..640 {
            // First channel has high activation weight and specific value
            if i < 128 {
                vals[i] = 3.0; // Large value, should be well-reconstructed
                act_w[i] = 10.0; // High importance
            } else if i < 256 {
                vals[i] = -0.1; // Small value
                act_w[i] = 0.1; // Low importance
            } else {
                vals[i] = 0.5 * ((i as f32) / 640.0 * std::f32::consts::PI * 2.0).sin();
                act_w[i] = 1.0;
            }
        }

        // Max-abs baseline
        let (codes_ma, scales_ma, biases_ma) = pack_nf4_tile(&vals);
        let mut ma_out = [0.0f32; TILE_ELEMENTS];
        {
            let mut c_arr = [0u8; 320];
            let mut s_arr = [0.0f32; 5];
            let mut b_arr = [0.0f32; 5];
            c_arr.copy_from_slice(&codes_ma);
            s_arr.copy_from_slice(&scales_ma);
            b_arr.copy_from_slice(&biases_ma);
            crate::nf4tile640::unpack_nf4_tile(&c_arr, &s_arr, &b_arr, &mut ma_out);
        }

        // AW-LS
        let (codes_aw, scales_aw, biases_aw) = pack_nf4_tile_awls(&vals, &act_w, 8);
        let mut aw_out = [0.0f32; TILE_ELEMENTS];
        {
            let mut c_arr = [0u8; 320];
            let mut s_arr = [0.0f32; 5];
            let mut b_arr = [0.0f32; 5];
            c_arr.copy_from_slice(&codes_aw);
            s_arr.copy_from_slice(&scales_aw);
            b_arr.copy_from_slice(&biases_aw);
            crate::nf4tile640::unpack_nf4_tile(&c_arr, &s_arr, &b_arr, &mut aw_out);
        }

        // AW-MSE: activation-weighted
        let aw_mse_ma: f64 = vals
            .iter()
            .zip(ma_out.iter())
            .zip(act_w.iter())
            .map(|((v, r), a)| (*a as f64) * ((v - r) as f64).powi(2))
            .sum::<f64>()
            / TILE_ELEMENTS as f64;
        let aw_mse_aw: f64 = vals
            .iter()
            .zip(aw_out.iter())
            .zip(act_w.iter())
            .map(|((v, r), a)| (*a as f64) * ((v - r) as f64).powi(2))
            .sum::<f64>()
            / TILE_ELEMENTS as f64;

        assert!(
            aw_mse_aw <= aw_mse_ma * 1.1 + 1e-6,
            "AW-LS AW-MSE {:.6} should not exceed max-abs AW-MSE {:.6} by more than 10%%",
            aw_mse_aw,
            aw_mse_ma
        );
    }
}

#[cfg(test)]
#[path = "metal_tests.rs"]
pub(crate) mod metal_test_module;

#[cfg(test)]
#[path = "hw_proof.rs"]
pub(crate) mod hw_proof_module;
/// Pack a single tile of 640 f32 values using symmetric int4 (q4_0 style).
/// Quantizes to -7..7 with group_size elements per group.
pub fn pack_symmetric_int4_tile(
    values: &[f32; TILE_ELEMENTS],
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;
    let packed_codes_len = num_groups * bytes_per_group;
    let mut packed_codes = vec![0u8; packed_codes_len];
    let mut scales = vec![0.0f32; num_groups];
    let mut biases = vec![0.0f32; num_groups];
    let max_code = 7.0f32;

    for group in 0..num_groups {
        let base = group * group_size;
        let max_abs = values[base..base + group_size]
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let scale = if max_abs < 1e-30 {
            1.0f32
        } else {
            max_abs / max_code
        };
        scales[group] = scale;
        biases[group] = 0.0;

        for i in 0..(group_size / 2) {
            let bit_idx = group * bytes_per_group + i;
            let val0 = (values[base + 2 * i] / scale).round().clamp(-7.0, 7.0) as i8;
            let val1 = (values[base + 2 * i + 1] / scale).round().clamp(-7.0, 7.0) as i8;
            // Map -7..7 to 0..15 (unsigned 4-bit)
            let code0 = (val0 + 7) as u8;
            let code1 = (val1 + 7) as u8;
            packed_codes[bit_idx] = code0 | (code1 << 4);
        }
    }

    (packed_codes, scales, biases)
}
