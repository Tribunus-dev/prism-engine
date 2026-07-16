//! NF4 tile640 packed weight format — local copy for crate-internal use.
//!
//! These types live in `tribunus-compute-core` and will be migrated to
//! `prism-ecs-core` once the extraction dependency chain is resolved.
//!
//! NOTE: This is a minimal copy of the module from compute-core. Only the
//! items referenced by this crate are included. Keep in sync.

use crate::sweep::spec::Nf4CodebookId;

// ════════════════════════════════════════════════════════════════════════════
// Format Constants
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
pub const TILE_ELEMENTS: usize = 640;

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
// NF4 Codebooks
// ════════════════════════════════════════════════════════════════════════════

/// Prism canonical NF4 codebook: 16 evenly-spaced quantiles of N(0,1).
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
    0.25665056705474854,
    0.3479790687561035,
    0.4470846354961395,
    0.5603545904159546,
    0.7071067690849304,
    1.0,
];

/// BitsAndBytes NF4 codebook (from Dettmers et al., 2023).
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
    0.4407098591327667,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Symmetric normal float NF4 codebook.
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

/// Canonical NF4 codebook used for pack/unpack operations.
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
    0.4407098591327667,
    0.5626170039176941,
    0.7229568362236023,
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

/// Quantize a normalized value to the nearest NF4 code index.
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

/// Decode a single 4-bit NF4 code index to its f32 value.
pub fn nf4_dequantize(code: u8) -> f32 {
    assert!(code < 16, "NF4 code index must be 0..15, got {code}");
    NF4_CODEBOOK[code as usize]
}

/// Decode a single 4-bit NF4 code index to its f32 value using the given codebook.
pub fn nf4_dequantize_with_codebook(code: u8, codebook: &[f32; 16]) -> f32 {
    assert!(code < 16, "NF4 code index must be 0..15, got {code}");
    codebook[code as usize]
}

/// Quantize an f32 value to the nearest NF4 codebook entry, returning the 4-bit index.
pub fn nf4_quantize(value: f32) -> u8 {
    if value.is_nan() {
        return 7;
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
// Tile pack/unpack
// ════════════════════════════════════════════════════════════════════════════

/// Pack a single tile of 640 f32 values using configurable group size.
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

    for g in 0..num_groups {
        let base = g * group_size;
        let mut max_abs = 0.0f32;
        for i in 0..group_size {
            let v = values[base + i].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let scale = if max_abs > 1e-12 { max_abs } else { 1.0 };
        let bias = 0.0f32;
        scales[g] = scale;
        biases[g] = bias;

        for i in 0..(group_size / 2) {
            let v0 = values[base + 2 * i] / scale;
            let v1 = values[base + 2 * i + 1] / scale;
            let c0 = nf4_quantize(v0);
            let c1 = nf4_quantize(v1);
            let byte_idx = g * bytes_per_group + i;
            packed_codes[byte_idx] = c0 | (c1 << 4);
        }
    }
    (packed_codes, scales, biases)
}

/// Pack a single tile of 640 f32 values (default group size 128).
pub fn pack_nf4_tile(values: &[f32; TILE_ELEMENTS]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    pack_nf4_tile_with_group_size(values, GROUP_SIZE)
}

/// Pack a weight matrix into NF4 tile640 format.
pub fn pack_nf4_weights(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    use rayon::prelude::*;

    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let _tile_cols = tiles_per_row * TILE_ELEMENTS;
    let total_tiles = rows * tiles_per_row;

    let mut packed_codes = vec![0u8; total_tiles * PACKED_BYTES_PER_TILE];
    let mut scales = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];
    let mut biases = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];

    let results: Vec<(usize, Vec<u8>, Vec<f32>, Vec<f32>)> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut row_codes = Vec::with_capacity(tiles_per_row * PACKED_BYTES_PER_TILE);
            let mut row_scales = Vec::with_capacity(tiles_per_row * SCALES_F32_PER_TILE);
            let mut row_biases = Vec::with_capacity(tiles_per_row * SCALES_F32_PER_TILE);
            for t in 0..tiles_per_row {
                let col_start = t * TILE_ELEMENTS;
                let mut tile = [0.0f32; TILE_ELEMENTS];
                for c in 0..TILE_ELEMENTS {
                    let src_col = col_start + c;
                    tile[c] = if src_col < cols {
                        weights[r * cols + src_col]
                    } else {
                        0.0
                    };
                }
                let (codes, sc, bias) = pack_nf4_tile(&tile);
                row_codes.extend_from_slice(&codes);
                row_scales.extend_from_slice(&sc);
                row_biases.extend_from_slice(&bias);
            }
            (r, row_codes, row_scales, row_biases)
        })
        .collect();

    for (r, codes, sc, bias) in results {
        let _src_start = r * tiles_per_row * PACKED_BYTES_PER_TILE;
        let dst_start = r * tiles_per_row * PACKED_BYTES_PER_TILE;
        packed_codes[dst_start..dst_start + codes.len()].copy_from_slice(&codes);
        scales[r * tiles_per_row * SCALES_F32_PER_TILE..][..sc.len()].copy_from_slice(&sc);
        biases[r * tiles_per_row * SCALES_F32_PER_TILE..][..bias.len()].copy_from_slice(&bias);
    }

    (packed_codes, scales, biases, rows as u32, cols as u32)
}

/// Pack NF4 weights with activation-weighted loss scaling (AWLS).
pub fn pack_nf4_weights_awls(
    weights: &[f32],
    rows: usize,
    cols: usize,
    _activation_weights: Option<&[f32]>,
    _max_iters: u8,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, u32, u32) {
    // AWLS not yet migrated — fall back to standard packing
    pack_nf4_weights(weights, rows, cols)
}

// ════════════════════════════════════════════════════════════════════════════
// Unpack
// ════════════════════════════════════════════════════════════════════════════

/// Unpack multiple tiles from NF4 format back to f32.
pub fn unpack_nf4_weights(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let tile_cols = tiles_per_row * TILE_ELEMENTS;
    let mut result = vec![0.0f32; rows * tile_cols];

    for t in 0..total_tiles {
        let r = t / tiles_per_row;
        let tc = t % tiles_per_row;
        let code_base = t * PACKED_BYTES_PER_TILE;
        let meta_base = t * SCALES_F32_PER_TILE;

        for g in 0..GROUPS_PER_TILE {
            let scale = scales[meta_base + g];
            let bias = biases[meta_base + g];
            let group_code_base = code_base + g * PACKED_BYTES_PER_GROUP;
            let out_base = r * tile_cols + tc * TILE_ELEMENTS + g * GROUP_SIZE;

            for i in 0..(GROUP_SIZE / 2) {
                let byte = packed_codes[group_code_base + i];
                let idx0 = byte & 0x0F;
                let idx1 = byte >> 4;
                result[out_base + 2 * i] =
                    nf4_dequantize_with_codebook(idx0, &NF4_CODEBOOK) * scale + bias;
                result[out_base + 2 * i + 1] =
                    nf4_dequantize_with_codebook(idx1, &NF4_CODEBOOK) * scale + bias;
            }
        }
    }

    if tile_cols > cols {
        result.truncate(rows * cols);
    }
    result
}

/// Unpack NF4 weights with configurable group size.
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
    let bytes_per_group = group_size / 2;
    let packed_per_tile = groups_per_tile * bytes_per_group;
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let tile_cols = tiles_per_row * TILE_ELEMENTS;
    let mut result = vec![0.0f32; rows * tile_cols];

    for t in 0..total_tiles {
        let r = t / tiles_per_row;
        let tc = t % tiles_per_row;
        let code_base = t * packed_per_tile;
        let meta_base = t * groups_per_tile;

        for g in 0..groups_per_tile {
            let scale = scales[meta_base + g];
            let bias = biases[meta_base + g];
            let gcode = code_base + g * bytes_per_group;
            let out_base = r * tile_cols + tc * TILE_ELEMENTS + g * group_size;

            for i in 0..(group_size / 2) {
                let byte = packed_codes[gcode + i];
                let idx0 = byte & 0x0F;
                let idx1 = byte >> 4;
                result[out_base + 2 * i] = NF4_CODEBOOK[idx0 as usize] * scale + bias;
                result[out_base + 2 * i + 1] = NF4_CODEBOOK[idx1 as usize] * scale + bias;
            }
        }
    }

    if tile_cols > cols {
        result.truncate(rows * cols);
    }
    result
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
    let packed_per_tile = groups_per_tile * bytes_per_group;
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let tile_cols = tiles_per_row * TILE_ELEMENTS;
    let mut result = vec![0.0f32; rows * tile_cols];

    for t in 0..total_tiles {
        let r = t / tiles_per_row;
        let tc = t % tiles_per_row;
        let code_base = t * packed_per_tile;
        let meta_base = t * groups_per_tile;

        for g in 0..groups_per_tile {
            let scale = scales[meta_base + g];
            let bias = biases[meta_base + g];
            let gcode = code_base + g * bytes_per_group;
            let out_base = r * tile_cols + tc * TILE_ELEMENTS + g * group_size;

            for i in 0..(group_size / 2) {
                let byte = packed_codes[gcode + i];
                let idx0 = byte & 0x0F;
                let idx1 = byte >> 4;
                result[out_base + 2 * i] = codebook[idx0 as usize] * scale + bias;
                result[out_base + 2 * i + 1] = codebook[idx1 as usize] * scale + bias;
            }
        }
    }

    if tile_cols > cols {
        result.truncate(rows * cols);
    }
    result
}

// ════════════════════════════════════════════════════════════════════════════
// INT8 tile640 pack/unpack
// ════════════════════════════════════════════════════════════════════════════

/// Pack a weight matrix into INT8 tile640 format.
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

    let results: Vec<(usize, Vec<u8>, Vec<f32>)> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let mut row_codes = Vec::with_capacity(tile_cols);
            let mut row_scales = Vec::with_capacity(num_tiles);
            for t in 0..num_tiles {
                let col_start = t * TILE_ELEMENTS;
                let mut max_abs = 0.0f32;
                for c in 0..TILE_ELEMENTS {
                    let src_col = col_start + c;
                    let v = if src_col < cols {
                        weights[r * cols + src_col]
                    } else {
                        0.0
                    };
                    let abs_v = v.abs();
                    if abs_v > max_abs {
                        max_abs = abs_v;
                    }
                }
                let scale = if max_abs > 1e-12 {
                    max_abs / 127.0
                } else {
                    1.0
                };
                row_scales.push(scale);
                for c in 0..TILE_ELEMENTS {
                    let src_col = col_start + c;
                    let v = if src_col < cols {
                        weights[r * cols + src_col]
                    } else {
                        0.0
                    };
                    let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
                    row_codes.push(q as u8);
                }
            }
            (r, row_codes, row_scales)
        })
        .collect();

    for (r, row_codes, row_scales) in results {
        codes[r * tile_cols..(r + 1) * tile_cols].copy_from_slice(&row_codes);
        scales.extend_from_slice(&row_scales);
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

    for r in 0..rows {
        for t in 0..num_tiles {
            let scale = scales[r * num_tiles + t];
            let bias = biases[r * num_tiles + t];
            let code_base = r * tile_cols + t * TILE_ELEMENTS;
            let out_base = r * cols + t * TILE_ELEMENTS;
            let limit = TILE_ELEMENTS.min(cols - t * TILE_ELEMENTS);

            for c in 0..limit {
                result[out_base + c] = (codes[code_base + c] as i8) as f32 * scale + bias;
            }
        }
    }
    result
}

/// Unpack INT8 weights with configurable group size.
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
    let mut result = vec![0.0f32; rows * tile_cols];

    for r in 0..rows {
        for t in 0..num_tiles {
            for g in 0..groups_per_tile {
                let scale_idx = r * num_tiles * groups_per_tile + t * groups_per_tile + g;
                let scale = scales[scale_idx];
                let bias = biases[scale_idx];
                let code_base = r * tile_cols + t * TILE_ELEMENTS + g * group_size;
                let out_base = r * tile_cols + t * TILE_ELEMENTS + g * group_size;

                for c in 0..group_size {
                    let code = codes[code_base + c];
                    result[out_base + c] = (code as i8) as f32 * scale + bias;
                }
            }
        }
    }

    if tile_cols > cols {
        result.truncate(rows * cols);
    }
    result
}

// ════════════════════════════════════════════════════════════════════════════
// Dequant matmul reference
// ════════════════════════════════════════════════════════════════════════════

/// Compute reference dequant + matmul for NF4 tile640 weights.
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
    let tiles_per_col = n.div_ceil(TILE_ELEMENTS);
    let total_tiles = k * tiles_per_col;

    if packed_codes.len() < total_tiles * PACKED_BYTES_PER_TILE {
        return Err(format!(
            "packed_codes too short: {} < {}",
            packed_codes.len(),
            total_tiles * PACKED_BYTES_PER_TILE
        ));
    }
    if output.len() < m * n {
        return Err(format!("output too short: {} < {}", output.len(), m * n));
    }

    let weights = unpack_nf4_weights(packed_codes, scales, biases, k, n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += input[i * k + kk] * weights[kk * n + j];
            }
            output[i * n + j] = sum;
        }
    }
    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// Accelerate helpers module
// ════════════════════════════════════════════════════════════════════════════

pub mod accelerate {
    /// Squared Euclidean distance between two f32 slices.
    /// Uses Accelerate vDSP on macOS, pure-Rust fallback elsewhere.
    #[cfg(target_os = "macos")]
    pub fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut diff = vec![0.0f32; a.len()];
        unsafe {
            extern "C" {
                fn vDSP_vsub(
                    a: *const f32,
                    a_stride: i32,
                    b: *const f32,
                    b_stride: i32,
                    result: *mut f32,
                    result_stride: i32,
                    n: i32,
                );
                fn vDSP_svesq(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
            }
            vDSP_vsub(
                a.as_ptr(),
                1,
                b.as_ptr(),
                1,
                diff.as_mut_ptr(),
                1,
                a.len() as i32,
            );
            let mut sum_sq = 0.0f32;
            vDSP_svesq(diff.as_ptr(), 1, &mut sum_sq, a.len() as i32);
            sum_sq
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Maximum absolute error between two f32 slices.
    #[cfg(target_os = "macos")]
    pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len() as i32;
        let mut diff = vec![0.0f32; a.len()];
        unsafe {
            extern "C" {
                fn vDSP_vsub(
                    a: *const f32,
                    a_stride: i32,
                    b: *const f32,
                    b_stride: i32,
                    result: *mut f32,
                    result_stride: i32,
                    n: i32,
                );
                fn vDSP_maxmgv(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
            }
            vDSP_vsub(a.as_ptr(), 1, b.as_ptr(), 1, diff.as_mut_ptr(), 1, n);
            let mut max_val = 0.0f32;
            vDSP_maxmgv(diff.as_ptr(), 1, &mut max_val, n);
            max_val
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }
}

/// Pack a 640-element tile using symmetric int4 (q4_0 style).
pub fn pack_symmetric_int4_tile(
    values: &[f32; TILE_ELEMENTS],
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let num_groups = TILE_ELEMENTS / group_size;
    let bytes_per_group = group_size / 2;
    let packed_codes_len = num_groups * bytes_per_group;
    let mut packed_codes = vec![0u8; packed_codes_len];
    let mut scales = vec![0.0f32; num_groups];
    let biases = vec![0.0f32; num_groups];
    let max_code = 7.0f32;

    for g in 0..num_groups {
        let base = g * group_size;
        let mut max_abs = 0.0f32;
        for i in 0..group_size {
            let v = values[base + i].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let scale = if max_abs > 1e-12 {
            max_abs / max_code
        } else {
            1.0
        };
        scales[g] = scale;

        for i in 0..(group_size / 2) {
            let v0 = (values[base + 2 * i] / scale)
                .round()
                .clamp(-max_code, max_code) as i8;
            let v1 = (values[base + 2 * i + 1] / scale)
                .round()
                .clamp(-max_code, max_code) as i8;
            let c0 = (v0 as u8) & 0x0F;
            let c1 = ((v1 as u8) & 0x0F) << 4;
            packed_codes[g * bytes_per_group + i] = c0 | c1;
        }
    }

    (packed_codes, scales, biases)
}

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
            required_kernel: "prism.nf4tile640.dequant_mul.v1".into(),
        }
    }
}

/// Compute the expected packed byte sizes for a weight matrix.
pub fn packed_size(rows: usize, cols: usize) -> usize {
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
// Validation oracle
// ════════════════════════════════════════════════════════════════════════════

/// Result of comparing two f32 matrices.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub mismatches: usize,
    pub total_elements: usize,
    pub passed: bool,
}

/// Validate a matrix against a reference with given tolerance.
pub fn validate_matmul(reference: &[f32], candidate: &[f32], tolerance: f32) -> ValidationResult {
    assert_eq!(reference.len(), candidate.len());
    let total_elements = reference.len();
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut mismatches = 0usize;

    for (r, c) in reference.iter().zip(candidate.iter()) {
        let abs_err = (r - c).abs();
        let threshold = tolerance * r.abs().max(c.abs()).max(1e-10);
        sum_abs_error += abs_err;
        if abs_err > max_abs_error {
            max_abs_error = abs_err;
        }
        if abs_err > threshold {
            mismatches += 1;
        }
    }

    let mean_abs_error = sum_abs_error / total_elements as f32;
    ValidationResult {
        max_abs_error,
        mean_abs_error,
        mismatches,
        total_elements,
        passed: mismatches == 0,
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
