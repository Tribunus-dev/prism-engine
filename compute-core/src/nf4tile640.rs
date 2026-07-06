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
pub const TILE_ELEMENTS: usize = 640;

/// Bytes per u32 word.
pub const PACKED_WORD_BYTES: usize = 4;

/// Bytes of packed codes per group (128 / 2).
pub const PACKED_BYTES_PER_GROUP: usize = GROUP_SIZE / 2; // 64

/// Bytes of packed codes per tile (640 / 2).
pub const PACKED_BYTES_PER_TILE: usize = TILE_ELEMENTS / 2; // 320

/// Number of f32 scale values per tile (one per group).
pub const SCALES_F32_PER_TILE: usize = GROUPS_PER_TILE; // 5

// ════════════════════════════════════════════════════════════════════════════
// Section 2: NF4 Codebook
// ════════════════════════════════════════════════════════════════════════════

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
    let mut packed_codes = vec![0u8; PACKED_BYTES_PER_TILE];
    let mut scales = vec![0.0f32; SCALES_F32_PER_TILE];
    let mut biases = vec![0.0f32; SCALES_F32_PER_TILE];

    for group in 0..GROUPS_PER_TILE {
        let base = group * GROUP_SIZE;
        let group_slice = &values[base..base + GROUP_SIZE];

        // Compute scale: max absolute value in the group.
        let max_abs = group_slice
            .iter()
            .map(|v| v.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let scale = if max_abs < 1e-30 {
            // All zeros: use a scale of 1.0 and code 7 (0.0) for every element.
            1.0f32
        } else {
            max_abs
        };

        scales[group] = scale;
        biases[group] = 0.0; // NF4 bias is always zero

        // Pack codes: 2 per byte (low nibble = even element, high = odd+1).
        let codes_base = group * PACKED_BYTES_PER_GROUP;
        for i in 0..(GROUP_SIZE / 2) {
            let val0 = group_slice[2 * i] / scale;
            let val1 = group_slice[2 * i + 1] / scale;
            let code0 = nf4_quantize(val0);
            let code1 = nf4_quantize(val1);
            packed_codes[codes_base + i] = code0 | (code1 << 4);
        }
    }

    (packed_codes, scales, biases)
}

/// Pack multiple tiles from a flat f32 array (shapes are M×K layout).
/// Each contiguous K-length row is split into (K / TILE_ELEMENTS) tiles.
///
/// Returns `(packed_codes, scales, biases)` with data for all tiles stored
/// contiguously per buffer (tile-major order).
///
/// # Panics
///
/// Panics if `weights.len()` does not equal `rows * cols` or if `cols` is not
/// a multiple of `TILE_ELEMENTS`.
pub fn pack_nf4_weights(weights: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    assert_eq!(
        weights.len(),
        rows * cols,
        "weights length {} must equal rows {} × cols {}",
        weights.len(),
        rows,
        cols
    );
    assert!(
        cols % TILE_ELEMENTS == 0,
        "cols {cols} must be a multiple of tile size {TILE_ELEMENTS}"
    );

    let tiles_per_row = cols / TILE_ELEMENTS;
    let total_tiles = rows * tiles_per_row;

    let mut packed_codes = vec![0u8; total_tiles * PACKED_BYTES_PER_TILE];
    let mut scales = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];
    let mut biases = vec![0.0f32; total_tiles * SCALES_F32_PER_TILE];

    for row in 0..rows {
        for tile_in_row in 0..tiles_per_row {
            let values_base = row * cols + tile_in_row * TILE_ELEMENTS;
            let values: &[f32; TILE_ELEMENTS] = weights[values_base..values_base + TILE_ELEMENTS]
                .try_into()
                .expect("tile slice fits exactly");

            let tile_idx = row * tiles_per_row + tile_in_row;
            let (codes_tile, scale_tile, bias_tile) = pack_nf4_tile(values);

            let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
            packed_codes[codes_off..codes_off + PACKED_BYTES_PER_TILE]
                .copy_from_slice(&codes_tile);

            let scale_off = tile_idx * SCALES_F32_PER_TILE;
            scales[scale_off..scale_off + SCALES_F32_PER_TILE].copy_from_slice(&scale_tile);

            let bias_off = tile_idx * SCALES_F32_PER_TILE;
            biases[bias_off..bias_off + SCALES_F32_PER_TILE].copy_from_slice(&bias_tile);
        }
    }

    (packed_codes, scales, biases)
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
/// Panics if the buffer lengths do not match `rows * tiles_per_row` tiles.
pub fn unpack_nf4_weights(
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    assert!(
        cols % TILE_ELEMENTS == 0,
        "cols {cols} must be a multiple of tile size {TILE_ELEMENTS}"
    );

    let tiles_per_row = cols / TILE_ELEMENTS;
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
        let out_base = row * cols + tile_in_row * TILE_ELEMENTS;

        let mut tile_out = [0.0f32; TILE_ELEMENTS];
        unpack_nf4_tile(codes_slice, scale_slice, bias_slice, &mut tile_out);
        output[out_base..out_base + TILE_ELEMENTS].copy_from_slice(&tile_out);
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
    pub tiles_per_row: u32,
    pub total_tiles: u32,
    pub required_kernel: String,
}

impl Nf4Tile640Manifest {
    /// Build a manifest for the given logical shape.
    pub fn new(rows: u32, cols: u32) -> Self {
        let tiles_per_row = cols.div_ceil(TILE_ELEMENTS as u32);
        let total_tiles = rows * tiles_per_row;
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
            tiles_per_row,
            total_tiles,
            required_kernel: "dequant_mul_nf4tile640".to_string(),
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Section 7: Fused dequantize + matmul (CPU reference oracle)
// ════════════════════════════════════════════════════════════════════════════

/// Compute the expected packed byte sizes for a weight matrix of shape `[rows, cols]`.
///
/// This is the size of the packed_codes buffer (contiguous bytes of
/// 4-bit NF4 indices, 8 codes per u32, LE byte order).
pub fn packed_size(rows: usize, cols: usize) -> usize {
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    total_tiles * PACKED_BYTES_PER_TILE
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
    let tiles_per_row = n.div_ceil(TILE_ELEMENTS);
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

    // Zero the output buffer.
    output.fill(0.0);

    // Iterate tiles in packed order.
    for tile_idx in 0..total_tiles {
        let kr = tile_idx / tiles_per_row;
        let tile_in_row = tile_idx % tiles_per_row;
        let col_base = tile_in_row * TILE_ELEMENTS;

        let codes_off = tile_idx * PACKED_BYTES_PER_TILE;
        let scale_off = tile_idx * SCALES_F32_PER_TILE;

        // Extract slices for this tile.
        let codes_slice: &[u8; PACKED_BYTES_PER_TILE] = packed_codes
            [codes_off..codes_off + PACKED_BYTES_PER_TILE]
            .try_into()
            .expect("codes slice fits exactly");
        let scale_slice: &[f32; SCALES_F32_PER_TILE] =
            scales[scale_off..scale_off + SCALES_F32_PER_TILE]
                .try_into()
                .expect("scale slice fits exactly");
        let bias_slice: &[f32; SCALES_F32_PER_TILE] =
            biases[scale_off..scale_off + SCALES_F32_PER_TILE]
                .try_into()
                .expect("bias slice fits exactly");

        // Dequantize this tile once.
        let mut tile_f32 = [0.0f32; TILE_ELEMENTS];
        unpack_nf4_tile(codes_slice, scale_slice, bias_slice, &mut tile_f32);

        // Accumulate contribution across all M rows of the output.
        for row_m in 0..m {
            let x_val = input[row_m * k + kr];
            if x_val == 0.0 {
                continue;
            }
            let out_base = row_m * n + col_base;
            let limit = TILE_ELEMENTS.min(n - col_base);
            for j in 0..limit {
                output[out_base + j] += x_val * tile_f32[j];
            }
        }
    }

    Ok(())
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
        // A small 640×640 identity packed then unpacked.  Since TILE_ELEMENTS = 640,
        // this is one row with one tile of an identity matrix -> 1.0 on diagonal,
        // zeros elsewhere.
        let rows = 1usize;
        let cols = TILE_ELEMENTS;
        let mut identity = vec![0.0f32; rows * cols];
        identity[0] = 1.0;

        let (packed_codes, scales, biases) = pack_nf4_weights(&identity, rows, cols);

        let tile_count = 1;
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
        // We use n = TILE_ELEMENTS (pack_nf4_weights requires cols % TILE_ELEMENTS == 0)
        // but only the first 3 columns per row are non-zero.
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

        let (packed_codes, scales, biases) = pack_nf4_weights(&weights, k, n);

        let mut output = vec![0.0f32; m * n];
        dequant_matmul_reference(
            &input, &packed_codes, &scales, &biases, m, k, n, &mut output,
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

        let (packed_codes, scales, biases) = pack_nf4_weights(&weights, k, n);

        // Input: [m, k] with distinct values.
        let input: Vec<f32> = (0..m * k).map(|x| (x as f32 + 1.0) * 0.1).collect();

        let mut output = vec![0.0f32; m * n];
        dequant_matmul_reference(
            &input, &packed_codes, &scales, &biases, m, k, n, &mut output,
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
        assert!(
            dequant_matmul_reference(&input, &codes, &scales, &biases, 3, 1, TILE_ELEMENTS, &mut output)
                .is_err()
        );
        // Wrong output length: m=2, n=TILE_ELEMENTS needs 1280, got 6.
        assert!(
            dequant_matmul_reference(&input, &codes, &scales, &biases, 2, 2, TILE_ELEMENTS, &mut output)
                .is_err()
        );
        // Wrong packed codes size: need 1 tile = 320 codes bytes, got 100.
        let mut big_output = vec![0.0f32; 2 * TILE_ELEMENTS];
        assert!(
            dequant_matmul_reference(
                &input,
                &[0u8; 100],
                &scales,
                &biases,
                2,
                2,
                TILE_ELEMENTS,
                &mut big_output
            )
            .is_err()
        );
        // Wrong scales size.
        assert!(
            dequant_matmul_reference(
                &input,
                &codes,
                &[0.0f32; 1], // wrong scale count
                &biases,
                2,
                2,
                TILE_ELEMENTS,
                &mut big_output
            )
            .is_err()
        );
        // Wrong biases size.
        assert!(
            dequant_matmul_reference(
                &input,
                &codes,
                &scales,
                &[0.0f32; 1], // wrong bias count
                2,
                2,
                TILE_ELEMENTS,
                &mut big_output
            )
            .is_err()
        );
    }
}
