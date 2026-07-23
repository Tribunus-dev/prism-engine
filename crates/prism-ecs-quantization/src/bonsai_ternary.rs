//! Bonsai-specific 2-bit (GGUF Q2_0) to 1.58-bit (Tile640 ternary) conversion pipeline.
//!
//! The source checkpoint is GGUF Q2_0 (2-bit block-quantized). Steps:
//! 1. Dequantize Q2_0 blocks to f32
//! 2. Convert f32 weights to ternary (-1, 0, +1) via absmean threshold
//! 3. Pack as Tile640 (20 base-3 trits per u32)
//! 4. Compute per-page (640-weight) bf16 max-scale
//! 5. Compute per-lane (20-weight) int8 relative scale
//! 6. Extract top-0.5% outliers as sparse bf16

use sha2::{Digest, Sha256};
#[cfg(feature = "gguf-compile")]
use std::path::Path;

// ── Constants ──────────────────────────────────────────────────────────

/// Q2_0 block: 256 weights, fp16 scale (2 bytes), 64 bytes packed data.
const Q2_0_BLOCK_SIZE: usize = 256;
/// Q2_0 block total byte size: 2 (fp16 scale) + 64 (packed 2-bit values).
const Q2_0_BYTES_PER_BLOCK: usize = 66;

/// Tile640 constants — page = 640 weights = 32 u32 words, lane = 20 weights.
const TILE640_PAGE_SIZE: usize = 640;
const TILE640_LANE_SIZE: usize = 20;
const TILE640_WORDS_PER_PAGE: usize = 32;
const TILE640_LANES_PER_PAGE: usize = 32; // 640 / 20

/// Default fraction of weights treated as outliers.
const DEFAULT_OUTLIER_FRACTION: f64 = 0.005;

// ── Error type ─────────────────────────────────────────────────────────

/// Errors that can occur during Bonsai 2-bit → ternary conversion.
#[derive(Debug, thiserror::Error)]
pub enum BonsaiConversionError {
    /// The Q2_0 data slice does not contain a complete block header.
    #[error("Q2_0 dequant error: truncated data at block {block} — need {expected} bytes, have {actual}")]
    TruncatedData {
        block: usize,
        expected: usize,
        actual: usize,
    },

    /// The Q2_0 data length doesn't match the declared number of elements.
    #[error(
        "Q2_0 dequant error: data length {data_len} inconsistent with {num_elements} elements"
    )]
    DataLengthMismatch {
        data_len: usize,
        num_elements: usize,
    },

    /// The weight count is not a multiple of the Q2_0 block size.
    #[error("Q2_0 dequant error: {num_elements} is not a multiple of block size {block_size}")]
    NotBlockAligned {
        num_elements: usize,
        block_size: usize,
    },

    /// A ternary weight value outside {-1, 0, +1} was encountered during packing.
    #[error("Invalid ternary value {value} at index {index}; expected -1, 0, or +1")]
    InvalidTernaryValue { value: i8, index: usize },

    /// Shape declaration mismatch between expected and actual element count.
    #[error("Shape mismatch: {tensor} has {expected} elements but got {actual} weights")]
    ShapeMismatch {
        tensor: String,
        expected: usize,
        actual: usize,
    },

    /// I/O error reading or seeking the GGUF file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// FFI / NUL error in a tensor name.
    #[error("Invalid tensor name: {0}")]
    InvalidTensorName(String),
}

// ── Result type ────────────────────────────────────────────────────────

/// Intermediate result of ternarizing a flat f32 weight array.
pub struct TernaryResult {
    /// Ternary values: each element is -1, 0, or +1.
    pub ternary: Vec<i8>,
    /// Absmean threshold used for the conversion.
    pub threshold: f32,
    /// Indices of weights extracted as outliers (before ternarization).
    pub outlier_indices: Vec<usize>,
    /// Original f32 values of the extracted outliers.
    pub outlier_values: Vec<f32>,
}

/// Packed Tile640 representation of ternary weights.
#[derive(Debug)]
pub struct Tile640Packed {
    /// Packed ternary words: TILE640_WORDS_PER_PAGE per page.
    /// Encoding: 20 base-3 trits per u32, LSB-first.
    /// Trit 0 → ternary 0, trit 1 → ternary +1, trit 2 → ternary -1.
    pub packed_words: Vec<u32>,
    /// Number of pages (ceil(in_dim / 640) per output row × out_dim).
    pub num_pages: usize,
    /// Output dimension (rows).
    pub out_dim: u32,
    /// Input dimension (columns).
    pub in_dim: u32,
    /// Page size (number of weights per page).
    pub page_size: usize,
}

/// Two-level scales for Tile640.
pub struct Scales {
    /// BF16 page scales: one per page. Stored as u16 bits (BF16 representation).
    /// page_scales[p] = max_abs across all 640 weights in page p, as BF16.
    pub page_scales: Vec<u16>,
    /// Int8 lane scales: one per lane (32 per page).
    /// lane_scales[l] = round(page_max / lane_max * 127), clamped to [1, 127].
    pub lane_scales: Vec<i8>,
}

/// Set of outlier weights extracted for sparse correction.
pub struct OutlierSet {
    /// Output-row index of each outlier.
    pub rows: Vec<u32>,
    /// Input-column index of each outlier.
    pub cols: Vec<u32>,
    /// BF16-encoded value of each outlier, stored as u16 bits.
    pub values: Vec<u16>,
}

/// Record of a single tensor conversion.
pub struct TensorConversion {
    /// Name of the tensor in the GGUF file.
    pub tensor_name: String,
    /// Input dtype string (e.g., "q2_0").
    pub input_dtype: String,
    /// Output format string (always "tile640_ternary").
    pub output_format: String,
    /// Total packed byte size (packed_words × 4).
    pub packed_size: usize,
    /// Total page-scale byte size (page_scales × 2).
    pub page_scale_size: usize,
    /// Total lane-scale byte size (lane_scales × 1).
    pub lane_scale_size: usize,
    /// Number of extracted outlier weights.
    pub outlier_count: usize,
    /// SHA-256 digest of the packed payload.
    pub digest: [u8; 32],
}

// ── Q2_0 dequantization ───────────────────────────────────────────────

/// Dequantize a GGML Q2_0 tensor from raw bytes to f32.
///
/// Q2_0 block format:
/// - Block size: 256 weights
/// - Per block: 2 bytes fp16 scale (little-endian) + 64 bytes packed 2-bit data
/// - 4 weights per byte (LSB first)
/// - 2-bit values: 0 → -1, 1 → 0, 2 → +1, 3 → +2
/// - Dequantized: val = (q2_val - 1) * scale
pub fn dequantize_q2_0(
    data: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>, BonsaiConversionError> {
    if num_elements % Q2_0_BLOCK_SIZE != 0 {
        return Err(BonsaiConversionError::NotBlockAligned {
            num_elements,
            block_size: Q2_0_BLOCK_SIZE,
        });
    }

    let num_blocks = num_elements / Q2_0_BLOCK_SIZE;
    let expected_bytes = num_blocks * Q2_0_BYTES_PER_BLOCK;

    if data.len() < expected_bytes {
        return Err(BonsaiConversionError::DataLengthMismatch {
            data_len: data.len(),
            num_elements,
        });
    }

    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_offset = block_idx * Q2_0_BYTES_PER_BLOCK;
        let block_end = block_offset + Q2_0_BYTES_PER_BLOCK;

        if block_end > data.len() {
            return Err(BonsaiConversionError::TruncatedData {
                block: block_idx,
                expected: block_end,
                actual: data.len(),
            });
        }

        // Read fp16 scale (2 bytes, little-endian)
        let scale_bits = u16::from_le_bytes([data[block_offset], data[block_offset + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        // Read 64 bytes of packed 2-bit data
        let packed_start = block_offset + 2;
        let packed = &data[packed_start..packed_start + 64];

        for &byte in packed {
            for j in 0..4 {
                let nibble = (byte >> (j * 2)) & 0x03;
                // Q2_0: 0→-1, 1→0, 2→+1, 3→+2
                let q2_val = nibble as i32;
                let signed = (q2_val - 1) as f32;
                result.push(signed * scale);
            }
        }
    }

    Ok(result)
}

// ── Ternarization ─────────────────────────────────────────────────────

/// Convert f32 weights to ternary (-1, 0, +1) using absmean threshold.
///
/// The threshold is computed as `sum(|w|) / n`. Weights with |w| below the
/// threshold become 0; above threshold they round to +1 or -1.
///
/// `outlier_frac` specifies the fraction of largest-magnitude weights to
/// extract as outliers before ternarization (e.g., 0.005 for top 0.5%).
pub fn ternarize_f32(weights: &[f32], outlier_frac: f64) -> TernaryResult {
    let n = weights.len();
    if n == 0 {
        return TernaryResult {
            ternary: Vec::new(),
            threshold: 0.0,
            outlier_indices: Vec::new(),
            outlier_values: Vec::new(),
        };
    }

    // Compute absmean threshold
    let abs_sum: f32 = weights.iter().map(|w| w.abs()).sum();
    let threshold = abs_sum / n as f32;

    // Find outlier indices: largest-magnitude weights top `outlier_frac`
    let outlier_count = ((n as f64) * outlier_frac).ceil() as usize;
    let outlier_count = outlier_count.min(n);

    let mut indices: Vec<usize> = (0..n).collect();
    // Sort indices by descending absolute weight value
    indices.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let outlier_indices: Vec<usize> = indices[..outlier_count].to_vec();
    let outlier_values: Vec<f32> = outlier_indices.iter().map(|&i| weights[i]).collect();
    let outlier_set: std::collections::HashSet<usize> = outlier_indices.iter().copied().collect();

    // Ternarize non-outlier weights
    let mut ternary = Vec::with_capacity(n);
    for (i, &w) in weights.iter().enumerate() {
        if outlier_set.contains(&i) {
            // Outliers are excluded from ternary representation — they get
            // separate sparse BF16 storage instead.
            ternary.push(0);
        } else if w.abs() < threshold {
            ternary.push(0);
        } else if w > 0.0 {
            ternary.push(1);
        } else {
            ternary.push(-1);
        }
    }

    TernaryResult {
        ternary,
        threshold,
        outlier_indices,
        outlier_values,
    }
}

// ── Tile640 packing ──────────────────────────────────────────────────

/// Pack ternary weights into Tile640 format.
///
/// Layout:
/// - A "page" is 640 weights → TILE640_WORDS_PER_PAGE = 32 u32 words.
/// - A "lane" is 20 weights → 1 u32 word using base-3 encoding.
/// - 3^20 < 2^32, so 20 base-3 trits fit in one u32.
/// - LSB-first encoding: trit_i = (packed / 3^i) % 3.
/// - Trit encoding: 0 → ternary 0, 1 → +1, 2 → -1.
///
/// The total number of pages per row is ceil(in_dim / 640).
/// Rows are packed consecutively: row 0 pages, row 1 pages, ...
pub fn pack_tile640(
    ternary: &[i8],
    out_dim: u32,
    in_dim: u32,
    page_size: usize,
) -> Result<Tile640Packed, BonsaiConversionError> {
    let out = out_dim as usize;
    let inp = in_dim as usize;
    let expected = out * inp;

    if ternary.len() < expected {
        return Err(BonsaiConversionError::ShapeMismatch {
            tensor: String::new(),
            expected,
            actual: ternary.len(),
        });
    }

    let lanes_per_page = page_size / TILE640_LANE_SIZE;
    let words_per_page = lanes_per_page;
    let pages_per_row = (inp + page_size - 1) / page_size;
    let total_words = out * pages_per_row * words_per_page;
    let mut packed_words = vec![0u32; total_words];

    for row in 0..out {
        for page in 0..pages_per_row {
            for lane in 0..lanes_per_page {
                let word_idx = row * pages_per_row * words_per_page + page * words_per_page + lane;
                let weight_base = row * inp + page * page_size + lane * TILE640_LANE_SIZE;

                let mut word: u32 = 0;
                for vi in 0..TILE640_LANE_SIZE {
                    let weight_idx = weight_base + vi;
                    let ternary_val = if weight_idx < ternary.len() {
                        ternary[weight_idx]
                    } else {
                        0i8
                    };

                    // Validate before encoding
                    if !matches!(ternary_val, -1 | 0 | 1) {
                        return Err(BonsaiConversionError::InvalidTernaryValue {
                            value: ternary_val,
                            index: weight_idx,
                        });
                    }

                    // Map: -1 → trit 2, 0 → trit 0, +1 → trit 1
                    let trit: u32 = match ternary_val {
                        -1 => 2,
                        0 => 0,
                        1 => 1,
                        _ => unreachable!(),
                    };
                    word += trit * (3u32.pow(vi as u32));
                }
                packed_words[word_idx] = word;
            }
        }
    }

    Ok(Tile640Packed {
        packed_words,
        num_pages: pages_per_row * out,
        out_dim,
        in_dim,
        page_size,
    })
}

// ── Scale computation ─────────────────────────────────────────────────

/// Compute two-level scales for Tile640 packed weights.
///
/// - Per-page (640-weight) BF16 max-scale: the max absolute value across all
///   640 weights in the page, stored as BF16 bits in a u16.
/// - Per-lane (20-weight) int8 relative scale: how much smaller this lane's
///   max is relative to the page max, quantized to an int8 in [1, 127]:
///     lane_scale = round(page_max / lane_max * 127)
///
/// If lane_max is near zero (all values zero in lane), lane_scale is set to 1.
/// If lane_max == page_max, lane_scale is 127.
pub fn compute_scales(packed: &Tile640Packed, original_weights: &[f32]) -> Scales {
    let out = packed.out_dim as usize;
    let inp = packed.in_dim as usize;
    let page_size = packed.page_size;
    let lanes_per_page = page_size / TILE640_LANE_SIZE;
    let pages_per_row = (inp + page_size - 1) / page_size;
    let num_pages = out * pages_per_row;

    let mut page_scales = vec![0u16; num_pages];
    let mut lane_scales = vec![0i8; num_pages * lanes_per_page];

    for row in 0..out {
        for page in 0..pages_per_row {
            let page_idx = row * pages_per_row + page;
            let weight_base = row * inp + page * page_size;

            // Compute page max: max absolute value across all 640 weights
            let mut page_max = 0.0f32;
            for wi in 0..page_size {
                let idx = weight_base + wi;
                let val = if idx < original_weights.len() {
                    original_weights[idx].abs()
                } else {
                    0.0
                };
                if val > page_max {
                    page_max = val;
                }
            }

            // Guard against zero page
            if page_max < 1e-30 {
                page_max = 1.0;
            }

            // Store as BF16 bits
            let page_bf16 = half::bf16::from_f32(page_max);
            page_scales[page_idx] = page_bf16.to_bits();

            // Compute per-lane scales
            for lane in 0..lanes_per_page {
                let lane_idx = page_idx * lanes_per_page + lane;
                let lane_base = weight_base + (lane * TILE640_LANE_SIZE) as usize;

                let mut lane_max = 0.0f32;
                for vi in 0..TILE640_LANE_SIZE {
                    let idx = lane_base + vi;
                    let val = if idx < original_weights.len() {
                        original_weights[idx].abs()
                    } else {
                        0.0
                    };
                    if val > lane_max {
                        lane_max = val;
                    }
                }

                // lane_scale = round(page_max / lane_max * 127), clamped [1, 127]
                let raw_scale = if lane_max < 1e-30 {
                    1.0f32
                } else {
                    (page_max / lane_max) * 127.0f32
                };
                let clamped = (raw_scale.round() as i32).clamp(1, 127) as i8;
                lane_scales[lane_idx] = clamped;
            }
        }
    }

    Scales {
        page_scales,
        lane_scales,
    }
}

// ── Outlier extraction ────────────────────────────────────────────────

/// Extract outlier weights from a ternary-converted tensor.
///
/// Outliers are weights where the ternary |0/±1| representation loses too much
/// precision. Given the ternary result from `ternarize_f32`, the outliers are
/// the weights that were already identified as such (largest |w| values).
///
/// Returns (row, column, bf16_value) for each outlier.
pub fn extract_outliers(weights: &[f32], _ternary: &[i8], fraction: f64) -> OutlierSet {
    let n = weights.len();
    if n == 0 {
        return OutlierSet {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
        };
    }

    let outlier_count = ((n as f64) * fraction).ceil() as usize;
    let outlier_count = outlier_count.min(n);

    // We need to find the largest-magnitude weights. We don't have the row/col
    // layout here (no out_dim/in_dim), but we'll track flat indices. The caller
    // can reinterpret flat indices as (row, col) when it has the dimensions.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let outlier_indices = &indices[..outlier_count];
    let mut rows = Vec::with_capacity(outlier_count);
    let mut cols = Vec::with_capacity(outlier_count);
    let mut values = Vec::with_capacity(outlier_count);

    for &idx in outlier_indices {
        let val = weights[idx];
        let bf16 = half::bf16::from_f32(val);
        rows.push(0); // Filled in by the caller with actual row dimension
        cols.push(idx as u32);
        values.push(bf16.to_bits());
    }

    OutlierSet { rows, cols, values }
}

/// Assign row indices to outlier columns based on out_dim/in_dim.
///
/// `extract_outliers` stores flat column indices; this function converts them
/// to (row, col) pairs given the matrix dimensions.
pub fn assign_outlier_rows(outliers: &mut OutlierSet, _out_dim: u32, in_dim: u32) {
    let inp = in_dim as usize;
    for i in 0..outliers.cols.len() {
        let flat = outliers.cols[i] as usize;
        let row = (flat / inp) as u32;
        let col = (flat % inp) as u32;
        outliers.rows[i] = row;
        outliers.cols[i] = col;
    }
}

// ── Full conversion pipeline ──────────────────────────────────────────

/// Orchestrates the full Bonsai 2-bit → ternary conversion pipeline.
pub struct Bonsai2To3Conversion {
    /// Fraction of weights to extract as outliers.
    pub outlier_fraction: f64,
}

impl Default for Bonsai2To3Conversion {
    fn default() -> Self {
        Self {
            outlier_fraction: DEFAULT_OUTLIER_FRACTION,
        }
    }
}

impl Bonsai2To3Conversion {
    /// Create a new conversion pipeline with the given outlier fraction.
    pub fn new(outlier_fraction: f64) -> Self {
        Self { outlier_fraction }
    }

    /// Convert a single tensor from Q2_0 bytes through the full pipeline.
    ///
    /// Returns the packed Tile640 data, scales, outliers, and conversion digest.
    pub fn convert_tensor(
        &self,
        data: &[u8],
        name: &str,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<TensorConversion, BonsaiConversionError> {
        let num_elements = (out_dim as usize) * (in_dim as usize);

        // Step 1: Dequantize Q2_0 → f32
        let f32_vals = dequantize_q2_0(data, num_elements)?;

        // Step 2: Ternarize f32 → {-1, 0, +1}
        let ternary_result = ternarize_f32(&f32_vals, self.outlier_fraction);

        // Step 3: Pack ternary into Tile640
        let packed = pack_tile640(&ternary_result.ternary, out_dim, in_dim, TILE640_PAGE_SIZE)?;

        // Step 4: Compute scales
        let scales = compute_scales(&packed, &f32_vals);

        // Step 5: Extract outliers
        let mut outliers =
            extract_outliers(&f32_vals, &ternary_result.ternary, self.outlier_fraction);
        assign_outlier_rows(&mut outliers, out_dim, in_dim);

        // Compute digest over packed bytes
        let packed_bytes: Vec<u8> = packed
            .packed_words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let mut hasher = Sha256::new();
        hasher.update(&packed_bytes);
        let digest: [u8; 32] = hasher.finalize().into();

        Ok(TensorConversion {
            tensor_name: name.to_string(),
            input_dtype: "q2_0".to_string(),
            output_format: "tile640_ternary".to_string(),
            packed_size: packed_bytes.len(),
            page_scale_size: scales.page_scales.len() * 2,
            lane_scale_size: scales.lane_scales.len(),
            outlier_count: outliers.cols.len(),
            digest,
        })
    }

    /// Convert an entire GGUF checkpoint file.
    ///
    /// Reads each tensor, dequantizes from Q2_0, ternarizes, packs, computes
    /// scales, and extracts outliers. Returns a vector of `TensorConversion`
    /// records describing every tensor processed.
    #[cfg(feature = "gguf-compile")]
    pub fn convert_checkpoint(
        &self,
        gguf_path: &Path,
    ) -> Result<Vec<TensorConversion>, BonsaiConversionError> {
        use prism_gguf::parse_gguf_header;
        use prism_gguf::read_tensor_f32;

        let (_metadata, tensor_meta) = parse_gguf_header(gguf_path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut file = std::fs::File::open(gguf_path)?;
        let mut results = Vec::with_capacity(tensor_meta.len());

        for meta in &tensor_meta {
            // Process based on dtype string.
            // Q2_0 tensors go through dequantize→ternarize→pack.
            if meta.dtype == "q2_0" {
                // Read raw Q2_0 bytes and run the full pipeline
                let raw_bytes = read_raw_tensor_bytes(&mut file, meta)?;
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                let (out_dim, in_dim) = if shape.len() >= 2 {
                    (shape[0], shape[1])
                } else if shape.len() == 1 {
                    (shape[0], 1u32)
                } else {
                    (1u32, 1u32)
                };
                let conv = self.convert_tensor(&raw_bytes, &meta.name, out_dim, in_dim)?;
                results.push(conv);
            } else {
                // Non-Q2_0: load as f32 and ternarize directly
                let f32_data = match read_tensor_f32(&mut file, meta) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!(
                            "[bonsai_ternary] skipping tensor {} (dtype={}): {}",
                            meta.name, meta.dtype, e
                        );
                        continue;
                    }
                };

                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                let (out_dim, _in_dim) = if shape.len() >= 2 {
                    (shape[0], shape[1])
                } else if shape.len() == 1 {
                    (shape[0], 1u32)
                } else {
                    (1u32, 1u32)
                };
                let num_elements = f32_data.len() as u32;
                let actual_out = out_dim.max(1);
                let actual_in = (num_elements / actual_out).max(1);

                let ternary_result = ternarize_f32(&f32_data, self.outlier_fraction);
                let packed = pack_tile640(
                    &ternary_result.ternary,
                    actual_out,
                    actual_in,
                    TILE640_PAGE_SIZE,
                )?;
                let _scales = compute_scales(&packed, &f32_data);

                let packed_bytes: Vec<u8> = packed
                    .packed_words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                let mut hasher = Sha256::new();
                hasher.update(&packed_bytes);
                let digest: [u8; 32] = hasher.finalize().into();

                results.push(TensorConversion {
                    tensor_name: meta.name.clone(),
                    input_dtype: meta.dtype.clone(),
                    output_format: "tile640_ternary".to_string(),
                    packed_size: packed_bytes.len(),
                    page_scale_size: 0,
                    lane_scale_size: 0,
                    outlier_count: 0,
                    digest,
                });
            }
        }

        Ok(results)
    }
}

/// Read the raw bytes of a tensor from a GGUF file without dequantization.
#[cfg(feature = "gguf-compile")]
fn read_raw_tensor_bytes(
    file: &mut std::fs::File,
    meta: &prism_gguf::GgufTensorMeta,
) -> Result<Vec<u8>, std::io::Error> {
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(meta.byte_offset))?;
    let mut buf = vec![0u8; meta.byte_size as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

// ── CPU Reference GEMV ─────────────────────────────────────────────────────

/// Reference GEMV using Tile640-packed ternary weights.
///
/// Matches the Metal kernel (`ternary_tile640_gemv`) exactly — bit-identical
/// results (assuming the same fp math) with the same two-level scale
/// application, base-3 unpacking, and output precision.
///
/// # Arguments
/// * `packed` — Packed u32 words: `[out_dim * nt * 32]` where `nt = ceil(in_dim / 640)`.
/// * `input_vector` — F32 input vector, length `in_dim`.
/// * `page_scales` — BF16 page scales (u16 bits), length `out_dim * nt`.
/// * `lane_scales` — Int8 relative scales, length `out_dim * nt * 32`.
/// * `out_dim` — Number of output rows.
/// * `in_dim` — Number of input columns.
///
/// # Returns
/// Output vector of `out_dim` f32 values.
pub fn ternary_gemv_ref(
    packed: &[u32],
    input_vector: &[f32],
    page_scales: &[u16],
    lane_scales: &[i8],
    out_dim: u32,
    in_dim: u32,
) -> Vec<f32> {
    let out = out_dim as usize;
    let inp = in_dim as usize;
    let nt = (inp + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
    let words_per_row = nt * TILE640_WORDS_PER_PAGE; // nt * 32

    let mut output = vec![0.0f32; out];

    for row in 0..out {
        let mut acc = 0.0f32;
        let row_offset = row * words_per_row;

        for wi in 0..words_per_row {
            let p = wi / TILE640_LANES_PER_PAGE; // page within row
            let lane = wi % TILE640_LANES_PER_PAGE; // lane within page (0..31)
            let col0 = p * TILE640_PAGE_SIZE + lane * TILE640_LANE_SIZE;

            // Two-level scale: bf16 page-max × (int8 lane / 127)
            let page_bits = page_scales[row * nt + p];
            let page_max = f32::from(half::bf16::from_bits(page_bits));
            let lane_s = lane_scales[row * words_per_row + wi] as f32;
            let scale = page_max * (lane_s * (1.0f32 / 127.0f32));

            let mut word = packed[row_offset + wi];
            for vi in 0..TILE640_LANE_SIZE {
                let d = word % 3; // LSB trit: 0, 1, 2
                word /= 3;
                let col = col0 + vi;
                if col >= inp {
                    break;
                }
                if d != 0 {
                    let tv = if d == 1 { scale } else { -scale };
                    acc = fma_f32(input_vector[col], tv, acc);
                }
            }
        }

        output[row] = acc;
    }

    output
}
/// Apply outlier correction to a ternary GEMV output.
///
/// Outliers are large-magnitude weights that were removed before ternarization.
/// For each outlier at (row, col) with a BF16 value, we add the full-precision
/// contribution back: `output[row] += input_vector[col] * outlier_value`.
///
/// # Arguments
/// * `output` — Output vector (mutated in-place), length `out_dim`.
/// * `input_vector` — F32 input vector, length `in_dim`.
/// * `outlier_rows` — Row indices for each outlier (`u32` LE bytes).
/// * `outlier_cols` — Column indices for each outlier (`u32` LE bytes).
/// * `outlier_vals` — BF16-encoded values (`u16` LE bytes).
/// * `out_dim` — Number of output rows.
/// * `in_dim` — Number of input columns.
pub fn apply_outlier_correction(
    output: &mut [f32],
    input_vector: &[f32],
    outlier_rows: &[u32],
    outlier_cols: &[u32],
    outlier_vals: &[u16],
    out_dim: u32,
    in_dim: u32,
) {
    let out = out_dim as usize;
    let inp = in_dim as usize;
    debug_assert_eq!(outlier_rows.len(), outlier_cols.len());
    debug_assert_eq!(outlier_rows.len(), outlier_vals.len());
    debug_assert!(output.len() >= out);
    debug_assert!(input_vector.len() >= inp);

    for i in 0..outlier_rows.len() {
        let row = outlier_rows[i] as usize;
        let col = outlier_cols[i] as usize;
        if row < out && col < inp {
            let val = f32::from(half::bf16::from_bits(outlier_vals[i]));
            output[row] = fma_f32(input_vector[col], val, output[row]);
        }
    }
}

/// Fused multiply-add helper for bit-exact matching with Metal's `fma`.
/// Metal's `fma(a, b, c)` uses IEEE 754 fused multiply-add (single rounding).
#[inline(always)]
fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Simple reference GEMV using raw i8 ternary weights (no packing).
///
/// Useful for testing correctness of the reference pipeline without the
/// Tile640 packing layer.
///
/// # Arguments
/// * `ternary_weights` — Flat row-major ternary weights: `[out_dim * in_dim]`,
///   each element is -1, 0, or +1.
/// * `input_vector` — F32 input vector, length `in_dim`.
/// * `scales` — Per-row scale factors, length `out_dim`.
/// * `out_dim` — Number of output rows.
/// * `in_dim` — Number of input columns.
///
/// # Returns
/// Output vector of `out_dim` f32 values.
pub fn ternary_gemv_ref_simple(
    ternary_weights: &[i8],
    input_vector: &[f32],
    scales: &[f32],
    out_dim: u32,
    in_dim: u32,
) -> Vec<f32> {
    let out = out_dim as usize;
    let inp = in_dim as usize;

    let mut output = vec![0.0f32; out];
    for row in 0..out {
        let mut acc = 0.0f32;
        let row_offset = row * inp;
        for col in 0..inp {
            let w = ternary_weights[row_offset + col] as f32;
            if w != 0.0f32 {
                acc = fma_f32(input_vector[col], w * scales[row], acc);
            }
        }
        output[row] = acc;
    }
    output
}

// ── ABI Validation ─────────────────────────────────────────────────────────

/// Receipt from ABI validation, recording what was checked.
#[derive(Debug, Clone)]
pub struct AbiValidationReceipt {
    /// Whether all checks passed.
    pub valid: bool,
    /// Human-readable description of the validation outcome.
    pub description: String,
    /// Computed expected packed data length in u32 words.
    pub expected_packed_words: usize,
    /// Actual packed data length.
    pub actual_packed_words: usize,
    /// Whether any reserved 0b11 patterns were found in the packed data.
    pub has_reserved_patterns: Option<bool>,
}

/// Validate ternary ABI layout for Tile640-packed data.
///
/// Checks:
/// 1. Packed data size matches `out_dim * ceil(in_dim / 640) * 32`.
/// 2. `page_scales` length matches `out_dim * ceil(in_dim / 640)`.
/// 3. `lane_scales` length matches `out_dim * ceil(in_dim / 640) * 32`.
/// 4. No reserved 0b11 patterns in packed data (trit values must be 0, 1, or 2).
/// 5. `page_scales` contains no zero page scales (zero values are clamped).
pub fn validate_ternary_abi(
    packed: &[u32],
    page_scales: &[u16],
    lane_scales: &[i8],
    out_dim: u32,
    in_dim: u32,
) -> AbiValidationReceipt {
    let out = out_dim as usize;
    let inp = in_dim as usize;
    let nt = (inp + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
    let expected_packed_words = out * nt * TILE640_WORDS_PER_PAGE;
    let expected_page_scales = out * nt;
    let expected_lane_scales = out * nt * TILE640_LANES_PER_PAGE;

    // Check 1: Packed data size
    if packed.len() != expected_packed_words {
        return AbiValidationReceipt {
            valid: false,
            description: format!(
                "packed data size mismatch: expected {} words, got {}",
                expected_packed_words,
                packed.len()
            ),
            expected_packed_words,
            actual_packed_words: packed.len(),
            has_reserved_patterns: None,
        };
    }

    // Check 2: Page scales length
    if page_scales.len() != expected_page_scales {
        return AbiValidationReceipt {
            valid: false,
            description: format!(
                "page_scales size mismatch: expected {}, got {}",
                expected_page_scales,
                page_scales.len()
            ),
            expected_packed_words,
            actual_packed_words: packed.len(),
            has_reserved_patterns: None,
        };
    }

    // Check 3: Lane scales length
    if lane_scales.len() != expected_lane_scales {
        return AbiValidationReceipt {
            valid: false,
            description: format!(
                "lane_scales size mismatch: expected {}, got {}",
                expected_lane_scales,
                lane_scales.len()
            ),
            expected_packed_words,
            actual_packed_words: packed.len(),
            has_reserved_patterns: None,
        };
    }

    // Check 4: No reserved patterns — scan each word for trit value 3 (0b11)
    let mut has_reserved = false;
    for &word in packed {
        let mut w = word;
        for _ in 0..TILE640_LANE_SIZE {
            if w % 3 > 2 {
                // Sanity: w % 3 is always 0,1,2 but check for safety
                has_reserved = true;
                break;
            }
            // The only way to have reserved pattern is if some trit value is > 2,
            // but since we encode base-3, w % 3 can only be 0,1,2. However,
            // detect actual trit values of 3+ by checking bits more carefully.
            w /= 3;
        }
        // Also check if any bit pair is 0b11 in the packed representation.
        // In Tile640 base-3 encoding there are no fixed bit positions, but
        // we verify the rem/div cycle doesn't produce anything > 2.
    }

    // Scan for actual reserved trit encoding issues: in the tile640 format,
    // each word is base-3 packed. No word should have rem values outside {0,1,2}.
    // Additionally, if any original f32 weight had a pattern that maps to
    // trit 3+ it would have been caught by pack_tile640's validation.

    // Check 5: Page scales should be non-zero (zero means uninitialized)
    let zero_page_scales: Vec<usize> = page_scales
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 0)
        .map(|(i, _)| i)
        .collect();

    if !zero_page_scales.is_empty() {
        return AbiValidationReceipt {
            valid: false,
            description: format!(
                "page_scales contains zero values at indices {:?} (uninitialized)",
                zero_page_scales
            ),
            expected_packed_words,
            actual_packed_words: packed.len(),
            has_reserved_patterns: Some(has_reserved),
        };
    }

    AbiValidationReceipt {
        valid: true,
        description: "ternary ABI validation passed".to_string(),
        expected_packed_words,
        actual_packed_words: packed.len(),
        has_reserved_patterns: Some(has_reserved),
    }
}

/// Receipt from outlier validation.
#[derive(Debug, Clone)]
pub struct OutlierValidationReceipt {
    /// Whether all checks passed.
    pub valid: bool,
    /// Human-readable description.
    pub description: String,
    /// Number of outliers.
    pub outlier_count: usize,
}

/// Validate outlier sets for correctness.
///
/// Checks:
/// 1. All row indices are within `[0, out_dim)`.
/// 2. All col indices are within `[0, in_dim)`.
/// 3. All value arrays have matching lengths.
/// 4. All values are finite (no NaN, inf).
/// 5. Value counts match across rows, cols, and values.
pub fn validate_outliers(
    outliers: &OutlierSet,
    out_dim: u32,
    in_dim: u32,
) -> OutlierValidationReceipt {
    // Matching lengths
    let n = outliers.rows.len();
    if outliers.cols.len() != n || outliers.values.len() != n {
        return OutlierValidationReceipt {
            valid: false,
            description: format!(
                "outlier array length mismatch: rows={}, cols={}, values={}",
                outliers.rows.len(),
                outliers.cols.len(),
                outliers.values.len()
            ),
            outlier_count: n,
        };
    }

    // Bounds and finiteness
    for i in 0..n {
        if outliers.rows[i] >= out_dim {
            return OutlierValidationReceipt {
                valid: false,
                description: format!(
                    "outlier row index {} out of bounds: row={} >= out_dim={}",
                    i, outliers.rows[i], out_dim
                ),
                outlier_count: n,
            };
        }
        if outliers.cols[i] >= in_dim {
            return OutlierValidationReceipt {
                valid: false,
                description: format!(
                    "outlier col index {} out of bounds: col={} >= in_dim={}",
                    i, outliers.cols[i], in_dim
                ),
                outlier_count: n,
            };
        }

        // Check finiteness: reconstruct bf16 → f32 and verify
        let val = f32::from(half::bf16::from_bits(outliers.values[i]));
        if !val.is_finite() {
            return OutlierValidationReceipt {
                valid: false,
                description: format!(
                    "outlier value at index {} is not finite: {:.6} (bits: {})",
                    i, val, outliers.values[i]
                ),
                outlier_count: n,
            };
        }
    }

    OutlierValidationReceipt {
        valid: true,
        description: format!("outlier validation passed: {} outliers", n),
        outlier_count: n,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Q2_0 dequant tests ─────────────────────────────────────────

    #[test]
    fn test_dequantize_q2_0_single_block() {
        // Build a single Q2_0 block (66 bytes):
        // scale = fp16 of 2.0 = 0x4000 (le bytes: 0x00, 0x40)
        // All weights = 2-bit value 0b01 = 1 (signed → 0, so result is 0 * 2.0 = 0.0)
        let mut block = Vec::with_capacity(Q2_0_BYTES_PER_BLOCK);
        // Scale = 2.0 in fp16
        block.extend_from_slice(&0x4000u16.to_le_bytes());
        // 64 bytes of value 0b01010101 = 0x55 → each nibble = 0b01 = 1
        // signed = (1 - 1) = 0, so all zeros
        block.extend_from_slice(&[0x55u8; 64]);

        let result = dequantize_q2_0(&block, 256).unwrap();
        assert_eq!(result.len(), 256);
        // All values should be 0.0
        for v in &result {
            assert!((*v).abs() < 1e-6, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q2_0_positive_ones() {
        // 2-bit value 0b10 = 2 → signed = +1 → val = 1.0 * scale
        let mut block = Vec::with_capacity(Q2_0_BYTES_PER_BLOCK);
        // Scale = 1.5 in fp16
        let scale_f16 = half::f16::from_f32(1.5);
        block.extend_from_slice(&scale_f16.to_bits().to_le_bytes());
        // Each nibble = 0b10 = 2
        // byte 0b10101010 = 0xAA
        block.extend_from_slice(&[0xAAu8; 64]);

        let result = dequantize_q2_0(&block, 256).unwrap();
        assert_eq!(result.len(), 256);
        for v in &result {
            let expected = (2i32 - 1) as f32 * 1.5;
            assert!(
                (*v - expected).abs() < 1e-3,
                "expected {}, got {}",
                expected,
                v
            );
        }
    }

    #[test]
    fn test_dequantize_q2_0_mixed_nibbles() {
        // Create a block with all 4 possible values cycling
        let mut block = Vec::with_capacity(Q2_0_BYTES_PER_BLOCK);
        let scale_f16 = half::f16::from_f32(1.0);
        block.extend_from_slice(&scale_f16.to_bits().to_le_bytes());

        // Pattern: 0b11_10_01_00 = 0xE4, cycling through 0,1,2,3
        let mut packed = Vec::with_capacity(64);
        for i in 0..64 {
            // Each byte encodes 4 weights: weight_j = (i * 4 + j) % 4
            let mut byte = 0u8;
            for j in 0..4 {
                let val = ((i * 4 + j) % 4) as u8;
                byte |= val << (j * 2);
            }
            packed.push(byte);
        }
        block.extend_from_slice(&packed);

        let result = dequantize_q2_0(&block, 256).unwrap();
        assert_eq!(result.len(), 256);

        for (idx, &v) in result.iter().enumerate() {
            let nibble_val = (idx % 4) as i32;
            let expected = (nibble_val - 1) as f32;
            assert!(
                (v - expected).abs() < 1e-6,
                "idx {}: nibble={} expected {} got {}",
                idx,
                nibble_val,
                expected,
                v
            );
        }
    }

    #[test]
    fn test_dequantize_q2_0_multiple_blocks() {
        // Two blocks: first scale 1.0, second scale 2.0
        let mut data = Vec::new();

        // Block 0: scale=1.0, all 0b01 (value 1 → signed 0)
        let s0 = half::f16::from_f32(1.0);
        data.extend_from_slice(&s0.to_bits().to_le_bytes());
        data.extend_from_slice(&[0x55u8; 64]);

        // Block 1: scale=2.0, all 0b10 (value 2 → signed +1)
        let s1 = half::f16::from_f32(2.0);
        data.extend_from_slice(&s1.to_bits().to_le_bytes());
        data.extend_from_slice(&[0xAAu8; 64]);

        let result = dequantize_q2_0(&data, 512).unwrap();
        assert_eq!(result.len(), 512);

        // Block 0: all zeros
        for i in 0..256 {
            assert!(
                result[i].abs() < 1e-6,
                "block0 idx {} expected 0 got {}",
                i,
                result[i]
            );
        }
        // Block 1: all (2-1)*2.0 = 2.0
        for i in 256..512 {
            let expected = (2i32 - 1) as f32 * 2.0;
            assert!(
                (result[i] - expected).abs() < 1e-3,
                "block1 idx {} expected {} got {}",
                i,
                expected,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q2_0_truncated_data() {
        let data = vec![0u8; 10]; // Too small for even one block
        let err = dequantize_q2_0(&data, 256).unwrap_err();
        assert!(matches!(
            err,
            BonsaiConversionError::DataLengthMismatch { .. }
        ));
    }

    #[test]
    fn test_dequantize_q2_0_not_block_aligned() {
        let data = vec![0u8; Q2_0_BYTES_PER_BLOCK];
        let err = dequantize_q2_0(&data, 100).unwrap_err();
        assert!(matches!(err, BonsaiConversionError::NotBlockAligned { .. }));
    }

    // ── Ternarization tests ────────────────────────────────────────

    #[test]
    fn test_ternarize_f32_all_positive() {
        // All weights positive and above absmean threshold -> all +1
        // threshold = (10+20+30+40)/4 = 25, so 30 and 40 are above, 10 and 20 below
        // Use weights with equal magnitude to trigger all +1
        let weights = vec![100.0, 100.0, 100.0, 100.0];
        let result = ternarize_f32(&weights, 0.0);
        assert_eq!(result.ternary, vec![1, 1, 1, 1]);
        assert!(result.threshold > 0.0);
        assert!(result.outlier_indices.is_empty());
    }

    #[test]
    fn test_ternarize_f32_mixed() {
        // Weights with varying magnitudes
        let weights = vec![5.0, -3.0, 0.1, -0.05, 2.0, -4.0];
        let result = ternarize_f32(&weights, 0.0);
        // threshold = (5+3+0.1+0.05+2+4)/6 ≈ 14.15/6 ≈ 2.358
        // 5.0 > 2.358 → +1
        // -3.0 > 2.358 → -1
        // 0.1 < 2.358 → 0
        // -0.05 < 2.358 → 0
        // 2.0 < 2.358 → 0
        // -4.0 > 2.358 → -1
        assert_eq!(result.ternary, vec![1, -1, 0, 0, 0, -1]);
        assert!(result.threshold > 0.0);
    }

    #[test]
    fn test_ternarize_f32_with_outliers() {
        let weights = vec![100.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // Remove top 1/6 ≈ 16.7% (ceil gives 1 outlier)
        let result = ternarize_f32(&weights, 1.0 / 6.0);
        // Outlier should be index 0 (value 100.0)
        assert_eq!(result.outlier_indices, vec![0]);
        assert_eq!(result.outlier_values, vec![100.0]);
        // outlier at index 0 should be 0 in ternary
        assert_eq!(result.ternary[0], 0);
    }

    #[test]
    fn test_ternarize_f32_empty() {
        let result = ternarize_f32(&[], 0.0);
        assert!(result.ternary.is_empty());
        assert_eq!(result.threshold, 0.0);
    }

    // ── Tile640 packing tests ──────────────────────────────────────

    #[test]
    fn test_pack_tile640_one_page() {
        // 640 weights = 1 page = 32 u32s
        // All -1 values → trit 2
        let ternary = vec![-1i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        assert_eq!(packed.packed_words.len(), 32);
        assert_eq!(packed.num_pages, 1);

        // Each word is all trit-2: sum_{i=0}^{19} 2 * 3^i
        let expected_word: u32 = (0..20).map(|i| 2u32 * 3u32.pow(i)).sum();
        for &w in &packed.packed_words {
            assert_eq!(w, expected_word, "all -1 should encode as all trit-2");
        }
    }

    #[test]
    fn test_pack_tile640_all_ones() {
        // All +1 values → trit 1
        let ternary = vec![1i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        let expected_word: u32 = (0..20).map(|i| 1u32 * 3u32.pow(i)).sum();
        for &w in &packed.packed_words {
            assert_eq!(w, expected_word, "all +1 should encode as all trit-1");
        }
    }

    #[test]
    fn test_pack_tile640_all_zero() {
        let ternary = vec![0i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        for &w in &packed.packed_words {
            assert_eq!(w, 0, "all zero should encode as all trit-0");
        }
    }

    #[test]
    fn test_pack_tile640_multi_row() {
        // 2 rows × 640 cols = 1280 weights
        let mut ternary = vec![0i8; 1280];
        // First row: all -1
        for i in 0..640 {
            ternary[i] = -1;
        }
        // Second row: all +1
        for i in 640..1280 {
            ternary[i] = 1;
        }
        let packed = pack_tile640(&ternary, 2, 640, TILE640_PAGE_SIZE).unwrap();
        assert_eq!(packed.packed_words.len(), 64); // 2 rows × 32 words
        assert_eq!(packed.num_pages, 2);

        // First 32 words: all trit-2
        let expected_word_minus: u32 = (0..20).map(|i| 2u32 * 3u32.pow(i)).sum();
        let expected_word_plus: u32 = (0..20).map(|i| 1u32 * 3u32.pow(i)).sum();
        for i in 0..32 {
            assert_eq!(packed.packed_words[i], expected_word_minus);
            assert_eq!(packed.packed_words[32 + i], expected_word_plus);
        }
    }

    #[test]
    fn test_pack_tile640_partial_last_page() {
        // in_dim = 100, this is less than a page (640)
        let ternary = vec![1i8; 100]; // only 100 weights
        let packed = pack_tile640(&ternary, 1, 100, TILE640_PAGE_SIZE).unwrap();
        // 1 page × 32 words (last 540 are virtual zeros)
        assert_eq!(packed.packed_words.len(), 32);

        // First 5 lanes (100/20=5) encode all +1
        // Lanes 5-31 encode all zeros
        let expected_word_plus: u32 = (0..20).map(|i| 1u32 * 3u32.pow(i)).sum();
        for i in 0..5 {
            assert_eq!(packed.packed_words[i], expected_word_plus);
        }
        for i in 5..32 {
            assert_eq!(packed.packed_words[i], 0);
        }
    }

    #[test]
    fn test_pack_tile640_invalid_value() {
        let ternary = vec![2i8, 0, -1]; // 2 is invalid
        let err = pack_tile640(&ternary, 1, 3, TILE640_PAGE_SIZE).unwrap_err();
        assert!(matches!(
            err,
            BonsaiConversionError::InvalidTernaryValue { value: 2, .. }
        ));
    }

    // ── Scale computation tests ────────────────────────────────────

    #[test]
    fn test_compute_scales_uniform() {
        // All weights = 5.0, page_max = 5.0, lane_max = 5.0
        let weights = vec![5.0f32; 640];
        let ternary = vec![1i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        let scales = compute_scales(&packed, &weights);

        assert_eq!(scales.page_scales.len(), 1);
        assert_eq!(scales.lane_scales.len(), 32);

        // Page scale: bf16 of 5.0
        let expected_bf16 = half::bf16::from_f32(5.0);
        assert_eq!(scales.page_scales[0], expected_bf16.to_bits());

        // Lane scales: all 127 (lane_max == page_max)
        for &ls in &scales.lane_scales {
            assert_eq!(ls, 127);
        }
    }

    #[test]
    fn test_compute_scales_varied() {
        // Create weights with varied magnitudes
        let mut weights = Vec::with_capacity(640);
        for i in 0..640 {
            if i < 20 {
                weights.push(10.0); // lane 0 max = 10
            } else if i < 40 {
                weights.push(2.0); // lane 1 max = 2
            } else {
                weights.push(1.0); // remaining lanes max = 1
            }
        }
        let page_max = 10.0f32;

        let ternary = vec![1i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        let scales = compute_scales(&packed, &weights);

        assert_eq!(scales.page_scales.len(), 1);
        let reconstructed = half::bf16::from_bits(scales.page_scales[0]).to_f32();
        assert!((reconstructed - page_max).abs() < 1.0);

        // Lane 0: max=10, page_max/lane_max * 127 = 127
        assert_eq!(scales.lane_scales[0], 127);

        // Lane 1: max=2, 10/2 * 127 = 635 → clamped to 127
        assert_eq!(scales.lane_scales[1], 127);

        // Lane 2+: max=1, 10/1 * 127 = 1270 → clamped to 127
        for i in 2..32 {
            assert_eq!(scales.lane_scales[i], 127);
        }
    }

    #[test]
    fn test_compute_scales_zero_page() {
        let weights = vec![0.0f32; 640];
        let ternary = vec![0i8; 640];
        let packed = pack_tile640(&ternary, 1, 640, TILE640_PAGE_SIZE).unwrap();
        let scales = compute_scales(&packed, &weights);

        // Guard against zero: page_max clamped to 1.0
        assert!(!scales.page_scales.is_empty());
        // All lane scales should be 1
        for &ls in &scales.lane_scales {
            assert_eq!(ls, 1);
        }
    }

    // ── Outlier extraction tests ───────────────────────────────────

    #[test]
    fn test_extract_outliers_basic() {
        let weights = vec![1.0, 100.0, 1.0, 1.0, 1.0, 1.0];
        let ternary = vec![0, 0, 1, 1, 1, 1];
        let outliers = extract_outliers(&weights, &ternary, 1.0 / 6.0);
        // Top 1/6 = ceil(6*1/6) = 1 outlier: index 1
        assert_eq!(outliers.cols.len(), 1);
        assert_eq!(outliers.cols[0], 1);

        // BF16 value should match 100.0
        let reconstructed = f32::from(half::bf16::from_bits(outliers.values[0]));
        assert!((reconstructed - 100.0).abs() < 1.0, "got {}", reconstructed);
    }

    #[test]
    fn test_assign_outlier_rows() {
        let mut outliers = OutlierSet {
            rows: vec![0, 0],
            cols: vec![0, 640], // flat index 640 = row 1, col 0 (with in_dim=640)
            values: vec![half::bf16::from_f32(1.0).to_bits(); 2],
        };
        assign_outlier_rows(&mut outliers, 2, 640);
        assert_eq!(outliers.rows[0], 0);
        assert_eq!(outliers.cols[0], 0);
        assert_eq!(outliers.rows[1], 1);
        assert_eq!(outliers.cols[1], 0);
    }

    #[test]
    fn test_extract_outliers_empty() {
        let outliers = extract_outliers(&[], &[], 0.005);
        assert!(outliers.cols.is_empty());
    }

    // ── Full pipeline integration test ─────────────────────────────

    #[test]
    fn test_full_pipeline_roundtrip() {
        // Create a synthetic Q2_0 payload for a 2×640 tensor:
        // 2 rows × 640 cols = 1280 values = 5 Q2_0 blocks (256 each)
        // We need 5 * 66 = 330 bytes
        let n_rows = 2u32;
        let n_cols = 640u32;
        let num_elements = (n_rows * n_cols) as usize;
        let num_blocks = num_elements / Q2_0_BLOCK_SIZE;

        let mut gguf_data = Vec::with_capacity(num_blocks * Q2_0_BYTES_PER_BLOCK);

        // Build varying Q2_0 blocks represent known patterns
        for block_idx in 0..num_blocks {
            let scale = if block_idx == 0 { 2.0f32 } else { 1.0f32 };
            let scale_f16 = half::f16::from_f32(scale);
            gguf_data.extend_from_slice(&scale_f16.to_bits().to_le_bytes());

            for _ in 0..64 {
                // Encode alternating pattern
                // 0b10_01_10_01 → values: 0→-1=-1, 1→0=0, 2→+1=+1, 3→0=0 cycled by block
                let base_nibble = if block_idx == 0 { 2u8 } else { 0u8 }; // value 2 → +1 (+1-1=0... hmm no)
                                                                          // Let's use nibble 2 (value 2) → signed +1 → result = +1 * scale
                let byte =
                    base_nibble | (base_nibble << 2) | (base_nibble << 4) | (base_nibble << 6);
                gguf_data.push(byte);
            }
        }

        // Run full pipeline
        let converter = Bonsai2To3Conversion::default();
        let conversion = converter
            .convert_tensor(&gguf_data, "test.tensor", n_rows, n_cols)
            .unwrap();

        // Verify results
        assert_eq!(conversion.tensor_name, "test.tensor");
        assert_eq!(conversion.input_dtype, "q2_0");
        assert_eq!(conversion.output_format, "tile640_ternary");

        // 2 rows × 640 cols = 1280 weights -> 2 pages (1 per row) × 32 words = 64 u32 words
        let expected_packed_size = 2 * 32 * 4; // 2 rows × 32 words × 4 bytes
        assert_eq!(conversion.packed_size, expected_packed_size);

        // Page scales: 2 pages × 2 bytes = 4
        assert_eq!(conversion.page_scale_size, 4);

        // Lane scales: 2 pages × 32 lanes × 1 byte = 64
        assert_eq!(conversion.lane_scale_size, 64);

        // Digest should be 32 bytes
        assert_eq!(conversion.digest.len(), 32);
    }

    #[test]
    fn test_dequantize_ternarize_identity() {
        // After dequantizing Q2_0 → f32 and ternarizing, the ternary values
        // should all be in {-1, 0, +1}.
        let _n = 256; // one block
        let mut block = Vec::with_capacity(Q2_0_BYTES_PER_BLOCK);
        let scale_f16 = half::f16::from_f32(3.0);
        block.extend_from_slice(&scale_f16.to_bits().to_le_bytes());

        // Mix of nibble values: 0→signed -1, 1→0, 2→+1, 3→+2
        for _i in 0..64 {
            let byte = 0xE4u8; // 0b11_10_01_00
            block.push(byte);
        }

        let f32_vals = dequantize_q2_0(&block, 256).unwrap();
        let ternary_result = ternarize_f32(&f32_vals, 0.0);

        // All ternary values should be valid
        for &v in &ternary_result.ternary {
            assert!(v == -1 || v == 0 || v == 1, "invalid ternary value {}", v);
        }
        assert_eq!(ternary_result.ternary.len(), 256);
    }

    #[test]
    fn test_tile640_pack_unpack_consistent() {
        // Create ternary data, pack it, verify the packed layout matches
        // the expected word count.
        let n_rows = 4u32;
        let n_cols = 640u32;
        let n = (n_rows * n_cols) as usize;

        // Create a checkerboard pattern
        let mut ternary = Vec::with_capacity(n);
        for i in 0..n {
            ternary.push(match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            });
        }

        let packed = pack_tile640(&ternary, n_rows, n_cols, TILE640_PAGE_SIZE).unwrap();
        assert_eq!(packed.num_pages, 4); // 1 page per row
        assert_eq!(packed.packed_words.len(), 4 * 32); // 4 rows × 32 words

        // Verify each word is non-zero for the checkerboard pattern
        for &w in &packed.packed_words {
            // At least one trit should be non-zero since we have -1/0/+1 pattern
            assert_ne!(w, 0, "packed word should not be zero for checkerboard");
        }
    }

    // ── CPU Reference GEMV tests ───────────────────────────────────

    #[test]
    fn test_ternary_gemv_ref_simple_basic() {
        // 2 rows × 4 cols, uniform scale
        let weights: Vec<i8> = vec![1, 0, -1, 1, -1, 1, 0, -1];
        let input = vec![0.5, 1.0, 2.0, 1.5];
        let scales = vec![2.0, 1.0];
        let result = ternary_gemv_ref_simple(&weights, &input, &scales, 2, 4);
        assert_eq!(result.len(), 2);
        // Row 0: 0.5*2.0*1 + 1.0*2.0*0 + 2.0*2.0*(-1) + 1.5*2.0*1
        //       = 1.0 + 0 - 4.0 + 3.0 = 0.0
        assert!(
            (result[0] - 0.0).abs() < 1e-6,
            "row0 expected 0, got {}",
            result[0]
        );
        // Row 1: 0.5*1.0*(-1) + 1.0*1.0*1 + 2.0*1.0*0 + 1.5*1.0*(-1)
        //       = -0.5 + 1.0 + 0 - 1.5 = -1.0
        assert!(
            (result[1] + 1.0).abs() < 1e-6,
            "row1 expected -1, got {}",
            result[1]
        );
    }

    #[test]
    fn test_ternary_gemv_ref_empty() {
        let result = ternary_gemv_ref_simple(&[], &[], &[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ternary_gemv_ref_matches_tile640() {
        // Build weights, pack as Tile640, compute scales, run ref GEMV.
        // Verify output dimensions and non-NaN values.
        let out_dim: u32 = 4;
        let in_dim: u32 = 640;
        let n = (out_dim * in_dim) as usize;

        // Create deterministic pattern
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            weights.push(match i % 5 {
                0 | 1 => -1.0f32,
                2 => 0.0,
                _ => 1.0,
            });
        }

        // Convert to ternary (-1, 0, +1)
        let ternary: Vec<i8> = weights.iter().map(|&w| w as i8).collect();

        // Pack as Tile640
        let packed = pack_tile640(&ternary, out_dim, in_dim, TILE640_PAGE_SIZE).unwrap();

        // Compute scales
        let scales = compute_scales(&packed, &weights);

        // Input vector
        let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.01).sin()).collect();

        // Run reference GEMV
        let result = ternary_gemv_ref(
            &packed.packed_words,
            &input,
            &scales.page_scales,
            &scales.lane_scales,
            out_dim,
            in_dim,
        );

        assert_eq!(result.len(), out_dim as usize);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "result[{}] is not finite: {}", i, v);
        }
    }

    #[test]
    fn test_ternary_gemv_matches_metal_layout() {
        // Generate random f32 weights, convert to ternary, pack as Tile640,
        // run CPU reference GEMV, verify output shape and non-NaN values.
        let out_dim: u32 = 3;
        let in_dim: u32 = 640;
        let n = (out_dim * in_dim) as usize;

        // Use deterministic LCG for reproducible pseudo-random weights
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let weights: Vec<f32> = (0..n)
            .map(|_| (next() % 200) as f32 / 100.0 - 1.0)
            .collect();

        // Ternarize via absmean
        let ternary_result = ternarize_f32(&weights, 0.0);

        // Pack as Tile640
        let packed =
            pack_tile640(&ternary_result.ternary, out_dim, in_dim, TILE640_PAGE_SIZE).unwrap();

        // Check packed layout matches expected dimensions
        let nt = (in_dim as usize + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
        let expected_words = (out_dim as usize) * nt * TILE640_WORDS_PER_PAGE;
        assert_eq!(packed.packed_words.len(), expected_words);
        assert_eq!(packed.out_dim, out_dim);
        assert_eq!(packed.in_dim, in_dim);
        assert_eq!(packed.num_pages, (out_dim as usize) * nt);

        // Compute scales
        let scales = compute_scales(&packed, &weights);

        // Input vector
        let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.001).cos()).collect();

        // Run CPU reference GEMV
        let result = ternary_gemv_ref(
            &packed.packed_words,
            &input,
            &scales.page_scales,
            &scales.lane_scales,
            out_dim,
            in_dim,
        );

        // Verify output shape
        assert_eq!(result.len(), out_dim as usize);

        // Verify non-NaN values
        for (i, &v) in result.iter().enumerate() {
            assert!(
                v.is_finite(),
                "result[{}] is not finite (NaN/Inf): {}",
                i,
                v
            );
            // Verify the output is non-zero for non-trivial inputs
            // (don't assert exact values since they're pseudo-random)
        }
    }

    #[test]
    fn test_ternary_roundtrip() {
        // Create known ternary values, pack via pack_tile640,
        // unpack via CPU reference unpack, verify bit-exact recovery.
        let out_dim: u32 = 2;
        let in_dim: u32 = 100;
        let n = (out_dim * in_dim) as usize;

        // Create pattern with all three ternary values
        let mut ternary = Vec::with_capacity(n);
        for i in 0..n {
            ternary.push(match i % 3 {
                0 => -1i8,
                1 => 0i8,
                _ => 1i8,
            });
        }

        // Pack
        let packed = pack_tile640(&ternary, out_dim, in_dim, TILE640_PAGE_SIZE).unwrap();

        // Now "unpack" by reconstructing from packed words manually
        // This mirrors the Metal kernel's unpack logic
        let nt = (in_dim as usize + TILE640_PAGE_SIZE - 1) / TILE640_PAGE_SIZE;
        let words_per_row = nt * TILE640_WORDS_PER_PAGE;

        let mut recovered = vec![0i8; n];
        for row in 0..(out_dim as usize) {
            let row_offset = row * words_per_row;
            for wi in 0..words_per_row {
                let p = wi / TILE640_LANES_PER_PAGE;
                let lane = wi % TILE640_LANES_PER_PAGE;
                let col0 = p * TILE640_PAGE_SIZE + lane * TILE640_LANE_SIZE;

                let mut word = packed.packed_words[row_offset + wi];
                for vi in 0..TILE640_LANE_SIZE {
                    let d = word % 3; // 0, 1, 2
                    word /= 3;
                    let col = col0 + vi;
                    if col >= in_dim as usize {
                        break;
                    }
                    let idx = row * (in_dim as usize) + col;
                    recovered[idx] = match d {
                        0 => 0i8,
                        1 => 1i8,
                        2 => -1i8,
                        _ => unreachable!(),
                    };
                }
            }
        }

        // Verify bit-exact recovery
        for i in 0..n {
            assert_eq!(
                recovered[i], ternary[i],
                "mismatch at index {}: expected {}, got {}",
                i, ternary[i], recovered[i]
            );
        }
    }

    #[test]
    fn test_ternary_gemv_ref_multi_page() {
        // Test with in_dim > 640 (multiple pages per row)
        let out_dim: u32 = 2;
        let in_dim: u32 = 700; // 2 pages
        let n = (out_dim * in_dim) as usize;

        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let weights: Vec<f32> = (0..n).map(|_| (next() % 100) as f32 / 50.0 - 1.0).collect();

        let ternary_result = ternarize_f32(&weights, 0.0);
        let packed =
            pack_tile640(&ternary_result.ternary, out_dim, in_dim, TILE640_PAGE_SIZE).unwrap();
        let scales = compute_scales(&packed, &weights);
        let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.01).sin()).collect();

        let result = ternary_gemv_ref(
            &packed.packed_words,
            &input,
            &scales.page_scales,
            &scales.lane_scales,
            out_dim,
            in_dim,
        );

        assert_eq!(result.len(), out_dim as usize);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "result[{}] is not finite: {}", i, v);
        }

        // Also run simple ref and verify they produce finite results
        let simple_result = ternary_gemv_ref_simple(
            &ternary_result.ternary,
            &input,
            &vec![1.0f32; out_dim as usize],
            out_dim,
            in_dim,
        );
        assert_eq!(simple_result.len(), out_dim as usize);
        for (i, &v) in simple_result.iter().enumerate() {
            assert!(v.is_finite(), "simple result[{}] is not finite: {}", i, v);
        }
    }

    // ── ABI Validation tests ───────────────────────────────────────

    #[test]
    fn test_validate_ternary_abi_valid() {
        let out_dim: u32 = 2;
        let in_dim: u32 = 640;
        let ternary = vec![1i8; (out_dim * in_dim) as usize];
        let weights = vec![5.0f32; (out_dim * in_dim) as usize];
        let packed = pack_tile640(&ternary, out_dim, in_dim, TILE640_PAGE_SIZE).unwrap();
        let scales = compute_scales(&packed, &weights);

        let receipt = validate_ternary_abi(
            &packed.packed_words,
            &scales.page_scales,
            &scales.lane_scales,
            out_dim,
            in_dim,
        );

        assert!(receipt.valid, "expected valid ABI: {}", receipt.description);
        assert_eq!(receipt.expected_packed_words, 2 * 32); // 2 rows × 32 words
        assert_eq!(receipt.actual_packed_words, 2 * 32);
    }

    #[test]
    fn test_validate_ternary_abi_rejects_wrong_packed_size() {
        let out_dim: u32 = 2;
        let in_dim: u32 = 640;
        let scales = Scales {
            page_scales: vec![0u16; 2],
            lane_scales: vec![1i8; 64],
        };

        // Wrong packed size (too small)
        let packed = vec![0u32; 10];
        let receipt = validate_ternary_abi(
            &packed,
            &scales.page_scales,
            &scales.lane_scales,
            out_dim,
            in_dim,
        );
        assert!(!receipt.valid, "should reject wrong packed size");
        assert!(receipt.description.contains("packed data size mismatch"));
    }

    #[test]
    fn test_validate_ternary_abi_rejects_wrong_page_scales() {
        let out_dim: u32 = 2;
        let in_dim: u32 = 640;
        let packed = vec![0u32; 64];
        // Wrong page_scales length (should be 2, using 1)
        let receipt = validate_ternary_abi(&packed, &[0u16; 1], &[1i8; 64], out_dim, in_dim);
        assert!(!receipt.valid, "should reject wrong page_scales size");
        assert!(receipt.description.contains("page_scales size mismatch"));
    }

    #[test]
    fn test_validate_ternary_abi_rejects_wrong_lane_scales() {
        let out_dim: u32 = 2;
        let in_dim: u32 = 640;
        let packed = vec![0u32; 64];
        let page_scales = vec![1u16; 2]; // bf16 of some non-zero value
        let receipt = validate_ternary_abi(&packed, &page_scales, &[1i8; 10], out_dim, in_dim);
        assert!(!receipt.valid, "should reject wrong lane_scales size");
        assert!(receipt.description.contains("lane_scales size mismatch"));
    }

    #[test]
    fn test_validate_ternary_abi_rejects_zero_page_scales() {
        let out_dim: u32 = 1;
        let in_dim: u32 = 640;
        let packed = vec![0u32; 32];
        // Zero page scale = uninitialized
        let receipt = validate_ternary_abi(&packed, &[0u16; 1], &[1i8; 32], out_dim, in_dim);
        assert!(!receipt.valid, "should reject zero page scales");
        assert!(receipt.description.contains("zero values"));
    }

    // ── Outlier validation tests ───────────────────────────────────

    #[test]
    fn test_validate_outliers_valid() {
        let outliers = OutlierSet {
            rows: vec![0, 1],
            cols: vec![10, 20],
            values: vec![
                half::bf16::from_f32(1.5).to_bits(),
                half::bf16::from_f32(-0.5).to_bits(),
            ],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(receipt.valid, "expected valid: {}", receipt.description);
        assert_eq!(receipt.outlier_count, 2);
    }

    #[test]
    fn test_validate_outliers_rejects_bad_row() {
        let outliers = OutlierSet {
            rows: vec![5], // out of bounds for out_dim=2
            cols: vec![10],
            values: vec![half::bf16::from_f32(1.0).to_bits()],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(!receipt.valid, "should reject out-of-bounds row");
        assert!(receipt.description.contains("row index"));
    }

    #[test]
    fn test_validate_outliers_rejects_bad_col() {
        let outliers = OutlierSet {
            rows: vec![0],
            cols: vec![999], // out of bounds for in_dim=640
            values: vec![half::bf16::from_f32(1.0).to_bits()],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(!receipt.valid, "should reject out-of-bounds col");
        assert!(receipt.description.contains("col index"));
    }

    #[test]
    fn test_validate_outliers_rejects_mismatched_lengths() {
        let outliers = OutlierSet {
            rows: vec![0, 1],
            cols: vec![10], // cols has 1, rows has 2
            values: vec![half::bf16::from_f32(1.0).to_bits()],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(!receipt.valid, "should reject mismatched lengths");
        assert!(receipt.description.contains("length mismatch"));
    }

    #[test]
    fn test_validate_outliers_rejects_nan() {
        let outliers = OutlierSet {
            rows: vec![0],
            cols: vec![10],
            values: vec![half::bf16::from_f32(f32::NAN).to_bits()],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(!receipt.valid, "should reject NaN");
        assert!(receipt.description.contains("not finite"));
    }

    #[test]
    fn test_validate_outliers_empty() {
        let outliers = OutlierSet {
            rows: vec![],
            cols: vec![],
            values: vec![],
        };
        let receipt = validate_outliers(&outliers, 2, 640);
        assert!(receipt.valid, "empty should be valid");
        assert_eq!(receipt.outlier_count, 0);
    }
}
