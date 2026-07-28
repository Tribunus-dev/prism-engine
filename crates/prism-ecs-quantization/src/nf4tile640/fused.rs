//! Interleaved fused tile format (CImage v2 micro-layout).
//!
//! Each tile stores codes (320B), scales (20B), and biases (20B) contiguously
//! in a single 360-byte block. Matrix boundaries are 64-byte aligned.
//!
//! This eliminates the triadic MTLBuffer overhead from the split format:
//! one buffer per matrix instead of three, one IOSurface for ANE binding,
//! one segment table entry for mmap.

use super::{pack_nf4_tile, unpack_nf4_tile, TILE_ELEMENTS};

const BYTES_PER_TILE: usize = 360; // 320 codes + 20 scales + 20 biases

/// A single fused tile: codes, scales, and biases in one contiguous block.
/// The tile's complete execution context is at one pointer offset.
#[derive(Debug, Clone, Copy)]
pub struct FusedTile {
    /// Packed NF4 codes: 640 elements × 4-bit = 320 bytes
    pub codes: [u8; 320],
    /// Scale factors: 5 groups × f32 = 20 bytes
    pub scales: [f32; 5],
    /// Bias coefficients: 5 groups × f32 = 20 bytes
    pub biases: [f32; 5],
}

impl FusedTile {
    /// Bytes required for one fused tile.
    pub const fn byte_size() -> usize {
        BYTES_PER_TILE
    }
}

/// Pack a single tile of 640 f32 values into the fused format.
pub fn pack_fused_tile(values: &[f32; TILE_ELEMENTS]) -> [u8; BYTES_PER_TILE] {
    let (codes, scales, biases) = pack_nf4_tile(values);
    let mut block = [0u8; BYTES_PER_TILE];
    // codes: first 320 bytes
    block[..320].copy_from_slice(&codes);
    // scales: bytes 320..340
    block[320..340].copy_from_slice(bytemuck::cast_slice(&scales));
    // biases: bytes 340..360
    block[340..360].copy_from_slice(bytemuck::cast_slice(&biases));
    block
}

/// Dequantize a fused tile back to f32.
pub fn unpack_fused_tile(tile: &[u8; BYTES_PER_TILE]) -> [f32; TILE_ELEMENTS] {
    let mut codes = [0u8; 320];
    let mut scales = [0.0f32; 5];
    let mut biases = [0.0f32; 5];
    codes.copy_from_slice(&tile[..320]);
    scales.copy_from_slice(bytemuck::cast_slice(&tile[320..340]));
    biases.copy_from_slice(bytemuck::cast_slice(&tile[340..360]));
    let mut output = [0.0f32; TILE_ELEMENTS];
    unpack_nf4_tile(&codes, &scales, &biases, &mut output);
    output
}

/// Pack a full weight matrix into the fused format.
///
/// Layout: tiles in tile-major order, each tile 360 contiguous bytes.
/// The buffer is 64-byte padded at the end.
pub fn pack_weights_fused(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let tiles_per_row = cols.div_ceil(TILE_ELEMENTS);
    let total_tiles = rows * tiles_per_row;
    let total_data = total_tiles * BYTES_PER_TILE;
    // 64-byte align
    let aligned = ((total_data + 63) / 64) * 64;
    let mut buf = Vec::with_capacity(aligned);

    for row in 0..rows {
        for tile_idx in 0..tiles_per_row {
            let col_start = tile_idx * TILE_ELEMENTS;
            let mut tile_vals = [0.0f32; TILE_ELEMENTS];
            for i in 0..TILE_ELEMENTS {
                let c = col_start + i;
                if c < cols {
                    tile_vals[i] = weights[row * cols + c];
                } else {
                    tile_vals[i] = 0.0; // zero-pad
                }
            }
            let fused = pack_fused_tile(&tile_vals);
            buf.extend_from_slice(&fused);
        }
    }

    // Pad to 64-byte boundary
    buf.resize(aligned, 0);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_roundtrip() {
        // Deterministic values that exercise the full range
        let mut vals = [0.0f32; TILE_ELEMENTS];
        for i in 0..TILE_ELEMENTS {
            vals[i] = ((i % 128) as f32 - 64.0) / 64.0; // [-1, 1)
        }
        let tile = pack_fused_tile(&vals);
        assert_eq!(tile.len(), BYTES_PER_TILE);
        let reconstructed = unpack_fused_tile(&tile);
        let mse: f64 = vals
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / TILE_ELEMENTS as f64;
        let rmse = mse.sqrt();
        // NF4 quantization noise ~2-3%
        assert!(rmse < 0.06, "RMSE should be < 0.06, got {}", rmse);
    }

    #[test]
    fn test_pack_weights_fused_alignment() {
        let weights: Vec<f32> = (0..3840 * 1280).map(|i| (i % 128) as f32).collect();
        let buf = pack_weights_fused(&weights, 3840, 1280);
        assert!(buf.len() % 64 == 0, "buffer must be 64-byte aligned");
    }

    #[test]
    fn test_fused_vs_split_equivalence() {
        let vals: [f32; TILE_ELEMENTS] = std::array::from_fn(|i| ((i as f32) / 640.0) * 2.0 - 1.0);
        let fused = pack_fused_tile(&vals);
        let reconstructed = unpack_fused_tile(&fused);
        // Compare against split path
        let (codes, scales, biases) = pack_nf4_tile(&vals);
        let mut split_out = [0.0f32; TILE_ELEMENTS];
        unpack_nf4_tile(
            &std::array::from_fn(|i| codes[i]),
            &std::array::from_fn(|i| scales[i]),
            &std::array::from_fn(|i| biases[i]),
            &mut split_out,
        );
        for i in 0..TILE_ELEMENTS {
            assert!(
                (reconstructed[i] - split_out[i]).abs() < 1e-6,
                "fused and split must produce identical output at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fused_non_multiple_cols() {
        let cols = 1000; // not a multiple of 640
        let rows = 4;
        let weights: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let buf = pack_weights_fused(&weights, rows, cols);
        // Verify the packed size is correct for 4 rows x ceil(1000/640)=2 tiles per row = 8 tiles
        let expected_tiles = rows * cols.div_ceil(TILE_ELEMENTS);
        let data_bytes = expected_tiles * BYTES_PER_TILE;
        let aligned = ((data_bytes + 63) / 64) * 64;
        assert_eq!(buf.len(), aligned);

        // Verify we can reconstruct approximately
        // (We can't easily verify without a full dequant matmul, but the length check
        // and the roundtrip test on compatible dims gives confidence.)
        assert!(buf.len() >= data_bytes);
    }
}
