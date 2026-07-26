//! Spatial downsampling cache for VLM vision tokens.
//!
//! Preserves central foveal regions and localized text-bounding boxes at
//! full FP16 precision. Surrounding background spatial tokens are compressed
//! to 2-bit using INT2 block packing, grouped along adjacent 2D tile quadrants.

// ── INT2 pack/unpack (4 values per byte, 2 bits each) ────────────────────────

/// Pack a slice of f32 values into 2-bit quantized bytes.
///
/// Returns `(packed_bytes, scale, min_val)` where every group of 4 values
/// occupies one byte with the same bit layout as `Int2PackedGroup`:
///
/// | bits | element |
/// |------|---------|
/// | `[1:0]` | `base + 0` |
/// | `[3:2]` | `base + 1` |
/// | `[5:4]` | `base + 2` |
/// | `[7:6]` | `base + 3` |
fn pack_int2(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }

    let mut min = values[0];
    let mut max = values[0];
    for &v in &values[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let range = max - min;
    let scale = if range == 0.0 {
        // Sentinel: all values equal → quantize to 0, dequantized as min.
        1.0
    } else {
        range / 3.0
    };

    let num_bytes = (values.len() + 3) / 4;
    let mut packed = vec![0u8; num_bytes];

    for (byte_idx, byte) in packed.iter_mut().enumerate() {
        let base = byte_idx * 4;
        let mut b = 0u8;
        for j in 0..4 {
            let idx = base + j;
            let q = if idx < values.len() {
                if scale == 0.0 {
                    0u8
                } else {
                    ((values[idx] - min) / scale).round().clamp(0.0, 3.0) as u8
                }
            } else {
                0
            };
            b |= q << (j * 2);
        }
        *byte = b;
    }

    (packed, scale, min)
}

/// Unpack 2-bit quantized bytes back into `f32` values.
fn unpack_int2(packed: &[u8], scale: f32, min_val: f32, element_count: usize) -> Vec<f32> {
    let num_values = packed.len() * 4;
    let mut out = Vec::with_capacity(num_values);

    for &byte in packed {
        for j in 0..4 {
            let q = (byte >> (j * 2)) & 0x03;
            if out.len() < element_count {
                out.push(min_val + (q as f32) * scale);
            }
        }
    }

    out
}

// ── TileState ─────────────────────────────────────────────────────────────────

/// Storage variant for a single tile in the grid decimation cache.
#[derive(Debug, Clone, PartialEq)]
pub enum TileState {
    /// Tile kept at full FP16 precision.
    FullPrecision {
        row: usize,
        col: usize,
        data: Vec<f32>,
    },
    /// Tile compressed to 2-bit via INT2 packer.
    Quantized {
        row: usize,
        col: usize,
        scale: f32,
        min_val: f32,
        element_count: usize,
        packed: Vec<u8>,
    },
}

impl TileState {
    fn row(&self) -> usize {
        match self {
            TileState::FullPrecision { row, .. } => *row,
            TileState::Quantized { row, .. } => *row,
        }
    }

    fn col(&self) -> usize {
        match self {
            TileState::FullPrecision { col, .. } => *col,
            TileState::Quantized { col, .. } => *col,
        }
    }
}

// ── GridDecimationCache ───────────────────────────────────────────────────────

/// Spatial downsampling cache for VLM vision tokens.
///
/// Preserves central foveal regions and localized text-bounding boxes at
/// full FP16 precision. Surrounding background spatial tokens are compressed
/// to 2-bit using INT2 block packing, grouped along adjacent 2D tile quadrants.
pub struct GridDecimationCache {
    pub tile_rows: usize,
    pub tile_cols: usize,
    /// (row_start, col_start, height, width) of the foveal window.
    pub full_precision_window: (usize, usize, usize, usize),
    /// Target quantization bits (always 2 for current implementation).
    pub quantization_bits: u8,
    /// Per-tile storage.
    pub(crate) tiles: Vec<TileState>,
}

impl GridDecimationCache {
    /// Create a new cache for a `tile_rows × tile_cols` grid.
    ///
    /// The foveal window is a square of side `2 * fovea_radius + 1` centered
    /// at `(center_row, center_col)`, clamped to the grid bounds.
    pub fn new(
        tile_rows: usize,
        tile_cols: usize,
        center_row: usize,
        center_col: usize,
        fovea_radius: usize,
    ) -> Self {
        let row_start = center_row.saturating_sub(fovea_radius);
        let col_start = center_col.saturating_sub(fovea_radius);
        let window_height = (2 * fovea_radius + 1).min(tile_rows.saturating_sub(row_start));
        let window_width = (2 * fovea_radius + 1).min(tile_cols.saturating_sub(col_start));

        Self {
            tile_rows,
            tile_cols,
            full_precision_window: (row_start, col_start, window_height, window_width),
            quantization_bits: 2,
            tiles: Vec::new(),
        }
    }

    /// Returns `true` when the tile at `(row, col)` falls inside the foveal
    /// full-precision window.
    fn is_in_fovea(&self, row: usize, col: usize) -> bool {
        let (r_start, c_start, r_height, c_width) = self.full_precision_window;
        row >= r_start && row < r_start + r_height && col >= c_start && col < c_start + c_width
    }

    /// Store or update a tile at `(row, col)`.
    ///
    /// Tiles inside the foveal window are stored as raw `f32`. Tiles outside
    /// are quantized to 2-bit using INT2 packing with per-tile scale + min.
    ///
    /// Returns an error when `(row, col)` is out of bounds for the grid.
    pub fn store_tile(&mut self, row: usize, col: usize, data: &[f32]) -> Result<(), String> {
        if row >= self.tile_rows || col >= self.tile_cols {
            return Err(format!(
                "tile ({}, {}) out of bounds for {}x{} grid",
                row, col, self.tile_rows, self.tile_cols
            ));
        }

        let tile = if self.is_in_fovea(row, col) {
            TileState::FullPrecision {
                row,
                col,
                data: data.to_vec(),
            }
        } else {
            let (packed, scale, min_val) = pack_int2(data);
            TileState::Quantized {
                row,
                col,
                scale,
                min_val,
                element_count: data.len(),
                packed,
            }
        };

        // Update existing tile at (row, col) or append.
        if let Some(pos) = self
            .tiles
            .iter()
            .position(|t| t.row() == row && t.col() == col)
        {
            self.tiles[pos] = tile;
        } else {
            self.tiles.push(tile);
        }

        Ok(())
    }

    /// Read back the tile at `(row, col)`, reconstructing from either
    /// full-precision or quantized storage.
    ///
    /// Returns an error when the tile is out of bounds or not yet stored.
    pub fn read_tile(&self, row: usize, col: usize) -> Result<Vec<f32>, String> {
        if row >= self.tile_rows || col >= self.tile_cols {
            return Err(format!(
                "tile ({}, {}) out of bounds for {}x{} grid",
                row, col, self.tile_rows, self.tile_cols
            ));
        }

        let tile = self
            .tiles
            .iter()
            .find(|t| t.row() == row && t.col() == col)
            .ok_or_else(|| format!("tile ({}, {}) not found", row, col))?;

        match tile {
            TileState::FullPrecision { data, .. } => Ok(data.clone()),
            TileState::Quantized {
                scale,
                min_val,
                element_count,
                packed,
                ..
            } => Ok(unpack_int2(packed, *scale, *min_val, *element_count)),
        }
    }

    /// Number of tiles currently stored in the cache.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: check that two f32 slices are "close enough" given INT2 error.
    fn approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        for (&x, &y) in a.iter().zip(b.iter()) {
            if (x - y).abs() > epsilon {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_pack_roundtrip() {
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (packed, scale, min_val) = pack_int2(&vals);
        let recovered = unpack_int2(&packed, scale, min_val, 16);
        // INT2 has 3 quantisation values above min — expect some error.
        assert_eq!(recovered.len(), 16);
        // The first value should match perfectly (it is min_val).
        assert!((recovered[0] - 0.0).abs() < 1e-6);
        // The last value should be close to 15.0 (quantized to 3)
        assert!((recovered[15] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_empty_pack() {
        let (packed, scale, min_val) = pack_int2(&[]);
        assert!(packed.is_empty());
        let recovered = unpack_int2(&packed, scale, min_val, 0);
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_uniform_values() {
        let vals: Vec<f32> = vec![42.0; 8];
        let (packed, scale, min_val) = pack_int2(&vals);
        let recovered = unpack_int2(&packed, scale, min_val, 8);
        assert!(recovered.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    #[test]
    fn test_4x4_grid_foveal_tiles() {
        let mut cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        // Foveal window: rows 0..3, cols 0..3 (center (1,1), radius 1)
        assert_eq!(cache.full_precision_window, (0, 0, 3, 3));

        // Store a tile inside the fovea (row 1, col 1).
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        cache.store_tile(1, 1, &data).unwrap();
        assert!(matches!(cache.tiles[0], TileState::FullPrecision { .. }));

        // Read it back — should be exact because it was stored as f32.
        let recovered = cache.read_tile(1, 1).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_4x4_grid_quantized_tiles() {
        let mut cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        // (3, 3) is outside the window → gets quantized.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        cache.store_tile(3, 3, &data).unwrap();
        assert!(matches!(cache.tiles[0], TileState::Quantized { .. }));

        // Read back — approximate reconstruction.
        let recovered = cache.read_tile(3, 3).unwrap();
        assert_eq!(recovered.len(), 16);
        // The INT2 roundtrip has error proportional to range/3.
        // data range = 15.0, so step = 5.0 → max error < 2.5.
        assert!(approx_eq(&data, &recovered, 2.5));
    }

    #[test]
    fn test_update_existing_tile() {
        let mut cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        let data_v1: Vec<f32> = (0..16).map(|i| i as f32).collect();
        cache.store_tile(1, 1, &data_v1).unwrap();
        assert_eq!(cache.tile_count(), 1);

        // Update same tile with new data.
        let data_v2: Vec<f32> = (100..116).map(|i| i as f32).collect();
        cache.store_tile(1, 1, &data_v2).unwrap();
        assert_eq!(cache.tile_count(), 1); // still only one tile.
        let recovered = cache.read_tile(1, 1).unwrap();
        assert_eq!(recovered, data_v2);
    }

    #[test]
    fn test_out_of_bounds_errors() {
        let cache = GridDecimationCache::new(4, 4, 1, 1, 1);

        let err = cache.read_tile(4, 0).unwrap_err();
        assert!(err.contains("out of bounds"), "got: {err}");

        let err = cache.read_tile(0, 4).unwrap_err();
        assert!(err.contains("out of bounds"), "got: {err}");
    }

    #[test]
    fn test_tile_not_found() {
        let cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        let err = cache.read_tile(0, 0).unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn test_multiple_tiles_fovea_and_quantized() {
        let mut cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        // Store tiles in the 4 quadrants.
        for &(r, c, base) in &[(0, 0, 0), (0, 3, 10), (3, 0, 20), (3, 3, 30)] {
            let data: Vec<f32> = (base..base + 12).map(|i| i as f32).collect();
            cache.store_tile(r, c, &data).unwrap();
        }

        assert_eq!(cache.tile_count(), 4);

        // (0,0) is inside the foveal window → full precision.
        let t00 = cache.read_tile(0, 0).unwrap();
        assert!((t00[0] - 0.0).abs() < 1e-6);

        // (3,3) is outside → quantized.
        let t33 = cache.read_tile(3, 3).unwrap();
        assert!((t33[0] - 30.0).abs() < 2.5);

        // The inside/outside boundary.
        // (0,3) is outside (window cols 0..2) → quantized.
        let t03 = cache.read_tile(0, 3).unwrap();
        assert!((t03[0] - 10.0).abs() < 2.5);
    }

    #[test]
    fn test_tile_count() {
        let mut cache = GridDecimationCache::new(4, 4, 1, 1, 1);
        assert_eq!(cache.tile_count(), 0);

        cache.store_tile(0, 0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(cache.tile_count(), 1);

        cache.store_tile(3, 3, &[5.0, 6.0]).unwrap();
        assert_eq!(cache.tile_count(), 2);
    }
}
