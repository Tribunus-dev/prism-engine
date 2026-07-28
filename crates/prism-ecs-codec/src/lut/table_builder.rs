//! Palettized LUT matrix construction utilities.
//!
//! This module owns the canonical authority for the on-disk
//! binary format of palettized matrices used by
//! [`crate::lut::evaluator`] and token embedding lookup. The
//! format is the only contract this module exposes; callers
//! build or parse a [`LutMatrix`] and pass the resulting bytes
//! to the math kernels or the embedding lookup.
//!
//! # Format
//!
//! Each palettized matrix row has:
//! - **Row header**: codebook of 16 × `u16` centroids (32 bytes,
//!   little-endian)
//! - **Index bytes**: packed 4-bit indices, 1 byte per 2 indices
//!   (little-endian)
//!
//! Total row payload = `dim_n / 2` bytes for indices + 32 bytes
//! for codebook.

/// A single row of a palettized LUT matrix.
#[derive(Debug, Clone)]
pub struct LutRow {
    /// 16-entry centroid codebook (FP16 bit patterns).
    pub codebook: [u16; 16],
    /// Packed 4-bit indices into the codebook, `dim / 2` bytes.
    pub indices: Vec<u8>,
}

impl LutRow {
    /// Number of output dimensions (elements) this row
    /// represents.
    pub fn dim(&self) -> usize {
        self.indices.len() * 2
    }

    /// Look up the FP16 value at position `col` (0-indexed).
    pub fn get(&self, col: usize) -> u16 {
        let byte = self.indices[col / 2];
        let nibble = if col % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        self.codebook[nibble as usize]
    }

    /// Serialize this row to the binary format:
    /// `[codebook (32 bytes)][indices (dim/2 bytes)]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(32 + self.indices.len());
        for &c in &self.codebook {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf.extend_from_slice(&self.indices);
        buf
    }
}

/// A complete palettized matrix (multiple rows).
///
/// Rows are stored sequentially: row 0 header + indices, row 1
/// header + indices, etc.
#[derive(Debug, Clone)]
pub struct LutMatrix {
    rows: Vec<LutRow>,
    dim_m: u32,
    dim_n: u32,
}

impl LutMatrix {
    /// Build a LUT matrix from raw centroid + index data.
    ///
    /// `codebooks_per_row` should be
    /// `[codebook_for_row_0, codebook_for_row_1, …]` where each
    /// inner slice has length 16. `indices_per_row` should be
    /// the packed 4-bit indices for each row.
    pub fn new(
        codebooks_per_row: &[[u16; 16]],
        indices_per_row: &[Vec<u8>],
        dim_m: u32,
        dim_n: u32,
    ) -> Self {
        let rows = codebooks_per_row
            .iter()
            .zip(indices_per_row.iter())
            .map(|(cb, idx)| LutRow {
                codebook: *cb,
                indices: idx.clone(),
            })
            .collect();
        LutMatrix { rows, dim_m, dim_n }
    }

    /// Number of rows (output dim M).
    pub fn dim_m(&self) -> u32 {
        self.dim_m
    }

    /// Number of columns (input dim N).
    pub fn dim_n(&self) -> u32 {
        self.dim_n
    }

    /// Access a row by index.
    pub fn row(&self, r: usize) -> Option<&LutRow> {
        self.rows.get(r)
    }

    /// Serialize the full matrix to the binary payload format.
    pub fn to_payload(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(self.rows.len() * (32 + self.dim_n as usize / 2));
        for row in &self.rows {
            payload.extend_from_slice(&row.to_bytes());
        }
        payload
    }

    /// Build a codebook from k-means centroids
    /// (f32 → u16 FP16 bit pattern).
    pub fn centroids_to_codebook(centroids: &[f32; 16]) -> [u16; 16] {
        let mut cb = [0u16; 16];
        for (i, &c) in centroids.iter().enumerate() {
            cb[i] = half::f16::from_f32(c).to_bits();
        }
        cb
    }
}

/// Encode a sequence of 4-bit indices into the packed byte
/// format.
///
/// Each byte holds two indices: low nibble = first index, high
/// nibble = second. Each index must be in `0..=15`.
pub fn pack_indices(indices: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((indices.len() + 1) / 2);
    for chunk in indices.chunks(2) {
        let a = chunk[0] & 0x0F;
        let b = if chunk.len() > 1 {
            (chunk[1] & 0x0F) << 4
        } else {
            0
        };
        packed.push(a | b);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_row_round_trips_indices() {
        let mut row = LutRow {
            codebook: [0u16; 16],
            indices: pack_indices(&[0, 1, 2, 3, 4, 5, 6, 7]),
        };
        for (i, c) in row.codebook.iter_mut().enumerate() {
            *c = half::f16::from_f32(i as f32).to_bits();
        }
        // Each get(i) returns codebook[i]
        for i in 0..8 {
            let got = row.get(i);
            assert_eq!(got, half::f16::from_f32(i as f32).to_bits());
        }
        // dim() is indices.len() * 2 (4 indices → 8 elements)
        assert_eq!(row.dim(), 8);
    }

    #[test]
    fn lut_row_serialization_layout() {
        let row = LutRow {
            codebook: [0x3c00u16; 16],
            indices: vec![0x12, 0x34, 0x56, 0x78],
        };
        let bytes = row.to_bytes();
        // 32 bytes codebook + 4 bytes indices = 36
        assert_eq!(bytes.len(), 36);
        for i in 0..16 {
            let lo = bytes[i * 2];
            let hi = bytes[i * 2 + 1];
            assert_eq!(u16::from_le_bytes([lo, hi]), 0x3c00);
        }
        assert_eq!(&bytes[32..], &[0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn lut_matrix_dimensions_and_rows() {
        let cbs: Vec<[u16; 16]> = vec![[0u16; 16], [0u16; 16], [0u16; 16]];
        let idx: Vec<Vec<u8>> = vec![vec![0u8; 4], vec![0u8; 4], vec![0u8; 4]];
        let m = LutMatrix::new(&cbs, &idx, 3, 8);
        assert_eq!(m.dim_m(), 3);
        assert_eq!(m.dim_n(), 8);
        assert_eq!(m.row(0).map(|r| r.dim()), Some(8));
        assert!(m.row(99).is_none());
    }

    #[test]
    fn lut_matrix_to_payload_is_concatenation() {
        let cbs: Vec<[u16; 16]> = vec![[0x3c00u16; 16], [0x4000u16; 16]];
        let idx: Vec<Vec<u8>> = vec![vec![0xab; 4], vec![0xcd; 4]];
        let m = LutMatrix::new(&cbs, &idx, 2, 8);
        let payload = m.to_payload();
        // 2 rows × (32 + 4) = 72 bytes
        assert_eq!(payload.len(), 72);
        // First row codebook: 16 × 0x3c00
        for i in 0..16 {
            assert_eq!(
                u16::from_le_bytes([payload[i * 2], payload[i * 2 + 1]]),
                0x3c00
            );
        }
        // First row indices: 0xab × 4
        assert_eq!(&payload[32..36], &[0xab; 4]);
        // Second row codebook: 16 × 0x4000
        for i in 0..16 {
            assert_eq!(
                u16::from_le_bytes([payload[36 + i * 2], payload[36 + i * 2 + 1]]),
                0x4000
            );
        }
        // Second row indices: 0xcd × 4
        assert_eq!(&payload[68..72], &[0xcd; 4]);
    }

    #[test]
    fn centroids_to_codebook_matches_half_bits() {
        let centroids: [f32; 16] = std::array::from_fn(|i| i as f32);
        let cb = LutMatrix::centroids_to_codebook(&centroids);
        for (i, &c) in cb.iter().enumerate() {
            assert_eq!(c, half::f16::from_f32(i as f32).to_bits());
        }
    }

    #[test]
    fn pack_indices_low_and_high_nibbles() {
        // [0, 1] → byte 0 = 0x10 (low=0, high=1)
        assert_eq!(pack_indices(&[0, 1]), vec![0x10]);
        // [2, 3, 4] → byte 0 = 0x32 (low=2, high=3), then trailing nibble = 0x04
        assert_eq!(pack_indices(&[2, 3, 4]), vec![0x32, 0x04]);
        // Empty input → empty output
        assert!(pack_indices(&[]).is_empty());
    }
}
