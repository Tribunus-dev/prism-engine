//! LUT (palettized weight table) construction utilities.
//!
//! Builds the on-disk binary format for palettized matrices used by
//! [`crate::lut::evaluator::lut_gemv_cpu`] and token embedding lookup.
//!
//! # Format
//! Each palettized matrix row has:
//! - **Row header**: codebook of 16 × `u16` centroids (32 bytes, little-endian)
//! - **Index bytes**: packed 4-bit indices, 1 byte per 2 indices (little-endian)
//!
//! Total row payload = `dim_n / 2` bytes for indices + 32 bytes for codebook.

/// A single row of a palettized LUT matrix.
#[derive(Debug, Clone)]
pub struct LutRow {
    /// 16-entry centroid codebook (FP16 bit patterns).
    pub codebook: [u16; 16],
    /// Packed 4-bit indices into the codebook, `dim / 2` bytes.
    pub indices: Vec<u8>,
}

impl LutRow {
    /// Number of output dimensions (elements) this row represents.
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
/// Rows are stored sequentially: row 0 header + indices, row 1 header + indices, etc.
#[derive(Debug, Clone)]
pub struct LutMatrix {
    rows: Vec<LutRow>,
    dim_m: u32,
    dim_n: u32,
}

impl LutMatrix {
    /// Build a LUT matrix from raw centroid + index data.
    ///
    /// `codebooks_per_row` should be `[codebook_for_row_0, codebook_for_row_1, …]`
    /// where each inner slice has length 16.
    /// `indices_per_row` should be the packed 4-bit indices for each row.
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

    /// Build a codebook from k-means centroids (f32 → u16 FP16 bit pattern).
    pub fn centroids_to_codebook(centroids: &[f32; 16]) -> [u16; 16] {
        let mut cb = [0u16; 16];
        for (i, &c) in centroids.iter().enumerate() {
            cb[i] = half::f16::from_f32(c).to_bits();
        }
        cb
    }
}

/// Encode a sequence of 4-bit indices into the packed byte format.
///
/// Each byte holds two indices: low nibble = first index, high nibble = second.
/// Each index must be in `0..=15`.
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
