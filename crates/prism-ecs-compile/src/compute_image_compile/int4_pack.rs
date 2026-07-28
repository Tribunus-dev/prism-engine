//! CPU-side ternary repacker: `.cimage` (20 trits/u32) → 5 trits/byte
//! (`TernaryBlock32`) + fused interleave.
//!
//! At load time, the CPU decompresses `.cimage` ternary weights and repacks
//! them into a 5-trits-per-byte format with 32-element block scales. Then
//! it fuses 7 matrices per layer into a single contiguous interleaved
//! buffer optimised for cache-line coalescing.
//!
//! Authority: pure std-only repacker math. Uses the `half` crate for FP16
//! conversion. Engine-coupled dispatch lives in
//! `legacy_compute_image_compile::int4_pack`.

/// Ternary block of 32 elements: 7 bytes (5 trits/byte) + 2 bytes (FP16 scale) = 9 bytes.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TernaryBlock32 {
    /// 5-trits-per-byte packed payload (32 trits / 5 = 7 bytes, last byte partial).
    pub packed_trits: [u8; 7],
    /// FP16 block scale bits.
    pub block_scale: u16,
}

/// 16-byte aligned version for coalesced GPU uint4 vector loads.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AlignedTernaryBlock32 {
    /// 5-trits-per-byte packed payload.
    pub packed_trits: [u8; 7],
    /// FP16 block scale bits.
    pub block_scale: u16,
    /// Padding to reach 16 bytes total.
    pub padding: [u8; 7],
}

impl From<TernaryBlock32> for AlignedTernaryBlock32 {
    fn from(b: TernaryBlock32) -> Self {
        AlignedTernaryBlock32 {
            packed_trits: b.packed_trits,
            block_scale: b.block_scale,
            padding: [0u8; 7],
        }
    }
}

/// Unpack 5 ternary digits (0, 1, 2) from one byte into `out`.
///
/// `out[0]` is the least-significant trit.
pub fn unpack_byte_5_trits(byte: u8, out: &mut [u8; 5]) {
    let mut v = byte as u32;
    for i in 0..5 {
        let q = (v * 171) >> 9; // fast_div3 for u8 (v < 256, so shift by 9)
        out[i] = (v - q * 3) as u8;
        v = q;
    }
}

/// Pack 5 ternary digits (0,1,2) into one byte.
pub fn pack_5_trits(digits: &[u8; 5]) -> u8 {
    let mut val = 0u32;
    let mut mul = 1u32;
    for i in 0..5 {
        val += (digits[i] as u32) * mul;
        mul *= 3;
    }
    val as u8
}

/// Convert a 32-element slice of f32 weights to one `TernaryBlock32`.
///
/// Finds the max magnitude → uses it as scale → quantises to [-1, 0, 1] →
/// packs 5 trits per byte (with the 7th byte holding 2 trits).
pub fn quantize_to_ternary_block32(weights: &[f32; 32]) -> TernaryBlock32 {
    let mut max_abs = 0.0f32;
    for &w in weights.iter() {
        let a = w.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = if max_abs > 1e-7 { max_abs } else { 1.0f32 };
    let inv = 1.0 / scale;

    let mut trits = [0u8; 32];
    for i in 0..32 {
        let q = (weights[i] * inv).round() as i32;
        let c = q.clamp(-1, 1);
        trits[i] = (c + 1) as u8; // -1→0, 0→1, +1→2
    }

    let mut packed = [0u8; 7];
    for byte_idx in 0..6 {
        let base = byte_idx * 5;
        packed[byte_idx] = pack_5_trits(&[
            trits[base],
            trits[base + 1],
            trits[base + 2],
            trits[base + 3],
            trits[base + 4],
        ]);
    }
    // Last byte holds 2 trits.
    packed[6] = (trits[30] + trits[31] * 3) as u8;

    TernaryBlock32 {
        packed_trits: packed,
        block_scale: half::f16::from_f32(scale).to_bits(),
    }
}

/// Expand a ternary tensor from `.cimage` format (20 trits/u32) to f32.
///
/// Each u32 = 20 ternary digits base-3 packed.
pub fn decompress_ternary_u32_tensor(src: &[u32]) -> Vec<f32> {
    let total_weights = src.len() * 20;
    let mut out = vec![0.0f32; total_weights];
    for (i, &val) in src.iter().enumerate() {
        let mut v = val;
        for j in 0..20 {
            let rem = v.wrapping_sub(((v as u64 * 2863311531u64) >> 33) as u32 * 3); // fast_mod3
            let wgt = (rem as i32) - 1;
            out[i * 20 + j] = wgt as f32;
            v = ((v as u64 * 2863311531u64) >> 33) as u32; // fast_div3
        }
    }
    out
}

/// Repack a `.cimage` weight tensor to `TernaryBlock32` format.
///
/// Input: `&[u32]` in 20-trits-per-u32 format (as stored in `.cimage`).
/// Output: `Vec<TernaryBlock32>` — one per 32-element block.
pub fn repack_ternary_tensor(src: &[u32]) -> Vec<TernaryBlock32> {
    let f32_vals = decompress_ternary_u32_tensor(src);
    let total_weights = f32_vals.len();
    let num_blocks = (total_weights + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * 32;
        let mut block = [0.0f32; 32];
        for i in 0..32 {
            block[i] = if start + i < total_weights {
                f32_vals[start + i]
            } else {
                0.0
            };
        }
        out.push(quantize_to_ternary_block32(&block));
    }
    out
}

/// Fuse-interleave all 7 weight matrices for one layer.
///
/// Each input matrix slice is the serialised `TernaryBlock32` blocks for
/// that matrix, organised as `rows_of_blocks × 20 blocks × 9 bytes`.
#[allow(clippy::too_many_arguments)]
pub fn interleave_fused_ternary_layer(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    o: &[u8],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    q_rows: usize,
    kv_rows: usize,
    o_rows: usize,
    hid_rows: usize,
    ffn_rows: usize,
) -> Vec<u8> {
    // Each matrix row = TILE/32=20 blocks × 9 bytes = 180 bytes
    let sub_tile = 180usize;
    let matrices: &[(usize, &[u8])] = &[
        (q_rows, q),
        (kv_rows, k),
        (kv_rows, v),
        (o_rows, o),
        (hid_rows, gate),
        (hid_rows, up),
        (ffn_rows, down),
    ];
    // Number of tile positions = max rows across all matrices (each row is one tile position)
    let max_tiles = matrices
        .iter()
        .map(|(r, _)| (*r + 31) / 32)
        .max()
        .unwrap_or(24);

    let fused_tile_bytes = 7 * sub_tile; // 1260 bytes
    let mut fused = vec![0u8; max_tiles * fused_tile_bytes];

    for t in 0..max_tiles {
        let tile_base = t * fused_tile_bytes;
        for (m_idx, (rows, data)) in matrices.iter().enumerate() {
            if t < (*rows + 31) / 32 {
                let src_start = t * sub_tile;
                let dst_start = tile_base + m_idx * sub_tile;
                if src_start + sub_tile <= data.len() {
                    fused[dst_start..dst_start + sub_tile]
                        .copy_from_slice(&data[src_start..src_start + sub_tile]);
                }
            }
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_5_trits_roundtrip() {
        let test_cases = [
            [0u8, 0, 0, 0, 0],
            [1u8, 0, 0, 0, 0],
            [2u8, 2, 2, 2, 2],
            [2u8, 1, 0, 1, 2],
            [2u8, 2, 2, 2, 1],
        ];
        for &digits in &test_cases {
            let packed = pack_5_trits(&digits);
            let mut unpacked = [0u8; 5];
            unpack_byte_5_trits(packed, &mut unpacked);
            assert_eq!(digits, unpacked, "round-trip failed for {:?}", digits);
        }
    }

    #[test]
    fn test_quantize_all_positive() {
        let weights = [1.0f32; 32];
        let block = quantize_to_ternary_block32(&weights);
        let first_byte =
            unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(block.packed_trits[0])) };
        assert_eq!(first_byte, 242, "first byte should be 242");
        let scale = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(block.block_scale)) };
        assert_eq!(scale, half::f16::from_f32(1.0).to_bits());
    }

    #[test]
    fn test_decompress_ternary_u32_tensor() {
        let src = [0u32];
        let f32_vals = decompress_ternary_u32_tensor(&src);
        assert_eq!(f32_vals.len(), 20);
        for (i, &v) in f32_vals.iter().enumerate() {
            assert_eq!(v, -1.0, "weight {} should be -1.0, got {}", i, v);
        }
    }
}
