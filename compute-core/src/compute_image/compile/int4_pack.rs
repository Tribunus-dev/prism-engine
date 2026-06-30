#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

//! CPU-side ternary repacker: .cimage (20 trits/u32) → 5 trits/byte (TernaryBlock32) + fused interleave.
//!
//! At load time, the CPU decompresses .cimage ternary weights and repacks them into
//! 5-trits-per-byte format with 32-element block scales. Then fuses 7 matrices per layer
//! into a single contiguous interleaved buffer optimized for cache-line coalescing.
//!
//! TernaryBlock32 (per 32 elements): 7 bytes packed data + 2 bytes FP16 scale = 9 bytes.
//! Fused interleave: [tile0(Q,K,V,O,Gate,Up,Down), tile1(...), ...]

use half;

/// Ternary block of 32 elements: 7 bytes (5 trits/byte) + 2 bytes (FP16 scale) = 9 bytes.
#[repr(C, packed)]
pub struct TernaryBlock32 {
    pub packed_trits: [u8; 7],
    pub block_scale: u16, // f16 bits
}

/// 16-byte aligned version for coalesced GPU uint4 vector loads.
#[repr(C)]
pub struct AlignedTernaryBlock32 {
    pub packed_trits: [u8; 7],
    pub block_scale: u16,
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

/// 16-byte aligned version for coalesced GPU uint4 vector loads.
/// 7 trit bytes + 2 scale bytes + 7 padding = exactly 16 bytes.
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

/// Convert a 32-element slice of f32 weights to one TernaryBlock32.
/// Finds max magnitude → scale → quantize to [−1,0,1] → pack 5/byte.
pub fn quantize_to_ternary_block32(weights: &[f32; 32]) -> TernaryBlock32 {
    let mut max_abs = 0.0f32;
    for &w in weights.iter() { let a = w.abs(); if a > max_abs { max_abs = a; } }
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
            trits[base], trits[base+1], trits[base+2], trits[base+3], trits[base+4]
        ]);
    }
    // Last byte holds 2 trits
    packed[6] = (trits[30] + trits[31] * 3) as u8;

    TernaryBlock32 {
        packed_trits: packed,
        block_scale: half::f16::from_f32(scale).to_bits(),
    }
}

/// Expand a ternary tensor from .cimage format (20 trits/u32) to f32.
/// Each u32 = 20 ternary digits base-3 packed.
pub fn decompress_ternary_u32_tensor(src: &[u32]) -> Vec<f32> {
    let total_weights = src.len() * 20;
    let mut out = vec![0.0f32; total_weights];
    for (i, &val) in src.iter().enumerate() {
        let mut v = val;
        for j in 0..20 {
            let rem = v - ((v as u64 * 2863311531u64) >> 33) as u32 * 3; // fast_mod3
            let wgt = (rem as i32) - 1;
            out[i * 20 + j] = wgt as f32;
            v = (v as u64 * 2863311531u64 >> 33) as u32; // fast_div3
        }
    }
    out
}

/// Repack a .cimage weight tensor to TernaryBlock32 format.
/// Input: &[u32] in 20-trits-per-u32 format (as stored in .cimage).
/// Output: Vec<TernaryBlock32> — one per 32-element block.
pub fn repack_ternary_tensor(src: &[u32]) -> Vec<TernaryBlock32> {
    let f32_vals = decompress_ternary_u32_tensor(src);
    let total_weights = f32_vals.len();
    let num_blocks = (total_weights + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * 32;
        let mut block = [0.0f32; 32];
        for i in 0..32 {
            block[i] = if start + i < total_weights { f32_vals[start + i] } else { 0.0 };
        }
        out.push(quantize_to_ternary_block32(&block));
    }
    out
}

/// Fuse-interleave all 7 weight matrices for one layer.
/// Per layer: Q(3840 rows), K(512), V(512), O(4096), Gate(3840), Up(3840), Down(15360).
/// Fused tile layout: [tile0(Q_180B,K_180B,...,Down_180B), tile1(...), ...]
/// Each sub-tile = 20 TernaryBlock32 × 9 bytes = 180 bytes.
/// Pad each fused tile to 1280 bytes (10 cache lines) for alignment.
///
/// Each input matrix slice is the serialized TernaryBlock32 blocks for that
/// matrix, organized as rows_of_blocks × 20 blocks × 9 bytes.
pub fn interleave_fused_ternary_layer(
    q: &[u8], k: &[u8], v: &[u8], o: &[u8],
    gate: &[u8], up: &[u8], down: &[u8],
    q_rows: usize, kv_rows: usize, o_rows: usize,
    hid_rows: usize, ffn_rows: usize,
) -> Vec<u8> {
    // Each matrix row = TILE/32=20 blocks × 9 bytes = 180 bytes
    let sub_tile = 180usize;
    let matrices: &[(usize, &[u8])] = &[
        (q_rows, q), (kv_rows, k), (kv_rows, v), (o_rows, o),
        (hid_rows, gate), (hid_rows, up), (ffn_rows, down),
    ];
    // Number of tile positions = max rows across all matrices (each row is one tile position)
    let max_tiles = matrices.iter().map(|(r,_)| (*r + 31) / 32).max().unwrap_or(24);

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

/// Repack all 48 layers of .cimage weights to fused ternary format.
pub fn repack_all_layers_fused(_ternary_src: &[u8]) -> Vec<u8> {
    // Read .cimage ternary buffer (layers × 7 matrices in the old offset layout)
    // Re-layout to fused ternary
    unimplemented!() // caller knows the layer offsets — this is called with per-layer slices
}

// ── Tests ──────────────────────────────────────────────────────────────

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
        // All +1 → digit 2. Packed should be: each byte = 242 (2 * 3^4 + 2 * 3^3 + ...)
        // For 5 trits all = 2: 2 + 2*3 + 2*9 + 2*27 + 2*81 = 2+6+18+54+162 = 242
        let first_byte = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(block.packed_trits[0])) };
        assert_eq!(first_byte, 242, "first byte should be 242");
        // Scale should be f16 of 1.0
        let scale = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(block.block_scale)) };
        assert_eq!(scale, half::f16::from_f32(1.0).to_bits());
    }

    #[test]
    fn test_decompress_ternary_u32_tensor() {
        // Single u32 with 20 values all = 0 (ternary -1)
        // Each digit = 0 in base-3 means packed value = 0
        let src = [0u32];
        let f32_vals = decompress_ternary_u32_tensor(&src);
        assert_eq!(f32_vals.len(), 20);
        for (i, &v) in f32_vals.iter().enumerate() {
            assert_eq!(v, -1.0, "weight {} should be -1.0, got {}", i, v);
        }
    }

    #[test]
    fn test_decompress_mixed_values() {
        // Pack digits [0,1,2,0,1,2,...] into one u32: base-3 encoding
        // digit[0]*3^0 + digit[1]*3^1 + digit[2]*3^2 + ...
        // digits = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1]
        let digits: [u32; 20] = [
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
            1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
        ];
        let mut val = 0u32;
        let mut mul = 1u32;
        for i in 0..20 {
            val += digits[i] * mul;
            mul *= 3;
        }

        let src = [val];
        let f32_vals = decompress_ternary_u32_tensor(&src);
        assert_eq!(f32_vals.len(), 20);
        // digit 0 → -1, digit 1 → 0, digit 2 → +1
        for (i, &v) in f32_vals.iter().enumerate() {
            let expected: f32 = match digits[i] {
                0 => -1.0,
                1 => 0.0,
                2 => 1.0,
                _ => unreachable!(),
            };
            assert!((v - expected).abs() < 1e-6, "weight {}: expected {} got {}", i, expected, v);
        }
    }

    #[test]
    fn test_repack_roundtrip_via_decompress() {
        // Pack digits that decode to [-1,0,1] repeated
        let digits: [u32; 20] = [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1];
        let mut val = 0u32;
        let mut mul = 1u32;
        for i in 0..20 {
            val += digits[i] * mul;
            mul *= 3;
        }

        let src = vec![val; 33]; // 33 u32 = 660 weights = enough for 21 blocks
        let blocks = repack_ternary_tensor(&src);
        // 660 / 32 = 20.625 → 21 blocks
        assert_eq!(blocks.len(), 21);

        // Verify block scale is reasonable
        for block in &blocks {
            let scale_bits = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!(block.block_scale)) };
            let scale = half::f16::from_bits(scale_bits).to_f32();
            assert!(scale > 0.0, "scale should be positive, got {}", scale);
        }
    }

    #[test]
    fn test_interleave_fused_ternary_layer() {
        // Build minimal test data: 1 row per matrix (1 row = 20 blocks × 9 = 180 bytes)
        let rows = [
            (32usize, 180 * 32),  // q: 1 row
            (32, 180 * 32),       // k: 1 row
            (32, 180 * 32),       // v: 1 row
            (32, 180 * 32),       // o: 1 row
            (32, 180 * 32),       // gate: 1 row
            (32, 180 * 32),       // up: 1 row
            (32, 180 * 32),       // down: 1 row
        ];

        let make_data = |len: usize, fill: u8| -> Vec<u8> { vec![fill; len] };

        let q = make_data(rows[0].1, 1);
        let k = make_data(rows[1].1, 2);
        let v = make_data(rows[2].1, 3);
        let o = make_data(rows[3].1, 4);
        let gate = make_data(rows[4].1, 5);
        let up = make_data(rows[5].1, 6);
        let down = make_data(rows[6].1, 7);

        let fused = interleave_fused_ternary_layer(
            &q, &k, &v, &o, &gate, &up, &down,
            32, 32, 32, 32, 32,
        );

        // 1 tile position × 7 matrices × 180 bytes = 1260 bytes
        assert_eq!(fused.len(), 1260);

        // Verify first byte of each sub-tile
        assert_eq!(fused[0], 1, "Q first byte");
        assert_eq!(fused[180], 2, "K first byte");
        assert_eq!(fused[360], 3, "V first byte");
        assert_eq!(fused[540], 4, "O first byte");
        assert_eq!(fused[720], 5, "Gate first byte");
        assert_eq!(fused[900], 6, "Up first byte");
        assert_eq!(fused[1080], 7, "Down first byte");
    }

    #[test]
    fn test_interleave_asymmetric_matrices() {
        // Q has 2 rows, others have 1 row (64 vs 32 elements)
        let q = vec![0xAAu8; 360];    // 2 rows of 180
        let k = vec![0xBBu8; 180];    // 1 row
        let v = vec![0xCCu8; 180];
        let o = vec![0xDDu8; 360];    // 2 rows
        let gate = vec![0xEEu8; 180];
        let up = vec![0xFFu8; 180];
        let down = vec![0x00u8; 180];

        let fused = interleave_fused_ternary_layer(
            &q, &k, &v, &o, &gate, &up, &down,
            64, 32, 64, 32, 32,
        );

        // max tiles = max(ceil(64/32), ceil(32/32), ...) = max(2,1,1,2,1,1,1) = 2
        // Each tile = 1260 bytes → 2520 total
        assert_eq!(fused.len(), 2520);

        // Tile 0: all matrices present
        assert_eq!(fused[0], 0xAA);
        assert_eq!(fused[180], 0xBB);
        assert_eq!(fused[1260], 0xAA); // Q row 1 at tile 1, Q sub-tile start

        // Tile 1: K,V,Gate,Up,Down should be zeros (only 1 row each)
        for off in [180, 360, 720, 900, 1080] {
            assert_eq!(fused[1260 + off], 0, "matrix with 1 row should be zero-padded at tile 1");
        }
    }
}
