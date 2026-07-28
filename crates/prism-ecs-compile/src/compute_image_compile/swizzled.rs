//! Block-swizzled ternary repack helpers — pure 16×16 swizzling math used
//! by ANE Planar Engine gather.
//!
//! Authority: swizzling math. Std-only — no engine dependencies.

/// Map linear `(row, col)` → `(byte_offset, shift_within_byte)`.
#[inline(always)]
pub fn swizzled_byte_offset(row: usize, col: usize, width: usize) -> (usize, usize) {
    let bpr = width / 16;
    let br = row / 16;
    let bc = col / 16;
    let bi = br * bpr + bc;
    let ir = row % 16;
    let ic = col % 16;
    let ii = ir * 16 + ic;
    (bi * 64 + ii / 4, ii % 4)
}

/// Size of swizzled u8 buffer for a tensor of `(rows, cols)`.
pub fn swizzled_buffer_size(rows: usize, cols: usize) -> usize {
    ((rows + 15) / 16) * ((cols + 15) / 16) * 64
}

/// Decode a u32 base-3 pack into an array of 20 ternary digits `[0..2]`.
#[inline(always)]
pub fn decode_ternary_u32(packed: u32, digits: &mut [u8; 20]) {
    let mut rem = packed;
    for d in digits.iter_mut() {
        *d = (rem % 3) as u8;
        rem /= 3;
    }
}

/// Re-pack ternary u32 packs from DRAM into 16×16 swizzled u8 in SLC.
///
/// The ternary data uses the tile640 format: u32s at
///   offset = (row × num_tiles × 32 + tile × 32 + lane) × 4
/// Each u32 encodes 20 ternary values in base-3: digit 0→0, 1→+1, 2→-1.
pub fn repack_ternary_to_swizzled_u8(
    ternary_bytes: &[u8],
    rows: usize,
    cols: usize,
    slc_buf: &mut [u8],
    slc_width: usize,
) {
    let expected = swizzled_buffer_size(rows, cols);
    if slc_buf.len() < expected {
        return;
    }
    slc_buf[..expected].fill(0);

    let ts = 640usize;
    let nt = (cols + ts - 1) / ts;

    // Accumulate quartets per SLC byte, then encode once all 4 slots fill
    let mut temp: Vec<[u8; 4]> = vec![[0u8; 4]; expected];
    let mut count: Vec<u8> = vec![0u8; expected];

    for row in 0..rows {
        for t in 0..nt {
            for lane in 0..32 {
                let po = row * nt * 32 * 4 + t * 32 * 4 + lane * 4;
                if po + 4 > ternary_bytes.len() {
                    break;
                }

                let packed = u32::from_le_bytes([
                    ternary_bytes[po],
                    ternary_bytes[po + 1],
                    ternary_bytes[po + 2],
                    ternary_bytes[po + 3],
                ]);

                let mut rem = packed;
                for vi in 0..20 {
                    let col = t * ts + lane * 20 + vi;
                    if col >= cols {
                        break;
                    }

                    let digit = (rem % 3) as u8;
                    rem /= 3;

                    let (byte_off, shift) = swizzled_byte_offset(row, col, slc_width);
                    if byte_off >= expected {
                        continue;
                    }

                    temp[byte_off][shift as usize] = digit;
                    count[byte_off] += 1;
                }
            }
        }
    }

    // Encode fully-filled quartets into base-3 state bytes
    for bi in 0..expected {
        if count[bi] == 4 {
            let q = &temp[bi];
            slc_buf[bi] = q[0] + q[1] * 3 + q[2] * 9 + q[3] * 27;
        } else if count[bi] > 0 {
            // Partial quartet at tensor edge — encode what's filled
            let mut state: u8 = 0;
            for s in (0..4).rev() {
                state = state * 3
                    + if s < count[bi] {
                        temp[bi][s as usize]
                    } else {
                        0
                    };
            }
            slc_buf[bi] = state;
        }
    }
}
