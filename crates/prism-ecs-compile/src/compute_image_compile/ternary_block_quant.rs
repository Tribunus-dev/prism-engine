//! Ternary block quantizer — 4 ternary digits per byte (`ternary_quantize_block`),
//! ANE swizzled-weight generation, and FP16→ternary helpers.
//!
//! Authority: pure 256-element block ternary quantisation. Engine-coupled
//! dispatch lives in the engine's `legacy_compute_image_compile::ternary`.

use crate::compute_image_compile::fp16::fp16_to_f32;
use crate::compute_image_compile::swizzled::{swizzled_byte_offset, swizzled_buffer_size};

/// Quantize a 256-element block of f32 weights to (FP16 scale, 64 packed bytes).
///
/// Each byte holds 4 ternary digits (2 bits each). Trits are 0 (zero), 1 (+1), 2 (-1).
pub fn ternary_quantize_block(block: &[f32; 256]) -> ([u8; 2], [u8; 64]) {
    let max_mag = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_mag > 1e-12 { max_mag } else { 1.0f32 };
    let su = f32_to_fp16_bits(scale);
    let mut nib = [0u8; 64];
    for (i, chk) in block.chunks_exact(4).enumerate() {
        let mut b: u8 = 0;
        for (j, &v) in chk.iter().enumerate() {
            let sn = (v / scale).round().clamp(-1.0, 1.0) as i8;
            b |= (match sn {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            }) << (j * 2);
        }
        nib[i] = b;
    }
    (su.to_le_bytes(), nib)
}

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = (bits >> 23) & 0xFF;
    let m = bits & 0x7FFFFF;
    if e == 0 {
        return s;
    }
    if e == 0xFF {
        return if m == 0 {
            if s != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let ef = e as i32 - 127 + 15;
    if ef >= 0x1F {
        return if s != 0 { 0xFC00 } else { 0x7C00 };
    }
    if ef <= 0 {
        return s;
    }
    s | ((ef as u16) << 10) | ((m >> 13) as u16)
}

/// Generate ANE-planar swizzled u8 weights from raw BF16 input bytes.
pub fn generate_ane_swizzled_weights(raw_bf16: &[u8], out_dim: u32, in_dim: u32) -> Vec<u8> {
    let rows = out_dim as usize;
    let cols = in_dim as usize;
    let total = swizzled_buffer_size(rows, cols);
    if total == 0 {
        return Vec::new();
    }
    let mut swz = vec![0u8; total];
    let mut temp = vec![[0u8; 4]; total];
    let mut cnt = vec![0u8; total];

    let tv = rows * cols;
    let nb = (tv + 255) / 256;
    for bi in 0..nb {
        let st = bi * 256;
        let n = (tv - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < raw_bf16.len() {
                blk[j] = f32::from_bits(
                    (u16::from_le_bytes([raw_bf16[bo], raw_bf16[bo + 1]]) as u32) << 16,
                );
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                0b01 => 1,
                0b10 => 2,
                _ => 0,
            };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / cols, vi % cols, cols);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..total {
        if cnt[b] == 0 {
            continue;
        }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() {
            s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 };
        }
        swz[b] = s;
    }
    swz
}

/// Requantize FP16 KV cache → swizzled u8 ternary format.
///
/// `fp16_kv`: raw FP16 bytes from KV cache (`seq_len * kv_dim * 2` bytes).
/// `slc_buf`: pre-allocated output buffer (size = `swizzled_buffer_size`).
pub fn requantize_kv_to_swizzled_u8(
    fp16_kv: &[u8],
    seq_len: usize,
    kv_dim: usize,
    slc_buf: &mut [u8],
) {
    let total = seq_len * kv_dim;
    let nb = (total + 255) / 256;
    let expected = swizzled_buffer_size(seq_len, kv_dim);
    if slc_buf.len() < expected {
        return;
    }
    slc_buf[..expected].fill(0);

    let mut temp = vec![[0u8; 4]; expected];
    let mut cnt = vec![0u8; expected];

    for bi in 0..nb {
        let st = bi * 256;
        let n = (total - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < fp16_kv.len() {
                let bits = u16::from_le_bytes([fp16_kv[bo], fp16_kv[bo + 1]]);
                blk[j] = fp16_to_f32(bits.to_le_bytes());
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                0b01 => 1u8,
                0b10 => 2u8,
                _ => 0u8,
            };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / kv_dim, vi % kv_dim, kv_dim);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..expected {
        if cnt[b] == 0 {
            continue;
        }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() {
            s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 };
        }
        slc_buf[b] = s;
    }
}
