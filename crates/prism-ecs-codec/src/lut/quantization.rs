//! INT8 KV cache and ternary weight quantization helpers.
//!
//! This module owns the canonical authority for symmetric
//! per-token INT8 quantization of FP16 vectors (used by the
//! per-token KV cache format) and the 2-bit ternary packing of
//! `{-1, 0, +1}` weight values. The byte format is the only
//! contract this module exposes; all callers consume
//! quantized bytes and validate the format themselves.
//!
//! # Per-token INT8 layout
//!
//! Each token block is `kv_dim + 4` bytes:
//! - bytes `[0..4)` — `f32` scale in little-endian
//! - bytes `[4..)` — `i8` quantized values (padded to `kv_dim`),
//!   stored as `u8`
//!
//! # Ternary packing
//!
//! Each `f32` input maps to a 2-bit value:
//! - values >= 0.5 → `0b01` (ternary +1)
//! - values <= -0.5 → `0b10` (ternary -1)
//! - values between -0.5 and 0.5 → `0b00` (ternary 0)
//!
//! 4 values pack per byte; the lowest-index element occupies the
//! lowest 2 bits of the byte.

/// Dequantize a contiguous INT8 KV cache block back to FP16.
///
/// See module docs for the on-disk layout. Returns an empty
/// vector if the buffer is too small to contain a single
/// `kv_dim + 4` token block.
pub fn dequant_inline(data: &[u8], kv_dim: usize) -> Vec<u16> {
    let ts = kv_dim + 4;
    if data.len() < ts {
        return Vec::new();
    }
    let nt = data.len() / ts;
    let mut out = Vec::with_capacity(nt * kv_dim);
    for t in 0..nt {
        let o = t * ts;
        let s = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        for j in 0..kv_dim {
            out.push(
                half::f16::from_f32(((data[o + 4 + j] as i8) as f32) * (1.0 / s)).to_bits(),
            );
        }
    }
    out
}

/// Symmetric per-token INT8 quantization:
/// `data → [scale: f32 LE][i8 × n]`.
///
/// The scale is `127 / max_abs(data)` so quantized values fill
/// `[-127, 127]`. Returns an empty vector for empty input.
pub fn quantize_token(data: &[u16]) -> Vec<u8> {
    let token_size = data.len() + 4;
    let mut out = Vec::with_capacity(token_size);
    if data.is_empty() {
        return out;
    }
    let max_abs = data
        .iter()
        .fold(0.0f32, |a, &v| a.max(half::f16::from_bits(v).to_f32().abs()));
    let scale = if max_abs > 1e-10 { 127.0 / max_abs } else { 1.0 };
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in data {
        let f = half::f16::from_bits(v).to_f32();
        out.push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8);
    }
    out
}

/// Pack ternary weights (-1, 0, +1) into a compact 2-bit
/// representation. See module docs for the threshold rules.
///
/// Returns 4 f32 values packed per byte (2 bits each). The
/// output length is `(data.len() + 3) / 4` bytes.
pub fn pack_ternary_weights(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; (data.len() + 3) / 4];
    for (i, &v) in data.iter().enumerate() {
        let bit = if v >= 0.5 {
            0b01
        } else if v <= -0.5 {
            0b10
        } else {
            0b00
        };
        let byte_idx = i / 4;
        let shift = (i % 4) * 2;
        out[byte_idx] |= bit << shift;
    }
    out
}

/// Extract the f32 scale from the first 4 bytes (little-endian)
/// of a data slice. Returns `1.0` if the buffer is shorter than
/// 4 bytes. Used for INT8 KV cache blocks where the first 4
/// bytes encode the scale.
pub fn extract_scale(data: &[u8]) -> f32 {
    if data.len() < 4 {
        return 1.0;
    }
    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrip_preserves_values() {
        let original: Vec<u16> = [1.0f32, 2.5, -3.0, 0.5]
            .iter()
            .map(|&f| half::f16::from_f32(f).to_bits())
            .collect();
        let quantized = quantize_token(&original);
        let dequantized = dequant_inline(&quantized, 4);
        assert_eq!(dequantized.len(), 4);
        for (o, d) in original.iter().zip(dequantized.iter()) {
            let of = half::f16::from_bits(*o).to_f32();
            let df = half::f16::from_bits(*d).to_f32();
            assert!(
                (of - df).abs() < 0.02,
                "quant error: orig={} deq={}",
                of,
                df
            );
        }
    }

    #[test]
    fn dequant_inline_returns_empty_for_short_input() {
        assert_eq!(dequant_inline(&[], 4).len(), 0);
        assert_eq!(dequant_inline(&[0u8; 3], 4).len(), 0);
    }

    #[test]
    fn quantize_token_returns_empty_for_empty_input() {
        let out = quantize_token(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn pack_ternary_weights_matches_bit_layout() {
        // 1.0 (+1) → 0b01, 0.0 → 0b00, -1.0 → 0b10, 0.3 → 0b00
        // packed byte 0 = 0b00_10_00_01 = 0x21
        let data = vec![1.0, 0.0, -1.0, 0.3, -0.7];
        let packed = pack_ternary_weights(&data);
        assert_eq!(packed[0], 0b00_10_00_01);
        // -0.7 → 0b10 in lowest 2 bits of byte 1
        assert_eq!(packed[1], 0b10);
    }

    #[test]
    fn extract_scale_reads_le_f32() {
        let scale = 127.0f32;
        let mut bytes = scale.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[1u8; 10]);
        assert!((extract_scale(&bytes) - 127.0).abs() < 1e-6);
        assert!((extract_scale(&[]) - 1.0).abs() < 1e-6);
        assert!((extract_scale(&[0, 0, 0]) - 1.0).abs() < 1e-6);
    }
}
