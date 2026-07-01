//! Quantization helpers for INT8 KV cache and weight dequantization.
//!
//! Provides symmetric per-token INT8 quantization and dequantization
//! for KV cache entries stored as scaled `i8` byte arrays.

/// Dequantize a contiguous INT8 KV cache block back to FP16.
///
/// # Format
/// Each token block is `kv_dim + 4` bytes:
/// - bytes `[0..4)` — `f32` scale in little-endian
/// - bytes `[4..)` — `i8` quantized values (padded to `kv_dim`), stored as `u8`
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
            out.push(half::f16::from_f32(((data[o + 4 + j] as i8) as f32) * (1.0 / s)).to_bits());
        }
    }
    out
}

/// Symmetric per-token INT8 quantization: `data → [scale: f32 LE][i8 × n]`.
///
/// The scale is `127 / max_abs(data)` so quantized values fill `[-127, 127]`.
pub fn quantize_token(data: &[u16]) -> Vec<u8> {
    let token_size = data.len() + 4;
    let mut out = Vec::with_capacity(token_size);
    if data.is_empty() {
        return out;
    }
    let max_abs = data.iter().fold(0.0f32, |a, &v| {
        a.max(half::f16::from_bits(v).to_f32().abs())
    });
    let scale = if max_abs > 1e-10 {
        127.0 / max_abs
    } else {
        1.0
    };
    out.extend_from_slice(&scale.to_le_bytes());
    for &v in data {
        let f = half::f16::from_bits(v).to_f32();
        out.push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8);
    }
    out
}

/// Pack ternary weights (-1, 0, +1) into a compact 2-bit representation.
///
/// Each f32 input maps to a 2-bit value:
/// - values <= -0.5 → 0b10 (ternary -1)
/// - values between -0.5 and 0.5 → 0b00 (ternary 0)
/// - values >= 0.5 → 0b01 (ternary +1)
///
/// Returns 4 f32 values packed per byte (2 bits each).
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

/// Extract the f32 scale from the first 4 bytes (little-endian) of a data slice.
///
/// Used for INT8 KV cache blocks where the first 4 bytes encode the scale.
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
    fn test_quantize_roundtrip() {
        let original: Vec<u16> = [1.0, 2.5, -3.0, 0.5]
            .iter()
            .map(|&f| half::f16::from_f32(f).to_bits())
            .collect();
        let quantized = quantize_token(&original);
        let dequantized = dequant_inline(&quantized, 4);
        assert_eq!(dequantized.len(), 4);
        for (o, d) in original.iter().zip(dequantized.iter()) {
            let of = half::f16::from_bits(*o).to_f32();
            let df = half::f16::from_bits(*d).to_f32();
            assert!((of - df).abs() < 0.02, "quant error: orig={} deq={}", of, df);
        }
    }

    #[test]
    fn test_dequant_empty() {
        assert_eq!(dequant_inline(&[], 4).len(), 0);
        assert_eq!(dequant_inline(&[0u8; 3], 4).len(), 0);
    }

    #[test]
    fn test_pack_ternary_weights() {
        let data = vec![1.0, 0.0, -1.0, 0.3, -0.7];
        let packed = pack_ternary_weights(&data);
        // First byte: 01|00|10|00 (bit order within each 2-bit group: LSB first)
        // bit 0-1: 1.0 => 01 = 0x01
        // bit 2-3: 0.0 => 00 = 0x00
        // bit 4-5: -1.0 => 10 = 0x02
        // bit 6-7: 0.3 => 00 = 0x00
        // Byte 0: 0b00_10_00_01 = 0x21
        assert_eq!(packed[0], 0b00_10_00_01);
        // Second byte: -0.7 => 10 = 0x02 in lowest 2 bits
        assert_eq!(packed[1], 0b10);
    }

    #[test]
    fn test_extract_scale() {
        let scale = 127.0f32;
        let mut bytes = scale.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[1u8; 10]);
        assert!((extract_scale(&bytes) - 127.0).abs() < 1e-6);
        assert!((extract_scale(&[]) - 1.0).abs() < 1e-6);
        assert!((extract_scale(&[0, 0, 0]) - 1.0).abs() < 1e-6);
    }
}
