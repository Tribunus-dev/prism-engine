//! IEEE 754 FP16 ↔ f32 conversion helpers (engine-decoupled).
//!
//! Authority: pure FP16 conversion math. Std-only — no engine dependencies.

/// Convert a 2-byte little-endian FP16 payload to f32.
pub fn fp16_to_f32(b: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(b);
    let s = (((bits >> 15) & 1) as f32) * -2.0 + 1.0;
    let e = (bits >> 10) & 0x1F;
    let m = (bits & 0x03FF) as f32;
    if e == 0 {
        return if m == 0.0 {
            0.0
        } else {
            s * (m / 1024.0) * 2.0_f32.powi(-14)
        };
    }
    if e == 0x1F {
        return if m == 0.0 {
            if s > 0.0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        } else {
            f32::NAN
        };
    }
    s * (1.0 + m / 1024.0) * 2.0_f32.powi(e as i32 - 15)
}

/// Convert a single-precision float to IEEE 754 FP16 bit pattern.
pub fn half_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        return if mant == 0 {
            if (bits >> 31) != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let exp_f16: i32 = exp as i32 - 127 + 15;
    if exp_f16 >= 0x1F {
        return if (bits >> 31) != 0 { 0xFC00 } else { 0x7C00 };
    }
    if exp_f16 <= 0 {
        return sign;
    }
    sign | ((exp_f16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert an IEEE 754 FP16 bit pattern to f32.
pub fn f32_from_half(x: u16) -> f32 {
    let bits = x as u32;
    let sign = bits & 0x8000;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return 0.0;
        }
        let norm_exp: i32 = -14;
        let norm_mant = mant;
        let fp32_bits = sign << 16 | ((norm_exp + 127) as u32) << 23 | norm_mant << 13;
        return f32::from_bits(fp32_bits);
    }
    if exp == 0x1F {
        let fp32_bits = sign << 16 | 0x7F800000u32 | mant << 13;
        return f32::from_bits(fp32_bits);
    }
    let fp32_exp = exp.wrapping_add(127 - 15);
    let fp32_bits = sign << 16 | fp32_exp << 23 | mant << 13;
    f32::from_bits(fp32_bits)
}

/// Alias of [`fp16_to_f32`] used by some engine call sites that pass a u16.
#[inline]
pub fn half_to_f32(bits: u16) -> f32 {
    f32_from_half(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zero() {
        assert_eq!(fp16_to_f32(half_from_f32(0.0).to_le_bytes()), 0.0);
        assert_eq!(fp16_to_f32(half_from_f32(-0.0).to_le_bytes()), -0.0);
    }

    #[test]
    fn roundtrip_one() {
        let x = 1.0f32;
        let h = half_from_f32(x);
        let y = fp16_to_f32(h.to_le_bytes());
        assert!((x - y).abs() < 1e-3, "got {y}");
    }

    #[test]
    fn inf_roundtrip() {
        let h = half_from_f32(f32::INFINITY);
        assert_eq!(h, 0x7C00);
        let h_neg = half_from_f32(f32::NEG_INFINITY);
        assert_eq!(h_neg, 0xFC00);
    }
}
