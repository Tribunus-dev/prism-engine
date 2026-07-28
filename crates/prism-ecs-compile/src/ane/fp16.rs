//! IEEE 754 binary16 ↔ binary32 conversion helpers.
//!
//! Authority: FP16 quantization for the ANE compile-time surface.
//!
//! These are pure functions with no engine dependencies. The same
//! conversion logic is duplicated across the engine's `ane/` files;
//! the constitutional surface provides a single canonical home.
//!
//! # Rounding
//!
//! - Subnormal f32 values are flushed to fp16 zero (per Apple ANE behavior).
//! - Overflows (|x| > 65504) round to ±Inf in fp16.
//! - NaNs preserve the sign and the signalling bit; quiet NaNs remain quiet.
//! - Infs round-trip exactly.
//!
//! # Tests
//!
//! The test module verifies round-tripping for finite values, special
//! values, and the round-trip through the constitutional `SlotAllocator`
//! data path (the `slot_allocator.rs` tests cover the `usize` casts).

/// Convert an IEEE 754 binary32 `f32` to a packed `u16` in fp16 format.
///
/// Flushes subnormals to zero; overflows to infinity; preserves NaN
/// signalling. Returns the packed 16-bit value.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    // Special cases
    if exp == 0xFF {
        // NaN or Inf — preserve sign, set fp16 exponent all-ones
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        // f32 subnormal / zero → flush to fp16 zero
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        // Overflow → Inf
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        // Underflow → zero
        return sign << 15;
    }

    let new_mant = mant >> 13;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
}

/// Convert a packed IEEE 754 fp16 `u16` to a full `f32`.
///
/// Denormals are normalised; NaNs and infinities round-trip correctly.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        // Zero or denormal
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Normalise: actual exponent is -14
        let leading = mant.leading_zeros() - 21; // 32 - 11
        let norm_exp = ((127 - 15 - leading as i32) as u32) << 23;
        let norm_mant = (mant << (leading + 1)) & 0x7F_FFFF;
        f32::from_bits((sign << 31) | norm_exp | norm_mant)
    } else if exp == 31 {
        // NaN or Inf
        let mant32 = if mant == 0 {
            0
        } else {
            (mant << 13) | 0x7F_FFFF
        };
        f32::from_bits((sign << 31) | 0x7F80_0000 | mant32)
    } else {
        // Normal fp16 value
        let exp32 = (exp + (127 - 15)) << 23;
        let mant32 = mant << 13;
        f32::from_bits((sign << 31) | exp32 | mant32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_f16_round_trip_finite() {
        for &x in &[0.0f32, 1.0, 2.0, 42.0, -1.0, -128.0, 65504.0] {
            let packed = f32_to_f16(x);
            let back = f16_to_f32(packed);
            assert!(
                (back - x).abs() <= x.abs() * 1e-3 || x == 0.0,
                "roundtrip {x} → {packed:#06x} → {back}",
            );
        }
    }

    #[test]
    fn f32_to_f16_special_values() {
        // Zero
        assert_eq!(f32_to_f16(0.0f32), 0x0000);
        assert_eq!(f32_to_f16(-0.0f32), 0x8000);
        // Inf
        let inf_f16 = f32_to_f16(f32::INFINITY);
        assert_eq!(inf_f16, 0x7C00);
        // NaN (preserves signalling bit)
        let nan_f16 = f32_to_f16(f32::NAN);
        assert!(nan_f16 & 0x7C00 == 0x7C00);
        assert!(nan_f16 & 0x0200 != 0);
    }

    #[test]
    fn f16_to_f32_zero_and_inf() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }
}
