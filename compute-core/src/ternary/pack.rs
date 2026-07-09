//! Ternary packing and unpacking — 2-bit {-1, 0, +1} encoding.
//!
//! Encoding scheme:
//! - `-1` → `00`
//! - `0`  → `01`
//! - `+1` → `10`
//! - `11` is reserved / invalid
//!
//! Each byte packs four values: the first value occupies bits 0-1 (LSB),
//! the second bits 2-3, the third bits 4-5, the fourth bits 6-7.

use crate::ternary::codec::TernaryCodecError;

/// Pack ternary values (-1, 0, +1) into 2-bit codes.
///
/// Encoding (per value, LSB-first within each byte):
///   00 → -1
///   01 →  0
///   10 → +1
///   11 → reserved (rejected at pack time)
///
/// Four values are packed into each byte. If `values.len()` is not a
/// multiple of 4, the final byte is zero-padded with 01 (zero) codes.
pub fn pack_ternary_codes(values: &[i8]) -> Result<Vec<u8>, TernaryCodecError> {
    for &v in values.iter() {
        if v != -1 && v != 0 && v != 1 {
            return Err(TernaryCodecError::InvalidWeight(v));
        }
    }

    let num_bytes = (values.len() + 3) / 4;
    let mut bytes = vec![0u8; num_bytes];

    for (i, &v) in values.iter().enumerate() {
        let code: u8 = match v {
            -1 => 0b00,
            0 => 0b01,
            1 => 0b10,
            _ => unreachable!(), // validated above
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        bytes[byte_idx] |= code << bit_offset;
    }

    // Zero-pad the remaining nibbles in the last byte with 01 (zero).
    let remainder = values.len() % 4;
    if remainder != 0 {
        for j in remainder..4 {
            let bit_offset = j * 2;
            bytes[num_bytes - 1] |= 0b01u8 << bit_offset;
        }
    }

    Ok(bytes)
}

/// Unpack 2-bit ternary codes back into i8 values.
///
/// Returns exactly `expected_values` i8 values. Any padding bits in the
/// final byte beyond `expected_values` are ignored.
pub fn unpack_ternary_codes(
    bytes: &[u8],
    expected_values: usize,
) -> Result<Vec<i8>, TernaryCodecError> {
    let required_bytes = (expected_values + 3) / 4;
    if bytes.len() < required_bytes {
        return Err(TernaryCodecError::LengthMismatch {
            expected: required_bytes,
            actual: bytes.len(),
        });
    }

    let mut out = Vec::with_capacity(expected_values);
    for i in 0..expected_values {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let code = (bytes[byte_idx] >> bit_offset) & 0b11;
        let val = match code {
            0b00 => -1i8,
            0b01 => 0i8,
            0b10 => 1i8,
            0b11 => return Err(TernaryCodecError::ReservedCode11),
            _ => unreachable!(),
        };
        out.push(val);
    }

    Ok(out)
}

/// Validate that no reserved 0b11 codes exist in the packed data.
///
/// Checks every 2-bit code up to `expected_values`. Returns `Ok(())` if
/// all codes are valid (00, 01, 10).
pub fn validate_no_reserved_codes(
    bytes: &[u8],
    expected_values: usize,
) -> Result<(), TernaryCodecError> {
    let required_bytes = (expected_values + 3) / 4;
    if bytes.len() < required_bytes {
        return Err(TernaryCodecError::LengthMismatch {
            expected: required_bytes,
            actual: bytes.len(),
        });
    }

    for i in 0..expected_values {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let code = (bytes[byte_idx] >> bit_offset) & 0b11;
        if code == 0b11 {
            return Err(TernaryCodecError::ReservedCode11);
        }
    }

    Ok(())
}
