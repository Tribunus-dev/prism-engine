//! Ternary 2-bit codec — packed tensor layout, error taxonomy, and pack/unpack.
//!
//! This module is the canonical authority for the 2-bit {-1, 0, +1}
//! codec used by BitNet b1.58 weight tensors. The encoding is the
//! same one the engine used (LSB-first 2-bit codes, four values per
//! byte, 11 reserved as invalid), re-implemented in the constitutional
//! surface so that the bitnet module does not depend on engine-internal
//! codec types.
//!
//! Encoding scheme (per value, LSB-first within each byte):
//! - `00` → -1
//! - `01` →  0
//! - `10` → +1
//! - `11` is reserved / invalid (rejected at pack time)

use half::f16;
use thiserror::Error;

/// Errors that can occur during ternary codec operations.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum TernaryCodecError {
    /// A weight value outside the {-1, 0, +1} set was passed in.
    #[error("invalid ternary weight value: {0} (expected -1, 0, or +1)")]
    InvalidWeight(i8),
    /// A reserved 0b11 code was encountered while unpacking or
    /// validating packed bytes.
    #[error("reserved code 11 encountered in packed ternary data")]
    ReservedCode11,
    /// The byte buffer length did not match the expected number of
    /// values.
    #[error("length mismatch: expected {expected} values, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    /// A packing-layer error (used for adapter-side validation paths
    /// that surface a formatted message rather than a typed variant).
    #[error("packing error: {0}")]
    PackingError(String),
}

/// A tensor packed in the Ternary1_58 codec.
///
/// Weights are packed 4 values per byte (2 bits each, LSB-first).
/// Scales are stored as f16, one per group of `group_size` weights.
/// The layout matches the engine's pre-absorption `TernaryPackedTensor`
/// field-for-field so call sites can convert via a `From` impl when
/// the engine's cimage pipeline is being fed the constitutional output.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPackedTensor {
    /// Number of rows in the stored (in_features) dimension.
    pub rows: usize,
    /// Number of columns in the stored (out_features) dimension.
    pub cols: usize,
    /// Number of ternary values per quantization group.
    pub group_size: usize,
    /// Number of groups per row (`cols.div_ceil(group_size)`).
    pub groups_per_row: usize,
    /// Number of bytes per group of codes (`(group_size + 3) / 4`).
    pub bytes_per_group: usize,
    /// Packed 2-bit codes, four values per byte.
    pub codes: Vec<u8>,
    /// Per-group scales, length = `rows * groups_per_row`.
    pub scales: Vec<f16>,
}

/// Pack ternary values (-1, 0, +1) into 2-bit codes.
///
/// Encoding (per value, LSB-first within each byte):
/// - `00` → -1
/// - `01` →  0
/// - `10` → +1
/// - `11` is reserved (rejected at pack time)
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
/// Returns exactly `expected_values` i8 values. Any padding bits in
/// the final byte beyond `expected_values` are ignored.
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
/// Checks every 2-bit code up to `expected_values`. Returns `Ok(())`
/// if all codes are valid (00, 01, 10).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip_zero_padded() {
        // 5 values: -1, 0, +1, 0, -1 → 2 bytes (3 values in 2nd byte)
        let values = vec![-1i8, 0, 1, 0, -1];
        let packed = pack_ternary_codes(&values).unwrap();
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary_codes(&packed, 5).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_rejects_out_of_set_weight() {
        let values = vec![-1i8, 0, 2];
        let err = pack_ternary_codes(&values).unwrap_err();
        assert!(matches!(err, TernaryCodecError::InvalidWeight(2)));
    }

    #[test]
    fn unpack_rejects_reserved_code_11() {
        // Encode a 0b11 code in the first (LSB) 2 bits → byte 0x03.
        let bytes = vec![0x03u8];
        let err = unpack_ternary_codes(&bytes, 1).unwrap_err();
        assert!(matches!(err, TernaryCodecError::ReservedCode11));
    }

    #[test]
    fn unpack_length_mismatch() {
        let bytes = vec![0u8; 1];
        let err = unpack_ternary_codes(&bytes, 10).unwrap_err();
        assert!(matches!(err, TernaryCodecError::LengthMismatch { .. }));
    }

    #[test]
    fn validate_no_reserved_codes_clean() {
        let values = vec![-1i8, 0, 1, -1];
        let packed = pack_ternary_codes(&values).unwrap();
        validate_no_reserved_codes(&packed, 4).unwrap();
    }

    #[test]
    fn validate_no_reserved_codes_rejects() {
        let bytes = vec![0x03u8];
        let err = validate_no_reserved_codes(&bytes, 1).unwrap_err();
        assert!(matches!(err, TernaryCodecError::ReservedCode11));
    }

    #[test]
    fn empty_pack_returns_empty_bytes() {
        let packed = pack_ternary_codes(&[]).unwrap();
        assert!(packed.is_empty());
    }
}
