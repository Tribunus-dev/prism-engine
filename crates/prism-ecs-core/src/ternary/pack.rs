/// Error type for ternary packing/unpacking operations.
#[derive(Debug)]
pub enum TernaryPackError {
    /// An invalid weight value was encountered (must be -1, 0, or +1).
    InvalidWeight(i8),
    /// A packed byte contained the reserved 0b11 pattern.
    ReservedPattern,
}

impl std::fmt::Display for TernaryPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWeight(w) => write!(f, "invalid ternary weight: {w} (must be -1, 0, or +1)"),
            Self::ReservedPattern => write!(f, "reserved 0b11 pattern in packed ternary data"),
        }
    }
}

impl std::error::Error for TernaryPackError {}

/// Pack ternary weights (-1, 0, +1) into 2-bit packed bytes.
///
/// Each byte stores four ternary values:
///   - Byte bits [1:0] = value 0
///   - Byte bits [3:2] = value 1
///   - Byte bits [5:4] = value 2
///   - Byte bits [7:6] = value 3
///
/// Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10.  0b11 is reserved.
pub fn pack_ternary_codes(weights: &[i8]) -> Result<Vec<u8>, TernaryPackError> {
    let mut packed = Vec::with_capacity(weights.len().div_ceil(4));
    for chunk in weights.chunks(4) {
        // Unused lanes decode as zero, the canonical neutral value.
        let mut byte: u8 = 0x55;
        for (i, &w) in chunk.iter().enumerate() {
            let code: u8 = match w {
                -1 => 0b00,
                0 => 0b01,
                1 => 0b10,
                other => return Err(TernaryPackError::InvalidWeight(other)),
            };
            byte = (byte & !(0x03 << (i * 2))) | (code << (i * 2));
        }
        packed.push(byte);
    }
    Ok(packed)
}

/// Unpack 2-bit packed ternary data back to -1, 0, +1 values.
///
/// `expected` is the number of ternary values to extract. The packed slice
/// must contain at least `ceil(expected / 4)` bytes.
pub fn unpack_ternary_codes(packed: &[u8], expected: usize) -> Result<Vec<i8>, TernaryPackError> {
    let mut weights = Vec::with_capacity(expected);
    for &byte in packed {
        for i in 0..4 {
            if weights.len() >= expected {
                break;
            }
            let code = (byte >> (i * 2)) & 0x03;
            let val = match code {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                0b11 => return Err(TernaryPackError::ReservedPattern),
                _ => unreachable!(),
            };
            weights.push(val);
        }
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let input = vec![1, 0, -1, 1, 0, -1, 0, 1];
        let packed = pack_ternary_codes(&input).unwrap();
        assert_eq!(packed.len(), 2);
        let unpacked = unpack_ternary_codes(&packed, input.len()).unwrap();
        assert_eq!(unpacked, input);
    }

    #[test]
    fn test_invalid_weight() {
        let result = pack_ternary_codes(&[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reserved_pattern() {
        // Manually construct a byte with 0b11 pattern
        let packed = vec![0b0000_0011];
        let result = unpack_ternary_codes(&packed, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty() {
        assert_eq!(pack_ternary_codes(&[]).unwrap(), Vec::<u8>::new());
        assert_eq!(unpack_ternary_codes(&[], 0).unwrap(), Vec::<i8>::new());
    }

    #[test]
    fn test_partial_last_byte() {
        let input = vec![1, 0, -1];
        let packed = pack_ternary_codes(&input).unwrap();
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_ternary_codes(&packed, input.len()).unwrap();
        assert_eq!(unpacked, input);
    }
}
