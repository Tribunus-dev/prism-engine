//! Physical layout packaging for ternary candidates.
//!
//! Packages ternary weights and scales into a `TernaryPackage` suitable for
//! inclusion in a compiled artifact. The physical layout describes how the
//! packed data maps onto the target backend's memory model.

use super::candidate::PhysicalTileLayout;
use prism_ecs_core::ternary::pack::{pack_ternary_codes, unpack_ternary_codes};

/// Packaged ternary tensor — ready for artifact assembly.
#[derive(Debug, Clone)]
pub struct TernaryPackage {
    /// Ternary weights (-1, 0, +1).
    pub weights: Vec<i8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Physical tile layout variant.
    pub physical_layout: PhysicalTileLayout,
    /// Packed byte representation of the ternary weights.
    ///
    /// 4 ternary values per byte (2-bit packing), with the encoding:
    ///   -1 → 0b00, 0 → 0b01, +1 → 0b10.
    /// 0b11 is reserved/invalid.
    pub packed_bytes: Vec<u8>,
}

/// Pack ternary weights and scales into a `TernaryPackage`.
///
/// Each weight is packed into 2 bits (4 values per byte).
/// The physical layout is set to `PhysicalTileLayout::Tile640`.
///
/// Returns an error if any weight is not -1, 0, or +1.
pub fn pack_ternary(weights: &[i8], scales: &[f32]) -> Result<TernaryPackage, String> {
    let mut packed = pack_ternary_codes(weights).map_err(|e| e.to_string())?;
    if let Some(last) = packed.last_mut() {
        let used = weights.len() % 4;
        if used != 0 {
            for i in used..4 {
                *last |= 0b01 << (i * 2);
            }
        }
    }

    Ok(TernaryPackage {
        weights: weights.to_vec(),
        scales: scales.to_vec(),
        physical_layout: PhysicalTileLayout::Tile640,
        packed_bytes: packed,
    })
}

/// Unpack a `TernaryPackage` back to ternary weights.
///
/// Returns an error if the packed data contains reserved 0b11 codes.
pub fn unpack_ternary(package: &TernaryPackage) -> Result<Vec<i8>, String> {
    let expected = package.weights.len();
    unpack_ternary_codes(&package.packed_bytes, expected).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_ternary_roundtrip() {
        let weights = vec![1, 0, -1, 1, -1, 0, 1, 1, -1, 0];
        let scales = vec![0.5, 0.5, 0.5];
        let package = pack_ternary(&weights, &scales).unwrap();

        assert_eq!(package.weights, weights);
        assert_eq!(package.scales, scales);
        // 10 weights → ceil(10/4) = 3 bytes with 2-bit packing
        assert_eq!(package.packed_bytes.len(), 3);

        // Verify 2-bit encoding: first byte = 0b10_00_01_10 = 0x86
        assert_eq!(package.packed_bytes[0], 0x86);

        // Roundtrip back
        let unpacked = unpack_ternary(&package).unwrap();
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_pack_ternary_empty() {
        let weights: Vec<i8> = vec![];
        let scales: Vec<f32> = vec![];
        let package = pack_ternary(&weights, &scales).unwrap();
        assert!(package.weights.is_empty());
        assert!(package.packed_bytes.is_empty());
    }

    #[test]
    fn test_pack_ternary_physical_layout() {
        let package = pack_ternary(&[1, -1], &[1.0]).unwrap();
        match package.physical_layout {
            PhysicalTileLayout::Tile640 => {} // expected
        }
    }

    #[test]
    fn test_unpack_ternary_rejects_reserved_codes() {
        // Feed bytes with reserved 0b11 codes to verify rejection.
        let package = TernaryPackage {
            weights: vec![0i8; 5], // 5 values → required_bytes = 2
            scales: vec![],
            physical_layout: PhysicalTileLayout::Tile640,
            packed_bytes: vec![0x03, 0], // byte 0 has 0b11 in bits 0-1
        };
        let result = unpack_ternary(&package);
        assert!(
            result.is_err(),
            "reserved code 0b11 should cause unpack error"
        );
        assert!(
            result.unwrap_err().contains("reserved"),
            "error should mention reserved code"
        );
    }

    #[test]
    fn test_pack_ternary_all_zeros() {
        let weights = vec![0, 0, 0];
        let package = pack_ternary(&weights, &[1.0]).unwrap();
        // 3 zeros → 1 byte: each zero → 0b01, pad → 0b01: 0b01_01_01_01 = 0x55
        assert_eq!(package.packed_bytes, vec![0x55]);
        let unpacked = unpack_ternary(&package).unwrap();
        assert_eq!(unpacked, vec![0, 0, 0]);
    }

    #[test]
    fn test_pack_unpack_single_value() {
        // A single weight packs into 1 byte.
        let package = pack_ternary(&[1], &[1.0]).unwrap();
        assert_eq!(package.packed_bytes.len(), 1);
        let unpacked = unpack_ternary(&package).unwrap();
        assert_eq!(unpacked, vec![1]);
    }

    #[test]
    fn test_pack_multiple_of_four() {
        // Exactly 4 weights → packed_bytes.len() = 1 = 4/4
        let weights = vec![1, 0, -1, 1];
        let package = pack_ternary(&weights, &[1.0]).unwrap();
        assert_eq!(package.packed_bytes.len(), weights.len() / 4);
        let unpacked = unpack_ternary(&package).unwrap();
        assert_eq!(unpacked, weights);

        // 8 weights → 2 bytes = 8/4
        let weights2 = vec![1, -1, 0, 1, -1, 1, 0, -1];
        let package2 = pack_ternary(&weights2, &[1.0]).unwrap();
        assert_eq!(package2.packed_bytes.len(), weights2.len() / 4);
        let unpacked2 = unpack_ternary(&package2).unwrap();
        assert_eq!(unpacked2, weights2);
    }
}
