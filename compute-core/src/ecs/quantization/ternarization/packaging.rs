//! Physical layout packaging for ternary candidates.
//!
//! Packages ternary weights and scales into a `TernaryPackage` suitable for
//! inclusion in a compiled artifact. The physical layout describes how the
//! packed data maps onto the target backend's memory model.

use super::candidate::PhysicalTileLayout;

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
    /// The packing maps {-1, 0, +1} to {0, 1, 2} as `u8` values,
    /// one byte per weight.
    pub packed_bytes: Vec<u8>,
}

/// Pack ternary weights and scales into a `TernaryPackage`.
///
/// The ternary values {-1, 0, +1} are shifted to {0, 1, 2} for
/// packed byte storage. The physical layout is set to
/// `PhysicalTileLayout::Tile640`.
pub fn pack_ternary(weights: &[i8], scales: &[f32]) -> TernaryPackage {
    let packed: Vec<u8> = weights.iter().map(|&w| (w + 1) as u8).collect();

    TernaryPackage {
        weights: weights.to_vec(),
        scales: scales.to_vec(),
        physical_layout: PhysicalTileLayout::Tile640,
        packed_bytes: packed,
    }
}

/// Unpack a `TernaryPackage` back to ternary weights.
///
/// Reverses the shift: {0, 1, 2} → {-1, 0, +1}.
pub fn unpack_ternary(package: &TernaryPackage) -> Vec<i8> {
    package
        .packed_bytes
        .iter()
        .map(|&b| match b {
            0 => -1,
            1 => 0,
            2 => 1,
            other => {
                // Saturate any unexpected byte to nearest valid value.
                if other <= 1 {
                    other as i8 - 1
                } else {
                    1
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_ternary_roundtrip() {
        let weights = vec![1, 0, -1, 1, -1, 0, 1, 1, -1, 0];
        let scales = vec![0.5, 0.5, 0.5];
        let package = pack_ternary(&weights, &scales);

        assert_eq!(package.weights, weights);
        assert_eq!(package.scales, scales);
        assert_eq!(package.packed_bytes.len(), weights.len());

        // Verify shift: {-1, 0, +1} → {0, 1, 2}
        assert_eq!(package.packed_bytes[0], 2); // 1 → 2
        assert_eq!(package.packed_bytes[1], 1); // 0 → 1
        assert_eq!(package.packed_bytes[2], 0); // -1 → 0

        // Roundtrip back
        let unpacked = unpack_ternary(&package);
        assert_eq!(unpacked, weights);
    }

    #[test]
    fn test_pack_ternary_empty() {
        let weights: Vec<i8> = vec![];
        let scales: Vec<f32> = vec![];
        let package = pack_ternary(&weights, &scales);
        assert!(package.weights.is_empty());
        assert!(package.packed_bytes.is_empty());
    }

    #[test]
    fn test_pack_ternary_physical_layout() {
        let package = pack_ternary(&[1, -1], &[1.0]);
        match package.physical_layout {
            PhysicalTileLayout::Tile640 => {} // expected
        }
    }

    #[test]
    fn test_unpack_ternary_saturates() {
        // Feed bytes outside {0, 1, 2} to test saturation.
        let package = TernaryPackage {
            weights: vec![],
            scales: vec![],
            physical_layout: PhysicalTileLayout::Tile640,
            packed_bytes: vec![0, 1, 2, 3, 255],
        };
        let unpacked = unpack_ternary(&package);
        assert_eq!(unpacked, vec![-1, 0, 1, 1, 1]); // 3→1, 255→1
    }

    #[test]
    fn test_pack_ternary_all_zeros() {
        let weights = vec![0, 0, 0];
        let package = pack_ternary(&weights, &[1.0]);
        assert_eq!(package.packed_bytes, vec![1, 1, 1]);
        let unpacked = unpack_ternary(&package);
        assert_eq!(unpacked, vec![0, 0, 0]);
    }
}
