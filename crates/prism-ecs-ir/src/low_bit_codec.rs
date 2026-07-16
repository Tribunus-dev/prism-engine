//! Bonsai codec: ternary 1.58-bit and binary 1-bit dot products.
//!
//! Pure-math utilities for efficient dot-product computation on quantized
//! values in Tile640 format (640 elements packed into 80 bytes).
//!
//! Each codegen module (Metal, CPU, etc.) calls these directly instead of
//! inlining the bit-level logic.

/// Compute the dot product of two ternary-valued vectors.
///
/// Each element of `a` and `b` must be -1, 0, or +1. The computation uses
/// only addition/subtraction — no general integer multiplication — because
/// `a[i] * b[i]` reduces to a sign comparison when neither operand is zero.
///
/// The Tile640 format packs 640 elements; both slices **must** have at least
/// 640 entries (only the first 640 are read).
///
/// `scale_a` and `scale_b` are the pre-resolved FP16-per-128-block scales
/// (promoted to `f32`). The result is `scale_a * scale_b * Σ(a[i]·b[i])`.
#[inline]
pub fn ternary_dot_product(a: &[i8], b: &[i8], scale_a: f32, scale_b: f32) -> f32 {
    debug_assert!(
        a.len() >= 640,
        "ternary_dot_product: a too short (need 640)"
    );
    debug_assert!(
        b.len() >= 640,
        "ternary_dot_product: b too short (need 640)"
    );

    let sum: i32 = a[..640]
        .iter()
        .zip(b[..640].iter())
        .map(|(&va, &vb)| {
            // va, vb ∈ {-1, 0, +1}
            // product = +1 when equal and non-zero,
            //           -1 when opposite and non-zero,
            //            0 when either operand is zero.
            match (va, vb) {
                (0, _) | (_, 0) => 0i32,
                (x, y) if x == y => 1i32,
                _ => -1i32,
            }
        })
        .sum();

    scale_a * scale_b * sum as f32
}

/// Compute the dot product of two binary (1-bit) valued vectors.
///
/// Each bit represents {0, +1}. The Tile640 format packs 640 bits into 80
/// bytes; both slices **must** have at least 80 entries (only the first 80
/// are read).
///
/// The computation is `popcount(a ^ b)` — the number of bit positions where
/// the two vectors differ. This yields a similarity measure that, combined
/// with the scale factors, recovers the quantized dot product.
///
/// `scale_a` and `scale_b` are the pre-resolved FP16-per-128-block scales
/// (promoted to `f32`). The result is `scale_a * scale_b * popcount(a ^ b)`.
#[inline]
pub fn binary_popcount_dot(a: &[u8], b: &[u8], scale_a: f32, scale_b: f32) -> f32 {
    debug_assert!(a.len() >= 80, "binary_popcount_dot: a too short (need 80)");
    debug_assert!(b.len() >= 80, "binary_popcount_dot: b too short (need 80)");

    let pop: u32 = a[..80]
        .iter()
        .zip(b[..80].iter())
        .map(|(&va, &vb)| (va ^ vb).count_ones())
        .sum();

    scale_a * scale_b * pop as f32
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ternary_dot_product ───────────────────────────────────────────────

    #[test]
    fn ternary_all_ones_same_sign() {
        let mut a = [0i8; 640];
        let mut b = [0i8; 640];
        a.fill(1);
        b.fill(1);
        let result = ternary_dot_product(&a, &b, 1.0, 1.0);
        assert_eq!(result, 640.0, "all +1 · all +1 should be 640");
    }

    #[test]
    fn ternary_all_opposite_signs() {
        let mut a = [0i8; 640];
        let mut b = [0i8; 640];
        a.fill(1);
        b.fill(-1);
        let result = ternary_dot_product(&a, &b, 1.0, 1.0);
        assert_eq!(result, -640.0, "all +1 · all -1 should be -640");
    }

    #[test]
    fn ternary_half_sparse_with_scales() {
        // a: 320 leading +1s, 320 zeros
        // b: all +1
        // sum = 320 * (+1*+1) = 320, scale_a=2.0, scale_b=0.5 → 320.0
        let mut a = [0i8; 640];
        let b = [1i8; 640];
        a[..320].fill(1);
        let result = ternary_dot_product(&a, &b, 2.0, 0.5);
        assert_eq!(
            result, 320.0,
            "320 matches with scales 2.0×0.5 should be 320"
        );
    }

    #[test]
    fn ternary_alternating_pattern() {
        // a: +1, -1, +1, -1, ... (320 each)
        // b: all +1
        // 320 × (+1*+1 = +1) + 320 × (-1*+1 = -1) = 0
        let a: Vec<i8> = (0..640).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let b = [1i8; 640];
        let result = ternary_dot_product(&a, &b, 3.0, 1.0);
        assert_eq!(result, 0.0, "balanced alternating pattern should be 0");
    }

    #[test]
    fn ternary_all_zero() {
        let a = [0i8; 640];
        let b = [0i8; 640];
        let result = ternary_dot_product(&a, &b, 1.0, 1.0);
        assert_eq!(result, 0.0, "all zero should be 0");
    }

    #[test]
    fn ternary_zero_scale() {
        let mut a = [1i8; 640];
        let mut b = [1i8; 640];
        a.fill(1);
        b.fill(1);
        let result = ternary_dot_product(&a, &b, 0.0, 2.0);
        assert_eq!(result, 0.0, "zero scale should give 0");
    }

    #[test]
    fn ternary_manual_known_values() {
        // a = [1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, ...]
        // b = [1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, ...]
        // Manual:
        //   i0:  1*1  = +1
        //   i1:  1*-1 = -1
        //   i2:  1*1  = +1
        //   i3:  1*-1 = -1
        //   i4: -1*1  = -1
        //   i5: -1*-1 = +1
        //   i6: -1*1  = -1
        //   i7: -1*-1 = +1
        //   i8:  0*1  =  0
        //   ...
        // First 8: sum = 0, remaining all-zero → total sum = 0
        let mut a = [0i8; 640];
        let mut b = [0i8; 640];
        for i in 0..4 {
            a[i] = 1;
            b[i] = if i % 2 == 0 { 1 } else { -1 };
        }
        for i in 4..8 {
            a[i] = -1;
            b[i] = if i % 2 == 0 { 1 } else { -1 };
        }
        // Fill rest of b with +1
        b[8..].fill(1);
        // Remaining 632 elements: a[i]=0, b[i]=1 → 0
        let result = ternary_dot_product(&a, &b, 1.0, 1.0);
        assert_eq!(result, 0.0, "manual calculation should be 0");
    }

    // ── binary_popcount_dot ───────────────────────────────────────────────

    #[test]
    fn binary_all_ones_vs_all_zeros() {
        let a = [0xFFu8; 80];
        let b = [0x00u8; 80];
        // XOR = 0xFF for each byte → popcount 8 per byte → 80×8 = 640
        let result = binary_popcount_dot(&a, &b, 1.0, 1.0);
        assert_eq!(result, 640.0, "all 1s vs all 0s XOR-popcount should be 640");
    }

    #[test]
    fn binary_identical_vectors() {
        let a = [0x55u8; 80];
        let b = [0x55u8; 80];
        // XOR = 0x00 → popcount 0 per byte → 0
        let result = binary_popcount_dot(&a, &b, 1.0, 1.0);
        assert_eq!(result, 0.0, "identical vectors XOR-popcount should be 0");
    }

    #[test]
    fn binary_alternating_pattern() {
        // a: 0xAA (1010_1010), b: 0x55 (0101_0101)
        // XOR = 0xFF per byte → popcount 8 per byte → 80×8 = 640
        let a = [0xAAu8; 80];
        let b = [0x55u8; 80];
        let result = binary_popcount_dot(&a, &b, 1.0, 1.0);
        assert_eq!(
            result, 640.0,
            "complementary patterns XOR-popcount should be 640"
        );
    }

    #[test]
    fn binary_mixed_with_scales() {
        // a: all 0xFF, b: all 0xFF → XOR = 0x00 → pop = 0
        let a = [0xFFu8; 80];
        let b = [0xFFu8; 80];
        let result = binary_popcount_dot(&a, &b, 3.0, 2.0);
        assert_eq!(
            result, 0.0,
            "identical vectors with non-zero scales should be 0"
        );
    }

    #[test]
    fn binary_partial_overlap() {
        // a: 0x0F (0000_1111), b: 0x03 (0000_0011)
        // XOR = 0x0C (0000_1100) → popcount 2
        let a = [0x0Fu8; 80];
        let b = [0x03u8; 80];
        let result = binary_popcount_dot(&a, &b, 1.0, 1.0);
        assert_eq!(result, 160.0, "80 bytes × popcount(0x0C)=2 should be 160");
    }

    #[test]
    fn binary_zero_scale() {
        let a = [0xFFu8; 80];
        let b = [0x00u8; 80];
        let result = binary_popcount_dot(&a, &b, 0.0, 1.0);
        assert_eq!(result, 0.0, "zero scale should give 0");
    }

    #[test]
    fn binary_all_zeros() {
        let a = [0x00u8; 80];
        let b = [0x00u8; 80];
        let result = binary_popcount_dot(&a, &b, 1.0, 1.0);
        assert_eq!(result, 0.0, "all zero XOR-popcount should be 0");
    }

    #[test]
    fn binary_known_popcount() {
        // 0xF0 (1111_0000) ^ 0xCC (1100_1100) = 0x3C (0011_1100) → popcount 4
        let a = [0xF0u8; 80];
        let b = [0xCCu8; 80];
        let result = binary_popcount_dot(&a, &b, 2.0, 0.5);
        assert_eq!(result, 320.0, "80×4=320 with scales 2.0×0.5");
    }
}
