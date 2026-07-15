//! Residual codec for ternary quantization.
//!
//! Encodes the residual between the original float tensor and its
//! ternary-quantized reconstruction. Supports both dense (full vector)
//! and sparse (threshold-gated) encoding.

/// Residual codec — encode/decode residual compensation for ternary quantization.
#[derive(Debug, Clone)]
pub struct ResidualCodec;

impl ResidualCodec {
    /// Encode dense residuals: `original - reconstructed` for every element.
    pub fn encode_dense(original: &[f32], reconstructed: &[f32]) -> Vec<f32> {
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| o - r)
            .collect()
    }

    /// Encode sparse residuals: only keep elements whose absolute error
    /// exceeds `threshold`.
    ///
    /// Returns `(indices, values)`.
    pub fn encode_sparse(
        original: &[f32],
        reconstructed: &[f32],
        threshold: f64,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (i, (o, r)) in original.iter().zip(reconstructed.iter()).enumerate() {
            let err = (o - r).abs() as f64;
            if err > threshold {
                indices.push(i);
                values.push(o - r);
            }
        }
        (indices, values)
    }

    /// Apply dense residuals: `base + residuals` element-wise.
    ///
    /// # Panics
    /// Panics if `base` and `residuals` have different lengths.
    pub fn apply_dense(base: &[f32], residuals: &[f32]) -> Vec<f32> {
        assert_eq!(
            base.len(),
            residuals.len(),
            "dense residual apply: length mismatch {} vs {}",
            base.len(),
            residuals.len()
        );
        base.iter()
            .zip(residuals.iter())
            .map(|(b, r)| b + r)
            .collect()
    }

    /// Apply sparse residuals in-place: `base[indices[i]] += values[i]`.
    ///
    /// # Panics
    /// Panics if `indices` and `values` have different lengths, or any
    /// `indices[i]` is out of bounds.
    pub fn apply_sparse(base: &mut [f32], indices: &[usize], values: &[f32]) {
        assert_eq!(
            indices.len(),
            values.len(),
            "sparse residual apply: length mismatch {} vs {}",
            indices.len(),
            values.len()
        );
        for (&i, &v) in indices.iter().zip(values.iter()) {
            assert!(
                i < base.len(),
                "sparse residual apply: index {} out of bounds (len {})",
                i,
                base.len()
            );
            base[i] += v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_dense_all() {
        let original = vec![1.0, 2.0, 3.0];
        let reconstructed = vec![0.9, 2.1, 2.8];
        let residuals = ResidualCodec::encode_dense(&original, &reconstructed);
        assert_eq!(residuals.len(), 3);
        // 1.0 - 0.9 = 0.1, 2.0 - 2.1 = -0.1, 3.0 - 2.8 = 0.2
        assert!((residuals[0] - 0.1).abs() < 1e-6);
        assert!((residuals[1] + 0.1).abs() < 1e-6);
        assert!((residuals[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_encode_dense_empty() {
        let original: Vec<f32> = vec![];
        let reconstructed: Vec<f32> = vec![];
        let residuals = ResidualCodec::encode_dense(&original, &reconstructed);
        assert!(residuals.is_empty());
    }

    #[test]
    fn test_encode_sparse_threshold() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let reconstructed = vec![1.05, 2.0, 2.5, 4.2];
        let (indices, values) = ResidualCodec::encode_sparse(&original, &reconstructed, 0.1);
        // index 0: error = 0.05 ≤ 0.1 → skipped
        // index 1: error = 0.0 ≤ 0.1 → skipped
        // index 2: error = 0.5 > 0.1 → kept (residual = 0.5)
        // index 3: error = 0.2 > 0.1 → kept (residual = -0.2)
        assert_eq!(indices.len(), 2);
        assert_eq!(values.len(), 2);
        assert_eq!(indices[0], 2);
        assert!((values[0] - 0.5).abs() < 1e-6);
        assert_eq!(indices[1], 3);
        assert!((values[1] + 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_encode_sparse_all_below_threshold() {
        let original = vec![1.0, 2.0];
        let reconstructed = vec![1.01, 2.01];
        let (indices, values) = ResidualCodec::encode_sparse(&original, &reconstructed, 0.1);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn test_apply_dense() {
        let base = vec![1.0, 2.0, 3.0];
        let residuals = vec![0.1, -0.1, 0.2];
        let result = ResidualCodec::apply_dense(&base, &residuals);
        assert!((result[0] - 1.1).abs() < 1e-6);
        assert!((result[1] - 1.9).abs() < 1e-6);
        assert!((result[2] - 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_apply_sparse() {
        let mut base = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 2];
        let values = vec![0.5, -0.5];
        ResidualCodec::apply_sparse(&mut base, &indices, &values);
        assert!((base[0] - 1.5).abs() < 1e-6);
        assert!((base[1] - 2.0).abs() < 1e-6);
        assert!((base[2] - 2.5).abs() < 1e-6);
        assert!((base[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_apply_dense_mismatch() {
        ResidualCodec::apply_dense(&[1.0, 2.0], &[0.1]);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn test_apply_sparse_mismatch() {
        let mut base = vec![1.0, 2.0];
        ResidualCodec::apply_sparse(&mut base, &[0], &[0.1, 0.2]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_apply_sparse_oob() {
        let mut base = vec![1.0, 2.0];
        ResidualCodec::apply_sparse(&mut base, &[5], &[0.1]);
    }
}
