//! `pipeline::lowering::dataset` — known-answer F32 matmul dataset.
//!
//! This file owns the canonical authority for the simple
//! [`F32MatmulDataset`] used to validate that real-backend lowering
//! preserves the already-qualified MLX, Accelerate, and Core ML
//! routes. The dataset is the unit-test contract for the lowering
//! adapters.

use prism_ecs_backend::routing::LogicalShape;

/// A known-answer dataset for a single F32 matmul operation.
///
/// Input A is `[1, 4]`, weight W is `[4, 1]`, expected output C is
/// `[1, 1]` with value `30.0` for the canonical `1..=4` inputs.
#[derive(Debug, Clone)]
pub struct F32MatmulDataset {
    /// Input matrix A in row-major order.
    pub input_data: Vec<f32>,
    /// Weight matrix W in row-major order.
    pub weight_data: Vec<f32>,
    /// Expected output vector in row-major order.
    pub expected_output: Vec<f32>,
    /// Logical shape of the input matrix.
    pub input_shape: Vec<u32>,
    /// Logical shape of the weight matrix.
    pub weight_shape: Vec<u32>,
    /// Logical shape of the output matrix.
    pub output_shape: Vec<u32>,
}

impl Default for F32MatmulDataset {
    fn default() -> Self {
        Self {
            input_data: vec![1.0, 2.0, 3.0, 4.0],
            weight_data: vec![1.0, 2.0, 3.0, 4.0],
            // [1,4] × [4,1] = [1,1] with result [30.0]
            expected_output: vec![30.0],
            input_shape: vec![1, 4],
            weight_shape: vec![4, 1],
            output_shape: vec![1, 1],
        }
    }
}

impl F32MatmulDataset {
    /// The expected output shape as a [`LogicalShape`].
    pub fn output_contract(&self) -> LogicalShape {
        LogicalShape {
            dims: self.output_shape.iter().map(|&d| d as u64).collect(),
        }
    }

    /// Verify that the actual output matches the expected output
    /// element-wise within the given tolerance.
    pub fn verify(&self, actual: &[f32], tolerance: f32) -> Result<(), String> {
        if actual.len() != self.expected_output.len() {
            return Err(format!(
                "output length mismatch: expected {}, got {}",
                self.expected_output.len(),
                actual.len()
            ));
        }
        for (i, (&got, &want)) in actual.iter().zip(self.expected_output.iter()).enumerate() {
            let diff = (got - want).abs();
            if diff > tolerance {
                return Err(format!(
                    "output[{i}]: expected {want}, got {got}, diff {diff}"
                ));
            }
        }
        if actual.iter().any(|x| !x.is_finite()) {
            return Err("output contains non-finite values".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dataset_is_1x4_x_4x1() {
        let d = F32MatmulDataset::default();
        assert_eq!(d.input_shape, vec![1, 4]);
        assert_eq!(d.weight_shape, vec![4, 1]);
        assert_eq!(d.output_shape, vec![1, 1]);
    }

    #[test]
    fn verify_accepts_exact_match() {
        let d = F32MatmulDataset::default();
        assert!(d.verify(&[30.0], 1e-5).is_ok());
    }

    #[test]
    fn verify_rejects_length_mismatch() {
        let d = F32MatmulDataset::default();
        assert!(d.verify(&[30.0, 60.0], 1e-5).is_err());
    }

    #[test]
    fn verify_rejects_non_finite() {
        let d = F32MatmulDataset::default();
        assert!(d.verify(&[f32::NAN], 1e-5).is_err());
    }
}
