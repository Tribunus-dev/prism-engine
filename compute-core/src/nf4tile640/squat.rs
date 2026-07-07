//! Squat (SQuashed Activation Training) — requantization helpers for
//! teacher activations using the nf4tile640 packed format.

use serde::{Deserialize, Serialize};

use crate::nf4tile640::{pack_nf4_weights, unpack_nf4_weights};

/// Configuration for squat requantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquatConfig {
    pub target_mode: String,
}

impl Default for SquatConfig {
    fn default() -> Self {
        Self {
            target_mode: "nf4tile640".to_string(),
        }
    }
}

/// Quantize and dequantize teacher activations through the nf4tile640
/// packed format, returning the requantized (reconstructed) activations.
///
/// This simulates the loss introduced during training by packing activations
/// as NF4 weights and unpacking them back.
pub fn squat_requantize(teacher_activations: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (codes, scales, biases, _, _) = pack_nf4_weights(teacher_activations, rows, cols);
    unpack_nf4_weights(&codes, &scales, &biases, rows, cols)
}

/// Requantize using an explicit [`SquatConfig`].  Currently delegates to
/// [`squat_requantize`]; future implementations may vary strategy based on
/// `config.target_mode`.
pub fn squat_requantize_with(
    teacher_activations: &[f32],
    rows: usize,
    cols: usize,
    _config: &SquatConfig,
) -> Vec<f32> {
    squat_requantize(teacher_activations, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squat_roundtrip() {
        let rows = 2;
        let cols = 640;
        let n = rows * cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect();
        let output = squat_requantize(&input, rows, cols);
        assert_eq!(output.len(), input.len());
        let max_out = output.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_out > 0.0, "output should contain non-zero values");
    }

    #[test]
    fn test_squat_non_multiple_cols() {
        let rows = 3;
        let cols = 1000;
        let n = rows * cols;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let output = squat_requantize(&input, rows, cols);
        assert_eq!(output.len(), input.len());
        let cfg = SquatConfig::default();
        let output2 = squat_requantize_with(&input, rows, cols, &cfg);
        assert_eq!(output, output2);
    }
}
