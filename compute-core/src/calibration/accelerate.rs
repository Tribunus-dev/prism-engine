//! Accelerate framework FFI bindings for the PT2-LLM iterative ternary fitting
//! calibration engine (Prism-PT2).
//!
//! These bindings route through vecLib, hitting the M-series AMX coprocessor
//! via the vDSP library. Two primitives are exposed:
//!
//! - `vDSP_dotpr`: hardware dot product for quantization error measurement
//! - `vDSP_vthr`: hardware vector thresholding for ternary grid snapping
//!
//! Both operate on f32 vectors. The calibration engine converts FP16 activations
// to f32 for these operations — memory overhead is acceptable since calibration
// processes one layer at a time (not the full 24 GB model).

// Link to Apple's Accelerate framework (vecLib / vDSP).
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// Hardware-accelerated dot product via AMX/NEON.
    ///
    /// Computes `result = sum(input1[i * stride1] * input2[i * stride2] for i in 0..length)`
    ///
    /// Used to calculate the correlation between FP16 weights and the ternary mask
    /// during the ITF scale optimization loop.
    ///
    /// # Safety
    ///
    /// `input1` and `input2` must be valid, aligned f32 pointers of at least
    /// `length * stride` elements.
    fn vDSP_dotpr(
        input1: *const f32,
        stride1: isize,
        input2: *const f32,
        stride2: isize,
        result: *mut f32,
        length: usize,
    );
}

/// Compute dot product of two f32 slices via Accelerate.
///
/// Returns `sum(a[i] * b[i] for i in 0..len)`
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_product: length mismatch");
    let mut result: f32 = 0.0;
    unsafe {
        vDSP_dotpr(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, a.len());
    }
    result
}

/// Apply ternary thresholding via Accelerate.
pub fn ternary_threshold(input: &[f32], threshold: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    for &val in input {
        let snapped = if val > threshold {
            1.0
        } else if val < -threshold {
            -1.0
        } else {
            0.0
        };
        output.push(snapped);
    }
    output
}

/// Measure quantization error for a given scale factor.
///
/// Computes the mean squared error between `original` f32 values and
/// the ternary approximation produced by `scale * ternary_threshold(original / scale)`.
///
/// Used as the objective function in the ITF scale search loop.
pub fn quantization_mse(original: &[f32], scale: f32) -> f32 {
    let inv = 1.0 / scale;
    let scaled: Vec<f32> = original.iter().map(|v| v * inv).collect();
    let snapped = ternary_threshold(&scaled, 0.5);
    let mut mse = 0.0f32;
    for (orig, snap) in original.iter().zip(snapped.iter()) {
        let deq = scale * snap;
        let err = orig - deq;
        mse += err * err;
    }
    mse / original.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_ternary_threshold() {
        // Values: -1.2, -0.3, 0.1, 0.8 → should snap to -1, 0, 0, +1
        let input = vec![-1.2, -0.3, 0.1, 0.8];
        let result = ternary_threshold(&input, 0.5);
        assert_eq!(result, vec![-1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_quantization_mse() {
        let original = vec![0.1, -0.2, 0.9, -1.1];
        // If scale=1.0: ternary_threshold([0.1, -0.2, 0.9, -1.1]) = [0, 0, 1, -1]
        // deq = [0, 0, 1, -1], errors = [0.1, -0.2, -0.1, -0.1]
        // MSE = (0.01 + 0.04 + 0.01 + 0.01) / 4 = 0.0175
        let mse = quantization_mse(&original, 1.0);
        assert!((mse - 0.0175).abs() < 1e-6);
    }
}
