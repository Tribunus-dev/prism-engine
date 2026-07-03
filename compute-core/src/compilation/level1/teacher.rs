//! Level 1 dense teacher forward executor via Accelerate vDSP.
//!
//! Loads teacher weights, runs a dense FP16 GEMV forward pass on CPU via
//! vDSP dot product (AMX coprocessor). Stateless: receives a microbatch
//! index and produces activations into an internal output buffer that the
//! scheduler and reducer access for deterministic CPU-side comparisons.

use crate::calibration::accelerate::dot_product;

/// Default hidden dimension used for the teacher weight matrix.
const HIDDEN_DIM: usize = 3840;

/// Simulated FP16 teacher weights and output buffer.
struct WeightStore {
    /// Flat weight matrix: out_dim × in_dim, row-major.
    data: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl WeightStore {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let n = in_dim * out_dim;
        let mut data = vec![0.0f32; n];
        // Seed with a deterministic sinusoidal pattern (synthetic proxy for
        // real teacher weights loaded from a checkpoint).  The amplitude is
        // low (0.01) so the dot-product output is in a reasonable FP16 range.
        for i in 0..n {
            data[i] = ((i as f64).sin() * 0.01) as f32;
        }
        WeightStore { data, in_dim, out_dim }
    }
}

/// The dense teacher forward executor.
///
/// Wraps the Accelerate/vDSP-based dense GEMV that runs the teacher model on
/// a microbatch of tokens.  Stateless — each call receives a microbatch index
/// and produces output activations into the internal output buffer.
pub struct MetalTeacher {
    weights: WeightStore,
    output: Vec<f32>,
    sample_input: Vec<f32>,
}

impl MetalTeacher {
    /// Create a new MetalTeacher with default 3840×3840 weight matrix.
    pub fn new() -> Self {
        let in_dim = HIDDEN_DIM;
        let out_dim = HIDDEN_DIM;
        // Pre-generate the synthetic sample input (doesn't vary per microbatch
        // in this synthetic Level 1 mode; real weights would load from disk).
        let sample_input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f64).cos() * 0.1) as f32)
            .collect();

        MetalTeacher {
            weights: WeightStore::new(in_dim, out_dim),
            output: vec![0.0f32; out_dim],
            sample_input,
        }
    }

    /// Create a MetalTeacher with a specific weight matrix shape.
    pub fn with_shape(in_dim: usize, out_dim: usize) -> Self {
        let sample_input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f64).cos() * 0.1) as f32)
            .collect();

        MetalTeacher {
            weights: WeightStore::new(in_dim, out_dim),
            output: vec![0.0f32; out_dim],
            sample_input,
        }
    }

    /// Execute dense teacher forward via Accelerate vDSP.
    ///
    /// Computes `y[o] = Σᵢ W[o,i] · x[i]` for each output row using the
    /// hardware-accelerated `dot_product` (vDSP on AMX/NEON).
    pub fn forward(&mut self, _microbatch: usize, _slot_id: u64) {
        let out_dim = self.weights.out_dim;
        let in_dim = self.weights.in_dim;

        for o in 0..out_dim {
            let start = o * in_dim;
            let row = &self.weights.data[start..start + in_dim];
            // Accelerate vDSP dot product — hits AMX coprocessor on M-series.
            self.output[o] = dot_product(row, &self.sample_input);
        }
    }

    /// Access the internal output buffer for reduction.
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the hidden dimension (output width).
    pub fn hidden_dim(&self) -> usize {
        self.weights.out_dim
    }
}

impl Default for MetalTeacher {
    fn default() -> Self {
        Self::new()
    }
}
