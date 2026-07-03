//! Level 1 Accelerate control-plane reducer.
//!
//! Accelerate owns all control-plane numerical work: MSE computation, cosine
//! similarity, residual relative error, moment accumulation, Gram or Hessian-
//! diagonal estimates, threshold selection, per-page and per-channel scale
//! solves, sidecar ranking, deterministic reductions, and receipt hashing.

use crate::calibration::accelerate::dot_product;

/// The Accelerate reducer — runs on the CPU/Accelerate control plane.
///
/// Compares teacher and student outputs for a microbatch, computes metrics
/// via vDSP primitives, and stores reduction results for the candidate
/// optimizer and receipt.
pub struct AccelerateReducer {
    /// Last computed output MSE between teacher and student.
    pub output_mse: Option<f64>,
    /// Last computed cosine similarity between teacher and student outputs.
    pub cosine_similarity: Option<f64>,
    /// Last computed residual relative error: ‖teacher - student‖₂ / ‖teacher‖₂.
    pub residual_relative_error: Option<f64>,
    /// Hidden dimension (output vector length).
    hidden_dim: usize,
}

impl AccelerateReducer {
    /// Create a new AccelerateReducer with default hidden dimension.
    pub fn new() -> Self {
        AccelerateReducer::with_hidden_dim(3840)
    }

    /// Create a new AccelerateReducer with a specific hidden dimension.
    pub fn with_hidden_dim(hidden_dim: usize) -> Self {
        AccelerateReducer {
            output_mse: None,
            cosine_similarity: None,
            residual_relative_error: None,
            hidden_dim,
        }
    }

    /// Compute reduction metrics between teacher and student outputs.
    ///
    /// Takes pre-computed teacher and student activation vectors (f32 slices)
    /// and computes three metrics using Accelerate vDSP primitives:
    ///
    /// * **MSE**: Mean squared error: `(1/N) Σ (t[i] - s[i])²`
    /// * **Cosine similarity**: `Σ t[i]·s[i] / (‖t‖ · ‖s‖)`
    /// * **Residual relative error**: `‖t - s‖₂ / ‖t‖₂`
    pub fn reduce(
        &mut self,
        _microbatch: usize,
        teacher_out: &[f32],
        student_out: &[f32],
    ) {
        let len = teacher_out.len().min(student_out.len()).min(self.hidden_dim);
        if len == 0 {
            self.output_mse = Some(f64::INFINITY);
            self.cosine_similarity = Some(0.0);
            self.residual_relative_error = Some(f64::INFINITY);
            return;
        }

        let t = &teacher_out[..len];
        let s = &student_out[..len];

        // ── MSE: (1/N) Σ (t[i] - s[i])² ──────────────────────────────────
        let mut diff = Vec::with_capacity(len);
        for i in 0..len {
            diff.push(t[i] - s[i]);
        }
        let mse = dot_product(&diff, &diff) / len as f32;
        self.output_mse = Some(mse as f64);

        // ── Cosine similarity ────────────────────────────────────────────
        let tt = dot_product(t, t).sqrt();
        let ss = dot_product(s, s).sqrt();
        let ts = dot_product(t, s);
        self.cosine_similarity = if tt > 1e-10 && ss > 1e-10 {
            Some((ts / (tt * ss)) as f64)
        } else {
            Some(if tt == ss { 1.0 } else { 0.0 })
        };

        // ── Residual relative error ──────────────────────────────────────
        let diff_norm = dot_product(&diff, &diff).sqrt();
        self.residual_relative_error = Some(if tt > 1e-10 {
            (diff_norm / tt) as f64
        } else if diff_norm > 1e-10 {
            f64::INFINITY
        } else {
            0.0
        });
    }

    /// Return the hidden dimension this reducer was configured with.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

impl Default for AccelerateReducer {
    fn default() -> Self {
        Self::new()
    }
}
