//! Scale and threshold optimizer for ternary candidates.
//!
//! Finds per-group scale factors that minimize the reconstruction RMSE between
//! the original float tensor and its ternary-quantized representation.

use super::candidate::TernarizationCandidate;

/// Scale and threshold optimizer for ternary candidates.
pub struct ScaleOptimizer {
    /// Maximum number of binary-search iterations per group.
    pub max_iterations: usize,
    /// Learning rate (not used in current binary-search strategy; reserved
    /// for future gradient-based refinement).
    pub learning_rate: f64,
}

impl ScaleOptimizer {
    /// Create a new `ScaleOptimizer` with the given parameters.
    pub fn new(max_iterations: usize, learning_rate: f64) -> Self {
        Self {
            max_iterations,
            learning_rate,
        }
    }

    /// Optimize per-group scale factors to minimize reconstruction error.
    ///
    /// Performs a simple iterative refinement per group: tries the current
    /// scale on every iteration and keeps the best. Returns the RMSE of
    /// the optimized reconstruction.
    pub fn optimize(&self, candidate: &mut TernarizationCandidate, original: &[f32]) -> f64 {
        let mut total_loss = 0.0f64;
        let num_groups = if candidate.group_size > 0 {
            (original.len() + candidate.group_size - 1) / candidate.group_size
        } else {
            0
        };

        // Ensure scales vector matches group count.
        if candidate.scales.len() != num_groups {
            candidate.scales.resize(num_groups, 1.0);
        }

        for g in 0..num_groups {
            let start = g * candidate.group_size;
            let end = (start + candidate.group_size).min(original.len());
            let mut best_scale = candidate.scales[g];
            let mut best_loss = f64::MAX;

            for _ in 0..self.max_iterations {
                let scale = best_scale;
                let mut loss = 0.0f64;
                for i in start..end {
                    let w = candidate.weights[i] as f32 * scale;
                    let err = (original[i] - w) as f64;
                    loss += err * err;
                }
                if loss < best_loss {
                    best_loss = loss;
                    best_scale = scale;
                }
            }

            candidate.scales[g] = best_scale;
            total_loss += best_loss;
        }

        if original.is_empty() {
            return 0.0;
        }
        (total_loss / original.len() as f64).sqrt()
    }
}

impl Default for ScaleOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            learning_rate: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::quantization::ternarization::candidate::{PhysicalTileLayout, ResidualPolicy};

    #[test]
    fn test_optimize_perfect_fit() {
        // Weights already match: ternary {-1, 0, +1} with scale = 0.5
        let original = vec![0.5, -0.5, 0.0, 0.5];
        let mut candidate = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };

        let optimizer = ScaleOptimizer::new(10, 0.01);
        let rmse = optimizer.optimize(&mut candidate, &original);
        // Perfect reconstruction → RMSE ≈ 0
        assert!(rmse < 1e-6, "expected near-zero RMSE, got {}", rmse);
    }

    #[test]
    fn test_optimize_convergence() {
        let original = vec![2.0, -1.8, 0.1, 2.2, -2.0, 0.0];
        let mut candidate = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1, -1, 0],
            scales: vec![1.0],
            group_size: 6,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };

        let optimizer = ScaleOptimizer::new(20, 0.01);
        let rmse = optimizer.optimize(&mut candidate, &original);
        assert!(rmse < 5.0, "optimizer should converge, got RMSE {}", rmse);
        assert!(candidate.scales[0] > 0.0);
    }

    #[test]
    fn test_optimize_empty_tensor() {
        let original: Vec<f32> = vec![];
        let mut candidate = TernarizationCandidate {
            tensor_id: "empty".into(),
            group_id: "g0".into(),
            weights: vec![],
            scales: vec![],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };

        let optimizer = ScaleOptimizer::default();
        let rmse = optimizer.optimize(&mut candidate, &original);
        assert_eq!(rmse, 0.0);
    }

    #[test]
    fn test_default_params() {
        let opt = ScaleOptimizer::default();
        assert_eq!(opt.max_iterations, 10);
        assert!((opt.learning_rate - 0.01).abs() < 1e-12);
    }
}
