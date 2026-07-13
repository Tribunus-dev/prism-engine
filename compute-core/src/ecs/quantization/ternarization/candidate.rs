//! Ternarization candidate types.
//!
//! A `TernarizationCandidate` represents a single tensor's ternary assignment:
//! per-element weights in {-1, 0, +1}, per-group scale factors, and metadata
//! for residual compensation and physical layout.

/// A ternarization candidate — a single tensor's ternary assignment.
#[derive(Debug, Clone)]
pub struct TernarizationCandidate {
    pub tensor_id: String,
    pub group_id: String,
    /// Ternary weights: each element is -1, 0, or +1.
    pub weights: Vec<i8>,
    /// Per-group scale factors.
    pub scales: Vec<f32>,
    /// Number of weights per scale group.
    pub group_size: usize,
    /// Residual compensation policy.
    pub residual_policy: ResidualPolicy,
    /// Physical tile layout for the candidate.
    pub physical_layout: PhysicalTileLayout,
    /// Kernel variant selected for this candidate.
    pub kernel_selection: String,
}

/// Policy for compensating residual errors after ternary quantization.
#[derive(Debug, Clone)]
pub enum ResidualPolicy {
    /// No residual compensation.
    None,
    /// Dense residual vector — one residual per element.
    Dense { residuals: Vec<f32> },
    /// Sparse residual vector — only significant residuals are stored.
    Sparse {
        indices: Vec<usize>,
        values: Vec<f32>,
    },
}

/// Physical tile layout variants for ternary candidates.
#[derive(Debug, Clone)]
pub enum PhysicalTileLayout {
    /// Tile-640 format (640-element tiles with metadata).
    Tile640,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_defaults() {
        let c = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, 0, -1, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        assert_eq!(c.weights.len(), 4);
        assert_eq!(c.scales.len(), 1);
    }

    #[test]
    fn test_residual_policy_dense() {
        let r = ResidualPolicy::Dense {
            residuals: vec![0.1, -0.2, 0.05, -0.1],
        };
        match &r {
            ResidualPolicy::Dense { residuals } => assert_eq!(residuals.len(), 4),
            _ => panic!("expected Dense"),
        }
    }

    #[test]
    fn test_residual_policy_sparse() {
        let r = ResidualPolicy::Sparse {
            indices: vec![0, 3],
            values: vec![0.1, -0.2],
        };
        match &r {
            ResidualPolicy::Sparse { indices, values } => {
                assert_eq!(indices.len(), 2);
                assert_eq!(values.len(), 2);
            }
            _ => panic!("expected Sparse"),
        }
    }

    #[test]
    fn test_residual_policy_none() {
        let r: ResidualPolicy = ResidualPolicy::None;
        match r {
            ResidualPolicy::None => {} // ok
            _ => panic!("expected None"),
        }
    }
}
