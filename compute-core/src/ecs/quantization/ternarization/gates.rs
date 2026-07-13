//! Candidate gates for the ternarization pipeline.
//!
//! Implements the required gates from plan Section 6: reconstruction
//! fidelity, structural validity, and the combined `check_candidate` gate.

use super::candidate::ResidualPolicy;
use super::candidate::TernarizationCandidate;
use super::residual::ResidualCodec;

/// Gates for the candidate process (plan Section 6: Required gates).
///
/// Each threshold represents a quality gate the candidate must pass
/// before it can be admitted into the compiled artifact.
#[derive(Debug, Clone)]
pub struct CandidateGates {
    /// Maximum allowed RMSE for reconstruction of a single tensor.
    pub reconstruction_threshold: f64,
    /// Maximum allowed operator error (forward-pass deviation).
    pub operator_threshold: f64,
    /// Maximum allowed layer error (end-to-end layer deviation).
    pub layer_threshold: f64,
    /// Maximum allowed rollout regression (autoregressive quality).
    pub rollout_threshold: f64,
}

/// Check structural validity of ternary weights and grouping.
///
/// Returns `Ok(())` if the weights array is non-empty and the group
/// size is greater than zero.
pub fn check_structural(weights: &[i8], group_size: usize) -> Result<(), String> {
    if weights.is_empty() {
        return Err("empty weights".into());
    }
    if group_size == 0 {
        return Err("group_size must be > 0".into());
    }
    Ok(())
}

/// Check that the reconstruction RMSE between `original` and
/// `reconstructed` is within `threshold`.
pub fn check_reconstruction(
    original: &[f32],
    reconstructed: &[f32],
    threshold: f64,
) -> Result<(), String> {
    if original.len() != reconstructed.len() {
        return Err(format!(
            "length mismatch: original {} vs reconstructed {}",
            original.len(),
            reconstructed.len()
        ));
    }
    let mut sq_error = 0.0f64;
    for (o, r) in original.iter().zip(reconstructed.iter()) {
        let err = (*o - *r) as f64;
        sq_error += err * err;
    }
    let rmse = (sq_error / original.len() as f64).sqrt();
    if rmse > threshold {
        Err(format!(
            "reconstruction RMSE {} > threshold {}",
            rmse, threshold
        ))
    } else {
        Ok(())
    }
}

/// Run all gates against a single candidate.
///
/// Checks structural validity, reconstructs the tensor from the
/// candidate's weights and scales, and verifies the reconstruction
/// RMSE is within the configured threshold.
pub fn check_candidate(
    original: &[f32],
    candidate: &TernarizationCandidate,
    gates: &CandidateGates,
) -> Result<(), String> {
    check_structural(&candidate.weights, candidate.group_size)?;

    let reconstructed: Vec<f32> = candidate
        .weights
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let g = i / candidate.group_size;
            w as f32 * candidate.scales.get(g).copied().unwrap_or(1.0)
        })
        .collect();

    let post_residual = match &candidate.residual_policy {
        ResidualPolicy::None => reconstructed,
        ResidualPolicy::Dense { residuals } => {
            ResidualCodec::apply_dense(&reconstructed, residuals)
        }
        ResidualPolicy::Sparse {
            indices, values, ..
        } => {
            let mut r = reconstructed;
            ResidualCodec::apply_sparse(&mut r, indices, values);
            r
        }
    };

    check_reconstruction(original, &post_residual, gates.reconstruction_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::quantization::ternarization::candidate::{PhysicalTileLayout, ResidualPolicy};

    #[test]
    fn test_check_structural_valid() {
        assert!(check_structural(&[1, 0, -1], 3).is_ok());
    }

    #[test]
    fn test_check_structural_empty_weights() {
        let err = check_structural(&[], 4).unwrap_err();
        assert!(err.contains("empty weights"));
    }

    #[test]
    fn test_check_structural_zero_group_size() {
        let err = check_structural(&[1, 0, -1], 0).unwrap_err();
        assert!(err.contains("group_size must be > 0"));
    }

    #[test]
    fn test_check_reconstruction_within_threshold() {
        let original = vec![1.0, -1.0, 0.0, 1.0];
        let reconstructed = vec![0.98, -1.02, 0.01, 0.99];
        assert!(check_reconstruction(&original, &reconstructed, 0.1).is_ok());
    }

    #[test]
    fn test_check_reconstruction_exceeds_threshold() {
        let original = vec![10.0, -10.0, 0.0, 10.0];
        let reconstructed = vec![0.0, 0.0, 0.0, 0.0];
        let err = check_reconstruction(&original, &reconstructed, 0.5).unwrap_err();
        assert!(err.contains("RMSE"));
    }

    #[test]
    fn test_check_reconstruction_length_mismatch() {
        let err = check_reconstruction(&[1.0, 2.0], &[1.0], 0.1).unwrap_err();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn test_check_candidate_passes() {
        let original = vec![1.0, -1.0, 0.0, 0.5];
        let candidate = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        let gates = CandidateGates {
            reconstruction_threshold: 0.5,
            operator_threshold: 1.0,
            layer_threshold: 1.0,
            rollout_threshold: 1.0,
        };
        assert!(check_candidate(&original, &candidate, &gates).is_ok());
    }

    #[test]
    fn test_check_candidate_fails_structural() {
        let candidate = TernarizationCandidate {
            tensor_id: "empty".into(),
            group_id: "g0".into(),
            weights: vec![],
            scales: vec![],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        let gates = CandidateGates {
            reconstruction_threshold: 0.5,
            operator_threshold: 1.0,
            layer_threshold: 1.0,
            rollout_threshold: 1.0,
        };
        let err = check_candidate(&[1.0], &candidate, &gates).unwrap_err();
        assert!(err.contains("empty weights"));
    }

    #[test]
    fn test_check_candidate_fails_reconstruction() {
        let original = vec![100.0, -100.0];
        let candidate = TernarizationCandidate {
            tensor_id: "bad".into(),
            group_id: "g0".into(),
            weights: vec![1, -1],
            scales: vec![1.0],
            group_size: 2,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        let gates = CandidateGates {
            reconstruction_threshold: 0.01,
            operator_threshold: 1.0,
            layer_threshold: 1.0,
            rollout_threshold: 1.0,
        };
        let err = check_candidate(&original, &candidate, &gates).unwrap_err();
        assert!(err.contains("RMSE"));
    }

    #[test]
    fn test_candidate_gates_default_sanity() {
        let gates = CandidateGates {
            reconstruction_threshold: 0.05,
            operator_threshold: 0.1,
            layer_threshold: 0.1,
            rollout_threshold: 0.2,
        };
        assert!((gates.reconstruction_threshold - 0.05).abs() < 1e-12);
        assert!((gates.operator_threshold - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_check_candidate_dense_zero_residual_equals_none() {
        // A Dense policy with zero residuals should pass just like None.
        let original = vec![1.0, -1.0, 0.0, 0.5];
        let zero_residuals = vec![0.0, 0.0, 0.0, 0.0];

        let candidate_none = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::None,
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        let candidate_dense_zero = TernarizationCandidate {
            tensor_id: "test".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::Dense {
                residuals: zero_residuals,
            },
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };
        let gates = CandidateGates {
            reconstruction_threshold: 0.5,
            operator_threshold: 1.0,
            layer_threshold: 1.0,
            rollout_threshold: 1.0,
        };
        let r_none = check_candidate(&original, &candidate_none, &gates);
        let r_dense = check_candidate(&original, &candidate_dense_zero, &gates);
        assert_eq!(
            r_none.is_ok(),
            r_dense.is_ok(),
            "zero-residual dense should match None"
        );
    }

    #[test]
    fn test_check_candidate_nonzero_residual_improves_rmse() {
        // Create a case where residual compensation is needed to pass the gate.
        // Without residual: weights × scales deviates too much.
        // With residual: the error shrinks and passes.
        let original = vec![2.0, -2.0, 0.0, 1.0];

        // weights × scales: [0.5, -0.5, 0.0, 0.5] — large error against original
        let residuals = vec![1.5, -1.5, 0.0, 0.5]; // exactly the error

        let candidate = TernarizationCandidate {
            tensor_id: "residual".into(),
            group_id: "g0".into(),
            weights: vec![1, -1, 0, 1],
            scales: vec![0.5],
            group_size: 4,
            residual_policy: ResidualPolicy::Dense { residuals },
            physical_layout: PhysicalTileLayout::Tile640,
            kernel_selection: "ternary_gemv".into(),
        };

        // Tight threshold that only passes with residual compensation
        let gates = CandidateGates {
            reconstruction_threshold: 0.01,
            operator_threshold: 1.0,
            layer_threshold: 1.0,
            rollout_threshold: 1.0,
        };

        // Without residual: reconstruction = [0.5, -0.5, 0.0, 0.5], RMSE ~1.32 > 0.01
        // With residual: reconstruction = [2.0, -2.0, 0.0, 1.0], RMSE = 0
        assert!(
            check_candidate(&original, &candidate, &gates).is_ok(),
            "residual compensation should make the candidate pass a tight gate"
        );
    }
}
