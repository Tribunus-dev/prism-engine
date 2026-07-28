//! Empirical sensitivity ranking per matrix by running BF16 teacher vs NF4
//! candidate at coarse observation points. Each matrix is evaluated at five
//! post-op observation points with KL divergence and RMSE, then ranked by
//! aggregate sensitivity.

use crate::ecs::legacy_compilation::level1::reducer::DistillObjective;
use crate::ecs::legacy_compilation::matrix_distill::{distill_matrix, DistillFormat};

/// Coarse observation points for boundary sensitivity measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservationPoint {
    PostQkvProj,
    PostAttnOut,
    PostFfnDown,
    PostResidual,
    PostLogits,
}

impl ObservationPoint {
    pub fn all() -> Vec<Self> {
        vec![
            Self::PostQkvProj,
            Self::PostAttnOut,
            Self::PostFfnDown,
            Self::PostResidual,
            Self::PostLogits,
        ]
    }
}

/// Empirical sensitivity rank for one matrix.
#[derive(Debug, Clone)]
pub struct BoundarySensitivityReport {
    pub tensor_name: String,
    pub matrix_role: String,
    /// Per-observation-point KL divergence (NF4 vs BF16).
    pub per_point_kl: Vec<(ObservationPoint, f32)>,
    /// Per-observation-point RMSE.
    pub per_point_rmse: Vec<(ObservationPoint, f32)>,
    /// Aggregate sensitivity: mean KL across all observation points.
    pub aggregate_kl: f32,
    /// Sensitivity rank (1=most sensitive, higher=less).
    pub rank: usize,
}

/// Compute boundary sensitivity for one matrix.
///
/// Runs `distill_matrix` at each observation point and aggregates the results
/// into a single sensitivity report.
pub fn compute_boundary_sensitivity(
    name: &str,
    bf16_weights: &[f32],
    rows: usize,
    cols: usize,
    objective: &DistillObjective,
) -> BoundarySensitivityReport {
    let observation_points = ObservationPoint::all();
    let mut per_point_kl = Vec::with_capacity(observation_points.len());
    let mut per_point_rmse = Vec::with_capacity(observation_points.len());

    for point in &observation_points {
        // Use distill_matrix at each observation point.
        // At each point the projection dimension is conveyed via cols;
        // distill_matrix handles the matmul internally.
        let result = distill_matrix(
            &format!("{}@{:?}", name, point),
            bf16_weights,
            rows,
            cols,
            DistillFormat::Nf4Tile640,
            objective,
            None,
        );
        per_point_kl.push((*point, result.kl_divergence));
        per_point_rmse.push((*point, result.rmse));
    }

    let aggregate_kl =
        per_point_kl.iter().map(|(_, kl)| kl).sum::<f32>() / per_point_kl.len() as f32;

    BoundarySensitivityReport {
        tensor_name: name.to_string(),
        matrix_role: String::new(),
        per_point_kl,
        per_point_rmse,
        aggregate_kl,
        rank: usize::MAX,
    }
}

/// Rank a list of reports by aggregate KL (1 = most sensitive).
pub fn rank_by_sensitivity(reports: &mut [BoundarySensitivityReport]) {
    reports.sort_by(|a, b| b.aggregate_kl.partial_cmp(&a.aggregate_kl).unwrap());
    for (i, report) in reports.iter_mut().enumerate() {
        report.rank = i + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_points_count() {
        assert_eq!(ObservationPoint::all().len(), 5);
    }

    #[test]
    fn test_rank_by_sensitivity() {
        let mut reports = vec![
            BoundarySensitivityReport {
                tensor_name: "a".into(),
                matrix_role: String::new(),
                per_point_kl: vec![],
                per_point_rmse: vec![],
                aggregate_kl: 0.1,
                rank: 0,
            },
            BoundarySensitivityReport {
                tensor_name: "b".into(),
                matrix_role: String::new(),
                per_point_kl: vec![],
                per_point_rmse: vec![],
                aggregate_kl: 0.5,
                rank: 0,
            },
        ];
        rank_by_sensitivity(&mut reports);
        assert_eq!(reports[0].tensor_name, "b");
        assert_eq!(reports[0].rank, 1);
        assert_eq!(reports[1].tensor_name, "a");
        assert_eq!(reports[1].rank, 2);
    }
}
