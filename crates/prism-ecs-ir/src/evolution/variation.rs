//! Adaptive variation operators for evolutionary compilation search.

use crate::evolution::foundation::MetalGeometryAxis;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VariationOperator {
    Unknown,
    Representation,
    Packing,
    Geometry,
    Decomposition,
    Memory,
    Fusion,
    Runtime,
    AneUnit,
}

const OPERATORS: [VariationOperator; 8] = [
    VariationOperator::Representation,
    VariationOperator::Packing,
    VariationOperator::Geometry,
    VariationOperator::Decomposition,
    VariationOperator::Memory,
    VariationOperator::Fusion,
    VariationOperator::Runtime,
    VariationOperator::AneUnit,
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct OperatorStats {
    pub attempts: u64,
    pub successes: u64,
    pub reward_sum: f64,
}

impl OperatorStats {
    pub fn mean_reward(self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.reward_sum / self.attempts as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveVariationController {
    pub stats: std::collections::HashMap<VariationOperator, OperatorStats>,
    pub exploration: f64,
    pub geometry_covariance: GeometryCovariance,
}

impl Default for AdaptiveVariationController {
    fn default() -> Self {
        Self {
            stats: OPERATORS
                .into_iter()
                .map(|op| (op, OperatorStats::default()))
                .collect(),
            exploration: 0.25,
            geometry_covariance: GeometryCovariance::default(),
        }
    }
}

impl AdaptiveVariationController {
    /// Combine evidence from another worker without replacing locally learned
    /// statistics. This is used when federated runtime snapshots converge.
    pub fn merge(&mut self, remote: &Self) {
        for (operator, incoming) in &remote.stats {
            let current = self.stats.entry(*operator).or_default();
            current.attempts = current.attempts.saturating_add(incoming.attempts);
            current.successes = current.successes.saturating_add(incoming.successes);
            current.reward_sum += incoming.reward_sum;
        }
        let local_weight = self.geometry_covariance.observations as f64;
        let remote_weight = remote.geometry_covariance.observations as f64;
        let total = local_weight + remote_weight;
        if total > 0.0 {
            for index in 0..4 {
                self.geometry_covariance.mean[index] = (self.geometry_covariance.mean[index]
                    * local_weight
                    + remote.geometry_covariance.mean[index] * remote_weight)
                    / total;
                self.geometry_covariance.variance[index] =
                    (self.geometry_covariance.variance[index] * local_weight
                        + remote.geometry_covariance.variance[index] * remote_weight)
                        / total;
            }
            for row in 0..4 {
                for column in 0..4 {
                    self.geometry_covariance.covariance[row][column] =
                        (self.geometry_covariance.covariance[row][column] * local_weight
                            + remote.geometry_covariance.covariance[row][column] * remote_weight)
                            / total;
                }
            }
            self.geometry_covariance.observations = self
                .geometry_covariance
                .observations
                .saturating_add(remote.geometry_covariance.observations);
        }
    }
}

/// Diagonal CMA-style online covariance estimate for the correlated hardware
/// regime [tile_m, tile_n, tile_k, shared_memory]. A diagonal estimate is the
/// safe first implementation for bounded integer hardware parameters; the
/// shared update still couples proposal scale across all four dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryCovariance {
    pub mean: [f64; 4],
    pub variance: [f64; 4],
    #[serde(default = "default_geometry_covariance_matrix")]
    pub covariance: [[f64; 4]; 4],
    pub observations: u64,
    pub learning_rate: f64,
}

fn default_geometry_covariance_matrix() -> [[f64; 4]; 4] {
    [
        [1024.0, 0.0, 0.0, 0.0],
        [0.0, 1024.0, 0.0, 0.0],
        [0.0, 0.0, 256.0, 0.0],
        [0.0, 0.0, 0.0, 16384.0 * 16384.0],
    ]
}

impl Default for GeometryCovariance {
    fn default() -> Self {
        Self {
            mean: [64.0, 64.0, 32.0, 65536.0],
            variance: [1024.0, 1024.0, 256.0, 16384.0 * 16384.0],
            covariance: default_geometry_covariance_matrix(),
            observations: 0,
            learning_rate: 0.2,
        }
    }
}

impl GeometryCovariance {
    pub fn observe(&mut self, geometry: &MetalGeometryAxis, shared_memory_bytes: u32, score: f64) {
        if !score.is_finite() || score <= 0.0 {
            return;
        }
        let values = [
            geometry.grid_tile_m as f64,
            geometry.grid_tile_n as f64,
            geometry.grid_tile_k as f64,
            shared_memory_bytes as f64,
        ];
        self.observations = self.observations.saturating_add(1);
        let weight = self.learning_rate * score.clamp(0.0, 1.0);
        let mut deltas = [0.0; 4];
        for (index, ((mean, variance), value)) in self
            .mean
            .iter_mut()
            .zip(self.variance.iter_mut())
            .zip(values)
            .enumerate()
        {
            let delta = value - *mean;
            deltas[index] = delta;
            *mean += weight * delta;
            *variance = ((1.0 - weight) * *variance + weight * delta * delta).max(1.0);
        }
        for row in 0..4 {
            for column in 0..4 {
                let updated = (1.0 - weight) * self.covariance[row][column]
                    + weight * deltas[row] * deltas[column];
                self.covariance[row][column] = if row == column {
                    updated.max(1.0)
                } else {
                    updated
                };
            }
        }
    }

    pub fn sample(&self, rng: &mut impl Rng) -> (MetalGeometryAxis, u32) {
        let mut lower = [[0.0; 4]; 4];
        for row in 0..4 {
            for column in 0..=row {
                let prior = (0..column)
                    .map(|index| lower[row][index] * lower[column][index])
                    .sum::<f64>();
                if row == column {
                    lower[row][column] = (self.covariance[row][row] - prior).max(1.0).sqrt();
                } else {
                    lower[row][column] =
                        (self.covariance[row][column] - prior) / lower[column][column].max(1e-9);
                }
            }
        }
        let standard = [
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        ];
        let sample = |index: usize| {
            self.mean[index]
                + (0..=index)
                    .map(|column| lower[index][column] * standard[column])
                    .sum::<f64>()
        };
        let clamp =
            |value: f64, min: u32, max: u32| value.round().clamp(min as f64, max as f64) as u32;
        (
            MetalGeometryAxis {
                threadgroup_width: 32,
                threadgroup_height: 8,
                grid_tile_m: clamp(sample(0), 1, 256),
                grid_tile_n: clamp(sample(1), 1, 256),
                grid_tile_k: clamp(sample(2), 1, 128),
            },
            clamp(sample(3), 4096, 262144),
        )
    }
}

impl AdaptiveVariationController {
    pub fn observe_geometry(
        &mut self,
        geometry: &MetalGeometryAxis,
        shared_memory_bytes: u32,
        score: f64,
    ) {
        self.geometry_covariance
            .observe(geometry, shared_memory_bytes, score);
    }

    pub fn sample_geometry(&self, rng: &mut impl Rng) -> (MetalGeometryAxis, u32) {
        self.geometry_covariance.sample(rng)
    }

    pub fn select(&self, rng: &mut impl Rng) -> VariationOperator {
        let total: u64 = self.stats.values().map(|s| s.attempts).sum();
        let log_total = (total.max(1) as f64).ln();
        let weights: Vec<f64> = OPERATORS
            .iter()
            .map(|op| {
                let stats = self.stats.get(op).copied().unwrap_or_default();
                if stats.attempts == 0 {
                    f64::INFINITY
                } else {
                    stats.mean_reward()
                        + self.exploration * (log_total / stats.attempts as f64).sqrt()
                }
            })
            .collect();
        if let Some(index) = weights.iter().position(|weight| weight.is_infinite()) {
            return OPERATORS[index];
        }
        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let adjusted: Vec<f64> = weights
            .iter()
            .map(|weight| (weight - min).max(0.001))
            .collect();
        let total_weight: f64 = adjusted.iter().sum();
        let mut draw = rng.gen::<f64>() * total_weight;
        for (index, weight) in adjusted.iter().enumerate() {
            if draw <= *weight {
                return OPERATORS[index];
            }
            draw -= weight;
        }
        *OPERATORS.last().unwrap()
    }

    pub fn record(&mut self, operator: VariationOperator, reward: f64) {
        let stats = self.stats.entry(operator).or_default();
        stats.attempts += 1;
        if reward > 0.0 {
            stats.successes += 1;
        }
        stats.reward_sum += reward.clamp(-1.0, 1.0);
    }
}

/// Correlated geometry proposal. The same scale factor is applied to M/N/K
/// and shared memory, preserving useful hardware regimes instead of treating
/// correlated compiler parameters as independent genes.
pub fn correlated_geometry_mutation(
    geometry: &MetalGeometryAxis,
    shared_memory_bytes: u32,
    rng: &mut impl Rng,
) -> (MetalGeometryAxis, u32) {
    let scale_up = rng.gen::<f64>() >= 0.5;
    let factor = if scale_up { 2 } else { 1 };
    let divisor = if scale_up { 1 } else { 2 };
    let scale = |value: u32, max: u32| (value.saturating_mul(factor) / divisor).clamp(1, max);
    (
        MetalGeometryAxis {
            threadgroup_width: scale(geometry.threadgroup_width, 256),
            threadgroup_height: scale(geometry.threadgroup_height, 64),
            grid_tile_m: scale(geometry.grid_tile_m, 256),
            grid_tile_n: scale(geometry.grid_tile_n, 256),
            grid_tile_k: scale(geometry.grid_tile_k, 128),
        },
        scale(shared_memory_bytes, 262_144).max(4096),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn untried_operators_are_explored() {
        let controller = AdaptiveVariationController::default();
        let mut rng = StdRng::seed_from_u64(7);
        assert!(matches!(
            controller.select(&mut rng),
            VariationOperator::Representation
        ));
    }

    #[test]
    fn geometry_mutation_keeps_parameters_correlated() {
        let geometry = MetalGeometryAxis::default();
        let mut rng = StdRng::seed_from_u64(4);
        let (mutated, memory) = correlated_geometry_mutation(&geometry, 65536, &mut rng);
        assert_eq!(mutated.grid_tile_m, mutated.grid_tile_n);
        assert!((4096..=262144).contains(&memory));
    }

    #[test]
    fn covariance_update_moves_toward_successful_regime() {
        let mut covariance = GeometryCovariance::default();
        let geometry = MetalGeometryAxis {
            grid_tile_m: 128,
            grid_tile_n: 128,
            grid_tile_k: 64,
            ..Default::default()
        };
        covariance.observe(&geometry, 131072, 1.0);
        assert!(covariance.observations == 1);
        assert!(covariance.mean[0] > 64.0);
        assert!(covariance.mean[3] > 65536.0);
    }

    #[test]
    fn covariance_tracks_correlated_geometry_and_merges() {
        let mut local = GeometryCovariance::default();
        let geometry = MetalGeometryAxis {
            grid_tile_m: 128,
            grid_tile_n: 128,
            grid_tile_k: 64,
            ..Default::default()
        };
        local.observe(&geometry, 131072, 1.0);
        local.observe(&geometry, 131072, 1.0);
        assert!(local.covariance[0][1] > 0.0);

        let mut controller = AdaptiveVariationController::default();
        controller.merge(&AdaptiveVariationController {
            geometry_covariance: local.clone(),
            ..AdaptiveVariationController::default()
        });
        assert_eq!(
            controller.geometry_covariance.covariance[0][1],
            local.covariance[0][1]
        );
    }
}
