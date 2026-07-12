//! Metal decomposition search — evolves tile geometry and reduction strategies
//! for NF4/ternary operations using the evolutionary search engine (M7).
//!
//! M8: Connects canonical Metal fragments (M4) to the evolution pipeline.
//! Each [`MetalDecompositionSearch`] produces a [`DecompositionResult`] that
//! captures the winning program and its cost.

use crate::ecs::component::backend::BackendTarget;
use crate::ecs::evolution::foundation::{CostFunction, CostMetrics, EvolveProgram, SearchConfig};
use crate::ecs::plan::CodecFamily;

/// Configuration for a metal decomposition search.
///
/// Wraps a tensor id, target backend, codec family, and search config.
/// Construct via [`Self::for_nf4`] or [`Self::for_ternary`].
#[derive(Debug, Clone)]
pub struct MetalDecompositionSearch {
    pub tensor_id: String,
    pub backend: BackendTarget,
    pub format: CodecFamily,
    pub config: SearchConfig,
}

impl MetalDecompositionSearch {
    /// Create a search for NF4 tile geometry on the given backend.
    pub fn for_nf4(tensor_id: &str, backend: BackendTarget) -> Self {
        Self {
            tensor_id: tensor_id.to_string(),
            backend,
            format: CodecFamily::Nf4,
            config: SearchConfig {
                population_size: 8,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 50,
                convergence_threshold: 0.01,
                cost_function: CostFunction::WallTime,
            },
        }
    }

    /// Create a search for ternary tile geometry on the given backend.
    pub fn for_ternary(tensor_id: &str, backend: BackendTarget) -> Self {
        Self {
            tensor_id: tensor_id.to_string(),
            backend,
            format: CodecFamily::Ternary,
            config: SearchConfig {
                population_size: 8,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 50,
                convergence_threshold: 0.01,
                cost_function: CostFunction::WallTime,
            },
        }
    }
}

/// Results from a decomposition search.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub tensor_id: String,
    pub format: CodecFamily,
    pub generations: u64,
    pub winning_program: EvolveProgram,
    pub cost: CostMetrics,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_decomposition_config() {
        let search = MetalDecompositionSearch::for_nf4("test.weight", BackendTarget::Metal);
        assert_eq!(search.config.population_size, 8);
        assert_eq!(search.format, CodecFamily::Nf4);
    }

    #[test]
    fn test_ternary_decomposition_config() {
        let search = MetalDecompositionSearch::for_ternary("test.weight", BackendTarget::Metal);
        assert_eq!(search.config.cost_function, CostFunction::WallTime);
    }

    #[test]
    fn test_backend_target_variant() {
        // Verify that BackendTarget::Metal is the correct variant for
        // Metal compilation targets in the evolution-specific enum.
        assert_eq!(format!("{:?}", BackendTarget::Metal), "Metal");
    }
}
