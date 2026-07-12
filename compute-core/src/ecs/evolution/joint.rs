//! Joint engram + representation search.
//!
//! Extends the search genome to include engram parameters alongside
//! program decomposition, enabling Pareto-optimal tradeoffs between
//! quality, memory, latency, and energy.

use crate::ecs::evolution::foundation::{CostFunction, EvolveProgram, SearchConfig};

/// A joint genome — program variant plus engram configuration.
#[derive(Debug, Clone)]
pub struct JointGenome {
    pub program: EvolveProgram,
    pub engram_codec: String,
    pub engram_capacity: usize,
    pub insertion_point: String,
    pub retrieval_threshold: f64,
    pub tensor_representation: String,
    pub kernel_variant: String,
}

/// Configuration for a joint search.
#[derive(Debug, Clone)]
pub struct JointSearchConfig {
    pub tensor_id: String,
    pub config: SearchConfig,
    pub engram_codecs: Vec<String>,
    pub insertion_points: Vec<String>,
    pub retrieval_thresholds: Vec<f64>,
    pub kernel_variants: Vec<String>,
}

impl JointSearchConfig {
    pub fn for_tensor(tensor_id: &str) -> Self {
        Self {
            tensor_id: tensor_id.to_string(),
            config: SearchConfig {
                population_size: 10,
                mutation_rate: 0.25,
                crossover_rate: 0.25,
                max_generations: 75,
                convergence_threshold: 0.01,
                cost_function: CostFunction::Weighted {
                    wall: 0.4,
                    energy: 0.4,
                    bandwidth: 0.2,
                },
            },
            engram_codecs: vec!["nf4".into(), "ternary".into(), "int8".into()],
            insertion_points: vec![
                "after.linear.q_proj".into(),
                "after.linear.k_proj".into(),
                "after.linear.v_proj".into(),
                "after.linear.o_proj".into(),
            ],
            retrieval_thresholds: vec![0.5, 0.7, 0.9],
            kernel_variants: vec![
                "tile640_gemv".into(),
                "persistent_gemv".into(),
                "batched_gemv".into(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_search_config() {
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        assert_eq!(config.engram_codecs.len(), 3);
        assert_eq!(config.insertion_points.len(), 4);
        assert_eq!(config.kernel_variants.len(), 3);
    }

    #[test]
    fn test_joint_genome() {
        let genome = JointGenome {
            program: EvolveProgram::MetalShader("kernel".into()),
            engram_codec: "nf4".into(),
            engram_capacity: 1024,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
            tensor_representation: "TernaryTile640".into(),
            kernel_variant: "tile640_gemv".into(),
        };
        assert_eq!(genome.engram_codec, "nf4");
        assert_eq!(genome.kernel_variant, "tile640_gemv");
    }
}
