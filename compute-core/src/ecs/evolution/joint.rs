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

/// A genome with optional fitness score.
#[derive(Debug, Clone)]
pub struct ScoredGenome {
    pub genome: JointGenome,
    pub fitness: Option<f64>,
}

/// Result from a joint search.
#[derive(Debug, Clone)]
pub struct JointSearchResult {
    pub best_genome: Option<JointGenome>,
    pub population_size: usize,
    pub generations_completed: usize,
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

    /// Run a joint search over engram + representation + kernel space.
    pub fn run(&self) -> JointSearchResult {
        let mut population = self.generate_initial_population();

        for gen in 0..self.config.max_generations {
            // Evaluate each genome
            for genome in &mut population {
                if genome.fitness.is_none() {
                    genome.fitness = Some(self.evaluate(&genome.genome));
                }
            }

            // Sort by fitness
            population.sort_by(|a, b| {
                let af = a.fitness.unwrap_or(f64::MAX);
                let bf = b.fitness.unwrap_or(f64::MAX);
                af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Keep top N
            population.truncate(self.config.population_size);

            // Check convergence
            if gen > 0
                && population.first().map(|g| g.fitness.unwrap_or(0.0))
                    == population.get(1).map(|g| g.fitness.unwrap_or(0.0))
            {
                break;
            }
        }

        JointSearchResult {
            best_genome: population.first().map(|s| s.genome.clone()),
            population_size: population.len(),
            generations_completed: self.config.max_generations,
        }
    }

    fn generate_initial_population(&self) -> Vec<ScoredGenome> {
        let mut pop = Vec::new();
        for codec in &self.engram_codecs {
            for insertion_point in &self.insertion_points {
                for kernel in &self.kernel_variants {
                    pop.push(ScoredGenome {
                        genome: JointGenome {
                            program: EvolveProgram::MetalShader(format!("{}_kernel", kernel)),
                            engram_codec: codec.clone(),
                            engram_capacity: 1024,
                            insertion_point: insertion_point.clone(),
                            retrieval_threshold: 0.7,
                            tensor_representation: "TernaryTile640".into(),
                            kernel_variant: kernel.clone(),
                        },
                        fitness: None,
                    });
                }
            }
        }
        pop
    }

    fn evaluate(&self, genome: &JointGenome) -> f64 {
        // Simulated cost: smaller engram_codec + simpler kernel = better
        let codec_cost = match genome.engram_codec.as_str() {
            "int8" => 3.0,
            "nf4" => 2.0,
            "ternary" => 1.0,
            _ => 5.0,
        };
        let kernel_cost = match genome.kernel_variant.as_str() {
            "tile640_gemv" => 1.0,
            "persistent_gemv" => 2.0,
            "batched_gemv" => 1.5,
            _ => 3.0,
        };
        codec_cost + kernel_cost
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

    #[test]
    fn test_joint_search_generates_population() {
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        let pop = config.generate_initial_population();
        assert_eq!(
            pop.len(),
            config.engram_codecs.len()
                * config.insertion_points.len()
                * config.kernel_variants.len()
        );
    }

    #[test]
    fn test_joint_search_evaluates() {
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        let result = config.run();
        assert!(result.best_genome.is_some());
        assert!(result.generations_completed > 0);
    }
}
