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

/// Crossover two genomes: mix program from parent A, engram_codec from parent B,
/// insertion point from A, kernel variant from A, and average the thresholds.
fn joint_crossover(a: &JointGenome, b: &JointGenome) -> JointGenome {
    JointGenome {
        program: a.program.clone(),
        engram_codec: b.engram_codec.clone(),
        engram_capacity: a.engram_capacity.max(b.engram_capacity),
        insertion_point: a.insertion_point.clone(),
        retrieval_threshold: (a.retrieval_threshold + b.retrieval_threshold) / 2.0,
        tensor_representation: b.tensor_representation.clone(),
        kernel_variant: a.kernel_variant.clone(),
    }
}

/// Mutate a genome: slightly perturb retrieval_threshold or switch to a
/// different codec/kernel. Uses `seed` as a source of deterministic variation
/// — no external RNG required.
fn joint_mutate(genome: &JointGenome, config: &JointSearchConfig, seed: usize) -> JointGenome {
    let mut result = genome.clone();
    match seed % 5 {
        0 => {
            // Perturb retrieval threshold by a small delta
            let delta = (seed as f64 * 0.07 - 0.03) * 0.1;
            result.retrieval_threshold = (result.retrieval_threshold + delta).clamp(0.1, 1.0);
        }
        1 => {
            // Switch to a different engram codec
            if !config.engram_codecs.is_empty() {
                let idx = seed % config.engram_codecs.len();
                result.engram_codec = config.engram_codecs[idx].clone();
            }
        }
        2 => {
            // Switch insertion point
            if !config.insertion_points.is_empty() {
                let idx = seed % config.insertion_points.len();
                result.insertion_point = config.insertion_points[idx].clone();
            }
        }
        3 => {
            // Switch kernel variant
            if !config.kernel_variants.is_empty() {
                let idx = seed % config.kernel_variants.len();
                result.kernel_variant = config.kernel_variants[idx].clone();
            }
        }
        _ => {
            // Toggle tensor representation
            result.tensor_representation = if result.tensor_representation == "TernaryTile640" {
                "Int8Tile640".into()
            } else {
                "TernaryTile640".into()
            };
        }
    }
    result
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
    ///
    /// Each generation: evaluates all un-scored genomes, sorts by fitness,
    /// keeps the top N, checks for convergence, then refills the population
    /// with crossover + mutation offspring so evolution actually progresses.
    pub fn run(&self) -> JointSearchResult {
        let mut population = self.generate_initial_population();
        let mut generations_completed = 0;

        for gen in 0..self.config.max_generations {
            generations_completed = gen + 1;

            // Evaluate each genome
            for genome in &mut population {
                if genome.fitness.is_none() {
                    genome.fitness = Some(self.evaluate(&genome.genome));
                }
            }

            // Sort by fitness (lower is better)
            population.sort_by(|a, b| {
                let af = a.fitness.unwrap_or(f64::MAX);
                let bf = b.fitness.unwrap_or(f64::MAX);
                af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Check convergence on the FULL sorted population BEFORE elite
            // truncation — otherwise the small elite pool can look artificially
            // converged and the refill never fires.
            if gen > 0 {
                if let (Some(best), Some(second)) = (
                    population.first().and_then(|g| g.fitness),
                    population.get(1).and_then(|g| g.fitness),
                ) {
                    if (best - second).abs()
                        < self.config.convergence_threshold * best.abs().max(1.0)
                    {
                        break;
                    }
                }
            }

            // Keep only elite subset (30%) — makes room for offspring
            let budget = self.config.population_size;
            let elite_count = (budget as f64 * 0.3).ceil() as usize;
            let elite_count = elite_count.max(1).min(budget.saturating_sub(1));
            population.truncate(elite_count);

            // Refill population with mutated/crossover offspring so evolution
            // actually produces diversity each generation.
            let mut i = 0;
            while population.len() < budget {
                let idx_a = i % population.len();
                let idx_b = (i + 1) % population.len();
                let child_genome = {
                    let parent_a = &population[idx_a];
                    let parent_b = &population[idx_b];
                    joint_crossover(&parent_a.genome, &parent_b.genome)
                };
                let mutated = joint_mutate(&child_genome, self, i);
                population.push(ScoredGenome {
                    genome: mutated,
                    fitness: None,
                });
                i += 1;
            }
        }

        JointSearchResult {
            best_genome: population.first().map(|s| s.genome.clone()),
            population_size: population.len(),
            generations_completed,
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

    #[test]
    fn test_joint_crossover() {
        let a = JointGenome {
            program: EvolveProgram::MetalShader("tile640_gemv_kernel".into()),
            engram_codec: "nf4".into(),
            engram_capacity: 512,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
            tensor_representation: "TernaryTile640".into(),
            kernel_variant: "tile640_gemv".into(),
        };
        let b = JointGenome {
            program: EvolveProgram::MetalShader("persistent_gemv_kernel".into()),
            engram_codec: "int8".into(),
            engram_capacity: 2048,
            insertion_point: "after.linear.k_proj".into(),
            retrieval_threshold: 0.5,
            tensor_representation: "Int8Tile640".into(),
            kernel_variant: "persistent_gemv".into(),
        };
        let child = joint_crossover(&a, &b);
        // program from a, engram_codec from b
        assert_eq!(format!("{:?}", child.program), format!("{:?}", a.program));
        assert_eq!(child.engram_codec, b.engram_codec);
        // capacity is max
        assert_eq!(child.engram_capacity, 2048);
        // threshold is average
        assert!((child.retrieval_threshold - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_joint_mutate_perturbs_threshold() {
        let genome = JointGenome {
            program: EvolveProgram::MetalShader("kernel".into()),
            engram_codec: "nf4".into(),
            engram_capacity: 1024,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
            tensor_representation: "TernaryTile640".into(),
            kernel_variant: "tile640_gemv".into(),
        };
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        // seed=0 triggers threshold perturbation (seed%5==0)
        let mutated = joint_mutate(&genome, &config, 0);
        assert_ne!(mutated.retrieval_threshold, 0.7);
        assert!((0.1..=1.0).contains(&mutated.retrieval_threshold));
    }

    #[test]
    fn test_offspring_generated() {
        // Verify that offspring are bred each generation: the elite truncation
        // (keeping 30%) creates room for crossover+mutate offspring,
        // which are evaluated in subsequent generations. Without the fix,
        // the refill loop `while next_gen.len() < population_size` never fires
        // because `next_gen` starts at full capacity.
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        let result = config.run();

        assert!(
            result.generations_completed > 0,
            "search should complete at least one generation"
        );
        assert!(
            result.best_genome.is_some(),
            "search should produce a best genome"
        );
        // With elite truncation the population should be refilled to budget
        assert_eq!(
            result.population_size, 10,
            "population should be at budget ({}) after breeding, got {}",
            10, result.population_size,
        );

        let best = result.best_genome.unwrap();
        assert!(
            !best.engram_codec.is_empty(),
            "best genome should have an engram codec"
        );
        assert!(
            !best.kernel_variant.is_empty(),
            "best genome should have a kernel variant"
        );
        // With elite truncation, multiple generations should run
        assert!(
            result.generations_completed > 1,
            "search should run multiple generations (got {}): \
             offspring were not bred",
            result.generations_completed,
        );
    }
}
