//! Joint engram + representation search.
//!
//! Extends the search genome to include engram parameters alongside
//! program decomposition, enabling Pareto-optimal tradeoffs between
//! quality, memory, latency, and energy.

use crate::ecs::canonical::identity::CandidateId;
use crate::ecs::evolution::evaluator::{CandidateEvaluator, Workload};
use crate::ecs::evolution::foundation::{
    CandidateGenome, CandidateStatus, CostFunction, DecompositionStrategy, EvolutionCandidate,
    MemoryConfig, MetalGeometry, SearchConfig,
};
use crate::ecs::cimage::PhysicalTileLayout;
use crate::execution_plan::CodecFamily;

/// Full joint genome — representation, kernel, and engram genes.
#[derive(Debug, Clone)]
pub struct JointGenome {
    // Representation genes
    pub codec: CodecFamily,
    pub group_size: usize,
    pub residual_policy: String, // "none", "dense", "sparse"

    // Kernel genes
    pub kernel_variant: String, // "tile640_gemv", "persistent_gemv", etc.
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub reduction_strategy: String, // "sequential", "split-k", "tree"

    // Engram genes
    pub engram_codec: String,
    pub engram_capacity: usize,
    pub insertion_point: String,
    pub retrieval_threshold: f64,
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
        // Representation: codec from A, group_size from B, residual_policy from A
        codec: a.codec,
        group_size: b.group_size,
        residual_policy: a.residual_policy.clone(),

        // Kernel: variant from B, tiles from A, reduction from B
        kernel_variant: b.kernel_variant.clone(),
        tile_m: a.tile_m,
        tile_n: a.tile_n,
        tile_k: a.tile_k,
        reduction_strategy: b.reduction_strategy.clone(),

        // Engram: codec from A, max capacity, insertion from B, averaged threshold
        engram_codec: a.engram_codec.clone(),
        engram_capacity: a.engram_capacity.max(b.engram_capacity),
        insertion_point: b.insertion_point.clone(),
        retrieval_threshold: (a.retrieval_threshold + b.retrieval_threshold) / 2.0,
    }
}

/// Mutate a genome: slightly perturb retrieval_threshold or switch to a
/// different codec/kernel. Uses `seed` as a source of deterministic variation
/// — no external RNG required.
fn joint_mutate(genome: &JointGenome, config: &JointSearchConfig, seed: usize) -> JointGenome {
    let mut result = genome.clone();
    match seed % 7 {
        n @ 0..=6 => match n {
            0 => {
                // Perturb retrieval threshold by a small delta
                let delta = (seed as f64 * 0.07 - 0.03) * 0.1;
                result.retrieval_threshold = (result.retrieval_threshold + delta).clamp(0.1, 1.0);
            }
            1 => {
                // Switch to a different codec family (representation gene)
                let codecs = [
                    CodecFamily::Nf4,
                    CodecFamily::Int8,
                    CodecFamily::Ternary,
                    CodecFamily::RawF32,
                ];
                let idx = seed % codecs.len();
                result.codec = codecs[idx];
            }
            2 => {
                // Perturb group_size
                let sizes = [16usize, 32, 64, 128, 256];
                let idx = seed % sizes.len();
                result.group_size = sizes[idx];
            }
            3 => {
                // Switch residual policy
                let policies = ["none", "dense", "sparse"];
                let idx = seed % policies.len();
                result.residual_policy = policies[idx].to_string();
            }
            4 => {
                // Switch to a different engram codec
                if !config.engram_codecs.is_empty() {
                    let idx = seed % config.engram_codecs.len();
                    result.engram_codec = config.engram_codecs[idx].clone();
                }
            }
            5 => {
                // Switch insertion point
                if !config.insertion_points.is_empty() {
                    let idx = seed % config.insertion_points.len();
                    result.insertion_point = config.insertion_points[idx].clone();
                }
            }
            6 => {
                // Switch kernel variant
                if !config.kernel_variants.is_empty() {
                    let idx = seed % config.kernel_variants.len();
                    result.kernel_variant = config.kernel_variants[idx].clone();
                }
            }
            _ => unreachable!(),
        },
        _ => {}
    }
    result
}

impl JointSearchConfig {
    /// Build a full-search config spanning representation, kernel, and engram dimensions.
    pub fn for_full_search(tensor_id: &str) -> Self {
        Self {
            tensor_id: tensor_id.to_string(),
            config: SearchConfig {
                population_size: 20,
                mutation_rate: 0.25,
                crossover_rate: 0.25,
                max_generations: 100,
                convergence_threshold: 0.01,
                cost_function: CostFunction::Weighted {
                    wall: 0.5,
                    energy: 0.3,
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

    /// Run the same population search against an executable evaluator. A
    /// candidate is admitted to selection only when static, numerical, and
    /// performance receipts all pass; failed candidates receive infinite
    /// fitness and cannot win by synthetic cost estimates.
    pub fn run_measured<E: CandidateEvaluator>(
        &self,
        evaluator: &E,
    ) -> Result<JointSearchResult, String> {
        let mut population = self.generate_initial_population();
        let workload = Workload {
            tensor_id: crate::ecs::canonical::identity::LogicalTensorId(self.tensor_id.clone()),
            shape: vec![2, 4, 640],
            repetitions: 3,
        };
        let mut generations_completed = 0;
        for gen in 0..self.config.max_generations {
            generations_completed = gen + 1;
            for scored in &mut population {
                if scored.fitness.is_none() {
                    scored.fitness = Some(
                        self.measured_fitness(&scored.genome, evaluator, &workload)
                            .unwrap_or(f64::INFINITY),
                    );
                }
            }
            population.sort_by(|a, b| {
                a.fitness
                    .unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.fitness.unwrap_or(f64::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if gen > 0 {
                if let (Some(a), Some(b)) = (population.first(), population.get(1)) {
                    let af = a.fitness.unwrap_or(f64::INFINITY);
                    let bf = b.fitness.unwrap_or(f64::INFINITY);
                    if af.is_finite()
                        && (af - bf).abs()
                            < self.config.convergence_threshold * af.abs().max(1.0)
                    {
                        break;
                    }
                }
            }
            let budget = self.config.population_size.max(1);
            let elite_count = ((budget as f64 * 0.3).ceil() as usize)
                .max(1)
                .min(budget.saturating_sub(1).max(1));
            population.truncate(elite_count);
            let mut i = 0;
            while population.len() < budget {
                let a = population[i % population.len()].genome.clone();
                let b = population[(i + 1) % population.len()].genome.clone();
                population.push(ScoredGenome {
                    genome: joint_mutate(&joint_crossover(&a, &b), self, i + gen),
                    fitness: None,
                });
                i += 1;
            }
        }
        Ok(JointSearchResult {
            best_genome: population
                .first()
                .filter(|x| x.fitness.is_some_and(|f| f.is_finite()))
                .map(|x| x.genome.clone()),
            population_size: population.len(),
            generations_completed,
        })
    }

    fn measured_fitness<E: CandidateEvaluator>(
        &self,
        genome: &JointGenome,
        evaluator: &E,
        workload: &Workload,
    ) -> Result<f64, String> {
        let candidate = EvolutionCandidate {
            candidate_id: CandidateId(format!("joint.{}.{}", self.tensor_id, genome.kernel_variant)),
            parent_ids: vec![],
            generation: 0,
            genome: CandidateGenome {
                representation: genome.codec,
                packing: PhysicalTileLayout {
                    tile_m: genome.tile_m as u32,
                    tile_n: genome.tile_n as u32,
                    tiles_per_row: 1,
                    total_tiles: 1,
                    padded_cols: genome.tile_n as u32,
                    group_size: genome.group_size as u32,
                    groups_per_tile: 1,
                    packed_bytes_per_tile: 0,
                    metadata_f32_per_tile: 0,
                },
                metal_geometry: MetalGeometry {
                    grid_width: 1,
                    grid_height: 1,
                    simd_width: 32,
                    threadgroup_width: 256,
                    threadgroup_height: 1,
                    threadgroup_depth: 1,
                },
                decomposition: match genome.reduction_strategy.as_str() {
                    "split-k" => DecompositionStrategy::SplitK(genome.tile_k as u32),
                    "tree" => DecompositionStrategy::ReductionTree(2),
                    _ => DecompositionStrategy::Sequential,
                },
                memory_config: MemoryConfig {
                    vector_width: 32,
                    cache_policy: "writeback".into(),
                    threadgroup_staging: 32768,
                },
                fusion_strategy: None,
                engram_config: None,
                kernel_variant: genome.kernel_variant.clone(),
            },
            compiled_artifacts: vec![],
            correctness_receipt: None,
            quality_receipt: None,
            performance_receipt: None,
            fitness: None,
            status: CandidateStatus::Created,
        };
        let static_receipt = evaluator.validate_static(&candidate)?;
        if !static_receipt.passed {
            return Err(static_receipt.violations.join(", "));
        }
        let compiled = evaluator.compile(&candidate)?;
        let numerical = evaluator.validate_numerical(&compiled)?;
        if !numerical.passed {
            return Err(format!("numerical gate failed: {}", numerical.max_absolute_error));
        }
        let performance = evaluator.measure(&compiled, workload)?;
        Ok(performance.latency_p50_ns as f64)
    }

    fn generate_initial_population(&self) -> Vec<ScoredGenome> {
        let mut pop = Vec::new();
        for codec in &self.engram_codecs {
            for insertion_point in &self.insertion_points {
                for kernel in &self.kernel_variants {
                    pop.push(ScoredGenome {
                        genome: JointGenome {
                            // Representation defaults
                            codec: CodecFamily::Nf4,
                            group_size: 32,
                            residual_policy: "sparse".into(),

                            // Kernel defaults
                            kernel_variant: kernel.clone(),
                            tile_m: 64,
                            tile_n: 64,
                            tile_k: 64,
                            reduction_strategy: "sequential".into(),

                            // Engram defaults
                            engram_codec: codec.clone(),
                            engram_capacity: 1024,
                            insertion_point: insertion_point.clone(),
                            retrieval_threshold: 0.7,
                        },
                        fitness: None,
                    });
                }
            }
        }
        pop
    }

    fn evaluate(&self, genome: &JointGenome) -> f64 {
        // Simulated cost across all three dimensions
        let repr_cost = match genome.codec {
            CodecFamily::Int8 => 3.0,
            CodecFamily::Nf4 => 2.0,
            CodecFamily::Ternary => 1.0,
            _ => 5.0,
        };
        let engram_cost = match genome.engram_codec.as_str() {
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
        repr_cost + engram_cost + kernel_cost
    }
}

/// Map a seed codon into a full JointGenome.
pub fn codon_to_genome(
    codec: &str,
    kernel: &str,
    engram_codec: &str,
    genome: &JointGenome,
) -> JointGenome {
    JointGenome {
        codec: match codec {
            "nf4" => CodecFamily::Nf4,
            "ternary" => CodecFamily::Ternary,
            _ => genome.codec,
        },
        group_size: genome.group_size,
        residual_policy: genome.residual_policy.clone(),
        kernel_variant: kernel.to_string(),
        tile_m: genome.tile_m,
        tile_n: genome.tile_n,
        tile_k: genome.tile_k,
        reduction_strategy: genome.reduction_strategy.clone(),
        engram_codec: engram_codec.to_string(),
        engram_capacity: genome.engram_capacity,
        insertion_point: genome.insertion_point.clone(),
        retrieval_threshold: genome.retrieval_threshold,
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
    fn test_joint_genome_all_dimensions_present() {
        let genome = JointGenome {
            codec: CodecFamily::Nf4,
            group_size: 32,
            residual_policy: "sparse".into(),
            kernel_variant: "tile640_gemv".into(),
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            reduction_strategy: "sequential".into(),
            engram_codec: "nf4".into(),
            engram_capacity: 1024,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
        };
        assert_eq!(genome.tile_m, 64);
        assert_eq!(genome.engram_codec, "nf4");
        assert_eq!(genome.reduction_strategy, "sequential");
    }

    #[test]
    fn test_full_search_config_has_all_genres() {
        let config = JointSearchConfig::for_full_search("attention.q_proj");
        assert_eq!(config.engram_codecs.len(), 3);
        assert_eq!(config.kernel_variants.len(), 3);
        assert!(config.config.population_size >= 20);
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
            codec: CodecFamily::Nf4,
            group_size: 32,
            residual_policy: "sparse".into(),
            kernel_variant: "tile640_gemv".into(),
            tile_m: 64,
            tile_n: 128,
            tile_k: 64,
            reduction_strategy: "sequential".into(),
            engram_codec: "nf4".into(),
            engram_capacity: 512,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
        };
        let b = JointGenome {
            codec: CodecFamily::Int8,
            group_size: 64,
            residual_policy: "dense".into(),
            kernel_variant: "persistent_gemv".into(),
            tile_m: 128,
            tile_n: 256,
            tile_k: 128,
            reduction_strategy: "split-k".into(),
            engram_codec: "int8".into(),
            engram_capacity: 2048,
            insertion_point: "after.linear.k_proj".into(),
            retrieval_threshold: 0.5,
        };
        let child = joint_crossover(&a, &b);
        // Representation: codec from A, group_size from B
        assert_eq!(child.codec, CodecFamily::Nf4);
        assert_eq!(child.group_size, 64);
        // Kernel: variant from B, tiles from A
        assert_eq!(child.kernel_variant, "persistent_gemv");
        assert_eq!(child.tile_m, 64);
        // Engram: codec from A, max capacity, averaged threshold
        assert_eq!(child.engram_codec, "nf4");
        assert_eq!(child.engram_capacity, 2048);
        assert!((child.retrieval_threshold - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_joint_mutate_perturbs_threshold() {
        let genome = JointGenome {
            codec: CodecFamily::Nf4,
            group_size: 32,
            residual_policy: "sparse".into(),
            kernel_variant: "tile640_gemv".into(),
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            reduction_strategy: "sequential".into(),
            engram_codec: "nf4".into(),
            engram_capacity: 1024,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
        };
        let config = JointSearchConfig::for_tensor("attention.q_proj");
        // seed=0 triggers threshold perturbation (seed%5==0)
        let mutated = joint_mutate(&genome, &config, 0);
        assert_ne!(mutated.retrieval_threshold, 0.7);
        assert!((0.1..=1.0).contains(&mutated.retrieval_threshold));
    }

    #[test]
    fn test_offspring_generated() {
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
        assert_eq!(best.codec, CodecFamily::Nf4);
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

    #[test]
    fn test_codon_to_genome_resolves() {
        let base = JointGenome {
            codec: CodecFamily::Nf4,
            group_size: 32,
            residual_policy: "sparse".into(),
            kernel_variant: "tile640_gemv".into(),
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            reduction_strategy: "sequential".into(),
            engram_codec: "nf4".into(),
            engram_capacity: 1024,
            insertion_point: "after.linear.q_proj".into(),
            retrieval_threshold: 0.7,
        };
        let mapped = codon_to_genome("ternary", "persistent_gemv", "int8", &base);
        assert_eq!(mapped.codec, CodecFamily::Ternary);
        assert_eq!(mapped.kernel_variant, "persistent_gemv");
        assert_eq!(mapped.engram_codec, "int8");
        // Non-overridden fields preserve defaults
        assert_eq!(mapped.group_size, 32);
        assert_eq!(mapped.reduction_strategy, "sequential");
    }
}
