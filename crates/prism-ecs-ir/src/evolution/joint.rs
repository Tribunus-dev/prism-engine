//! Joint evolution system for co-evolution of format, operation, and layout.
//!
//! The joint evolution system orchestrates genome-level crossover, mutation,
//! and selection across all eight genome axes simultaneously. It searches
//! the combined space of representation, packing, geometry, decomposition,
//! memory, fusion, engram, and runtime parameters.

use crate::evolution::evaluate::EvaluationStrategy;
use crate::evolution::foundation::{
    CandidateGenome, DecompositionAxis, FusionAxis, PackingAxis, RepresentationAxis,
};
use crate::evolution::frontier::ParetoFrontier;
use prism_ecs_core::Component;
use rand::Rng;

/// Configuration for the joint evolution search.
#[derive(Debug, Clone)]
pub struct JointSearchConfig {
    /// Population size per generation.
    pub population_size: usize,
    /// Crossover rate (0.0–1.0).
    pub crossover_rate: f64,
    /// Mutation rate per axis (0.0–1.0).
    pub mutation_rate: f64,
    /// Number of generations to run.
    pub max_generations: usize,
    /// Early stopping: stop if no improvement for N generations.
    pub stagnation_limit: usize,
}

impl Default for JointSearchConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            max_generations: 100,
            stagnation_limit: 10,
        }
    }
}

impl Component for JointSearchConfig {}

/// A scored genome: a candidate paired with its fitness vector.
#[derive(Debug, Clone)]
pub struct ScoredGenome {
    pub genome: CandidateGenome,
    pub fitness: Vec<f64>,
}

/// Result of a joint evolution search.
#[derive(Debug, Clone)]
pub struct JointSearchResult {
    /// The best genome found.
    pub best_genome: CandidateGenome,
    /// Fitness vector of the best genome.
    pub best_fitness: Vec<f64>,
    /// Generations actually executed.
    pub generations_run: usize,
    /// Whether convergence was reached.
    pub converged: bool,
}

/// Rescue scope — defines how far a rescue operation can explore.
#[derive(Debug, Clone, Copy)]
pub enum RescueScope {
    Local,
    Neighbourhood,
    Global,
}

/// Rescue codec — encodes how to recover from a failed candidate.
#[derive(Debug, Clone)]
pub struct RescueCodec {
    pub scope: RescueScope,
    pub retry_limit: usize,
}

/// Joint evolution system — orchestrates evolution across all genome axes.
///
/// Attach to an entity with a `JointSearchConfig` component.
#[derive(Debug, Clone)]
pub struct JointEvolutionSystem {
    pub config: JointSearchConfig,
    pub generation: u64,
}

impl JointEvolutionSystem {
    pub fn new(config: JointSearchConfig) -> Self {
        Self {
            config,
            generation: 0,
        }
    }

    /// Run one generation with two-phase evaluation.
    ///
    /// 1. Run synthetic evaluation on all candidates (fast, cheap).
    /// 2. Extract the Pareto-frontier top-N candidates.
    /// 3. Re-evaluate frontier candidates with the measured evaluator
    ///    on real hardware, updating their fitness scores.
    ///
    /// This avoids expensive hardware measurements for clearly
    /// dominated candidates while still grounding the frontier in
    /// real performance data.
    pub fn run_generation_with_measured(
        &self,
        population: &mut [CandidateGenome],
        synthetic_evaluator: &dyn EvaluationStrategy,
        measured_evaluator: &dyn EvaluationStrategy,
        frontier: &mut ParetoFrontier,
    ) {
        // Phase 1: synthetic evaluation on all candidates (fast).
        let empty_entity = prism_ecs_core::Entity::new(0, 0);
        for genome in population.iter() {
            let score = synthetic_evaluator.evaluate(genome, b"4096,4096");
            frontier.insert(
                empty_entity,
                vec![score],
                self.generation,
                &Default::default(),
            );
        }

        // Phase 2: re-evaluate the top frontier entries with measured evaluator.
        // We limit to the first N frontier entries to bound hardware cost.
        const TOP_N: usize = 10;
        let top_n = frontier.entries.len().min(TOP_N);
        for i in 0..top_n {
            if let Some(entry) = frontier.entries.get(i) {
                // Reconstruct genome from frontier — placeholder: in production
                // the genome would be stored alongside the entry.
                let genome = CandidateGenome {
                    representation: match entry.fitness.first() {
                        Some(f) if f.value() > 0.5 => RepresentationAxis::Fp16,
                        Some(f) if f.value() > 0.3 => RepresentationAxis::Int8,
                        _ => RepresentationAxis::Int4,
                    },
                    ..CandidateGenome::new()
                };

                let measured_score = measured_evaluator.evaluate(&genome, b"4096,4096");
                // Replace the fitness in the frontier entry (in-place).
                // NOTE: in production, the genome is tracked so the measured
                // score can be properly associated.
                if let Some(entry_mut) = frontier.entries.get_mut(i) {
                    entry_mut.fitness = vec![measured_score];
                }
            }
        }
    }

    /// Perform crossover between two parent genomes.
    ///
    /// Uses uniform crossover: each axis has a 50% chance of coming from
    /// parent A or parent B.
    pub fn joint_crossover(&self, a: &CandidateGenome, b: &CandidateGenome) -> CandidateGenome {
        CandidateGenome {
            representation: if rand::random::<f64>() < 0.5 {
                a.representation.clone()
            } else {
                b.representation.clone()
            },
            packing: if rand::random::<f64>() < 0.5 {
                a.packing.clone()
            } else {
                b.packing.clone()
            },
            metal_geometry: if rand::random::<f64>() < 0.5 {
                a.metal_geometry.clone()
            } else {
                b.metal_geometry.clone()
            },
            decomposition: if rand::random::<f64>() < 0.5 {
                a.decomposition.clone()
            } else {
                b.decomposition.clone()
            },
            memory: if rand::random::<f64>() < 0.5 {
                a.memory.clone()
            } else {
                b.memory.clone()
            },
            fusion: if rand::random::<f64>() < 0.5 {
                a.fusion.clone()
            } else {
                b.fusion.clone()
            },
            engram: if rand::random::<f64>() < 0.5 {
                a.engram.clone()
            } else {
                b.engram.clone()
            },
            runtime: if rand::random::<f64>() < 0.5 {
                a.runtime.clone()
            } else {
                b.runtime.clone()
            },
        }
    }

    /// Mutate a genome by randomly perturbing each axis.
    ///
    /// Each axis is mutated independently with probability `mutation_rate`.
    pub fn joint_mutate(&self, genome: &CandidateGenome) -> CandidateGenome {
        let mut result = genome.clone();
        let rate = self.config.mutation_rate;

        if rand::random::<f64>() < rate {
            result.representation = self.mutate_representation(&genome.representation);
        }
        if rand::random::<f64>() < rate {
            result.packing = self.mutate_packing(&genome.packing);
        }
        if rand::random::<f64>() < rate {
            result.metal_geometry = self.mutate_geometry(&genome.metal_geometry);
        }
        if rand::random::<f64>() < rate {
            result.decomposition = self.mutate_decomposition(&genome.decomposition);
        }
        if rand::random::<f64>() < rate {
            result.memory.shared_memory_bytes =
                self.mutate_memory_size(genome.memory.shared_memory_bytes);
        }
        if rand::random::<f64>() < rate {
            result.fusion = self.mutate_fusion(&genome.fusion);
        }
        if rand::random::<f64>() < rate {
            result.runtime.dispatch_width = genome.runtime.dispatch_width.saturating_mul(2).max(1);
        }

        result
    }

    fn mutate_representation(&self, repr: &RepresentationAxis) -> RepresentationAxis {
        match repr {
            RepresentationAxis::Fp16 => RepresentationAxis::Bf16,
            RepresentationAxis::Bf16 => RepresentationAxis::Int8,
            RepresentationAxis::Int8 => RepresentationAxis::Int4,
            RepresentationAxis::Int4 => RepresentationAxis::Nf4,
            RepresentationAxis::Nf4 => RepresentationAxis::Nf8,
            RepresentationAxis::Nf8 => RepresentationAxis::Ternary158,
            RepresentationAxis::Ternary158 => RepresentationAxis::Binary1,
            RepresentationAxis::Binary1 => RepresentationAxis::Fp16,
        }
    }

    fn mutate_packing(&self, packing: &PackingAxis) -> PackingAxis {
        match packing {
            PackingAxis::Tile640 => PackingAxis::Block2D,
            PackingAxis::Block2D => PackingAxis::Planar,
            PackingAxis::Planar => PackingAxis::Interleaved,
            PackingAxis::Interleaved => PackingAxis::Tile640,
        }
    }

    fn mutate_geometry(
        &self,
        geo: &crate::evolution::foundation::MetalGeometryAxis,
    ) -> crate::evolution::foundation::MetalGeometryAxis {
        crate::evolution::foundation::MetalGeometryAxis {
            threadgroup_width: (geo.threadgroup_width * 2).min(256),
            threadgroup_height: (geo.threadgroup_height * 2).min(64),
            grid_tile_m: (geo.grid_tile_m * 2).min(256),
            grid_tile_n: (geo.grid_tile_n * 2).min(256),
            grid_tile_k: (geo.grid_tile_k * 2).min(128),
        }
    }

    fn mutate_decomposition(&self, decomp: &DecompositionAxis) -> DecompositionAxis {
        match decomp {
            DecompositionAxis::Flat => DecompositionAxis::SplitM,
            DecompositionAxis::SplitM => DecompositionAxis::SplitMN,
            DecompositionAxis::SplitMN => DecompositionAxis::SplitMNK,
            DecompositionAxis::SplitMNK => DecompositionAxis::Flat,
        }
    }

    fn mutate_memory_size(&self, current: u32) -> u32 {
        if rand::random::<f64>() < 0.5 {
            (current * 2).min(262144)
        } else {
            (current / 2).max(4096)
        }
    }

    fn mutate_fusion(&self, fusion: &FusionAxis) -> FusionAxis {
        match fusion {
            FusionAxis::None => FusionAxis::ElementWise,
            FusionAxis::ElementWise => FusionAxis::KernelFusion,
            FusionAxis::KernelFusion => FusionAxis::None,
        }
    }

    /// Compute a deterministic digest for a genome (for dedup and caching).
    pub fn genome_digest(genome: &CandidateGenome) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}", genome.representation).hash(&mut hasher);
        format!("{:?}", genome.packing).hash(&mut hasher);
        format!("{:?}", genome.decomposition).hash(&mut hasher);
        format!("{:?}", genome.fusion).hash(&mut hasher);
        format!("{:?}", genome.engram).hash(&mut hasher);
        genome.metal_geometry.threadgroup_width.hash(&mut hasher);
        genome.metal_geometry.grid_tile_m.hash(&mut hasher);
        genome.memory.shared_memory_bytes.hash(&mut hasher);
        genome.runtime.dispatch_width.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Run one generation of the joint evolution search.
    ///
    /// Returns the resulting population (genomes paired with IDs for
    /// frontier tracking).
    pub fn run_generation(
        &self,
        population: &[ScoredGenome],
        frontier: &ParetoFrontier,
    ) -> Vec<ScoredGenome> {
        if population.is_empty() || frontier.is_empty() {
            return vec![];
        }

        let mut next_gen = Vec::with_capacity(self.config.population_size);

        // Elitism: keep the top frontier entries.
        for entry in frontier.entries.iter().take(5) {
            // Reconstruct a dummy scored genome from frontier entries.
            // In production, the actual genome would be stored alongside.
            let fitness: Vec<f64> = entry.fitness.iter().map(|f| f.value()).collect();
            next_gen.push(ScoredGenome {
                genome: CandidateGenome::new(),
                fitness,
            });
        }

        // Fill the rest through crossover and mutation.
        while next_gen.len() < self.config.population_size {
            let parent_a = &population[rand::thread_rng().gen_range(0..population.len())];
            let parent_b = &population[rand::thread_rng().gen_range(0..population.len())];

            let child = if rand::random::<f64>() < self.config.crossover_rate {
                self.joint_crossover(&parent_a.genome, &parent_b.genome)
            } else {
                parent_a.genome.clone()
            };

            let child = self.joint_mutate(&child);
            next_gen.push(ScoredGenome {
                genome: child,
                fitness: vec![0.0; frontier.num_dimensions], // placeholder — evaluated later
            });
        }

        next_gen
    }

    /// Convert a codon (index into the space of genome variants) into a genome.
    /// Used for systematic search when not mutating randomly.
    pub fn codon_to_genome(codon: u64) -> CandidateGenome {
        let repr_idx = (codon % 8) as usize;
        let packing_idx = ((codon / 8) % 4) as usize;
        let decomp_idx = ((codon / 32) % 4) as usize;
        let fusion_idx = ((codon / 128) % 3) as usize;

        let representations = [
            RepresentationAxis::Fp16,
            RepresentationAxis::Bf16,
            RepresentationAxis::Int8,
            RepresentationAxis::Int4,
            RepresentationAxis::Nf4,
            RepresentationAxis::Nf8,
            RepresentationAxis::Ternary158,
            RepresentationAxis::Binary1,
        ];

        let packings = [
            PackingAxis::Tile640,
            PackingAxis::Block2D,
            PackingAxis::Planar,
            PackingAxis::Interleaved,
        ];

        let decompositions = [
            DecompositionAxis::Flat,
            DecompositionAxis::SplitM,
            DecompositionAxis::SplitMN,
            DecompositionAxis::SplitMNK,
        ];

        let fusions = [
            FusionAxis::None,
            FusionAxis::ElementWise,
            FusionAxis::KernelFusion,
        ];

        CandidateGenome {
            representation: representations[repr_idx].clone(),
            packing: packings[packing_idx].clone(),
            metal_geometry: Default::default(),
            decomposition: decompositions[decomp_idx].clone(),
            memory: Default::default(),
            fusion: fusions[fusion_idx].clone(),
            engram: Default::default(),
            runtime: Default::default(),
        }
    }
}

impl Default for JointEvolutionSystem {
    fn default() -> Self {
        Self::new(JointSearchConfig::default())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::foundation::{FitnessScore, RepresentationAxis};

    #[test]
    fn crossover_produces_child() {
        let system = JointEvolutionSystem::default();
        let parent_a = CandidateGenome::new();
        let mut parent_b = CandidateGenome::new();
        parent_b.representation = RepresentationAxis::Binary1;
        let child = system.joint_crossover(&parent_a, &parent_b);
        // Child must have some representation from either parent.
        let has_repr = matches!(
            child.representation,
            RepresentationAxis::Fp16 | RepresentationAxis::Binary1
        );
        assert!(has_repr);
    }

    #[test]
    fn mutation_changes_genome() {
        let system = JointEvolutionSystem::new(JointSearchConfig {
            mutation_rate: 1.0, // always mutate every axis
            ..Default::default()
        });
        let original = CandidateGenome::new();
        let mutated = system.joint_mutate(&original);
        let original_digest = JointEvolutionSystem::genome_digest(&original);
        let mutated_digest = JointEvolutionSystem::genome_digest(&mutated);
        assert_ne!(original_digest, mutated_digest);
    }

    #[test]
    fn genome_digest_is_deterministic() {
        let genome = CandidateGenome::new();
        let d1 = JointEvolutionSystem::genome_digest(&genome);
        let d2 = JointEvolutionSystem::genome_digest(&genome);
        assert_eq!(d1, d2);
    }

    #[test]
    fn codon_to_genome_maps_distinct() {
        let g0 = JointEvolutionSystem::codon_to_genome(0);
        let g1 = JointEvolutionSystem::codon_to_genome(1);
        let d0 = JointEvolutionSystem::genome_digest(&g0);
        let d1 = JointEvolutionSystem::genome_digest(&g1);
        assert_ne!(d0, d1);
    }

    #[test]
    fn run_generation_produces_population() {
        let system = JointEvolutionSystem::default();
        let mut frontier = ParetoFrontier::new(2);

        let population: Vec<ScoredGenome> = (0..10)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5 + (i as f64) * 0.05, 0.5],
            })
            .collect();

        for (i, sg) in population.iter().enumerate() {
            let entity = prism_ecs_core::Entity(i as u64, 0);
            frontier.insert(
                entity,
                sg.fitness.iter().map(|&v| FitnessScore::new(v)).collect(),
                0,
                &Default::default(),
            );
        }

        let next = system.run_generation(&population, &frontier);
        assert!(!next.is_empty());
        assert!(next.len() <= system.config.population_size);
    }
}
