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
use crate::evolution::memory::{EvolutionReceipt, EvolutionaryMemory};
use crate::evolution::objectives::QualityDiversityArchive;
use crate::evolution::variation::{AdaptiveVariationController, VariationOperator};
use prism_ecs_core::Component;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::Cell;
use std::sync::Mutex;

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
    /// Optional seed for deterministic random operations.
    pub seed: Option<u64>,
}

impl Default for JointSearchConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            max_generations: 100,
            stagnation_limit: 10,
            seed: None,
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
#[derive(Debug)]
pub struct JointEvolutionSystem {
    pub config: JointSearchConfig,
    /// Current generation counter (interior mutability for &self methods).
    pub generation: Cell<u64>,
    /// Best single-dimensional fitness value observed so far.
    pub best_fitness: Cell<f64>,
    /// Generation at which best_fitness was last improved.
    pub best_fitness_generation: Cell<u64>,
    /// Seeded RNG wrapped in Mutex for interior mutability.
    pub rng: Mutex<Option<StdRng>>,
    /// Operator bandit state. It is deliberately separate from evaluator
    /// state so variation can adapt from receipt outcomes at the coordinator.
    pub variation: Mutex<AdaptiveVariationController>,
    pub memory: Mutex<EvolutionaryMemory>,
}

impl JointEvolutionSystem {
    pub fn new(config: JointSearchConfig) -> Self {
        let rng = config.seed.map(StdRng::seed_from_u64);
        Self {
            generation: Cell::new(0),
            best_fitness: Cell::new(f64::NEG_INFINITY),
            best_fitness_generation: Cell::new(0),
            config,
            rng: Mutex::new(rng),
            variation: Mutex::new(AdaptiveVariationController::default()),
            memory: Mutex::new(EvolutionaryMemory::default()),
        }
    }

    /// Return the search configuration (for inspection, mutation is via `new`).
    pub fn config(&self) -> &JointSearchConfig {
        &self.config
    }

    pub fn merge_variation_controller(&self, controller: &AdaptiveVariationController) {
        if let Ok(mut local) = self.variation.lock() {
            local.merge(controller);
        }
    }

    /// Persist an externally measured mutation receipt for later replay. The
    /// evaluator remains the authority; this method only records evidence.
    pub fn record_evolution_receipt(&self, receipt: EvolutionReceipt) {
        if let Ok(mut memory) = self.memory.lock() {
            memory.record(receipt);
        }
    }

    /// Evaluate a single genome using the given strategy and return a ScoredGenome.
    ///
    /// This is the canonical entry point for scoring: it converts the
    /// `FitnessScore` returned by the evaluator into a `Vec<f64>` fitness
    /// vector that the frontier and selection operators expect.
    pub fn estimate_and_score_genome(
        &self,
        genome: &CandidateGenome,
        evaluator: &dyn EvaluationStrategy,
        context: &[u8],
    ) -> ScoredGenome {
        let fitness_score = evaluator.evaluate(genome, context);
        if let Ok(mut controller) = self.variation.lock() {
            controller.observe_geometry(
                &genome.metal_geometry,
                genome.memory.shared_memory_bytes,
                fitness_score.value(),
            );
        }
        ScoredGenome {
            genome: genome.clone(),
            fitness: vec![fitness_score.value()],
        }
    }

    /// Check whether the evolution loop should stop.
    ///
    /// Examines generation count and frontier health to determine whether
    /// the maximum generation limit has been reached or the search has
    /// stagnated (no fitness improvement for `stagnation_limit` generations).
    ///
    /// Returns `Some(reason)` when the loop should stop, or `None` to
    /// continue. Also updates internal best-fitness tracking based on
    /// the current frontier state.
    pub fn should_stop(&self, frontier: &ParetoFrontier) -> Option<String> {
        let gen = self.generation.get();

        // Generation limit check.
        if gen >= self.config.max_generations as u64 {
            return Some(format!(
                "generation_limit: reached generation {} of {}",
                gen, self.config.max_generations
            ));
        }

        // Stagnation check: if the frontier's best entry has improved,
        // update tracking. Otherwise, if we've been stalled too long, stop.
        if let Some(best) = frontier.best_by_dimension(0) {
            let best_fit = best.fitness[0].value();
            let prev_best = self.best_fitness.get();

            if best_fit > prev_best + 1e-12 {
                self.best_fitness.set(best_fit);
                self.best_fitness_generation.set(gen);
            } else {
                let stalled_gens = gen - self.best_fitness_generation.get();
                if stalled_gens >= self.config.stagnation_limit as u64 {
                    return Some(format!(
                        "stall_limit: no improvement for {} generations (limit {})",
                        stalled_gens, self.config.stagnation_limit
                    ));
                }
            }
        } else {
            // Empty frontier — first evaluation pass hasn't happened yet.
            // Initialize best_fitness so the next generation can detect
            // improvement.
            self.best_fitness.set(f64::NEG_INFINITY);
            self.best_fitness_generation.set(gen);
        }

        None
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
        for genome in population.iter() {
            let score = synthetic_evaluator.evaluate(genome, b"4096,4096"); // run_generation_with_measured uses default context — callers pass real context via estimate_and_score_genome
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                vec![score],
                self.generation.get(),
                &Default::default(),
            );
        }

        // Phase 2: re-evaluate the top frontier entries with measured evaluator.
        // We limit to the first N frontier entries to bound hardware cost.
        const TOP_N: usize = 10;
        let top_n = frontier.entries.len().min(TOP_N);

        // Clone genomes first to avoid borrow conflict between immutable
        // read and mutable update of the frontier entries slice.
        let genomes_to_remeasure: Vec<_> = (0..top_n)
            .filter_map(|i| population.get(i).map(|g| (i, g.clone())))
            .collect();

        for (i, genome) in genomes_to_remeasure {
            let measured_score = measured_evaluator.evaluate(&genome, b"4096,4096"); // run_generation_with_measured uses default context
            if let Some(entry_mut) = frontier.entries.get_mut(i) {
                entry_mut.fitness = vec![measured_score];
            }
        }
    }

    /// Perform crossover between two parent genomes.
    ///
    /// Uses uniform crossover: each axis has a 50% chance of coming from
    /// parent A or parent B.
    pub fn joint_crossover(
        &self,
        a: &CandidateGenome,
        b: &CandidateGenome,
        rng: &mut impl Rng,
    ) -> CandidateGenome {
        CandidateGenome {
            representation: if rng.gen::<f64>() < 0.5 {
                a.representation.clone()
            } else {
                b.representation.clone()
            },
            packing: if rng.gen::<f64>() < 0.5 {
                a.packing.clone()
            } else {
                b.packing.clone()
            },
            metal_geometry: if rng.gen::<f64>() < 0.5 {
                a.metal_geometry.clone()
            } else {
                b.metal_geometry.clone()
            },
            decomposition: if rng.gen::<f64>() < 0.5 {
                a.decomposition.clone()
            } else {
                b.decomposition.clone()
            },
            memory: if rng.gen::<f64>() < 0.5 {
                a.memory.clone()
            } else {
                b.memory.clone()
            },
            fusion: if rng.gen::<f64>() < 0.5 {
                a.fusion.clone()
            } else {
                b.fusion.clone()
            },
            engram: if rng.gen::<f64>() < 0.5 {
                a.engram.clone()
            } else {
                b.engram.clone()
            },
            runtime: if rng.gen::<f64>() < 0.5 {
                a.runtime.clone()
            } else {
                b.runtime.clone()
            },
            ane_unit: if rng.gen::<f64>() < 0.5 {
                a.ane_unit.clone()
            } else {
                b.ane_unit.clone()
            },
        }
    }

    /// Mutate a genome by randomly perturbing each axis.
    ///
    /// Each axis is mutated independently with probability `mutation_rate`.
    pub fn joint_mutate(&self, genome: &CandidateGenome, rng: &mut impl Rng) -> CandidateGenome {
        self.joint_mutate_with_operator(genome, rng).0
    }

    pub fn joint_mutate_with_operator(
        &self,
        genome: &CandidateGenome,
        rng: &mut impl Rng,
    ) -> (CandidateGenome, VariationOperator) {
        let mut result = genome.clone();
        let rate = self.config.mutation_rate;

        // Ensure every generation has one explicitly selected adaptive
        // operator. The legacy per-axis mutations below remain as a low-rate
        // exploration floor for compatibility with existing search behavior.
        let selected_operator = self
            .variation
            .lock()
            .ok()
            .map(|controller| controller.select(rng));
        if rng.gen::<f64>() < rate {
            match selected_operator {
                Some(VariationOperator::Representation) => {
                    result.representation = self.mutate_representation(&genome.representation, rng)
                }
                Some(VariationOperator::Packing) => {
                    result.packing = self.mutate_packing(&genome.packing, rng)
                }
                Some(VariationOperator::Geometry) => {
                    let (geometry, memory) = self
                        .variation
                        .lock()
                        .map(|controller| controller.sample_geometry(rng))
                        .unwrap_or_else(|_| {
                            (
                                genome.metal_geometry.clone(),
                                genome.memory.shared_memory_bytes,
                            )
                        });
                    result.metal_geometry = geometry;
                    result.memory.shared_memory_bytes = memory;
                }
                Some(VariationOperator::Decomposition) => {
                    result.decomposition = self.mutate_decomposition(&genome.decomposition, rng)
                }
                Some(VariationOperator::Memory) => {
                    result.memory.shared_memory_bytes =
                        self.mutate_memory_size(genome.memory.shared_memory_bytes, rng)
                }
                Some(VariationOperator::Fusion) => {
                    result.fusion = self.mutate_fusion(&genome.fusion, rng)
                }
                Some(VariationOperator::Runtime) => {
                    result.runtime.dispatch_width =
                        genome.runtime.dispatch_width.saturating_mul(2).max(1)
                }
                Some(VariationOperator::AneUnit) => {}
                Some(VariationOperator::Unknown) => {}
                None => {}
            }
        }

        (
            result,
            selected_operator.unwrap_or(VariationOperator::Unknown),
        )
    }

    pub fn record_operator_feedback(&self, operator: VariationOperator, reward: f64) {
        if operator == VariationOperator::Unknown {
            return;
        }
        if let Ok(mut controller) = self.variation.lock() {
            controller.record(operator, reward);
        }
    }

    pub fn record_geometry_feedback(
        &self,
        geometry: &crate::evolution::foundation::MetalGeometryAxis,
        shared_memory_bytes: u32,
        score: f64,
    ) {
        if let Ok(mut controller) = self.variation.lock() {
            controller
                .geometry_covariance
                .observe(geometry, shared_memory_bytes, score);
        }
    }

    fn mutate_representation(
        &self,
        repr: &RepresentationAxis,
        _rng: &mut impl Rng,
    ) -> RepresentationAxis {
        match repr {
            RepresentationAxis::Fp16 => RepresentationAxis::Bf16,
            RepresentationAxis::Bf16 => RepresentationAxis::Int8,
            RepresentationAxis::Int8 => RepresentationAxis::Int4,
            RepresentationAxis::Int4 => RepresentationAxis::Nf4,
            RepresentationAxis::Nf4 => RepresentationAxis::Nf8,
            RepresentationAxis::Nf8 => RepresentationAxis::Ternary158,
            RepresentationAxis::Ternary158 => RepresentationAxis::TernaryTile640,
            RepresentationAxis::Binary1 => RepresentationAxis::Fp16,
            RepresentationAxis::TernaryTile640 => RepresentationAxis::Binary1,
        }
    }

    fn mutate_packing(&self, packing: &PackingAxis, _rng: &mut impl Rng) -> PackingAxis {
        match packing {
            PackingAxis::Tile640 => PackingAxis::Block2D,
            PackingAxis::Block2D => PackingAxis::Planar,
            PackingAxis::Planar => PackingAxis::Interleaved,
            PackingAxis::Interleaved => PackingAxis::Tile640,
        }
    }

    fn mutate_decomposition(
        &self,
        decomp: &DecompositionAxis,
        _rng: &mut impl Rng,
    ) -> DecompositionAxis {
        match decomp {
            DecompositionAxis::Flat => DecompositionAxis::SplitM,
            DecompositionAxis::SplitM => DecompositionAxis::SplitMN,
            DecompositionAxis::SplitMN => DecompositionAxis::SplitMNK,
            DecompositionAxis::SplitMNK => DecompositionAxis::Flat,
        }
    }

    fn mutate_memory_size(&self, current: u32, rng: &mut impl Rng) -> u32 {
        if rng.gen::<f64>() < 0.5 {
            (current * 2).min(262144)
        } else {
            (current / 2).max(4096)
        }
    }

    fn mutate_fusion(&self, fusion: &FusionAxis, _rng: &mut impl Rng) -> FusionAxis {
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
    /// Returns the resulting population paired with an optional stopping
    /// reason. `None` means the loop should continue; `Some(reason)` means
    /// the caller should terminate the evolution loop.
    pub fn run_generation(
        &self,
        population: &[ScoredGenome],
        frontier: &ParetoFrontier,
    ) -> (Vec<ScoredGenome>, Option<String>) {
        self.run_generation_with_archive(population, frontier, None)
    }

    /// Run a generation while allowing a quality-diversity archive to supply
    /// behavioral-niche elites and parents. The scalar frontier remains part
    /// of the compatibility path, but the archive now influences variation.
    pub fn run_generation_with_archive(
        &self,
        population: &[ScoredGenome],
        frontier: &ParetoFrontier,
        archive: Option<&QualityDiversityArchive>,
    ) -> (Vec<ScoredGenome>, Option<String>) {
        let (next, stop, _, _) = self.run_generation_with_feedback(population, frontier, archive);
        (next, stop)
    }

    pub fn run_generation_with_feedback(
        &self,
        population: &[ScoredGenome],
        frontier: &ParetoFrontier,
        archive: Option<&QualityDiversityArchive>,
    ) -> (
        Vec<ScoredGenome>,
        Option<String>,
        Vec<VariationOperator>,
        Vec<Vec<CandidateGenome>>,
    ) {
        if population.is_empty() || frontier.is_empty() {
            return (vec![], None, vec![], vec![]);
        }

        let mut rng = self.get_rng_for_selection();
        let mut next_gen: Vec<ScoredGenome> = Vec::with_capacity(self.config.population_size);
        let mut operators = Vec::with_capacity(self.config.population_size);
        let mut parents = Vec::with_capacity(self.config.population_size);

        // Elitism is archive-first: nondominated behavioral niches are the
        // primary selection source. The scalar frontier is only a fallback
        // compatibility source when archive cells do not fill the population.
        if let Some(archive) = archive {
            for entry in archive.ranked_elites() {
                if next_gen.len() >= self.config.population_size {
                    break;
                }
                let entry_digest = Self::genome_digest(&entry.genome);
                if !next_gen.iter().any(|candidate: &ScoredGenome| {
                    Self::genome_digest(&candidate.genome) == entry_digest
                }) {
                    next_gen.push(ScoredGenome {
                        genome: entry.genome.clone(),
                        fitness: vec![entry.objectives.scalar_compatibility_score()],
                    });
                    operators.push(VariationOperator::Unknown);
                    parents.push(Vec::new());
                }
            }
        }

        for entry in frontier.entries.iter().take(5) {
            if next_gen.len() >= self.config.population_size {
                break;
            }
            let Some(source) = population.get(entry.entity.id().saturating_sub(1) as usize) else {
                continue;
            };
            let entry_genome = source.genome.clone();
            let entry_digest = Self::genome_digest(&entry_genome);
            if !next_gen
                .iter()
                .any(|candidate| Self::genome_digest(&candidate.genome) == entry_digest)
            {
                let fitness: Vec<f64> = entry.fitness.iter().map(|f| f.value()).collect();
                next_gen.push(ScoredGenome {
                    genome: entry_genome,
                    fitness,
                });
                operators.push(VariationOperator::Unknown);
                parents.push(Vec::new());
            }
        }

        let archive_parents: Vec<ScoredGenome> = archive
            .map(|archive| {
                archive
                    .elites()
                    .map(|entry| ScoredGenome {
                        genome: entry.genome.clone(),
                        fitness: vec![entry.objectives.scalar_compatibility_score()],
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Fill the rest through crossover and mutation.
        while next_gen.len() < self.config.population_size {
            let parent_pool: Vec<&ScoredGenome> =
                population.iter().chain(archive_parents.iter()).collect();
            let parent_a = parent_pool[rng.gen_range(0..parent_pool.len())];
            let parent_b = parent_pool[rng.gen_range(0..parent_pool.len())];

            let child = if rng.gen::<f64>() < self.config.crossover_rate {
                self.joint_crossover(&parent_a.genome, &parent_b.genome, &mut rng)
            } else {
                parent_a.genome.clone()
            };

            let (child, operator) = self.joint_mutate_with_operator(&child, &mut rng);
            parents.push(vec![parent_a.genome.clone(), parent_b.genome.clone()]);
            next_gen.push(ScoredGenome {
                genome: child,
                fitness: vec![0.0; frontier.num_dimensions], // filled by caller on next eval pass
            });
            operators.push(operator);
        }

        // Advance generation counter and check stopping conditions.
        self.generation.set(self.generation.get() + 1);
        let stop_reason = if archive
            .map(|archive| archive.cells.len() > 1)
            .unwrap_or(false)
        {
            if self.generation.get() >= self.config.max_generations as u64 {
                Some(format!(
                    "generation_limit: reached generation {} of {}",
                    self.generation.get(),
                    self.config.max_generations
                ))
            } else {
                None
            }
        } else {
            self.should_stop(frontier)
        };

        (next_gen, stop_reason, operators, parents)
    }

    /// Get a seeded or unseeded RNG for parent selection.
    ///
    /// When the config specifies a seed, the RNG is deterministically
    /// seeded from that value. Otherwise, entropy-based RNG is used.
    fn get_rng_for_selection(&self) -> StdRng {
        // The old implementation cloned the stored RNG on every generation,
        // replaying identical parent choices. Derive a generation-specific
        // stream for seeded searches so reproducibility and progression both
        // hold. Unseeded searches retain entropy-based behavior.
        match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed.wrapping_add(self.generation.get())),
            None => StdRng::from_entropy(),
        }
    }

    /// Convert a codon (index into the space of genome variants) into a genome.
    /// Used for systematic search when not mutating randomly.
    pub fn codon_to_genome(codon: u64) -> CandidateGenome {
        let packing_idx = ((codon / 8) % 4) as usize;
        let decomp_idx = ((codon / 32) % 4) as usize;
        let fusion_idx = ((codon / 128) % 3) as usize;

        let repr_idx = (codon % 8) as usize;
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
            ane_unit: Default::default(),
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
    use crate::evolution::evaluate::SyntheticEvaluator;
    use crate::evolution::foundation::{FitnessScore, RepresentationAxis};

    #[test]
    fn crossover_produces_child() {
        let system = JointEvolutionSystem::default();
        let parent_a = CandidateGenome::new();
        let mut parent_b = CandidateGenome::new();
        parent_b.representation = RepresentationAxis::Binary1;
        let mut rng = rand::thread_rng();
        let child = system.joint_crossover(&parent_a, &parent_b, &mut rng);
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
        let mut rng = rand::thread_rng();
        let mutated = system.joint_mutate(&original, &mut rng);
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
    fn run_generation_produces_population_and_tracks_generation() {
        let system = JointEvolutionSystem::default();
        let mut frontier = ParetoFrontier::new(2);

        let population: Vec<ScoredGenome> = (0..10)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5 + (i as f64) * 0.05, 0.5],
            })
            .collect();

        for sg in population.iter() {
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                sg.fitness.iter().map(|&v| FitnessScore::new(v)).collect(),
                0,
                &Default::default(),
            );
        }

        assert_eq!(system.generation.get(), 0);
        let (next, stop_reason) = system.run_generation(&population, &frontier);
        assert!(!next.is_empty());
        assert!(next.len() <= system.config.population_size);
        assert_eq!(system.generation.get(), 1);
        // With default config of 100 max generations and 10 stall limit,
        // generation 1 should not stop.
        assert!(stop_reason.is_none());
    }

    #[test]
    fn should_stop_returns_reason_at_limit() {
        let system = JointEvolutionSystem::new(JointSearchConfig {
            max_generations: 2,
            stagnation_limit: 1,
            ..Default::default()
        });

        let mut frontier = ParetoFrontier::new(1);
        let population: Vec<ScoredGenome> = (0..5)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5],
            })
            .collect();

        for sg in &population {
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                vec![FitnessScore::new(sg.fitness[0])],
                0,
                &Default::default(),
            );
        }

        for sg in &population {
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                vec![FitnessScore::new(sg.fitness[0])],
                0,
                &Default::default(),
            );
        }

        // Generation 0: not at limit, should continue.
        assert!(system.should_stop(&frontier).is_none());

        // Advance to generation 2, which equals max_generations.
        system.generation.set(2);
        let reason = system.should_stop(&frontier);
        assert!(reason.is_some());
        assert!(reason.as_ref().unwrap().contains("generation_limit"));
    }

    #[test]
    fn should_stop_detects_stagnation() {
        // Separate system with high max_generations so stagnation check fires
        // before the generation-limit check.
        let system = JointEvolutionSystem::new(JointSearchConfig {
            max_generations: 100,
            stagnation_limit: 5,
            ..Default::default()
        });

        let mut frontier = ParetoFrontier::new(1);
        for i in 0..5 {
            frontier.insert(
                prism_ecs_core::Entity::new(i as u64 + 1, 0),
                vec![FitnessScore::new(0.5)],
                0,
                &Default::default(),
            );
        }

        // Generation 0: first call initialises tracking.
        assert!(system.should_stop(&frontier).is_none());
        assert_eq!(system.best_fitness.get(), 0.5);
        assert_eq!(system.best_fitness_generation.get(), 0);

        // Advance to generation 2 (stalled = 2 - 0 = 2, within limit 5).
        system.generation.set(2);
        assert!(system.should_stop(&frontier).is_none());

        // Advance to generation 7 (stalled = 7 - 1 = 6, past limit 5).
        system.generation.set(7);
        system.best_fitness.set(0.5);
        system.best_fitness_generation.set(1);
        let reason = system.should_stop(&frontier);
        assert!(reason.is_some());
        let msg = reason.unwrap();
        assert!(
            msg.contains("stall_limit"),
            "expected stall_limit in '{msg}'"
        );
    }

    #[test]
    fn estimate_and_score_genome_returns_scored_genome() {
        let system = JointEvolutionSystem::default();
        let genome = CandidateGenome::new();
        let evaluator = SyntheticEvaluator::new();

        let scored = system.estimate_and_score_genome(&genome, &evaluator, b"4096,4096");

        // The genome should be the original.
        assert_eq!(
            JointEvolutionSystem::genome_digest(&scored.genome),
            JointEvolutionSystem::genome_digest(&genome)
        );
        // Fitness should be a positive scalar value.
        assert_eq!(scored.fitness.len(), 1);
        assert!(scored.fitness[0] > 0.0);
        assert!(scored.fitness[0] <= 1.0);
    }

    #[test]
    fn seeded_rng_produces_deterministic_results() {
        let config = JointSearchConfig {
            seed: Some(42),
            ..Default::default()
        };
        let system = JointEvolutionSystem::new(config.clone());
        let system2 = JointEvolutionSystem::new(config);

        let frontier = ParetoFrontier::new(1);
        let population: Vec<ScoredGenome> = (0..5)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5],
            })
            .collect();

        let (next1, _) = system.run_generation(&population, &frontier);
        let (next2, _) = system2.run_generation(&population, &frontier);

        // With the same seed, the crossover and mutation should produce
        // the same next generation.
        assert_eq!(next1.len(), next2.len());
        for (a, b) in next1.iter().zip(next2.iter()) {
            assert_eq!(
                JointEvolutionSystem::genome_digest(&a.genome),
                JointEvolutionSystem::genome_digest(&b.genome)
            );
        }
    }

    #[test]
    fn different_seed_produces_different_results() {
        let config_a = JointSearchConfig {
            seed: Some(42),
            ..Default::default()
        };
        let config_b = JointSearchConfig {
            seed: Some(99),
            ..Default::default()
        };
        let system_a = JointEvolutionSystem::new(config_a);
        let system_b = JointEvolutionSystem::new(config_b);

        let mut frontier = ParetoFrontier::new(1);
        let population: Vec<ScoredGenome> = (0..5)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5],
            })
            .collect();

        for sg in &population {
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                vec![FitnessScore::new(sg.fitness[0])],
                0,
                &Default::default(),
            );
        }

        let (next_a, _) = system_a.run_generation(&population, &frontier);
        let (next_b, _) = system_b.run_generation(&population, &frontier);
        // Different seeds should produce different offspring genomes.
        let a_digests: Vec<String> = next_a
            .iter()
            .map(|sg| JointEvolutionSystem::genome_digest(&sg.genome))
            .collect();
        let b_digests: Vec<String> = next_b
            .iter()
            .map(|sg| JointEvolutionSystem::genome_digest(&sg.genome))
            .collect();
        assert_ne!(
            a_digests, b_digests,
            "different seeds must produce different results"
        );
    }

    #[test]
    fn run_generation_stops_on_stagnation() {
        let system = JointEvolutionSystem::new(JointSearchConfig {
            max_generations: 100,
            stagnation_limit: 3,
            ..Default::default()
        });

        let mut frontier = ParetoFrontier::new(1);
        let population: Vec<ScoredGenome> = (0..5)
            .map(|i| ScoredGenome {
                genome: JointEvolutionSystem::codon_to_genome(i),
                fitness: vec![0.5],
            })
            .collect();

        for sg in &population {
            frontier.insert(
                prism_ecs_core::Entity::new(frontier.entries.len() as u64 + 1, 0),
                vec![FitnessScore::new(sg.fitness[0])],
                0,
                &Default::default(),
            );
        }

        // First call: initialises tracking, generation 0 -> 1, not stalled.
        let (_next, stop1) = system.run_generation(&population, &frontier);
        assert!(stop1.is_none());

        // Manually set state so that stagnation fires on next run_generation.
        // best_fitness_generation was set to 0 on first call, so at generation 3
        // we have stalled 3 generations (3 - 0 = 3) >= limit 3.
        system.generation.set(3);
        system.best_fitness.set(0.5);
        system.best_fitness_generation.set(0);

        let (_, stop2) = system.run_generation(&population, &frontier);
        // Generation becomes 4, stalled = 4 - 0 = 4 >= 3.
        assert!(stop2.is_some());
        let msg = stop2.unwrap();
        assert!(
            msg.contains("stall_limit"),
            "expected stall_limit in '{msg}'"
        );
    }
}
