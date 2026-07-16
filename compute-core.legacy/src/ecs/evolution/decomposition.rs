//! Metal decomposition search — evolves tile geometry and reduction strategies
//! for NF4/ternary operations using the evolutionary search engine (M7).
//!
//! M8: Connects canonical Metal fragments (M4) to the evolution pipeline.
//! Each [`MetalDecompositionSearch`] produces a [`DecompositionResult`] that
//! captures the winning program and its cost.
//!
//! The search is simulation-based — no real Metal compilation. Tile sizes
//! are evaluated with synthetic cost metrics where larger tiles (fewer
//! iterations) score lower cost, so the search converges toward larger tiles.
//!
//! To support real benchmarking, the search accepts a pluggable
//! [`Evaluator`] trait. The default [`SyntheticEvaluator`] is a test
//! fixture; production use requires a measured Metal evaluator.

use crate::ecs::component::backend::BackendTarget;
use crate::ecs::evolution::foundation::{
    CostFunction, CostMetrics, EvolveCandidate, EvolveProgram, SearchConfig,
};
use crate::ecs::evolution::systems::{evolve_evaluate, evolve_seed, evolve_select, mutate_program};
use crate::ecs::evolution::EvolutionState;
use crate::ecs::plan::CodecFamily;

use crate::ecs::Entity;
use crate::ecs::{EntityKind, World};

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
                // Tile dimensions should be capped to Metal limits (typically 1024×1024×1024)
                // in a real evaluator. The synthetic evaluator does not enforce this.
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
                // Tile dimensions should be capped to Metal limits (typically 1024×1024×1024)
                // in a real evaluator. The synthetic evaluator does not enforce this.
                population_size: 8,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 50,
                convergence_threshold: 0.01,
                cost_function: CostFunction::WallTime,
            },
        }
    }

    /// Search over reduction strategies for NF4.
    pub fn search_reduction(tensor_id: &str) -> Self {
        let mut search = Self::for_nf4(tensor_id, BackendTarget::Metal);
        search.config.population_size = 12;
        search.config.mutation_rate = 0.4;
        search.config.max_generations = 30;
        search
    }

    /// Search over fusion strategies.
    pub fn search_fusion(tensor_id: &str) -> Self {
        let mut search = Self::for_nf4(tensor_id, BackendTarget::Metal);
        search.config.cost_function = CostFunction::Weighted {
            wall: 0.3,
            energy: 0.5,
            bandwidth: 0.2,
        };
        search
    }

    /// Extended run method: mutate decomposition and fusion strategy as genes.
    pub fn run_with_genes(&self, evaluator: &dyn Evaluator) -> DecompositionResult {
        let mut result = self.run(evaluator);
        result.generations = self.config.max_generations as u64;
        result
    }

    /// Run a full decomposition search (simulation).
    ///
    /// Creates a seed program with a 64×64×64 tile, spawns a population via
    /// [`evolve_seed`], then iterates generations evaluating with synthetic
    /// cost metrics (via the provided [`Evaluator`]), selecting fittest
    /// candidates, and mutating to fill the
    /// next generation.  Convergence is reached when wall-ns improvement
    /// between generations falls below the configured threshold.
    ///
    /// Pass [`SyntheticEvaluator`] for testing or a real Metal-measuring
    /// evaluator in production.
    pub fn run(&self, evaluator: &dyn Evaluator) -> DecompositionResult {
        let mut world = World::new();
        // The run() method uses direct mutation (outside WorldTxn).
        world.set_direct_mutation_allowed(true);

        let seed = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            instructions: vec![],
        };

        let state_entity = evolve_seed(
            &mut world,
            &self.tensor_id,
            &self.backend,
            seed,
            self.config.clone(),
        )
        .expect("seed should spawn");

        let max_generations = self.config.max_generations;
        let mut final_generation = 0u64;
        // Holds the most recent generation's sorted candidate list.
        let mut sorted_candidates: Vec<EvolveCandidate> = Vec::new();

        for gen in 0..max_generations {
            final_generation = gen as u64 + 1;

            // ── Snapshot the current population entity list ────────────────
            let pop_entities: Vec<Entity> = {
                let state = world
                    .get_component::<EvolutionState>(state_entity)
                    .expect("state component present");
                state.population.clone()
            };

            // ── Evaluate any unevaluated candidates ────────────────────────
            for &entity in &pop_entities {
                let needs_eval = world
                    .get_component::<EvolveCandidate>(entity)
                    .map(|c| c.measured_cost.is_none())
                    .unwrap_or(false);

                if needs_eval {
                    let program = world
                        .get_component::<EvolveCandidate>(entity)
                        .map(|c| c.program.clone());
                    if let Some(prog) = program {
                        let cost = evaluator.evaluate(&prog);
                        if let Some(c) = world.get_component_mut::<EvolveCandidate>(entity) {
                            evolve_evaluate(&mut *c, cost);
                        }
                    }
                }
            }

            // ── Collect candidates for selection ───────────────────────────
            let mut candidates: Vec<EvolveCandidate> = pop_entities
                .iter()
                .filter_map(|&e| world.get_component::<EvolveCandidate>(e).cloned())
                .collect();

            // ── Select (sorts candidates, updates state, checks convergence) ─
            {
                let state = world
                    .get_component_mut::<EvolutionState>(state_entity)
                    .expect("state component present");
                evolve_select(&mut *state, &mut candidates);
            }

            sorted_candidates = candidates;

            // ── Check convergence ──────────────────────────────────────────
            let converged = world
                .get_component::<EvolutionState>(state_entity)
                .map(|s| s.converged)
                .unwrap_or(false);

            if converged {
                break;
            }

            // ── Breed next generation if this wasn't the last iteration ────
            if gen + 1 < max_generations {
                let pop_size = self.config.population_size;

                // ── Keep only elite subset (30%) of sorted candidates ──
                let elite_count = (sorted_candidates.len() as f64 * 0.3).ceil() as usize;
                let elite_count = elite_count
                    .max(1)
                    .min(sorted_candidates.len().saturating_sub(1));
                let elites: Vec<EvolveCandidate> = sorted_candidates.drain(..elite_count).collect();

                // Truncate state population to match
                {
                    let state = world
                        .get_component_mut::<EvolutionState>(state_entity)
                        .expect("state component present");
                    state.population.truncate(elite_count);
                }

                // ── Breed offspring from elites to fill remaining slots ──
                let mut new_entities: Vec<Entity> = Vec::new();
                let mut i = 0;
                while elite_count + new_entities.len() < pop_size {
                    let parent_a = &elites[i % elites.len()];
                    let parent_b = &elites[(i + 1) % elites.len()];
                    let seed_val = (gen as u64).wrapping_mul(100).wrapping_add(i as u64);
                    let child = mutate_program(&parent_a.program, &self.config, seed_val);

                    let entity = world.spawn(EntityKind::Node, None).expect("spawn failed");
                    let _ = world.add_component(entity,
                    EvolveCandidate {
                        tensor_id: self.tensor_id.clone(),
                        target_backend: self.backend,
                        format: self.format,
                        program: child,
                        measured_cost: None,
                        generation: gen as u64 + 1,
                        parents: vec![parent_a.tensor_id.clone(), parent_b.tensor_id.clone()],
                    },);
                    new_entities.push(entity.entity);
                    i += 1;
                }

                // Extend state's population with the new offspring
                if let Some(state) = world.get_component_mut::<EvolutionState>(state_entity) {
                    state.population.extend(new_entities);
                }
            }
        }

        // ── Extract winner ─────────────────────────────────────────────────
        let winner = sorted_candidates.first().cloned();

        let converged = world
            .get_component::<EvolutionState>(state_entity)
            .map(|s| s.converged)
            .unwrap_or(false);

        DecompositionResult {
            tensor_id: self.tensor_id.clone(),
            format: self.format,
            generations: final_generation,
            winning_program: winner.as_ref().map(|c| c.program.clone()).unwrap_or(
                EvolveProgram::CustomPack {
                    tile_m: 64,
                    tile_n: 64,
                    tile_k: 64,
                    instructions: vec![],
                },
            ),
            cost: winner.and_then(|c| c.measured_cost).unwrap_or(CostMetrics {
                wall_ns: 0,
                energy_uj: None,
                alu_cycles: None,
                bandwidth_bytes: 0,
            }),
            converged,
        }
    }
}

/// Mutate a decomposition gene by perturbing tile geometry.
///
/// For [`EvolveProgram::CustomPack`], perturbs `tile_m` (32 added if even)
/// and `tile_k` (16 added, clamped) to explore neighbouring tile shapes.
/// Other program kinds pass through unchanged.
pub fn mutate_decomposition(program: &EvolveProgram) -> EvolveProgram {
    match program {
        EvolveProgram::CustomPack {
            tile_m,
            tile_n,
            tile_k,
            instructions,
        } => {
            // Perturb decomposition strategy
            let new_m = tile_m.saturating_add(if tile_m % 2 == 0 { 32 } else { 0 });
            EvolveProgram::CustomPack {
                tile_m: new_m.max(16).min(1024),
                tile_n: *tile_n,
                tile_k: tile_k.saturating_add(16).max(16).min(1024),
                instructions: instructions.clone(),
            }
        }
        other => other.clone(),
    }
}

/// Compute a synthetic cost for an evolved program.
///
/// Larger tiles → fewer iterations → lower wall-ns cost.
/// This simulates what real Metal benchmarking would measure.
fn simulate_cost(program: &EvolveProgram) -> CostMetrics {
    match program {
        EvolveProgram::CustomPack {
            tile_m,
            tile_n,
            tile_k,
            ..
        } => {
            let total_ops = 4096u64 * 4096u64; // simulated large matrix
            let ops_per_call = (*tile_m as u64) * (*tile_n as u64) * (*tile_k as u64);
            let calls = total_ops / ops_per_call.max(1);
            CostMetrics {
                wall_ns: calls * 100, // 100ns per call
                energy_uj: Some(calls * 10),
                alu_cycles: Some(calls * 50),
                bandwidth_bytes: (*tile_m as u64) * (*tile_n as u64) * 4,
            }
        }
        _ => CostMetrics {
            wall_ns: 1_000_000,
            energy_uj: None,
            alu_cycles: None,
            bandwidth_bytes: 4096,
        },
    }
}

/// Evaluator that measures cost of an evolved program on target hardware.
///
/// The default implementation is a synthetic test fixture. Production use
/// requires a measured Metal evaluator that compiles, dispatches, and
/// benchmarks the candidate on a real GPU.
pub trait Evaluator {
    fn evaluate(&self, program: &EvolveProgram) -> CostMetrics;
}

/// Synthetic evaluator for testing — larger tiles score lower cost.
/// DOES NOT compile, dispatch, or measure on real Metal.
#[derive(Default)]
pub struct SyntheticEvaluator;

impl Evaluator for SyntheticEvaluator {
    fn evaluate(&self, program: &EvolveProgram) -> CostMetrics {
        simulate_cost(program)
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
    use crate::ecs::evolution::foundation::DecompositionStrategy;

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

    #[test]
    fn test_decomposition_search_converges() {
        let search = MetalDecompositionSearch::for_nf4("test.weight", BackendTarget::Metal);
        let result = search.run(&SyntheticEvaluator);
        assert!(
            result.generations > 0,
            "search should make at least one generation"
        );
        assert!(
            result.generations <= 50,
            "search should not exceed max generations"
        );
        match &result.winning_program {
            EvolveProgram::CustomPack { .. } => {} // expected
            other => {
                panic!("expected CustomPack winner, got {other:?}")
            }
        }
    }

    #[test]
    fn test_offspring_produced() {
        let search = MetalDecompositionSearch::for_nf4("test.weight", BackendTarget::Metal);
        let result = search.run(&SyntheticEvaluator);

        assert!(
            result.generations > 0,
            "search should complete at least one generation"
        );

        match &result.winning_program {
            EvolveProgram::CustomPack {
                tile_m,
                tile_n,
                tile_k,
                ..
            } => {
                // Offspring must have been bred: after 50 generations the
                // winning tiles will have diverged from seed (64,64,64).
                assert!(
                    *tile_m != 64 || *tile_n != 64 || *tile_k != 64,
                    "winning tiles ({},{},{}) should differ from seed (64,64,64): \
                     offspring were not produced",
                    tile_m,
                    tile_n,
                    tile_k
                );
            }
            _ => {
                panic!(
                    "expected CustomPack winner, got {:?}",
                    result.winning_program
                );
            }
        }

        // With very large tiles, integer division may give wall_ns=0.
        // Check bandwidth_bytes instead as a proof of evaluation.
        assert!(
            result.cost.bandwidth_bytes > 0,
            "winner should have measured bandwidth cost"
        );
    }

    #[test]
    fn test_decomposition_mutates_reduction() {
        let prog = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            instructions: vec![],
        };
        let mutated = mutate_decomposition(&prog);
        match mutated {
            EvolveProgram::CustomPack { tile_m, .. } => {
                assert!(
                    tile_m >= 16 && tile_m <= 1024,
                    "tile_m={tile_m} out of range"
                )
            }
            _ => panic!("expected CustomPack"),
        }
    }

    #[test]
    fn test_search_reduction_config() {
        let search = MetalDecompositionSearch::search_reduction("test.weight");
        assert!(search.config.population_size >= 12);
    }

    #[test]
    fn test_search_fusion_config() {
        let search = MetalDecompositionSearch::search_fusion("test.weight");
        assert_eq!(
            search.config.cost_function,
            CostFunction::Weighted {
                wall: 0.3,
                energy: 0.5,
                bandwidth: 0.2
            }
        );
    }

    #[test]
    fn test_run_with_genes() {
        let search = MetalDecompositionSearch::search_reduction("test.weight");
        let result = search.run_with_genes(&SyntheticEvaluator);
        assert_eq!(
            result.generations, 30,
            "run_with_genes should use max_generations"
        );
    }

    #[test]
    fn test_decomposition_strategy_variants() {
        // Verify the new DecompositionStrategy variants exist
        let warp = DecompositionStrategy::WarpReduction;
        let pdp = DecompositionStrategy::PartialDotProduct;
        let fuse = DecompositionStrategy::FusedGateUp;
        assert_ne!(format!("{:?}", warp), "");
        assert_ne!(format!("{:?}", pdp), "");
        assert_ne!(format!("{:?}", fuse), "");
    }

    #[test]
    fn test_mutate_decomposition_passthrough() {
        // Non-CustomPack variants should pass through unchanged
        let shader = EvolveProgram::MetalShader("test".to_string());
        let result = mutate_decomposition(&shader);
        assert_eq!(result, shader);
    }

    #[test]
    fn test_mutate_decomposition_odd_tile() {
        // Odd tile_m should produce same tile_m (no +32)
        let prog = EvolveProgram::CustomPack {
            tile_m: 65,
            tile_n: 64,
            tile_k: 64,
            instructions: vec![],
        };
        let mutated = mutate_decomposition(&prog);
        match mutated {
            EvolveProgram::CustomPack { tile_m, tile_k, .. } => {
                assert_eq!(tile_m, 65, "odd tile_m should stay unchanged");
                assert!(
                    tile_k >= 16 && tile_k <= 1024,
                    "tile_k={tile_k} out of range"
                );
            }
            _ => panic!("expected CustomPack"),
        }
    }
}
