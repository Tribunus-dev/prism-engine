//! Stage 1: Evolutionary search systems.
//!
//! These functions seed, evaluate, select, mutate, and notify on the
//! evolutionary search lifecycle.  They operate on the foundation types
//! from [`super::foundation`] and the crate's custom `CompWorld` ECS.

use crate::ecs::component::backend::BackendTarget;
use crate::ecs::evolution::foundation::{
    CostMetrics, EvolutionState, EvolveCandidate, EvolveProgram, SearchConfig,
};
use crate::ecs::plan::CodecFamily;
use crate::ecs::{CompEntity, CompWorld, Component, EntityKind};

// ── Component registration ─────────────────────────────────────────────────
// Foundation types used as ECS components need the Component marker trait.
impl Component for EvolutionState {}
impl Component for EvolveCandidate {}

// ── Public API ─────────────────────────────────────────────────────────────

/// Spawn the initial population from a seed program by introducing random
/// mutations to create `population_size` candidates.
///
/// Returns the entity id of the `EvolutionState` entity (which also carries an
/// `EvolveCandidate` component for the seed candidate).
pub fn evolve_seed(
    world: &mut CompWorld,
    tensor_id: &str,
    target_backend: &BackendTarget,
    seed: EvolveProgram,
    config: SearchConfig,
) -> Result<CompEntity, String> {
    let state_entity = world.spawn(EntityKind::Node, Some("evolution_state".into()));

    world.add_component(
        state_entity,
        EvolutionState {
            tensor_id: tensor_id.to_string(),
            target_backend: *target_backend,
            seed_program: seed.clone(),
            population: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: config.clone(),
        },
    );

    world.add_component(
        state_entity,
        EvolveCandidate {
            tensor_id: tensor_id.to_string(),
            target_backend: *target_backend,
            format: CodecFamily::Ternary,
            program: seed.clone(),
            measured_cost: None,
            generation: 0,
            parents: Vec::new(),
        },
    );

    // Spawn population entities with perturbed programs
    let mut population_entities: Vec<CompEntity> = Vec::with_capacity(config.population_size);

    for _ in 0..config.population_size.saturating_sub(1) {
        let child = mutate_program(&seed, &config);
        let pop_entity = world.spawn(EntityKind::Node, None);
        world.add_component(
            pop_entity,
            EvolveCandidate {
                tensor_id: tensor_id.to_string(),
                target_backend: *target_backend,
                format: CodecFamily::Ternary,
                program: child,
                measured_cost: None,
                generation: 0,
                parents: vec![format!("seed-{}", tensor_id)],
            },
        );
        population_entities.push(pop_entity);
    }

    // Link population into the state
    if let Some(state) = world.get_component_mut::<EvolutionState>(state_entity) {
        state.population = population_entities;
    }

    Ok(state_entity)
}

/// Evaluate one candidate by recording its measured cost.
pub fn evolve_evaluate(candidate: &mut EvolveCandidate, measured: CostMetrics) {
    candidate.measured_cost = Some(measured);
}

/// Select the fittest candidates up to `population_size`.
///
/// Sorts `population` by wall‑clock cost (lower is better), updates the state
/// with the best measurement, checks convergence, and truncates to the
/// configured population size.
pub fn evolve_select(state: &mut EvolutionState, population: &mut [EvolveCandidate]) {
    // Sort by cost (lower is better)
    population.sort_by(|a, b| {
        let a_cost = a
            .measured_cost
            .as_ref()
            .map(|c| c.wall_ns)
            .unwrap_or(u64::MAX);
        let b_cost = b
            .measured_cost
            .as_ref()
            .map(|c| c.wall_ns)
            .unwrap_or(u64::MAX);
        a_cost.cmp(&b_cost)
    });

    // Record best
    if let Some(best) = population.first() {
        // Capture previous best before overwriting
        let prev_best = state.best_cost.clone();
        state.best_cost = best.measured_cost.clone();

        // Check convergence relative to previous best
        if let (Some(prev), Some(curr)) = (prev_best.as_ref(), best.measured_cost.as_ref()) {
            if prev.wall_ns > curr.wall_ns {
                let improvement = (prev.wall_ns - curr.wall_ns) as f64 / prev.wall_ns as f64;
                if improvement < state.search_config.convergence_threshold {
                    state.converged = true;
                }
            }
        }
    }

    // Keep only the fittest
    state
        .population
        .truncate(state.search_config.population_size);
    state.generation += 1;
}

/// Mutate a program to create an offspring.
///
/// For shader programs a comment is appended; for custom‑pack programs the
/// tile dimensions are perturbed; other variants pass through unchanged.
pub fn mutate_program(program: &EvolveProgram, _config: &SearchConfig) -> EvolveProgram {
    match program {
        EvolveProgram::MetalShader(src) => {
            let mut mutated = src.clone();
            mutated.push_str("\n// mutated\n");
            EvolveProgram::MetalShader(mutated)
        }
        EvolveProgram::CustomPack {
            tile_m,
            tile_n,
            tile_k,
            instructions,
        } => EvolveProgram::CustomPack {
            tile_m: tile_m.saturating_add(16),
            tile_n: *tile_n,
            tile_k: *tile_k,
            instructions: instructions.clone(),
        },
        other => other.clone(),
    }
}

/// Create offspring from two parents via crossover.
///
/// Simple implementation: returns a clone of `parent_a`.  A production
/// implementation would blend program features from both parents.
pub fn crossover(parent_a: &EvolveProgram, _parent_b: &EvolveProgram) -> EvolveProgram {
    parent_a.clone()
}

/// Return the best (lowest‑cost) candidate after selection.
pub fn evolve_winner<'a>(
    _state: &EvolutionState,
    population: &'a [EvolveCandidate],
) -> Option<&'a EvolveCandidate> {
    population.first()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::evolution::foundation::CostFunction;

    #[test]
    fn test_evolve_seed_creates_population() {
        let mut world = CompWorld::new();
        world.set_direct_mutation_allowed(true);

        let config = SearchConfig {
            population_size: 5,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };

        let prog = EvolveProgram::MetalShader("kernel void foo() {}".into());
        let entity = evolve_seed(&mut world, "t0", &BackendTarget::Metal, prog, config)
            .expect("evolve_seed should succeed");

        let state = world
            .get_component::<EvolutionState>(entity)
            .expect("state component should exist");
        assert_eq!(state.population.len(), 4); // population_size - 1 (seed is on state entity)
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn test_mutate_shader() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let prog = EvolveProgram::MetalShader("kernel void foo() {}".into());
        let mutated = mutate_program(&prog, &config);
        match mutated {
            EvolveProgram::MetalShader(s) => {
                assert!(s.len() > 20, "mutated shader should be longer")
            }
            _ => panic!("expected MetalShader variant"),
        }
    }

    #[test]
    fn test_mutate_custom_pack() {
        let config = SearchConfig {
            population_size: 10,
            mutation_rate: 0.3,
            crossover_rate: 0.2,
            max_generations: 100,
            convergence_threshold: 0.01,
            cost_function: CostFunction::WallTime,
        };
        let prog = EvolveProgram::CustomPack {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            instructions: vec![],
        };
        let mutated = mutate_program(&prog, &config);
        match mutated {
            EvolveProgram::CustomPack {
                tile_m,
                tile_n,
                tile_k,
                ..
            } => {
                assert_eq!(tile_m, 80, "tile_m should be mutated by +16");
                assert_eq!(tile_n, 128);
                assert_eq!(tile_k, 32);
            }
            _ => panic!("expected CustomPack variant"),
        }
    }

    #[test]
    fn test_crossover_clones_parent_a() {
        let a = EvolveProgram::MetalShader("kernel A".into());
        let b = EvolveProgram::MetalShader("kernel B".into());
        let child = crossover(&a, &b);
        match (&child, &a) {
            (EvolveProgram::MetalShader(c), EvolveProgram::MetalShader(pa)) => {
                assert_eq!(c, pa, "crossover should clone parent_a");
            }
            _ => panic!("expected MetalShader variant for both"),
        }
    }

    #[test]
    fn test_evolve_evaluate_records_metrics() {
        let mut candidate = EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            measured_cost: None,
            generation: 0,
            parents: vec![],
        };

        let metrics = CostMetrics {
            wall_ns: 1500,
            energy_uj: Some(500),
            alu_cycles: Some(1234),
            bandwidth_bytes: 4096,
        };

        evolve_evaluate(&mut candidate, metrics);
        let recorded = candidate.measured_cost.expect("cost should be recorded");
        assert_eq!(recorded.wall_ns, 1500);
        assert_eq!(recorded.energy_uj, Some(500));
    }

    #[test]
    fn test_evolve_select_picks_lowest_cost() {
        let mut state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            population: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.5,
                cost_function: CostFunction::WallTime,
            },
        };

        let mut pop: Vec<EvolveCandidate> = vec![
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("a".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 200,
                    energy_uj: None,
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
            EvolveCandidate {
                tensor_id: "test".into(),
                target_backend: BackendTarget::Metal,
                format: CodecFamily::Ternary,
                program: EvolveProgram::MetalShader("b".into()),
                measured_cost: Some(CostMetrics {
                    wall_ns: 100,
                    energy_uj: None,
                    alu_cycles: None,
                    bandwidth_bytes: 100,
                }),
                generation: 0,
                parents: vec![],
            },
        ];

        state.best_cost = Some(CostMetrics {
            wall_ns: 300,
            energy_uj: None,
            alu_cycles: None,
            bandwidth_bytes: 100,
        });
        evolve_select(&mut state, &mut pop);
        // 66% improvement (300→100) exceeds 0.5 threshold — still improving, not converged
        assert!(
            !state.converged,
            "66% improvement > 50% threshold: not converged"
        );
        assert_eq!(state.generation, 1);
        // After sorting, the lowest-cost (100) should be first
        assert_eq!(
            pop.first()
                .and_then(|c| c.measured_cost.as_ref())
                .map(|m| m.wall_ns),
            Some(100)
        );
    }

    #[test]
    fn test_evolve_select_converges_on_small_improvement() {
        let mut state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("kernel void foo() {}".into()),
            population: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.5, // requires 50% improvement
                cost_function: CostFunction::WallTime,
            },
        };

        let mut pop: Vec<EvolveCandidate> = vec![EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("a".into()),
            measured_cost: Some(CostMetrics {
                wall_ns: 290,
                energy_uj: None,
                alu_cycles: None,
                bandwidth_bytes: 100,
            }),
            generation: 0,
            parents: vec![],
        }];

        state.best_cost = Some(CostMetrics {
            wall_ns: 300,
            energy_uj: None,
            alu_cycles: None,
            bandwidth_bytes: 100,
        });
        evolve_select(&mut state, &mut pop);
        // 3.3% improvement (300→290) < 50% threshold → converged
        assert!(
            state.converged,
            "3.3% improvement < 50% threshold should trigger convergence"
        );
        assert_eq!(state.generation, 1);
    }

    #[test]
    fn test_evolve_winner_returns_first_after_sort() {
        let state = EvolutionState {
            tensor_id: "test".to_string(),
            target_backend: BackendTarget::Metal,
            seed_program: EvolveProgram::MetalShader("k".into()),
            population: Vec::new(),
            generation: 0,
            best_cost: None,
            best_candidate: None,
            converged: false,
            search_config: SearchConfig {
                population_size: 4,
                mutation_rate: 0.3,
                crossover_rate: 0.2,
                max_generations: 100,
                convergence_threshold: 0.01,
                cost_function: CostFunction::WallTime,
            },
        };

        let pop = vec![EvolveCandidate {
            tensor_id: "test".into(),
            target_backend: BackendTarget::Metal,
            format: CodecFamily::Ternary,
            program: EvolveProgram::MetalShader("best".into()),
            measured_cost: Some(CostMetrics {
                wall_ns: 50,
                energy_uj: None,
                alu_cycles: None,
                bandwidth_bytes: 100,
            }),
            generation: 0,
            parents: vec![],
        }];

        let winner = evolve_winner(&state, &pop);
        assert!(winner.is_some());
        assert_eq!(
            winner
                .and_then(|c| c.measured_cost.as_ref())
                .map(|m| m.wall_ns),
            Some(50)
        );
    }
}
