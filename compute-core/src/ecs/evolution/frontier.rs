use crate::ecs::evolution::foundation::{EvolutionCandidate, FitnessVector};



/// Pareto frontier — multi-objective ranking.
/// Plan Section 10: "Fitness is a vector rather than one arbitrary scalar.
/// Policy determines Pareto dominance, hard refusals, and final selection."
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    pub candidates: Vec<EvolutionCandidate>,
}

impl ParetoFrontier {
    pub fn new(candidates: Vec<EvolutionCandidate>) -> Self {
        Self { candidates }
    }

    /// Filter to only Pareto-optimal candidates (those not dominated by any other).
    pub fn compute(&self) -> Vec<&EvolutionCandidate> {
        let mut pareto: Vec<&EvolutionCandidate> = Vec::new();
        for candidate in &self.candidates {
            if let Some(fitness) = &candidate.fitness {
                let dominated = self.candidates.iter().any(|other| {
                    if let Some(of) = &other.fitness {
                        other.candidate_id != candidate.candidate_id && dominates(of, fitness)
                    } else {
                        false
                    }
                });
                if !dominated {
                    pareto.push(candidate);
                }
            }
        }
        pareto
    }
}

/// Returns true if `a` dominates `b` (a is better or equal in all objectives
/// and strictly better in at least one).
fn dominates(a: &FitnessVector, b: &FitnessVector) -> bool {
    // All objectives are minimized (lower is better)
    let better_in_any = a.task_quality < b.task_quality
        || a.interference < b.interference
        || a.operator_error < b.operator_error
        || a.memory_bytes < b.memory_bytes
        || a.latency_p50_ns < b.latency_p50_ns
        || a.compile_cost_ms < b.compile_cost_ms;
    let worse_in_none = a.task_quality <= b.task_quality
        && a.interference <= b.interference
        && a.operator_error <= b.operator_error
        && a.memory_bytes <= b.memory_bytes
        && a.latency_p50_ns <= b.latency_p50_ns
        && a.compile_cost_ms <= b.compile_cost_ms;
    better_in_any && worse_in_none
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::identity::CandidateId;
    use crate::ecs::cimage::PhysicalTileLayout;
    use crate::ecs::evolution::foundation::{
        CandidateGenome, CandidateStatus, DecompositionStrategy, MemoryConfig, MetalGeometry,
    };
    use crate::ecs::plan::CodecFamily;

    fn sample_genome() -> CandidateGenome {
        CandidateGenome {
            representation: CodecFamily::Nf4,
            packing: PhysicalTileLayout {
                tile_m: 1,
                tile_n: 640,
                tiles_per_row: 1,
                total_tiles: 1,
                padded_cols: 640,
                group_size: 32,
                groups_per_tile: 20,
                packed_bytes_per_tile: 320,
                metadata_f32_per_tile: 40,
            },
            metal_geometry: MetalGeometry {
                grid_width: 1,
                grid_height: 1,
                simd_width: 32,
                threadgroup_width: 32,
                threadgroup_height: 1,
                threadgroup_depth: 1,
            },
            decomposition: DecompositionStrategy::Sequential,
            memory_config: MemoryConfig {
                vector_width: 4,
                cache_policy: "default".into(),
                threadgroup_staging: 32768,
            },
            fusion_strategy: None,
            engram_config: None,
            kernel_variant: "gemv_nf4_tile640".into(),
        }
    }

    fn make_candidate(id: &str, quality: f64) -> EvolutionCandidate {
        EvolutionCandidate {
            candidate_id: CandidateId(id.into()),
            parent_ids: vec![],
            generation: 0,
            genome: sample_genome(),
            compiled_artifacts: vec![],
            correctness_receipt: None,
            quality_receipt: None,
            performance_receipt: None,
            fitness: Some(FitnessVector {
                task_quality: quality,
                interference: 0.1,
                operator_error: 0.01,
                memory_bytes: 100,
                latency_p50_ns: 50,
                latency_p95_ns: 60,
                energy_uj: None,
                compile_cost_ms: 10,
            }),
            ternary_recipe: None,
            status: CandidateStatus::Measured,
        }
    }

    #[test]
    fn test_pareto_frontier_selects_optimal() {
        let candidates = vec![make_candidate("a", 1.0), make_candidate("b", 2.0)];
        let frontier = ParetoFrontier::new(candidates);
        let optimal = frontier.compute();
        assert_eq!(optimal.len(), 1);
        assert_eq!(optimal[0].candidate_id.0, "a");
    }

    #[test]
    fn test_pareto_frontier_multiple_front() {
        // a dominates in quality/interference/error/latency/cost
        // b dominates in memory — neither fully dominates the other
        let mut a = make_candidate("a", 1.0);
        a.fitness = Some(FitnessVector {
            task_quality: 1.0,
            interference: 0.5,
            operator_error: 0.01,
            memory_bytes: 200,
            latency_p50_ns: 50,
            latency_p95_ns: 60,
            energy_uj: None,
            compile_cost_ms: 10,
        });
        let mut b = make_candidate("b", 3.0);
        b.fitness = Some(FitnessVector {
            task_quality: 2.0,
            interference: 0.1,
            operator_error: 0.02,
            memory_bytes: 50,
            latency_p50_ns: 100,
            latency_p95_ns: 120,
            energy_uj: None,
            compile_cost_ms: 20,
        });
        let candidates = vec![a, b];
        let frontier = ParetoFrontier::new(candidates);
        let optimal = frontier.compute();
        assert_eq!(optimal.len(), 2);
    }

    #[test]
    fn test_pareto_frontier_dominated_excluded() {
        // c is strictly worse than a in every objective
        let mut a = make_candidate("a", 1.0);
        a.fitness = Some(FitnessVector {
            task_quality: 1.0,
            interference: 0.1,
            operator_error: 0.01,
            memory_bytes: 100,
            latency_p50_ns: 50,
            latency_p95_ns: 60,
            energy_uj: None,
            compile_cost_ms: 10,
        });
        let mut c = make_candidate("c", 3.0);
        c.fitness = Some(FitnessVector {
            task_quality: 2.0,
            interference: 0.2,
            operator_error: 0.02,
            memory_bytes: 200,
            latency_p50_ns: 100,
            latency_p95_ns: 120,
            energy_uj: None,
            compile_cost_ms: 20,
        });
        let candidates = vec![a, c];
        let frontier = ParetoFrontier::new(candidates);
        let optimal = frontier.compute();
        assert_eq!(optimal.len(), 1);
        assert_eq!(optimal[0].candidate_id.0, "a");
    }

    #[test]
    fn test_pareto_frontier_skips_no_fitness() {
        let mut no_fit = make_candidate("nope", 1.0);
        no_fit.fitness = None;
        let with_fit = make_candidate("ok", 1.0);
        let candidates = vec![no_fit, with_fit];
        let frontier = ParetoFrontier::new(candidates);
        let optimal = frontier.compute();
        assert_eq!(optimal.len(), 1);
        assert_eq!(optimal[0].candidate_id.0, "ok");
    }
}
