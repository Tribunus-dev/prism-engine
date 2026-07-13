use crate::ecs::canonical::identity::CandidateId;
use crate::ecs::evolution::foundation::EvolutionCandidate;

/// Deterministic replay engine.
/// Plan Section 9: "Deterministic replay: Identical source, policy, corpus,
/// seed, compiler, and hardware profile reproduce the same candidate sequence
/// and artifact identities."
pub struct ReplayEngine {
    /// Seed that produced the recorded sequence (stored for contract
    /// enforcement even if not actively used during replay).
    #[allow(dead_code)]
    seed: u64,
    recorded: Vec<CandidateId>,
}

impl ReplayEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            recorded: Vec::new(),
        }
    }

    pub fn record(&mut self, candidate_id: CandidateId) {
        self.recorded.push(candidate_id);
    }

    pub fn replay(&self, candidates: &[EvolutionCandidate]) -> ReplayOutcome {
        let mut matches = 0;
        let mut mismatches = 0;
        for (expected, actual) in self.recorded.iter().zip(candidates.iter()) {
            if expected.0 == actual.candidate_id.0 {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }
        ReplayOutcome {
            matches,
            mismatches,
            total_expected: self.recorded.len(),
            total_actual: candidates.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    pub matches: usize,
    pub mismatches: usize,
    pub total_expected: usize,
    pub total_actual: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::cimage::PhysicalTileLayout;
    use crate::ecs::evolution::foundation::{
        CandidateGenome, CandidateStatus, DecompositionStrategy, FitnessVector, MemoryConfig,
        MetalGeometry,
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

    fn sample_candidate(id: &str, quality: f64) -> EvolutionCandidate {
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
            status: CandidateStatus::Measured,
        }
    }

    #[test]
    fn test_replay_engine_records_and_replays_match() {
        let mut engine = ReplayEngine::new(42);
        engine.record(CandidateId("a".into()));
        engine.record(CandidateId("b".into()));

        let candidates = vec![sample_candidate("a", 1.0), sample_candidate("b", 2.0)];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 2);
        assert_eq!(outcome.mismatches, 0);
        assert_eq!(outcome.total_expected, 2);
        assert_eq!(outcome.total_actual, 2);
    }

    #[test]
    fn test_replay_engine_detects_mismatch() {
        let mut engine = ReplayEngine::new(42);
        engine.record(CandidateId("a".into()));
        engine.record(CandidateId("c".into()));

        let candidates = vec![sample_candidate("a", 1.0), sample_candidate("b", 2.0)];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 1);
        assert_eq!(outcome.mismatches, 1);
    }

    #[test]
    fn test_replay_engine_empty() {
        let engine = ReplayEngine::new(42);
        let candidates: Vec<EvolutionCandidate> = vec![];
        let outcome = engine.replay(&candidates);
        assert_eq!(outcome.matches, 0);
        assert_eq!(outcome.total_expected, 0);
        assert_eq!(outcome.total_actual, 0);
    }
}
