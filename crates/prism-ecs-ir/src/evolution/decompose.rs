//! Decomposition system for splitting candidate genomes into sub-problems.
//!
//! The decomposition system breaks a candidate genome into independently
//! evolvable sub-problems (tile geometry, decomposition strategy, memory
//! configuration). Each sub-problem can be searched in parallel, reducing
//! the effective search space dimensionality.

use crate::evolution::foundation::CandidateGenome;
use prism_ecs_core::Component;

/// Configuration for the decomposition search.
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Whether to enable sub-problem decomposition.
    pub enabled: bool,
    /// Maximum number of sub-problems to create.
    pub max_sub_problems: usize,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_sub_problems: 4,
        }
    }
}

impl Component for DecompositionConfig {}

/// A single sub-problem derived from a candidate genome.
///
/// Each sub-problem focuses on a subset of the genome axes.
#[derive(Debug, Clone)]
pub struct SubProblem {
    /// Index of this sub-problem within the decomposition set.
    pub index: usize,
    /// Name describing this sub-problem (e.g. "tile-geometry", "memory-config").
    pub name: String,
    /// The partial genome for this sub-problem (only relevant axes filled).
    pub partial_genome: CandidateGenome,
    /// Which genome axes are active for this sub-problem (bitmask).
    pub active_axes: u8,
}

impl SubProblem {
    /// Create a new sub-problem exploring a subset of genome axes.
    pub fn new(
        index: usize,
        name: impl Into<String>,
        partial_genome: CandidateGenome,
        active_axes: u8,
    ) -> Self {
        Self {
            index,
            name: name.into(),
            partial_genome,
            active_axes,
        }
    }
}

/// Result of a decomposition search.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// The sub-problems produced by decomposition.
    pub sub_problems: Vec<SubProblem>,
    /// Overall fitness after recombining sub-problem solutions.
    pub recombined_fitness: f64,
    /// Whether the decomposition converged.
    pub converged: bool,
}

/// Decomposition system — splits a genome into sub-problems.
///
/// Attach to an entity with a `DecompositionConfig` component.
#[derive(Debug, Clone)]
pub struct DecompositionSystem {
    /// Configuration governing how genomes are decomposed.
    pub config: DecompositionConfig,
}

impl DecompositionSystem {
    pub fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Decompose a candidate genome into sub-problems.
    ///
    /// Each sub-problem focuses on a different subset of genome axes:
    /// 1. Tile geometry + decomposition axis
    /// 2. Memory configuration
    /// 3. Fusion strategy
    /// 4. Runtime parameters
    ///
    /// The representation and packing axes are held constant across all
    /// sub-problems (they are the highest-impact choices and benefit from
    /// joint optimization with each sub-problem).
    pub fn decompose(&self, genome: &CandidateGenome) -> Vec<SubProblem> {
        if !self.config.enabled {
            return vec![SubProblem::new(
                0,
                "full-genome".to_string(),
                genome.clone(),
                0xFF,
            )];
        }

        let mut problems = Vec::new();

        // Axis bitmask encoding:
        // Bit 0: representation, Bit 1: packing, Bit 2: metal_geometry,
        // Bit 3: decomposition, Bit 4: memory, Bit 5: fusion,
        // Bit 6: engram, Bit 7: runtime
        const GEOMETRY_MASK: u8 = 0b0000_0100; // bit 2
        const DECOMP_MASK: u8 = 0b0000_1000; // bit 3
        const MEMORY_MASK: u8 = 0b0001_0000; // bit 4
        const FUSION_MASK: u8 = 0b0010_0000; // bit 5
        const RUNTIME_MASK: u8 = 0b1000_0000; // bit 7

        // Sub-problem 1: tile geometry + decomposition
        {
            let mut pg = genome.clone();
            pg.memory = Default::default();
            pg.fusion = Default::default();
            pg.runtime = Default::default();
            problems.push(SubProblem::new(
                0,
                "tile-geometry-decomp",
                pg,
                GEOMETRY_MASK | DECOMP_MASK,
            ));
        }

        // Sub-problem 2: memory configuration
        {
            let mut pg = genome.clone();
            pg.metal_geometry = Default::default();
            pg.decomposition = Default::default();
            pg.fusion = Default::default();
            pg.runtime = Default::default();
            problems.push(SubProblem::new(
                1,
                "memory-config",
                pg,
                MEMORY_MASK,
            ));
        }

        // Sub-problem 3: fusion strategy
        {
            let mut pg = genome.clone();
            pg.metal_geometry = Default::default();
            pg.decomposition = Default::default();
            pg.memory = Default::default();
            pg.runtime = Default::default();
            problems.push(SubProblem::new(
                2,
                "fusion-strategy",
                pg,
                FUSION_MASK,
            ));
        }

        // Sub-problem 4: runtime parameters
        {
            let mut pg = genome.clone();
            pg.metal_geometry = Default::default();
            pg.decomposition = Default::default();
            pg.memory = Default::default();
            pg.fusion = Default::default();
            problems.push(SubProblem::new(
                3,
                "runtime-params",
                pg,
                RUNTIME_MASK,
            ));
        }

        problems.truncate(self.config.max_sub_problems);
        problems
    }

    /// Execute a full decomposition search on a genome.
    ///
    /// Returns the decomposition result including sub-problems and a
    /// recombined fitness estimate.
    pub fn run(&self, genome: &CandidateGenome) -> DecompositionResult {
        let sub_problems = self.decompose(genome);
        // Synthetic recombined fitness: average of sub-problem quality scores.
        // In production, this would run each sub-problem through an evaluator.
        let avg_fitness = sub_problems
            .iter()
            .map(|sp| {
                let active_count = sp.active_axes.count_ones() as f64;
                active_count / 8.0
            })
            .sum::<f64>()
            / sub_problems.len().max(1) as f64;

        DecompositionResult {
            sub_problems,
            recombined_fitness: avg_fitness,
            converged: avg_fitness > 0.5,
        }
    }
}

impl Default for DecompositionSystem {
    fn default() -> Self {
        Self::new(DecompositionConfig::default())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_full_genome() {
        let system = DecompositionSystem::default();
        let genome = CandidateGenome::new();
        let problems = system.decompose(&genome);
        assert_eq!(problems.len(), 4);
        // First sub-problem focuses on geometry + decomposition
        assert_eq!(problems[0].name, "tile-geometry-decomp");
    }

    #[test]
    fn decomposition_disabled_returns_single_problem() {
        let config = DecompositionConfig {
            enabled: false,
            max_sub_problems: 4,
        };
        let system = DecompositionSystem::new(config);
        let genome = CandidateGenome::new();
        let problems = system.decompose(&genome);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].name, "full-genome");
    }

    #[test]
    fn decomposition_result_has_fitness() {
        let system = DecompositionSystem::default();
        let genome = CandidateGenome::new();
        let result = system.run(&genome);
        assert_eq!(result.sub_problems.len(), 4);
        assert!(result.recombined_fitness > 0.0);
    }

    #[test]
    fn decomposition_config_component() {
        let config = DecompositionConfig {
            enabled: true,
            max_sub_problems: 2,
        };
        assert_eq!(config.max_sub_problems, 2);
    }
}
