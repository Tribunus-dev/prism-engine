//! Decomposition system for splitting candidate genomes into sub-problems.

use crate::evolution::foundation::{
    CandidateGenome, GenomeAxis, GenomeAxisSet, GENOME_AXIS_COUNT,
};
use prism_ecs_core::Component;

#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    pub enabled: bool,
    pub max_sub_problems: usize,
}

impl Default for DecompositionConfig {
    fn default() -> Self { Self { enabled: true, max_sub_problems: 4 } }
}
impl Component for DecompositionConfig {}

#[derive(Debug, Clone)]
pub struct SubProblem {
    pub index: usize,
    pub name: String,
    pub partial_genome: CandidateGenome,
    pub active_axes: GenomeAxisSet,
}

impl SubProblem {
    pub fn new(index: usize, name: impl Into<String>, partial_genome: CandidateGenome, active_axes: GenomeAxisSet) -> Self {
        Self { index, name: name.into(), partial_genome, active_axes }
    }
}

#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub sub_problems: Vec<SubProblem>,
    pub recombined_fitness: f64,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct DecompositionSystem { pub config: DecompositionConfig }

impl DecompositionSystem {
    pub fn new(config: DecompositionConfig) -> Self { Self { config } }

    pub fn decompose(&self, genome: &CandidateGenome) -> Vec<SubProblem> {
        if !self.config.enabled {
            return vec![SubProblem::new(0, "full-genome", genome.clone(), GenomeAxisSet::all())];
        }
        let geometry = GenomeAxisSet::from_axis(GenomeAxis::MetalGeometry)
            .union(GenomeAxisSet::from_axis(GenomeAxis::Decomposition));
        let memory = GenomeAxisSet::from_axis(GenomeAxis::Memory);
        let fusion = GenomeAxisSet::from_axis(GenomeAxis::Fusion);
        let runtime = GenomeAxisSet::from_axis(GenomeAxis::Runtime);
        let mut problems = Vec::new();

        let mut pg = genome.clone();
        pg.memory = Default::default(); pg.fusion = Default::default(); pg.runtime = Default::default();
        problems.push(SubProblem::new(0, "tile-geometry-decomp", pg, geometry));

        let mut pg = genome.clone();
        pg.metal_geometry = Default::default(); pg.decomposition = Default::default(); pg.fusion = Default::default(); pg.runtime = Default::default();
        problems.push(SubProblem::new(1, "memory-config", pg, memory));

        let mut pg = genome.clone();
        pg.metal_geometry = Default::default(); pg.decomposition = Default::default(); pg.memory = Default::default(); pg.runtime = Default::default();
        problems.push(SubProblem::new(2, "fusion-strategy", pg, fusion));

        let mut pg = genome.clone();
        pg.metal_geometry = Default::default(); pg.decomposition = Default::default(); pg.memory = Default::default(); pg.fusion = Default::default();
        problems.push(SubProblem::new(3, "runtime-params", pg, runtime));

        problems.truncate(self.config.max_sub_problems);
        problems
    }

    pub fn run(&self, genome: &CandidateGenome) -> DecompositionResult {
        let sub_problems = self.decompose(genome);
        let avg_fitness = sub_problems.iter().map(|sp| sp.active_axes.count() as f64 / GENOME_AXIS_COUNT as f64).sum::<f64>() / sub_problems.len().max(1) as f64;
        DecompositionResult { converged: avg_fitness > 0.5, recombined_fitness: avg_fitness, sub_problems }
    }
}

impl Default for DecompositionSystem { fn default() -> Self { Self::new(DecompositionConfig::default()) } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn decompose_full_genome() { let problems = DecompositionSystem::default().decompose(&CandidateGenome::new()); assert_eq!(problems.len(), 4); assert_eq!(problems[0].name, "tile-geometry-decomp"); assert!(problems[0].active_axes.contains(GenomeAxis::MetalGeometry)); }
    #[test] fn decomposition_disabled_returns_single_problem() { let system = DecompositionSystem::new(DecompositionConfig { enabled: false, max_sub_problems: 4 }); let problems = system.decompose(&CandidateGenome::new()); assert_eq!(problems.len(), 1); assert_eq!(problems[0].active_axes.count() as usize, GENOME_AXIS_COUNT); }
    #[test] fn decomposition_result_has_fitness() { let result = DecompositionSystem::default().run(&CandidateGenome::new()); assert_eq!(result.sub_problems.len(), 4); assert!(result.recombined_fitness > 0.0); }
}
