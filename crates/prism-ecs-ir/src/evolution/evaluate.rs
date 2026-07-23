//! Evaluation system for the evolutionary search pipeline.
//!
//! Defines the `EvaluationSystem` component that marks entities as evaluators,
//! and the `EvaluationStrategy` trait that pluggable evaluation strategies must
//! implement. The trait provides a common interface for synthetic and measured
//! evaluation backends.

use crate::evolution::foundation::{CandidateGenome, FitnessScore};
use prism_ecs_core::Component;

/// Component that marks an entity as an evaluation system.
///
/// Attach this to an entity in the ECS world to register it as the active
/// evaluator for the evolution pipeline. The evaluator reads candidate
/// genomes and produces fitness scores.
#[derive(Debug, Clone)]
pub struct EvolutionSystem {
    /// Name of this evaluation strategy.
    pub name: String,
}

impl Component for EvolutionSystem {}

/// The evaluation strategy trait.
///
/// Implementations convert a candidate genome into a scalar fitness score
/// by running the candidate through a model of the target hardware (synthetic
/// evaluator) or by dispatching to a concrete runtime (measured evaluator).
pub trait EvaluationStrategy: Send + Sync {
    /// Evaluate a candidate genome and return its fitness score.
    ///
    /// The `context` parameter carries any evaluator-specific metadata needed
    /// to interpret the genome (e.g., model architecture, device parameters).
    fn evaluate(&self, genome: &CandidateGenome, context: &[u8]) -> FitnessScore;

    /// Human-readable label for this evaluation strategy.
    fn name(&self) -> &str;
}

// ── Default Synthetic Evaluator ─────────────────────────────────────────────

/// A synthetic evaluator that scores genomes based on a simple cost model.
///
/// Used during early search phases before measured evaluation is available.
/// The score is derived from bit width, decomposition depth, memory usage,
/// and fusion policy — all computed from the genome axes without touching
/// hardware.
#[derive(Debug, Clone)]
pub struct SyntheticEvaluator {
    name: String,
}

impl SyntheticEvaluator {
    pub fn new() -> Self {
        Self {
            name: "synthetic".into(),
        }
    }
}

impl Default for SyntheticEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl EvaluationStrategy for SyntheticEvaluator {
    fn evaluate(&self, genome: &CandidateGenome, _context: &[u8]) -> FitnessScore {
        // Cost model: lower cost = higher fitness.
        // Base cost starts at 1.0 and increases with resource usage.
        let repr_bits = match genome.representation {
            crate::evolution::foundation::RepresentationAxis::Fp16 => 16.0,
            crate::evolution::foundation::RepresentationAxis::Bf16 => 16.0,
            crate::evolution::foundation::RepresentationAxis::Int8 => 8.0,
            crate::evolution::foundation::RepresentationAxis::Int4 => 4.0,
            crate::evolution::foundation::RepresentationAxis::Nf4 => 4.0,
            crate::evolution::foundation::RepresentationAxis::Nf8 => 8.0,
            crate::evolution::foundation::RepresentationAxis::Ternary158 => 2.0,
            crate::evolution::foundation::RepresentationAxis::TernaryTile640 => 2.0,
            crate::evolution::foundation::RepresentationAxis::Binary1 => 1.0,
        };

        // Lower bit width → lower memory cost → higher fitness.
        let mem_cost = repr_bits / 16.0;

        // More decomposition → more parallelism → higher fitness.
        let decomp_bonus = match genome.decomposition {
            crate::evolution::foundation::DecompositionAxis::Flat => 0.0,
            crate::evolution::foundation::DecompositionAxis::SplitM => 0.05,
            crate::evolution::foundation::DecompositionAxis::SplitMN => 0.10,
            crate::evolution::foundation::DecompositionAxis::SplitMNK => 0.15,
        };

        // Fusion reduces launch overhead.
        let fusion_bonus = match genome.fusion {
            crate::evolution::foundation::FusionAxis::None => 0.0,
            crate::evolution::foundation::FusionAxis::ElementWise => 0.05,
            crate::evolution::foundation::FusionAxis::KernelFusion => 0.10,
        };

        // Higher fitness = lower cost + bonuses.
        let raw = (1.0 - mem_cost * 0.6) + decomp_bonus + fusion_bonus;
        FitnessScore::new(raw)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::foundation::CandidateGenome;

    #[test]
    fn synthetic_evaluator_default_genome() {
        let eval = SyntheticEvaluator::new();
        let genome = CandidateGenome::new();
        let score = eval.evaluate(&genome, &[]);
        // FP16 (16 bits), SplitMN decomposition, ElementWise fusion
        assert!(score.value() > 0.0);
        assert!(score.value() <= 1.0);
    }

    #[test]
    fn binary_genome_scores_higher_than_fp16() {
        let eval = SyntheticEvaluator::new();
        let mut binary_genome = CandidateGenome::new();
        binary_genome.representation =
            crate::evolution::foundation::RepresentationAxis::Binary1;
        let binary_score = eval.evaluate(&binary_genome, &[]);

        let fp16_genome = CandidateGenome::new();
        let fp16_score = eval.evaluate(&fp16_genome, &[]);

        assert!(
            binary_score.value() > fp16_score.value(),
            "binary ({}) should score higher than fp16 ({})",
            binary_score.value(),
            fp16_score.value()
        );
    }

    #[test]
    fn evolution_system_component() {
        let eval = EvolutionSystem {
            name: "test-evaluator".into(),
        };
        assert_eq!(eval.name, "test-evaluator");
    }
}
