//! Reference-aware, progressive ternary search primitives.
//!
//! This module deliberately contains policy and search interfaces only.  It does
//! not know how a tensor is quantized or packaged; callers provide the measured
//! evidence and mutation/execution callbacks.

use super::{CandidateGenome, FitnessScore, FrontierConfig, ParetoFrontier};

/// Mean absolute error between the reference and candidate router margins.
/// A margin is the gap between the kth selected logit and the best excluded
/// logit, so this metric detects fragile expert decisions before top-k changes.
pub fn router_margin_error(reference: &[f64], candidate: &[f64], top_k: usize) -> f64 {
    if reference.len() != candidate.len() || reference.is_empty() || top_k == 0 {
        return f64::INFINITY;
    }
    fn margin(values: &[f64], k: usize) -> Option<f64> {
        if k > values.len() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let selected = sorted[k - 1];
        let excluded = sorted.get(k).copied().unwrap_or(selected);
        Some(selected - excluded)
    }
    match (margin(reference, top_k), margin(candidate, top_k)) {
        (Some(a), Some(b)) => (a - b).abs(),
        _ => f64::INFINITY,
    }
}

/// Cross entropy H(p_ref, p_candidate) in a numerically stable logit form.
pub fn logit_cross_entropy(reference: &[f64], candidate: &[f64]) -> f64 {
    if reference.len() != candidate.len() || reference.is_empty() {
        return f64::INFINITY;
    }
    let max_ref = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_candidate = candidate.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ref_exp: Vec<f64> = reference.iter().map(|v| (v - max_ref).exp()).collect();
    let cand_exp: Vec<f64> = candidate
        .iter()
        .map(|v| (v - max_candidate).exp())
        .collect();
    let ref_sum: f64 = ref_exp.iter().sum();
    let cand_log_sum = max_candidate + cand_exp.iter().sum::<f64>().ln();
    reference
        .iter()
        .zip(candidate)
        .zip(ref_exp)
        .enumerate()
        .map(|(i, ((_, _), p))| {
            let probability = p / ref_sum;
            probability * (cand_log_sum - candidate[i])
        })
        .sum()
}

/// Structured evidence collected for one candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TernaryObjectiveEvidence {
    pub quality: f64,
    pub activation_error: f64,
    pub logit_divergence: f64,
    pub task_loss: f64,
    pub router_agreement: f64,
    /// Difference between the reference and candidate top-k router margins.
    /// This catches expert re-ordering even when the selected set is unchanged.
    pub router_margin_error: f64,
    /// Drift in the reference-vs-candidate expert activation distribution.
    pub expert_balance_error: f64,
    /// Cross entropy from reference logits to candidate probabilities.
    pub logit_cross_entropy: f64,
    /// Optional rollout/generation loss measured against the reference.
    pub generation_loss: f64,
    pub memory_bytes: u64,
    pub latency_ms: f64,
    pub residual_bytes: u64,
    pub native_ternary_fraction: f64,
    pub energy: f64,
}

/// The objective vector is normalized so every dimension is maximized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TernaryObjectives {
    pub quality: FitnessScore,
    pub latency: FitnessScore,
    pub memory: FitnessScore,
    pub native_coverage: FitnessScore,
    pub residual_cost: FitnessScore,
    pub energy: FitnessScore,
    pub router_margin: FitnessScore,
    pub logit_behavior: FitnessScore,
    pub generation: FitnessScore,
    pub expert_balance: FitnessScore,
}

impl TernaryObjectiveEvidence {
    /// Whether the measured behavioral dimensions satisfy the hard reference
    /// gates. This is the value that must feed CImage promotion; a scalar
    /// backend fitness score is never sufficient.
    pub fn behavioral_passes(self, limits: &TernaryAdmissionLimits) -> bool {
        self.activation_error.is_finite()
            && self.logit_divergence.is_finite()
            && self.task_loss.is_finite()
            && self.router_agreement.is_finite()
            && self.router_margin_error.is_finite()
            && self.logit_cross_entropy.is_finite()
            && self.generation_loss.is_finite()
            && self.expert_balance_error.is_finite()
            && self.activation_error <= limits.max_activation_error
            && self.logit_divergence <= limits.max_logit_divergence
            && self.task_loss <= limits.max_task_loss
            && self.router_agreement >= limits.min_router_agreement
            && self.router_margin_error <= limits.max_router_margin_error
            && self.logit_cross_entropy <= limits.max_logit_cross_entropy
            && self.generation_loss <= limits.max_generation_loss
            && self.expert_balance_error <= limits.max_expert_balance_error
    }

    pub fn objectives(self, limits: &TernaryAdmissionLimits) -> TernaryObjectives {
        let inverse = |value: f64, limit: f64| {
            FitnessScore::new(if limit.is_finite() && limit > 0.0 {
                1.0 - (value.max(0.0) / limit).min(1.0)
            } else {
                0.0
            })
        };
        TernaryObjectives {
            quality: FitnessScore::new(self.quality),
            latency: inverse(self.latency_ms, limits.max_latency_ms),
            memory: inverse(self.memory_bytes as f64, limits.max_memory_bytes as f64),
            native_coverage: FitnessScore::new(self.native_ternary_fraction),
            residual_cost: inverse(self.residual_bytes as f64, limits.max_residual_bytes as f64),
            energy: inverse(self.energy, limits.max_energy),
            router_margin: inverse(self.router_margin_error, limits.max_router_margin_error),
            logit_behavior: inverse(self.logit_cross_entropy, limits.max_logit_cross_entropy),
            generation: inverse(self.generation_loss, limits.max_generation_loss),
            expert_balance: inverse(self.expert_balance_error, limits.max_expert_balance_error),
        }
    }

    pub fn vector(self, limits: &TernaryAdmissionLimits) -> Vec<FitnessScore> {
        let o = self.objectives(limits);
        vec![
            o.quality,
            o.latency,
            o.memory,
            o.native_coverage,
            o.residual_cost,
            o.energy,
            o.router_margin,
            o.logit_behavior,
            o.generation,
            o.expert_balance,
        ]
    }
}

/// Hard correctness and resource gates applied before Pareto insertion.
#[derive(Debug, Clone, Copy)]
pub struct TernaryAdmissionLimits {
    pub max_activation_error: f64,
    pub max_logit_divergence: f64,
    pub max_task_loss: f64,
    pub min_router_agreement: f64,
    pub max_router_margin_error: f64,
    pub max_logit_cross_entropy: f64,
    pub max_generation_loss: f64,
    pub max_expert_balance_error: f64,
    pub min_native_ternary_fraction: f64,
    pub max_latency_ms: f64,
    pub max_memory_bytes: u64,
    pub max_residual_bytes: u64,
    pub max_energy: f64,
}

impl Default for TernaryAdmissionLimits {
    fn default() -> Self {
        Self {
            max_activation_error: 0.05,
            max_logit_divergence: 0.05,
            max_task_loss: 0.1,
            min_router_agreement: 0.95,
            max_router_margin_error: 0.05,
            max_logit_cross_entropy: 0.05,
            max_generation_loss: 0.1,
            max_expert_balance_error: 0.05,
            min_native_ternary_fraction: 0.8,
            max_latency_ms: f64::MAX,
            max_memory_bytes: u64::MAX,
            max_residual_bytes: u64::MAX,
            max_energy: f64::MAX,
        }
    }
}

impl TernaryAdmissionLimits {
    /// Production policy for aggressive ternary search. The evolutionary
    /// scorer may explore high-error candidates, but promotion remains gated
    /// against the BF16/reference behavioral measurements.
    pub fn from_environment() -> Self {
        let mut limits = Self::default();
        if std::env::var("PRISM_AGGRESSIVE_TERNARY").ok().as_deref() == Some("1") {
            limits.min_native_ternary_fraction = 0.95;
            limits.max_activation_error = 0.02;
            limits.max_logit_divergence = 0.02;
            limits.max_task_loss = 0.05;
            limits.max_logit_cross_entropy = 0.02;
            limits.max_generation_loss = 0.05;
            limits.max_router_margin_error = 0.02;
            limits.max_expert_balance_error = 0.02;
        }
        if let Ok(value) = std::env::var("PRISM_MIN_TERNARY_FRACTION") {
            if let Ok(value) = value.parse::<f64>() {
                limits.min_native_ternary_fraction = value.clamp(0.0, 1.0);
            }
        }
        limits
    }
}

impl TernaryAdmissionLimits {
    pub fn admits(&self, e: &TernaryObjectiveEvidence) -> bool {
        e.quality.is_finite()
            && e.activation_error.is_finite()
            && e.logit_divergence.is_finite()
            && e.task_loss.is_finite()
            && e.router_agreement.is_finite()
            && e.router_margin_error.is_finite()
            && e.logit_cross_entropy.is_finite()
            && e.generation_loss.is_finite()
            && e.expert_balance_error.is_finite()
            && e.latency_ms.is_finite()
            && e.energy.is_finite()
            && e.behavioral_passes(self)
            && e.native_ternary_fraction >= self.min_native_ternary_fraction
            && e.latency_ms <= self.max_latency_ms
            && e.memory_bytes <= self.max_memory_bytes
            && e.residual_bytes <= self.max_residual_bytes
            && e.energy <= self.max_energy
    }
}

pub trait ProgressiveStageExecutor: Send + Sync {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        stage: usize,
        context: &[u8],
    ) -> TernaryObjectiveEvidence;
}

pub struct ProgressiveSearchConfig {
    pub stages: usize,
    pub frontier: FrontierConfig,
    pub limits: TernaryAdmissionLimits,
}

impl Default for ProgressiveSearchConfig {
    fn default() -> Self {
        Self {
            stages: 1,
            frontier: FrontierConfig::default(),
            limits: TernaryAdmissionLimits::default(),
        }
    }
}

impl ProgressiveSearchConfig {
    pub fn from_environment() -> Self {
        let mut config = Self::default();
        config.limits = TernaryAdmissionLimits::from_environment();
        config
    }
}

pub struct ProgressiveParetoSearch<'a> {
    pub config: ProgressiveSearchConfig,
    pub executor: &'a dyn ProgressiveStageExecutor,
}

impl<'a> ProgressiveParetoSearch<'a> {
    pub fn run<F>(&self, seed: Vec<CandidateGenome>, mutate: F) -> ParetoFrontier
    where
        F: FnMut(&CandidateGenome, usize) -> Vec<CandidateGenome>,
    {
        self.run_with_context(seed, &[], mutate)
    }

    pub fn run_with_context<F>(
        &self,
        seed: Vec<CandidateGenome>,
        context: &[u8],
        mut mutate: F,
    ) -> ParetoFrontier
    where
        F: FnMut(&CandidateGenome, usize) -> Vec<CandidateGenome>,
    {
        let mut population = seed;
        let mut frontier = ParetoFrontier::new(10);
        for stage in 0..self.config.stages.max(1) {
            let current = std::mem::take(&mut population);
            let mut next = Vec::new();
            for genome in current {
                let evidence = self.executor.evaluate(&genome, stage, context);
                if self.config.limits.admits(&evidence) {
                    frontier.insert(
                        genome.clone(),
                        evidence.vector(&self.config.limits),
                        stage as u64,
                        &self.config.frontier,
                    );
                    next.extend(mutate(&genome, stage));
                }
            }
            population = next;
        }
        frontier
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Executor;
    impl ProgressiveStageExecutor for Executor {
        fn evaluate(
            &self,
            g: &CandidateGenome,
            stage: usize,
            _context: &[u8],
        ) -> TernaryObjectiveEvidence {
            TernaryObjectiveEvidence {
                quality: if stage == 0 { 0.99 } else { 1.0 },
                native_ternary_fraction: if matches!(
                    g.representation,
                    super::super::RepresentationAxis::Ternary158
                ) {
                    1.0
                } else {
                    0.0
                },
                router_agreement: 1.0,
                ..Default::default()
            }
        }
    }
    #[test]
    fn rejects_non_ternary_candidate_before_frontier() {
        let mut g = CandidateGenome::new();
        g.representation = super::super::RepresentationAxis::Ternary158;
        let mut bad = g.clone();
        bad.representation = super::super::RepresentationAxis::Fp16;
        let search = ProgressiveParetoSearch {
            config: ProgressiveSearchConfig {
                stages: 1,
                ..Default::default()
            },
            executor: &Executor,
        };
        let f = search.run(vec![bad, g], |candidate, _| vec![candidate.clone()]);
        assert_eq!(f.len(), 1);
    }
    #[test]
    fn stages_mutate_survivors_only() {
        let mut g = CandidateGenome::new();
        g.representation = super::super::RepresentationAxis::Ternary158;
        let search = ProgressiveParetoSearch {
            config: ProgressiveSearchConfig {
                stages: 2,
                ..Default::default()
            },
            executor: &Executor,
        };
        let f = search.run(vec![g], |candidate, stage| {
            if stage == 0 {
                vec![candidate.clone()]
            } else {
                vec![]
            }
        });
        assert_eq!(f.entries.iter().filter(|e| e.generation == 1).count(), 1);
    }

    #[test]
    fn behavioral_metrics_capture_margin_and_logit_drift() {
        assert_eq!(
            router_margin_error(&[4.0, 1.0, 0.0], &[3.0, 2.0, 0.0], 1),
            2.0
        );
        let identical = logit_cross_entropy(&[2.0, 1.0], &[2.0, 1.0]);
        let shifted = logit_cross_entropy(&[2.0, 1.0], &[1.0, 2.0]);
        assert!(identical > 0.0 && identical < shifted);
    }
}
