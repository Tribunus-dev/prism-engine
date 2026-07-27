//! Production-mode fail-closed semantics for the search-system
//! measurement path.
//!
//! This module owns the single authority for "fail closed when the
//! real measurement is unavailable." Three entry points enforce it:
//!
//! 1. [`extract_measurements`] — turns a fitness score into
//!    [`CandidateMeasurements`], refusing to fabricate a measurement
//!    when the wrapped evaluator is synthetic or the score is
//!    non-finite / non-positive.
//! 2. [`evaluate_ternary_evidence`] — produces
//!    [`TernaryObjectiveEvidence`] from the adapter, refusing to
//!    proceed when the adapter is synthetic or the inner score is
//!    invalid. The behavioral probe (if installed) supplies the
//!    activation, logit, and router fields.
//! 3. [`create_measured_evaluator_from_daemon`] — the daemon
//!    integration point; returns an explicit "MeasuredEvaluator not
//!    available" error when the daemon cannot supply a real
//!    evaluator. Callers must not retry with a synthetic fallback.
//!
//! The fail-closed contract is the production-mode safety gate: a
//! non-real evaluator is never silently coerced into producing a
//! measurement that downstream CImage promotion would treat as
//! canonical. All three entry points return
//! [`SearchError::SyntheticDataInProductionMode`] or
//! [`SearchError::CorrectnessValidationFailed`] when the contract is
//! violated; the executor wrapper at [`super::strategy`] collapses
//! these errors to a missing-evidence marker so its trait signature
//! is preserved, but every other caller surfaces them.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Instant;

use prism_ecs_ir::evolution::evaluate::EvaluationStrategy as EcsEvaluationStrategy;
use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::FitnessScore;
use prism_ecs_ir::evolution::TernaryObjectiveEvidence;

use crate::search::SearchError;
use crate::CandidateMeasurements;

use super::objective::mapped_probe_evaluate_for_fail_closed;
use super::strategy::MeasuredEvaluatorAdapter;

/// Extract measurements from the wrapped evaluator for a candidate.
/// Fails closed when the evaluator is synthetic or when the fitness
/// score is non-finite / non-positive — those would otherwise flow
/// downstream as canonical measurements.
pub fn extract_measurements(
    adapter: &MeasuredEvaluatorAdapter,
    genome: &CandidateGenome,
    context: &[u8],
    fitness_score: f64,
) -> Result<CandidateMeasurements, SearchError> {
    if adapter.is_synthetic() {
        return Err(SearchError::SyntheticDataInProductionMode);
    }

    if !fitness_score.is_finite() || fitness_score <= 0.0 {
        return Err(SearchError::CorrectnessValidationFailed);
    }

    let start = Instant::now();
    let measured_score = adapter
        .inner
        .as_ref()
        .evaluate(genome, context)
        .value();
    let wall_time_ms = start.elapsed().as_secs_f64() * 1_000.0;
    if !measured_score.is_finite() || measured_score <= 0.0 {
        return Err(SearchError::CorrectnessValidationFailed);
    }

    Ok(CandidateMeasurements {
        wall_time_ms,
        gpu_time_ms: wall_time_ms,
        bandwidth_gbps: if wall_time_ms > 0.0 {
            context.len() as f64 / wall_time_ms / 1_000.0
        } else {
            0.0
        },
        peak_memory_mb: 0.0,
        reconstruction_error: 1.0 - measured_score,
        accuracy_score: measured_score,
    })
}

/// Produce structured evidence for progressive Pareto admission. The
/// backend score is the reference-quality signal; callers that have
/// activation/logit/router probes can replace the remaining fields
/// via the optional `behavioral_probe` slot on the adapter.
///
/// Fails closed when the adapter is synthetic or the inner score is
/// invalid. A ternary candidate without a behavioral probe is also
/// rejected — a ternary admission cannot be made on a backend score
/// alone.
pub fn evaluate_ternary_evidence(
    adapter: &MeasuredEvaluatorAdapter,
    genome: &CandidateGenome,
    context: &[u8],
) -> Result<TernaryObjectiveEvidence, SearchError> {
    if adapter.is_synthetic() {
        return Err(SearchError::SyntheticDataInProductionMode);
    }
    let start = Instant::now();
    let quality = adapter
        .inner
        .as_ref()
        .evaluate(genome, context)
        .value();
    let latency_ms = start.elapsed().as_secs_f64() * 1_000.0;
    if !quality.is_finite() || quality <= 0.0 {
        return Err(SearchError::CorrectnessValidationFailed);
    }
    let ternary_candidate = matches!(
        genome.representation,
        prism_ecs_ir::evolution::RepresentationAxis::Ternary158
            | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
    );
    if ternary_candidate && adapter.behavioral_probe.is_none() {
        return Err(SearchError::CorrectnessValidationFailed);
    }
    let mut evidence = TernaryObjectiveEvidence {
        quality,
        latency_ms,
        memory_bytes: context.len() as u64,
        native_ternary_fraction: if matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        ) {
            1.0
        } else {
            0.0
        },
        activation_error: f64::NAN,
        logit_divergence: f64::NAN,
        task_loss: f64::NAN,
        router_agreement: f64::NAN,
        router_margin_error: f64::NAN,
        logit_cross_entropy: f64::NAN,
        generation_loss: f64::NAN,
        energy: f64::NAN,
        ..Default::default()
    };
    if let Some(probe) = &adapter.behavioral_probe {
        let behavioral = mapped_probe_evaluate_for_fail_closed(probe.as_ref(), genome, context)?;
        evidence.activation_error = behavioral.activation_error;
        evidence.logit_divergence = behavioral.logit_divergence;
        evidence.task_loss = behavioral.task_loss;
        evidence.router_agreement = behavioral.router_agreement;
        evidence.router_margin_error = behavioral.router_margin_error;
        evidence.logit_cross_entropy = behavioral.logit_cross_entropy;
        evidence.generation_loss = behavioral.generation_loss;
        evidence.expert_balance_error = behavioral.expert_balance_error;
        evidence.residual_bytes = behavioral.residual_bytes;
        evidence.energy = behavioral.energy;
    }
    Ok(evidence)
}

/// Create a [`MeasuredEvaluatorAdapter`] from the daemon resource.
/// The daemon integration is the production source of measured
/// evaluators; when the daemon is unavailable, the call must fail
/// rather than coerce a synthetic evaluator into the role. The
/// single-authority failure here is the `SearchFailed` error path —
/// callers downstream treat it as the production gate.
pub fn create_measured_evaluator_from_daemon(
) -> Result<MeasuredEvaluatorAdapter, SearchError> {
    // The daemon integration lives in the engine-side
    // `compute-core`; when the engine is absent (e.g. the
    // constitutional crate is consumed by a headless compiler
    // pipeline) we surface an explicit, structured failure.
    Err(SearchError::SearchFailed(
        "MeasuredEvaluator not available - daemon integration required".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inner evaluator whose `name` contains "Synthetic" so the
    /// adapter recognises it as synthetic.
    struct SyntheticInner;

    impl EcsEvaluationStrategy for SyntheticInner {
        fn evaluate(
            &self,
            _genome: &CandidateGenome,
            _context: &[u8],
        ) -> FitnessScore {
            FitnessScore(0.5)
        }
        fn name(&self) -> &str {
            "SyntheticInner"
        }
    }

    struct MeasuredInner;

    impl EcsEvaluationStrategy for MeasuredInner {
        fn evaluate(
            &self,
            _genome: &CandidateGenome,
            _context: &[u8],
        ) -> FitnessScore {
            FitnessScore(0.5)
        }
        fn name(&self) -> &str {
            "MeasuredInner"
        }
    }

    #[test]
    fn fail_closed_rejects_synthetic_in_production_mode() {
        let adapter = MeasuredEvaluatorAdapter::new(Arc::new(SyntheticInner));
        let genome = CandidateGenome::new();
        let res = extract_measurements(&adapter, &genome, b"ctx", 0.5);
        assert!(matches!(
            res,
            Err(SearchError::SyntheticDataInProductionMode)
        ));
    }

    #[test]
    fn fail_closed_rejects_non_finite_fitness() {
        let adapter = MeasuredEvaluatorAdapter::new(Arc::new(MeasuredInner));
        let genome = CandidateGenome::new();
        let res = extract_measurements(&adapter, &genome, b"ctx", f64::NAN);
        assert!(matches!(
            res,
            Err(SearchError::CorrectnessValidationFailed)
        ));
    }

    #[test]
    fn fail_closed_daemon_integration_returns_explicit_error() {
        let res = create_measured_evaluator_from_daemon();
        assert!(matches!(res, Err(SearchError::SearchFailed(_))));
    }
}
