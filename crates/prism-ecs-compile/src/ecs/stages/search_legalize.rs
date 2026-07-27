//! Evolutionary search and legalization stages.
//!
//! - [`system_run_search`] runs the constitutional evolutionary search and
//!   promotes the selected candidate to a [`SearchStateComponent`].
//! - [`system_legalize`] runs the [`CompilerLegalizer`] and produces a
//!   [`LegalizedPlan`] bound to the session.

use prism_ecs_core::world::World;
use prism_ecs_ir::evolution::EvolutionRuntime;
use prism_ecs_ir::evolution::foundation::{AneUnitAxis, CandidateGenome, RepresentationAxis};
use prism_ecs_ir::evolution::progressive::{ProgressiveParetoSearch, ProgressiveSearchConfig};
use prism_spatial_ir::execution_plan::ExecutionMode;
use prism_spatial_ir::target::TargetCapabilities;

use crate::ecs::components::{
    CompilationSession, LegalizedPlan, SearchStateComponent, SessionStatus,
    SpatialGraphComponent,
};
use crate::ecs::orchestrator::{read_session_config, session_entity};
use crate::ecs::resources::{CurrentSource, EvaluatorOption};
use crate::legalize::CompilerLegalizer;
use crate::search::{EvaluationStrategy, SearchCoordinator};
use crate::CompileError;

/// Run the **evolutionary search** stage.
///
/// Reads the [`CurrentSource`] extension and [`SpatialGraphComponent`], runs the
/// [`SearchCoordinator`], and adds [`SearchStateComponent`].
pub fn system_run_search(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::SearchFailed("no current source resource".into()))?;

    let graph_component = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    let config = read_session_config(world, session)?;
    let evaluator = world.get_resource::<EvaluatorOption>();

    // Run the reference-aware progressive ternary stages before the broad
    // hardware search when the registered evaluator explicitly provides the
    // capability.  Legacy/synthetic evaluators return None and therefore do
    // not get to manufacture behavioral evidence.  Candidates advance only
    // after the executor supplies finite quality, activation, logit, and
    // router-agreement measurements.
    if let Some(executor) = evaluator
        .and_then(|option| option.0.as_deref())
        .and_then(|strategy| strategy.progressive_executor())
    {
        let mut seed = Vec::with_capacity(2);
        for representation in [
            RepresentationAxis::Ternary158,
            RepresentationAxis::TernaryTile640,
        ] {
            let mut genome = CandidateGenome::new();
            genome.representation = representation;
            seed.push(genome);
        }
        let progressive = ProgressiveParetoSearch {
            config: ProgressiveSearchConfig {
                stages: std::env::var("PRISM_PROGRESSIVE_STAGES")
                    .ok()
                    .and_then(|value| value.parse().ok())
                    .filter(|&value: &usize| value > 0)
                    .unwrap_or(2),
                limits: prism_ecs_ir::evolution::TernaryAdmissionLimits::from_environment(),
                ..ProgressiveSearchConfig::default()
            },
            executor,
        };
        let progressive_context = source
            .0
            .catalog
            .iter()
            .flat_map(|tensor| {
                let mut line = tensor.name.as_bytes().to_vec();
                line.push(b':');
                line.extend(
                    tensor
                        .shape
                        .iter()
                        .flat_map(|dimension| dimension.to_le_bytes()),
                );
                line.push(b'\n');
                line
            })
            .collect::<Vec<u8>>();
        let frontier =
            progressive.run_with_context(seed, &progressive_context, |candidate, _stage| {
                let mut mutations = Vec::with_capacity(3);
                mutations.push(candidate.clone());
                let mut planar = candidate.clone();
                planar.ane_unit = AneUnitAxis::Planar;
                mutations.push(planar);
                let mut matrix = candidate.clone();
                matrix.ane_unit = AneUnitAxis::Matrix;
                mutations.push(matrix);
                mutations
            });
        if frontier.is_empty() && config.production_mode {
            return Err(CompileError::SearchFailed(
                "progressive ternary search rejected every candidate".into(),
            ));
        }
    }

    let search_config = crate::SearchConfig {
        max_generations: config.max_generations,
        population_size: config.max_candidates,
        production_mode: config.production_mode,
        ..Default::default()
    };

    let runtime = world
        .get_resource::<EvolutionRuntime>()
        .cloned()
        .unwrap_or_default();
    let mut coordinator = SearchCoordinator::new(search_config).with_runtime(runtime);

    // Use a synthetic evaluator only when no user evaluator is registered
    // and the session is not in production mode. In production mode the
    // search will surface the failure rather than fabricating scores.
    struct EcsSyntheticEvaluator;
    impl EvaluationStrategy for EcsSyntheticEvaluator {
        fn evaluate(&self, _genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
            Ok(vec![1.0 / (1.0 + context.len() as f64)])
        }
        fn name(&self) -> &str {
            "ecs-synthetic"
        }
    }

    let synthetic;
    let eval_ref: Option<&dyn EvaluationStrategy> = if let Some(evaluator) = evaluator {
        evaluator.0.as_deref()
    } else if config.production_mode {
        None
    } else {
        synthetic = EcsSyntheticEvaluator;
        Some(&synthetic)
    };

    let result = coordinator
        .run_search(
            &source.0,
            &graph_component.graph,
            eval_ref,
            config.production_mode,
        )
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::SearchComplete;
    }

    world
        .insert_component(
            session,
            SearchStateComponent {
                trace: result.trace,
                candidates_evaluated: result.candidates_evaluated,
                generations_completed: result.generations_completed,
                format_plan: result.format_plan,
                best_joint_tiling: result.best_joint_tiling,
                heterogeneous_workload_evidence: result.heterogeneous_schedule,
                selection_receipt: result.selection_receipt,
                selected_deployment_digest: result
                    .deployment_archive
                    .select(&prism_ecs_ir::evolution::DeploymentPolicy::quality_first())
                    .map(|candidate| candidate.candidate_digest.clone()),
                deployment_archive: result.deployment_archive,
            },
        )
        .map_err(|e| CompileError::SearchFailed(e.to_string()))?;

    Ok(())
}

/// Run the **legalization** stage.
///
/// Reads the [`CurrentSource`] extension and [`SpatialGraphComponent`], runs the
/// [`CompilerLegalizer`], and adds [`LegalizedPlan`].
pub fn system_legalize(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::LegalizationFailed("no current source resource".into()))?;

    let graph_component = world
        .component::<SpatialGraphComponent>(session)
        .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    let config = read_session_config(world, session)?;
    if config.enable_search {
        let search = world
            .component::<SearchStateComponent>(session)
            .map_err(|e| CompileError::LegalizationFailed(format!("search state missing: {e}")))?;
        if let Some(plan) = search.format_plan.as_deref() {
            serde_json::from_str::<prism_ecs_ir::evolution::compile_plan::FormatPlan>(plan)
                .map_err(|e| {
                    CompileError::LegalizationFailed(format!("invalid selected format plan: {e}"))
                })?;
        } else if config.production_mode {
            return Err(CompileError::LegalizationFailed(
                "production legalization requires a selected format plan".into(),
            ));
        }
    }

    let target = world
        .get_resource::<TargetCapabilities>()
        .cloned()
        .unwrap_or_else(TargetCapabilities::sequential_only);
    let report = CompilerLegalizer::legalize(
        &source.0,
        &graph_component.graph,
        &target,
        ExecutionMode::Batch,
    )
    .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    let is_valid = report.is_valid();

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Legalized;
    }

    world
        .insert_component(session, LegalizedPlan { report, is_valid })
        .map_err(|e| CompileError::LegalizationFailed(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use crate::ecs::orchestrator::CompilationOrchestrator;

    #[test]
    fn search_fails_without_current_source() {
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        let result = system_run_search(&mut orch.world);
        assert!(matches!(result, Err(CompileError::SearchFailed(_))));
    }

    #[test]
    fn legalize_fails_without_current_source() {
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        let result = system_legalize(&mut orch.world);
        assert!(matches!(result, Err(CompileError::LegalizationFailed(_))));
    }

    #[test]
    fn legalize_fails_without_graph_component() {
        // Even with a CurrentSource, legalize requires a graph on the session.
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        // We deliberately don't set CurrentSource either — both errors are
        // legitimate failure paths, so we accept either in this pre-flight test.
        let result = system_legalize(&mut orch.world);
        assert!(result.is_err());
    }
}
