//! Compilation orchestrator — the world + session-entity pipeline driver.
//!
//! Single authority: the pipeline driver that owns a [`World`], spawns the
//! session entity, attaches the constitutional resources, and dispatches the
//! enabled [`CompilationStage`]s in order. The orchestrator itself does not
//! know how each stage works; it only schedules the system functions
//! defined in [`crate::ecs::stages`].
//!
//! Per the constitutional prime directive, the orchestrator never directly
//! spawns canonical entities or mutates components outside the
//! `WorldTxn`/transit boundary. It only owns the world lifecycle and
//! stage dispatch; the per-stage state changes happen in the system
//! functions, each of which uses the canonical session-entity contract.
//!
//! # Usage
//!
//! ```ignore
//! let mut orch = CompilationOrchestrator::new(config);
//! orch.world.add_resource(SourceAdapterList(adapters));
//! let result = orch.run_pipeline()?;
//! ```

use prism_ecs_core::entity::{Entity, EntityKind};
use prism_ecs_core::identity::{CompilerIdentity, SourceFormat};
use prism_ecs_core::world::World;
use prism_ecs_core::{global_context, StateStream};
use prism_ecs_ir::evolution::EvolutionRuntime;
use prism_ecs_source::SourceIdentity;

use crate::compilation_systems::sync_compilation_entity;
use crate::ecs::components::{
    CompilationSession, CImageArtifact, SearchStateComponent, SessionStatus, SourceModel,
    SpatialGraphComponent,
};
use crate::ecs::resources::{ModelManifestResource, SessionHandle, VecEventSink};
use crate::ecs::stages::{
    system_build_graph, system_build_receipt, system_certify, system_detect_source,
    system_emit_cimage, system_generate_kernels, system_legalize, system_run_search,
};
use crate::{
    CompileConfig, CompileError, CompileReceipt, CompileResult, CompileStatus,
    CompilationEvent, CompilationPolicy, CompilationStage, StageResult,
};

/// Owns an ECS world and session entity for a single compilation run.
pub struct CompilationOrchestrator {
    /// The session entity carrying all compilation components.
    pub session: Entity,
    /// The ECS world containing the session entity and all resources.
    pub world: World,
}

impl CompilationOrchestrator {
    /// Create a new orchestrator with an empty world and a session entity.
    ///
    /// The session entity is spawned with [`CompilationSession`] in the
    /// [`SessionStatus::Initialized`] state. Callers must add required
    /// resources (adapters, evaluator, target capabilities) before calling
    /// [`run_pipeline`](Self::run_pipeline).
    pub fn new(config: CompileConfig) -> Self {
        let mut world = World::new();
        let spawned = world
            .spawn(EntityKind::Model, Some("compilation_session".into()))
            .expect("spawn session entity");
        let session = spawned.entity;

        let session_id = uuid::Uuid::new_v4().to_string();
        world
            .insert_component(
                session,
                CompilationSession {
                    config,
                    status: SessionStatus::Initialized,
                    session_id,
                },
            )
            .expect("insert CompilationSession component");

        world.add_resource(SessionHandle(session));
        world.add_resource(VecEventSink::new());
        world.add_resource(EvolutionRuntime::global());
        let trace = global_context();
        world.add_resource(trace.clone());
        world.add_resource(StateStream::global());

        Self { session, world }
    }

    /// Attach the validated multi-model registry that will be embedded in the
    /// emitted CImage header.
    pub fn set_model_manifest(
        &mut self,
        manifest: crate::model_manifest::MultiModelManifest,
    ) -> Result<(), String> {
        manifest.validate()?;
        self.world.set_extension(ModelManifestResource(manifest));
        Ok(())
    }

    /// Run every enabled pipeline stage in order.
    ///
    /// Returns a [`CompileResult`] summarising the outcome. Individual stage
    /// failures are captured as [`CompileStatus::Partial`] or
    /// [`CompileStatus::Failed`].
    pub fn run_pipeline(&mut self) -> Result<CompileResult, CompileError> {
        let (config, policy) = {
            let session_comp = self
                .world
                .component::<CompilationSession>(self.session)
                .map_err(|_| CompileError::PolicyViolation("no session component".into()))?;
            (session_comp.config.clone(), CompilationPolicy::default())
        };

        let stages = policy.enabled_stages().to_vec();
        let mut stage_results: Vec<StageResult> = Vec::with_capacity(stages.len());

        for stage in &stages {
            let started_at = std::time::Instant::now();
            let result = self.run_stage(*stage);
            let duration_ms = started_at.elapsed().as_millis() as u64;

            match result {
                Ok(()) => {
                    stage_results.push(StageResult {
                        stage: *stage,
                        success: true,
                        duration_ms,
                        error: None,
                    });
                    // Emit stage-completed event.
                    if let Some(sink) = self.world.get_resource_mut::<VecEventSink>() {
                        let _ = sink.stage_completed(*stage, duration_ms, "ok");
                    }
                }
                Err(e) => {
                    stage_results.push(StageResult {
                        stage: *stage,
                        success: false,
                        duration_ms,
                        error: Some(e.to_string()),
                    });
                    if let Some(sink) = self.world.get_resource_mut::<VecEventSink>() {
                        let _ = sink.stage_failed(*stage, duration_ms, &e.to_string());
                    }
                    // If a strict policy would abort here, we still stop.
                    break;
                }
            }
        }

        // Build the final CompileResult.
        let all_ok = stage_results.iter().all(|r| r.success);

        let (status, _maybe_error) = if all_ok {
            if let Ok(session) = self.world.component_mut::<CompilationSession>(self.session) {
                session.status = SessionStatus::Complete;
            }
            (CompileStatus::Completed, None)
        } else if stage_results.iter().any(|r| r.success) {
            (CompileStatus::Partial(stage_results.clone()), None)
        } else {
            let err = stage_results
                .first()
                .and_then(|r| r.error.clone())
                .unwrap_or_else(|| "pipeline failed".into());
            (
                CompileStatus::Failed(CompileError::PolicyViolation(err).to_string()),
                Some(CompileError::PolicyViolation(
                    stage_results
                        .last()
                        .and_then(|r| r.error.clone())
                        .unwrap_or_default(),
                )),
            )
        };
        if !all_ok {
            if let Ok(session) = self.world.component_mut::<CompilationSession>(self.session) {
                session.status = SessionStatus::Failed(
                    stage_results
                        .last()
                        .and_then(|stage| stage.error.clone())
                        .unwrap_or_else(|| "pipeline failed".into()),
                );
            }
            let _ = sync_compilation_entity(&mut self.world);
        }

        let output_path = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.output_path.to_string_lossy().to_string())
            .unwrap_or_default();

        let output_digest = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default();

        let events = self
            .world
            .get_resource::<VecEventSink>()
            .map(|s| s.events().to_vec())
            .unwrap_or_default();

        let _config_clone = config.clone();
        let mut receipt = self.make_receipt(&stage_results);

        // Only populate receipt.output_digest when the cimage stage ran.
        if stage_results
            .iter()
            .any(|r| r.stage == CompilationStage::CImageEmission && r.success)
        {
            receipt.output_digest = output_digest.clone();
        }

        Ok(CompileResult {
            request_id: uuid::Uuid::new_v4(),
            status,
            receipt,
            events,
            output_digest,
            output_path: output_path.into(),
        })
    }

    /// Run a single compilation stage.
    ///
    /// Dispatches to the appropriate system function in [`crate::ecs::stages`].
    pub fn run_stage(&mut self, stage: CompilationStage) -> Result<(), CompileError> {
        let result = match stage {
            CompilationStage::SourceDetection => system_detect_source(&mut self.world),
            CompilationStage::SourceIngestion => {
                // Source ingestion is folded into source detection in this
                // pipeline — both happen in system_detect_source.
                Ok(())
            }
            CompilationStage::GraphConstruction => system_build_graph(&mut self.world),
            CompilationStage::EvolutionarySearch => system_run_search(&mut self.world),
            CompilationStage::CandidateMeasurement => {
                // Candidate measurement is folded into the search phase.
                Ok(())
            }
            CompilationStage::Legalization => system_legalize(&mut self.world),
            CompilationStage::TargetLowering => {
                // Target lowering is handled inside legalization.
                Ok(())
            }
            CompilationStage::KernelGeneration => system_generate_kernels(&mut self.world),
            CompilationStage::CImageEmission => system_emit_cimage(&mut self.world),
            CompilationStage::Certification | CompilationStage::Certify => {
                system_certify(&mut self.world)
            }
            CompilationStage::ReceiptBuild => system_build_receipt(&mut self.world),
        };
        if result.is_ok() {
            sync_compilation_entity(&mut self.world)?;
        }
        result
    }

    // ── internal helpers ──────────────────────────────────────────────────

    fn make_receipt(&self, stage_results: &[StageResult]) -> CompileReceipt {
        let source_identity = self
            .world
            .component::<SourceModel>(self.session)
            .ok()
            .map(|m| m.identity.clone())
            .unwrap_or_else(|| SourceIdentity {
                format: SourceFormat::Raw,
                source_digest: String::new(),
                architecture: String::new(),
                model_family: String::new(),
            });

        let candidate_count = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.candidates_evaluated as u32)
            .unwrap_or(0);

        let generations = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.generations_completed as u32)
            .unwrap_or(0);

        let output_digest = self
            .world
            .component::<CImageArtifact>(self.session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default();

        let source_digest = source_identity.source_digest.clone();

        let graph_digest = self
            .world
            .component::<SpatialGraphComponent>(self.session)
            .ok()
            .map(|g| g.graph_digest.clone())
            .unwrap_or_default();

        let search_trace_digest = self
            .world
            .component::<SearchStateComponent>(self.session)
            .ok()
            .map(|s| s.trace.trace_digest.clone())
            .filter(|d| !d.is_empty());

        let events = self
            .world
            .get_resource::<VecEventSink>()
            .map(|s| s.events().to_vec())
            .unwrap_or_default();

        let events_digest = if events.is_empty() {
            String::new()
        } else {
            let json = serde_json::to_string(&events).unwrap_or_default();
            {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(json.as_bytes());
                hex::encode(hasher.finalize())
            }
        };

        CompileReceipt {
            receipt_id: String::new(),
            request_id: uuid::Uuid::new_v4(),
            compiler_identity: CompilerIdentity {
                name: "ecs-orchestrator".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                build_hash: option_env!("PRISM_BUILD_HASH").map(str::to_owned),
                build_timestamp: option_env!("PRISM_BUILD_TIMESTAMP").map(str::to_owned),
            },
            source_identity,
            started_at: chrono::Utc::now(),
            completed_at: chrono::Utc::now(),
            duration_ms: stage_results.iter().map(|r| r.duration_ms).sum(),
            stages: stage_results.to_vec(),
            candidate_count,
            generations,
            output_digest,
            source_digest: Some(source_digest),
            graph_digest: Some(graph_digest),
            search_trace_digest,
            kernel_manifest_digest: None,
            events_digest: Some(events_digest),
            legalization_mode: Some("target_default".into()),
            selection_receipt: self
                .world
                .component::<SearchStateComponent>(self.session)
                .ok()
                .map(|search| search.selection_receipt.clone()),
            uop_tuning_receipt: self
                .world
                .component::<crate::ecs::components::KernelCollection>(self.session)
                .ok()
                .and_then(|kernels| kernels.uop_tuning_receipt.clone()),
            error: None,
            status: CompileStatus::Completed,
            finished_at: chrono::Utc::now(),
            output_path: std::path::PathBuf::new(),
            schema_version: "1.0".into(),
        }
    }
}

// ── private helpers used by the stage systems ─────────────────────────────

pub(crate) fn session_entity(world: &World) -> Result<Entity, CompileError> {
    world
        .get_resource::<SessionHandle>()
        .map(|h| h.0)
        .ok_or_else(|| CompileError::PolicyViolation("no session handle resource".into()))
}

pub(crate) fn read_session_config(
    world: &World,
    session: Entity,
) -> Result<CompileConfig, CompileError> {
    world
        .component::<CompilationSession>(session)
        .map(|s| s.config.clone())
        .map_err(|e| CompileError::PolicyViolation(e.to_string()))
}

// ===========================================================================
// Tests — the orchestrator must be constructible, the session entity must
// carry its initial component, and the stage dispatch must reject
// out-of-order calls.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use prism_ecs_core::entity::EntityKind;

    #[test]
    fn orchestrator_new_constructs_session_with_initial_status() {
        let config = CompileConfig::default();
        let orch = CompilationOrchestrator::new(config.clone());

        assert!(orch.world.is_alive(orch.session));

        let session = orch
            .world
            .component::<CompilationSession>(orch.session)
            .expect("session component should exist");
        assert!(matches!(session.status, SessionStatus::Initialized));
        assert_eq!(session.config.max_candidates, config.max_candidates);
        assert_eq!(session.config.max_generations, config.max_generations);
        assert!(!session.session_id.is_empty());

        // Session handle resource exists.
        assert!(orch.world.get_resource::<SessionHandle>().is_some());

        // VecEventSink resource exists.
        assert!(orch.world.get_resource::<VecEventSink>().is_some());
    }

    #[test]
    fn orchestrator_pipeline_no_adapters() {
        // A pipeline without adapters should fail on source detection when
        // running the full pipeline.
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);

        // Attempt source detection with no adapters registered.
        let result = orch.run_stage(CompilationStage::SourceDetection);
        assert!(result.is_err());
        if let Err(CompileError::SourceDetectionFailed(msg)) = result {
            assert!(msg.contains("no source adapters"));
        } else {
            panic!("expected SourceDetectionFailed");
        }
    }

    #[test]
    fn orchestrator_stage_dispatch_rejects_unmet_prereqs() {
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);

        // Source detection without adapters should fail.
        assert!(orch.run_stage(CompilationStage::SourceDetection).is_err());

        // Graph construction without previous stages should fail.
        assert!(orch.run_stage(CompilationStage::GraphConstruction).is_err());

        // Search without previous stages should fail.
        assert!(orch
            .run_stage(CompilationStage::EvolutionarySearch)
            .is_err());

        // Legalization without previous stages should fail.
        assert!(orch.run_stage(CompilationStage::Legalization).is_err());

        // Kernel generation requires the legalized SpatialIR graph.
        assert!(orch.run_stage(CompilationStage::KernelGeneration).is_err());

        // Receipt build cannot bypass certification.
        assert!(orch.run_stage(CompilationStage::ReceiptBuild).is_err());

        // The session remains initialized because no certified artifact exists.
        let session = orch
            .world
            .component::<CompilationSession>(orch.session)
            .expect("session component");
        assert!(matches!(session.status, SessionStatus::Initialized));
    }

    #[test]
    fn session_entity_helper_resolves_when_handle_present() {
        let config = CompileConfig::default();
        let orch = CompilationOrchestrator::new(config);
        let resolved = session_entity(&orch.world).expect("session entity helper");
        assert_eq!(resolved, orch.session);
    }

    #[test]
    fn read_session_config_returns_attached_config() {
        let config = CompileConfig {
            max_candidates: 42,
            ..CompileConfig::default()
        };
        let orch = CompilationOrchestrator::new(config.clone());
        let read_back = read_session_config(&orch.world, orch.session).expect("read config");
        assert_eq!(read_back.max_candidates, 42);
    }

    #[test]
    fn read_session_config_fails_when_entity_missing_component() {
        let config = CompileConfig::default();
        let mut orch = CompilationOrchestrator::new(config);
        // Spawn an unrelated entity; it has no CompilationSession.
        let stray = orch
            .world
            .spawn(EntityKind::Model, None)
            .expect("spawn stray");
        let err = read_session_config(&orch.world, stray.entity)
            .expect_err("read on entity without CompilationSession must fail");
        assert!(matches!(err, CompileError::PolicyViolation(_)));
    }
}
