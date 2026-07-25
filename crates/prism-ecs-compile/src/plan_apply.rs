//! Plan-apply: world-mutating constitutional system for compile requests.
//!
//! This module owns the canonical authority for applying a compile request
//! through the constitutional ECS world. It does not own source ingestion,
//! IR construction, or compile orchestration; those live in [`crate::compiler`]
//! (the orchestrator) and [`crate::ir_build`] (pure IR construction, when
//! that split is completed).
//!
//! The migration out of `compiler.rs` is the constitutional fix called out in
//! `references/module-discipline.md` §Concrete decomposition patterns for
//! Prism: the compile path must not mutate the world during "code
//! generation." The world-mutating setup that used to live in
//! `compiler::compile_source_ecs` now lives here as a typed constitutional
//! system that stages a session entity, registers resources, and runs the
//! existing pipeline systems.
//!
//! The orchestrator at [`crate::compiler::compile_path_with_backend`] calls
//! [`crate::compiler::compile_source`] which calls into the systems but does
//! not itself touch the world. World-mutation is isolated here.

use prism_ecs_core::entity::EntityKind;
use prism_ecs_core::world::World;
use prism_ecs_ir::evolution::EvolutionRuntime;
use prism_ecs_source::{CanonicalSource, CanonicalSourceAdapter};

use crate::compilation_entity::CompilationEntity;
use crate::compilation_systems::*;
use crate::ecs::{
    system_build_graph, system_build_receipt, system_certify, system_detect_source,
    system_emit_cimage, system_generate_kernels, system_legalize, system_run_search,
    CompilationReceipt, CompilationSession, CurrentSource, SessionHandle, SessionStatus,
    SourceAdapterList,
};
use crate::{CompileConfig, CompileError, CompileResult, CompileStatus, VecEventSink};

/// Run the full compilation pipeline using ECS systems.
///
/// This function creates a compilation entity and runs the complete schedule
/// of systems to perform the compilation. It is the only entry point in
/// `prism_ecs_compile` that directly mutates the canonical world during the
/// compile path; every other public function either reads the world
/// (e.g. `compile_ecs_op_to_xdna_cimage`) or orchestrates without touching it
/// (e.g. `compile_source`).
pub fn compile_source_ecs(
    world: &mut World,
    source: CanonicalSource,
    config: CompileConfig,
) -> Result<CompileResult, CompileError> {
    // Create a session entity for this compilation
    let session_entity = world
        .spawn(EntityKind::Session, None)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?
        .entity;

    // Initialize the session with basic configuration
    world.insert_component(
        session_entity,
        CompilationSession {
            config: config.clone(),
            status: SessionStatus::Initialized,
            session_id: uuid::Uuid::new_v4().to_string(),
        },
    )?;

    // Initialize the compilation entity
    world.insert_component(session_entity, CompilationEntity::new(config.clone()))?;

    // Set up world resources
    world
        .insert_resource(SessionHandle(session_entity))
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
    world
        .insert_resource(EvolutionRuntime::global())
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;

    // Set up source adapters
    let adapters = vec![
        Box::new(prism_ecs_source::gguf_adapter::GgufAdapter) as Box<dyn CanonicalSourceAdapter>,
        Box::new(prism_ecs_source::onnx_adapter::OnnxAdapter) as Box<dyn CanonicalSourceAdapter>,
        Box::new(prism_ecs_source::safetensors_adapter::SafetensorsAdapter)
            as Box<dyn CanonicalSourceAdapter>,
        Box::new(prism_ecs_source::mlx_adapter::MlxAdapter) as Box<dyn CanonicalSourceAdapter>,
    ];
    world
        .insert_resource(SourceAdapterList(adapters))
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;

    // Set up event sink
    let event_sink = VecEventSink::new();
    world
        .insert_resource(event_sink)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;

    // Store the canonical source as a world extension
    world.set_extension(CurrentSource(source));

    // Run the compilation pipeline through systems

    // 1. Source detection and ingestion
    system_detect_source(world)?;
    system_transition_ingest_to_plan(world)?;

    // 2. Graph construction
    system_build_graph(world)?;
    system_transition_plan_to_evaluate(world)?;

    // 3. Evolutionary search (if enabled)
    if config.enable_search {
        system_run_search(world)?;
    } else if let Ok(session) = world.component_mut::<CompilationSession>(session_entity) {
        session.status = SessionStatus::SearchComplete;
    }
    system_transition_evaluate_to_legalize(world)?;

    // 3.5 Build a constitutional `QuantizationResultComponent` from
    // the search's `format_plan` (or the default policy). The emit
    // stage requires this component; legacy code paths that did not
    // produce one were emitting source bytes under the source dtype
    // regardless of what the search selected.
    system_build_quantization_result(world)?;

    // 4. Legalization
    system_legalize(world)?;
    system_transition_legalize_to_compile(world)?;

    // 5. Kernel generation
    system_generate_kernels(world)?;
    system_transition_compile_to_emit(world)?;

    // 6. CImage emission
    system_emit_cimage(world)?;
    // 7. Reopen and certify the artifact before any receipt can claim
    // completion. This keeps the direct ECS entry point aligned with the
    // orchestrator path.
    system_certify(world)?;
    system_transition_emit_to_complete(world)?;

    // 8. Receipt building
    system_build_receipt(world)?;

    // Extract the final result
    let session_status = world.component::<CompilationSession>(session_entity)?;
    match &session_status.status {
        SessionStatus::Complete => {
            let receipt = world
                .component::<CompilationReceipt>(session_entity)
                .map_err(|e| {
                    CompileError::CompilationFailed(format!("receipt component missing: {e}"))
                })?
                .0
                .clone();
            Ok(CompileResult {
                status: CompileStatus::Completed,
                request_id: receipt.request_id,
                events: world
                    .get_resource::<VecEventSink>()
                    .map(|s| s.events())
                    .unwrap_or_default(),
                output_digest: receipt.output_digest.clone(),
                output_path: receipt.output_path.clone(),
                receipt,
            })
        }
        SessionStatus::Failed(error) => Err(CompileError::CompilationFailed(error.clone())),
        _ => Err(CompileError::CompilationFailed(
            "pipeline did not complete".into(),
        )),
    }
}
