use crate::compilation_entity::{CompilationEntity, CompilationStatus};
use crate::ecs::{
    CImageArtifact, CompilationSession, KernelCollection, LegalizedPlan, SearchStateComponent,
    SessionHandle, SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
};
use crate::CompileError;
use prism_ecs_core::component::Component;
use prism_ecs_core::world::World;

/// Mirror the session lifecycle into the legacy compilation entity so the two
/// ECS representations cannot silently drift.
pub fn sync_compilation_entity(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    let session_status = world
        .component::<CompilationSession>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?
        .status
        .clone();
    let mapped = match session_status {
        SessionStatus::Complete => CompilationStatus::Complete,
        SessionStatus::Failed(_) => CompilationStatus::Failed,
        SessionStatus::Initialized => CompilationStatus::Created,
        _ => CompilationStatus::Running,
    };
    world
        .component_mut::<CompilationEntity>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(format!("compilation entity missing: {e}")))?
        .status = mapped;
    Ok(())
}

fn session_mut(world: &mut World) -> Result<crate::ecs::SessionHandle, CompileError> {
    world
        .get_resource::<SessionHandle>()
        .copied()
        .ok_or_else(|| CompileError::CompilationFailed("session handle missing".into()))
}

fn require_status(world: &mut World, expected: SessionStatus) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    let actual = world
        .component::<CompilationSession>(handle.0)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?
        .status
        .clone();
    if std::mem::discriminant(&actual) != std::mem::discriminant(&expected) {
        return Err(CompileError::PolicyViolation(format!(
            "invalid ECS stage transition: expected {expected:?}, found {actual:?}"
        )));
    }
    Ok(())
}

fn require_component<T: Component>(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    world
        .component::<T>(handle.0)
        .map(|_| ())
        .map_err(|e| CompileError::PolicyViolation(format!("stage output missing: {e}")))
}

pub fn system_transition_ingest_to_plan(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::Ingested)?;
    require_component::<SourceModel>(world)?;
    require_component::<TensorCollection>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_plan_to_evaluate(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::GraphBuilt)?;
    require_component::<SpatialGraphComponent>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_evaluate_to_legalize(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::SearchComplete)?;
    require_component::<SearchStateComponent>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_legalize_to_compile(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::Legalized)?;
    require_component::<LegalizedPlan>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_compile_to_emit(world: &mut World) -> Result<(), CompileError> {
    require_status(world, SessionStatus::KernelsGenerated)?;
    require_component::<KernelCollection>(world)?;
    sync_compilation_entity(world)
}

pub fn system_transition_emit_to_complete(world: &mut World) -> Result<(), CompileError> {
    let handle = session_mut(world)?;
    {
        let status = world
            .component::<CompilationSession>(handle.0)
            .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
        if !matches!(status.status, SessionStatus::Certified) {
            return Err(CompileError::PolicyViolation(
                "completion requires a certified CImage".into(),
            ));
        }
    }
    require_component::<CImageArtifact>(world)?;
    {
        let status = world
            .component_mut::<CompilationSession>(handle.0)
            .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
        status.status = SessionStatus::Complete;
    }
    sync_compilation_entity(world)
}
