//! Receipt build stage.
//!
//! Collects events and builds the forensic receipt, attaching it to the
//! session entity as a [`CompilationReceipt`] component.

use sha2::Digest;

use prism_ecs_core::identity::{CompilerIdentity, SourceFormat};
use prism_ecs_core::world::World;
use prism_ecs_source::SourceIdentity;

use crate::ecs::components::{
    CImageArtifact, CompilationReceipt, CompilationSession, KernelCollection, SearchStateComponent,
    SessionStatus, SourceModel, SpatialGraphComponent,
};
use crate::ecs::orchestrator::session_entity;
use crate::ecs::resources::VecEventSink;
use crate::forensic::build_forensic_receipt;
use crate::{
    CompilationEvent, CompileError, CompileReceipt, CompileStatus,
};

/// Run the **receipt build** stage.
pub fn system_build_receipt(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let session_state = world
        .component::<CompilationSession>(session)
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;
    if !matches!(
        session_state.status,
        SessionStatus::Certified | SessionStatus::Complete
    ) {
        return Err(CompileError::CompilationFailed(
            "receipt build requires a certified CImage artifact".into(),
        ));
    }
    world
        .component::<CImageArtifact>(session)
        .map_err(|e| CompileError::CompilationFailed(format!("artifact missing: {e}")))?;
    let events: Vec<CompilationEvent> = world
        .get_resource::<VecEventSink>()
        .map(|s| s.events())
        .unwrap_or_default();

    let mut receipt = CompileReceipt {
        receipt_id: String::new(),
        request_id: uuid::Uuid::new_v4(),
        compiler_identity: CompilerIdentity {
            name: "ecs-orchestrator".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_hash: option_env!("PRISM_BUILD_HASH").map(str::to_owned),
            build_timestamp: option_env!("PRISM_BUILD_TIMESTAMP").map(str::to_owned),
        },
        source_identity: world
            .component::<SourceModel>(session)
            .ok()
            .map(|m| m.identity.clone())
            .unwrap_or_else(|| SourceIdentity {
                format: SourceFormat::Raw,
                source_digest: String::new(),
                architecture: String::new(),
                model_family: String::new(),
            }),
        started_at: chrono::Utc::now(),
        completed_at: chrono::Utc::now(),
        duration_ms: 0,
        stages: Vec::new(),
        candidate_count: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.candidates_evaluated as u32)
            .unwrap_or(0),
        generations: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.generations_completed as u32)
            .unwrap_or(0),
        output_digest: world
            .component::<CImageArtifact>(session)
            .ok()
            .map(|a| a.digest.clone())
            .unwrap_or_default(),
        source_digest: Some(
            world
                .component::<SourceModel>(session)
                .ok()
                .map(|m| m.identity.source_digest.clone())
                .unwrap_or_default(),
        ),
        graph_digest: Some(
            world
                .component::<SpatialGraphComponent>(session)
                .ok()
                .map(|g| g.graph_digest.clone())
                .unwrap_or_default(),
        ),
        search_trace_digest: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|s| s.trace.trace_digest.clone())
            .filter(|d| !d.is_empty()),
        kernel_manifest_digest: None,
        events_digest: Some(String::new()),
        legalization_mode: Some("target_default".into()),
        selection_receipt: world
            .component::<SearchStateComponent>(session)
            .ok()
            .map(|search| search.selection_receipt.clone()),
        uop_tuning_receipt: world
            .component::<KernelCollection>(session)
            .ok()
            .and_then(|kernels| kernels.uop_tuning_receipt.clone()),
        error: None,
        status: CompileStatus::Completed,
        finished_at: chrono::Utc::now(),
        output_path: std::path::PathBuf::new(),
        schema_version: "1.0".into(),
    };

    // Build and retain the forensic receipt on the session entity.
    if !events.is_empty() {
        let bytes = build_forensic_receipt(&events);
        receipt.events_digest = Some(hex::encode(sha2::Sha256::digest(&bytes)));
    }

    receipt.receipt_id = hex::encode(sha2::Sha256::digest(
        format!(
            "{}:{}",
            receipt.output_digest,
            receipt.search_trace_digest.clone().unwrap_or_default()
        )
        .as_bytes(),
    ));
    let receipt_id = receipt.receipt_id.clone();
    let cimage_digest = receipt.output_digest.clone();

    world
        .insert_component(session, CompilationReceipt(receipt))
        .map_err(|e| CompileError::CompilationFailed(e.to_string()))?;

    // Close the evidence chain for every admitted deployment candidate with
    // the emitted artifact and the compilation receipt that certified it.
    if let Ok(search) = world.component_mut::<SearchStateComponent>(session) {
        for candidate in search.deployment_archive.candidates.values_mut() {
            candidate.evidence.cimage_digest = Some(cimage_digest.clone());
            candidate.evidence.receipt_ids.push(receipt_id.clone());
        }
    }

    // Update session status to Complete.
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Complete;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::orchestrator::CompilationOrchestrator;
    use crate::CompileConfig;

    #[test]
    fn receipt_build_rejects_session_not_certified() {
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        // Session is Initialized, not Certified — receipt build must fail.
        let result = system_build_receipt(&mut orch.world);
        assert!(matches!(result, Err(CompileError::CompilationFailed(_))));
    }
}
