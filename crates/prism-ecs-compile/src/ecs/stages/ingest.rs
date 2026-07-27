//! Source ingestion and graph construction stages.
//!
//! These two stages are paired because they are the canonical front of the
//! pipeline: detect a source, construct a spatial graph. Each stage reads
//! from world resources and writes one component onto the session entity.

use prism_ecs_core::world::World;

use crate::ecs::components::{
    CompilationSession, SessionStatus, SourceModel, SpatialGraphComponent, TensorCollection,
};
use crate::ecs::orchestrator::session_entity;
use crate::ecs::resources::CurrentSource;
use crate::graph::CanonicalGraphBuilder;
use crate::CompileError;

/// Run the **source detection** stage.
///
/// Iterates registered format adapters to detect and ingress the source model.
/// On success adds [`SourceModel`] and [`TensorCollection`] components to the
/// session entity, stores the [`CanonicalSource`] as [`CurrentSource`], and
/// transitions session status to [`SessionStatus::Ingested`].
pub fn system_detect_source(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;
    let canonical_source = world.get_extension::<CurrentSource>().ok_or_else(|| {
        CompileError::SourceDetectionFailed(
            "no source adapters or canonical source provided".into(),
        )
    })?;
    let identity = canonical_source.0.identity.clone();

    // Extract metadata for components
    let architecture = identity.architecture.clone();
    let catalog = canonical_source.0.catalog.clone();
    let count = catalog.tensors.len();

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::Ingested;
    }

    // Attach components
    world
        .insert_component(
            session,
            SourceModel {
                identity: identity.clone(),
                architecture: architecture.clone(),
            },
        )
        .map_err(|e| CompileError::SourceDetectionFailed(e.to_string()))?;

    world
        .insert_component(
            session,
            TensorCollection {
                catalog: catalog.clone(),
                count,
            },
        )
        .map_err(|e| CompileError::SourceDetectionFailed(e.to_string()))?;

    Ok(())
}

/// Run the **graph construction** stage.
///
/// Reads the [`CurrentSource`] extension and produces a [`SpatialGraph`].
/// Adds [`SpatialGraphComponent`] to the session entity.
pub fn system_build_graph(world: &mut World) -> Result<(), CompileError> {
    let session = session_entity(world)?;

    let source = world
        .get_extension::<CurrentSource>()
        .ok_or_else(|| CompileError::GraphBuildFailed("no current source resource".into()))?;

    let result = CanonicalGraphBuilder::build(&source.0)
        .map_err(|e| CompileError::GraphBuildFailed(e.to_string()))?;

    // Update session status
    if let Ok(status) = world.component_mut::<CompilationSession>(session) {
        status.status = SessionStatus::GraphBuilt;
    }

    world
        .insert_component(
            session,
            SpatialGraphComponent {
                graph: result.graph,
                graph_digest: result.graph_digest,
                architecture: result.architecture,
            },
        )
        .map_err(|e| CompileError::GraphBuildFailed(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompileConfig;
    use crate::ecs::orchestrator::CompilationOrchestrator;
    use crate::ecs::resources::SessionHandle;
    use prism_ecs_core::entity::EntityKind;
    use prism_ecs_core::identity::SourceFormat;
    use prism_ecs_core::world::World;
    use prism_ecs_source::{SourceIdentity, TensorCatalog, TensorDataProvider};

    struct EmptyProvider;
    impl TensorDataProvider for EmptyProvider {
        fn read_tensor(
            &self,
            _tensor: &prism_ecs_source::TensorDescriptor,
        ) -> Result<Vec<u8>, prism_ecs_source::SourceError> {
            Ok(Vec::new())
        }
    }

    fn install_session(world: &mut World, source: prism_ecs_source::CanonicalSource) {
        let spawned = world
            .spawn(EntityKind::Session, None)
            .expect("spawn session")
            .entity;
        world
            .insert_component(
                spawned,
                CompilationSession {
                    config: CompileConfig::default(),
                    status: SessionStatus::Initialized,
                    session_id: "test".into(),
                },
            )
            .expect("insert session");
        world.set_extension(CurrentSource(source));
        world
            .insert_resource(SessionHandle(spawned))
            .expect("session handle");
    }

    #[test]
    fn detect_source_attaches_source_model_and_tensor_collection() {
        let mut world = World::new();
        let tensors = vec![prism_ecs_source::TensorDescriptor {
            name: "weight".into(),
            shape: vec![2, 2],
            dtype: "f16".into(),
            byte_offset: 0,
            byte_length: 8,
            element_size: 2,
            original_dtype: "F16".into(),
            data_offset: None,
            data_size_bytes: 8,
            layout: "row-major".into(),
        }];
        let catalog = TensorCatalog {
            tensors,
            ..Default::default()
        };
        let source = prism_ecs_source::CanonicalSource {
            identity: SourceIdentity {
                format: SourceFormat::SafeTensors,
                source_digest: "abc".into(),
                architecture: "llama".into(),
                model_family: "test".into(),
            },
            catalog,
            provider: Some(std::sync::Arc::new(EmptyProvider)),
            capabilities: Default::default(),
        };
        install_session(&mut world, source);

        system_detect_source(&mut world).expect("detect ok");

        // Session status advanced to Ingested.
        let session_comp = world
            .component::<CompilationSession>(world.get_resource::<SessionHandle>().unwrap().0)
            .unwrap();
        assert!(matches!(session_comp.status, SessionStatus::Ingested));
    }

    #[test]
    fn detect_source_fails_without_current_source() {
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        let result = system_detect_source(&mut orch.world);
        assert!(matches!(result, Err(CompileError::SourceDetectionFailed(_))));
    }

    #[test]
    fn build_graph_fails_without_current_source() {
        let mut orch = CompilationOrchestrator::new(CompileConfig::default());
        let result = system_build_graph(&mut orch.world);
        assert!(matches!(result, Err(CompileError::GraphBuildFailed(_))));
    }
}
