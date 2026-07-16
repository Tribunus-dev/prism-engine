//! PipelineBridge — thin bridge from ECS inference systems to the
//! constitutional pipeline command layer.
//!
//! Stores session context and a pre-wired [`SchemaRegistry`] so that
//! inference systems can call [`CreatePipelineCommand`] before inference
//! and [`SubmitStageOutputCommand`] after, without importing constitutional
//! types directly.

use crate::command::DomainEvent;
use crate::multimodal::{
    CreatePipelineCommand, PipelineModality, PipelineStage, SubmitStageOutputCommand,
    SCHEMA_INPUT_ARTIFACT, SCHEMA_OUTPUT_ARTIFACT, SCHEMA_PIPELINE, SCHEMA_PIPELINE_LIFECYCLE,
    SCHEMA_PIPELINE_MODALITY, SCHEMA_PIPELINE_STAGE, SCHEMA_WORK_LEASE_REF,
};
use crate::schema::{ComponentDurability, SchemaEntry, SchemaRegistry};
use crate::types::{ComponentSchemaId, MessageId, SchemaVersion};
use crate::world_txn::CommittedEpoch;
use prism_ecs_core::{Entity, World};

// ---------------------------------------------------------------------------
// PipelineBridge
// ---------------------------------------------------------------------------

/// Bridge between ECS runtime inference systems and the constitutional
/// pipeline command layer.
///
/// Pre-configured with a session entity and a [`SchemaRegistry`] so that
/// [`create_pipeline`](PipelineBridge::create_pipeline) and
/// [`submit_stage_output`](PipelineBridge::submit_stage_output) can
/// construct and execute the corresponding constitutional commands
/// from within a system's `run()` method.
///
/// The bridge stores the pipeline entity created by the most recent call
/// to `create_pipeline`, which is then used by `submit_stage_output`.
#[derive(Clone)]
pub struct PipelineBridge {
    /// The session entity this pipeline belongs to (legacy u64 format).
    session_entity: u64,
    /// Schema registry with all multimodal types registered.
    schema_registry: SchemaRegistry,
    /// The pipeline entity created by the most recent `create_pipeline`.
    pipeline_entity: Option<Entity>,
}

impl PipelineBridge {
    /// Create a new pipeline bridge for the given session.
    ///
    /// The schema registry is pre-populated with all multimodal component
    /// types so that callers do not need to manage schema registration.
    pub fn new(session_entity: u64) -> Self {
        Self {
            session_entity,
            schema_registry: Self::make_multimodal_registry(),
            pipeline_entity: None,
        }
    }

    /// Build a `SchemaRegistry` with all multimodal types registered.
    fn make_multimodal_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE),
            version: SchemaVersion(1),
            type_name: "Pipeline".into(),
            description: "Multimodal generation pipeline".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<
                crate::multimodal::Pipeline,
            >()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE_STAGE),
            version: SchemaVersion(1),
            type_name: "PipelineStage".into(),
            description: "Pipeline stage".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<PipelineStage>()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE_MODALITY),
            version: SchemaVersion(1),
            type_name: "PipelineModality".into(),
            description: "Pipeline modality".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<PipelineModality>()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_INPUT_ARTIFACT),
            version: SchemaVersion(1),
            type_name: "InputArtifactRef".into(),
            description: "Input artifact reference".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<
                crate::multimodal::InputArtifactRef,
            >()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_OUTPUT_ARTIFACT),
            version: SchemaVersion(1),
            type_name: "OutputArtifactRef".into(),
            description: "Output artifact reference".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<
                crate::multimodal::OutputArtifactRef,
            >()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE),
            version: SchemaVersion(1),
            type_name: "PipelineLifecycle".into(),
            description: "Pipeline lifecycle".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<
                crate::multimodal::PipelineLifecycle,
            >()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_WORK_LEASE_REF),
            version: SchemaVersion(1),
            type_name: "WorkLeaseRef".into(),
            description: "Work lease reference".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<
                crate::multimodal::WorkLeaseRef,
            >()),
        });
        reg
    }

    /// Create a multimodal pipeline and record the pipeline entity.
    ///
    /// Constructs and executes a [`CreatePipelineCommand`] with the stored
    /// session entity, the given modality, and the provided stages.
    /// The resulting pipeline entity is stored for later use by
    /// [`submit_stage_output`](PipelineBridge::submit_stage_output).
    ///
    /// Returns `Ok((epoch, event))` on success, or the
    /// [`MultimodalError`](crate::multimodal::MultimodalError)
    /// on failure.
    pub fn create_pipeline(
        &mut self,
        world: &mut World,
        target_modality: PipelineModality,
        stages: Vec<PipelineStage>,
    ) -> Result<
        (CommittedEpoch, DomainEvent),
        crate::multimodal::MultimodalError,
    > {
        let cmd = CreatePipelineCommand {
            id: MessageId::compute(
                format!(
                    "pipeline:session={}:{:?}",
                    self.session_entity, target_modality
                )
                .as_bytes(),
            ),
            session_entity: self.session_entity,
            target_modality,
            stages,
            input_artifacts: vec![],
        };
        let (epoch, event) = cmd.execute(world, &self.schema_registry)?;

        // Extract the pipeline entity from the event payload.
        if let Some(pipeline_id) = event.payload.get("pipeline_id").and_then(|v| v.as_u64()) {
            self.pipeline_entity = Some(Entity::new(pipeline_id, 0));
        }

        Ok((epoch, event))
    }

    /// Submit stage output for the most recently created pipeline.
    ///
    /// Constructs and executes a [`SubmitStageOutputCommand`] for the
    /// pipeline entity that was created by the preceding
    /// [`create_pipeline`](PipelineBridge::create_pipeline) call.
    ///
    /// Returns `Ok((epoch, event))` on success, or the
    /// [`MultimodalError`](crate::multimodal::MultimodalError)
    /// on failure. Returns `Err` with a message string if no pipeline
    /// entity has been created yet.
    pub fn submit_stage_output(
        &self,
        world: &mut World,
        stage_index: u32,
        output_artifact_id: Option<u64>,
    ) -> Result<(CommittedEpoch, DomainEvent), String> {
        let pipeline_entity = self.pipeline_entity.ok_or_else(|| {
            "PipelineBridge: no pipeline entity; call create_pipeline first".to_string()
        })?;

        let cmd = SubmitStageOutputCommand {
            id: MessageId::compute(
                format!(
                    "stage_output:pipeline={}:stage={}",
                    pipeline_entity.id(),
                    stage_index
                )
                .as_bytes(),
            ),
            pipeline_entity,
            stage_index,
            output_artifact_id,
        };
        cmd.execute(world, &self.schema_registry)
            .map_err(|e| format!("PipelineBridge::submit_stage_output: {e}"))
    }

    /// The session entity this bridge was created for.
    pub fn session_entity(&self) -> u64 {
        self.session_entity
    }

    /// The pipeline entity created by [`create_pipeline`], if any.
    pub fn pipeline_entity(&self) -> Option<Entity> {
        self.pipeline_entity
    }
}

impl std::fmt::Debug for PipelineBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineBridge")
            .field("session_entity", &self.session_entity)
            .field("pipeline_entity", &self.pipeline_entity)
            .finish()
    }
}
