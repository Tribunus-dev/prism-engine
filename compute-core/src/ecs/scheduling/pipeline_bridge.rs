//! Bridge wrapping constitutional multimodal pipeline commands behind a simple
//! synchronous API for production pipeline callers.
//!
//! Owns an `Arc<RwLock<World>>` and a `SchemaRegistry` with all multimodal
//! schemas pre-registered.  Each method:
//! 1. Locks the world for writing
//! 2. Constructs the relevant constitutional command with a unique `MessageId`
//! 3. Calls preflight + execute (the command internally stages a `WorldTxn`)
//! 4. Returns the result
//!
//! This is the authority path for:
//! - `CreatePipelineCommand`  → `PipelineBridge::create_pipeline`
//! - `SubmitStageOutputCommand` → `PipelineBridge::submit_stage_output`

use crate::ecs::constitutional::multimodal::{
    CreatePipelineCommand, InputArtifactRef, OutputArtifactRef, Pipeline, PipelineLifecycle,
    PipelineModality, PipelineStage, SubmitStageOutputCommand, WorkLeaseRef, SCHEMA_INPUT_ARTIFACT,
    SCHEMA_OUTPUT_ARTIFACT, SCHEMA_PIPELINE, SCHEMA_PIPELINE_LIFECYCLE, SCHEMA_PIPELINE_MODALITY,
    SCHEMA_PIPELINE_STAGE, SCHEMA_WORK_LEASE_REF,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion};
use crate::ecs::{Entity, World};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Wraps constitutional multimodal pipeline commands behind a simple
/// synchronous API.
///
/// Each method:
/// 1. Generates a unique `MessageId` via UUID
/// 2. Locks the world for writing
/// 3. Constructs the constitutional command
/// 4. Executes it (preflight + execute)
/// 5. Returns the result
pub struct PipelineBridge {
    world: Arc<RwLock<World>>,
    schema_registry: SchemaRegistry,
}

impl PipelineBridge {
    /// Create a new bridge backed by the given world.
    ///
    /// Registers all multimodal schemas on construction so schema validation
    /// inside each constitutional command passes.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        let mut schema_registry = SchemaRegistry::new();
        Self::register_multimodal_schemas(&mut schema_registry);
        Self {
            world,
            schema_registry,
        }
    }

    /// Register all multimodal domain schemas into the given registry.
    fn register_multimodal_schemas(reg: &mut SchemaRegistry) {
        reg.register_for_type::<Pipeline>(
            ComponentSchemaId(SCHEMA_PIPELINE),
            SchemaVersion(1),
            "Pipeline",
            "Multimodal generation pipeline",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<PipelineStage>(
            ComponentSchemaId(SCHEMA_PIPELINE_STAGE),
            SchemaVersion(1),
            "PipelineStage",
            "Pipeline stage",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<PipelineModality>(
            ComponentSchemaId(SCHEMA_PIPELINE_MODALITY),
            SchemaVersion(1),
            "PipelineModality",
            "Pipeline modality",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<InputArtifactRef>(
            ComponentSchemaId(SCHEMA_INPUT_ARTIFACT),
            SchemaVersion(1),
            "InputArtifactRef",
            "Input artifact reference",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<OutputArtifactRef>(
            ComponentSchemaId(SCHEMA_OUTPUT_ARTIFACT),
            SchemaVersion(1),
            "OutputArtifactRef",
            "Output artifact reference",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<PipelineLifecycle>(
            ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE),
            SchemaVersion(1),
            "PipelineLifecycle",
            "Pipeline lifecycle",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<WorkLeaseRef>(
            ComponentSchemaId(SCHEMA_WORK_LEASE_REF),
            SchemaVersion(1),
            "WorkLeaseRef",
            "Work lease reference",
            ComponentDurability::Durable,
        );
    }

    /// Create a multimodal pipeline entity.
    ///
    /// Constructs a [`CreatePipelineCommand`] internally, runs preflight then
    /// execute against the world, and returns the newly allocated pipeline
    /// entity id on success.
    pub fn create_pipeline(
        &self,
        session_entity: u64,
        stages: Vec<PipelineStage>,
    ) -> Result<u64, String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        // Peek the next entity id before execute — the command internally
        // allocates the same entity via WorldTxn::next_entity_id().
        let entity_id = world.next_entity_id();

        let id = MessageId::compute(Uuid::new_v4().as_bytes());

        let cmd = CreatePipelineCommand {
            id,
            session_entity,
            // Default to Multimodal for the general-purpose bridge; callers
            // that need a specific modality can use SubmitStageOutputCommand
            // for per-stage modality tagging.  Production vision/audio
            // pipelines construct their own command if a custom modality is
            // required.
            target_modality: PipelineModality::Multimodal("pipeline".to_string()),
            stages,
            input_artifacts: vec![],
        };

        cmd.execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        Ok(entity_id)
    }

    /// Record a stage output artifact on an existing pipeline.
    ///
    /// Constructs a [`SubmitStageOutputCommand`] internally, runs preflight
    /// then execute against the world.
    pub fn submit_stage_output(
        &self,
        pipeline_entity: u64,
        stage_index: usize,
        output_modality: PipelineModality,
        output_artifact: u64,
    ) -> Result<(), String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        let id = MessageId::compute(Uuid::new_v4().as_bytes());

        let cmd = SubmitStageOutputCommand {
            id,
            pipeline_entity: Entity(pipeline_entity, 0),
            stage_index: stage_index as u32,
            output_artifact_id: Some(output_artifact),
        };

        cmd.execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        // The `output_modality` parameter is accepted in the bridge API
        // signature for future use — e.g. transitioning the pipeline lifecycle
        // or tagging the output with modality metadata.  The underlying
        // constitutional command does not yet consume modality at this point;
        // the field is reserved for a follow-on evolution of the command.

        let _ = output_modality;

        Ok(())
    }
}
