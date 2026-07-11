use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::lifecycle::{LifecycleError, SessionLifecycle};
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{ClassifiedComponent, DurableClass, DurableComponent};
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn};
use crate::ecs::{CompWorld, EntityKind};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs ──────────────────────────────────────────────────
// Session: 13-17, Work: 18-23, Execution: 24-29, Agent: 30-38,
// Compilation: 39-46, Multimodal: 47-53

pub const SCHEMA_PIPELINE: u64 = 47;
pub const SCHEMA_PIPELINE_STAGE: u64 = 48;
pub const SCHEMA_PIPELINE_MODALITY: u64 = 49;
pub const SCHEMA_INPUT_ARTIFACT: u64 = 50;
pub const SCHEMA_OUTPUT_ARTIFACT: u64 = 51;
pub const SCHEMA_PIPELINE_LIFECYCLE: u64 = 52;
pub const SCHEMA_WORK_LEASE_REF: u64 = 53;

// ── Pipeline Component ────────────────────────────────────────────────────

/// One multimodal generation pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pipeline {
    pub pipeline_id: u64,
    pub session_entity: u64,
    pub target_modality: String,
    pub created_at: Timestamp,
}

impl crate::ecs::Component for Pipeline {}

impl ClassifiedComponent for Pipeline {
    type Class = DurableClass;
}
impl DurableComponent for Pipeline {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 47,
        version: 1,
    };
}

/// A stage within a pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineStage {
    pub stage_index: u32,
    pub stage_type: String,
    pub model_entity: u64,
    pub input_transform: String,
    pub output_transform: String,
}

impl crate::ecs::Component for PipelineStage {}

impl ClassifiedComponent for PipelineStage {
    type Class = DurableClass;
}
impl DurableComponent for PipelineStage {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 48,
        version: 1,
    };
}

/// What modality this pipeline handles.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineModality {
    Vision,
    Audio,
    Tts,
    Diffusion,
    Image,
    Video,
    Multimodal(String),
}

impl crate::ecs::Component for PipelineModality {}

impl ClassifiedComponent for PipelineModality {
    type Class = DurableClass;
}
impl DurableComponent for PipelineModality {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 49,
        version: 1,
    };
}

/// Artifact used as input to a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputArtifactRef {
    pub artifact_id: u64,
    pub role: String,
    pub loaded: bool,
}

impl crate::ecs::Component for InputArtifactRef {}

impl ClassifiedComponent for InputArtifactRef {
    type Class = DurableClass;
}
impl DurableComponent for InputArtifactRef {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 50,
        version: 1,
    };
}

/// Artifact produced by a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputArtifactRef {
    pub artifact_id: Option<u64>,
    pub stage_index: u32,
    pub mime_type: String,
    pub size_bytes: u64,
}

impl crate::ecs::Component for OutputArtifactRef {}

impl ClassifiedComponent for OutputArtifactRef {
    type Class = DurableClass;
}
impl DurableComponent for OutputArtifactRef {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 51,
        version: 1,
    };
}

/// Lifecycle for a multimodal pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineLifecycle {
    Created,
    Assembled,
    Executing,
    Paused,
    Cancelled,
    Completed,
    Failed,
}

impl crate::ecs::Component for PipelineLifecycle {}

impl ClassifiedComponent for PipelineLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for PipelineLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 52,
        version: 1,
    };
}

impl PipelineLifecycle {
    /// Returns true if this state is terminal (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Completed | Self::Failed)
    }

    /// Validate a lifecycle transition. Returns Ok(()) if allowed.
    pub fn can_transition_to(&self, target: Self) -> Result<(), LifecycleError> {
        let allowed = match (*self, target) {
            (Self::Created, Self::Assembled)
            | (Self::Assembled, Self::Executing)
            | (Self::Executing, Self::Completed)
            | (Self::Executing, Self::Paused)
            | (Self::Paused, Self::Executing)
            | (Self::Paused, Self::Cancelled)
            | (Self::Executing, Self::Cancelled)
            | (Self::Created, Self::Cancelled) => true,
            // Failure allowed from any non-terminal state.
            _ if target == Self::Failed && !self.is_terminal() => true,
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(LifecycleError::InvalidPipelineTransition {
                from: *self,
                to: target,
            })
        }
    }
}

/// Links a work lease to a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkLeaseRef {
    pub lease_id: u64,
    pub work_entity: u64,
    pub stage_index: u32,
}

impl crate::ecs::Component for WorkLeaseRef {}

impl ClassifiedComponent for WorkLeaseRef {
    type Class = DurableClass;
}
impl DurableComponent for WorkLeaseRef {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.multimodal",
        id: 53,
        version: 1,
    };
}

// ── CreatePipelineCommand ─────────────────────────────────────────────────

/// Command to create a multimodal pipeline with stage list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreatePipelineCommand {
    pub id: MessageId,
    pub session_entity: u64,
    pub target_modality: PipelineModality,
    pub stages: Vec<PipelineStage>,
    pub input_artifacts: Vec<InputArtifactRef>,
}

impl CreatePipelineCommand {
    /// Validate all schemas are registered for multimodal components.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        schema_registry
            .verify_type::<Pipeline>(ComponentSchemaId(SCHEMA_PIPELINE))
            .map_err(|e| format!("Pipeline schema: {e}"))?;
        schema_registry
            .verify_type::<PipelineStage>(ComponentSchemaId(SCHEMA_PIPELINE_STAGE))
            .map_err(|e| format!("PipelineStage schema: {e}"))?;
        schema_registry
            .verify_type::<PipelineModality>(ComponentSchemaId(SCHEMA_PIPELINE_MODALITY))
            .map_err(|e| format!("PipelineModality schema: {e}"))?;
        schema_registry
            .verify_type::<InputArtifactRef>(ComponentSchemaId(SCHEMA_INPUT_ARTIFACT))
            .map_err(|e| format!("InputArtifactRef schema: {e}"))?;
        schema_registry
            .verify_type::<OutputArtifactRef>(ComponentSchemaId(SCHEMA_OUTPUT_ARTIFACT))
            .map_err(|e| format!("OutputArtifactRef schema: {e}"))?;
        schema_registry
            .verify_type::<PipelineLifecycle>(ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE))
            .map_err(|e| format!("PipelineLifecycle schema: {e}"))?;
        schema_registry
            .verify_type::<WorkLeaseRef>(ComponentSchemaId(SCHEMA_WORK_LEASE_REF))
            .map_err(|e| format!("WorkLeaseRef schema: {e}"))?;
        Ok(())
    }

    /// Preflight: session exists and is Active.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), MultimodalError> {
        Self::validate_schemas(schema_registry).map_err(|e| MultimodalError::SchemaError(e))?;

        if self.stages.is_empty() {
            return Err(MultimodalError::NoStages);
        }

        // Validate session exists and is Active
        let session_entity = crate::ecs::CompEntity(self.session_entity);
        if !world.has_entity(session_entity) {
            return Err(MultimodalError::SessionNotFound(self.session_entity));
        }
        if world.entity_kind(session_entity) != Some(EntityKind::Session) {
            return Err(MultimodalError::SessionNotFound(self.session_entity));
        }
        if let Some(lifecycle) = world.get_component::<SessionLifecycle>(session_entity) {
            if *lifecycle != SessionLifecycle::Active {
                return Err(MultimodalError::SessionNotActive(self.session_entity));
            }
        } else {
            return Err(MultimodalError::SessionNotActive(self.session_entity));
        }

        // Validate stages reference valid model entities
        for stage in &self.stages {
            let model_entity = crate::ecs::CompEntity(stage.model_entity);
            if !world.has_entity(model_entity) {
                return Err(MultimodalError::ModelNotFound(stage.model_entity));
            }
        }

        Ok(())
    }

    /// Execute pipeline creation: validate, create pipeline entity with all
    /// components, commit atomically.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), MultimodalError> {
        // 0. Preflight
        self.preflight(world, schema_registry)?;

        // Save modality string before consuming self
        let modality_str = modality_to_string(&self.target_modality);

        // 1. Reserve entity ID for pipeline
        let pipeline_id = WorldTxn::next_entity_id(world);

        let mut txn = WorldTxn::new(world);

        // 2. Spawn pipeline entity
        txn.stage_spawn(pipeline_id, EntityKind::Pipeline);

        // 3. Attach pipeline metadata
        txn.put_durable(
            pipeline_id,
            Pipeline {
                pipeline_id,
                session_entity: self.session_entity,
                target_modality: modality_to_string(&self.target_modality),
                created_at: Timestamp::now(),
            },
        );

        // 4. Attach modality
        txn.put_durable(pipeline_id, self.target_modality);

        // 5. Attach lifecycle
        txn.put_durable(pipeline_id, PipelineLifecycle::Created);

        // 6. Attach stages as separate entities
        // Use sequential IDs starting after pipeline_id
        let stage_entities: Vec<u64> = (0..self.stages.len())
            .map(|i| {
                let stage_id = pipeline_id + 1 + i as u64;
                txn.stage_spawn(stage_id, EntityKind::Pipeline);
                stage_id
            })
            .collect();

        for (i, stage) in self.stages.into_iter().enumerate() {
            let stage_entity = stage_entities[i];
            txn.put_durable(stage_entity, stage);
        }

        // 7. Attach input artifacts
        for artifact in self.input_artifacts {
            txn.put_durable(pipeline_id, artifact);
        }

        // 8. Emit event
        let event = DomainEvent {
            id: self.id,
            kind: "pipeline_created".to_string(),
            entity_id: Some(EntityKindId(pipeline_id)),
            payload: serde_json::json!({
                "pipeline_id": pipeline_id,
                "session_entity": self.session_entity,
                "target_modality": modality_str,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(MultimodalError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── SubmitStageOutputCommand ──────────────────────────────────────────────

/// Command to record pipeline stage completion with an output artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitStageOutputCommand {
    pub id: MessageId,
    pub pipeline_entity: u64,
    pub stage_index: u32,
    pub output_artifact_id: Option<u64>,
}

impl SubmitStageOutputCommand {
    /// Preflight: pipeline exists, is Executing or Assembled.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), MultimodalError> {
        schema_registry
            .verify_type::<Pipeline>(ComponentSchemaId(SCHEMA_PIPELINE))
            .map_err(|e| MultimodalError::SchemaError(e))?;

        let entity = crate::ecs::CompEntity(self.pipeline_entity);
        if !world.has_entity(entity) {
            return Err(MultimodalError::PipelineNotFound(self.pipeline_entity));
        }
        if let Some(lifecycle) = world.get_component::<PipelineLifecycle>(entity) {
            match lifecycle {
                PipelineLifecycle::Executing | PipelineLifecycle::Assembled => {}
                _ => {
                    return Err(MultimodalError::PipelineNotExecuting(self.pipeline_entity));
                }
            }
        } else {
            return Err(MultimodalError::PipelineNotFound(self.pipeline_entity));
        }

        Ok(())
    }

    /// Execute: attach output artifact, potentially transition lifecycle.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), MultimodalError> {
        self.preflight(world, schema_registry)?;

        let mut txn = WorldTxn::new(world);

        txn.put_durable(
            self.pipeline_entity,
            OutputArtifactRef {
                artifact_id: self.output_artifact_id,
                stage_index: self.stage_index,
                mime_type: "application/octet-stream".to_string(),
                size_bytes: 0,
            },
        );

        let event = DomainEvent {
            id: self.id,
            kind: "stage_output_submitted".to_string(),
            entity_id: Some(EntityKindId(self.pipeline_entity)),
            payload: serde_json::json!({
                "pipeline_entity": self.pipeline_entity,
                "stage_index": self.stage_index,
                "output_artifact_id": self.output_artifact_id,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(MultimodalError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn modality_to_string(modality: &PipelineModality) -> String {
    match modality {
        PipelineModality::Vision => "vision".to_string(),
        PipelineModality::Audio => "audio".to_string(),
        PipelineModality::Tts => "tts".to_string(),
        PipelineModality::Diffusion => "diffusion".to_string(),
        PipelineModality::Image => "image".to_string(),
        PipelineModality::Video => "video".to_string(),
        PipelineModality::Multimodal(custom) => custom.clone(),
    }
}

/// Validate all multimodal schemas are registered.
pub fn validate_multimodal_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<Pipeline>(ComponentSchemaId(SCHEMA_PIPELINE))
        .map_err(|e| format!("Pipeline schema: {e}"))?;
    reg.verify_type::<PipelineStage>(ComponentSchemaId(SCHEMA_PIPELINE_STAGE))
        .map_err(|e| format!("PipelineStage schema: {e}"))?;
    reg.verify_type::<PipelineModality>(ComponentSchemaId(SCHEMA_PIPELINE_MODALITY))
        .map_err(|e| format!("PipelineModality schema: {e}"))?;
    reg.verify_type::<InputArtifactRef>(ComponentSchemaId(SCHEMA_INPUT_ARTIFACT))
        .map_err(|e| format!("InputArtifactRef schema: {e}"))?;
    reg.verify_type::<OutputArtifactRef>(ComponentSchemaId(SCHEMA_OUTPUT_ARTIFACT))
        .map_err(|e| format!("OutputArtifactRef schema: {e}"))?;
    reg.verify_type::<PipelineLifecycle>(ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE))
        .map_err(|e| format!("PipelineLifecycle schema: {e}"))?;
    reg.verify_type::<WorkLeaseRef>(ComponentSchemaId(SCHEMA_WORK_LEASE_REF))
        .map_err(|e| format!("WorkLeaseRef schema: {e}"))?;
    Ok(())
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MultimodalError {
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("no stages specified")]
    NoStages,
    #[error("session entity {0} not found")]
    SessionNotFound(u64),
    #[error("session entity {0} not Active")]
    SessionNotActive(u64),
    #[error("pipeline entity {0} not found")]
    PipelineNotFound(u64),
    #[error("pipeline entity {0} not Executing or Assembled")]
    PipelineNotExecuting(u64),
    #[error("model entity {0} not found")]
    ModelNotFound(u64),
    #[error("commit failed: {0}")]
    CommitFailed(crate::ecs::constitutional::world_txn::WorldTxnError),
    #[error("invalid pipeline lifecycle transition: {0}")]
    InvalidTransition(LifecycleError),
}

/// Replay a `pipeline_created` event to reconstruct a pipeline entity.
///
/// Restores: Pipeline, PipelineLifecycle::Created. Idempotent.
pub fn replay_pipeline_created(
    world: &mut CompWorld,
    event: &DomainEvent,
) -> Result<CommittedEpoch, MultimodalError> {
    let pipeline_id = event
        .entity_id
        .ok_or(MultimodalError::PipelineNotFound(0))?
        .0;
    let session_entity = event
        .payload
        .get("session_entity")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut txn = WorldTxn::new(world);
    if !world.has_entity(crate::ecs::CompEntity(pipeline_id)) {
        txn.stage_spawn(pipeline_id, EntityKind::Pipeline);
    }
    txn.add_component(
        pipeline_id,
        ComponentSchemaId(SCHEMA_PIPELINE),
        SchemaVersion(1),
        Pipeline {
            pipeline_id,
            session_entity,
            target_modality: "replay".to_string(),
            created_at: Timestamp::now(),
        },
    );
    txn.add_component(
        pipeline_id,
        ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE),
        SchemaVersion(1),
        PipelineLifecycle::Created,
    );
    let epoch = world.transit(txn).map_err(MultimodalError::CommitFailed)?;
    Ok(epoch)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::lifecycle::SessionLifecycle;
    use crate::ecs::constitutional::schema::{ComponentDurability, SchemaEntry};
    use crate::ecs::constitutional::types::SchemaVersion;

    fn make_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE),
            version: SchemaVersion(1),
            type_name: "Pipeline".into(),
            description: "Multimodal generation pipeline".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<Pipeline>()),
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
            type_id: Some(std::any::TypeId::of::<InputArtifactRef>()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_OUTPUT_ARTIFACT),
            version: SchemaVersion(1),
            type_name: "OutputArtifactRef".into(),
            description: "Output artifact reference".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<OutputArtifactRef>()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_PIPELINE_LIFECYCLE),
            version: SchemaVersion(1),
            type_name: "PipelineLifecycle".into(),
            description: "Pipeline lifecycle".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<PipelineLifecycle>()),
        });
        reg.register(SchemaEntry {
            schema_id: ComponentSchemaId(SCHEMA_WORK_LEASE_REF),
            version: SchemaVersion(1),
            type_name: "WorkLeaseRef".into(),
            description: "Work lease reference".into(),
            durability: ComponentDurability::Durable,
            type_id: Some(std::any::TypeId::of::<WorkLeaseRef>()),
        });
        reg
    }

    fn make_session(world: &mut CompWorld) -> u64 {
        let id = WorldTxn::next_entity_id(world);
        let mut txn = WorldTxn::new(world);
        txn.stage_spawn(id, EntityKind::Session);
        txn.add_component(
            id,
            ComponentSchemaId(16), // SCHEMA_SESSION_LIFECYCLE
            SchemaVersion(1),
            SessionLifecycle::Active,
        );
        world.transit(txn).unwrap();
        id
    }

    // ── test_pipeline_lifecycle_transitions ───────────────────────────────

    #[test]
    fn test_pipeline_lifecycle_transitions() {
        // Happy path
        assert!(PipelineLifecycle::Created
            .can_transition_to(PipelineLifecycle::Assembled)
            .is_ok());
        assert!(PipelineLifecycle::Assembled
            .can_transition_to(PipelineLifecycle::Executing)
            .is_ok());
        assert!(PipelineLifecycle::Executing
            .can_transition_to(PipelineLifecycle::Completed)
            .is_ok());

        // Pause/Resume
        assert!(PipelineLifecycle::Executing
            .can_transition_to(PipelineLifecycle::Paused)
            .is_ok());
        assert!(PipelineLifecycle::Paused
            .can_transition_to(PipelineLifecycle::Executing)
            .is_ok());

        // Cancellation
        assert!(PipelineLifecycle::Paused
            .can_transition_to(PipelineLifecycle::Cancelled)
            .is_ok());
        assert!(PipelineLifecycle::Executing
            .can_transition_to(PipelineLifecycle::Cancelled)
            .is_ok());
        assert!(PipelineLifecycle::Created
            .can_transition_to(PipelineLifecycle::Cancelled)
            .is_ok());

        // Failure paths — can fail from any non-terminal state
        let non_terminal = [
            PipelineLifecycle::Created,
            PipelineLifecycle::Assembled,
            PipelineLifecycle::Executing,
            PipelineLifecycle::Paused,
        ];
        for state in &non_terminal {
            assert!(
                state.can_transition_to(PipelineLifecycle::Failed).is_ok(),
                "{:?} should be able to fail",
                state
            );
        }

        // Terminal states cannot transition
        assert!(PipelineLifecycle::Completed
            .can_transition_to(PipelineLifecycle::Executing)
            .is_err());
        assert!(PipelineLifecycle::Cancelled
            .can_transition_to(PipelineLifecycle::Created)
            .is_err());
        assert!(PipelineLifecycle::Failed
            .can_transition_to(PipelineLifecycle::Assembled)
            .is_err());

        // Invalid forward jumps
        assert!(PipelineLifecycle::Created
            .can_transition_to(PipelineLifecycle::Executing)
            .is_err());
        assert!(PipelineLifecycle::Assembled
            .can_transition_to(PipelineLifecycle::Completed)
            .is_err());
    }

    // ── test_pipeline_serde ──────────────────────────────────────────────

    #[test]
    fn test_pipeline_serde() {
        let pipeline = Pipeline {
            pipeline_id: 42,
            session_entity: 7,
            target_modality: "image".to_string(),
            created_at: Timestamp(1_700_000_000_000_000_000),
        };
        let json = serde_json::to_string(&pipeline).unwrap();
        let deserialized: Pipeline = serde_json::from_str(&json).unwrap();
        assert_eq!(pipeline, deserialized);

        let stage = PipelineStage {
            stage_index: 0,
            stage_type: "encode".to_string(),
            model_entity: 100,
            input_transform: "tensor_to_input".to_string(),
            output_transform: "latent_to_output".to_string(),
        };
        let json = serde_json::to_string(&stage).unwrap();
        let deserialized: PipelineStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, deserialized);
    }

    // ── test_pipeline_modality_discriminants ──────────────────────────────

    #[test]
    fn test_pipeline_modality_discriminants() {
        let variants = [
            PipelineModality::Vision,
            PipelineModality::Audio,
            PipelineModality::Tts,
            PipelineModality::Diffusion,
            PipelineModality::Image,
            PipelineModality::Video,
            PipelineModality::Multimodal("custom".to_string()),
        ];
        // All discriminants are distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    variants[i], variants[j],
                    "{:?} should not equal {:?}",
                    variants[i], variants[j]
                );
            }
        }
    }

    // ── test_work_lease_ref_serde ─────────────────────────────────────────

    #[test]
    fn test_work_lease_ref_serde() {
        let lease = WorkLeaseRef {
            lease_id: 10,
            work_entity: 20,
            stage_index: 1,
        };
        let json = serde_json::to_string(&lease).unwrap();
        let deserialized: WorkLeaseRef = serde_json::from_str(&json).unwrap();
        assert_eq!(lease, deserialized);
    }

    // ── test_input_output_artifact_ref_serde ──────────────────────────────

    #[test]
    fn test_input_output_artifact_ref_serde() {
        let input = InputArtifactRef {
            artifact_id: 100,
            role: "prompt".to_string(),
            loaded: true,
        };
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: InputArtifactRef = serde_json::from_str(&json).unwrap();
        assert_eq!(input, deserialized);

        let output = OutputArtifactRef {
            artifact_id: Some(200),
            stage_index: 2,
            mime_type: "image/png".to_string(),
            size_bytes: 65536,
        };
        let json = serde_json::to_string(&output).unwrap();
        let deserialized: OutputArtifactRef = serde_json::from_str(&json).unwrap();
        assert_eq!(output, deserialized);
    }

    // ── test_pipeline_terminal ────────────────────────────────────────────

    #[test]
    fn test_pipeline_terminal() {
        assert!(PipelineLifecycle::Cancelled.is_terminal());
        assert!(PipelineLifecycle::Completed.is_terminal());
        assert!(PipelineLifecycle::Failed.is_terminal());

        assert!(!PipelineLifecycle::Created.is_terminal());
        assert!(!PipelineLifecycle::Assembled.is_terminal());
        assert!(!PipelineLifecycle::Executing.is_terminal());
        assert!(!PipelineLifecycle::Paused.is_terminal());
    }

    // ── test_create_pipeline_preflight ────────────────────────────────────

    #[test]
    fn test_create_pipeline_preflight() {
        let mut world = CompWorld::new();
        let reg = make_registry();

        // No session — should fail
        let cmd = CreatePipelineCommand {
            id: MessageId::compute(b"test-no-session"),
            session_entity: 1,
            target_modality: PipelineModality::Image,
            stages: vec![PipelineStage {
                stage_index: 0,
                stage_type: "encode".to_string(),
                model_entity: 100,
                input_transform: "a".to_string(),
                output_transform: "b".to_string(),
            }],
            input_artifacts: vec![],
        };
        assert!(cmd.preflight(&world, &reg).is_err());

        // Create a session
        let session_id = make_session(&mut world);

        // Valid command
        let cmd = CreatePipelineCommand {
            id: MessageId::compute(b"test-valid"),
            session_entity: session_id,
            target_modality: PipelineModality::Image,
            stages: vec![PipelineStage {
                stage_index: 0,
                stage_type: "encode".to_string(),
                model_entity: 100,
                input_transform: "a".to_string(),
                output_transform: "b".to_string(),
            }],
            input_artifacts: vec![],
        };
        assert!(cmd.preflight(&world, &reg).is_err()); // model 100 doesn't exist

        // Empty stages
        let cmd = CreatePipelineCommand {
            id: MessageId::compute(b"test-no-stages"),
            session_entity: session_id,
            target_modality: PipelineModality::Image,
            stages: vec![],
            input_artifacts: vec![],
        };
        assert!(cmd.preflight(&world, &reg).is_err());
    }

    // ── test_create_pipeline_execute ──────────────────────────────────────

    #[test]
    fn test_create_pipeline_execute() {
        let mut world = CompWorld::new();
        let reg = make_registry();
        let session_id = make_session(&mut world);
        // Model at entity 100, artifact at entity 200 — explicit IDs to avoid
        // conflict with session_entity (entity 1 from make_session).
        // Entity 100: Model with Deployable lifecycle
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(100, EntityKind::Model);
        txn.add_component(
            100,
            ComponentSchemaId(7), // SCHEMA_MODEL_LIFECYCLE
            SchemaVersion(1),
            crate::ecs::constitutional::residency::ModelLifecycle::Deployable,
        );
        world.transit(txn).unwrap();

        // Entity 200: Artifact (no components)
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(200, EntityKind::Artifact);
        world.transit(txn).unwrap();

        let cmd = CreatePipelineCommand {
            id: MessageId::compute(b"test-execute"),
            session_entity: session_id,
            target_modality: PipelineModality::Tts,
            stages: vec![PipelineStage {
                stage_index: 0,
                stage_type: "generate".to_string(),
                model_entity: 100,
                input_transform: "text".to_string(),
                output_transform: "audio".to_string(),
            }],
            input_artifacts: vec![InputArtifactRef {
                artifact_id: 200,
                role: "prompt".to_string(),
                loaded: true,
            }],
        };

        // Execute should succeed
        let (epoch, event) = cmd.execute(&mut world, &reg).unwrap();
        assert!(epoch.0 .0 > 0);
        assert_eq!(event.kind, "pipeline_created");
    }
}
