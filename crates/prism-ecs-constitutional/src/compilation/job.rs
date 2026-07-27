//! `CompilationJob` and its supporting components, commands, and replay.
//!
//! **Single authority:** owns the canonical shape of a `CompilationJob`
//! and the state-machine (`JobLifecycle`) that governs its transitions
//! from creation to promotion. Also owns `CreateCompilationJobCommand`
//! (the only sanctioned entry point that mints a job) and the
//! `replay_compilation_job_created` re-applier.
//!
//! Adjacent authorities that live in their own sub-modules:
//! - [`super::validation`] — receipt submission and validator outcomes.
//! - [`super::quantization`] — per-tensor result submission.
//! - [`super::cimage_promotion`] — terminal promotion to `Promoted`.
//!
//! The constitutional crate is the source of truth for this authority.
//! The engine (`compute-core`) has no duplicate; the engine's
//! `core/compile_state.rs` is execution-boundary (it writes/reads a
//! `compile.state.json` checkpoint file on disk, which is file
//! descriptor I/O per AGENTS.md criterion 1) and is documented as a
//! typed port below.

use crate::artifact::ArtifactDigest;
use crate::command::{DomainEvent, EffectRequest};
use crate::compilation::schema_ids::{
    SCHEMA_COMPILATION_JOB, SCHEMA_JOB_CONFIG, SCHEMA_JOB_INPUT, SCHEMA_JOB_LIFECYCLE,
};
use crate::schema::SchemaRegistry;
use crate::types::{
    ComponentSchemaId, EntityKindId, MessageId, SchemaKey, SchemaVersion, Timestamp,
};
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTransitExt, WorldTxn,
    WorldTxnError,
};
use prism_ecs_core::{Component, Entity, EntityKind, World};
use serde::{Deserialize, Serialize};

// ── Component types ─────────────────────────────────────────────────────────

/// A compilation job — a unit of work to compile a model artifact into a
/// target binary.
///
/// A `CompilationJob` is the durable record that names the source
/// artifact, the target device profile, and when the job was created.
/// Job state changes are tracked by the [`JobLifecycle`] component
/// attached to the same entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationJob {
    /// Logical job identifier.
    pub job_id: u64,
    /// Entity ID of the model artifact being compiled. See [`Entity`]
    /// for the canonical generational entity handle.
    pub target_artifact: u64,
    pub target_device_profile: String,
    pub created_at: Timestamp,
}

/// Input specification for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobInput {
    /// Entity ID of the model artifact to compile. See [`Entity`] for
    /// the canonical generational entity handle.
    pub model_artifact: u64,
    pub source_format: String,
    pub quantization_profile: Option<String>,
}

/// Configuration for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobConfig {
    pub target_format: String,
    pub optimization_level: u32,
    pub enable_validation: bool,
}

/// Output of a compilation job — populated on completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobOutput {
    /// Entity ID of the compiled CImage, if available. See [`Entity`]
    /// for the canonical generational entity handle.
    pub cimage_entity: Option<u64>,
    pub output_digest: Option<ArtifactDigest>,
    pub size_bytes: u64,
}

/// Lifecycle state of a compilation job.
///
/// The state machine here is the canonical one. Any other surface that
/// reports job progress (e.g. the engine's `CompileStage` enum) must
/// map onto these variants. The mapping is:
///
/// | `JobLifecycle`        | `compile_state::CompileStage`     |
/// |-----------------------|-----------------------------------|
/// | `Pending`             | (pre-creation)                    |
/// | `Compiling`           | `Planning`                        |
/// | `Planned`             | `Planning`                        |
/// | `Validating`          | `Verifying`                       |
/// | `Failed`              | `Failed { reason }`               |
/// | `Sealed`              | (post-verify, pre-promote)        |
/// | `Promoted`            | `Complete`                        |
///
/// `Cancelled` (engine-only) is not part of the constitutional state
/// machine because cancellation is an execution-plane event that does
/// not produce a durable `CompilationJob` transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobLifecycle {
    Pending,
    Compiling,
    /// A structured `QuantizationResultComponent` has been attached to
    /// the job. The per-tensor decisions exist; the artifact has not
    /// yet been sealed.
    Planned,
    Validating,
    Failed,
    Sealed,
    Promoted,
}

impl JobLifecycle {
    /// Returns `true` if a transition from `self` to `target` is valid.
    ///
    /// Valid transitions:
    /// - Pending → Compiling
    /// - Compiling → Planned
    /// - Planned → Validating | Failed
    /// - Validating → Failed | Sealed
    /// - Sealed → Promoted
    #[must_use]
    pub fn can_transition_to(&self, target: Self) -> bool {
        matches!(
            (self, target),
            (Self::Pending, Self::Compiling)
                | (Self::Compiling, Self::Planned)
                | (Self::Planned, Self::Validating | Self::Failed)
                | (Self::Validating, Self::Failed | Self::Sealed)
                | (Self::Sealed, Self::Promoted)
        )
    }
}

// ── Component / `DurableComponent` impls ────────────────────────────────────

impl Component for CompilationJob {}
impl ClassifiedComponent for CompilationJob {
    type Class = DurableClass;
}
impl DurableComponent for CompilationJob {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 31,
        version: 1,
    };
}

impl Component for JobInput {}
impl ClassifiedComponent for JobInput {
    type Class = DurableClass;
}
impl DurableComponent for JobInput {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 32,
        version: 1,
    };
}

impl Component for JobConfig {}
impl ClassifiedComponent for JobConfig {
    type Class = DurableClass;
}
impl DurableComponent for JobConfig {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 33,
        version: 1,
    };
}

impl Component for JobOutput {}
impl ClassifiedComponent for JobOutput {
    type Class = DurableClass;
}
impl DurableComponent for JobOutput {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 34,
        version: 1,
    };
}

impl Component for JobLifecycle {}
impl ClassifiedComponent for JobLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for JobLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 35,
        version: 1,
    };
}

// ── Error type (used by all sub-modules) ────────────────────────────────────

/// Errors produced by the compilation sub-modules.
///
/// Categorization (per AGENTS.md "per-crate error enums"):
/// - `Rejected` (preflight) — `SchemaError`, `ModelArtifactNotFound`,
///   `JobNotFound`, `InvalidState`, `MissingReceipt`.
/// - `Failed` (commit) — `CommitFailed` wrapping a `WorldTxnError`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompilationError {
    #[error("schema error: {0}")]
    SchemaError(String),

    /// The model artifact entity (u64 ID) was not found. See [`Entity`]
    /// for the canonical generational entity handle.
    #[error("model artifact {0} not found")]
    ModelArtifactNotFound(u64),

    /// The job entity (u64 ID) was not found. See [`Entity`] for the
    /// canonical generational entity handle.
    #[error("job entity {0} not found")]
    JobNotFound(u64),

    /// The job is in an invalid lifecycle state for the requested
    /// operation. `job_id` refers to the entity ID. See [`Entity`]
    /// for the canonical generational entity handle.
    #[error("invalid state for job {job_id}: expected {expected:?}, actual {actual:?}")]
    InvalidState {
        job_id: u64,
        expected: JobLifecycle,
        actual: JobLifecycle,
    },

    /// A required validation receipt entity (u64 ID) was not found.
    /// See [`Entity`] for the canonical generational entity handle.
    #[error("missing validation receipt for cimage {0}")]
    MissingReceipt(u64),

    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Create job command ──────────────────────────────────────────────────────

/// Command to create a new compilation job in the `Pending` state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateCompilationJobCommand {
    pub id: MessageId,
    /// Logical job identifier.
    pub job_id: u64,
    /// Entity ID of the model artifact to compile. See [`Entity`] for
    /// the canonical generational entity handle.
    pub model_artifact: u64,
    pub target_profile: String,
    pub config: JobConfig,
}

impl CreateCompilationJobCommand {
    /// Create the effect request for compilation preparation.
    #[must_use]
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("compile_prep:{}", self.job_id).as_bytes()),
            kind: crate::command::EffectKind::CompileModel,
            params: serde_json::json!({
                "job_id": self.job_id,
                "model_artifact": self.model_artifact,
                "target_profile": self.target_profile,
                "target_format": self.config.target_format,
            }),
        }
    }

    /// Preflight: validate schemas and model entity existence.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        super::validate_compilation_schemas(schema_registry)
            .map_err(CompilationError::SchemaError)?;

        // Validate model artifact entity exists
        let model_entity = Entity::new(self.model_artifact, 0);
        if !world.has_entity(model_entity) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }
        if world.entity_kind(model_entity) != Some(EntityKind::Artifact) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }

        Ok(())
    }

    /// Execute: spawn job entity, attach components, emit event, commit.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;
        // Execute: spawn job entity, attach components, emit event, commit.
        let job_entity = WorldTxn::next_entity_id(world);
        let now = Timestamp::now();

        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(job_entity, EntityKind::Executable);

        // Job metadata
        txn.put_durable(
            job_entity,
            CompilationJob {
                job_id: self.job_id,
                target_artifact: self.model_artifact,
                target_device_profile: self.target_profile.clone(),
                created_at: now,
            },
        );

        // Input
        txn.put_durable(
            job_entity,
            JobInput {
                model_artifact: self.model_artifact,
                source_format: String::new(),
                quantization_profile: None,
            },
        );

        // Config
        txn.put_durable(job_entity, self.config.clone());

        // Lifecycle — starts Pending
        txn.put_durable(job_entity, JobLifecycle::Pending);

        let event = DomainEvent {
            id: self.id,
            kind: "compilation_job_created".to_string(),
            entity_id: Some(EntityKindId(job_entity.id())),
            payload: serde_json::json!({
                "job_id": self.job_id,
                "model_artifact": self.model_artifact,
                "target_profile": self.target_profile,
                "target_format": self.config.target_format,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

        Ok((epoch, event))
    }
}

// ── Replay ──────────────────────────────────────────────────────────────────

/// Replay a `compilation_job_created` event to reconstruct job state.
///
/// Restores: `CompilationJob`, `JobInput`, `JobConfig`, `JobLifecycle`
/// (`Pending`). Ephemeral state is not restored; the job starts
/// `Pending` and will await fresh commands to progress.
///
/// Returns the committed epoch and the entity ID (u64) of the
/// reconstructed job entity. See [`Entity`] for the canonical
/// generational entity handle.
pub fn replay_compilation_job_created(
    world: &mut World,
    event: &DomainEvent,
) -> Result<(CommittedEpoch, u64), CompilationError> {
    let job_id = event
        .entity_id
        .ok_or_else(|| CompilationError::JobNotFound(0))?
        .0;

    let payload = &event.payload;
    let model_artifact = payload["model_artifact"]
        .as_u64()
        .ok_or_else(|| CompilationError::ModelArtifactNotFound(0))?;
    let target_profile = payload["target_profile"]
        .as_str()
        .unwrap_or("default")
        .to_string();

    let now = Timestamp::now();
    let entity = Entity::new(job_id, 0);
    let mut txn = WorldTxn::new(world);

    if !world.has_entity(Entity::new(job_id, 0)) {
        txn.stage_spawn(entity, EntityKind::Executable);
    }

    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_COMPILATION_JOB),
        SchemaVersion(1),
        CompilationJob {
            job_id,
            target_artifact: model_artifact,
            target_device_profile: target_profile,
            created_at: now,
        },
    );

    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_JOB_INPUT),
        SchemaVersion(1),
        JobInput {
            model_artifact,
            source_format: String::new(),
            quantization_profile: None,
        },
    );

    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_JOB_CONFIG),
        SchemaVersion(1),
        JobConfig {
            target_format: payload["target_format"].as_str().unwrap_or("").to_string(),
            optimization_level: 0,
            enable_validation: false,
        },
    );

    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
        SchemaVersion(1),
        JobLifecycle::Pending,
    );

    let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

    Ok((epoch, job_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_job_lifecycle_transitions ──────────────────────────────────

    #[test]
    fn job_lifecycle_can_transition_to_valid_edges() {
        assert!(JobLifecycle::Pending.can_transition_to(JobLifecycle::Compiling));
        assert!(JobLifecycle::Compiling.can_transition_to(JobLifecycle::Planned));
        assert!(JobLifecycle::Planned.can_transition_to(JobLifecycle::Validating));
        assert!(JobLifecycle::Planned.can_transition_to(JobLifecycle::Failed));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Failed));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Sealed));
        assert!(JobLifecycle::Sealed.can_transition_to(JobLifecycle::Promoted));
    }

    #[test]
    fn job_lifecycle_rejects_invalid_edges() {
        // Skip transitions
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Failed));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Planned));
        // Backwards
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Failed));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Compiling));
        // Terminal
        assert!(!JobLifecycle::Promoted.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Promoted.can_transition_to(JobLifecycle::Sealed));
        // Failed is absorbing
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Promoted));
        // Misc
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Planned.can_transition_to(JobLifecycle::Compiling));
        assert!(!JobLifecycle::Planned.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Validating.can_transition_to(JobLifecycle::Compiling));
    }

    // ── test_compilation_job_serde ──────────────────────────────────────

    #[test]
    fn compilation_job_serde_roundtrip() {
        let job = CompilationJob {
            job_id: 42,
            target_artifact: 7,
            target_device_profile: "m1-ane".to_string(),
            created_at: Timestamp::from_nanos(1_000_000),
        };

        let json = serde_json::to_string(&job).expect("serialize");
        let deserialized: CompilationJob = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(job, deserialized);
    }

    #[test]
    fn job_lifecycle_serde_roundtrip_all_variants() {
        for state in &[
            JobLifecycle::Pending,
            JobLifecycle::Compiling,
            JobLifecycle::Planned,
            JobLifecycle::Validating,
            JobLifecycle::Failed,
            JobLifecycle::Sealed,
            JobLifecycle::Promoted,
        ] {
            let json = serde_json::to_string(state).expect("serialize");
            let deserialized: JobLifecycle = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*state, deserialized);
        }
    }

    // ── test_job_config_serde ──────────────────────────────────────────

    #[test]
    fn job_config_serde_roundtrip() {
        let cfg = JobConfig {
            target_format: "mlmodelc".to_string(),
            optimization_level: 3,
            enable_validation: true,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: JobConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    // ── test_create_compilation_job_command ────────────────────────────

    #[test]
    fn create_compilation_job_execute_succeeds_with_artifact() {
        let mut world = World::new();
        let mut reg = SchemaRegistry::new();
        super::super::register_compilation_schemas(&mut reg);

        // Set up an artifact entity for the model
        // Use id 2 (past next_id=1) so spawn_entity_with_id bumps next_id to 3,
        // ensuring cmd.execute() gets a fresh id for the job entity.
        let artifact_id = 2u64;
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(Entity::new(artifact_id, 0), EntityKind::Artifact);
        world.transit(txn).unwrap();

        let config = JobConfig {
            target_format: "mlmodelc".to_string(),
            optimization_level: 3,
            enable_validation: true,
        };

        let cmd = CreateCompilationJobCommand {
            id: MessageId::compute(b"create_job_1"),
            job_id: 100,
            model_artifact: 2,
            target_profile: "m1-ane".to_string(),
            config,
        };

        let result = cmd.execute(&mut world, &reg);
        assert!(result.is_ok(), "execute failed: {:?}", result.err());

        let (epoch, event) = result.unwrap();
        assert_eq!(event.kind, "compilation_job_created");
        assert!(epoch.0 .0 > 0);

        // Verify the job entity was created
        let job_entity = event.entity_id.unwrap().0;
        let entity = Entity::new(job_entity, 0);
        assert!(world.has_entity(entity));

        let job = world
            .get_component::<CompilationJob>(entity)
            .expect("CompilationJob component");
        assert_eq!(job.job_id, 100);
        assert_eq!(job.target_artifact, 2);
        assert_eq!(job.target_device_profile, "m1-ane");

        let lifecycle = world
            .get_component::<JobLifecycle>(entity)
            .expect("JobLifecycle component");
        assert_eq!(*lifecycle, JobLifecycle::Pending);
    }

    #[test]
    fn create_compilation_job_preflight_rejects_missing_artifact() {
        let world = World::new();
        let mut reg = SchemaRegistry::new();
        super::super::register_compilation_schemas(&mut reg);

        let cmd = CreateCompilationJobCommand {
            id: MessageId::compute(b"bad_job"),
            job_id: 101,
            model_artifact: 999, // doesn't exist
            target_profile: "m1-ane".to_string(),
            config: JobConfig {
                target_format: "mlmodelc".to_string(),
                optimization_level: 0,
                enable_validation: false,
            },
        };

        let result = cmd.preflight(&world, &reg);
        assert!(matches!(
            result,
            Err(CompilationError::ModelArtifactNotFound(999))
        ));
    }
}
