use crate::ecs::constitutional::artifact::ArtifactDigest;
use crate::ecs::constitutional::command::{DomainEvent, EffectRequest};
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::CompWorld;
use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// Component Schema IDs (31–38)
// ══════════════════════════════════════════════════════════════════════════════

pub const SCHEMA_COMPILATION_JOB: u64 = 31;
pub const SCHEMA_JOB_INPUT: u64 = 32;
pub const SCHEMA_JOB_CONFIG: u64 = 33;
pub const SCHEMA_JOB_OUTPUT: u64 = 34;
pub const SCHEMA_JOB_LIFECYCLE: u64 = 35;
pub const SCHEMA_VALIDATION_RECEIPT: u64 = 36;
pub const SCHEMA_QUANTIZATION_PLAN: u64 = 37;
pub const SCHEMA_CIMAGE_PROMOTION: u64 = 38;

// ══════════════════════════════════════════════════════════════════════════════
// Component Types
// ══════════════════════════════════════════════════════════════════════════════

/// A compilation job — a unit of work to compile a model artifact into a target binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationJob {
    pub job_id: u64,
    pub target_artifact: u64,
    pub target_device_profile: String,
    pub created_at: Timestamp,
}

/// Input specification for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobInput {
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
    pub cimage_entity: Option<u64>,
    pub output_digest: Option<ArtifactDigest>,
    pub size_bytes: u64,
}

/// Lifecycle state of a compilation job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobLifecycle {
    Pending,
    Compiling,
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
    /// - Compiling → Validating
    /// - Validating → Failed | Sealed
    /// - Sealed → Promoted
    pub fn can_transition_to(&self, target: Self) -> bool {
        matches!(
            (self, target),
            (Self::Pending, Self::Compiling)
                | (Self::Compiling, Self::Validating)
                | (Self::Validating, Self::Failed | Self::Sealed)
                | (Self::Sealed, Self::Promoted)
        )
    }
}

/// A validation receipt — evidence that a validator checked the compiled output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReceipt {
    pub job_id: u64,
    pub validator_type: String,
    pub passed: bool,
    pub evidence_digest: [u8; 32],
    pub validated_at: Timestamp,
}

/// A quantization plan — describes how weights should be quantized.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationPlan {
    pub codec: String,
    pub group_size: u32,
    pub target_bitwidth: u8,
    pub validation_gate: String,
}

/// Promotion record — marks a CImage as promoted to Sealed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CimagePromotion {
    pub cimage_entity: u64,
    pub promotion_generation: u32,
    pub validation_receipt_ids: Vec<u64>,
    pub promoted_at: Timestamp,
}

// ── Component Trait impls ────────────────────────────────────────────────────

impl crate::ecs::Component for CompilationJob {}
impl crate::ecs::Component for JobInput {}
impl crate::ecs::Component for JobConfig {}
impl crate::ecs::Component for JobOutput {}
impl crate::ecs::Component for JobLifecycle {}
impl crate::ecs::Component for ValidationReceipt {}
impl crate::ecs::Component for QuantizationPlan {}
impl crate::ecs::Component for CimagePromotion {}

// ══════════════════════════════════════════════════════════════════════════════
// Schema Validation
// ══════════════════════════════════════════════════════════════════════════════

/// Validate all compilation schemas are registered for the correct types.
pub fn validate_compilation_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<CompilationJob>(ComponentSchemaId(SCHEMA_COMPILATION_JOB))
        .map_err(|e| format!("CompilationJob schema: {e}"))?;
    reg.verify_type::<JobInput>(ComponentSchemaId(SCHEMA_JOB_INPUT))
        .map_err(|e| format!("JobInput schema: {e}"))?;
    reg.verify_type::<JobConfig>(ComponentSchemaId(SCHEMA_JOB_CONFIG))
        .map_err(|e| format!("JobConfig schema: {e}"))?;
    reg.verify_type::<JobOutput>(ComponentSchemaId(SCHEMA_JOB_OUTPUT))
        .map_err(|e| format!("JobOutput schema: {e}"))?;
    reg.verify_type::<JobLifecycle>(ComponentSchemaId(SCHEMA_JOB_LIFECYCLE))
        .map_err(|e| format!("JobLifecycle schema: {e}"))?;
    reg.verify_type::<ValidationReceipt>(ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT))
        .map_err(|e| format!("ValidationReceipt schema: {e}"))?;
    reg.verify_type::<QuantizationPlan>(ComponentSchemaId(SCHEMA_QUANTIZATION_PLAN))
        .map_err(|e| format!("QuantizationPlan schema: {e}"))?;
    reg.verify_type::<CimagePromotion>(ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION))
        .map_err(|e| format!("CimagePromotion schema: {e}"))?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Commands
// ══════════════════════════════════════════════════════════════════════════════

/// Command to create a new compilation job in the Pending state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateCompilationJobCommand {
    pub id: MessageId,
    pub job_id: u64,
    pub model_artifact: u64,
    pub target_profile: String,
    pub config: JobConfig,
}

impl CreateCompilationJobCommand {
    /// Create the effect request for compilation preparation.
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("compile_prep:{}", self.job_id).as_bytes()),
            kind: crate::ecs::constitutional::command::EffectKind::CompileModel,
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
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        // Validate model artifact entity exists
        let model_entity = crate::ecs::CompEntity(self.model_artifact);
        if !world.has_entity(model_entity) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }
        if world.entity_kind(model_entity) != Some(crate::ecs::EntityKind::Artifact) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }

        Ok(())
    }

    /// Execute: spawn job entity, attach components, emit event, commit.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity_id = WorldTxn::next_entity_id(world);
        let now = Timestamp::now();

        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(entity_id, crate::ecs::EntityKind::Executable);

        // Job metadata
        txn.add_component(
            entity_id,
            ComponentSchemaId(SCHEMA_COMPILATION_JOB),
            SchemaVersion(1),
            CompilationJob {
                job_id: self.job_id,
                target_artifact: self.model_artifact,
                target_device_profile: self.target_profile.clone(),
                created_at: now,
            },
        );

        // Input
        txn.add_component(
            entity_id,
            ComponentSchemaId(SCHEMA_JOB_INPUT),
            SchemaVersion(1),
            JobInput {
                model_artifact: self.model_artifact,
                source_format: "".to_string(),
                quantization_profile: None,
            },
        );

        // Config
        txn.add_component(
            entity_id,
            ComponentSchemaId(SCHEMA_JOB_CONFIG),
            SchemaVersion(1),
            self.config.clone(),
        );

        // Lifecycle — starts Pending
        txn.add_component(
            entity_id,
            ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
            SchemaVersion(1),
            JobLifecycle::Pending,
        );

        let event = DomainEvent {
            id: self.id,
            kind: "compilation_job_created".to_string(),
            entity_id: Some(EntityKindId(entity_id)),
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

/// Command to submit a validation receipt for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitValidationReceiptCommand {
    pub id: MessageId,
    pub job_entity: u64,
    pub receipt: ValidationReceipt,
}

impl SubmitValidationReceiptCommand {
    /// Preflight: validate schemas and that the job entity exists with a compatible lifecycle.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        let entity = crate::ecs::CompEntity(self.job_entity);
        if !world.has_entity(entity) {
            return Err(CompilationError::JobNotFound(self.job_entity));
        }

        // Job must be in Validating state to accept receipts
        let lifecycle = world
            .get_component::<JobLifecycle>(entity)
            .ok_or(CompilationError::JobNotFound(self.job_entity))?;
        if *lifecycle != JobLifecycle::Validating {
            return Err(CompilationError::InvalidState {
                job_id: self.job_entity,
                expected: JobLifecycle::Validating,
                actual: *lifecycle,
            });
        }

        Ok(())
    }

    /// Execute: attach validation receipt to the job entity.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity = self.job_entity;
        let mut txn = WorldTxn::new(world);

        txn.add_component(
            entity,
            ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT),
            SchemaVersion(1),
            self.receipt.clone(),
        );

        let event = DomainEvent {
            id: self.id,
            kind: "validation_receipt_submitted".to_string(),
            entity_id: Some(EntityKindId(self.job_entity)),
            payload: serde_json::json!({
                "job_entity": self.job_entity,
                "validator_type": self.receipt.validator_type,
                "passed": self.receipt.passed,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

        Ok((epoch, event))
    }
}

/// Command to promote a compiled CImage to the Sealed state.
///
/// Validation gates must all pass before promotion is allowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromoteCimageCommand {
    pub id: MessageId,
    pub cimage_entity: u64,
    pub receipt_ids: Vec<u64>,
}

impl PromoteCimageCommand {
    /// Preflight: validate schemas, entity existence, and gate conditions.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        let cimage = crate::ecs::CompEntity(self.cimage_entity);
        if !world.has_entity(cimage) {
            return Err(CompilationError::JobNotFound(self.cimage_entity));
        }

        // Check lifecycle: must be Sealed to transition to Promoted
        let lifecycle = world
            .get_component::<JobLifecycle>(cimage)
            .ok_or(CompilationError::JobNotFound(self.cimage_entity))?;
        if *lifecycle != JobLifecycle::Sealed {
            return Err(CompilationError::InvalidState {
                job_id: self.cimage_entity,
                expected: JobLifecycle::Sealed,
                actual: *lifecycle,
            });
        }

        // All receipt IDs must have been submitted
        for rid in &self.receipt_ids {
            let receipt_entity = crate::ecs::CompEntity(*rid);
            if world
                .get_component::<ValidationReceipt>(receipt_entity)
                .is_none()
            {
                return Err(CompilationError::MissingReceipt(self.cimage_entity));
            }
        }

        Ok(())
    }

    /// Execute: attach promotion record and update lifecycle to Promoted.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let now = Timestamp::now();
        let mut txn = WorldTxn::new(world);

        // Attach promotion record
        txn.add_component(
            self.cimage_entity,
            ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION),
            SchemaVersion(1),
            CimagePromotion {
                cimage_entity: self.cimage_entity,
                promotion_generation: 1,
                validation_receipt_ids: self.receipt_ids.clone(),
                promoted_at: now,
            },
        );

        // Transition lifecycle to Promoted (add replaces the old value)
        txn.add_component(
            self.cimage_entity,
            ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
            SchemaVersion(1),
            JobLifecycle::Promoted,
        );

        let event = DomainEvent {
            id: self.id,
            kind: "cimage_promoted".to_string(),
            entity_id: Some(EntityKindId(self.cimage_entity)),
            payload: serde_json::json!({
                "cimage_entity": self.cimage_entity,
                "receipt_count": self.receipt_ids.len(),
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

        Ok((epoch, event))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Errors
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompilationError {
    #[error("schema error: {0}")]
    SchemaError(String),

    #[error("model artifact {0} not found")]
    ModelArtifactNotFound(u64),

    #[error("job entity {0} not found")]
    JobNotFound(u64),

    #[error("invalid state for job {job_id}: expected {expected:?}, actual {actual:?}")]
    InvalidState {
        job_id: u64,
        expected: JobLifecycle,
        actual: JobLifecycle,
    },

    #[error("missing validation receipt for cimage {0}")]
    MissingReceipt(u64),

    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ══════════════════════════════════════════════════════════════════════════════
// Replay Helper
// ══════════════════════════════════════════════════════════════════════════════

/// Replay a `compilation_job_created` event to reconstruct job state.
///
/// Restores: CompilationJob, JobInput, JobConfig, JobLifecycle (Pending).
/// Ephemeral state is not restored; the job starts Pending and will
/// await fresh commands to progress.
pub fn replay_compilation_job_created(
    world: &mut CompWorld,
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
    let _target_profile = payload["target_profile"]
        .as_str()
        .unwrap_or("default")
        .to_string();

    let now = Timestamp::now();
    let mut txn = WorldTxn::new(world);

    if !world.has_entity(crate::ecs::CompEntity(job_id)) {
        txn.stage_spawn(job_id, crate::ecs::EntityKind::Executable);
    }

    txn.add_component(
        job_id,
        ComponentSchemaId(SCHEMA_COMPILATION_JOB),
        SchemaVersion(1),
        CompilationJob {
            job_id,
            target_artifact: model_artifact,
            target_device_profile: _target_profile,
            created_at: now,
        },
    );

    txn.add_component(
        job_id,
        ComponentSchemaId(SCHEMA_JOB_INPUT),
        SchemaVersion(1),
        JobInput {
            model_artifact,
            source_format: String::new(),
            quantization_profile: None,
        },
    );

    txn.add_component(
        job_id,
        ComponentSchemaId(SCHEMA_JOB_CONFIG),
        SchemaVersion(1),
        JobConfig {
            target_format: payload["target_format"].as_str().unwrap_or("").to_string(),
            optimization_level: 0,
            enable_validation: false,
        },
    );

    txn.add_component(
        job_id,
        ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
        SchemaVersion(1),
        JobLifecycle::Pending,
    );

    let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

    Ok((epoch, job_id))
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::schema::SchemaEntry;
    use crate::ecs::CompWorld;

    /// Build a schema registry with all compilation types registered.
    fn make_compilation_schema_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register_for_type::<CompilationJob>(
            ComponentSchemaId(SCHEMA_COMPILATION_JOB),
            SchemaVersion(1),
            "CompilationJob",
            "Compilation job metadata",
            Default::default(),
        );
        reg.register_for_type::<JobInput>(
            ComponentSchemaId(SCHEMA_JOB_INPUT),
            SchemaVersion(1),
            "JobInput",
            "Compilation job input",
            Default::default(),
        );
        reg.register_for_type::<JobConfig>(
            ComponentSchemaId(SCHEMA_JOB_CONFIG),
            SchemaVersion(1),
            "JobConfig",
            "Compilation job config",
            Default::default(),
        );
        reg.register_for_type::<JobOutput>(
            ComponentSchemaId(SCHEMA_JOB_OUTPUT),
            SchemaVersion(1),
            "JobOutput",
            "Compilation job output",
            Default::default(),
        );
        reg.register_for_type::<JobLifecycle>(
            ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
            SchemaVersion(1),
            "JobLifecycle",
            "Compilation job lifecycle",
            Default::default(),
        );
        reg.register_for_type::<ValidationReceipt>(
            ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT),
            SchemaVersion(1),
            "ValidationReceipt",
            "Validation receipt",
            Default::default(),
        );
        reg.register_for_type::<QuantizationPlan>(
            ComponentSchemaId(SCHEMA_QUANTIZATION_PLAN),
            SchemaVersion(1),
            "QuantizationPlan",
            "Quantization plan",
            Default::default(),
        );
        reg.register_for_type::<CimagePromotion>(
            ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION),
            SchemaVersion(1),
            "CimagePromotion",
            "CImage promotion record",
            Default::default(),
        );
        reg
    }

    // ── test_job_lifecycle_transitions ───────────────────────────────────

    #[test]
    fn test_job_lifecycle_transitions() {
        // Valid transitions
        assert!(JobLifecycle::Pending.can_transition_to(JobLifecycle::Compiling));
        assert!(JobLifecycle::Compiling.can_transition_to(JobLifecycle::Validating));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Failed));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Sealed));
        assert!(JobLifecycle::Sealed.can_transition_to(JobLifecycle::Promoted));

        // Invalid transitions
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Failed));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Validating.can_transition_to(JobLifecycle::Compiling));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Failed));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Sealed.can_transition_to(JobLifecycle::Compiling));
        assert!(!JobLifecycle::Promoted.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Promoted.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Failed.can_transition_to(JobLifecycle::Promoted));
    }

    // ── test_compilation_job_serde ───────────────────────────────────────

    #[test]
    fn test_compilation_job_serde() {
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
    fn test_job_lifecycle_serde() {
        for state in &[
            JobLifecycle::Pending,
            JobLifecycle::Compiling,
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

    // ── test_validation_receipt_construction ─────────────────────────────

    #[test]
    fn test_validation_receipt_construction() {
        let now = Timestamp::now();
        let receipt = ValidationReceipt {
            job_id: 42,
            validator_type: "accuracy_gate".to_string(),
            passed: true,
            evidence_digest: [0xab; 32],
            validated_at: now,
        };

        assert_eq!(receipt.job_id, 42);
        assert_eq!(receipt.validator_type, "accuracy_gate");
        assert!(receipt.passed);
        assert_eq!(receipt.evidence_digest, [0xab; 32]);
        assert_eq!(receipt.validated_at, now);
    }

    #[test]
    fn test_validation_receipt_serde() {
        let receipt = ValidationReceipt {
            job_id: 99,
            validator_type: "perf_gate".to_string(),
            passed: false,
            evidence_digest: [0xcd; 32],
            validated_at: Timestamp::from_nanos(2_000_000_000),
        };

        let json = serde_json::to_string(&receipt).expect("serialize");
        let deserialized: ValidationReceipt = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(receipt, deserialized);
    }

    // ── test_quantization_plan_construction ──────────────────────────────

    #[test]
    fn test_quantization_plan_construction() {
        let plan = QuantizationPlan {
            codec: "nf4".to_string(),
            group_size: 64,
            target_bitwidth: 4,
            validation_gate: "quantization_admission".to_string(),
        };

        assert_eq!(plan.codec, "nf4");
        assert_eq!(plan.group_size, 64);
        assert_eq!(plan.target_bitwidth, 4);
        assert_eq!(plan.validation_gate, "quantization_admission");
    }

    // ── test_schema_validation ───────────────────────────────────────────

    #[test]
    fn test_compilation_schema_validation() {
        let reg = make_compilation_schema_registry();
        assert!(validate_compilation_schemas(&reg).is_ok());
    }

    #[test]
    fn test_compilation_schema_validation_fails_when_missing() {
        let reg = SchemaRegistry::new();
        assert!(validate_compilation_schemas(&reg).is_err());
    }

    // ── test_create_compilation_job_command ──────────────────────────────

    #[test]
    fn test_create_compilation_job_execute() {
        let mut world = CompWorld::new();
        let reg = make_compilation_schema_registry();

        // Set up an artifact entity for the model
        world.spawn(crate::ecs::EntityKind::Artifact, Some("test_model".into()));

        let config = JobConfig {
            target_format: "mlmodelc".to_string(),
            optimization_level: 3,
            enable_validation: true,
        };

        let cmd = CreateCompilationJobCommand {
            id: MessageId::compute(b"create_job_1"),
            job_id: 100,
            model_artifact: 1,
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
        let entity = crate::ecs::CompEntity(job_entity);
        assert!(world.has_entity(entity));

        let job = world
            .get_component::<CompilationJob>(entity)
            .expect("CompilationJob component");
        assert_eq!(job.job_id, 100);
        assert_eq!(job.target_artifact, 1);
        assert_eq!(job.target_device_profile, "m1-ane");

        let lifecycle = world
            .get_component::<JobLifecycle>(entity)
            .expect("JobLifecycle component");
        assert_eq!(*lifecycle, JobLifecycle::Pending);
    }

    #[test]
    fn test_create_compilation_job_preflight_rejects_missing_artifact() {
        let world = CompWorld::new();
        let reg = make_compilation_schema_registry();

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
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(CompilationError::ModelArtifactNotFound(999))
        ));
    }

    // ── test_submit_validation_receipt_command ───────────────────────────

    #[test]
    fn test_submit_validation_receipt_rejects_non_validating_state() {
        let mut world = CompWorld::new();
        let reg = make_compilation_schema_registry();

        // Spawn an entity as a job (must be in some lifecycle state)
        let entity_id = {
            let e = world.spawn(
                crate::ecs::EntityKind::Executable,
                Some("job_entity".into()),
            );
            e.0
        };

        // Lifecycle is Pending, not Validating — should fail
        world.add_component(crate::ecs::CompEntity(entity_id), JobLifecycle::Pending);

        let cmd = SubmitValidationReceiptCommand {
            id: MessageId::compute(b"receipt_1"),
            job_entity: entity_id,
            receipt: ValidationReceipt {
                job_id: 100,
                validator_type: "accuracy_gate".to_string(),
                passed: true,
                evidence_digest: [0xab; 32],
                validated_at: Timestamp::from_nanos(1_000_000),
            },
        };

        let result = cmd.preflight(&world, &reg);
        assert!(result.is_err());
        assert!(matches!(result, Err(CompilationError::InvalidState { .. })));
    }
}
