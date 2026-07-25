use crate::artifact::ArtifactDigest;
use crate::command::{DomainEvent, EffectRequest};
use crate::schema::SchemaRegistry;
use crate::types::*;
use crate::world_txn::{ClassifiedComponent, DurableClass, DurableComponent};
use crate::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use prism_ecs_core::Entity;
use prism_ecs_core::World;

use crate::world_txn::WorldTransitExt;
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
pub const SCHEMA_QUANTIZATION_RESULT: u64 = 39;

// ══════════════════════════════════════════════════════════════════════════════
// Component Types
// ══════════════════════════════════════════════════════════════════════════════

/// A compilation job — a unit of work to compile a model artifact into a target binary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationJob {
    /// Logical job identifier.
    pub job_id: u64,
    /// Entity ID of the model artifact being compiled. See [`Entity`] for the
    /// canonical generational entity handle.
    pub target_artifact: u64,
    pub target_device_profile: String,
    pub created_at: Timestamp,
}

/// Input specification for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobInput {
    /// Entity ID of the model artifact to compile. See [`Entity`] for the
    /// canonical generational entity handle.
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
    /// Entity ID of the compiled CImage, if available. See [`Entity`] for the
    /// canonical generational entity handle.
    pub cimage_entity: Option<u64>,
    pub output_digest: Option<ArtifactDigest>,
    pub size_bytes: u64,
}

/// Lifecycle state of a compilation job.
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

/// Per-tensor result of a quantization pass, attached to a `CompilationJob`.
///
/// This is the constitutional counterpart of
/// `prism_ecs_quantization::QuantizationResult`. It is the structured
/// evidence that the per-tensor codecs ran and produced specific
/// representations. The job cannot transition from `Compiling` to
/// `Validating` without this component attached.
///
/// `selections` is the canonical ordered list of decisions. Its digest
/// is what `CImage` emission later seals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizationResultComponent {
    /// Content digest of the source model the selections were derived from.
    pub source_digest: String,
    /// Target hardware identifier (e.g. "apple-m1", "cpu").
    pub target_hardware: String,
    /// Per-tensor selections, in source-graph iteration order.
    pub selections: Vec<QuantizedTensorSelectionComponent>,
    /// Default format used for tensors that did not appear in the
    /// caller-supplied `FormatPlan`. Surfaced explicitly so receipts
    /// can prove the default policy was applied.
    pub default_format: String,
    /// Schema version of the originating `prism_ecs_quantization` plan.
    /// Bumped if the on-wire shape changes.
    pub schema_version: u32,
}

/// A single per-tensor selection — the constitutional counterpart of
/// `prism_ecs_quantization::QuantizedTensorSelection`. Stores the
/// payload bytes and the format that was actually applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedTensorSelectionComponent {
    pub key: String,
    /// `TensorFormat` discriminant, encoded as a string for
    /// forward-compatibility. Decoded via
    /// `prism_ecs_quantization::TensorFormat::discriminant_byte` /
    /// `from_discriminant_byte` (the latter is added when the
    /// first version of this component ships).
    pub format_discriminant: u8,
    /// CImage-ready payload bytes for the applied codec.
    pub payload: Vec<u8>,
    /// `TensorType` discriminant (CImage physical type).
    pub tensor_type_discriminant: u8,
    pub dim_m: u32,
    pub dim_n: u32,
    pub effective_bpp: f32,
    pub payload_bytes: u64,
}

// ── Component Trait impls ────────────────────────────────────────────────────

impl prism_ecs_core::Component for CompilationJob {}
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

impl prism_ecs_core::Component for JobInput {}
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

impl prism_ecs_core::Component for JobConfig {}
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

impl prism_ecs_core::Component for JobOutput {}
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

impl prism_ecs_core::Component for JobLifecycle {}
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

impl prism_ecs_core::Component for ValidationReceipt {}
impl ClassifiedComponent for ValidationReceipt {
    type Class = DurableClass;
}
impl DurableComponent for ValidationReceipt {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 36,
        version: 1,
    };
}

impl prism_ecs_core::Component for QuantizationPlan {}
impl ClassifiedComponent for QuantizationPlan {
    type Class = DurableClass;
}
impl DurableComponent for QuantizationPlan {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 37,
        version: 1,
    };
}

impl prism_ecs_core::Component for CimagePromotion {}
impl ClassifiedComponent for CimagePromotion {
    type Class = DurableClass;
}
impl DurableComponent for CimagePromotion {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 38,
        version: 1,
    };
}

impl prism_ecs_core::Component for QuantizationResultComponent {}
impl ClassifiedComponent for QuantizationResultComponent {
    type Class = DurableClass;
}
impl DurableComponent for QuantizationResultComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 39,
        version: 1,
    };
}

impl prism_ecs_core::Component for QuantizedTensorSelectionComponent {}
impl ClassifiedComponent for QuantizedTensorSelectionComponent {
    type Class = DurableClass;
}
impl DurableComponent for QuantizedTensorSelectionComponent {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.compilation",
        id: 40,
        version: 1,
    };
}

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
    reg.verify_type::<QuantizationResultComponent>(ComponentSchemaId(SCHEMA_QUANTIZATION_RESULT))
        .map_err(|e| format!("QuantizationResultComponent schema: {e}"))?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Commands
// ══════════════════════════════════════════════════════════════════════════════

/// Command to create a new compilation job in the Pending state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateCompilationJobCommand {
    pub id: MessageId,
    /// Logical job identifier.
    pub job_id: u64,
    /// Entity ID of the model artifact to compile. See [`Entity`] for the
    /// canonical generational entity handle.
    pub model_artifact: u64,
    pub target_profile: String,
    pub config: JobConfig,
}

impl CreateCompilationJobCommand {
    /// Create the effect request for compilation preparation.
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
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        // Validate model artifact entity exists
        let model_entity = Entity::new(self.model_artifact, 0);
        if !world.has_entity(model_entity) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }
        if world.entity_kind(model_entity) != Some(prism_ecs_core::EntityKind::Artifact) {
            return Err(CompilationError::ModelArtifactNotFound(self.model_artifact));
        }

        Ok(())
    }

    // Execute: spawn job entity, attach components, emit event, commit.
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

        txn.stage_spawn(job_entity, prism_ecs_core::EntityKind::Executable);

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
                source_format: "".to_string(),
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

/// Command to submit a validation receipt for a compilation job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitValidationReceiptCommand {
    pub id: MessageId,
    /// Entity ID of the job receiving the validation receipt. See [`Entity`]
    /// for the canonical generational entity handle.
    pub job_entity: u64,
    pub receipt: ValidationReceipt,
}

impl SubmitValidationReceiptCommand {
    /// Preflight: validate schemas and that the job entity exists with a compatible lifecycle.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        let entity = Entity::new(self.job_entity, 0);
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
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity = prism_ecs_core::Entity::new(self.job_entity, 0);
        let mut txn = WorldTxn::new(world);

        txn.put_durable(entity, self.receipt.clone());

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

/// Command to submit a `QuantizationResultComponent` for a compilation
/// job. This is the chokepoint between per-tensor compilation (which
/// lives in `prism_ecs_quantization`) and the constitutional
/// `CompilationJob` entity. After this command, the job has a
/// `Planned` lifecycle and a structured plan that downstream
/// validation and emission can read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmitQuantizationResultCommand {
    pub id: MessageId,
    /// Entity ID of the job receiving the plan. See [`Entity`] for the
    /// canonical generational entity handle.
    pub job_entity: u64,
    /// The structured per-tensor result.
    pub result: QuantizationResultComponent,
}

impl SubmitQuantizationResultCommand {
    /// Preflight: validate schemas and that the job exists in a
    /// compatible lifecycle (`Compiling` is the only accepted source).
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(CompilationError::SchemaError)?;

        let entity = Entity::new(self.job_entity, 0);
        if !world.has_entity(entity) {
            return Err(CompilationError::JobNotFound(self.job_entity));
        }
        let lifecycle = world
            .get_component::<JobLifecycle>(entity)
            .ok_or(CompilationError::JobNotFound(self.job_entity))?;
        if *lifecycle != JobLifecycle::Compiling {
            return Err(CompilationError::InvalidState {
                job_id: self.job_entity,
                expected: JobLifecycle::Compiling,
                actual: *lifecycle,
            });
        }
        Ok(())
    }

    /// Execute: attach the result and transition the lifecycle to
    /// `Planned`.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity = Entity::new(self.job_entity, 0);
        let mut txn = WorldTxn::new(world);

        txn.put_durable(entity, self.result.clone());
        txn.put_durable(entity, JobLifecycle::Planned);

        let event = DomainEvent {
            id: self.id,
            kind: "quantization_result_submitted".to_string(),
            entity_id: Some(EntityKindId(self.job_entity)),
            payload: serde_json::json!({
                "job_entity": self.job_entity,
                "selection_count": self.result.selections.len(),
                "default_format": self.result.default_format,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(CompilationError::CommitFailed)?;

        Ok((epoch, event))
    }
}

/// Convert a `prism_ecs_quantization::QuantizationResult` into the
/// constitutional `QuantizationResultComponent`.
///
/// This lives in the constitutional crate because it is the only
/// place that knows the wire-format on the ECS side. The
/// `prism_ecs_quantization` crate stays pure of constitutional
/// dependencies.
#[cfg(feature = "quantization-bridge")]
pub fn quantization_result_to_component(
    result: &prism_ecs_quantization::QuantizationResult,
) -> QuantizationResultComponent {
    use prism_ecs_quantization::cimage::TensorType;
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;

    fn tensor_type_discriminant(t: &TensorType) -> u8 {
        t.discriminant_byte()[0]
    }

    QuantizationResultComponent {
        source_digest: result.source_digest.clone(),
        target_hardware: result.target_hardware.clone(),
        selections: result
            .selections
            .iter()
            .map(|s| QuantizedTensorSelectionComponent {
                key: s.key.clone(),
                format_discriminant: s.format.discriminant_byte(),
                payload: s.payload.clone(),
                tensor_type_discriminant: tensor_type_discriminant(&s.tensor_type),
                dim_m: s.dim_m,
                dim_n: s.dim_n,
                effective_bpp: s.effective_bpp,
                payload_bytes: s.payload_bytes,
            })
            .collect(),
        default_format: format!("{:?}", result.default_format),
        schema_version: 1,
    }
}

/// Reference to the default format for runtime use. Re-exported for
/// code that needs the typed value but cannot import the IR enum.
pub fn default_format_name() -> &'static str {
    "Palettized4Bit"
}

/// Command to promote a compiled CImage to the Sealed state.
///
/// Validation gates must all pass before promotion is allowed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromoteCimageCommand {
    pub id: MessageId,
    /// Entity ID of the CImage to promote. See [`Entity`] for the canonical
    /// generational entity handle.
    pub cimage_entity: prism_ecs_core::Entity,
    /// Entity IDs of the validation receipts that must pass before promotion.
    /// See [`Entity`] for the canonical generational entity handle.
    pub receipt_ids: Vec<u64>,
}

impl PromoteCimageCommand {
    /// Preflight: validate schemas, entity existence, and gate conditions.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        validate_compilation_schemas(schema_registry)
            .map_err(|e| CompilationError::SchemaError(e))?;

        let cimage = Entity::new(self.cimage_entity.id(), 0);
        if !world.has_entity(cimage) {
            return Err(CompilationError::JobNotFound(self.cimage_entity.id()));
        }

        // Check lifecycle: must be Sealed to transition to Promoted
        let lifecycle = world
            .get_component::<JobLifecycle>(cimage)
            .ok_or(CompilationError::JobNotFound(self.cimage_entity.id()))?;
        if *lifecycle != JobLifecycle::Sealed {
            return Err(CompilationError::InvalidState {
                job_id: self.cimage_entity.id(),
                expected: JobLifecycle::Sealed,
                actual: *lifecycle,
            });
        }

        // All receipt IDs must have been submitted
        for rid in &self.receipt_ids {
            let receipt_entity = Entity::new(*rid, 0);
            if world
                .get_component::<ValidationReceipt>(receipt_entity)
                .is_none()
            {
                return Err(CompilationError::MissingReceipt(self.cimage_entity.id()));
            }
        }

        Ok(())
    }

    /// Execute: attach promotion record and update lifecycle to Promoted.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let now = Timestamp::now();
        let cimage = self.cimage_entity;
        let mut txn = WorldTxn::new(world);

        // Attach promotion record
        txn.put_durable(
            cimage,
            CimagePromotion {
                cimage_entity: cimage.id(),
                promotion_generation: 1,
                validation_receipt_ids: self.receipt_ids.clone(),
                promoted_at: now,
            },
        );

        // Transition lifecycle to Promoted (add replaces the old value)
        txn.put_durable(cimage, JobLifecycle::Promoted);

        let event = DomainEvent {
            id: self.id,
            kind: "cimage_promoted".to_string(),
            entity_id: Some(EntityKindId(self.cimage_entity.id())),
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

    /// The model artifact entity (u64 ID) was not found. See [`Entity`] for
    /// the canonical generational entity handle.
    #[error("model artifact {0} not found")]
    ModelArtifactNotFound(u64),

    /// The job entity (u64 ID) was not found. See [`Entity`] for the canonical
    /// generational entity handle.
    #[error("job entity {0} not found")]
    JobNotFound(u64),

    /// The job is in an invalid lifecycle state for the requested operation.
    /// `job_id` refers to the entity ID. See [`Entity`] for the canonical
    /// generational entity handle.
    #[error("invalid state for job {job_id}: expected {expected:?}, actual {actual:?}")]
    InvalidState {
        job_id: u64,
        expected: JobLifecycle,
        actual: JobLifecycle,
    },

    /// A required validation receipt entity (u64 ID) was not found. See
    /// [`Entity`] for the canonical generational entity handle.
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
///
/// Returns the committed epoch and the entity ID (u64) of the reconstructed
/// job entity. See [`Entity`] for the canonical generational entity handle.
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
    let _target_profile = payload["target_profile"]
        .as_str()
        .unwrap_or("default")
        .to_string();

    let now = Timestamp::now();
    let entity = prism_ecs_core::Entity::new(job_id, 0);
    let mut txn = WorldTxn::new(world);

    if !world.has_entity(Entity::new(job_id, 0)) {
        txn.stage_spawn(entity, prism_ecs_core::EntityKind::Executable);
    }

    txn.add_component(
        entity,
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

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::World;

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
        reg.register_for_type::<QuantizationResultComponent>(
            ComponentSchemaId(SCHEMA_QUANTIZATION_RESULT),
            SchemaVersion(1),
            "QuantizationResultComponent",
            "Per-tensor quantization result attached to a job",
            Default::default(),
        );
        reg
    }

    // ── test_job_lifecycle_transitions ───────────────────────────────────

    #[test]
    fn test_job_lifecycle_transitions() {
        // Valid transitions
        assert!(JobLifecycle::Pending.can_transition_to(JobLifecycle::Compiling));
        assert!(JobLifecycle::Compiling.can_transition_to(JobLifecycle::Planned));
        assert!(JobLifecycle::Planned.can_transition_to(JobLifecycle::Validating));
        assert!(JobLifecycle::Planned.can_transition_to(JobLifecycle::Failed));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Failed));
        assert!(JobLifecycle::Validating.can_transition_to(JobLifecycle::Sealed));
        assert!(JobLifecycle::Sealed.can_transition_to(JobLifecycle::Promoted));

        // Invalid transitions
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Failed));
        assert!(!JobLifecycle::Pending.can_transition_to(JobLifecycle::Planned));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Pending));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Sealed));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Promoted));
        assert!(!JobLifecycle::Compiling.can_transition_to(JobLifecycle::Validating));
        assert!(!JobLifecycle::Planned.can_transition_to(JobLifecycle::Compiling));
        assert!(!JobLifecycle::Planned.can_transition_to(JobLifecycle::Promoted));
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
        let mut world = World::new();
        let reg = make_compilation_schema_registry();

        // Set up an artifact entity for the model
        // Use id 2 (past next_id=1) so spawn_entity_with_id bumps next_id to 3,
        // ensuring cmd.execute() gets a fresh id for the job entity.
        let artifact_id = 2u64;
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(
            prism_ecs_core::Entity::new(artifact_id, 0),
            prism_ecs_core::EntityKind::Artifact,
        );
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
    fn test_create_compilation_job_preflight_rejects_missing_artifact() {
        let world = World::new();
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
        let mut world = World::new();
        let reg = make_compilation_schema_registry();

        // Spawn an entity as a job (must be in some lifecycle state)
        let entity_id = WorldTxn::next_entity_id(&world);
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(entity_id, prism_ecs_core::EntityKind::Executable);
        txn.add_component(
            entity_id,
            ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
            SchemaVersion(1),
            JobLifecycle::Pending,
        );
        world.transit(txn).unwrap();

        let cmd = SubmitValidationReceiptCommand {
            id: MessageId::compute(b"receipt_1"),
            job_entity: entity_id.id(),
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
