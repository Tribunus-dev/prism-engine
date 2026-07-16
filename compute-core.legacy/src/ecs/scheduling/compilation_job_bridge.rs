//! Bridge wrapping constitutional compilation commands behind a simple API
//! for production compiler callers.
//!
//! Owns an `Arc<RwLock<World>>` and provides synchronous methods that
//! lock the world, construct the appropriate constitutional command,
//! then call its preflight + execute lifecycle.

use crate::ecs::constitutional::compilation::{
    CimagePromotion, CompilationJob, CreateCompilationJobCommand, JobConfig, JobInput,
    JobLifecycle, JobOutput, PromoteCimageCommand, QuantizationPlan,
    SubmitValidationReceiptCommand, ValidationReceipt, SCHEMA_CIMAGE_PROMOTION,
    SCHEMA_COMPILATION_JOB, SCHEMA_JOB_CONFIG, SCHEMA_JOB_INPUT, SCHEMA_JOB_LIFECYCLE,
    SCHEMA_JOB_OUTPUT, SCHEMA_QUANTIZATION_PLAN, SCHEMA_VALIDATION_RECEIPT,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion, Timestamp};
use crate::ecs::{Entity, World};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Wraps constitutional compilation commands behind a simple synchronous API.
///
/// Each method:
/// 1. Generates a unique `MessageId` via UUID
/// 2. Locks the world for writing
/// 3. Constructs the constitutional command
/// 4. Executes it (preflight + execute)
/// 5. Returns the result
pub struct CompilationJobBridge {
    world: Arc<RwLock<World>>,
    schema_registry: SchemaRegistry,
}

impl CompilationJobBridge {
    /// Create a new bridge backed by the given world.
    ///
    /// Registers all compilation schemas on construction so schema validation
    /// inside each constitutional command passes.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        let mut schema_registry = SchemaRegistry::new();
        Self::register_compilation_schemas(&mut schema_registry);
        Self {
            world,
            schema_registry,
        }
    }

    /// Register all compilation domain schemas into the given registry.
    fn register_compilation_schemas(reg: &mut SchemaRegistry) {
        reg.register_for_type::<CompilationJob>(
            ComponentSchemaId(SCHEMA_COMPILATION_JOB),
            SchemaVersion(1),
            "CompilationJob",
            "Compilation job metadata",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<JobInput>(
            ComponentSchemaId(SCHEMA_JOB_INPUT),
            SchemaVersion(1),
            "JobInput",
            "Compilation job input",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<JobConfig>(
            ComponentSchemaId(SCHEMA_JOB_CONFIG),
            SchemaVersion(1),
            "JobConfig",
            "Compilation job config",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<JobOutput>(
            ComponentSchemaId(SCHEMA_JOB_OUTPUT),
            SchemaVersion(1),
            "JobOutput",
            "Compilation job output",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<JobLifecycle>(
            ComponentSchemaId(SCHEMA_JOB_LIFECYCLE),
            SchemaVersion(1),
            "JobLifecycle",
            "Compilation job lifecycle",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<ValidationReceipt>(
            ComponentSchemaId(SCHEMA_VALIDATION_RECEIPT),
            SchemaVersion(1),
            "ValidationReceipt",
            "Validation receipt",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<QuantizationPlan>(
            ComponentSchemaId(SCHEMA_QUANTIZATION_PLAN),
            SchemaVersion(1),
            "QuantizationPlan",
            "Quantization plan",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<CimagePromotion>(
            ComponentSchemaId(SCHEMA_CIMAGE_PROMOTION),
            SchemaVersion(1),
            "CimagePromotion",
            "CImage promotion record",
            ComponentDurability::Durable,
        );
    }

    /// Create a compilation job entity, transitioning it to `Pending` state.
    ///
    /// Calls [`CreateCompilationJobCommand`] internally.
    /// Returns the job entity id on success.
    pub fn create_job(
        &self,
        model_artifact: u64,
        target_profile: &str,
        config: JobConfig,
    ) -> Result<u64, String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        // Peek the next entity id before execute — the command internally
        // allocates the same entity via WorldTxn::next_entity_id().
        let entity_id = world.next_entity_id();

        let uuid = uuid();
        let id = MessageId::compute(uuid.as_bytes());

        let cmd = CreateCompilationJobCommand {
            id,
            job_id: uuid.as_u128() as u64,
            model_artifact,
            target_profile: target_profile.to_string(),
            config,
        };

        cmd.execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        Ok(entity_id)
    }

    /// Submit a validation receipt for a compilation job.
    ///
    /// Calls [`SubmitValidationReceiptCommand`] internally.
    pub fn submit_validation(
        &self,
        job_entity: Entity,
        validator_id: &str,
        passed: bool,
        details: &str,
    ) -> Result<(), String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        let uuid = uuid();
        let id = MessageId::compute(uuid.as_bytes());

        // Hash the details string into a 32-byte evidence digest.
        let evidence_digest = blake3::hash(details.as_bytes());

        let cmd = SubmitValidationReceiptCommand {
            id,
            job_entity: job_entity.id(),
            receipt: ValidationReceipt {
                job_id: job_entity.id(),
                validator_type: validator_id.to_string(),
                passed,
                evidence_digest: *evidence_digest.as_bytes(),
                validated_at: Timestamp::now(),
            },
        };

        cmd.execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Promote a compiled CImage to the Sealed (later Promoted) state.
    ///
    /// Calls [`PromoteCimageCommand`] internally.
    ///
    /// The bridge does not track individual receipt entity IDs — callers
    /// that need receipt-gated promotion should manage receipt entity
    /// references externally and invoke the constitutional command directly.
    pub fn promote_cimage(
        &self,
        job_entity: Entity,
        _cimage_digest: &str,
        _cimage_size_bytes: u64,
        _target_hardware: &str,
    ) -> Result<(), String> {
        let mut world = self.world.write().map_err(|e| e.to_string())?;

        let uuid = uuid();
        let id = MessageId::compute(uuid.as_bytes());

        let cmd = PromoteCimageCommand {
            id,
            cimage_entity: job_entity,
            // The bridge API does not surface receipt entity IDs;
            // callers that need strict receipt gating should use the
            // constitutional command directly with populated receipt_ids.
            receipt_ids: Vec::new(),
        };

        cmd.execute(&mut world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        Ok(())
    }
}

/// Generate a random v4 UUID.
fn uuid() -> Uuid {
    Uuid::new_v4()
}
