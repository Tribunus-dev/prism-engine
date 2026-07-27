//! `ValidationReceipt` and `SubmitValidationReceiptCommand`.
//!
//! **Single authority:** owns the canonical evidence of a validator's
//! verdict on a compiled `CompilationJob` and the command that
//! attaches a receipt to a job. A receipt is a durable fact: once
//! submitted, it persists with the job and can be replayed.
//!
//! Adjacent authorities (separate sub-modules):
//! - [`super::job`] — the `CompilationJob` and its lifecycle.
//! - [`super::cimage_promotion`] — promotion flow that consults
//!   receipts before allowing the `Sealed → Promoted` transition.

use crate::compilation::job::CompilationError;
use crate::schema::SchemaRegistry;
use crate::types::{EntityKindId, MessageId, SchemaKey, Timestamp};
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTransitExt, WorldTxn,
};
use prism_ecs_core::{Component, Entity, World};
use serde::{Deserialize, Serialize};

use super::job::JobLifecycle;

// ── Component type ──────────────────────────────────────────────────────────

/// A validation receipt — evidence that a validator checked the
/// compiled output.
///
/// A receipt is bound to a specific `job_id` (a `u64` mirror of the
/// entity ID, retained for backwards compatibility with the durable
/// event payload). The `evidence_digest` is the canonical 32-byte
/// content address of whatever evidence the validator produced; the
/// `validator_type` names the validator's protocol so receipts are
/// self-describing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReceipt {
    pub job_id: u64,
    pub validator_type: String,
    pub passed: bool,
    pub evidence_digest: [u8; 32],
    pub validated_at: Timestamp,
}

impl Component for ValidationReceipt {}
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

// ── Submit command ──────────────────────────────────────────────────────────

/// Command to submit a validation receipt for a compilation job.
///
/// The job must be in the `Validating` state; otherwise the command
/// is rejected at preflight. Receipts are append-only: there is no
/// "amend" or "revoke" command. A failed receipt leaves the job in
/// `Failed` via the lifecycle transition in the receipt path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitValidationReceiptCommand {
    pub id: MessageId,
    /// Entity ID of the job receiving the validation receipt. See
    /// [`Entity`] for the canonical generational entity handle.
    pub job_entity: u64,
    pub receipt: ValidationReceipt,
}

impl SubmitValidationReceiptCommand {
    /// Preflight: validate schemas and that the job entity exists
    /// with a compatible lifecycle.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        super::validate_compilation_schemas(schema_registry)
            .map_err(CompilationError::SchemaError)?;

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
    ) -> Result<(CommittedEpoch, crate::command::DomainEvent), CompilationError> {
        self.preflight(world, schema_registry)?;

        let entity = Entity::new(self.job_entity, 0);
        let mut txn = WorldTxn::new(world);

        txn.put_durable(entity, self.receipt.clone());

        let event = crate::command::DomainEvent {
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── test_validation_receipt_construction ───────────────────────────

    #[test]
    fn validation_receipt_construction() {
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
    fn validation_receipt_serde_roundtrip() {
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

    // ── test_submit_validation_receipt_command ─────────────────────────

    #[test]
    fn submit_validation_receipt_rejects_non_validating_state() {
        let mut world = World::new();
        let mut reg = SchemaRegistry::new();
        super::super::register_compilation_schemas(&mut reg);

        // Spawn an entity as a job (must be in some lifecycle state)
        let entity_id = WorldTxn::next_entity_id(&world);
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(entity_id, prism_ecs_core::EntityKind::Executable);
        txn.add_component(
            entity_id,
            crate::types::ComponentSchemaId(super::super::schema_ids::SCHEMA_JOB_LIFECYCLE),
            crate::types::SchemaVersion(1),
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
        assert!(matches!(result, Err(CompilationError::InvalidState { .. })));
    }
}
