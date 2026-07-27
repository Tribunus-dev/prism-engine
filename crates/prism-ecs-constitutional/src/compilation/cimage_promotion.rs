//! `CimagePromotion` and `PromoteCimageCommand`.
//!
//! **Single authority:** owns the canonical promotion record that
//! marks a CImage as promoted to `Sealed`. Promotion is the terminal
//! transition of a `CompilationJob`: it requires that all referenced
//! `ValidationReceipt` entities have been submitted and that the
//! job's current [`super::job::JobLifecycle`] is `Sealed`.

use crate::compilation::job::CompilationError;
use crate::schema::SchemaRegistry;
use crate::types::{EntityKindId, MessageId, SchemaKey, Timestamp};
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTransitExt, WorldTxn,
};
use prism_ecs_core::{Component, Entity, World};
use serde::{Deserialize, Serialize};

use super::job::JobLifecycle;
use super::validation::ValidationReceipt;

// ── Component type ──────────────────────────────────────────────────────────

/// Promotion record — marks a CImage as promoted to `Sealed`.
///
/// `promotion_generation` is bumped each time the same CImage is
/// re-promoted (rare; the typical path is one-shot). The list of
/// `validation_receipt_ids` names the receipts that gated this
/// promotion; the receipts themselves live as separate
/// `ValidationReceipt` components on the referenced entities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CimagePromotion {
    pub cimage_entity: u64,
    pub promotion_generation: u32,
    pub validation_receipt_ids: Vec<u64>,
    pub promoted_at: Timestamp,
}

impl Component for CimagePromotion {}
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

// ── Promote command ─────────────────────────────────────────────────────────

/// Command to promote a compiled CImage to the `Sealed` state.
///
/// Validation gates must all pass before promotion is allowed: every
/// `receipt_id` must resolve to an existing `ValidationReceipt`
/// component, and the CImage entity's lifecycle must currently be
/// `Sealed` (i.e. the job has passed validation and is awaiting
/// promotion).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromoteCimageCommand {
    pub id: MessageId,
    /// Entity ID of the CImage to promote. See [`Entity`] for the
    /// canonical generational entity handle.
    pub cimage_entity: Entity,
    /// Entity IDs of the validation receipts that must pass before
    /// promotion. See [`Entity`] for the canonical generational
    /// entity handle.
    pub receipt_ids: Vec<u64>,
}

impl PromoteCimageCommand {
    /// Preflight: validate schemas, entity existence, and gate
    /// conditions.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), CompilationError> {
        super::validate_compilation_schemas(schema_registry)
            .map_err(CompilationError::SchemaError)?;

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

    /// Execute: attach promotion record and update lifecycle to
    /// `Promoted`.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, crate::command::DomainEvent), CompilationError> {
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

        let event = crate::command::DomainEvent {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cimage_promotion_construction() {
        let promotion = CimagePromotion {
            cimage_entity: 7,
            promotion_generation: 1,
            validation_receipt_ids: vec![100, 101],
            promoted_at: Timestamp::from_nanos(1_234_567),
        };

        assert_eq!(promotion.cimage_entity, 7);
        assert_eq!(promotion.promotion_generation, 1);
        assert_eq!(promotion.validation_receipt_ids, vec![100, 101]);
    }

    #[test]
    fn cimage_promotion_serde_roundtrip() {
        let promotion = CimagePromotion {
            cimage_entity: 7,
            promotion_generation: 3,
            validation_receipt_ids: vec![100, 101, 102],
            promoted_at: Timestamp::from_nanos(9_999_999),
        };
        let json = serde_json::to_string(&promotion).expect("serialize");
        let back: CimagePromotion = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(promotion, back);
    }

    #[test]
    fn promote_cimage_preflight_rejects_missing_entity() {
        let world = World::new();
        let mut reg = SchemaRegistry::new();
        super::super::register_compilation_schemas(&mut reg);

        let cmd = PromoteCimageCommand {
            id: MessageId::compute(b"promote_missing"),
            cimage_entity: Entity::new(999, 0),
            receipt_ids: Vec::new(),
        };

        let result = cmd.preflight(&world, &reg);
        assert!(matches!(result, Err(CompilationError::JobNotFound(999))));
    }
}
