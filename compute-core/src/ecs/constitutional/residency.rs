use crate::ecs::constitutional::artifact::ArtifactDigest;
use crate::ecs::constitutional::command::{DomainEvent, EffectOutcome, EffectRequest};
use crate::ecs::constitutional::device::DeviceStableId;
use crate::ecs::constitutional::lifecycle::ResidencyLifecycle;
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::CompWorld;
use serde::{Deserialize, Serialize};

// ── Model Components ─────────────────────────────────────────────────────

/// A model entity represents a deployable model derived from a canonical artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelId(pub DomainId);

/// Reference to the artifact this model was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactRef {
    pub artifact_id: u64,
    pub digest: ArtifactDigest,
}

/// Model lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelLifecycle {
    Created,
    Validated,
    Deployable,
    Deprecated,
    Removed,
}

impl ModelLifecycle {
    pub fn can_transition_to(&self, target: Self) -> bool {
        match (*self, target) {
            (Self::Created, Self::Validated)
            | (Self::Validated, Self::Deployable)
            | (Self::Deployable, Self::Deprecated)
            | (Self::Deprecated, Self::Removed) => true,
            _ => false,
        }
    }
}

// ── Residency Components ─────────────────────────────────────────────────

/// References the device this residency is placed on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyDeviceRef {
    pub device_id: u64,
    pub device_stable_id: DeviceStableId,
}

/// Memory allocated for this residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyMemoryClaim {
    pub bytes: u64,
    pub allocated: bool, // false = desired, true = actually allocated
}

/// Representation format of the resident weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyFormat {
    Native,
    Quantized,
    Distilled,
}

/// Ephemeral allocation key — NOT durable.
/// References the execution-plane allocation (Metal buffer, CUDA alloc, mmap, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocationToken(pub String);

impl AllocationToken {
    pub fn ephemeral_durability() -> ComponentDurability {
        ComponentDurability::Ephemeral
    }
}

// ── DeployModelCommand ───────────────────────────────────────────────────

/// Command to deploy a model from a canonical artifact onto a device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeployModelCommand {
    pub id: MessageId,
    pub artifact_entity: u64,
    pub device_entity: u64,
    pub device_stable_id: DeviceStableId,
    pub format: ResidencyFormat,
    pub memory_bytes: u64,
}

impl DeployModelCommand {
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("load-model:{}", self.artifact_entity).as_bytes()),
            kind: crate::ecs::constitutional::command::EffectKind::LoadFile,
            params: serde_json::json!({
                "artifact": self.artifact_entity,
                "device": self.device_entity,
                "memory": self.memory_bytes,
            }),
        }
    }

    /// Execute deployment: create model entity + residency entity atomically.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
        outcome: EffectOutcome,
    ) -> Result<(CommittedEpoch, DomainEvent), DeploymentError> {
        let _ = schema_registry; // schema verification placeholder
        if !outcome.success {
            return Err(DeploymentError::EffectFailed);
        }

        // Validate outcome request ID matches the effect request
        let expected_request_id = self.to_effect_request().id;
        if outcome.request_id != expected_request_id {
            return Err(DeploymentError::RequestMismatch);
        }

        // Reserve entity IDs for model + residency
        let model_id = WorldTxn::next_entity_id(world);
        // next_entity_id is a read-only peek, so we manually advance for the second ID
        let residency_id = model_id + 1;

        let mut txn = WorldTxn::new(world);

        // Spawn model entity
        txn.stage_spawn(model_id, crate::ecs::EntityKind::Model);

        // Spawn residency entity
        txn.stage_spawn(residency_id, crate::ecs::EntityKind::Residency);

        // Emit domain event
        let event = DomainEvent {
            id: self.id,
            kind: "model_deployed".to_string(),
            entity_id: None,
            payload: serde_json::json!({
                "model_id": model_id,
                "residency_id": residency_id,
                "device": self.device_entity,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(DeploymentError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeploymentError {
    #[error("deployment effect failed")]
    EffectFailed,
    #[error("request ID mismatch")]
    RequestMismatch,
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Component Trait impls ─────────────────────────────────────────────────

impl crate::ecs::Component for ModelId {}
impl crate::ecs::Component for ModelArtifactRef {}
impl crate::ecs::Component for ModelLifecycle {}
impl crate::ecs::Component for ResidencyDeviceRef {}
impl crate::ecs::Component for ResidencyMemoryClaim {}
impl crate::ecs::Component for ResidencyLifecycle {}
impl crate::ecs::Component for ResidencyFormat {}
impl crate::ecs::Component for AllocationToken {}
