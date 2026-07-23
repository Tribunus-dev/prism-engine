use crate::types::*;
use serde::{Deserialize, Serialize};

/// A command is a requested intent — queued, validated, and either committed or rejected.
/// Commands do not mutate the world until a system processes them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub id: MessageId,
    pub target_domain: DomainId,
    pub payload: serde_json::Value, // type-erased command body
}

/// An effect request — ask the execution plane to perform an external operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectRequest {
    pub id: MessageId,
    pub kind: EffectKind,
    pub params: serde_json::Value,
}

/// Kinds of effects the execution plane can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectKind {
    LoadFile,
    MapMemory,
    CreateDevice,
    EnumerateDrivers,
    CompileModel,
    RunInference,
    AcquireLease,
    ReleaseHandle,
    Custom(u64),
}

/// The outcome of executing an effect — untrusted until validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectOutcome {
    pub id: MessageId,
    pub request_id: MessageId,
    pub success: bool,
    pub output: serde_json::Value,
}

/// A domain event — committed ECS fact, emitted atomically after a world transaction commits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainEvent {
    pub id: MessageId,
    pub kind: String,
    pub entity_id: Option<EntityKindId>,
    pub payload: serde_json::Value,
}

/// The durability contract for an event emitted by an ECS transaction.
///
/// Durable events are facts that may be appended to
/// [`crate::persistence::EventStore`] and used
/// during replay. Advisory events are observations for the current runtime;
/// they are intentionally kept out of the durable event log and therefore
/// cannot become a second source of world authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventDurability {
    Durable,
    Advisory,
}

/// A runtime-only observation emitted at the same transaction boundary as a
/// durable domain event. It is never accepted by the durable event store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvisoryEvent {
    pub id: MessageId,
    pub kind: String,
    pub entity_id: Option<EntityKindId>,
    pub payload: serde_json::Value,
}

impl AdvisoryEvent {
    pub fn new(
        kind: impl Into<String>,
        entity_id: Option<EntityKindId>,
        payload: serde_json::Value,
    ) -> Self {
        let kind = kind.into();
        let id = MessageId::compute(
            serde_json::to_string(&(kind.as_str(), entity_id, &payload))
                .unwrap_or_default()
                .as_bytes(),
        );
        Self {
            id,
            kind,
            entity_id,
            payload,
        }
    }

    pub const fn durability(&self) -> EventDurability {
        EventDurability::Advisory
    }
}

/// A typed event boundary for code that needs to carry both event classes.
/// Persistence APIs accept [`DomainEvent`] directly, making accidental
/// advisory-event persistence a type-level mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifiedEvent {
    Durable(DomainEvent),
    Advisory(AdvisoryEvent),
}

impl ClassifiedEvent {
    pub const fn durability(&self) -> EventDurability {
        match self {
            Self::Durable(_) => EventDurability::Durable,
            Self::Advisory(_) => EventDurability::Advisory,
        }
    }
}

/// A receipt candidate — evidence of external work, awaiting validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptCandidate {
    pub id: MessageId,
    pub kind: String,
    pub payload: serde_json::Value,
    pub payload_hash: [u8; 32],
}
