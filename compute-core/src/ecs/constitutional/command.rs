use crate::ecs::constitutional::types::*;
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

/// A receipt candidate — evidence of external work, awaiting validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptCandidate {
    pub id: MessageId,
    pub kind: String,
    pub payload: serde_json::Value,
    pub payload_hash: [u8; 32],
}
