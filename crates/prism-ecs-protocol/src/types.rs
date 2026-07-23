use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_NAME: &str = "prism.ecs.application";
pub const CURRENT_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };
pub const MAX_AGENT_LIST_LIMIT: u16 = 256;

/// Wire version for every request, event, error, and capability snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        CURRENT_PROTOCOL_VERSION
    }
}

/// A request from an application client to the Prism runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolRequest {
    pub protocol: String,
    pub version: ProtocolVersion,
    pub request_id: Uuid,
    pub body: RequestBody,
}

impl ProtocolRequest {
    pub fn new(request_id: Uuid, body: RequestBody) -> Self {
        Self {
            protocol: PROTOCOL_NAME.into(),
            version: CURRENT_PROTOCOL_VERSION,
            request_id,
            body,
        }
    }
}

/// The bounded first-slice application operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum RequestBody {
    GetCapabilities,
    GetHealth,
    ListAgents {
        limit: u16,
    },
    SpawnAgent {
        parent_id: u64,
        task: String,
        max_steps: u32,
        #[serde(default)]
        expected_world_epoch: Option<u64>,
    },
    CancelAgent {
        agent_id: u64,
        #[serde(default)]
        expected_world_epoch: Option<u64>,
    },
}

/// An event returned by the runtime boundary. Events are correlated to the
/// originating request and are safe to forward to a UI or another client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub protocol: String,
    pub version: ProtocolVersion,
    pub request_id: Uuid,
    pub body: EventBody,
}

impl Event {
    pub fn new(request_id: Uuid, body: EventBody) -> Self {
        Self {
            protocol: PROTOCOL_NAME.into(),
            version: CURRENT_PROTOCOL_VERSION,
            request_id,
            body,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum EventBody {
    Capabilities(CapabilitySet),
    Health(Health),
    Agents(Vec<Agent>),
    CommandCommitted(CommandReceipt),
    Error(ProtocolError),
}

/// Capabilities advertised by this protocol endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub version: ProtocolVersion,
    #[serde(rename = "type")]
    pub kind: String,
    pub capabilities: Vec<Capability>,
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self {
            version: CURRENT_PROTOCOL_VERSION,
            kind: "capabilities".into(),
            capabilities: vec![
                Capability::GetCapabilities,
                Capability::GetHealth,
                Capability::ListAgents,
                Capability::SpawnAgent,
                Capability::CancelAgent,
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    GetCapabilities,
    GetHealth,
    ListAgents,
    SpawnAgent,
    CancelAgent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub entity_count: usize,
    pub world_epoch: u64,
    pub journal_sequence: u64,
    pub receipt_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Agent {
    pub entity_id: u64,
    pub phase: String,
    pub lifecycle: String,
    pub parent_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReceipt {
    pub sequence: u64,
    pub world_epoch: u64,
    pub result: CommandResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum CommandResult {
    Spawned { entity_id: u64 },
    Cancelled { entity_id: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProtocolError {
    #[serde(rename = "type")]
    pub kind: String,
    pub protocol: String,
    pub version: ProtocolVersion,
    pub request_id: Uuid,
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

impl ProtocolError {
    pub fn new(
        request_id: Uuid,
        code: ErrorCode,
        message: impl Into<String>,
        retryable: bool,
    ) -> Self {
        Self {
            kind: "error".into(),
            protocol: PROTOCOL_NAME.into(),
            version: CURRENT_PROTOCOL_VERSION,
            request_id,
            code,
            message: message.into(),
            retryable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    UnsupportedProtocol,
    UnsupportedVersion,
    InvalidRequest,
    UnsupportedCapability,
    EpochMismatch,
    IdempotencyConflict,
    EntityNotFound,
    RuntimeFailure,
}
