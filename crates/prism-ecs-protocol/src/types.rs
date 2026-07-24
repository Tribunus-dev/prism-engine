use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const PROTOCOL_NAME: &str = "prism.ecs.application";
pub const CURRENT_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };
pub const MAX_AGENT_LIST_LIMIT: u16 = 256;
pub const MAX_WORKFLOW_MESSAGES: usize = 1024;

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
    OpenThread {
        thread_id: Uuid,
    },
    GetThread {
        thread_id: Uuid,
    },
    AppendMessage {
        thread_id: Uuid,
        role: MessageRole,
        content: String,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
    RequestToolApproval {
        thread_id: Uuid,
        tool_name: String,
        arguments: serde_json::Value,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
    ResolveToolApproval {
        thread_id: Uuid,
        approval_id: Uuid,
        decision: ToolApprovalDecision,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
    CancelThread {
        thread_id: Uuid,
        #[serde(default)]
        expected_revision: Option<u64>,
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
    WorkflowSnapshot(WorkflowSnapshot),
    WorkflowEvent(WorkflowEvent),
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

impl CapabilitySet {
    pub fn workflow() -> Self {
        let mut capabilities = Self::default().capabilities;
        capabilities.extend([
            Capability::OpenThread,
            Capability::GetThread,
            Capability::AppendMessage,
            Capability::RequestToolApproval,
            Capability::ResolveToolApproval,
            Capability::CancelThread,
        ]);
        Self {
            version: CURRENT_PROTOCOL_VERSION,
            kind: "capabilities".into(),
            capabilities,
        }
    }

    pub fn supports(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
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
    OpenThread,
    GetThread,
    AppendMessage,
    RequestToolApproval,
    ResolveToolApproval,
    CancelThread,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatus {
    Active,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolApprovalDecision {
    Approve,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolApprovalState {
    Pending,
    Approved,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageRecord {
    pub message_id: Uuid,
    pub sequence: u64,
    pub role: MessageRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolApproval {
    pub approval_id: Uuid,
    pub sequence: u64,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub state: ToolApprovalState,
    pub resolved_sequence: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowSnapshot {
    pub thread_id: Uuid,
    pub revision: u64,
    pub status: ThreadStatus,
    pub messages: Vec<MessageRecord>,
    pub approvals: Vec<ToolApproval>,
}

impl WorkflowSnapshot {
    pub fn new(thread_id: Uuid) -> Self {
        Self {
            thread_id,
            revision: 0,
            status: ThreadStatus::Active,
            messages: Vec::new(),
            approvals: Vec::new(),
        }
    }

    pub fn apply(&mut self, event: &WorkflowEvent) -> Result<(), String> {
        if event.sequence != self.revision + 1 {
            return Err(format!(
                "workflow sequence gap: expected {}, got {}",
                self.revision + 1,
                event.sequence
            ));
        }
        match &event.kind {
            WorkflowEventKind::ThreadOpened => {
                if self.revision != 0 || self.thread_id != event.thread_id {
                    return Err("thread opened more than once".into());
                }
            }
            WorkflowEventKind::MessageAppended {
                message_id,
                role,
                content,
            } => {
                if self.status == ThreadStatus::Cancelled {
                    return Err("thread is cancelled".into());
                }
                if content.trim().is_empty() {
                    return Err("message content must not be empty".into());
                }
                if self.messages.len() >= MAX_WORKFLOW_MESSAGES {
                    return Err("workflow message limit exceeded".into());
                }
                self.messages.push(MessageRecord {
                    message_id: *message_id,
                    sequence: event.sequence,
                    role: *role,
                    content: content.clone(),
                });
            }
            WorkflowEventKind::ToolApprovalRequested {
                approval_id,
                tool_name,
                arguments,
            } => {
                if self.status == ThreadStatus::Cancelled {
                    return Err("thread is cancelled".into());
                }
                if tool_name.trim().is_empty() || !arguments.is_object() {
                    return Err("tool approval requires a name and object arguments".into());
                }
                self.approvals.push(ToolApproval {
                    approval_id: *approval_id,
                    sequence: event.sequence,
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    state: ToolApprovalState::Pending,
                    resolved_sequence: None,
                });
            }
            WorkflowEventKind::ToolApprovalResolved {
                approval_id,
                decision,
            } => {
                let approval = self
                    .approvals
                    .iter_mut()
                    .find(|approval| approval.approval_id == *approval_id)
                    .ok_or_else(|| "tool approval not found".to_string())?;
                if approval.state != ToolApprovalState::Pending {
                    return Err("tool approval is already resolved".into());
                }
                approval.state = match decision {
                    ToolApprovalDecision::Approve => ToolApprovalState::Approved,
                    ToolApprovalDecision::Deny => ToolApprovalState::Denied,
                };
                approval.resolved_sequence = Some(event.sequence);
            }
            WorkflowEventKind::RuntimeObservation { .. } => {
                // Runtime observations are provenance events. They do not
                // mutate the user-facing thread projection, but they remain
                // sequence-checked and replayable in the workflow log.
            }
            WorkflowEventKind::ThreadCancelled => {
                if self.status == ThreadStatus::Cancelled {
                    return Err("thread is already cancelled".into());
                }
                self.status = ThreadStatus::Cancelled;
                for approval in &mut self.approvals {
                    if approval.state == ToolApprovalState::Pending {
                        approval.state = ToolApprovalState::Denied;
                        approval.resolved_sequence = Some(event.sequence);
                    }
                }
            }
        }
        self.revision = event.sequence;
        Ok(())
    }

    pub fn replay(thread_id: Uuid, events: &[WorkflowEvent]) -> Result<Self, String> {
        let mut snapshot = Self::new(thread_id);
        for event in events {
            if event.thread_id != thread_id {
                return Err("workflow event belongs to another thread".into());
            }
            snapshot.apply(event)?;
        }
        Ok(snapshot)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowEvent {
    pub thread_id: Uuid,
    pub sequence: u64,
    pub kind: WorkflowEventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum WorkflowEventKind {
    ThreadOpened,
    MessageAppended {
        message_id: Uuid,
        role: MessageRole,
        content: String,
    },
    ToolApprovalRequested {
        approval_id: Uuid,
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolApprovalResolved {
        approval_id: Uuid,
        decision: ToolApprovalDecision,
    },
    /// Runtime-owned execution observation forwarded to application clients.
    RuntimeObservation {
        dispatch_id: u64,
        session_id: u64,
        model_id: String,
        modality: String,
        status: String,
        output_digest: Option<String>,
        output_units: u64,
    },
    ThreadCancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowRecord {
    pub snapshot: WorkflowSnapshot,
    pub events: Vec<WorkflowEvent>,
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
    ThreadNotFound,
    RevisionMismatch,
    ApprovalNotFound,
    ApprovalNotPending,
    CancellationFailed,
    RuntimeFailure,
}
