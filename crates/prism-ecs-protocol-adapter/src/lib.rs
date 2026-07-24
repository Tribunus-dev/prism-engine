//! Runtime-backed adapter for the provider-neutral Prism ECS application
//! protocol.
//!
//! The adapter owns no world, journal, scheduler, or persistence. It only
//! translates protocol DTOs to an existing `KernelHandle` and projects
//! committed runtime values back into protocol events.

use prism_ecs_protocol::{
    Agent, CapabilitySet, CommandReceipt, CommandResult, ErrorCode, Event, EventBody, Health,
    ProtocolError, ProtocolRequest, RequestBody, WorkflowEvent, WorkflowEventKind, WorkflowRecord,
    WorkflowSnapshot, CURRENT_PROTOCOL_VERSION, MAX_AGENT_LIST_LIMIT, PROTOCOL_NAME,
};
use prism_ecs_runtime::{Command, CommandEnvelope, CommandResult as RuntimeCommandResult};
use prism_ecs_runtime::{KernelHandle, RuntimeError};
use prism_mcp_core::ProjectionStore;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Client-side protocol boundary implemented by a runtime-backed adapter.
pub trait ApplicationClient: Send {
    fn send(&self, request: ProtocolRequest) -> Event;
}

/// Adapter from versioned application requests to the existing ECS kernel.
#[derive(Clone)]
pub struct RuntimeClient {
    handle: KernelHandle,
}

/// Persistence boundary for the Rust-owned conversation/workflow state.
///
/// Implementations may use the daemon's durable projection store. The
/// protocol adapter never owns ECS state and never writes directly to a
/// database.
pub trait WorkflowStore: Send + Sync {
    fn load(&self, thread_id: Uuid) -> Result<Option<WorkflowRecord>, String>;
    fn save(&self, record: &WorkflowRecord) -> Result<(), String>;
}

/// Runtime cancellation boundary. The daemon composition root can connect
/// this to its existing inference cancellation manager without importing that
/// runtime into the protocol contract crate.
pub trait WorkflowCancellation: Send + Sync {
    fn cancel(&self, thread_id: Uuid) -> Result<(), String>;
}

#[derive(Default)]
pub struct NoopWorkflowCancellation;

impl WorkflowCancellation for NoopWorkflowCancellation {
    fn cancel(&self, _thread_id: Uuid) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Default)]
pub struct InMemoryWorkflowStore {
    records: Mutex<HashMap<Uuid, WorkflowRecord>>,
}

impl WorkflowStore for InMemoryWorkflowStore {
    fn load(&self, thread_id: Uuid) -> Result<Option<WorkflowRecord>, String> {
        self.records
            .lock()
            .map_err(|_| "workflow store lock poisoned".to_string())
            .map(|records| records.get(&thread_id).cloned())
    }

    fn save(&self, record: &WorkflowRecord) -> Result<(), String> {
        let mut records = self
            .records
            .lock()
            .map_err(|_| "workflow store lock poisoned".to_string())?;
        records.insert(record.snapshot.thread_id, record.clone());
        Ok(())
    }
}

/// Adapter over the existing daemon projection storage seam. SQLite-backed
/// daemon profiles persist this record; a projection implementation that is
/// only in-memory remains intentionally non-durable.
pub struct ProjectionWorkflowStore {
    projection: Arc<dyn ProjectionStore>,
}

impl ProjectionWorkflowStore {
    pub fn new(projection: Arc<dyn ProjectionStore>) -> Self {
        Self { projection }
    }

    fn key(thread_id: Uuid) -> String {
        format!("prism.workflow.{thread_id}")
    }
}

impl WorkflowStore for ProjectionWorkflowStore {
    fn load(&self, thread_id: Uuid) -> Result<Option<WorkflowRecord>, String> {
        let value = match self.projection.get_trace(&Self::key(thread_id)) {
            Ok(value) => value,
            Err(error) if error.to_string().contains("no such table") => None,
            Err(error) => return Err(error.to_string()),
        };
        let Some(value) = value else {
            return Ok(None);
        };
        let record: WorkflowRecord = serde_json::from_value(value)
            .map_err(|error| format!("decode workflow record: {error}"))?;
        if record.snapshot.thread_id != thread_id {
            return Err("workflow record thread id does not match key".into());
        }
        Ok(Some(record))
    }

    fn save(&self, record: &WorkflowRecord) -> Result<(), String> {
        let value = serde_json::to_value(record)
            .map_err(|error| format!("encode workflow record: {error}"))?;
        self.projection
            .put_trace(&Self::key(record.snapshot.thread_id), &value)
            .map_err(|error| error.to_string())
    }
}

struct WorkflowService<S, C> {
    store: S,
    cancellation: C,
}

impl<S: WorkflowStore, C: WorkflowCancellation> WorkflowService<S, C> {
    fn load(&self, thread_id: Uuid) -> Result<WorkflowRecord, String> {
        let record = self
            .store
            .load(thread_id)?
            .ok_or_else(|| "thread not found".to_string())?;
        let replayed = WorkflowSnapshot::replay(thread_id, &record.events)?;
        if replayed != record.snapshot {
            return Err("workflow snapshot does not match replay log".into());
        }
        Ok(record)
    }

    fn save_event(
        &self,
        mut record: WorkflowRecord,
        expected_revision: Option<u64>,
        kind: WorkflowEventKind,
    ) -> Result<WorkflowEvent, String> {
        if let Some(expected) = expected_revision {
            if record.snapshot.revision != expected {
                return Err(format!(
                    "revision mismatch: expected {}, got {}",
                    expected, record.snapshot.revision
                ));
            }
        }
        let event = WorkflowEvent {
            thread_id: record.snapshot.thread_id,
            sequence: record.snapshot.revision + 1,
            kind,
        };
        let mut next_snapshot = record.snapshot.clone();
        next_snapshot.apply(&event)?;
        record.events.push(event.clone());
        record.snapshot = next_snapshot;
        self.store.save(&record)?;
        Ok(event)
    }

    fn open(&self, thread_id: Uuid) -> Result<WorkflowSnapshot, String> {
        if let Some(record) = self.store.load(thread_id)? {
            return Ok(record.snapshot);
        }
        let record = WorkflowRecord {
            snapshot: WorkflowSnapshot::new(thread_id),
            events: Vec::new(),
        };
        self.save_event(record, Some(0), WorkflowEventKind::ThreadOpened)?;
        Ok(self.load(thread_id)?.snapshot)
    }

    fn handle(&self, request_id: Uuid, body: RequestBody) -> EventBody {
        let result = match body {
            RequestBody::OpenThread { thread_id } => {
                self.open(thread_id).map(EventBody::WorkflowSnapshot)
            }
            RequestBody::GetThread { thread_id } => self
                .load(thread_id)
                .map(|record| EventBody::WorkflowSnapshot(record.snapshot)),
            RequestBody::AppendMessage {
                thread_id,
                role,
                content,
                expected_revision,
            } => self.load(thread_id).and_then(|record| {
                self.save_event(
                    record,
                    expected_revision,
                    WorkflowEventKind::MessageAppended {
                        message_id: Uuid::new_v4(),
                        role,
                        content,
                    },
                )
                .map(EventBody::WorkflowEvent)
            }),
            RequestBody::RequestToolApproval {
                thread_id,
                tool_name,
                arguments,
                expected_revision,
            } => self.load(thread_id).and_then(|record| {
                self.save_event(
                    record,
                    expected_revision,
                    WorkflowEventKind::ToolApprovalRequested {
                        approval_id: Uuid::new_v4(),
                        tool_name,
                        arguments,
                    },
                )
                .map(EventBody::WorkflowEvent)
            }),
            RequestBody::ResolveToolApproval {
                thread_id,
                approval_id,
                decision,
                expected_revision,
            } => self.load(thread_id).and_then(|record| {
                self.save_event(
                    record,
                    expected_revision,
                    WorkflowEventKind::ToolApprovalResolved {
                        approval_id,
                        decision,
                    },
                )
                .map(EventBody::WorkflowEvent)
            }),
            RequestBody::CancelThread {
                thread_id,
                expected_revision,
            } => self.load(thread_id).and_then(|record| {
                if let Some(expected) = expected_revision {
                    if record.snapshot.revision != expected {
                        return Err(format!(
                            "revision mismatch: expected {}, got {}",
                            expected, record.snapshot.revision
                        ));
                    }
                }
                if record.snapshot.status == prism_ecs_protocol::ThreadStatus::Cancelled {
                    return Err("thread is already cancelled".into());
                }
                self.cancellation.cancel(thread_id)?;
                self.save_event(record, None, WorkflowEventKind::ThreadCancelled)
                    .map(EventBody::WorkflowEvent)
            }),
            _ => Err("request is not a workflow operation".into()),
        };

        result.unwrap_or_else(|message| {
            let code = if message == "thread not found" {
                ErrorCode::ThreadNotFound
            } else if message.starts_with("revision mismatch") {
                ErrorCode::RevisionMismatch
            } else if message == "tool approval not found" {
                ErrorCode::ApprovalNotFound
            } else if message == "tool approval is already resolved" {
                ErrorCode::ApprovalNotPending
            } else if message.starts_with("cancellation") {
                ErrorCode::CancellationFailed
            } else {
                ErrorCode::InvalidRequest
            };
            EventBody::Error(ProtocolError::new(request_id, code, message, false))
        })
    }
}

/// Application client that combines the existing ECS runtime adapter with a
/// bounded Rust-owned workflow service.
pub struct WorkflowClient<S, C> {
    runtime: RuntimeClient,
    service: Arc<Mutex<WorkflowService<S, C>>>,
}

impl<S: WorkflowStore + 'static, C: WorkflowCancellation + 'static> WorkflowClient<S, C> {
    pub fn new(handle: KernelHandle, store: S, cancellation: C) -> Self {
        Self {
            runtime: RuntimeClient::new(handle),
            service: Arc::new(Mutex::new(WorkflowService {
                store,
                cancellation,
            })),
        }
    }

    pub fn runtime(&self) -> &RuntimeClient {
        &self.runtime
    }

    /// Append a runtime-owned execution observation to a user workflow.
    /// Scheduler/backend code calls this hook after it has committed a
    /// dispatch outcome; Swift only observes the resulting protocol event.
    pub fn publish_runtime_observation(
        &self,
        thread_id: Uuid,
        dispatch_id: u64,
        session_id: u64,
        model_id: String,
        modality: String,
        status: String,
        output_digest: Option<String>,
        output_units: u64,
    ) -> Result<WorkflowEvent, String> {
        let service = self
            .service
            .lock()
            .map_err(|_| "workflow service lock poisoned".to_string())?;
        let record = service.load(thread_id)?;
        service.save_event(
            record,
            None,
            WorkflowEventKind::RuntimeObservation {
                dispatch_id,
                session_id,
                model_id,
                modality,
                status,
                output_digest,
                output_units,
            },
        )
    }
}

impl<S: WorkflowStore + 'static, C: WorkflowCancellation + 'static> ApplicationClient
    for WorkflowClient<S, C>
{
    fn send(&self, request: ProtocolRequest) -> Event {
        let request_id = request.request_id;
        if matches!(
            request.body,
            RequestBody::OpenThread { .. }
                | RequestBody::GetThread { .. }
                | RequestBody::AppendMessage { .. }
                | RequestBody::RequestToolApproval { .. }
                | RequestBody::ResolveToolApproval { .. }
                | RequestBody::CancelThread { .. }
        ) {
            if request.protocol != PROTOCOL_NAME {
                return Event::new(
                    request_id,
                    EventBody::Error(ProtocolError::new(
                        request_id,
                        ErrorCode::UnsupportedProtocol,
                        format!("unsupported protocol: {}", request.protocol),
                        false,
                    )),
                );
            }
            if request.version.major != CURRENT_PROTOCOL_VERSION.major
                || request.version.minor > CURRENT_PROTOCOL_VERSION.minor
            {
                return Event::new(
                    request_id,
                    EventBody::Error(ProtocolError::new(
                        request_id,
                        ErrorCode::UnsupportedVersion,
                        format!(
                            "unsupported protocol version {}.{}",
                            request.version.major, request.version.minor
                        ),
                        false,
                    )),
                );
            }
            let body = self
                .service
                .lock()
                .map(|service| service.handle(request_id, request.body))
                .unwrap_or_else(|_| {
                    EventBody::Error(ProtocolError::new(
                        request_id,
                        ErrorCode::RuntimeFailure,
                        "workflow service lock poisoned",
                        true,
                    ))
                });
            return Event::new(request_id, body);
        }

        if matches!(request.body, RequestBody::GetCapabilities) {
            return Event::new(
                request_id,
                EventBody::Capabilities(CapabilitySet::workflow()),
            );
        }
        self.runtime.send(request)
    }
}

impl RuntimeClient {
    pub fn new(handle: KernelHandle) -> Self {
        Self { handle }
    }

    pub fn handle(&self) -> &KernelHandle {
        &self.handle
    }

    fn command(
        &self,
        request_id: Uuid,
        expected_world_epoch: Option<u64>,
        command: Command,
    ) -> EventBody {
        let mut envelope = CommandEnvelope::new(command);
        envelope.idempotency_key = request_id;
        envelope.correlation_id = request_id.to_string();
        envelope.expected_epoch = expected_world_epoch;

        match self.handle.submit(envelope) {
            Ok(outcome) => match outcome.result {
                RuntimeCommandResult::Spawned { entity_id } => {
                    EventBody::CommandCommitted(CommandReceipt {
                        sequence: outcome.sequence,
                        world_epoch: outcome.world_epoch,
                        result: CommandResult::Spawned { entity_id },
                    })
                }
                RuntimeCommandResult::Cancelled { entity_id } => {
                    EventBody::CommandCommitted(CommandReceipt {
                        sequence: outcome.sequence,
                        world_epoch: outcome.world_epoch,
                        result: CommandResult::Cancelled { entity_id },
                    })
                }
                other => EventBody::Error(ProtocolError::new(
                    request_id,
                    ErrorCode::RuntimeFailure,
                    format!("runtime returned an unsupported command result: {other:?}"),
                    false,
                )),
            },
            Err(error) => EventBody::Error(map_runtime_error(request_id, error)),
        }
    }
}

impl ApplicationClient for RuntimeClient {
    fn send(&self, request: ProtocolRequest) -> Event {
        let request_id = request.request_id;
        if request.protocol != PROTOCOL_NAME {
            return Event::new(
                request_id,
                EventBody::Error(ProtocolError::new(
                    request_id,
                    ErrorCode::UnsupportedProtocol,
                    format!("unsupported protocol: {}", request.protocol),
                    false,
                )),
            );
        }
        if request.version.major != CURRENT_PROTOCOL_VERSION.major
            || request.version.minor > CURRENT_PROTOCOL_VERSION.minor
        {
            return Event::new(
                request_id,
                EventBody::Error(ProtocolError::new(
                    request_id,
                    ErrorCode::UnsupportedVersion,
                    format!(
                        "unsupported protocol version {}.{}",
                        request.version.major, request.version.minor
                    ),
                    false,
                )),
            );
        }

        let body = match request.body {
            RequestBody::GetCapabilities => EventBody::Capabilities(CapabilitySet::default()),
            RequestBody::GetHealth => {
                let health = self.handle.health();
                EventBody::Health(Health {
                    status: health.status,
                    entity_count: health.entity_count,
                    world_epoch: health.world_epoch,
                    journal_sequence: health.journal_sequence,
                    receipt_sequence: health.receipt_sequence,
                })
            }
            RequestBody::ListAgents { limit } => EventBody::Agents(
                self.handle
                    .query_agents()
                    .into_iter()
                    .take(limit.clamp(1, MAX_AGENT_LIST_LIMIT) as usize)
                    .map(|agent| Agent {
                        entity_id: agent.entity_id,
                        phase: agent.phase,
                        lifecycle: agent.lifecycle,
                        parent_id: agent.parent_id,
                    })
                    .collect(),
            ),
            RequestBody::SpawnAgent {
                parent_id,
                task,
                max_steps,
                expected_world_epoch,
            } => {
                if task.trim().is_empty() || max_steps == 0 {
                    EventBody::Error(ProtocolError::new(
                        request_id,
                        ErrorCode::InvalidRequest,
                        "spawn_agent requires a non-empty task and max_steps greater than zero",
                        false,
                    ))
                } else {
                    self.command(
                        request_id,
                        expected_world_epoch,
                        Command::SpawnAgent {
                            parent_id,
                            task,
                            max_steps,
                        },
                    )
                }
            }
            RequestBody::CancelAgent {
                agent_id,
                expected_world_epoch,
            } => self.command(
                request_id,
                expected_world_epoch,
                Command::CancelAgent { agent_id },
            ),
            _ => EventBody::Error(ProtocolError::new(
                request_id,
                ErrorCode::UnsupportedCapability,
                "workflow request requires WorkflowClient",
                false,
            )),
        };

        Event::new(request_id, body)
    }
}

fn map_runtime_error(request_id: Uuid, error: RuntimeError) -> ProtocolError {
    let (code, retryable) = match error {
        RuntimeError::EpochMismatch { .. } => (ErrorCode::EpochMismatch, false),
        RuntimeError::IdempotencyConflict(_) => (ErrorCode::IdempotencyConflict, true),
        RuntimeError::Entity(_) => (ErrorCode::EntityNotFound, false),
        RuntimeError::UnknownCommand(_) => (ErrorCode::InvalidRequest, false),
        RuntimeError::Journal(_)
        | RuntimeError::Receipt(_)
        | RuntimeError::Io(_)
        | RuntimeError::Lease(_)
        | RuntimeError::Dispatch(_) => (ErrorCode::RuntimeFailure, true),
    };
    ProtocolError::new(request_id, code, error.to_string(), retryable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_protocol::{
        Capability, EventBody, ProtocolError, RequestBody, ToolApprovalDecision, ToolApprovalState,
    };
    use prism_ecs_runtime::{Command, CommandEnvelope, RuntimeKernel};
    use std::sync::Mutex;

    fn request(request_id: Uuid, body: RequestBody) -> ProtocolRequest {
        ProtocolRequest::new(request_id, body)
    }

    #[test]
    fn runtime_adapter_projects_commands_and_idempotent_replays() {
        let client = RuntimeClient::new(RuntimeKernel::new().handle());
        let request_id = Uuid::from_u128(11);
        let request = request(
            request_id,
            RequestBody::SpawnAgent {
                parent_id: 0,
                task: "bounded task".into(),
                max_steps: 4,
                expected_world_epoch: None,
            },
        );

        let response = client.send(request.clone());
        match response.body {
            EventBody::CommandCommitted(ref receipt) => {
                assert_eq!(response.request_id, request_id);
                assert_eq!(receipt.sequence, 1);
                assert_eq!(receipt.world_epoch, 0);
                assert_eq!(receipt.result, CommandResult::Spawned { entity_id: 1 });
            }
            other => panic!("expected command event, got {other:?}"),
        }
        assert_eq!(client.send(request), response);
    }

    #[test]
    fn runtime_adapter_projects_reads_without_exposing_runtime_types() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();
        handle
            .submit(CommandEnvelope::new(Command::SpawnAgent {
                parent_id: 0,
                task: "visible".into(),
                max_steps: 1,
            }))
            .expect("seed agent");
        let client = RuntimeClient::new(handle);

        let response = client.send(request(
            Uuid::from_u128(12),
            RequestBody::ListAgents { limit: 8 },
        ));
        match response.body {
            EventBody::Agents(agents) => {
                assert_eq!(agents.len(), 1);
                assert_eq!(agents[0].entity_id, 1);
            }
            other => panic!("expected agents event, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_and_runtime_errors_are_typed_events() {
        let client = RuntimeClient::new(RuntimeKernel::new().handle());
        let mut versioned_request = request(Uuid::from_u128(13), RequestBody::GetHealth);
        versioned_request.version.major = CURRENT_PROTOCOL_VERSION.major + 1;

        let response = client.send(versioned_request);
        assert!(matches!(
            response.body,
            EventBody::Error(ProtocolError {
                code: ErrorCode::UnsupportedVersion,
                retryable: false,
                ..
            })
        ));

        let response = client.send(request(
            Uuid::from_u128(14),
            RequestBody::CancelAgent {
                agent_id: 99,
                expected_world_epoch: Some(42),
            },
        ));
        match response.body {
            EventBody::Error(error) => {
                assert_eq!(error.code, ErrorCode::EpochMismatch);
                assert!(!error.retryable);
            }
            other => panic!("expected error event, got {other:?}"),
        }
    }

    struct RecordingCancellation(Arc<Mutex<Vec<Uuid>>>);

    impl WorkflowCancellation for RecordingCancellation {
        fn cancel(&self, thread_id: Uuid) -> Result<(), String> {
            self.0.lock().unwrap().push(thread_id);
            Ok(())
        }
    }

    #[test]
    fn workflow_client_persists_replayable_messages_and_approval_state() {
        let thread_id = Uuid::from_u128(21);
        let client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            InMemoryWorkflowStore::default(),
            RecordingCancellation(Arc::new(Mutex::new(Vec::new()))),
        );

        let response = client.send(request(
            Uuid::from_u128(22),
            RequestBody::OpenThread { thread_id },
        ));
        let snapshot = match response.body {
            EventBody::WorkflowSnapshot(snapshot) => snapshot,
            other => panic!("expected workflow snapshot, got {other:?}"),
        };
        assert_eq!(snapshot.revision, 1);

        let response = client.send(request(
            Uuid::from_u128(23),
            RequestBody::AppendMessage {
                thread_id,
                role: prism_ecs_protocol::MessageRole::User,
                content: "run the safe read".into(),
                expected_revision: Some(1),
            },
        ));
        assert!(matches!(response.body, EventBody::WorkflowEvent(_)));

        let response = client.send(request(
            Uuid::from_u128(24),
            RequestBody::RequestToolApproval {
                thread_id,
                tool_name: "repo_read".into(),
                arguments: serde_json::json!({"path":"Cargo.toml"}),
                expected_revision: Some(2),
            },
        ));
        let approval_id = match response.body {
            EventBody::WorkflowEvent(event) => match event.kind {
                WorkflowEventKind::ToolApprovalRequested { approval_id, .. } => approval_id,
                other => panic!("expected approval request, got {other:?}"),
            },
            other => panic!("expected workflow event, got {other:?}"),
        };

        let response = client.send(request(
            Uuid::from_u128(25),
            RequestBody::ResolveToolApproval {
                thread_id,
                approval_id,
                decision: ToolApprovalDecision::Approve,
                expected_revision: Some(3),
            },
        ));
        assert!(matches!(response.body, EventBody::WorkflowEvent(_)));

        let response = client.send(request(
            Uuid::from_u128(26),
            RequestBody::GetThread { thread_id },
        ));
        match response.body {
            EventBody::WorkflowSnapshot(snapshot) => {
                assert_eq!(snapshot.revision, 4);
                assert_eq!(snapshot.messages.len(), 1);
                assert_eq!(snapshot.approvals[0].state, ToolApprovalState::Approved);
            }
            other => panic!("expected workflow snapshot, got {other:?}"),
        }
    }

    #[test]
    fn workflow_client_fences_stale_writes_and_calls_runtime_cancellation() {
        let thread_id = Uuid::from_u128(31);
        let cancellation = RecordingCancellation(Arc::new(Mutex::new(Vec::new())));
        let calls = cancellation.0.clone();
        let client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            InMemoryWorkflowStore::default(),
            cancellation,
        );
        client.send(request(
            Uuid::from_u128(32),
            RequestBody::OpenThread { thread_id },
        ));
        let stale = client.send(request(
            Uuid::from_u128(33),
            RequestBody::AppendMessage {
                thread_id,
                role: prism_ecs_protocol::MessageRole::User,
                content: "stale".into(),
                expected_revision: Some(0),
            },
        ));
        assert!(matches!(
            stale.body,
            EventBody::Error(ProtocolError {
                code: ErrorCode::RevisionMismatch,
                ..
            })
        ));

        let cancelled = client.send(request(
            Uuid::from_u128(34),
            RequestBody::CancelThread {
                thread_id,
                expected_revision: Some(1),
            },
        ));
        assert!(matches!(cancelled.body, EventBody::WorkflowEvent(_)));
        assert_eq!(calls.lock().unwrap().as_slice(), &[thread_id]);
    }

    #[test]
    fn workflow_client_publishes_ecs_runtime_observation_with_session_identity() {
        let thread_id = Uuid::from_u128(35);
        let client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            InMemoryWorkflowStore::default(),
            NoopWorkflowCancellation,
        );
        client.send(request(
            Uuid::from_u128(36),
            RequestBody::OpenThread { thread_id },
        ));
        let event = client
            .publish_runtime_observation(
                thread_id,
                7,
                19,
                "audio/decoder".into(),
                "audio".into(),
                "completed".into(),
                Some("digest".into()),
                24_000,
            )
            .unwrap();
        match event.kind {
            WorkflowEventKind::RuntimeObservation {
                dispatch_id,
                session_id,
                model_id,
                output_units,
                ..
            } => {
                assert_eq!(dispatch_id, 7);
                assert_eq!(session_id, 19);
                assert_eq!(model_id, "audio/decoder");
                assert_eq!(output_units, 24_000);
            }
            other => panic!("expected runtime observation, got {other:?}"),
        }
    }

    #[test]
    fn projection_workflow_store_round_trips_through_daemon_storage_seam() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let db = Arc::new(
            prism_mcp_core::DbManager::open(&directory.path().join("workflow.db"), "", 2)
                .expect("open sqlite projection store"),
        );
        let thread_id = Uuid::from_u128(41);
        let client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            ProjectionWorkflowStore::new(db.clone()),
            NoopWorkflowCancellation,
        );
        client.send(request(
            Uuid::from_u128(42),
            RequestBody::OpenThread { thread_id },
        ));
        client.send(request(
            Uuid::from_u128(43),
            RequestBody::AppendMessage {
                thread_id,
                role: prism_ecs_protocol::MessageRole::Assistant,
                content: "persisted".into(),
                expected_revision: Some(1),
            },
        ));

        let second_client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            ProjectionWorkflowStore::new(db),
            NoopWorkflowCancellation,
        );
        let response = second_client.send(request(
            Uuid::from_u128(44),
            RequestBody::GetThread { thread_id },
        ));
        match response.body {
            EventBody::WorkflowSnapshot(snapshot) => {
                assert_eq!(snapshot.revision, 2);
                assert_eq!(snapshot.messages[0].content, "persisted");
            }
            other => panic!("expected persisted snapshot, got {other:?}"),
        }
    }

    #[test]
    fn workflow_client_advertises_only_workflow_capabilities_for_workflow_requests() {
        let client = WorkflowClient::new(
            RuntimeKernel::new().handle(),
            InMemoryWorkflowStore::default(),
            NoopWorkflowCancellation,
        );
        let response = client.send(request(Uuid::from_u128(51), RequestBody::GetCapabilities));
        match response.body {
            EventBody::Capabilities(capabilities) => {
                assert!(capabilities.supports(Capability::AppendMessage));
                assert!(capabilities.supports(Capability::CancelThread));
            }
            other => panic!("expected capability snapshot, got {other:?}"),
        }
    }
}
