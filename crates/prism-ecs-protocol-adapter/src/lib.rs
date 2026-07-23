//! Runtime-backed adapter for the provider-neutral Prism ECS application
//! protocol.
//!
//! The adapter owns no world, journal, scheduler, or persistence. It only
//! translates protocol DTOs to an existing `KernelHandle` and projects
//! committed runtime values back into protocol events.

use prism_ecs_protocol::{
    Agent, CapabilitySet, CommandReceipt, CommandResult, ErrorCode, Event, EventBody, Health,
    ProtocolError, ProtocolRequest, RequestBody, CURRENT_PROTOCOL_VERSION, MAX_AGENT_LIST_LIMIT,
    PROTOCOL_NAME,
};
use prism_ecs_runtime::{Command, CommandEnvelope, CommandResult as RuntimeCommandResult};
use prism_ecs_runtime::{KernelHandle, RuntimeError};
use uuid::Uuid;

/// Client-side protocol boundary implemented by a runtime-backed adapter.
pub trait ApplicationClient {
    fn send(&self, request: ProtocolRequest) -> Event;
}

/// Adapter from versioned application requests to the existing ECS kernel.
#[derive(Clone)]
pub struct RuntimeClient {
    handle: KernelHandle,
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
    use prism_ecs_protocol::{EventBody, ProtocolError, RequestBody};
    use prism_ecs_runtime::{Command, CommandEnvelope, RuntimeKernel};

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
}
