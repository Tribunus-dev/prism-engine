//! Versioned wire types for the Prism ECS application boundary.
//!
//! This crate is deliberately independent of the runtime implementation. A
//! client, daemon transport, or Swift bridge can depend on these DTOs without
//! importing orchestration, persistence, or provider-specific code.

mod types;

pub use types::{
    Agent, Capability, CapabilitySet, CommandReceipt, CommandResult, ErrorCode, Event, EventBody,
    Health, MessageRecord, MessageRole, ProtocolError, ProtocolRequest, ProtocolVersion,
    RequestBody, ThreadStatus, ToolApproval, ToolApprovalDecision, ToolApprovalState,
    WorkflowEvent, WorkflowEventKind, WorkflowRecord, WorkflowSnapshot, CURRENT_PROTOCOL_VERSION,
    MAX_AGENT_LIST_LIMIT, MAX_WORKFLOW_MESSAGES, PROTOCOL_NAME,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_value};
    use uuid::Uuid;

    #[test]
    fn request_round_trip_preserves_version_and_tagged_body() {
        let original = ProtocolRequest::new(
            Uuid::from_u128(7),
            RequestBody::SpawnAgent {
                parent_id: 0,
                task: "summarize the workspace".into(),
                max_steps: 12,
                expected_world_epoch: Some(3),
            },
        );

        let encoded = to_value(&original).expect("request encodes");
        assert_eq!(encoded["protocol"], PROTOCOL_NAME);
        assert_eq!(encoded["version"]["major"], 1);
        assert_eq!(encoded["body"]["type"], "spawn_agent");

        let decoded: ProtocolRequest = serde_json::from_value(encoded).expect("request decodes");
        assert_eq!(decoded, original);
    }

    #[test]
    fn capability_snapshot_is_versioned_and_stable() {
        let capabilities = CapabilitySet::default();
        assert_eq!(capabilities.version, CURRENT_PROTOCOL_VERSION);
        assert_eq!(
            capabilities.capabilities,
            vec![
                Capability::GetCapabilities,
                Capability::GetHealth,
                Capability::ListAgents,
                Capability::SpawnAgent,
                Capability::CancelAgent,
            ]
        );

        let encoded = to_value(&capabilities).expect("capabilities encode");
        assert_eq!(encoded["type"], "capabilities");
        assert_eq!(encoded["capabilities"][0], "get_capabilities");
    }

    #[test]
    fn protocol_error_is_json_friendly() {
        let error = ProtocolError::new(
            Uuid::from_u128(15),
            ErrorCode::InvalidRequest,
            "task must not be empty",
            false,
        );
        let encoded = to_value(error).expect("error encodes");
        assert_eq!(encoded["type"], "error");
        assert_eq!(encoded["code"], "invalid_request");
        assert_eq!(
            encoded["request_id"],
            json!(Uuid::from_u128(15).to_string())
        );
    }

    #[test]
    fn workflow_events_replay_into_a_snapshot() {
        let thread_id = Uuid::from_u128(16);
        let approval_id = Uuid::from_u128(17);
        let events = vec![
            WorkflowEvent {
                thread_id,
                sequence: 1,
                kind: WorkflowEventKind::ThreadOpened,
            },
            WorkflowEvent {
                thread_id,
                sequence: 2,
                kind: WorkflowEventKind::MessageAppended {
                    message_id: Uuid::from_u128(18),
                    role: MessageRole::User,
                    content: "inspect the build".into(),
                },
            },
            WorkflowEvent {
                thread_id,
                sequence: 3,
                kind: WorkflowEventKind::ToolApprovalRequested {
                    approval_id,
                    tool_name: "repo_read".into(),
                    arguments: json!({"path":"Cargo.toml"}),
                },
            },
            WorkflowEvent {
                thread_id,
                sequence: 4,
                kind: WorkflowEventKind::ToolApprovalResolved {
                    approval_id,
                    decision: ToolApprovalDecision::Approve,
                },
            },
        ];

        let snapshot = WorkflowSnapshot::replay(thread_id, &events).expect("replay succeeds");
        assert_eq!(snapshot.revision, 4);
        assert_eq!(snapshot.messages.len(), 1);
        assert_eq!(snapshot.approvals[0].state, ToolApprovalState::Approved);
    }

    #[test]
    fn workflow_capabilities_are_additive_and_queryable() {
        let capabilities = CapabilitySet::workflow();
        assert!(capabilities.supports(Capability::RequestToolApproval));
        assert!(capabilities.supports(Capability::CancelThread));
        assert!(CapabilitySet::default().supports(Capability::CancelAgent));
        assert!(!CapabilitySet::default().supports(Capability::CancelThread));
    }
}
