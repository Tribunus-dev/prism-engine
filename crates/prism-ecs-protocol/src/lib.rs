//! Versioned wire types for the Prism ECS application boundary.
//!
//! This crate is deliberately independent of the runtime implementation. A
//! client, daemon transport, or Swift bridge can depend on these DTOs without
//! importing orchestration, persistence, or provider-specific code.

mod types;

pub use types::{
    Agent, Capability, CapabilitySet, CommandReceipt, CommandResult, ErrorCode, Event, EventBody,
    Health, ProtocolError, ProtocolRequest, ProtocolVersion, RequestBody, CURRENT_PROTOCOL_VERSION,
    MAX_AGENT_LIST_LIMIT, PROTOCOL_NAME,
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
}
