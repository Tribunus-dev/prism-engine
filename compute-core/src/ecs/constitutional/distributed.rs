use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use crate::ecs::{CompEntity, CompWorld, EntityKind};
use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════════════════════
// Component Schema IDs (55–62)
// ══════════════════════════════════════════════════════════════════════════════

pub const SCHEMA_PEER_IDENTITY: u64 = 55;
pub const SCHEMA_NODE_MEMBERSHIP: u64 = 56;
pub const SCHEMA_PEER_CAPABILITIES: u64 = 57;
pub const SCHEMA_NODE_TOPOLOGY: u64 = 58;
pub const SCHEMA_TRUST_STATE: u64 = 59;
pub const SCHEMA_WORKER_HEALTH: u64 = 60;
pub const SCHEMA_REMOTE_LEASE: u64 = 61;
pub const SCHEMA_REMOTE_CAPABILITY_OBSERVATION: u64 = 62;

// ══════════════════════════════════════════════════════════════════════════════
// Component Types
// ══════════════════════════════════════════════════════════════════════════════

/// Cryptographic peer identity — stable domain identity for a node.
/// Connections and sockets remain ephemeral; this identity persists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerIdentity {
    pub peer_id: String,
    pub public_key: Vec<u8>,
    pub discovered_at: Timestamp,
}

/// Cluster membership — which cluster a node belongs to and when it joined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeMembership {
    pub node_id: u64,
    pub cluster_name: String,
    pub joined_epoch: WorldEpoch,
    pub last_seen: Timestamp,
}

/// Claimed capabilities list — OBSERVED, not trusted until validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerCapabilities(pub Vec<String>);

/// Topology information for a node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeTopology {
    pub node_entity: u64,
    pub role: String,
    pub region: String,
    pub instance_type: String,
}

/// Trust level after validation of a peer's identity and capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrustState {
    Untrusted,
    Observed,
    Validated,
    Trusted,
    Revoked,
}

impl TrustState {
    /// Attempt a forward transition. Returns Err if the transition is invalid.
    pub fn transition_to(&self, target: TrustState) -> Result<TrustState, String> {
        match (self, target) {
            // Forward: monotonic progression
            (TrustState::Untrusted, TrustState::Observed)
            | (TrustState::Observed, TrustState::Validated)
            | (TrustState::Validated, TrustState::Trusted) => Ok(target),
            // Revocation is always allowed from any pre-revocation state
            (TrustState::Untrusted, TrustState::Revoked)
            | (TrustState::Observed, TrustState::Revoked)
            | (TrustState::Validated, TrustState::Revoked)
            | (TrustState::Trusted, TrustState::Revoked) => Ok(target),
            // No backwards transitions
            _ => Err(format!(
                "invalid trust state transition: {:?} -> {:?}",
                self, target
            )),
        }
    }
}

/// Observation-based worker health — not authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerHealth {
    Healthy,
    Degraded(String),
    Unreachable(Timestamp),
    Removed,
}

/// Work lease on a remote worker — follows same lease + outcome validation rules as local execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteLease {
    pub lease_id: u64,
    pub worker_entity: u64,
    pub session_entity: u64,
    pub issued_at: Timestamp,
    pub expires_at: Timestamp,
    pub cancellation_epoch: WorldEpoch,
}

/// An observation about a remote worker's capability — NOT a trusted canonical fact until validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteCapabilityObservation {
    pub worker_entity: u64,
    pub observed_capability: String,
    pub observed_at: Timestamp,
    pub validated: bool,
}

// ── Component impls ─────────────────────────────────────────────────────────

impl crate::ecs::Component for PeerIdentity {}
impl crate::ecs::Component for NodeMembership {}
impl crate::ecs::Component for PeerCapabilities {}
impl crate::ecs::Component for NodeTopology {}
impl crate::ecs::Component for TrustState {}
impl crate::ecs::Component for WorkerHealth {}
impl crate::ecs::Component for RemoteLease {}
impl crate::ecs::Component for RemoteCapabilityObservation {}

// ── ClassifiedComponent / DurableComponent impls ─────────────────────────

impl ClassifiedComponent for PeerIdentity {
    type Class = DurableClass;
}
impl DurableComponent for PeerIdentity {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 55,
        version: 1,
    };
}

impl ClassifiedComponent for NodeMembership {
    type Class = DurableClass;
}
impl DurableComponent for NodeMembership {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 56,
        version: 1,
    };
}

impl ClassifiedComponent for PeerCapabilities {
    type Class = DurableClass;
}
impl DurableComponent for PeerCapabilities {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 57,
        version: 1,
    };
}

impl ClassifiedComponent for NodeTopology {
    type Class = DurableClass;
}
impl DurableComponent for NodeTopology {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 58,
        version: 1,
    };
}

impl ClassifiedComponent for TrustState {
    type Class = DurableClass;
}
impl DurableComponent for TrustState {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 59,
        version: 1,
    };
}

impl ClassifiedComponent for WorkerHealth {
    type Class = DurableClass;
}
impl DurableComponent for WorkerHealth {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 60,
        version: 1,
    };
}

impl ClassifiedComponent for RemoteLease {
    type Class = DurableClass;
}
impl DurableComponent for RemoteLease {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 61,
        version: 1,
    };
}

impl ClassifiedComponent for RemoteCapabilityObservation {
    type Class = DurableClass;
}
impl DurableComponent for RemoteCapabilityObservation {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.distributed",
        id: 62,
        version: 1,
    };
}

// ══════════════════════════════════════════════════════════════════════════════
// Schema Validation
// ══════════════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════════════════
// Replay
// ══════════════════════════════════════════════════════════════════════════════

/// Replay a `peer_registered` event to reconstruct a peer node entity.
///
/// Restores: PeerIdentity, NodeMembership, TrustState::Observed.
/// Spawns the node entity if not already present (idempotent).
pub fn replay_peer_registered(
    world: &mut CompWorld,
    event: &DomainEvent,
) -> Result<CommittedEpoch, DistributedError> {
    let node_id = event
        .payload
        .get("node_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let peer_id_str = event
        .payload
        .get("peer_id")
        .and_then(|v| v.as_str())
        .unwrap_or("replay");
    let cluster = event
        .payload
        .get("cluster_name")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let mut txn = WorldTxn::new(world);

    if !world.has_entity(crate::ecs::CompEntity(node_id)) {
        txn.stage_spawn(node_id, EntityKind::Node);
    }

    txn.add_component(
        node_id,
        ComponentSchemaId(SCHEMA_PEER_IDENTITY),
        SchemaVersion(1),
        PeerIdentity {
            peer_id: peer_id_str.to_string(),
            public_key: vec![],
            discovered_at: Timestamp::now(),
        },
    );
    txn.add_component(
        node_id,
        ComponentSchemaId(SCHEMA_NODE_MEMBERSHIP),
        SchemaVersion(1),
        NodeMembership {
            node_id,
            cluster_name: cluster.to_string(),
            joined_epoch: WorldEpoch(0),
            last_seen: Timestamp::now(),
        },
    );
    txn.add_component(
        node_id,
        ComponentSchemaId(SCHEMA_TRUST_STATE),
        SchemaVersion(1),
        TrustState::Observed,
    );

    let epoch = world.transit(txn).map_err(DistributedError::CommitFailed)?;
    Ok(epoch)
}

/// Validate all distributed schemas are registered for the correct types.
pub fn validate_distributed_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<PeerIdentity>(ComponentSchemaId(SCHEMA_PEER_IDENTITY))
        .map_err(|e| format!("PeerIdentity schema: {e}"))?;
    reg.verify_type::<NodeMembership>(ComponentSchemaId(SCHEMA_NODE_MEMBERSHIP))
        .map_err(|e| format!("NodeMembership schema: {e}"))?;
    reg.verify_type::<PeerCapabilities>(ComponentSchemaId(SCHEMA_PEER_CAPABILITIES))
        .map_err(|e| format!("PeerCapabilities schema: {e}"))?;
    reg.verify_type::<NodeTopology>(ComponentSchemaId(SCHEMA_NODE_TOPOLOGY))
        .map_err(|e| format!("NodeTopology schema: {e}"))?;
    reg.verify_type::<TrustState>(ComponentSchemaId(SCHEMA_TRUST_STATE))
        .map_err(|e| format!("TrustState schema: {e}"))?;
    reg.verify_type::<WorkerHealth>(ComponentSchemaId(SCHEMA_WORKER_HEALTH))
        .map_err(|e| format!("WorkerHealth schema: {e}"))?;
    reg.verify_type::<RemoteLease>(ComponentSchemaId(SCHEMA_REMOTE_LEASE))
        .map_err(|e| format!("RemoteLease schema: {e}"))?;
    reg.verify_type::<RemoteCapabilityObservation>(ComponentSchemaId(
        SCHEMA_REMOTE_CAPABILITY_OBSERVATION,
    ))
    .map_err(|e| format!("RemoteCapabilityObservation schema: {e}"))?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// Commands
// ══════════════════════════════════════════════════════════════════════════════

/// Register a peer node in the cluster with its identity, membership, and claimed capabilities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterPeerCommand {
    pub id: MessageId,
    pub peer_identity: PeerIdentity,
    pub node_membership: NodeMembership,
    pub capabilities: Vec<String>,
}

impl RegisterPeerCommand {
    /// Preflight: peer_id must not already be registered.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), DistributedError> {
        Self::validate_schemas(schema_registry)?;

        // Check no existing node entity has this peer_id
        for entity in world.entities_of_kind(EntityKind::Node) {
            if let Some(identity) = world.get_component::<PeerIdentity>(entity) {
                if identity.peer_id == self.peer_identity.peer_id {
                    return Err(DistributedError::PeerAlreadyRegistered(
                        self.peer_identity.peer_id.clone(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Execute registration: spawn a Node entity with all distributed components.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), DistributedError> {
        self.preflight(world, schema_registry)?;

        let node_id = WorldTxn::next_entity_id(world);
        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(node_id, EntityKind::Node);

        txn.put_durable(node_id, self.peer_identity.clone());
        txn.put_durable(node_id, self.node_membership.clone());
        txn.put_durable(node_id, PeerCapabilities(self.capabilities.clone()));
        txn.put_durable(node_id, TrustState::Untrusted);
        txn.put_durable(node_id, WorkerHealth::Healthy);

        let event = DomainEvent {
            id: self.id,
            kind: "peer_registered".to_string(),
            entity_id: Some(EntityKindId(node_id)),
            payload: serde_json::json!({
                "node_id": node_id,
                "peer_id": self.peer_identity.peer_id,
                "cluster": self.node_membership.cluster_name,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(DistributedError::CommitFailed)?;
        Ok((epoch, event))
    }

    /// Validate all distributed schemas are registered.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), DistributedError> {
        validate_distributed_schemas(schema_registry).map_err(|e| DistributedError::SchemaError(e))
    }
}

/// Record an observation about a worker's capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObserveWorkerCapabilityCommand {
    pub id: MessageId,
    pub worker_entity: u64,
    pub capability: String,
}

impl ObserveWorkerCapabilityCommand {
    /// Preflight: worker_entity must exist and be a Node.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), DistributedError> {
        Self::validate_schemas(schema_registry)?;

        let entity = CompEntity(self.worker_entity);
        if !world.has_entity(entity) {
            return Err(DistributedError::WorkerNotFound(self.worker_entity));
        }
        if world.entity_kind(entity) != Some(EntityKind::Node) {
            return Err(DistributedError::WorkerNotFound(self.worker_entity));
        }

        Ok(())
    }

    /// Execute: attach a new RemoteCapabilityObservation component to the worker entity.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), DistributedError> {
        self.preflight(world, schema_registry)?;

        // Observations are not authoritative — we attach them as components.
        // A worker may accumulate multiple observations; each gets its own entity.
        let observation_id = WorldTxn::next_entity_id(world);
        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(observation_id, EntityKind::Node);

        txn.put_durable(
            observation_id,
            RemoteCapabilityObservation {
                worker_entity: self.worker_entity,
                observed_capability: self.capability.clone(),
                observed_at: Timestamp::now(),
                validated: false,
            },
        );

        let event = DomainEvent {
            id: self.id,
            kind: "worker_capability_observed".to_string(),
            entity_id: Some(EntityKindId(observation_id)),
            payload: serde_json::json!({
                "worker_entity": self.worker_entity,
                "capability": self.capability,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(DistributedError::CommitFailed)?;
        Ok((epoch, event))
    }

    /// Validate all schemas.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), DistributedError> {
        validate_distributed_schemas(schema_registry).map_err(|e| DistributedError::SchemaError(e))
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Errors
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DistributedError {
    #[error("peer already registered: {0}")]
    PeerAlreadyRegistered(String),
    #[error("worker not found: {0}")]
    WorkerNotFound(u64),
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
    use crate::ecs::CompWorld;

    fn make_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register_for_type::<PeerIdentity>(
            ComponentSchemaId(SCHEMA_PEER_IDENTITY),
            SchemaVersion(1),
            "PeerIdentity",
            "Network peer identity",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<NodeMembership>(
            ComponentSchemaId(SCHEMA_NODE_MEMBERSHIP),
            SchemaVersion(1),
            "NodeMembership",
            "Cluster node membership state",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<PeerCapabilities>(
            ComponentSchemaId(SCHEMA_PEER_CAPABILITIES),
            SchemaVersion(1),
            "PeerCapabilities",
            "Capabilities advertised by a peer",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<NodeTopology>(
            ComponentSchemaId(SCHEMA_NODE_TOPOLOGY),
            SchemaVersion(1),
            "NodeTopology",
            "Topological position in the cluster",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<TrustState>(
            ComponentSchemaId(SCHEMA_TRUST_STATE),
            SchemaVersion(1),
            "TrustState",
            "Trust level for peer interactions",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<WorkerHealth>(
            ComponentSchemaId(SCHEMA_WORKER_HEALTH),
            SchemaVersion(1),
            "WorkerHealth",
            "Worker health and heartbeat",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<RemoteLease>(
            ComponentSchemaId(SCHEMA_REMOTE_LEASE),
            SchemaVersion(1),
            "RemoteLease",
            "Remote execution lease",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<RemoteCapabilityObservation>(
            ComponentSchemaId(SCHEMA_REMOTE_CAPABILITY_OBSERVATION),
            SchemaVersion(1),
            "RemoteCapabilityObservation",
            "Observed capability from remote peer",
            ComponentDurability::Durable,
        );
        reg
    }

    #[test]
    fn test_trust_state_transitions() {
        // Forward progression
        assert_eq!(
            TrustState::Untrusted.transition_to(TrustState::Observed),
            Ok(TrustState::Observed)
        );
        assert_eq!(
            TrustState::Observed.transition_to(TrustState::Validated),
            Ok(TrustState::Validated)
        );
        assert_eq!(
            TrustState::Validated.transition_to(TrustState::Trusted),
            Ok(TrustState::Trusted)
        );

        // Revocation from any pre-revocation state
        assert_eq!(
            TrustState::Untrusted.transition_to(TrustState::Revoked),
            Ok(TrustState::Revoked)
        );
        assert_eq!(
            TrustState::Observed.transition_to(TrustState::Revoked),
            Ok(TrustState::Revoked)
        );
        assert_eq!(
            TrustState::Validated.transition_to(TrustState::Revoked),
            Ok(TrustState::Revoked)
        );
        assert_eq!(
            TrustState::Trusted.transition_to(TrustState::Revoked),
            Ok(TrustState::Revoked)
        );

        // No backwards transitions
        assert!(TrustState::Observed
            .transition_to(TrustState::Untrusted)
            .is_err());
        assert!(TrustState::Validated
            .transition_to(TrustState::Observed)
            .is_err());
        assert!(TrustState::Validated
            .transition_to(TrustState::Untrusted)
            .is_err());
        assert!(TrustState::Trusted
            .transition_to(TrustState::Validated)
            .is_err());
        assert!(TrustState::Trusted
            .transition_to(TrustState::Untrusted)
            .is_err());

        // No escape from Revoked
        assert!(TrustState::Revoked
            .transition_to(TrustState::Untrusted)
            .is_err());
        assert!(TrustState::Revoked
            .transition_to(TrustState::Trusted)
            .is_err());
    }

    #[test]
    fn test_peer_identity_serde() {
        let identity = PeerIdentity {
            peer_id: "node-abc-123".to_string(),
            public_key: vec![0xab, 0xcd, 0xef, 0x01],
            discovered_at: Timestamp::from_nanos(1_000_000),
        };

        let json = serde_json::to_string(&identity).unwrap();
        let deserialized: PeerIdentity = serde_json::from_str(&json).unwrap();

        assert_eq!(identity, deserialized);
        assert_eq!(deserialized.peer_id, "node-abc-123");
        assert_eq!(deserialized.public_key, vec![0xab, 0xcd, 0xef, 0x01]);
    }

    #[test]
    fn test_remote_lease_construction() {
        let lease = RemoteLease {
            lease_id: 42,
            worker_entity: 1001,
            session_entity: 500,
            issued_at: Timestamp::from_nanos(1_000_000_000),
            expires_at: Timestamp::from_nanos(2_000_000_000),
            cancellation_epoch: WorldEpoch(0),
        };

        assert_eq!(lease.lease_id, 42);
        assert_eq!(lease.worker_entity, 1001);
        assert_eq!(lease.session_entity, 500);
        assert!(lease.issued_at.0 < lease.expires_at.0);
        assert_eq!(lease.cancellation_epoch, WorldEpoch(0));

        // Round-trip
        let json = serde_json::to_string(&lease).unwrap();
        let deserialized: RemoteLease = serde_json::from_str(&json).unwrap();
        assert_eq!(lease, deserialized);
    }

    #[test]
    fn test_register_peer_preflight_duplicate() {
        let reg = make_registry();
        let mut world = CompWorld::new();

        let cmd = RegisterPeerCommand {
            id: MessageId::compute(b"reg-1"),
            peer_identity: PeerIdentity {
                peer_id: "node-dup".to_string(),
                public_key: vec![0x01],
                discovered_at: Timestamp::now(),
            },
            node_membership: NodeMembership {
                node_id: 1,
                cluster_name: "alpha".to_string(),
                joined_epoch: WorldEpoch(1),
                last_seen: Timestamp::now(),
            },
            capabilities: vec!["compute".to_string()],
        };

        // First registration succeeds
        cmd.clone().execute(&mut world, &reg).unwrap();

        // Second registration with same peer_id fails preflight
        let dup = RegisterPeerCommand {
            id: MessageId::compute(b"reg-2"),
            peer_identity: PeerIdentity {
                peer_id: "node-dup".to_string(),
                public_key: vec![0x02],
                discovered_at: Timestamp::now(),
            },
            node_membership: NodeMembership {
                node_id: 2,
                cluster_name: "alpha".to_string(),
                joined_epoch: WorldEpoch(2),
                last_seen: Timestamp::now(),
            },
            capabilities: vec![],
        };
        let err = dup.preflight(&world, &reg).unwrap_err();
        assert!(matches!(err, DistributedError::PeerAlreadyRegistered(_)));
    }

    #[test]
    fn test_observe_capability_preflight_missing_worker() {
        let reg = make_registry();
        let world = CompWorld::new();

        let cmd = ObserveWorkerCapabilityCommand {
            id: MessageId::compute(b"obs-1"),
            worker_entity: 9999,
            capability: "inference".to_string(),
        };

        let err = cmd.preflight(&world, &reg).unwrap_err();
        assert!(matches!(err, DistributedError::WorkerNotFound(9999)));
    }
}
