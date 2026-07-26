//! Bridge wrapping constitutional distributed peer registration commands behind a
//! simple synchronous API.
//!
//! Owns an `Arc<RwLock<World>>` and a `SchemaRegistry` with all distributed
//! component schemas pre-registered.  Each method:
//! 1. Locks the world for writing
//! 2. Constructs the relevant constitutional command with a unique `MessageId`
//! 3. Executes it (the command internally runs preflight + WorldTxn + transit)
//! 4. Returns the result
//!
//! This is the authority path for:
//! - `RegisterPeerCommand` → `DistributedBridge::register_peer`

use crate::ecs::constitutional::distributed::{
    NodeMembership, NodeTopology, PeerCapabilities, PeerIdentity, RegisterPeerCommand,
    RemoteCapabilityObservation, RemoteLease, TrustState, WorkerHealth, SCHEMA_NODE_MEMBERSHIP,
    SCHEMA_NODE_TOPOLOGY, SCHEMA_PEER_CAPABILITIES, SCHEMA_PEER_IDENTITY,
    SCHEMA_REMOTE_CAPABILITY_OBSERVATION, SCHEMA_REMOTE_LEASE, SCHEMA_TRUST_STATE,
    SCHEMA_WORKER_HEALTH,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion};
use crate::ecs::World;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Wraps constitutional distributed peer registration commands behind a simple
/// synchronous API.
///
/// Each method:
/// 1. Generates a unique `MessageId` via UUID
/// 2. Locks the world for writing
/// 3. Constructs the constitutional command
/// 4. Executes it (preflight + execute)
/// 5. Returns the result
pub struct DistributedBridge {
    world: Arc<RwLock<World>>,
    schema_registry: SchemaRegistry,
}

impl DistributedBridge {
    /// Create a new bridge backed by the given world.
    ///
    /// Registers all distributed schemas on construction so schema validation
    /// inside each constitutional command passes.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        let mut reg = SchemaRegistry::new();
        Self::register_distributed_schemas(&mut reg);
        Self {
            world,
            schema_registry: reg,
        }
    }

    /// Register all distributed domain schemas into the given registry.
    fn register_distributed_schemas(reg: &mut SchemaRegistry) {
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
    }

    // ── helpers ────────────────────────────────────────────────────────────

    /// Generate a unique `MessageId` for each command invocation.
    fn msg_id() -> MessageId {
        MessageId::compute(Uuid::new_v4().as_bytes())
    }

    // ── public API ─────────────────────────────────────────────────────────

    /// Register a peer node in the distributed cluster.
    ///
    /// Constructs a [`RegisterPeerCommand`] internally, runs preflight then
    /// execute against the world, and returns the newly allocated node entity id
    /// on success.
    pub fn register_peer(
        &self,
        identity: PeerIdentity,
        membership: NodeMembership,
        capabilities: Vec<String>,
    ) -> Result<u64, String> {
        let mut world = self
            .world
            .write()
            .map_err(|e| format!("world lock poisoned: {e}"))?;

        let cmd = RegisterPeerCommand {
            id: Self::msg_id(),
            peer_identity: identity,
            node_membership: membership,
            capabilities,
        };

        let (_epoch, event) = cmd
            .execute(&mut *world, &self.schema_registry)
            .map_err(|e| e.to_string())?;

        // Extract the node entity id from the event's entity_id
        let node_id = event
            .entity_id
            .ok_or_else(|| "peer_registered event missing entity_id".to_string())?
            .0;

        Ok(node_id)
    }
}
