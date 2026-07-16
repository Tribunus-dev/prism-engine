//! Bridge wrapping the constitutional `SubmitIngressRequestCommand` behind a
//! simple synchronous API for worker ingress callers.
//!
//! Owns an `Arc<RwLock<World>>` and a pre-built `SchemaRegistry` with all
//! ingress schemas.  Each method:
//! 1. Locks the world for writing
//! 2. Constructs the relevant constitutional command with a unique `MessageId`
//! 3. Calls execute (which internally runs preflight then stages a `WorldTxn`)
//! 4. Returns the allocated entity id on success
//!
//! This is the authority path for `SubmitIngressRequestCommand`.

use crate::ecs::constitutional::ingress::{
    ApiKey, IngressLifecycle, IngressRequest, RateLimiterState, RequestQueue,
    SubmitIngressRequestCommand, TransportSession, SCHEMA_API_KEY, SCHEMA_INGRESS_LIFECYCLE,
    SCHEMA_INGRESS_REQUEST, SCHEMA_RATE_LIMITER, SCHEMA_REQUEST_QUEUE, SCHEMA_TRANSPORT_SESSION,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion};
use crate::ecs::World;
use std::fmt;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Wraps the constitutional `SubmitIngressRequestCommand` behind a simple
/// synchronous API for production ingress callers.
///
/// Each method:
/// 1. Generates a unique `MessageId` via UUID
/// 2. Locks the world for writing
/// 3. Constructs the constitutional command
/// 4. Executes it (preflight + execute via the command's own lifecycle)
/// 5. Returns the result
pub struct IngressBridge {
    world: Arc<RwLock<World>>,
    schema_registry: SchemaRegistry,
}

impl fmt::Debug for IngressBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IngressBridge")
            .field("schema_registry", &self.schema_registry)
            .finish_non_exhaustive()
    }
}

// SAFETY: IngressBridge only accesses the World through an RwLock(World) that
// properly synchronizes all access. The `!Sync` nature of `World` comes from
// internal FnOnce closures, not from any data-race-vulnerable state — the
// RwLock ensures mutual exclusion across threads. This is the same pattern
// used by other bridge structures in this codebase that hold Arc<RwLock<World>>.
unsafe impl Sync for IngressBridge {}
// SAFETY: See above for Sync. The RwLock provides mutual exclusion for all
// World access, making it safe to transfer the bridge across thread boundaries.
unsafe impl Send for IngressBridge {}
impl IngressBridge {
    /// Create a new bridge backed by the given world.
    ///
    /// Registers all ingress schemas on construction so schema validation
    /// inside the constitutional command passes.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        let mut reg = SchemaRegistry::new();
        Self::register_ingress_schemas(&mut reg);
        Self {
            world,
            schema_registry: reg,
        }
    }

    /// Register all ingress domain schemas into the given registry.
    fn register_ingress_schemas(reg: &mut SchemaRegistry) {
        reg.register_for_type::<IngressRequest>(
            ComponentSchemaId(SCHEMA_INGRESS_REQUEST),
            SchemaVersion(1),
            "IngressRequest",
            "incoming request from any transport",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<ApiKey>(
            ComponentSchemaId(SCHEMA_API_KEY),
            SchemaVersion(1),
            "ApiKey",
            "API key with permissions",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<RateLimiterState>(
            ComponentSchemaId(SCHEMA_RATE_LIMITER),
            SchemaVersion(1),
            "RateLimiterState",
            "rate limiter state for a transport or API key",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<RequestQueue>(
            ComponentSchemaId(SCHEMA_REQUEST_QUEUE),
            SchemaVersion(1),
            "RequestQueue",
            "queue of pending ingress requests",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<TransportSession>(
            ComponentSchemaId(SCHEMA_TRANSPORT_SESSION),
            SchemaVersion(1),
            "TransportSession",
            "persistent transport session",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<IngressLifecycle>(
            ComponentSchemaId(SCHEMA_INGRESS_LIFECYCLE),
            SchemaVersion(1),
            "IngressLifecycle",
            "lifecycle state of an ingress request",
            ComponentDurability::Durable,
        );
    }

    /// Generate a unique `MessageId` for each command invocation.
    fn msg_id() -> MessageId {
        MessageId::compute(Uuid::new_v4().as_bytes())
    }

    /// Submit an ingress request from any transport.
    ///
    /// Constructs a [`SubmitIngressRequestCommand`] internally, runs execute
    /// against the world, and returns the newly allocated entity id on success.
    pub fn submit_request(
        &self,
        transport: &str,
        method: &str,
        path: &str,
        body: Vec<u8>,
    ) -> Result<u64, String> {
        let mut world = self
            .world
            .write()
            .map_err(|e| format!("world lock poisoned: {e}"))?;

        let cmd = SubmitIngressRequestCommand {
            id: Self::msg_id(),
            transport: transport.to_string(),
            method: method.to_string(),
            path: path.to_string(),
            body,
            api_key_hash: None,
        };

        cmd.execute(&mut *world, &self.schema_registry)
            .map(|(_epoch, event)| {
                event
                    .payload
                    .get("request_id")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
            })
            .map_err(|e| e.to_string())
    }
}
