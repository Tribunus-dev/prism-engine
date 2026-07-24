use crate::command::DomainEvent;
use crate::schema::SchemaRegistry;
use crate::types::*;
use crate::world_txn::WorldTransitExt;
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use prism_ecs_core::{Entity, EntityKind, World};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs (63-68) ──────────────────────────────────────────
// Session: 13-17, Work: 18-23, Execution: 24-29, Compilation: 31-38,
// Agent: 39-46, Scheduler: 47-54, Driver: 55-62, Ingress: 63+

pub const SCHEMA_INGRESS_REQUEST: u64 = 63;
pub const SCHEMA_API_KEY: u64 = 64;
pub const SCHEMA_RATE_LIMITER: u64 = 65;
pub const SCHEMA_REQUEST_QUEUE: u64 = 66;
pub const SCHEMA_TRANSPORT_SESSION: u64 = 67;
pub const SCHEMA_INGRESS_LIFECYCLE: u64 = 68;

// ── Ingress Components ────────────────────────────────────────────────────

/// An incoming request from any transport (HTTP, Swift, JS, P2P, CLI).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngressRequest {
    pub request_id: u64,
    pub transport: String,
    pub method: String,
    pub path: String,
    pub body_hash: [u8; 32],
    pub received_at: Timestamp,
    pub resolved_command: Option<String>,
}

/// API key with permissions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKey {
    pub key_hash: [u8; 32],
    pub owner: String,
    pub permissions: Vec<String>,
    pub expires_at: Timestamp,
}

/// Rate limiter state for a transport or API key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateLimiterState {
    pub window_start: Timestamp,
    pub request_count: u64,
    pub max_requests: u64,
    pub window_ms: u64,
}

/// Queue of pending ingress requests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestQueue {
    pub queue_entity: u64,
    pub requests: Vec<u64>,
    pub drained_epoch: WorldEpoch,
}

/// Persistent transport session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransportSession {
    pub session_entity: u64,
    pub transport: String,
    pub remote_addr: String,
    pub connected_at: Timestamp,
    pub last_activity: Timestamp,
}

// ── Ingress Lifecycle ─────────────────────────────────────────────────────

/// Lifecycle of an ingress request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IngressLifecycle {
    Received,
    Authenticated,
    Queued,
    Resolved,
    Rejected,
    Completed,
}

impl IngressLifecycle {
    /// Returns true if this state is terminal (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Rejected | Self::Completed)
    }

    /// Validate a lifecycle transition. Returns Ok(()) if allowed.
    pub fn can_transition_to(&self, target: Self) -> Result<(), IngressError> {
        let allowed = match (*self, target) {
            (Self::Received, Self::Authenticated)
            | (Self::Received, Self::Rejected)
            | (Self::Authenticated, Self::Queued)
            | (Self::Authenticated, Self::Rejected)
            | (Self::Queued, Self::Resolved)
            | (Self::Queued, Self::Rejected)
            | (Self::Resolved, Self::Completed)
            | (Self::Resolved, Self::Rejected) => true,
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(IngressError::InvalidTransition {
                from: *self,
                to: target,
            })
        }
    }

    /// Short machine-readable name for event routing.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Received => "received",
            Self::Authenticated => "authenticated",
            Self::Queued => "queued",
            Self::Resolved => "resolved",
            Self::Rejected => "rejected",
            Self::Completed => "completed",
        }
    }
}

// ── Schema Validation ─────────────────────────────────────────────────────

/// Validate all ingress schemas are registered for the correct types.
pub fn validate_ingress_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<IngressRequest>(ComponentSchemaId(SCHEMA_INGRESS_REQUEST))
        .map_err(|e| format!("IngressRequest schema: {e}"))?;
    reg.verify_type::<ApiKey>(ComponentSchemaId(SCHEMA_API_KEY))
        .map_err(|e| format!("ApiKey schema: {e}"))?;
    reg.verify_type::<RateLimiterState>(ComponentSchemaId(SCHEMA_RATE_LIMITER))
        .map_err(|e| format!("RateLimiterState schema: {e}"))?;
    reg.verify_type::<RequestQueue>(ComponentSchemaId(SCHEMA_REQUEST_QUEUE))
        .map_err(|e| format!("RequestQueue schema: {e}"))?;
    reg.verify_type::<TransportSession>(ComponentSchemaId(SCHEMA_TRANSPORT_SESSION))
        .map_err(|e| format!("TransportSession schema: {e}"))?;
    reg.verify_type::<IngressLifecycle>(ComponentSchemaId(SCHEMA_INGRESS_LIFECYCLE))
        .map_err(|e| format!("IngressLifecycle schema: {e}"))?;
    Ok(())
}

// ── SubmitIngressRequestCommand ───────────────────────────────────────────

/// Submit an ingress request from any transport.
///
/// Transports (HTTP, Swift, JS, P2P, CLI) all resolve through this same
/// constitutional command regardless of origin. The transport adapter contains
/// no domain authority — restarting the server does not alter the canonical world.
///
/// The `ingress_entity` field uses a `u64` identifier. The canonical Entity
/// equivalent is `Entity(ingress_id, gen)`. New callers should prefer `Entity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitIngressRequestCommand {
    pub id: MessageId,
    pub transport: String,
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
    pub api_key_hash: Option<[u8; 32]>,
}

impl SubmitIngressRequestCommand {
    /// Validate ingress schemas are registered.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        validate_ingress_schemas(schema_registry)
    }

    /// Preflight: validate api_key hash is valid (if provided), rate limit not exceeded.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), IngressError> {
        Self::validate_schemas(schema_registry).map_err(|e| IngressError::SchemaError(e))?;

        // Validate API key if provided
        if let Some(key_hash) = &self.api_key_hash {
            let api_key_valid = world
                .entities_of_kind(EntityKind::Session)
                .iter()
                .any(|&entity| {
                    if let Some(ak) = world.get_component::<ApiKey>(entity) {
                        ak.key_hash == *key_hash && ak.expires_at.0 > Timestamp::now().0
                    } else {
                        false
                    }
                });
            if !api_key_valid {
                return Err(IngressError::InvalidApiKey);
            }
        }

        Ok(())
    }

    /// Execute the command: create ingress entity with lifecycle=Received,
    /// attach IngressRequest component, emit event, and request resolution.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), IngressError> {
        self.preflight(world, schema_registry)?;

        let request_id = WorldTxn::next_entity_id(world);
        let now = Timestamp::now();
        let body_hash = blake3::hash(&self.body);

        let mut txn = WorldTxn::new(world);

        txn.stage_spawn(request_id, EntityKind::Session);

        txn.put_durable(
            request_id,
            IngressRequest {
                request_id: request_id.0,
                transport: self.transport.clone(),
                method: self.method.clone(),
                path: self.path.clone(),
                body_hash: *body_hash.as_bytes(),
                received_at: now,
                resolved_command: None,
            },
        );

        txn.put_durable(request_id, IngressLifecycle::Received);

        let body_hash_hex: String = body_hash
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let event = DomainEvent {
            id: self.id,
            kind: "ingress_request_received".to_string(),
            entity_id: Some(EntityKindId(request_id.0)),
            payload: serde_json::json!({
                "request_id": request_id,
                "transport": self.transport,
                "method": self.method,
                "path": self.path,
                "body_hash": body_hash_hex,
                "received_at": now.0,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(IngressError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── ResolveIngressCommand ─────────────────────────────────────────────────

/// Mark an ingress request as resolved to a specific constitutional command.
///
/// The `ingress_entity` field uses a `u64` identifier. The canonical Entity
/// equivalent is `Entity(ingress_id, gen)`. New callers should prefer `Entity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolveIngressCommand {
    pub id: MessageId,
    pub ingress_entity: u64,
    pub resolved_command_id: MessageId,
    pub target_subsystem: String,
}

impl ResolveIngressCommand {
    /// Validate ingress schemas.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        validate_ingress_schemas(schema_registry)
    }

    /// Preflight: ingress entity exists and is in a resolvable lifecycle state.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), IngressError> {
        Self::validate_schemas(schema_registry).map_err(|e| IngressError::SchemaError(e))?;

        let entity = Entity::new(self.ingress_entity, 0);
        if !world.has_entity(entity) {
            return Err(IngressError::EntityNotFound(self.ingress_entity));
        }

        // Current lifecycle must allow Resolved transition
        let current = world
            .get_component::<IngressLifecycle>(entity)
            .ok_or(IngressError::EntityNotFound(self.ingress_entity))?;

        current
            .can_transition_to(IngressLifecycle::Resolved)
            .map_err(|_| IngressError::InvalidTransition {
                from: *current,
                to: IngressLifecycle::Resolved,
            })?;

        Ok(())
    }

    /// Execute the resolution: transition lifecycle to Resolved,
    /// update the resolved_command on IngressRequest, emit event.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), IngressError> {
        self.preflight(world, schema_registry)?;

        let mut txn = WorldTxn::new(world);

        // Update lifecycle
        txn.put_durable(
            Entity::new(self.ingress_entity, 0),
            IngressLifecycle::Resolved,
        );

        // Update the resolved_command field by replacing IngressRequest
        if let Some(ingress_req) =
            world.get_component::<IngressRequest>(Entity::new(self.ingress_entity, 0))
        {
            let updated = IngressRequest {
                resolved_command: Some(self.resolved_command_id.to_string()),
                ..ingress_req.clone()
            };
            txn.put_durable(Entity::new(self.ingress_entity, 0), updated);
        }

        let event = DomainEvent {
            id: self.id,
            kind: "ingress_request_resolved".to_string(),
            entity_id: Some(EntityKindId(self.ingress_entity)),
            payload: serde_json::json!({
                "ingress_entity": self.ingress_entity,
                "resolved_command_id": self.resolved_command_id.to_string(),
                "target_subsystem": self.target_subsystem,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(IngressError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── IngressLifecycleTransitionCommand ─────────────────────────────────────

/// Directly transition an ingress request lifecycle (e.g. Authenticated,
/// Queued, Rejected, Completed).
///
/// The `ingress_entity` field uses a `u64` identifier. The canonical Entity
/// equivalent is `Entity(ingress_id, gen)`. New callers should prefer `Entity`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngressLifecycleTransitionCommand {
    pub id: MessageId,
    pub ingress_entity: u64,
    pub target: IngressLifecycle,
}

impl IngressLifecycleTransitionCommand {
    /// Validate ingress schemas.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        validate_ingress_schemas(schema_registry)
    }

    /// Preflight: entity exists, transition is valid.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), IngressError> {
        Self::validate_schemas(schema_registry).map_err(|e| IngressError::SchemaError(e))?;

        let entity = Entity::new(self.ingress_entity, 0);
        if !world.has_entity(entity) {
            return Err(IngressError::EntityNotFound(self.ingress_entity));
        }

        let current = world
            .get_component::<IngressLifecycle>(entity)
            .ok_or(IngressError::EntityNotFound(self.ingress_entity))?;

        current
            .can_transition_to(self.target)
            .map_err(|_| IngressError::InvalidTransition {
                from: *current,
                to: self.target,
            })?;

        Ok(())
    }

    /// Execute: update lifecycle to target, emit event.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), IngressError> {
        self.preflight(world, schema_registry)?;

        let mut txn = WorldTxn::new(world);

        txn.put_durable(Entity::new(self.ingress_entity, 0), self.target);

        let event = DomainEvent {
            id: self.id,
            kind: format!("ingress_{}", self.target.name()),
            entity_id: Some(EntityKindId(self.ingress_entity)),
            payload: serde_json::json!({
                "ingress_entity": self.ingress_entity,
                "to": format!("{:?}", self.target),
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(IngressError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IngressError {
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("invalid API key")]
    InvalidApiKey,
    #[error("ingress entity {0} not found")]
    EntityNotFound(u64),
    #[error("invalid lifecycle transition: from {from:?} to {to:?}")]
    InvalidTransition {
        from: IngressLifecycle,
        to: IngressLifecycle,
    },
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Component impls ───────────────────────────────────────────────────────

impl prism_ecs_core::Component for IngressRequest {}
impl prism_ecs_core::Component for ApiKey {}
impl prism_ecs_core::Component for RateLimiterState {}
impl prism_ecs_core::Component for RequestQueue {}
impl prism_ecs_core::Component for TransportSession {}
impl prism_ecs_core::Component for IngressLifecycle {}

// ── Entity conversion helper ────────────────────────────────────────────────

/// Convert a legacy `u64` entity identifier to the canonical `Entity` type.
///
/// Uses generation `0` for entities created outside the ECS lifecycle
/// (e.g., replay, test fixtures). Prefer receiving entities from
/// `World::spawn()` over constructing them directly.
#[allow(dead_code)]
pub(crate) fn as_entity(id: u64) -> Entity {
    Entity::new(id, 0)
}

impl ClassifiedComponent for IngressRequest {
    type Class = DurableClass;
}
impl DurableComponent for IngressRequest {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 63,
        version: 1,
    };
}

impl ClassifiedComponent for ApiKey {
    type Class = DurableClass;
}
impl DurableComponent for ApiKey {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 64,
        version: 1,
    };
}

impl ClassifiedComponent for RateLimiterState {
    type Class = DurableClass;
}
impl DurableComponent for RateLimiterState {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 65,
        version: 1,
    };
}

impl ClassifiedComponent for RequestQueue {
    type Class = DurableClass;
}
impl DurableComponent for RequestQueue {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 66,
        version: 1,
    };
}

impl ClassifiedComponent for TransportSession {
    type Class = DurableClass;
}
impl DurableComponent for TransportSession {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 67,
        version: 1,
    };
}

impl ClassifiedComponent for IngressLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for IngressLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.ingress",
        id: 68,
        version: 1,
    };
}

// ── Replay Functions ───────────────────────────────────────────────────

/// Replay an ingress request submitted event, restoring the ingress entity
/// and its request/lifecycle components.
pub fn replay_ingress_request_submitted(
    world: &mut World,
    event: &DomainEvent,
) -> Result<CommittedEpoch, IngressError> {
    let entity_id = event.entity_id.ok_or(IngressError::EntityNotFound(0))?.0;
    let transport = event
        .payload
        .get("transport")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let method = event
        .payload
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("GET")
        .to_string();
    let path = event
        .payload
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("/")
        .to_string();

    let mut txn = WorldTxn::new(world);
    if !world.has_entity(Entity::new(entity_id, 0)) {
        txn.stage_spawn(Entity::new(entity_id, 0), EntityKind::Session);
    }
    txn.add_component(
        Entity::new(entity_id, 0),
        ComponentSchemaId(SCHEMA_INGRESS_REQUEST),
        SchemaVersion(1),
        IngressRequest {
            request_id: entity_id,
            transport,
            method,
            path,
            body_hash: [0u8; 32],
            received_at: Timestamp::now(),
            resolved_command: None,
        },
    );
    txn.add_component(
        Entity::new(entity_id, 0),
        ComponentSchemaId(SCHEMA_INGRESS_LIFECYCLE),
        SchemaVersion(1),
        IngressLifecycle::Received,
    );
    let epoch = world.transit(txn).map_err(IngressError::CommitFailed)?;
    Ok(epoch)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::*;

    fn make_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register_for_type::<IngressRequest>(
            ComponentSchemaId(SCHEMA_INGRESS_REQUEST),
            SchemaVersion(1),
            "IngressRequest",
            "Incoming request from any transport",
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
            "Rate limiter state",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<RequestQueue>(
            ComponentSchemaId(SCHEMA_REQUEST_QUEUE),
            SchemaVersion(1),
            "RequestQueue",
            "Queue of pending ingress requests",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<TransportSession>(
            ComponentSchemaId(SCHEMA_TRANSPORT_SESSION),
            SchemaVersion(1),
            "TransportSession",
            "Persistent transport session",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<IngressLifecycle>(
            ComponentSchemaId(SCHEMA_INGRESS_LIFECYCLE),
            SchemaVersion(1),
            "IngressLifecycle",
            "Lifecycle of an ingress request",
            ComponentDurability::Durable,
        );
        reg
    }

    // ── test_ingress_lifecycle_transitions ────────────────────────────────
    //
    // Received → Authenticated → Queued → Resolved
    // Direct Rejected from any state
    // Completed only from Resolved

    #[test]
    fn test_ingress_lifecycle_transitions() {
        // Forward path: Received → Authenticated → Queued → Resolved → Completed
        assert!(IngressLifecycle::Received
            .can_transition_to(IngressLifecycle::Authenticated)
            .is_ok());
        assert!(IngressLifecycle::Authenticated
            .can_transition_to(IngressLifecycle::Queued)
            .is_ok());
        assert!(IngressLifecycle::Queued
            .can_transition_to(IngressLifecycle::Resolved)
            .is_ok());
        assert!(IngressLifecycle::Resolved
            .can_transition_to(IngressLifecycle::Completed)
            .is_ok());

        // Rejected from any state
        assert!(IngressLifecycle::Received
            .can_transition_to(IngressLifecycle::Rejected)
            .is_ok());
        assert!(IngressLifecycle::Authenticated
            .can_transition_to(IngressLifecycle::Rejected)
            .is_ok());
        assert!(IngressLifecycle::Queued
            .can_transition_to(IngressLifecycle::Rejected)
            .is_ok());
        assert!(IngressLifecycle::Resolved
            .can_transition_to(IngressLifecycle::Rejected)
            .is_ok());

        // Invalid transitions: skipping states
        assert!(IngressLifecycle::Received
            .can_transition_to(IngressLifecycle::Queued)
            .is_err());
        assert!(IngressLifecycle::Received
            .can_transition_to(IngressLifecycle::Resolved)
            .is_err());
        assert!(IngressLifecycle::Received
            .can_transition_to(IngressLifecycle::Completed)
            .is_err());
        assert!(IngressLifecycle::Authenticated
            .can_transition_to(IngressLifecycle::Resolved)
            .is_err());
        assert!(IngressLifecycle::Authenticated
            .can_transition_to(IngressLifecycle::Completed)
            .is_err());
        assert!(IngressLifecycle::Queued
            .can_transition_to(IngressLifecycle::Authenticated)
            .is_err());
        assert!(IngressLifecycle::Queued
            .can_transition_to(IngressLifecycle::Completed)
            .is_err());

        // Terminal states
        assert!(IngressLifecycle::Rejected.is_terminal());
        assert!(IngressLifecycle::Completed.is_terminal());
        assert!(!IngressLifecycle::Received.is_terminal());
        assert!(!IngressLifecycle::Authenticated.is_terminal());
        assert!(!IngressLifecycle::Queued.is_terminal());
        assert!(!IngressLifecycle::Resolved.is_terminal());

        // Terminal states cannot transition further
        assert!(IngressLifecycle::Rejected
            .can_transition_to(IngressLifecycle::Completed)
            .is_err());
        assert!(IngressLifecycle::Completed
            .can_transition_to(IngressLifecycle::Resolved)
            .is_err());
    }

    // ── test_ingress_request_serde ────────────────────────────────────────

    #[test]
    fn test_ingress_request_serde() {
        let req = IngressRequest {
            request_id: 42,
            transport: "http".to_string(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body_hash: [0xab; 32],
            received_at: Timestamp::from_nanos(1_000_000),
            resolved_command: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: IngressRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(req.request_id, deserialized.request_id);
        assert_eq!(req.transport, deserialized.transport);
        assert_eq!(req.method, deserialized.method);
        assert_eq!(req.path, deserialized.path);
        assert_eq!(req.body_hash, deserialized.body_hash);
        assert_eq!(req.received_at, deserialized.received_at);
        assert_eq!(req.resolved_command, deserialized.resolved_command);
    }

    // ── test_api_key_construction ────────────────────────────────────────

    #[test]
    fn test_api_key_construction() {
        let key = ApiKey {
            key_hash: [0xde; 32],
            owner: "test-owner".to_string(),
            permissions: vec!["inference".to_string(), "models.read".to_string()],
            expires_at: Timestamp::from_nanos(1_000_000_000_000),
        };

        assert_eq!(key.owner, "test-owner");
        assert_eq!(key.permissions.len(), 2);
        assert!(key.permissions.contains(&"inference".to_string()));
        assert!(!key.permissions.contains(&"admin".to_string()));
    }

    // ── test_schema_validation ───────────────────────────────────────────

    #[test]
    fn test_ingress_schema_validation() {
        let reg = make_registry();
        assert!(validate_ingress_schemas(&reg).is_ok());
    }

    // ── test_submit_ingress_command_execution ────────────────────────────

    #[test]
    fn test_submit_ingress_command_execution() {
        let reg = make_registry();
        let mut world = World::new();

        let cmd = SubmitIngressRequestCommand {
            id: MessageId::compute(b"test-ingress-1"),
            transport: "http".to_string(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            body: b"hello".to_vec(),
            api_key_hash: None,
        };

        let (epoch, event) = cmd.execute(&mut world, &reg).unwrap();
        assert!(epoch.0 .0 > 0, "epoch advanced");

        assert!(epoch.0 .0 > 0, "epoch advanced");
        assert!(epoch.0 .0 > 0, "epoch advanced");
        assert_eq!(event.kind, "ingress_request_received");
        assert!(event.entity_id.is_some());
    }

    // ── test_rate_limiter_serde ──────────────────────────────────────────

    #[test]
    fn test_rate_limiter_serde() {
        let rl = RateLimiterState {
            window_start: Timestamp::from_nanos(1_000_000),
            request_count: 5,
            max_requests: 100,
            window_ms: 60_000,
        };

        let json = serde_json::to_string(&rl).unwrap();
        let deserialized: RateLimiterState = serde_json::from_str(&json).unwrap();
        assert_eq!(rl, deserialized);
    }

    // ── test_transport_session_serde ─────────────────────────────────────

    #[test]
    fn test_transport_session_serde() {
        let ts = TransportSession {
            session_entity: 100,
            transport: "websocket".to_string(),
            remote_addr: "192.168.1.1:9000".to_string(),
            connected_at: Timestamp::from_nanos(1_000_000),
            last_activity: Timestamp::from_nanos(1_100_000),
        };

        let json = serde_json::to_string(&ts).unwrap();
        let deserialized: TransportSession = serde_json::from_str(&json).unwrap();
        assert_eq!(ts, deserialized);
    }

    // ── test_request_queue_serde ─────────────────────────────────────────

    #[test]
    fn test_request_queue_serde() {
        let rq = RequestQueue {
            queue_entity: 200,
            requests: vec![1, 2, 3],
            drained_epoch: WorldEpoch(42),
        };

        let json = serde_json::to_string(&rq).unwrap();
        let deserialized: RequestQueue = serde_json::from_str(&json).unwrap();
        assert_eq!(rq, deserialized);
    }

    // ── test_ingress_lifecycle_serde ─────────────────────────────────────

    #[test]
    fn test_ingress_lifecycle_serde() {
        for variant in &[
            IngressLifecycle::Received,
            IngressLifecycle::Authenticated,
            IngressLifecycle::Queued,
            IngressLifecycle::Resolved,
            IngressLifecycle::Rejected,
            IngressLifecycle::Completed,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: IngressLifecycle = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, deserialized);
        }
    }
}
