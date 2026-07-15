use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::lifecycle::DeviceLifecycle;
use crate::ecs::constitutional::lifecycle::{ResidencyLifecycle, SessionLifecycle};
use crate::ecs::constitutional::residency::ModelLifecycle;
use crate::ecs::constitutional::residency::ResidencyDeviceRef;
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use crate::ecs::{Entity, EntityKind, World};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs ──────────────────────────────────────────────────
// Artifact: 1-4, Model/Residency: 5-12, Session: 13+

pub const SCHEMA_SESSION_CONFIG: u64 = 13;
pub const SCHEMA_SESSION_MODELS: u64 = 14;
pub const SCHEMA_SESSION_DEVICES: u64 = 15;
pub const SCHEMA_SESSION_LIFECYCLE: u64 = 16;
pub const SCHEMA_RESIDENCY_MODEL_REF: u64 = 17;

// ── Session Components ────────────────────────────────────────────────────

/// Session configuration — quotas, deadlines, priority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    pub max_tokens: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub batch_size: u64,
    pub priority: u32,
    pub deadline_epochs: u64,
}

/// Set of model entities this session uses.
///
/// The `u64` values correspond to entity identifiers. The canonical Entity
/// equivalent is `Entity(model_id, gen)`. Callers should migrate to using
/// `Entity` for type-safe entity references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionModels(pub Vec<u64>);

/// Set of device entities this session targets.
///
/// The `u64` values correspond to entity identifiers. The canonical Entity
/// equivalent is `Entity(device_id, gen)`. Callers should migrate to using
/// `Entity` for type-safe entity references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDevices(pub Vec<u64>);

/// Typed relationship: links a residency entity to the model it was built from.
/// Needed for session admission to find which models are resident on which devices.
///
/// The `u64` fields correspond to entity identifiers. The canonical Entity
/// equivalent is `Entity(residency_id, gen)` / `Entity(model_id, gen)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidencyModelRef {
    pub residency_id: u64,
    pub model_id: u64,
}

// ── CreateSessionCommand ──────────────────────────────────────────────────

/// Command to admit a new session.
///
/// Entity fields (`model_entities`, `device_entities`) use `u64` identifiers.
/// The canonical Entity equivalent is `Entity(id, gen)`. New callers should
/// prefer `Entity` for type safety.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSessionCommand {
    pub id: MessageId,
    pub config: SessionConfig,
    pub model_entities: Vec<u64>,
    pub device_entities: Vec<u64>,
}

impl CreateSessionCommand {
    /// Validate all schemas are registered for session components.
    pub fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        schema_registry
            .verify_type::<SessionConfig>(ComponentSchemaId(SCHEMA_SESSION_CONFIG))
            .map_err(|e| format!("SessionConfig schema: {e}"))?;
        schema_registry
            .verify_type::<SessionModels>(ComponentSchemaId(SCHEMA_SESSION_MODELS))
            .map_err(|e| format!("SessionModels schema: {e}"))?;
        schema_registry
            .verify_type::<SessionDevices>(ComponentSchemaId(SCHEMA_SESSION_DEVICES))
            .map_err(|e| format!("SessionDevices schema: {e}"))?;
        schema_registry
            .verify_type::<SessionLifecycle>(ComponentSchemaId(SCHEMA_SESSION_LIFECYCLE))
            .map_err(|e| format!("SessionLifecycle schema: {e}"))?;
        schema_registry
            .verify_type::<ResidencyModelRef>(ComponentSchemaId(SCHEMA_RESIDENCY_MODEL_REF))
            .map_err(|e| format!("ResidencyModelRef schema: {e}"))?;
        Ok(())
    }

    /// Preflight: validate every model exists, every device exists and is Ready,
    /// and every model has at least one admissible residency on a requested device.
    pub fn preflight(
        &self,
        world: &World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), SessionError> {
        Self::validate_schemas(schema_registry).map_err(|e| SessionError::SchemaError(e))?;

        if self.model_entities.is_empty() {
            return Err(SessionError::NoModels);
        }
        if self.device_entities.is_empty() {
            return Err(SessionError::NoDevices);
        }

        // Validate every model entity
        for &model in &self.model_entities {
            let entity = crate::ecs::CompEntity(model);
            if !world.has_entity(entity) {
                return Err(SessionError::ModelNotFound(model));
            }
            if world.entity_kind(entity) != Some(EntityKind::Model) {
                return Err(SessionError::ModelNotFound(model));
            }
            // Model must be Deployable (ready for use)
            if let Some(lifecycle) = world.get_component::<ModelLifecycle>(entity) {
                if !matches!(
                    lifecycle,
                    ModelLifecycle::Deployable
                        | ModelLifecycle::Created
                        | ModelLifecycle::Validated
                ) {
                    return Err(SessionError::ModelNotAdmissible(model));
                }
            }
        }

        // Validate every device entity
        for &device in &self.device_entities {
            let entity = crate::ecs::CompEntity(device);
            if !world.has_entity(entity) {
                return Err(SessionError::DeviceNotFound(device));
            }
            if world.entity_kind(entity) != Some(EntityKind::Device) {
                return Err(SessionError::DeviceNotFound(device));
            }
            // Device must be Ready
            if let Some(lifecycle) = world.get_component::<DeviceLifecycle>(entity) {
                if !lifecycle.is_available() {
                    return Err(SessionError::DeviceNotReady(device));
                }
            } else {
                return Err(SessionError::DeviceNotReady(device));
            }
        }

        // For each model, ensure at least one residency exists on a requested device
        for &model in &self.model_entities {
            let has_admissible_residency = self
                .device_entities
                .iter()
                .any(|&device| Self::find_residency(world, model, device).is_some());
            if !has_admissible_residency {
                return Err(SessionError::ModelNotAdmissible(model));
            }
        }

        Ok(())
    }

    /// Find a residency entity for a given model on a given device.
    fn find_residency(world: &World, model: u64, device: u64) -> Option<u64> {
        for entity in world.entities_of_kind(EntityKind::Residency) {
            // Check this residency targets the right device
            if let Some(dev_ref) = world.get_component::<ResidencyDeviceRef>(entity) {
                if dev_ref.device_id != device {
                    continue;
                }
            } else {
                continue;
            }
            // Check residency lifecycle is Resident or Binding (admissible)
            if let Some(lifecycle) = world.get_component::<ResidencyLifecycle>(entity) {
                if !matches!(
                    lifecycle,
                    ResidencyLifecycle::Resident | ResidencyLifecycle::Binding
                ) {
                    continue;
                }
            } else {
                continue;
            }
            // Verify this residency belongs to this model via ResidencyModelRef
            if let Some(model_ref) = world.get_component::<ResidencyModelRef>(entity) {
                if model_ref.model_id == model {
                    return Some(entity.0);
                }
            }
        }
        None
    }

    /// Find an existing session with the same model+device combination.
    /// Linear scan — in production, maintain index.
    fn find_existing_session(world: &World, models: &[u64], devices: &[u64]) -> Option<u64> {
        for entity in world.entities_of_kind(EntityKind::Session) {
            if let Some(session_models) =
                world.get_component::<SessionModels>(crate::ecs::CompEntity(entity.0))
            {
                if session_models.0.len() != models.len() {
                    continue;
                }
                if !models.iter().all(|m| session_models.0.contains(m)) {
                    continue;
                }
            } else {
                continue;
            }
            if let Some(session_devices) =
                world.get_component::<SessionDevices>(crate::ecs::CompEntity(entity.0))
            {
                if session_devices.0.len() != devices.len() {
                    continue;
                }
                if !devices.iter().all(|d| session_devices.0.contains(d)) {
                    continue;
                }
            } else {
                continue;
            }
            return Some(entity.0);
        }
        None
    }

    /// Execute session admission: validate, create session entity with all
    /// components, commit atomically.
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), SessionError> {
        // 0. Preflight
        self.preflight(world, schema_registry)?;

        // 1. Idempotency: check if session already exists for this model+device set
        if let Some(existing) =
            Self::find_existing_session(world, &self.model_entities, &self.device_entities)
        {
            // Update components on existing session
            let mut txn = WorldTxn::new(world);
            txn.put_durable(Entity(existing, 0), self.config.clone());

            let event = DomainEvent {
                id: self.id,
                kind: "session_admitted".to_string(),
                entity_id: Some(EntityKindId(existing)),
                payload: serde_json::json!({
                    "session_id": existing,
                    "idempotent": true,
                }),
            };
            txn.emit_event(event.clone());
            let epoch = world.transit(txn).map_err(SessionError::CommitFailed)?;
            return Ok((epoch, event));
        }

        // 2. Reserve entity ID
        let session_id = WorldTxn::next_entity_id(world);

        let mut txn = WorldTxn::new(world);

        // 3. Spawn session entity
        txn.stage_spawn(session_id, EntityKind::Session);

        // 4. Attach components
        txn.put_durable(session_id, self.config.clone());
        txn.put_durable(session_id, SessionModels(self.model_entities.clone()));
        txn.put_durable(session_id, SessionDevices(self.device_entities.clone()));
        txn.put_durable(session_id, SessionLifecycle::Created);

        // 5. Emit event
        let event = DomainEvent {
            id: self.id,
            kind: "session_admitted".to_string(),
            entity_id: Some(EntityKindId(session_id.id())),
            payload: serde_json::json!({
                "session_id": session_id.id(),
                "models": self.model_entities,
                "devices": self.device_entities,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(SessionError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── SessionLifecycleTransition ────────────────────────────────────────────

/// Command to transition a session lifecycle state.
///
/// The `session_entity` field uses a `u64` identifier. The canonical Entity
/// equivalent is `Entity(session_id, gen)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionSessionCommand {
    pub id: MessageId,
    pub session_entity: Entity,
    pub target: SessionLifecycle,
}

impl TransitionSessionCommand {
    pub fn execute(
        self,
        world: &mut World,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), SessionError> {
        Self::validate_schemas(schema_registry).map_err(|e| SessionError::SchemaError(e))?;

        if !world.has_entity(self.session_entity) {
            return Err(SessionError::SessionNotFound(self.session_entity.id()));
        }
        if world.entity_kind(self.session_entity) != Some(EntityKind::Session) {
            return Err(SessionError::SessionNotFound(self.session_entity.id()));
        }

        // Read current lifecycle
        let current = world
            .get_component::<SessionLifecycle>(self.session_entity)
            .ok_or(SessionError::SessionNotFound(self.session_entity.id()))?;

        // Validate transition
        current
            .can_transition_to(self.target)
            .map_err(|e| SessionError::InvalidTransition(e))?;

        let mut txn = WorldTxn::new(world);
        txn.put_durable(self.session_entity, self.target);

        let event = DomainEvent {
            id: self.id,
            kind: format!("session_{}", self.target.name()),
            entity_id: Some(EntityKindId(self.session_entity.id())),
            payload: serde_json::json!({
                "session_id": self.session_entity.id(),
                "from": format!("{:?}", current),
                "to": format!("{:?}", self.target),
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(SessionError::CommitFailed)?;
        Ok((epoch, event))
    }

    fn validate_schemas(schema_registry: &SchemaRegistry) -> Result<(), String> {
        CreateSessionCommand::validate_schemas(schema_registry)
    }
}

impl SessionLifecycle {
    /// Short machine-readable name for event routing.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Admitted => "admitted",
            Self::Active => "active",
            Self::Quiescing => "quiescing",
            Self::Saving => "saving",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Releasing => "releasing",
            Self::Released => "released",
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SessionError {
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("no models specified")]
    NoModels,
    #[error("no devices specified")]
    NoDevices,
    #[error("model entity {0} not found")]
    ModelNotFound(u64),
    #[error("model entity {0} not admissible (no resident residency)")]
    ModelNotAdmissible(u64),
    #[error("device entity {0} not found")]
    DeviceNotFound(u64),
    #[error("device entity {0} not Ready")]
    DeviceNotReady(u64),
    #[error("session entity {0} not found")]
    SessionNotFound(u64),
    #[error("invalid lifecycle transition: {0}")]
    InvalidTransition(crate::ecs::constitutional::lifecycle::LifecycleError),
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
}

// ── Replay helpers ─────────────────────────────────────────────────────────

/// Reconstruct a session entity from a `session_admitted` event.
/// Restores config, model refs, device refs, lifecycle. Idempotent.
pub fn replay_session_admitted(
    world: &mut World,
    event: &DomainEvent,
) -> Result<CommittedEpoch, SessionError> {
    let session_id = event
        .entity_id
        .map(|id| id.0)
        .or_else(|| event.payload.get("session_id").and_then(|v| v.as_u64()))
        .ok_or_else(|| SessionError::SchemaError("missing session_id in event".into()))?;

    let entity = Entity(session_id, 0);

    let models: Vec<u64> = event
        .payload
        .get("models")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();

    let devices: Vec<u64> = event
        .payload
        .get("devices")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();

    let mut txn = WorldTxn::new(world);

    if !world.has_entity(entity) {
        txn.stage_spawn(entity, EntityKind::Session);
    }

    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_SESSION_CONFIG),
        SchemaVersion(1),
        SessionConfig {
            max_tokens: 4096,
            max_input_tokens: 2048,
            max_output_tokens: 2048,
            batch_size: 1,
            priority: 1,
            deadline_epochs: 100,
        },
    );
    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_SESSION_MODELS),
        SchemaVersion(1),
        SessionModels(models),
    );
    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_SESSION_DEVICES),
        SchemaVersion(1),
        SessionDevices(devices),
    );
    txn.add_component(
        entity,
        ComponentSchemaId(SCHEMA_SESSION_LIFECYCLE),
        SchemaVersion(1),
        SessionLifecycle::Created,
    );

    let epoch = world.transit(txn).map_err(SessionError::CommitFailed)?;
    Ok(epoch)
}

// ── Component impls ───────────────────────────────────────────────────────

impl crate::ecs::Component for SessionConfig {}
impl crate::ecs::Component for SessionModels {}
impl crate::ecs::Component for SessionDevices {}
impl crate::ecs::Component for SessionLifecycle {}
impl crate::ecs::Component for ResidencyModelRef {}

// ── Entity conversion helper ────────────────────────────────────────────────

/// Convert a legacy `u64` entity identifier to the canonical `Entity` type.
///
/// Uses generation `0` for backward compatibility with the legacy
/// `World`/`CompEntity` storage. Callers migrating to the new
/// `World`+`Entity(u64, u32)` API should replace this with proper
/// generation-aware entity construction.
#[allow(dead_code)]
pub(crate) fn as_entity(id: u64) -> Entity {
    Entity(id, 0)
}

// ── Constitutional classification ────────────────────────────────────────

impl ClassifiedComponent for SessionConfig {
    type Class = DurableClass;
}
impl DurableComponent for SessionConfig {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.session",
        id: 13,
        version: 1,
    };
}

impl ClassifiedComponent for SessionModels {
    type Class = DurableClass;
}
impl DurableComponent for SessionModels {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.session",
        id: 14,
        version: 1,
    };
}

impl ClassifiedComponent for SessionDevices {
    type Class = DurableClass;
}
impl DurableComponent for SessionDevices {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.session",
        id: 15,
        version: 1,
    };
}

impl ClassifiedComponent for SessionLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for SessionLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.session",
        id: 16,
        version: 1,
    };
}

impl ClassifiedComponent for ResidencyModelRef {
    type Class = DurableClass;
}
impl DurableComponent for ResidencyModelRef {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.session",
        id: 17,
        version: 1,
    };
}
