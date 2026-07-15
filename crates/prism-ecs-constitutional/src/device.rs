use crate::command::{DomainEvent, EffectOutcome, EffectRequest};
use crate::driver::BackendCapability;
use crate::lifecycle::DeviceLifecycle;
use crate::schema::{ComponentDurability, SchemaRegistry};
use crate::types::*;
use crate::world_txn::{
    ClassifiedComponent, CommittedEpoch, DurableClass, DurableComponent, WorldTxn, WorldTxnError,
};
use prism_ecs_core::{Entity, World};

use serde::{Deserialize, Serialize};
use crate::world_txn::WorldTransitExt;

/// Stable hardware identity — backend-specific but deterministic.
/// PCIe: domain:bus:dev:func + vendor:device ID
/// Metal: platform registry ID + hardware family
/// Remote: cryptographic peer identity
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceStableId(pub String);

impl DeviceStableId {
    pub fn pcie(domain: u16, bus: u8, dev: u8, func: u8, vendor: u16, device: u16) -> Self {
        Self(format!(
            "pcie:{:04x}:{:02x}:{:02x}.{:02x}:{:04x}:{:04x}",
            domain, bus, dev, func, vendor, device
        ))
    }
}

/// The factory that discovered this device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DriverFactoryId(pub String);

/// Backend family identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendFamily(pub String);

/// Driver version — semver-like.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DriverVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

/// Device capability set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCapabilities(pub Vec<BackendCapability>);

/// Memory limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceMemoryLimits {
    pub total_bytes: u64,
    pub max_alloc_bytes: u64,
}

/// Compute topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceTopology {
    pub compute_units: u32,
    pub description: String,
}

/// Device health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Desired device state — what the system wants the device to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DesiredDeviceState {
    Active,
    Standby,
    Offline,
    Removed,
}

/// Observed device state — what was last observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservedDeviceState {
    Present,
    Absent,
    Degraded,
    Error,
}

/// Last successful observation timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LastObservation(pub Timestamp);

/// Ephemeral runtime handle key — NOT durable. Stored as ephemeral component.
/// The actual native handle (Metal device, ROCm handle, etc.) lives only in
/// the execution plane, referenced by this opaque key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuntimeHandleKey(pub String);

impl RuntimeHandleKey {
    pub fn ephemeral_durability() -> ComponentDurability {
        ComponentDurability::Ephemeral
    }
}

// ── Component Schema IDs (69-80) ─────────────────────────────────────────

pub const SCHEMA_DEVICE_STABLE_ID: u64 = 69;
pub const SCHEMA_DRIVER_FACTORY_ID: u64 = 70;
pub const SCHEMA_BACKEND_FAMILY: u64 = 71;
pub const SCHEMA_DEVICE_CAPABILITIES: u64 = 72;
pub const SCHEMA_DEVICE_MEMORY_LIMITS: u64 = 73;
pub const SCHEMA_DEVICE_TOPOLOGY: u64 = 74;
pub const SCHEMA_DEVICE_HEALTH: u64 = 75;
pub const SCHEMA_DEVICE_LIFECYCLE: u64 = 76;
pub const SCHEMA_DESIRED_DEVICE_STATE: u64 = 77;
pub const SCHEMA_OBSERVED_DEVICE_STATE: u64 = 78;
pub const SCHEMA_LAST_OBSERVATION: u64 = 79;
pub const SCHEMA_RUNTIME_HANDLE_KEY: u64 = 80;

// ── Discovery Command ────────────────────────────────────────────────────

/// Command to discover devices via a specific driver factory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoverDevicesCommand {
    pub id: MessageId,
    pub factory_name: String,
}

impl DiscoverDevicesCommand {
    /// Create the enumeration effect request.
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("enum:{}", self.factory_name).as_bytes()),
            kind: crate::command::EffectKind::EnumerateDrivers,
            params: serde_json::json!({"factory": self.factory_name}),
        }
    }

    /// Validate outcome, create or update device entity, commit.
    pub fn execute(
        self,
        world: &mut World,
        _schema_registry: &SchemaRegistry,
        outcome: EffectOutcome,
    ) -> Result<(CommittedEpoch, Vec<DomainEvent>), DeviceError> {
        if !outcome.success {
            return Err(DeviceError::DiscoveryFailed);
        }
        let expected_id = self.to_effect_request().id;
        if outcome.request_id != expected_id {
            return Err(DeviceError::RequestMismatch);
        }

        // Verify schemas
        // (register_for_type calls would go here — for now assume caller registered)

        let mut txn = WorldTxn::new(world);
        let mut events = Vec::new();

        // Parse outcome into device records
        let devices: Vec<serde_json::Value> = outcome
            .output
            .get("devices")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        for device_info in &devices {
            let stable_id = DeviceStableId(
                device_info
                    .get("stable_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            );

            // Check if device already exists — idempotent update
            let _entity_id = if let Some(existing) = find_device_by_stable_id(world, &stable_id) {
                existing
            } else {
                let id = WorldTxn::next_entity_id(world);
                txn.stage_spawn(id, prism_ecs_core::EntityKind::Device);
                id
            };

            // Attach/replace components
            // (component insertion via txn.add_component would go here once
            // the schema binding and generational entity API support it)

            events.push(DomainEvent {
                id: MessageId::compute(format!("dev:{:?}", stable_id).as_bytes()),
                kind: "device_discovered".to_string(),
                entity_id: None,
                payload: serde_json::json!({"stable_id": stable_id.0}),
            });
        }

        for event in events.clone() {
            txn.emit_event(event);
        }

        let epoch = world.transit(txn).map_err(DeviceError::CommitFailed)?;
        Ok((epoch, events))
    }
}

/// Find a device entity by its stable ID.
/// Linear scan — in production, maintain a reverse index.
///
/// Returns the entity ID of the matching device, if any. See [`Entity`] for the
/// canonical generational entity handle.
pub fn find_device_by_stable_id(_world: &World, _stable_id: &DeviceStableId) -> Option<Entity> {
    // TODO: maintain StableId → EntityId reverse index
    None
}

// ── Initialization Command ─────────────────────────────────────────────────

/// Command to initialize a discovered device (create a handle).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeDeviceCommand {
    pub id: MessageId,
    /// Entity ID of the device to initialize. Referenced via [`Entity`] in the
    /// canonical API.
    pub device_entity: u64,
    pub factory_name: String,
}

impl InitializeDeviceCommand {
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("init:{}", self.device_entity).as_bytes()),
            kind: crate::command::EffectKind::CreateDevice,
            params: serde_json::json!({"device": self.device_entity}),
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

// ── Replay ────────────────────────────────────────────────────────────────

/// Replay a `device_discovered` event to reconstruct a device entity.
///
/// Returns the committed epoch and the entity ID (u64) of the reconstructed
/// device entity. See [`Entity`] for the canonical generational handle.
pub fn replay_device_discovered(
    world: &mut World,
    event: &DomainEvent,
) -> Result<(CommittedEpoch, Entity), DeviceError> {
    let stable_id_str = event
        .payload
        .get("stable_id")
        .and_then(|v| v.as_str())
        .unwrap_or("replay");
    // Scan for existing device with this stable ID
    let entity_id = if let Some(existing) =
        find_device_by_stable_id(world, &DeviceStableId(stable_id_str.to_string()))
    {
        existing
    } else {
        let id = WorldTxn::next_entity_id(world);
        let mut txn = WorldTxn::new(world);
        txn.stage_spawn(id, prism_ecs_core::EntityKind::Device);
        txn.add_component(
            id,
            ComponentSchemaId(1),
            SchemaVersion(1),
            DeviceStableId(stable_id_str.to_string()),
        );
        txn.add_component(
            id,
            ComponentSchemaId(1),
            SchemaVersion(1),
            crate::lifecycle::DeviceLifecycle::Discovered,
        );
        let epoch = world
            .transit(txn)
            .map_err(|e| DeviceError::CommitFailed(e))?;
        return Ok((epoch, id));
    };
    Ok((CommittedEpoch(WorldEpoch(0)), entity_id))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeviceError {
    #[error("discovery effect failed")]
    DiscoveryFailed,
    #[error("request ID mismatch")]
    RequestMismatch,
    #[error("commit failed: {0}")]
    CommitFailed(WorldTxnError),
    #[error("device not found")]
    DeviceNotFound,
}

// ── Component impls ────────────────────────────────────────────────────────

impl prism_ecs_core::Component for DeviceStableId {}
impl prism_ecs_core::Component for DriverFactoryId {}
impl prism_ecs_core::Component for BackendFamily {}
impl prism_ecs_core::Component for DriverVersion {}
impl prism_ecs_core::Component for DeviceCapabilities {}
impl prism_ecs_core::Component for DeviceMemoryLimits {}
impl prism_ecs_core::Component for DeviceTopology {}
impl prism_ecs_core::Component for DeviceLifecycle {}
impl prism_ecs_core::Component for DeviceHealth {}
impl prism_ecs_core::Component for DesiredDeviceState {}
impl prism_ecs_core::Component for ObservedDeviceState {}
impl prism_ecs_core::Component for LastObservation {}
impl prism_ecs_core::Component for RuntimeHandleKey {}

// ── ClassifiedComponent / DurableComponent impls ─────────────────────────

impl ClassifiedComponent for DeviceStableId {
    type Class = DurableClass;
}
impl DurableComponent for DeviceStableId {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_STABLE_ID as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DriverFactoryId {
    type Class = DurableClass;
}
impl DurableComponent for DriverFactoryId {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DRIVER_FACTORY_ID as u32,
        version: 1,
    };
}

impl ClassifiedComponent for BackendFamily {
    type Class = DurableClass;
}
impl DurableComponent for BackendFamily {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_BACKEND_FAMILY as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DeviceCapabilities {
    type Class = DurableClass;
}
impl DurableComponent for DeviceCapabilities {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_CAPABILITIES as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DeviceMemoryLimits {
    type Class = DurableClass;
}
impl DurableComponent for DeviceMemoryLimits {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_MEMORY_LIMITS as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DeviceTopology {
    type Class = DurableClass;
}
impl DurableComponent for DeviceTopology {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_TOPOLOGY as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DeviceHealth {
    type Class = DurableClass;
}
impl DurableComponent for DeviceHealth {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_HEALTH as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DeviceLifecycle {
    type Class = DurableClass;
}
impl DurableComponent for DeviceLifecycle {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DEVICE_LIFECYCLE as u32,
        version: 1,
    };
}

impl ClassifiedComponent for DesiredDeviceState {
    type Class = DurableClass;
}
impl DurableComponent for DesiredDeviceState {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_DESIRED_DEVICE_STATE as u32,
        version: 1,
    };
}

impl ClassifiedComponent for ObservedDeviceState {
    type Class = DurableClass;
}
impl DurableComponent for ObservedDeviceState {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_OBSERVED_DEVICE_STATE as u32,
        version: 1,
    };
}

impl ClassifiedComponent for LastObservation {
    type Class = DurableClass;
}
impl DurableComponent for LastObservation {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_LAST_OBSERVATION as u32,
        version: 1,
    };
}

impl ClassifiedComponent for RuntimeHandleKey {
    type Class = DurableClass;
}
impl DurableComponent for RuntimeHandleKey {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.device",
        id: SCHEMA_RUNTIME_HANDLE_KEY as u32,
        version: 1,
    };
}
