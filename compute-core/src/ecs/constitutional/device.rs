use crate::ecs::constitutional::command::{DomainEvent, EffectOutcome, EffectRequest};
use crate::ecs::constitutional::driver::BackendCapability;
use crate::ecs::constitutional::lifecycle::DeviceLifecycle;
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn, WorldTxnError};
use crate::ecs::CompWorld;
use serde::{Deserialize, Serialize};

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
            kind: crate::ecs::constitutional::command::EffectKind::EnumerateDrivers,
            params: serde_json::json!({"factory": self.factory_name}),
        }
    }

    /// Validate outcome, create or update device entity, commit.
    pub fn execute(
        self,
        world: &mut CompWorld,
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
                txn.stage_spawn(id, crate::ecs::EntityKind::Device);
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
pub fn find_device_by_stable_id(_world: &CompWorld, _stable_id: &DeviceStableId) -> Option<u64> {
    // TODO: maintain StableId → EntityId reverse index
    None
}

// ── Initialization Command ─────────────────────────────────────────────────

/// Command to initialize a discovered device (create a handle).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitializeDeviceCommand {
    pub id: MessageId,
    pub device_entity: u64,
    pub factory_name: String,
}

impl InitializeDeviceCommand {
    pub fn to_effect_request(&self) -> EffectRequest {
        EffectRequest {
            id: MessageId::compute(format!("init:{}", self.device_entity).as_bytes()),
            kind: crate::ecs::constitutional::command::EffectKind::CreateDevice,
            params: serde_json::json!({"device": self.device_entity}),
        }
    }
}

// ── Errors ─────────────────────────────────────────────────────────────────

// ── Replay ────────────────────────────────────────────────────────────────

/// Replay a `device_discovered` event to reconstruct a device entity.
pub fn replay_device_discovered(
    world: &mut CompWorld,
    event: &DomainEvent,
) -> Result<(CommittedEpoch, u64), DeviceError> {
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
        txn.stage_spawn(id, crate::ecs::EntityKind::Device);
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
            crate::ecs::constitutional::lifecycle::DeviceLifecycle::Discovered,
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

impl crate::ecs::Component for DeviceStableId {}
impl crate::ecs::Component for DriverFactoryId {}
impl crate::ecs::Component for BackendFamily {}
impl crate::ecs::Component for DriverVersion {}
impl crate::ecs::Component for DeviceCapabilities {}
impl crate::ecs::Component for DeviceMemoryLimits {}
impl crate::ecs::Component for DeviceTopology {}
impl crate::ecs::Component for DeviceLifecycle {}
impl crate::ecs::Component for DeviceHealth {}
impl crate::ecs::Component for DesiredDeviceState {}
impl crate::ecs::Component for ObservedDeviceState {}
impl crate::ecs::Component for LastObservation {}
impl crate::ecs::Component for RuntimeHandleKey {}
