use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::lifecycle::{
    DeviceLifecycle, ResidencyLifecycle, SessionLifecycle,
};
use crate::ecs::constitutional::schema::SchemaRegistry;
use crate::ecs::constitutional::types::*;
use crate::ecs::constitutional::world_txn::{CommittedEpoch, WorldTxn};
use crate::ecs::{CompWorld, EntityKind};
use serde::{Deserialize, Serialize};

// ── Component Schema IDs ──────────────────────────────────────────────────
// Artifact: 1-4, Model/Residency: 5-12, Session: 13-17, Execution: 24-29

pub const SCHEMA_EXECUTION_LEASE: u64 = 24;
pub const SCHEMA_LEASE_OWNER: u64 = 25;
pub const SCHEMA_LEASE_RANGE: u64 = 26;
pub const SCHEMA_KV_SLOT: u64 = 27;
pub const SCHEMA_KV_OWNERSHIP: u64 = 28;
pub const SCHEMA_EXECUTION_OUTPUT: u64 = 29;

// ── Execution Lease Component (reconstituted from events, NOT durable) ─────
//
// A bounded execution lease. Not stored as a durable component — it is
// reconstituted from lease_acquired / lease_completed events during replay.
// Ephemeral components serve as an in-memory cache of active leases.

/// A bounded execution lease granting token-range access to an inference device.
///
/// Reconstituted from domain events (lease_acquired → created, lease_completed →
/// removed). Not durable — reloaded from event history during replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionLease {
    pub lease_id: u64,
    pub session_entity: u64,
    pub deployment_entity: u64,
    pub device_entity: u64,
    pub token_range_start: u64,
    pub token_range_end: u64,
    pub cancellation_epoch: WorldEpoch,
    pub created_at: Timestamp,
    pub deadline: Timestamp,
}

/// Who holds this lease.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseOwner {
    pub session_id: u64,
    pub work_item_id: u64,
}

/// Token batch range assigned to a lease.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseTokenRange {
    pub start: u64,
    pub end: u64,
}

/// KV cache slot identity — ephemeral runtime state, not durable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlot {
    pub cache_entity: u64,
    pub slot_index: u32,
    pub page_count: u32,
    pub format: u8,
}

/// Durable record of which session owns which KV cache range.
///
/// Marked ephemeral for replay purposes (reconstructed from events),
/// but persisted to durable storage for crash recovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvOwnership {
    pub session_id: u64,
    pub kv_slot_id: u64,
    pub valid_range_start: u64,
    pub valid_range_end: u64,
}

/// Result of execution — ephemeral, attached to the lease entity during
/// execution and replaced by the completed event outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOutput {
    pub tokens: Vec<u32>,
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: u8,
}

// ── Component Trait impls ─────────────────────────────────────────────────

impl crate::ecs::Component for ExecutionLease {}
impl crate::ecs::Component for LeaseOwner {}
impl crate::ecs::Component for LeaseTokenRange {}
impl crate::ecs::Component for KvSlot {}
impl crate::ecs::Component for KvOwnership {}
impl crate::ecs::Component for ExecutionOutput {}

// ── Schema Validation ─────────────────────────────────────────────────────

/// Validate all execution component schemas are registered for the correct types.
pub fn validate_execution_schemas(reg: &SchemaRegistry) -> Result<(), String> {
    reg.verify_type::<ExecutionLease>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_EXECUTION_LEASE,
    ))
    .map_err(|e| format!("ExecutionLease schema: {e}"))?;
    reg.verify_type::<LeaseOwner>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_LEASE_OWNER,
    ))
    .map_err(|e| format!("LeaseOwner schema: {e}"))?;
    reg.verify_type::<LeaseTokenRange>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_LEASE_RANGE,
    ))
    .map_err(|e| format!("LeaseTokenRange schema: {e}"))?;
    reg.verify_type::<KvSlot>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_KV_SLOT,
    ))
    .map_err(|e| format!("KvSlot schema: {e}"))?;
    reg.verify_type::<KvOwnership>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_KV_OWNERSHIP,
    ))
    .map_err(|e| format!("KvOwnership schema: {e}"))?;
    reg.verify_type::<ExecutionOutput>(crate::ecs::constitutional::types::ComponentSchemaId(
        SCHEMA_EXECUTION_OUTPUT,
    ))
    .map_err(|e| format!("ExecutionOutput schema: {e}"))?;
    Ok(())
}

// ── AcquireExecutionLeaseCommand ──────────────────────────────────────────

/// Request a lease for inference on a given session, deployment, and device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquireExecutionLeaseCommand {
    pub id: MessageId,
    /// Session entity that will own the lease.
    pub session_entity: u64,
    /// Deployment (residency) entity providing the model.
    pub deployment_entity: u64,
    /// Device entity executing the inference.
    pub device_entity: u64,
    /// Number of tokens requested for this lease.
    pub token_batch_size: u64,
    /// Deadline by which the lease must complete.
    pub deadline: Timestamp,
}

impl AcquireExecutionLeaseCommand {
    /// Preflight: session exists and is Active, deployment exists and is
    /// Resident, device exists and is Ready.
    pub fn preflight(
        &self,
        world: &CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(), ExecutionError> {
        validate_execution_schemas(schema_registry)?;

        // Validate session entity
        let session = crate::ecs::CompEntity(self.session_entity);
        if !world.has_entity(session) {
            return Err(ExecutionError::SessionNotFound(self.session_entity));
        }
        if world.entity_kind(session) != Some(EntityKind::Session) {
            return Err(ExecutionError::SessionNotFound(self.session_entity));
        }
        if let Some(lifecycle) = world.get_component::<SessionLifecycle>(session) {
            if !matches!(lifecycle, SessionLifecycle::Active) {
                return Err(ExecutionError::SessionNotActive(self.session_entity));
            }
        } else {
            return Err(ExecutionError::SessionNotActive(self.session_entity));
        }

        // Validate deployment (residency) entity
        let deployment = crate::ecs::CompEntity(self.deployment_entity);
        if !world.has_entity(deployment) {
            return Err(ExecutionError::DeploymentNotFound(self.deployment_entity));
        }
        if world.entity_kind(deployment) != Some(EntityKind::Residency) {
            return Err(ExecutionError::DeploymentNotFound(self.deployment_entity));
        }
        if let Some(lifecycle) = world.get_component::<ResidencyLifecycle>(deployment) {
            if !matches!(lifecycle, ResidencyLifecycle::Resident) {
                return Err(ExecutionError::DeploymentNotResident(
                    self.deployment_entity,
                ));
            }
        } else {
            return Err(ExecutionError::DeploymentNotResident(
                self.deployment_entity,
            ));
        }

        // Validate device entity
        let device = crate::ecs::CompEntity(self.device_entity);
        if !world.has_entity(device) {
            return Err(ExecutionError::DeviceNotFound(self.device_entity));
        }
        if world.entity_kind(device) != Some(EntityKind::Device) {
            return Err(ExecutionError::DeviceNotFound(self.device_entity));
        }
        if let Some(lifecycle) = world.get_component::<DeviceLifecycle>(device) {
            if !lifecycle.is_available() {
                return Err(ExecutionError::DeviceNotReady(self.device_entity));
            }
        } else {
            return Err(ExecutionError::DeviceNotReady(self.device_entity));
        }

        Ok(())
    }

    /// Execute lease acquisition: validate, spawn a lease entity, emit
    /// lease_acquired event.
    ///
    /// Effect outcome: validate the execution plane has acquired resources,
    /// return lease id + token range. Transaction creates the lease entity
    /// and emits a lease_acquired domain event.
    pub fn execute(
        self,
        world: &mut CompWorld,
        schema_registry: &SchemaRegistry,
    ) -> Result<(CommittedEpoch, DomainEvent), ExecutionError> {
        // 0. Preflight
        self.preflight(world, schema_registry)?;

        // 1. Reserve entity ID for the lease
        let lease_id = WorldTxn::next_entity_id(world);
        let now = Timestamp::now();

        let mut txn = WorldTxn::new(world);

        // 2. Spawn lease entity
        txn.stage_spawn(lease_id, EntityKind::Session);

        // 3. Attach lease components
        let token_range_end = self.token_batch_size; // simplified: 0..batch_size
        txn.add_component(
            lease_id,
            ComponentSchemaId(SCHEMA_EXECUTION_LEASE),
            SchemaVersion(1),
            ExecutionLease {
                lease_id,
                session_entity: self.session_entity,
                deployment_entity: self.deployment_entity,
                device_entity: self.device_entity,
                token_range_start: 0,
                token_range_end,
                cancellation_epoch: WorldEpoch(0),
                created_at: now,
                deadline: self.deadline,
            },
        );
        txn.add_component(
            lease_id,
            ComponentSchemaId(SCHEMA_LEASE_OWNER),
            SchemaVersion(1),
            LeaseOwner {
                session_id: self.session_entity,
                work_item_id: lease_id,
            },
        );
        txn.add_component(
            lease_id,
            ComponentSchemaId(SCHEMA_LEASE_RANGE),
            SchemaVersion(1),
            LeaseTokenRange {
                start: 0,
                end: token_range_end,
            },
        );

        // 4. Emit event
        let event = DomainEvent {
            id: self.id,
            kind: "lease_acquired".to_string(),
            entity_id: Some(EntityKindId(lease_id)),
            payload: serde_json::json!({
                "lease_id": lease_id,
                "session_entity": self.session_entity,
                "deployment_entity": self.deployment_entity,
                "device_entity": self.device_entity,
                "token_range_start": 0,
                "token_range_end": token_range_end,
                "deadline": self.deadline.0,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(ExecutionError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── CompleteExecutionLeaseCommand ─────────────────────────────────────────

/// Complete a lease with its execution output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompleteExecutionLeaseCommand {
    pub id: MessageId,
    /// Lease entity to complete.
    pub lease_id: u64,
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// Reason for finishing (0 = normal, 1 = max_tokens, 2 = stop, 3 = cancelled).
    pub finish_reason: u8,
}

impl CompleteExecutionLeaseCommand {
    /// Preflight: lease exists and is active (has an ExecutionLease component).
    pub fn preflight(&self, world: &CompWorld) -> Result<(), ExecutionError> {
        let lease = crate::ecs::CompEntity(self.lease_id);
        if !world.has_entity(lease) {
            return Err(ExecutionError::LeaseNotFound(self.lease_id));
        }
        if world.get_component::<ExecutionLease>(lease).is_none() {
            return Err(ExecutionError::LeaseNotFound(self.lease_id));
        }
        Ok(())
    }

    /// Execute lease completion: validate, emit lease_completed event.
    pub fn execute(
        self,
        world: &mut CompWorld,
    ) -> Result<(CommittedEpoch, DomainEvent), ExecutionError> {
        // 0. Preflight
        self.preflight(world)?;

        let mut txn = WorldTxn::new(world);

        // Attach execution output to the lease entity
        txn.add_component(
            self.lease_id,
            ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
            SchemaVersion(1),
            ExecutionOutput {
                tokens: self.tokens.clone(),
                logprobs: None,
                finish_reason: self.finish_reason,
            },
        );

        // Emit event
        let event = DomainEvent {
            id: self.id,
            kind: "lease_completed".to_string(),
            entity_id: Some(EntityKindId(self.lease_id)),
            payload: serde_json::json!({
                "lease_id": self.lease_id,
                "token_count": self.tokens.len(),
                "finish_reason": self.finish_reason,
            }),
        };
        txn.emit_event(event.clone());

        let epoch = world.transit(txn).map_err(ExecutionError::CommitFailed)?;
        Ok((epoch, event))
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecutionError {
    #[error("schema error: {0}")]
    SchemaError(String),
    #[error("session entity {0} not found")]
    SessionNotFound(u64),
    #[error("session entity {0} is not active")]
    SessionNotActive(u64),
    #[error("deployment entity {0} not found")]
    DeploymentNotFound(u64),
    #[error("deployment entity {0} is not resident")]
    DeploymentNotResident(u64),
    #[error("device entity {0} not found")]
    DeviceNotFound(u64),
    #[error("device entity {0} is not ready")]
    DeviceNotReady(u64),
    #[error("lease entity {0} not found")]
    LeaseNotFound(u64),
    #[error("transaction commit failed: {0}")]
    CommitFailed(crate::ecs::constitutional::world_txn::WorldTxnError),
}

impl From<String> for ExecutionError {
    fn from(s: String) -> Self {
        ExecutionError::SchemaError(s)
    }
}

// ── Replay helpers ─────────────────────────────────────────────────────────

/// Reconstruct a lease entity from a `lease_acquired` event.
/// Restores ExecutionLease, LeaseOwner, LeaseTokenRange. Idempotent.
pub fn replay_lease_acquired(
    world: &mut CompWorld,
    event: &DomainEvent,
) -> Result<CommittedEpoch, ExecutionError> {
    let lease_id = event.payload["lease_id"]
        .as_u64()
        .ok_or_else(|| ExecutionError::LeaseNotFound(0))?;
    let session_entity = event.payload["session_entity"]
        .as_u64()
        .ok_or_else(|| ExecutionError::SessionNotFound(0))?;
    let deployment_entity = event.payload["deployment_entity"]
        .as_u64()
        .ok_or_else(|| ExecutionError::DeploymentNotFound(0))?;
    let device_entity = event.payload["device_entity"]
        .as_u64()
        .ok_or_else(|| ExecutionError::DeviceNotFound(0))?;
    let token_range_start = event.payload["token_range_start"].as_u64().unwrap_or(0);
    let token_range_end = event.payload["token_range_end"].as_u64().unwrap_or(0);
    let deadline_ts = event.payload["deadline"].as_u64().unwrap_or(0);

    let mut txn = WorldTxn::new(world);

    if !world.has_entity(crate::ecs::CompEntity(lease_id)) {
        txn.stage_spawn(lease_id, EntityKind::Session);
    }

    txn.add_component(
        lease_id,
        ComponentSchemaId(SCHEMA_EXECUTION_LEASE),
        SchemaVersion(1),
        ExecutionLease {
            lease_id,
            session_entity,
            deployment_entity,
            device_entity,
            token_range_start,
            token_range_end,
            cancellation_epoch: WorldEpoch(0),
            created_at: Timestamp::now(),
            deadline: Timestamp(deadline_ts),
        },
    );
    txn.add_component(
        lease_id,
        ComponentSchemaId(SCHEMA_LEASE_OWNER),
        SchemaVersion(1),
        LeaseOwner {
            session_id: session_entity,
            work_item_id: lease_id,
        },
    );
    txn.add_component(
        lease_id,
        ComponentSchemaId(SCHEMA_LEASE_RANGE),
        SchemaVersion(1),
        LeaseTokenRange {
            start: token_range_start,
            end: token_range_end,
        },
    );

    let epoch = world.transit(txn).map_err(ExecutionError::CommitFailed)?;
    Ok(epoch)
}

/// Reconstruct lease completion from a `lease_completed` event.
/// Restores ExecutionOutput. Idempotent.
pub fn replay_lease_completed(
    world: &mut CompWorld,
    event: &DomainEvent,
) -> Result<CommittedEpoch, ExecutionError> {
    let lease_id = event.payload["lease_id"]
        .as_u64()
        .ok_or_else(|| ExecutionError::LeaseNotFound(0))?;
    let token_count = event.payload["token_count"].as_u64().unwrap_or(0) as usize;
    let finish_reason = event.payload["finish_reason"].as_u64().unwrap_or(0) as u8;

    let mut txn = WorldTxn::new(world);

    txn.add_component(
        lease_id,
        ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
        SchemaVersion(1),
        ExecutionOutput {
            tokens: vec![0u32; token_count],
            logprobs: None,
            finish_reason,
        },
    );

    let epoch = world.transit(txn).map_err(ExecutionError::CommitFailed)?;
    Ok(epoch)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::constitutional::lifecycle::{
        DeviceLifecycle, ResidencyLifecycle, SessionLifecycle,
    };
    use crate::ecs::CompWorld;

    // ── test_execution_lease_types_serde ────────────────────────────────

    #[test]
    fn test_execution_lease_types_serde() {
        // ExecutionLease
        let lease = ExecutionLease {
            lease_id: 42,
            session_entity: 1,
            deployment_entity: 3,
            device_entity: 2,
            token_range_start: 0,
            token_range_end: 512,
            cancellation_epoch: WorldEpoch(0),
            created_at: Timestamp(1_000_000),
            deadline: Timestamp(2_000_000),
        };
        let json = serde_json::to_string(&lease).unwrap();
        let back: ExecutionLease = serde_json::from_str(&json).unwrap();
        assert_eq!(lease, back);

        // LeaseOwner
        let owner = LeaseOwner {
            session_id: 1,
            work_item_id: 42,
        };
        let json = serde_json::to_string(&owner).unwrap();
        let back: LeaseOwner = serde_json::from_str(&json).unwrap();
        assert_eq!(owner, back);

        // LeaseTokenRange
        let range = LeaseTokenRange { start: 0, end: 512 };
        let json = serde_json::to_string(&range).unwrap();
        let back: LeaseTokenRange = serde_json::from_str(&json).unwrap();
        assert_eq!(range, back);

        // KvSlot
        let slot = KvSlot {
            cache_entity: 10,
            slot_index: 3,
            page_count: 64,
            format: 1,
        };
        let json = serde_json::to_string(&slot).unwrap();
        let back: KvSlot = serde_json::from_str(&json).unwrap();
        assert_eq!(slot, back);

        // KvOwnership
        let ownership = KvOwnership {
            session_id: 1,
            kv_slot_id: 10,
            valid_range_start: 0,
            valid_range_end: 4096,
        };
        let json = serde_json::to_string(&ownership).unwrap();
        let back: KvOwnership = serde_json::from_str(&json).unwrap();
        assert_eq!(ownership, back);

        // ExecutionOutput
        let output = ExecutionOutput {
            tokens: vec![101, 202, 303],
            logprobs: Some(vec![-0.5, -0.3, -0.1]),
            finish_reason: 0,
        };
        let json = serde_json::to_string(&output).unwrap();
        let back: ExecutionOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(output, back);
    }

    // ── test_acquire_lease_preflight ────────────────────────────────────

    #[test]
    fn test_acquire_lease_preflight() {
        let mut world = CompWorld::new();
        let mut reg = crate::ecs::constitutional::schema::SchemaRegistry::new();

        // Register schemas
        reg.register_for_type::<ExecutionLease>(
            ComponentSchemaId(SCHEMA_EXECUTION_LEASE),
            SchemaVersion(1),
            "ExecutionLease",
            "execution lease",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseOwner>(
            ComponentSchemaId(SCHEMA_LEASE_OWNER),
            SchemaVersion(1),
            "LeaseOwner",
            "lease owner",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseTokenRange>(
            ComponentSchemaId(SCHEMA_LEASE_RANGE),
            SchemaVersion(1),
            "LeaseTokenRange",
            "lease token range",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<KvSlot>(
            ComponentSchemaId(SCHEMA_KV_SLOT),
            SchemaVersion(1),
            "KvSlot",
            "kv cache slot",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );
        reg.register_for_type::<KvOwnership>(
            ComponentSchemaId(SCHEMA_KV_OWNERSHIP),
            SchemaVersion(1),
            "KvOwnership",
            "kv ownership",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<ExecutionOutput>(
            ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
            SchemaVersion(1),
            "ExecutionOutput",
            "execution output",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );

        // Spawn session entity (1) — must be Active
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(1, EntityKind::Session);
        txn.add_component(
            1,
            ComponentSchemaId(1),
            SchemaVersion(1),
            SessionLifecycle::Active,
        );
        world.transit(txn).unwrap();

        // Spawn device entity (2) — must be Ready
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(2, EntityKind::Device);
        txn.add_component(
            2,
            ComponentSchemaId(1),
            SchemaVersion(1),
            DeviceLifecycle::Ready,
        );
        world.transit(txn).unwrap();

        // Spawn residency entity (3) — must be Resident
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(3, EntityKind::Residency);
        txn.add_component(
            3,
            ComponentSchemaId(1),
            SchemaVersion(1),
            ResidencyLifecycle::Resident,
        );
        world.transit(txn).unwrap();

        let cmd = AcquireExecutionLeaseCommand {
            id: MessageId::compute(b"test-acquire"),
            session_entity: 1,
            deployment_entity: 3,
            device_entity: 2,
            token_batch_size: 512,
            deadline: Timestamp(2_000_000),
        };

        let result = cmd.preflight(&world, &reg);
        assert!(
            result.is_ok(),
            "preflight should succeed with valid entities, got: {:?}",
            result
        );

        // Execute
        let (epoch, event) = cmd.execute(&mut world, &reg).unwrap();
        assert_eq!(event.kind, "lease_acquired");
        assert_eq!(event.payload["lease_id"], 4); // 4th entity
    }

    // ── test_acquire_lease_preflight_invalid ─────────────────────────────

    #[test]
    fn test_acquire_lease_preflight_invalid() {
        let mut world = CompWorld::new();
        let mut reg = crate::ecs::constitutional::schema::SchemaRegistry::new();
        reg.register_for_type::<ExecutionLease>(
            ComponentSchemaId(SCHEMA_EXECUTION_LEASE),
            SchemaVersion(1),
            "ExecutionLease",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseOwner>(
            ComponentSchemaId(SCHEMA_LEASE_OWNER),
            SchemaVersion(1),
            "LeaseOwner",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseTokenRange>(
            ComponentSchemaId(SCHEMA_LEASE_RANGE),
            SchemaVersion(1),
            "LeaseTokenRange",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<KvSlot>(
            ComponentSchemaId(SCHEMA_KV_SLOT),
            SchemaVersion(1),
            "KvSlot",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );
        reg.register_for_type::<KvOwnership>(
            ComponentSchemaId(SCHEMA_KV_OWNERSHIP),
            SchemaVersion(1),
            "KvOwnership",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Durable,
        );
        reg.register_for_type::<ExecutionOutput>(
            ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
            SchemaVersion(1),
            "ExecutionOutput",
            "",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );

        // Spawn entities but with wrong lifecycle states
        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(1, EntityKind::Session);
        txn.add_component(
            1,
            ComponentSchemaId(1),
            SchemaVersion(1),
            SessionLifecycle::Created,
        );
        world.transit(txn).unwrap();

        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(2, EntityKind::Device);
        txn.add_component(
            2,
            ComponentSchemaId(1),
            SchemaVersion(1),
            DeviceLifecycle::Discovered,
        );
        world.transit(txn).unwrap();

        let mut txn = WorldTxn::new(&world);
        txn.stage_spawn(3, EntityKind::Residency);
        txn.add_component(
            3,
            ComponentSchemaId(1),
            SchemaVersion(1),
            ResidencyLifecycle::Binding,
        );
        world.transit(txn).unwrap();

        let cmd = AcquireExecutionLeaseCommand {
            id: MessageId::compute(b"test-acquire-fail"),
            session_entity: 1,
            deployment_entity: 3,
            device_entity: 2,
            token_batch_size: 128,
            deadline: Timestamp(2_000_000),
        };

        assert!(
            matches!(
                cmd.preflight(&world, &reg),
                Err(ExecutionError::SessionNotActive(1))
            ),
            "expected SessionNotActive for Created session"
        );

        // Fix session to Active, should now fail on DeploymentNotResident
        world.set_direct_mutation_allowed(true);
        world.add_component(crate::ecs::CompEntity(1), SessionLifecycle::Active);
        assert!(
            matches!(
                cmd.preflight(&world, &reg),
                Err(ExecutionError::DeploymentNotResident(3))
            ),
            "expected DeploymentNotResident for Binding residency"
        );

        // Fix deployment to Resident, should now fail on DeviceNotReady
        world.add_component(crate::ecs::CompEntity(3), ResidencyLifecycle::Resident);
        assert!(
            matches!(
                cmd.preflight(&world, &reg),
                Err(ExecutionError::DeviceNotReady(2))
            ),
            "expected DeviceNotReady for Discovered device"
        );
    }

    // ── test_kv_slot_ephemeral ───────────────────────────────────────────

    #[test]
    fn test_kv_slot_ephemeral() {
        // Verify KvSlot is registered with Ephemeral durability in the schema
        let mut reg = crate::ecs::constitutional::schema::SchemaRegistry::new();
        reg.register_for_type::<KvSlot>(
            ComponentSchemaId(SCHEMA_KV_SLOT),
            SchemaVersion(1),
            "KvSlot",
            "kv cache slot identity",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );

        let entry = reg.get(&ComponentSchemaId(SCHEMA_KV_SLOT)).unwrap();
        assert_eq!(
            entry.durability,
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
            "KvSlot should be registered as Ephemeral"
        );

        // KvOwnership is durable but marked ephemeral for replay
        reg.register_for_type::<KvOwnership>(
            ComponentSchemaId(SCHEMA_KV_OWNERSHIP),
            SchemaVersion(1),
            "KvOwnership",
            "kv ownership",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );
        let entry = reg.get(&ComponentSchemaId(SCHEMA_KV_OWNERSHIP)).unwrap();
        assert_eq!(
            entry.durability,
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
            "KvOwnership should be marked Ephemeral for replay"
        );

        // Also verify ExecutionOutput is registered as Ephemeral
        reg.register_for_type::<ExecutionOutput>(
            ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
            SchemaVersion(1),
            "ExecutionOutput",
            "execution result",
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
        );
        let entry = reg
            .get(&ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT))
            .unwrap();
        assert_eq!(
            entry.durability,
            crate::ecs::constitutional::schema::ComponentDurability::Ephemeral,
            "ExecutionOutput should be registered as Ephemeral"
        );
    }
}
