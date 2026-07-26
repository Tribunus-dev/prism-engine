//! Bridge between the `HeterogeneousExecutor` and the constitutional execution
//! lease command subsystem.  Wraps `AcquireExecutionLeaseCommand` and
//! `CompleteExecutionLeaseCommand` with a thin API owning an
//! `Arc<RwLock<World>>`.
//!
//! # Authority record, not hot path
//!
//! The `SlotLeaseManager` on the executor handles hot-path slot synchronization.
//! This bridge is the constitutional authority record — it records lease
//! acquire/complete in the ECS world via domain events.  Errors are discarded
//! at the call site (`let _ = …`) so a constitutional failure never stalls
//! the hot execution path.

use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::execution::{
    AcquireExecutionLeaseCommand, CompleteExecutionLeaseCommand, ExecutionLease, ExecutionOutput,
    LeaseOwner, LeaseTokenRange, SCHEMA_EXECUTION_LEASE, SCHEMA_EXECUTION_OUTPUT,
    SCHEMA_LEASE_OWNER, SCHEMA_LEASE_RANGE,
};
use crate::ecs::constitutional::schema::{ComponentDurability, SchemaRegistry};
use crate::ecs::constitutional::types::{ComponentSchemaId, MessageId, SchemaVersion, Timestamp};
use crate::ecs::constitutional::world_txn::CommittedEpoch;
use crate::ecs::{Entity, World};
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Thin API over constitutional execution lease commands.
pub struct ExecutionLeaseBridge {
    world: Arc<RwLock<World>>,
}

impl ExecutionLeaseBridge {
    /// Wrap an existing `Arc<RwLock<World>>`.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        Self { world }
    }

    // ── helpers ────────────────────────────────────────────────────────────

    /// Generate a unique `MessageId` for each command invocation.
    fn msg_id() -> MessageId {
        MessageId::compute(Uuid::new_v4().as_bytes())
    }

    /// Create a `SchemaRegistry` pre-populated with all execution component
    /// schemas so that the constitutional command's preflight succeeds.
    fn make_registry() -> SchemaRegistry {
        let mut reg = SchemaRegistry::new();
        reg.register_for_type::<ExecutionLease>(
            ComponentSchemaId(SCHEMA_EXECUTION_LEASE),
            SchemaVersion(1),
            "ExecutionLease",
            "bounded execution lease",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseOwner>(
            ComponentSchemaId(SCHEMA_LEASE_OWNER),
            SchemaVersion(1),
            "LeaseOwner",
            "lease ownership record",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<LeaseTokenRange>(
            ComponentSchemaId(SCHEMA_LEASE_RANGE),
            SchemaVersion(1),
            "LeaseTokenRange",
            "token range for lease",
            ComponentDurability::Durable,
        );
        reg.register_for_type::<ExecutionOutput>(
            ComponentSchemaId(SCHEMA_EXECUTION_OUTPUT),
            SchemaVersion(1),
            "ExecutionOutput",
            "execution result output",
            ComponentDurability::Ephemeral,
        );
        reg
    }

    // ── public API ─────────────────────────────────────────────────────────

    /// Acquire a constitutional execution lease.
    ///
    /// Constructs an [`AcquireExecutionLeaseCommand`], registers the execution
    /// schemas against an internal registry, and executes the command against
    /// the world.  Returns the committed epoch and domain event on success.
    pub fn acquire_lease(
        &self,
        session_entity: u64,
        deployment_entity: u64,
        device_entity: u64,
        token_batch_size: u64,
    ) -> Result<(CommittedEpoch, DomainEvent), String> {
        let mut world = self
            .world
            .write()
            .map_err(|e| format!("world lock poisoned: {e}"))?;

        let cmd = AcquireExecutionLeaseCommand {
            id: Self::msg_id(),
            session_entity,
            deployment_entity,
            device_entity,
            token_batch_size,
            deadline: Timestamp::now(),
        };

        let registry = Self::make_registry();

        cmd.execute(&mut *world, &registry)
            .map_err(|e| e.to_string())
    }

    /// Complete a constitutional execution lease.
    ///
    /// Constructs a [`CompleteExecutionLeaseCommand`] and executes it against
    /// the world.  Returns the committed epoch and domain event on success.
    pub fn complete_lease(
        &self,
        lease_id: Entity,
        tokens: Vec<u32>,
        finish_reason: u8,
    ) -> Result<(CommittedEpoch, DomainEvent), String> {
        let mut world = self
            .world
            .write()
            .map_err(|e| format!("world lock poisoned: {e}"))?;

        let cmd = CompleteExecutionLeaseCommand {
            id: Self::msg_id(),
            lease_id,
            tokens,
            finish_reason,
        };

        cmd.execute(&mut *world).map_err(|e| e.to_string())
    }
}
