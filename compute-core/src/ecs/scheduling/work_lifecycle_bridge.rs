//! Bridge between the `HeterogeneousExecutor` and the constitutional work
//! command subsystem.  Replaces the legacy `WorkRegistry` with a thin API
//! that owns an `Arc<RwLock<World>>` and wraps each constitutional command's
//! preflight + execute + world.transit cycle.
//!
//! All methods follow the same internal pattern:
//! 1. Lock the world (read for preflight, drop the read lock)
//! 2. Construct the relevant constitutional command
//! 3. Lock the world (write for execute + transit)
//! 4. Call preflight against the world
//! 5. Create a `WorldTxn`, call execute, call `world.transit(txn)`
//! 6. Return the result or a string error

use crate::ecs::constitutional::types::{MessageId, Timestamp, WorldEpoch};
use crate::ecs::constitutional::work::{
    CancelWorkCommand, CompleteWorkCommand, CreateWorkCommand, FailWorkCommand, LeaseWorkCommand,
    ResourceClaimComponent, WorkKind, WorkState,
};
use crate::ecs::constitutional::world_txn::WorldTxn;
use crate::ecs::{Entity, World};
use prism_ecs_constitutional::WorldTransitExt;
use std::sync::{Arc, RwLock};
use uuid::Uuid;

/// Thin API over constitutional work commands, intended as a drop-in
/// replacement for the legacy `WorkRegistry`.
pub struct WorkLifecycleBridge {
    world: Arc<RwLock<World>>,
}

impl WorkLifecycleBridge {
    /// Wrap an existing `Arc<RwLock<World>>`.
    pub fn new(world: Arc<RwLock<World>>) -> Self {
        Self { world }
    }

    // ── core helpers ─────────────────────────────────────────────────────────

    /// Generate a unique `MessageId` for each command invocation.
    fn msg_id() -> MessageId {
        MessageId::compute(Uuid::new_v4().as_bytes())
    }

    /// Lock the world for reading and extract the current lease generation
    /// from a `WorkState::Leased(gen)`.  Returns `None` if the entity is
    /// missing or in a non-Leased state.
    fn read_lease_generation(world: &World, work_entity: Entity) -> Option<u32> {
        match world.get_component::<WorkState>(work_entity) {
            Some(WorkState::Leased(gen)) => Some(*gen),
            _ => None,
        }
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// Create a work item and transition it to `Ready` in a single atomic
    /// transaction.
    ///
    /// Returns the newly allocated work [`Entity`].
    pub fn create_work(
        &self,
        kind: WorkKind,
        target_entity: Entity,
        resource_claim: ResourceClaimComponent,
    ) -> Result<Entity, String> {
        let cmd = CreateWorkCommand {
            id: Self::msg_id(),
            kind,
            target_entity,
            prerequisites: Vec::new(), // no prerequisites — go straight to Ready
            resource_claim,
        };

        // Read-lock for preflight
        {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            cmd.preflight(&guard).map_err(|e| e.to_string())?;
        }

        // Write-lock for execute + transit
        let mut guard = self.world.write().map_err(|e| e.to_string())?;

        let mut txn = WorldTxn::new(&*guard);
        let _event = cmd.execute(&*guard, &mut txn).map_err(|e| e.to_string())?;

        // The CreateWorkCommand puts the entity in Pending; push it to Ready
        // in the same transaction so the caller never sees a non-ready state.
        // (We know the entity id from the DomainEvent, but since
        // WorldTxn::next_entity_id gives us the predicted id we use
        // that.)
        let work_entity = WorldTxn::next_entity_id(&*guard);
        txn.put_durable::<WorkState>(work_entity, WorkState::Ready);

        guard.transit(txn).map_err(|e| e.to_string())?;

        Ok(work_entity)
    }

    /// Transition work from `Ready` to `Leased`.
    pub fn lease_work(&self, work_entity: Entity, lease_generation: u32) -> Result<(), String> {
        let cmd = LeaseWorkCommand {
            id: Self::msg_id(),
            work_entity,
            lease_generation,
            attempt: 0,
            cancellation_epoch: WorldEpoch(0),
            expiry: Timestamp(0),
        };

        // Read-lock for preflight
        {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            cmd.preflight(&guard).map_err(|e| e.to_string())?;
        }

        // Write-lock for execute + transit
        let mut guard = self.world.write().map_err(|e| e.to_string())?;
        let mut txn = WorldTxn::new(&*guard);
        let _event = cmd.execute(&*guard, &mut txn).map_err(|e| e.to_string())?;
        guard.transit(txn).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Complete a leased work item, attaching its output.
    pub fn complete_work(&self, work_entity: Entity, output: Vec<u8>) -> Result<(), String> {
        // Read current lease generation under the read lock
        let lease_generation = {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            Self::read_lease_generation(&guard, work_entity)
                .ok_or_else(|| "work not currently leased".to_string())?
        };

        let cmd = CompleteWorkCommand {
            id: Self::msg_id(),
            work_entity,
            output,
            lease_generation,
        };

        // Read-lock for preflight
        {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            cmd.preflight(&guard).map_err(|e| e.to_string())?;
        }

        // Write-lock for execute + transit
        let mut guard = self.world.write().map_err(|e| e.to_string())?;
        let mut txn = WorldTxn::new(&*guard);
        let _event = cmd.execute(&*guard, &mut txn).map_err(|e| e.to_string())?;
        guard.transit(txn).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Fail a leased work item with the given error message.
    pub fn fail_work(&self, work_entity: Entity, error: String) -> Result<(), String> {
        // Read current lease generation under the read lock
        let lease_generation = {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            Self::read_lease_generation(&guard, work_entity)
                .ok_or_else(|| "work not currently leased".to_string())?
        };

        let cmd = FailWorkCommand {
            id: Self::msg_id(),
            work_entity,
            error,
            lease_generation,
        };

        // Read-lock for preflight
        {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            cmd.preflight(&guard).map_err(|e| e.to_string())?;
        }

        // Write-lock for execute + transit
        let mut guard = self.world.write().map_err(|e| e.to_string())?;
        let mut txn = WorldTxn::new(&*guard);
        let _event = cmd.execute(&*guard, &mut txn).map_err(|e| e.to_string())?;
        guard.transit(txn).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Cancel a work item (Pending, Ready, or Leased → Cancelled).
    pub fn cancel_work(&self, work_entity: Entity) -> Result<(), String> {
        let cmd = CancelWorkCommand {
            id: Self::msg_id(),
            work_entity,
        };

        // Read-lock for preflight
        {
            let guard = self.world.read().map_err(|e| e.to_string())?;
            cmd.preflight(&guard).map_err(|e| e.to_string())?;
        }

        // Write-lock for execute + transit
        let mut guard = self.world.write().map_err(|e| e.to_string())?;
        let mut txn = WorldTxn::new(&*guard);
        let _event = cmd.execute(&*guard, &mut txn).map_err(|e| e.to_string())?;
        guard.transit(txn).map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Read the current state of a work item without mutating.
    ///
    /// Returns `None` if the entity does not exist or has no `WorkState`
    /// component.
    pub fn get_work_state(&self, work_entity: Entity) -> Option<WorkState> {
        let guard = self.world.read().ok()?;
        guard.get_component::<WorkState>(work_entity).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::EntityKind;

    /// Helper: spawn an executable entity in the world and return its Entity.
    fn spawn_executable(world: &mut World) -> Entity {
        let mut txn = WorldTxn::new(world);
        let e = WorldTxn::next_entity_id(world);
        txn.stage_spawn(e, EntityKind::Executable);
        world.transit(txn).unwrap();
        e
    }

    #[test]
    fn test_create_work() {
        let mut world = World::new();
        let target = spawn_executable(&mut world);
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        let work = bridge
            .create_work(
                WorkKind::RunInference,
                target,
                ResourceClaimComponent {
                    memory_bytes: 1024,
                    compute_units: 1,
                    priority: 0,
                },
            )
            .expect("create_work should succeed");

        // Verify the work entity exists and is in Ready state
        assert_eq!(
            bridge.get_work_state(work),
            Some(WorkState::Ready),
            "work should be created in Ready state"
        );
    }

    #[test]
    fn test_create_work_missing_target() {
        let world = World::new();
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        // Target entity 9999 does not exist
        let result = bridge.create_work(
            WorkKind::RunInference,
            Entity(9999, 0),
            ResourceClaimComponent {
                memory_bytes: 1024,
                compute_units: 1,
                priority: 0,
            },
        );
        assert!(result.is_err(), "create_work with missing target must fail");
    }

    #[test]
    fn test_full_lifecycle() {
        let mut world = World::new();

        let target = spawn_executable(&mut world);
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        // Create → Ready
        let work = bridge
            .create_work(
                WorkKind::RunInference,
                target,
                ResourceClaimComponent {
                    memory_bytes: 1024,
                    compute_units: 1,
                    priority: 0,
                },
            )
            .expect("create_work");

        // Lease
        bridge
            .lease_work(work, 1)
            .expect("lease_work should succeed");
        assert_eq!(
            bridge.get_work_state(work),
            Some(WorkState::Leased(1)),
            "work should be Leased(1)"
        );

        // Complete
        bridge
            .complete_work(work, vec![1, 2, 3])
            .expect("complete_work should succeed");
        assert_eq!(
            bridge.get_work_state(work),
            Some(WorkState::Completed),
            "work should be Completed"
        );

        // Re-completing must fail
        assert!(
            bridge.complete_work(work, vec![]).is_err(),
            "completing a Completed work must fail"
        );

        // Cancelling a Completed work must also fail
        assert!(
            bridge.cancel_work(work).is_err(),
            "cancelling a Completed work must fail"
        );
    }

    #[test]
    fn test_lease_and_fail() {
        let mut world = World::new();

        let target = spawn_executable(&mut world);
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        let work = bridge
            .create_work(
                WorkKind::CompileGraph,
                target,
                ResourceClaimComponent {
                    memory_bytes: 512,
                    compute_units: 2,
                    priority: 1,
                },
            )
            .expect("create_work");

        bridge
            .lease_work(work, 1)
            .expect("lease_work should succeed");
        assert_eq!(bridge.get_work_state(work), Some(WorkState::Leased(1)));

        // Fail
        bridge
            .fail_work(work, "compute error".to_string())
            .expect("fail_work should succeed");
        assert_eq!(
            bridge.get_work_state(work),
            Some(WorkState::Failed),
            "work should be Failed"
        );

        // Failing a Failed work must fail
        assert!(
            bridge.fail_work(work, "again".to_string()).is_err(),
            "failing a Failed work must fail"
        );
    }

    #[test]
    fn test_cancel_ready_work() {
        let mut world = World::new();

        let target = spawn_executable(&mut world);
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        let work = bridge
            .create_work(
                WorkKind::Validate,
                target,
                ResourceClaimComponent {
                    memory_bytes: 256,
                    compute_units: 1,
                    priority: 0,
                },
            )
            .expect("create_work");

        // Cancel before leasing
        bridge
            .cancel_work(work)
            .expect("cancel_work should succeed");
        assert_eq!(bridge.get_work_state(work), Some(WorkState::Cancelled));
    }

    #[test]
    fn test_get_work_state_missing() {
        let world = World::new();
        let bridge = WorkLifecycleBridge::new(Arc::new(RwLock::new(world)));

        // Non-existent entity → None
        assert_eq!(bridge.get_work_state(Entity(999, 0)), None);
    }
}
