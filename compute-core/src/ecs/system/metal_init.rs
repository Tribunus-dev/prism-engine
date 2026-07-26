use crate::ecs::component::backend::{BackendComponent, MetalDeviceState};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Initializes the Metal device — creates a Backend entity with
/// `MetalDeviceState` and `BackendComponent` on the ECS world.
///
/// Runs once during `SchedulePhase::ModelLoading`.
pub struct MetalInitSystem;
impl CompilerSystem for MetalInitSystem {
    fn name(&self) -> &str {
        "MetalInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Check if a Metal backend entity already exists.
        let existing: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);
        for entity in &existing {
            if world.get_component::<MetalDeviceState>(*entity).is_some() {
                return Ok(());
            }
        }

        // Spawn a new backend entity with Metal device state.
        //
        // Stage the spawn + both inserts on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        let token = txn.stage_spawn(EntityKind::Executable, Some("metal_backend".into()));
        if let Err(e) = txn.stage_insert_on(
            token,
            MetalDeviceState {
                device_handle: 0,
                command_queue_handle: 0,
                buffer_manager_handle: 0,
            },
        ) {
            tracing::warn!(error = %e, "metal_init: stage_insert_on MetalDeviceState");
        }
        if let Err(e) = txn.stage_insert_on(
            token,
            BackendComponent {
                backend_id: "metal".into(),
                capabilities: vec!["f32".into(), "f16".into(), "bfloat16".into()],
                instance_id: 1,
            },
        ) {
            tracing::warn!(error = %e, "metal_init: stage_insert_on BackendComponent");
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "metal_init: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("metal_init: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
