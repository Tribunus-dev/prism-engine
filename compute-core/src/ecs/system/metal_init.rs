use crate::ecs::component::backend::{BackendComponent, MetalDeviceState};

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
        let entity = world.spawn(EntityKind::Executable, Some("metal_backend".into()))?;
        let _ = world.add_component(entity,
        MetalDeviceState {
            device_handle: 0,
            command_queue_handle: 0,
            buffer_manager_handle: 0,
        },);;
        let _ = world.add_component(entity,
        BackendComponent {
            backend_id: "metal".into(),
            capabilities: vec!["f32".into(), "f16".into(), "bfloat16".into()],
            instance_id: 1,
        },);;

        Ok(())
    }
}
