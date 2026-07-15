use crate::ecs::component::backend::MetalDeviceState;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Cleans up Metal resources — removes `MetalDeviceState` components
/// and their owning entities.
///
/// Runs once during `SchedulePhase::Packaging`.
pub struct MetalCleanupSystem;
impl CompilerSystem for MetalCleanupSystem {
    fn name(&self) -> &str {
        "MetalCleanupSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        for entity in &entities {
            if world.get_component::<MetalDeviceState>(*entity).is_some() {
                world.remove_component::<MetalDeviceState>(*entity);
            }
        }

        Ok(())
    }
}
