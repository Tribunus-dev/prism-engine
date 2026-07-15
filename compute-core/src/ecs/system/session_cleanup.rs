use crate::ecs::component::scheduling::SessionState;

use crate::ecs::Entity;
use crate::ecs::{World, CompilerSystem, EntityKind, SchedulePhase};

/// Cleans up session state — removes `SessionState` components
/// from session entities.
///
/// Runs once during `SchedulePhase::Packaging`.
pub struct SessionCleanupSystem;
impl CompilerSystem for SessionCleanupSystem {
    fn name(&self) -> &str {
        "SessionCleanupSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);

        for entity in &entities {
            if world.get_component::<SessionState>(*entity).is_some() {
                world.remove_component::<SessionState>(*entity);
            }
        }

        Ok(())
    }
}
