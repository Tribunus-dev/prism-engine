use crate::ecs::component::scheduling::{PhaseDagState, ReadyQueueState};

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Cleans up phase DAG state — removes `PhaseDagState` and
/// `ReadyQueueState` components from the engine entity.
///
/// Runs once during `SchedulePhase::Packaging`.
pub struct PhaseEngineCleanupSystem;
impl CompilerSystem for PhaseEngineCleanupSystem {
    fn name(&self) -> &str {
        "PhaseEngineCleanupSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        for entity in &entities {
            if world.get_component::<PhaseDagState>(*entity).is_some() {
                world.remove_component::<PhaseDagState>(*entity);
            }
            if world.get_component::<ReadyQueueState>(*entity).is_some() {
                world.remove_component::<ReadyQueueState>(*entity);
            }
        }

        Ok(())
    }
}
