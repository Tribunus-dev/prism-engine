use crate::ecs::component::scheduling::{PhaseDagState, ReadyQueueState};
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

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
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Executable);

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
