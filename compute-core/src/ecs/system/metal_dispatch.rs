use crate::ecs::component::scheduling::WorkRegistryComponent;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Dispatches Metal compute kernels — scans Tensor entities with pending
/// work and advances them through the dispatch pipeline.
///
/// Runs every tick of the `Execution` phase.
pub struct MetalDispatchSystem;
impl CompilerSystem for MetalDispatchSystem {
    fn name(&self) -> &str {
        "MetalDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            // Check for a work registry entry to dispatch.
            let state = world
                .get_component::<WorkRegistryComponent>(*entity)
                .map(|w| w.state)
                .unwrap_or(crate::ecs::component::scheduling::WorkState::Created);
            if state != crate::ecs::component::scheduling::WorkState::Submitted {
                continue;
            }

            let Some(work) = world.get_component_mut::<WorkRegistryComponent>(*entity) else {
                continue;
            };
            // Advance to Running state — the actual Metal dispatch
            // is delegated to the kernel-specific systems.
            work.state = crate::ecs::component::scheduling::WorkState::Running;
        }

        Ok(())
    }
}
