use crate::ecs::component::scheduling::{
    BackpressureComponent, BackpressureLevel, WorkRegistryComponent, WorkState,
};

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Dispatches work from the registry to lane queues.
///
/// Scans entities with `WorkRegistryComponent` whose state is `Ready`
/// or `Selected`, checks for backpressure, and advances them into the
/// appropriate lane queue.
pub struct WorkDispatchSystem;
impl CompilerSystem for WorkDispatchSystem {
    fn name(&self) -> &str {
        "WorkDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        let lane_entities: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);

        for entity in &entities {
            // Check backpressure first (immutable borrow)
            let blocked = lane_entities.iter().any(|le| {
                world
                    .get_component::<BackpressureComponent>(*le)
                    .map(|bp| bp.level >= BackpressureLevel::Severe)
                    .unwrap_or(false)
            });

            if blocked {
                continue;
            }

            let Some(work) = world.get_component_mut::<WorkRegistryComponent>(*entity) else {
                continue;
            };
            if work.state != WorkState::Ready && work.state != WorkState::Selected {
                continue;
            }

            work.state = WorkState::Submitted;
        }

        Ok(())
    }
}
