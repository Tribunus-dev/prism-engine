use crate::ecs::component::scheduling::{
    BackpressureComponent, BackpressureLevel, ReadyQueueState, WorkRegistryComponent, WorkState,
};
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Ticks the work dispatch loop — scans entities with pending items
/// in `ReadyQueueState` and advances `WorkRegistryComponent` states
/// from `Ready` → `Submitted`, subject to backpressure.
///
/// Runs every tick of the `Execution` phase.
pub struct WorkDispatchTickSystem;
impl CompilerSystem for WorkDispatchTickSystem {
    fn name(&self) -> &str {
        "WorkDispatchTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);
        let engine_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Executable);

        // Check backpressure on engine entities.
        let has_backpressure = engine_entities.iter().any(|e| {
            world
                .get_component::<BackpressureComponent>(*e)
                .map(|bp| bp.level >= BackpressureLevel::Severe)
                .unwrap_or(false)
        });

        if has_backpressure {
            return Ok(());
        }

        // Drain pending items from ReadyQueueState.
        for engine_entity in &engine_entities {
            let Some(rq) = world.get_component_mut::<ReadyQueueState>(*engine_entity) else {
                continue;
            };
            if rq.pending_items.is_empty() {
                continue;
            }
            rq.pending_items.clear();
        }

        // Dispatch any work items in Ready state.
        for entity in &entities {
            let Some(work) = world.get_component_mut::<WorkRegistryComponent>(*entity) else {
                continue;
            };
            if work.state != WorkState::Ready && work.state != WorkState::Created {
                continue;
            }
            work.state = WorkState::Submitted;
        }

        Ok(())
    }
}
