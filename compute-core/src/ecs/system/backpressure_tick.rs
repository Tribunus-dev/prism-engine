use crate::ecs::component::scheduling::{BackpressureComponent, BackpressureLevel};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Ticks the backpressure state machine — decay active backpressure
/// levels over time as resources drain.
///
/// Reads every entity with a `BackpressureComponent`, decays the
/// level based on queue depth, and updates it.
pub struct BackpressureTickSystem;
impl CompilerSystem for BackpressureTickSystem {
    fn name(&self) -> &str {
        "BackpressureTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for entity in &entities {
            let Some(bp) = world.get_component_mut::<BackpressureComponent>(*entity) else {
                continue;
            };

            if bp.queue_depth == 0 && bp.level > BackpressureLevel::None {
                bp.level = match bp.level {
                    BackpressureLevel::Critical => BackpressureLevel::Severe,
                    BackpressureLevel::Severe => BackpressureLevel::Moderate,
                    BackpressureLevel::Moderate => BackpressureLevel::Mild,
                    _ => BackpressureLevel::None,
                };
            }
        }

        Ok(())
    }
}
