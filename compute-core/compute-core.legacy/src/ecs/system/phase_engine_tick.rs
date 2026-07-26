use crate::ecs::component::scheduling::PhaseDagState;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Ticks the phase engine state machine — advances `PhaseDagState.current_phase`
/// along the configured phase DAG edges.
///
/// Runs every tick of the `Execution` phase.
pub struct PhaseEngineTickSystem;
impl CompilerSystem for PhaseEngineTickSystem {
    fn name(&self) -> &str {
        "PhaseEngineTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        for entity in &entities {
            let Some(dag) = world.get_component_mut::<PhaseDagState>(*entity) else {
                continue;
            };

            // Find the next phase by looking at outgoing edges from the current phase.
            let next = dag
                .edges
                .iter()
                .find(|(from, _)| from == &dag.current_phase)
                .map(|(_, to)| to.clone());

            if let Some(next_phase) = next {
                dag.current_phase = next_phase;
            }
        }

        Ok(())
    }
}
