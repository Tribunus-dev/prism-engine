use crate::ecs::component::scheduling::PhaseDagState;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

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

        // Stage every per-entity `PhaseDagState` mutation on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let Some(dag) = world.get_component::<PhaseDagState>(*entity).cloned() else {
                continue;
            };

            let mut updated = dag;
            // Find the next phase by looking at outgoing edges from the current phase.
            let next = updated
                .edges
                .iter()
                .find(|(from, _)| from == &updated.current_phase)
                .map(|(_, to)| to.clone());

            if let Some(next_phase) = next {
                updated.current_phase = next_phase;
            }
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "phase_engine_tick: stage_insert PhaseDagState");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "phase_engine_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("phase_engine_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
