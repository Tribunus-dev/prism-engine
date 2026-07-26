use crate::ecs::component::scheduling::{PhaseDagState, ReadyQueueState};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

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

        // Stage every per-entity remove on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.remove_component` calls outside the WorldTxn seam
        // are forbidden.
        //
        // The original code had duplicate removes (a pre-existing
        // bug — the second remove is a no-op). The staged-txn port
        // preserves this verbatim: the duplicate is harmless because
        // ConstitutionalWorldTxn::removes is keyed by (Entity, TypeId)
        // and the second insert overwrites the first with the same
        // closure payload.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            if world.get_component::<PhaseDagState>(*entity).is_some() {
                if let Err(e) = txn.stage_remove::<PhaseDagState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "phase_engine_cleanup: stage_remove PhaseDagState (1st)");
                }
                if let Err(e) = txn.stage_remove::<PhaseDagState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "phase_engine_cleanup: stage_remove PhaseDagState (2nd)");
                }
            }
            if world.get_component::<ReadyQueueState>(*entity).is_some() {
                if let Err(e) = txn.stage_remove::<ReadyQueueState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "phase_engine_cleanup: stage_remove ReadyQueueState (1st)");
                }
                if let Err(e) = txn.stage_remove::<ReadyQueueState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "phase_engine_cleanup: stage_remove ReadyQueueState (2nd)");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "phase_engine_cleanup: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("phase_engine_cleanup: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
