use crate::ecs::component::scheduling::SessionState;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

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

        // Stage every per-entity remove on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.remove_component` calls outside the WorldTxn seam
        // are forbidden.
        //
        // The original code had duplicate removes (a pre-existing
        // bug — the second remove is a no-op). The staged-txn port
        // preserves this verbatim: ConstitutionalWorldTxn::removes is
        // keyed by (Entity, TypeId) and the second insert overwrites
        // the first with the same closure payload.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            if world.get_component::<SessionState>(*entity).is_some() {
                if let Err(e) = txn.stage_remove::<SessionState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "session_cleanup: stage_remove SessionState (1st)");
                }
                if let Err(e) = txn.stage_remove::<SessionState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "session_cleanup: stage_remove SessionState (2nd)");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "session_cleanup: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("session_cleanup: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
