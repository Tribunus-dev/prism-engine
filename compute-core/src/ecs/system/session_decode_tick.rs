use crate::ecs::component::scheduling::SessionState;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Advances one decode step — increments `SessionState.decode_step`
/// on each tick of the `Execution` phase.
///
/// Runs every tick of the `Execution` phase.
pub struct SessionDecodeTickSystem;
impl CompilerSystem for SessionDecodeTickSystem {
    fn name(&self) -> &str {
        "SessionDecodeTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);

        // Stage every per-entity `SessionState` mutation on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let Some(session) = world.get_component::<SessionState>(*entity).cloned() else {
                continue;
            };
            let mut updated = session;
            updated.decode_step += 1;
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "session_decode_tick: stage_insert SessionState");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "session_decode_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("session_decode_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
