use crate::ecs::component::scheduling::SessionState;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Initializes an inference session — spawns a session entity with
/// `SessionState` components.
///
/// Runs once during `SchedulePhase::ModelLoading`.
pub struct SessionInitSystem;
impl CompilerSystem for SessionInitSystem {
    fn name(&self) -> &str {
        "SessionInitSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Check if a session already exists.
        let existing: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);
        for entity in &existing {
            if world.get_component::<SessionState>(*entity).is_some() {
                return Ok(());
            }
        }

        // Spawn a session entity.
        //
        // Stage the spawn + insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        let token = txn.stage_spawn(EntityKind::Session, Some("session".into()));
        if let Err(e) = txn.stage_insert_on(
            token,
            SessionState {
                decode_step: 0,
                active_model: "".into(),
                generation_params_json: "{}".into(),
            },
        ) {
            tracing::warn!(error = %e, "session_init: stage_insert_on SessionState");
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "session_init: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("session_init: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
