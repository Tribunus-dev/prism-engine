use crate::ecs::component::backend::MetalDeviceState;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Cleans up Metal resources — removes `MetalDeviceState` components
/// and their owning entities.
///
/// Runs once during `SchedulePhase::Packaging`.
pub struct MetalCleanupSystem;
impl CompilerSystem for MetalCleanupSystem {
    fn name(&self) -> &str {
        "MetalCleanupSystem"
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
        // preserves this verbatim: ConstitutionalWorldTxn::removes is
        // keyed by (Entity, TypeId) and the second insert overwrites
        // the first with the same closure payload.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            if world.get_component::<MetalDeviceState>(*entity).is_some() {
                if let Err(e) = txn.stage_remove::<MetalDeviceState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "metal_cleanup: stage_remove MetalDeviceState (1st)");
                }
                if let Err(e) = txn.stage_remove::<MetalDeviceState>(*entity) {
                    tracing::warn!(entity = ?entity, error = %e, "metal_cleanup: stage_remove MetalDeviceState (2nd)");
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "metal_cleanup: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("metal_cleanup: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
