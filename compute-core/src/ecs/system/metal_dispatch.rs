use crate::ecs::component::scheduling::WorkRegistryComponent;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Dispatches Metal compute kernels — scans Tensor entities with pending
/// work and advances them through the dispatch pipeline.
///
/// Runs every tick of the `Execution` phase.
pub struct MetalDispatchSystem;
impl CompilerSystem for MetalDispatchSystem {
    fn name(&self) -> &str {
        "MetalDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Stage every per-entity `WorkRegistryComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            // Check for a work registry entry to dispatch.
            let state = world
                .get_component::<WorkRegistryComponent>(*entity)
                .map(|w| w.state)
                .unwrap_or(crate::ecs::component::scheduling::WorkState::Created);
            if state != crate::ecs::component::scheduling::WorkState::Submitted {
                continue;
            }

            let Some(work) = world.get_component::<WorkRegistryComponent>(*entity).cloned()
            else {
                continue;
            };
            // Advance to Running state — the actual Metal dispatch
            // is delegated to the kernel-specific systems.
            let mut updated = work;
            updated.state = crate::ecs::component::scheduling::WorkState::Running;
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "metal_dispatch: stage_insert WorkRegistryComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "metal_dispatch: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("metal_dispatch: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
