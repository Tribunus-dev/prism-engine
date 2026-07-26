use crate::ecs::component::scheduling::{
    BackpressureComponent, BackpressureLevel, WorkRegistryComponent, WorkState,
};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Dispatches work from the registry to lane queues.
///
/// Scans entities with `WorkRegistryComponent` whose state is `Ready`
/// or `Selected`, checks for backpressure, and advances them into the
/// appropriate lane queue.
pub struct WorkDispatchSystem;
impl CompilerSystem for WorkDispatchSystem {
    fn name(&self) -> &str {
        "WorkDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        let lane_entities: Vec<Entity> = world.entities_of_kind(EntityKind::CommandBuffer);

        // Stage every per-entity `WorkRegistryComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden.
        //
        // Strategy: snapshot the pre-mutation value via
        // `get_component` (immutable read), compute the post-mutation
        // value locally, and stage the new value as an insert. This
        // is the extract-mutate-insert pattern documented in the
        // Phase 2.5 changelog; it preserves the constitutional
        // discipline at the cost of one clone per WorkRegistry
        // mutation.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            // Check backpressure first (immutable borrow)
            let blocked = lane_entities.iter().any(|le| {
                world
                    .get_component::<BackpressureComponent>(*le)
                    .map(|bp| bp.level >= BackpressureLevel::Severe)
                    .unwrap_or(false)
            });

            if blocked {
                continue;
            }

            let Some(work) = world.get_component::<WorkRegistryComponent>(*entity).cloned()
            else {
                continue;
            };
            if work.state != WorkState::Ready && work.state != WorkState::Selected {
                continue;
            }

            let mut updated = work;
            updated.state = WorkState::Submitted;
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "work_dispatch: stage_insert WorkRegistryComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "work_dispatch: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("work_dispatch: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
