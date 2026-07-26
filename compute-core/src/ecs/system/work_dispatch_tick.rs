use crate::ecs::component::scheduling::{
    BackpressureComponent, BackpressureLevel, ReadyQueueState, WorkRegistryComponent, WorkState,
};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Ticks the work dispatch loop — scans entities with pending items
/// in `ReadyQueueState` and advances `WorkRegistryComponent` states
/// from `Ready` → `Submitted`, subject to backpressure.
///
/// Runs every tick of the `Execution` phase.
pub struct WorkDispatchTickSystem;
impl CompilerSystem for WorkDispatchTickSystem {
    fn name(&self) -> &str {
        "WorkDispatchTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);
        let engine_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        // Check backpressure on engine entities.
        let has_backpressure = engine_entities.iter().any(|e| {
            world
                .get_component::<BackpressureComponent>(*e)
                .map(|bp| bp.level >= BackpressureLevel::Severe)
                .unwrap_or(false)
        });

        if has_backpressure {
            return Ok(());
        }

        // Stage every per-entity mutation (ReadyQueueState +
        // WorkRegistryComponent) on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        // Drain pending items from ReadyQueueState.
        for engine_entity in &engine_entities {
            let Some(rq) = world.get_component::<ReadyQueueState>(*engine_entity).cloned()
            else {
                continue;
            };
            let mut updated = rq;
            if updated.pending_items.is_empty() {
                continue;
            }
            updated.pending_items.clear();
            if let Err(e) = txn.stage_insert(*engine_entity, updated) {
                tracing::warn!(entity = ?engine_entity, error = %e, "work_dispatch_tick: stage_insert ReadyQueueState");
            }
        }

        // Dispatch any work items in Ready state.
        for entity in &entities {
            let Some(work) = world.get_component::<WorkRegistryComponent>(*entity).cloned()
            else {
                continue;
            };
            if work.state != WorkState::Ready && work.state != WorkState::Created {
                continue;
            }
            let mut updated = work;
            updated.state = WorkState::Submitted;
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "work_dispatch_tick: stage_insert WorkRegistryComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "work_dispatch_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("work_dispatch_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
