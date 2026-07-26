use crate::ecs::component::scheduling::{BackpressureComponent, BackpressureLevel};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Ticks the backpressure state machine — decay active backpressure
/// levels over time as resources drain.
///
/// Reads every entity with a `BackpressureComponent`, decays the
/// level based on queue depth, and updates it.
pub struct BackpressureTickSystem;
impl CompilerSystem for BackpressureTickSystem {
    fn name(&self) -> &str {
        "BackpressureTickSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Stage every per-entity `BackpressureComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &entities {
            let Some(bp) = world.get_component::<BackpressureComponent>(*entity).cloned()
            else {
                continue;
            };

            let mut updated = bp;
            if updated.queue_depth == 0 && updated.level > BackpressureLevel::None {
                updated.level = match updated.level {
                    BackpressureLevel::Critical => BackpressureLevel::Severe,
                    BackpressureLevel::Severe => BackpressureLevel::Moderate,
                    BackpressureLevel::Moderate => BackpressureLevel::Mild,
                    _ => BackpressureLevel::None,
                };
            }
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "backpressure_tick: stage_insert BackpressureComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "backpressure_tick: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("backpressure_tick: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
