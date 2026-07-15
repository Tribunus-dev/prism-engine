use crate::ecs::component::fusion::{TileSize, WorkgroupCount};
use crate::ecs::component::tensor::Shape;

use crate::ecs::Entity;
use crate::ecs::{World, CompilerSystem, EntityKind, SchedulePhase};

pub struct ScalarDispatchSystem;
impl CompilerSystem for ScalarDispatchSystem {
    fn name(&self) -> &str {
        "ScalarDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        const SCALAR_THRESHOLD: u32 = 128;

        let dispatches = world.entities_of_kind(EntityKind::Dispatch);

        for entity in dispatches {
            if let Some(total) = dispatch_total_elements(world, entity) {
                if total < SCALAR_THRESHOLD {
                    world.add_component(entity, WorkgroupCount(1, 1, 1));
                }
            }
        }

        Ok(())
    }
}

/// Compute the total number of scalar invocations for a dispatch entity.
///
/// Prefers the product of `Shape` dimensions when present. Falls back to
/// `WorkgroupCount` x `TileSize` (defaulting tile to 1x1x1 when absent).
/// Returns `None` when neither signal is available or the product overflows.
fn dispatch_total_elements(world: &World, entity: Entity) -> Option<u32> {
    // Primary signal: the dispatch's output tensor shape.
    if let Some(shape) = world.get_component::<Shape>(entity) {
        return shape
            .0
            .iter()
            .try_fold(1u32, |acc, &dim| acc.checked_mul(dim));
    }

    // Fallback: grid count x thread count already decided by a prior pass.
    if let Some(wg) = world.get_component::<WorkgroupCount>(entity) {
        let tile_prod = world
            .get_component::<TileSize>(entity)
            .map(|t| t.0.checked_mul(t.1)?.checked_mul(t.2))
            .flatten()
            .unwrap_or(1);
        let wg_prod = wg.0.checked_mul(wg.1)?.checked_mul(wg.2)?;
        return wg_prod.checked_mul(tile_prod);
    }

    None
}
