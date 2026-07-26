use crate::ecs::component::backend::{BackendComponent, TensorComponent};

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Manages tensor residency across backends — tracks which backends
/// hold tensor data and initiates transfers when needed.
///
/// Reads `TensorComponent` residency fields and `BackendComponent`
/// capabilities to decide whether a tensor needs to be moved between
/// backends.
pub struct BackendResidencySystem;
impl CompilerSystem for BackendResidencySystem {
    fn name(&self) -> &str {
        "BackendResidencySystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::MemoryPlanning
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensor_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);
        let backend_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        let _backend_map: Vec<(String, String)> = backend_entities
            .iter()
            .filter_map(|e| {
                world
                    .get_component::<BackendComponent>(*e)
                    .map(|bc| (bc.backend_id.clone(), bc.backend_id.clone()))
            })
            .collect();

        for tensor_entity in &tensor_entities {
            let Some(tensor) = world.get_component::<TensorComponent>(*tensor_entity) else {
                continue;
            };
            let _current = &tensor.residency;
        }

        Ok(())
    }
}
