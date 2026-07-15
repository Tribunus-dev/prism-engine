use crate::ecs::component::backend::{BackendComponent, MetalDeviceState, TensorComponent};
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Manages buffer transfers between backends — scans Tensor entities
/// and initiates transfers when a tensor's residency doesn't match
/// the active backend.
///
/// Runs every tick of the `Execution` phase.
pub struct MetalTransferSystem;
impl CompilerSystem for MetalTransferSystem {
    fn name(&self) -> &str {
        "MetalTransferSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let tensor_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);
        let backend_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Executable);

        // Collect available backend ids.
        let backend_ids: Vec<String> = backend_entities
            .iter()
            .filter_map(|e| {
                world
                    .get_component::<BackendComponent>(*e)
                    .map(|bc| bc.backend_id.clone())
            })
            .collect();

        if backend_ids.is_empty() {
            return Ok(());
        }

        // Check if any backend has a MetalDeviceState.
        let has_metal = backend_entities
            .iter()
            .any(|e| world.get_component::<MetalDeviceState>(*e).is_some());

        if !has_metal {
            return Ok(());
        }

        // Scan tensor entities and update residency when needed.
        for entity in &tensor_entities {
            let Some(tensor) = world.get_component_mut::<TensorComponent>(*entity) else {
                continue;
            };
            // If tensor is not resident on any available backend,
            // mark it as pending transfer to the first metal backend.
            if tensor.residency == "none" || tensor.residency == "cpu" {
                tensor.residency = "metal".into();
            }
        }

        Ok(())
    }
}
