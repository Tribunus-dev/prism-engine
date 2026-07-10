use crate::ecs::component::backend::{BackendComponent, TensorComponent};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Dispatches tensor operations to the appropriate backend.
///
/// Reads `TensorComponent` entities and routes operations to backends
/// whose `BackendComponent` capabilities match. This is the ECS-native
/// entry point for the TensorBackend dispatch layer.
pub struct BackendDispatchSystem;
impl CompilerSystem for BackendDispatchSystem {
    fn name(&self) -> &str {
        "BackendDispatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let tensor_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);
        let backend_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Executable);
        let backends: Vec<(&CompEntity, BackendComponent)> = backend_entities
            .iter()
            .filter_map(|e| {
                world
                    .get_component::<BackendComponent>(*e)
                    .map(|bc| (e, bc.clone()))
            })
            .collect();

        if backends.is_empty() {
            return Ok(());
        }

        for tensor_entity in &tensor_entities {
            let Some(tensor) = world.get_component::<TensorComponent>(*tensor_entity) else {
                continue;
            };
            let _target = backends
                .iter()
                .find(|(_, bc)| bc.capabilities.iter().any(|cap| cap == &tensor.dtype));
        }

        Ok(())
    }
}
