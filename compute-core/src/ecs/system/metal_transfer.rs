use crate::ecs::component::backend::{BackendComponent, MetalDeviceState, TensorComponent};
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
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
        //
        // Stage every per-tensor `TensorComponent` mutation on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` calls outside the WorldTxn seam
        // are forbidden. Extract-mutate-insert pattern.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in &tensor_entities {
            let Some(tensor) = world.get_component::<TensorComponent>(*entity).cloned() else {
                continue;
            };
            let mut updated = tensor;
            // If tensor is not resident on any available backend,
            // mark it as pending transfer to the first metal backend.
            if updated.residency == "none" || updated.residency == "cpu" {
                updated.residency = "metal".into();
            }
            if let Err(e) = txn.stage_insert(*entity, updated) {
                tracing::warn!(entity = ?entity, error = %e, "metal_transfer: stage_insert TensorComponent");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "metal_transfer: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("metal_transfer: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
