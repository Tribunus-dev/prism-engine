//! ECS-native draft model loading — wraps `compute_image::compile::draft_loader`.
//!
//! Loads DSpark draft model weights from safetensors and stores the fused
//! ternary buffer as an ECS component.

use std::path::PathBuf;

use crate::ecs::component::model_source::DraftWeightsComp;
use crate::ecs::compute_image::compile::draft_loader::load_draft_weights;
#[allow(unused_imports)]
use crate::ecs::Entity;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Load draft model weights and attach as `DraftWeightsComp` on the model entity.
pub struct DraftModelSystem {
    pub ckpt_dir: PathBuf,
}

impl CompilerSystem for DraftModelSystem {
    fn name(&self) -> &str {
        "DraftModelSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let fused = load_draft_weights(&self.ckpt_dir)
            .map_err(|e| anyhow::anyhow!("draft weight load failed: {e}"))?;

        let model_entities = world.entities_of_kind(EntityKind::Model);
        if let Some(entity) = model_entities.first() {
            world.add_component(*entity, DraftWeightsComp(fused));
        } else {
            // No model entity yet — spawn one.
            let entity = world.spawn(EntityKind::Model, Some("draft_model".into()));
            world.add_component(entity, DraftWeightsComp(fused));
        }
        Ok(())
    }
}
