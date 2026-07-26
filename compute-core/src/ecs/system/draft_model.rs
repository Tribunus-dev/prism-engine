//! ECS-native draft model loading — wraps `compute_image::compile::draft_loader`.
//!
//! Loads DSpark draft model weights from safetensors and stores the fused
//! ternary buffer as an ECS component.

use std::path::PathBuf;

use crate::ecs::component::model_source::DraftWeightsComp;
use crate::ecs::compute_image::compile::draft_loader::load_draft_weights;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

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
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let fused = load_draft_weights(&self.ckpt_dir)
            .map_err(|e| anyhow::anyhow!("draft weight load failed: {e}"))?;

        // Stage every spawn + insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let model_entities = world.entities_of_kind(EntityKind::Model);
        let mut txn = ConstitutionalWorldTxn::new();
        if let Some(entity) = model_entities.first() {
            if let Err(e) = txn.stage_insert(*entity, DraftWeightsComp(fused)) {
                tracing::warn!(entity = ?entity, error = %e, "draft_model: stage_insert DraftWeightsComp (existing model)");
            }
        } else {
            // No model entity yet — spawn one.
            let token = txn.stage_spawn(EntityKind::Model, Some("draft_model".into()));
            if let Err(e) = txn.stage_insert_on(token, DraftWeightsComp(fused)) {
                tracing::warn!(error = %e, "draft_model: stage_insert_on DraftWeightsComp (new model)");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "draft_model: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("draft_model: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
