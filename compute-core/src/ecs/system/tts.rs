//! ECS-native TTS compilation — wraps `compute_image::compile::tts_compile`.
//!
//! Packs Qwen3-TTS weights as nf4tile640 cimage segments and stores
//! triplet paths on the model entity.

#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]

use std::path::PathBuf;

use crate::ecs::component::model_source::TtsWeightsComp;
use crate::ecs::compute_image::compile::tts_compile::pack_tts_weights;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Pack TTS weights into nf4tile640 cimage segment files.
///
/// Reads a `.safetensors` checkpoint from `safetensors_path`, writes
/// triplet segment files to `output_dir`, and attaches `TtsWeightsComp`
/// on the model entity pointing to the talker triplet.
pub struct TTSSystem {
    pub safetensors_path: PathBuf,
    pub output_dir: PathBuf,
}

impl CompilerSystem for TTSSystem {
    fn name(&self) -> &str {
        "TTSSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // pack_tts_weights writes files to output_dir and returns TensorEntries.
        // The filesystem side-effect is intentionally NOT routed
        // through the WorldTxn — the WorldTxn is the canonical
        // authority seam for ECS state, not for external resources.
        let _entries = pack_tts_weights(&self.safetensors_path, &self.output_dir)
            .map_err(|e| anyhow::anyhow!("TTS weight packing failed: {e}"))?;

        let weight_path = self.output_dir.join("tts_talker_weight.bin");
        let scale_path = self.output_dir.join("tts_talker_scale.bin");
        let bias_path = self.output_dir.join("tts_talker_bias.bin");

        let model_entities = world.entities_of_kind(EntityKind::Model);
        // Stage every spawn + insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.spawn` / `world.add_component` calls outside the
        // WorldTxn seam are forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        if let Some(entity) = model_entities.first() {
            if let Err(e) = txn.stage_insert(
                *entity,
                TtsWeightsComp {
                    weight_path,
                    scale_path,
                    bias_path,
                },
            ) {
                tracing::warn!(entity = ?entity, error = %e, "tts: stage_insert TtsWeightsComp (existing model)");
            }
        } else {
            let token = txn.stage_spawn(EntityKind::Model, Some("tts_weights".into()));
            if let Err(e) = txn.stage_insert_on(
                token,
                TtsWeightsComp {
                    weight_path,
                    scale_path,
                    bias_path,
                },
            ) {
                tracing::warn!(error = %e, "tts: stage_insert_on TtsWeightsComp (new model)");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "tts: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("tts: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
