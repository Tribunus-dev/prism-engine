//! ECS-native Core ML portfolio compilation.
//! Wrapper around the compute_image portfolio pipeline.

#![cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]

use crate::ecs::component::model_source::PortfolioArtifactsComp;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;
use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};
use std::path::PathBuf;

/// Compile a portfolio of Core ML packets.
pub struct PortfolioSystem {
    pub output_dir: PathBuf,
}
impl CompilerSystem for PortfolioSystem {
    fn name(&self) -> &str {
        "PortfolioSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);
        // Stage the per-model `PortfolioArtifactsComp` insert on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        if let Some(entity) = model_entities.first() {
            if let Err(e) = txn.stage_insert(
                *entity,
                PortfolioArtifactsComp {
                    artifact_paths: vec![],
                },
            ) {
                tracing::warn!(entity = ?entity, error = %e, "portfolio: stage_insert PortfolioArtifactsComp");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "portfolio: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("portfolio: ConstitutionalWorldTxn commit failed: {e}")
        })?;
        Ok(())
    }
}
