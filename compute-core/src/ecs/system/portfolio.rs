//! ECS-native Core ML portfolio compilation.
//! Wrapper around the compute_image portfolio pipeline.

#![cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]

use crate::ecs::component::model_source::PortfolioArtifactsComp;
use crate::ecs::Entity;
use crate::ecs::{CompEntity, World, CompilerSystem, EntityKind, SchedulePhase};
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
        let model_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Model);
        if let Some(entity) = model_entities.first() {
            world.add_component(
                *entity,
                PortfolioArtifactsComp {
                    artifact_paths: vec![],
                },
            );
        }
        Ok(())
    }
}
