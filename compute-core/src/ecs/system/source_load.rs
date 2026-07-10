//! ECS-native source loading — wraps `compute_image::compile::source` logic.
//!
//! Loads safetensors shards, extracts tensor metadata, and spawns tensor
//! entities in the ECS world with Shape / DataType / CanonicalRoleComp.

#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]

use std::collections::HashMap;
use std::path::PathBuf;

use crate::ecs::component::tensor::{DataType, Shape};
use crate::ecs::compute_image::compile::load_source_tensor_table;
use crate::ecs::compute_image::compile::source::load_source;
use crate::ecs::Component;
use crate::ecs::{CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Source tensor metadata wrapped as an ECS component.
#[derive(Debug, Clone)]
pub struct SourceTensorMeta {
    pub raw_name: String,
    pub raw_dtype: String,
    pub sha256: String,
}
impl Component for SourceTensorMeta {}

/// ECS system that loads model source from a directory and populates tensor
/// entities with shape / dtype metadata.
pub struct SourceLoadingSystem {
    pub source_dir: PathBuf,
    pub skip_validation: bool,
}

impl CompilerSystem for SourceLoadingSystem {
    fn name(&self) -> &str {
        "SourceLoadingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let loaded = load_source(&self.source_dir, self.skip_validation)
            .map_err(|e| anyhow::anyhow!("source load failed: {e}"))?;

        // Spawn a Tensor entity per source tensor with Shape and DataType.
        for (_name, tensor) in &loaded.source_tensors {
            let entity = world.spawn(EntityKind::Tensor, Some(tensor.name.clone()));
            world.add_component(entity, Shape(tensor.shape.clone()));
            let dt = map_dtype_str(&tensor.dtype);
            world.add_component(entity, DataType(dt));
            world.add_component(
                entity,
                SourceTensorMeta {
                    raw_name: tensor.name.clone(),
                    raw_dtype: tensor.dtype.clone(),
                    sha256: tensor.source_sha256.clone(),
                },
            );
        }

        Ok(())
    }
}

/// Load a tensor hash table (for diff / validation purposes).
pub struct TensorTableLoadingSystem {
    pub source_dir: PathBuf,
}

impl CompilerSystem for TensorTableLoadingSystem {
    fn name(&self) -> &str {
        "TensorTableLoadingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let table = load_source_tensor_table(&self.source_dir)
            .map_err(|e| anyhow::anyhow!("tensor table load failed: {e}"))?;

        // Store table as a component on a synthetic entity so downstream
        // systems can reference it for differential compilation decisions.
        let meta_entity = world.spawn(EntityKind::Model, Some("tensor_table".into()));
        world.add_component(meta_entity, TensorTableComp(table));
        Ok(())
    }
}

/// Wrapper around the source tensor table for differential compilation.
#[derive(Debug, Clone)]
pub struct TensorTableComp(
    pub HashMap<String, crate::ecs::compute_image::compile::source::SourceTensorInfo>,
);
impl Component for TensorTableComp {}

/// Compute a diff against a previous manifest.
pub struct DiffSystem {
    pub source_dir: PathBuf,
}

impl CompilerSystem for DiffSystem {
    fn name(&self) -> &str {
        "DiffSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, _world: &mut CompWorld) -> anyhow::Result<()> {
        // Load manifest from the source dir — requires a saved manifest path.
        // Stub: the real call is `diff_tensors(source_dir, &prev_manifest)`.
        // We store the result on a model entity for use by downstream systems.
        tracing::warn!("DiffSystem requires a prev_manifest path; skipping");
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn map_dtype_str(dtype: &str) -> crate::ecs::component::tensor::DType {
    match dtype.to_lowercase().as_str() {
        "f32" | "float32" => crate::ecs::component::tensor::DType::F32,
        "f16" | "float16" => crate::ecs::component::tensor::DType::F16,
        "bf16" | "bfloat16" => crate::ecs::component::tensor::DType::BF16,
        "i8" | "int8" => crate::ecs::component::tensor::DType::I8,
        "i4" | "int4" => crate::ecs::component::tensor::DType::I4,
        "i2" | "int2" => crate::ecs::component::tensor::DType::I2,
        _ => crate::ecs::component::tensor::DType::F32,
    }
}
