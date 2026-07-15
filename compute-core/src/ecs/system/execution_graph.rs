//! ECS-native execution graph building — wraps `compute_image::compile::execution_graph`.
//!
//! Builds an `ExecutionGraphDescriptor` from layer entities in the ECS world,
//! serialises it to binary, and stores as `ExecutionGraphComp` on the model entity.

use crate::ecs::component::model_source::ExecutionGraphComp;
use crate::ecs::component::tensor::LayerIndex;
use crate::ecs::compute_image::compile::execution_graph::{
    ExecutionGraphDescriptor, LayerExecutionNode,
};

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// Build the execution graph from the ECS world's layer/tensor state.
///
/// Iterates over Tensor entities that have a `LayerIndex`, groups them
/// by layer, and produces one `LayerExecutionNode` per layer.  Serialises
/// the descriptor and attaches `ExecutionGraphComp` to the first Model entity.
pub struct ExecutionGraphSystem;

impl CompilerSystem for ExecutionGraphSystem {
    fn name(&self) -> &str {
        "ExecutionGraphSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensor_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Tensor);

        // Collect per-layer information from tensor components.
        let mut layer_map: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        let mut nodes: Vec<LayerExecutionNode> = Vec::new();

        for &entity in &tensor_entities {
            let Some(layer_idx) = world.get_component::<LayerIndex>(entity) else {
                continue;
            };
            if !layer_map.contains_key(&layer_idx.0) {
                layer_map.insert(layer_idx.0, nodes.len());
                nodes.push(LayerExecutionNode {
                    node_kind: 0,         // DecoderLayer
                    attention_kind: 1,    // FullAttention default
                    device_capability: 1, // Gpu
                    compaction_epoch: 0xFF,
                    layer_index: layer_idx.0,
                    head_dim: 0, // filled from data
                    num_heads: 0,
                    hidden_dim: 0,
                    weight_offset: 0,
                    weight_length: 0,
                    scale_offset: 0,
                    _reserved: [0u8; 8],
                });
            }
        }

        let desc = ExecutionGraphDescriptor {
            layers: nodes,
            ..Default::default()
        };

        let serialized = desc.to_bytes();

        // Attach to the first model entity, or spawn one.
        let model_entities = world.entities_of_kind(EntityKind::Model);
        if let Some(entity) = model_entities.first() {
            let _ = world.add_component(*entity, ExecutionGraphComp(serialized));;
        } else {
            let spawn_result = world.spawn(EntityKind::Model, Some("execution_graph".into()))?;
            let _ = world.add_component(spawn_result, ExecutionGraphComp(serialized));;
        }

        Ok(())
    }
}
