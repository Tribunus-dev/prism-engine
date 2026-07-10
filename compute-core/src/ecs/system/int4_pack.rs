//! ECS-native INT4 packing — wraps `compute_image::compile::int4_pack`.
//!
//! Iterates over tensor entities that carry raw f32 weight data (stored
//! as `SourceWeightsData`) and repacks each into `TernaryBlock32` blocks,
//! storing the result as a `TernaryPackResult` component.

use crate::ecs::component::model_source::TernaryPackResult;
use crate::ecs::component::tensor::Shape;
use crate::ecs::compute_image::compile::int4_pack::{
    quantize_to_ternary_block32, repack_ternary_tensor,
};
use crate::ecs::Component;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Raw f32 weight data for a tensor — populated by source loading or
/// draft loading systems for downstream consumption.
#[derive(Debug, Clone)]
pub struct SourceWeightsData(pub Vec<f32>);
impl Component for SourceWeightsData {}

/// Iterates over Tensor entities that carry `SourceWeightsData` and
/// `Shape` components, repacks each into TernaryBlock32 blocks, and
/// attaches a `TernaryPackResult`.
pub struct Int4PackSystem;

impl CompilerSystem for Int4PackSystem {
    fn name(&self) -> &str {
        "Int4PackSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Quantization
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let tensor_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);
        if tensor_entities.is_empty() {
            return Ok(());
        }

        for &entity in &tensor_entities {
            // Skip if already packed.
            if world.get_component::<TernaryPackResult>(entity).is_some() {
                continue;
            }

            let shape = match world.get_component::<Shape>(entity) {
                Some(s) => s.0.clone(),
                None => continue,
            };
            let total_elements: usize = shape.iter().map(|&d| d as usize).product();
            if total_elements == 0 {
                continue;
            }

            // Read raw weights from the entity.
            let weights = match world.get_component::<SourceWeightsData>(entity) {
                Some(d) => &d.0,
                None => continue,
            };

            if weights.len() < total_elements {
                tracing::warn!(
                    "Int4PackSystem: entity {:?} has Shape {:?} ({} elements) \
                     but only {} weights; padding",
                    entity,
                    shape,
                    total_elements,
                    weights.len(),
                );
            }

            // Pack into TernaryBlock32 blocks.
            let packed = pack_f32_to_ternary_blocks(weights);
            let block_count = packed.len() as u32 / 9;

            world.add_component(
                entity,
                TernaryPackResult {
                    packed_blocks: packed,
                    block_count,
                },
            );
        }

        Ok(())
    }
}

/// Pack a contiguous f32 slice into TernaryBlock32 blocks (7 trit bytes
/// + 2 scale bytes per 32-element block).
///
/// This is the core utility used by downstream systems that have tensor
/// data in hand (e.g. from source loading or draft loading).
pub fn pack_f32_to_ternary_blocks(weights: &[f32]) -> Vec<u8> {
    let num_blocks = (weights.len() + 31) / 32;
    let mut out = Vec::with_capacity(num_blocks * 9);
    for b in 0..num_blocks {
        let start = b * 32;
        let mut block = [0.0f32; 32];
        for i in 0..32 {
            block[i] = if start + i < weights.len() {
                weights[start + i]
            } else {
                0.0
            };
        }
        let tb = quantize_to_ternary_block32(&block);
        out.extend_from_slice(&tb.packed_trits);
        out.extend_from_slice(&tb.block_scale.to_le_bytes());
    }
    out
}

/// Repack a .cimage-style u32 ternary tensor into TernaryBlock32 format.
pub fn repack_ternary_u32_tensor(src: &[u32]) -> Vec<u8> {
    let blocks = repack_ternary_tensor(src);
    let mut out = Vec::with_capacity(blocks.len() * 9);
    for tb in &blocks {
        out.extend_from_slice(&tb.packed_trits);
        out.extend_from_slice(&tb.block_scale.to_le_bytes());
    }
    out
}
