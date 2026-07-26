use crate::ecs::adapter::CanonicalRole;
use crate::ecs::component::memory::MemoryBudget;
use crate::ecs::component::tensor::{
    CanonicalRoleComp, CodecFamilyComp, ExpertIndex, MoEConfig, Shape,
};
use crate::ecs::config::TextArchitecture;
use crate::ecs::plan::CodecFamily;

use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};
use std::collections::{HashMap, HashSet};

/// Evaluates MoE expert budget against the configured codec and topology.
pub struct MoERoutingSystem;
impl CompilerSystem for MoERoutingSystem {
    fn name(&self) -> &str {
        "MoERoutingSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Quantization
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let all_tensors = world.entities_of_kind(EntityKind::Tensor);

        // (layer -> set of expert indices)
        let mut layer_experts: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut has_shared_expert = false;

        for entity in &all_tensors {
            let Some(role_comp) = world.get_component::<CanonicalRoleComp>(*entity) else {
                continue;
            };

            match role_comp.0 {
                CanonicalRole::GateEx(layer, expert)
                | CanonicalRole::UpEx(layer, expert)
                | CanonicalRole::DownEx(layer, expert) => {
                    layer_experts.entry(layer).or_default().insert(expert);
                }
                CanonicalRole::SharedGate
                | CanonicalRole::SharedUp
                | CanonicalRole::SharedDown
                | CanonicalRole::SharedGateL(_)
                | CanonicalRole::SharedUpL(_)
                | CanonicalRole::SharedDownL(_) => {
                    has_shared_expert = true;
                }
                _ => {}
            }
        }

        // Determine total expert count and top-k from the layer with the most experts.
        let max_expert_count = layer_experts
            .values()
            .map(|set| set.len() as u32)
            .max()
            .unwrap_or(0);

        let total = max_expert_count;
        let top_k = max_expert_count;

        // Create an expert entity for each (layer, expert) pair.
        for experts in layer_experts.values() {
            for expert_idx in experts {
                let expert_entity = world.spawn(EntityKind::Expert, None)?;
                let _ = world.add_component(expert_entity,
                ExpertIndex {
                    index: *expert_idx,
                    total,
                    top_k,
                },);
            }
        }

        // Add MoEConfig to the model entity (first one).
        if total > 0 {
            for model in world.entities_of_kind(EntityKind::Model) {
                let _ = world.add_component(model,
                MoEConfig {
                    shared_expert: has_shared_expert,
                    num_experts: total,
                    top_k,
                    intermediate_size: None,
                },);
                break;
            }
        }

        Ok(())
    }
}

fn codec_bytes_per_element(codec: CodecFamily) -> f64 {
    match codec {
        CodecFamily::Nf4 => 0.5,
        CodecFamily::Int8 => 1.0,
        CodecFamily::Fp16 => 2.0,
        CodecFamily::RawF32 => 4.0,
        CodecFamily::SymInt4 => 0.5,
        CodecFamily::Ternary | CodecFamily::Ternary1_58 => 0.25,
        CodecFamily::Mixed => 4.0,
        CodecFamily::Q8_0 => 1.0,
        CodecFamily::Q4_K => 0.5,
        CodecFamily::Q2_K | CodecFamily::IQ2_XXS => 0.25,
    }
}

/// Computes the per-model memory budget from the target backend's capabilities
/// and the selected codec family.
pub struct MemoryBudgetSystem;
impl CompilerSystem for MemoryBudgetSystem {
    fn name(&self) -> &str {
        "MemoryBudgetSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Quantization
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let tensor_entities = world.entities_of_kind(EntityKind::Tensor);

        let mut weight_bytes: f64 = 0.0;

        for entity in &tensor_entities {
            let (shape, codec) = match (
                world.get_component::<Shape>(*entity),
                world.get_component::<CodecFamilyComp>(*entity),
            ) {
                (Some(s), Some(c)) => (s, c),
                _ => continue,
            };

            let num_elements: u64 = shape.0.iter().map(|&d| d as u64).product();
            let bpe = codec_bytes_per_element(codec.0);
            weight_bytes += num_elements as f64 * bpe;
        }

        // Round weight_bytes up to the nearest whole byte.
        let weight_bytes = weight_bytes.ceil() as u64;

        // Estimate scratch as 10% of weight.
        let scratch_bytes = (weight_bytes as f64 * 0.10).ceil() as u64;

        // Estimate KV-cache from model architecture or heuristics.
        let model_entities = world.entities_of_kind(EntityKind::Model);
        let (num_layers, num_heads) = if let Some(model) = model_entities.first() {
            if let Some(arch) = world.get_component::<TextArchitecture>(*model) {
                (arch.num_hidden_layers, arch.num_attention_heads)
            } else {
                (64, 32)
            }
        } else {
            (64, 32)
        };
        let seq_len: u64 = 4096;
        let kv_cache_bytes = num_layers as u64 * num_heads as u64 * seq_len * 2;

        let total_bytes = weight_bytes + scratch_bytes + kv_cache_bytes;

        // Add MemoryBudget to the first model entity.
        for model in model_entities {
            let _ = world.add_component(model,
            MemoryBudget {
                total_bytes,
                weight_bytes,
                scratch_bytes,
                kv_cache_bytes,
            },);
            break;
        }

        Ok(())
    }
}
