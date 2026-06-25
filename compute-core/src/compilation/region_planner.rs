//! Region planner — builds a RegionExecutionPlan from a CanonicalModel
//! using the region catalogue for placement decisions.
//!
//! PRISM-MODEL-TO-CIMAGE-0001 — PhaseIR lowering step.
//!
//! For each layer, builds the canonical op chain (AttnNorm → Q/K/V → RoPE →
//! attention → O → residual; MlpNorm → gate → SiLU → up → down → residual),
//! looks up each op in the region catalogue, and partitions into Core ML
//! islands, Metal ops, and CPU ops.

use serde::{Deserialize, Serialize};

use crate::compilation::region_catalogue::{RegionAdmission, RegionCatalogue};
use crate::model_adapter::CanonicalModel;
use crate::compilation::ane_eligibility::AneEligibility;
use crate::compilation::phase_ir::{PhaseRegion, RegionId};

// ── Scheduled operation ──────────────────────────────────────────────────

/// A single scheduled operation in the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledOp {
    /// Index in the flat op list.
    pub op_index: u32,
    /// Layer number (n_layers for final norm / lm_head).
    pub layer: u32,
    /// Operator family from the catalogue (e.g. "q_projection").
    pub operator_family: String,
    /// Human-readable role (e.g. "attn_norm", "lm_head").
    pub role: String,
    /// Admission status from the catalogue.
    pub admission: RegionAdmission,
    /// Core ML island id if this op is part of a Core ML island.
    pub island_id: Option<u32>,
}

// ── Core ML island ───────────────────────────────────────────────────────

/// A contiguous block of CoreMlProduction ops compiled into one .mlpackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlIsland {
    pub island_id: u32,
    pub ops: Vec<ScheduledOp>,
    pub input_slots: Vec<u32>,
    pub output_slots: Vec<u32>,
}

// ── Region execution plan ────────────────────────────────────────────────

/// Complete region execution plan for a decoder model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionExecutionPlan {
    /// Number of decoder layers.
    pub n_layers: u32,
    /// Flat ordered list of all ops.
    pub ops: Vec<ScheduledOp>,
    /// Core ML islands (contiguous CoreMlProduction runs).
    pub coreml_islands: Vec<CoreMlIsland>,
    /// Metal production ops.
    pub metal_ops: Vec<ScheduledOp>,
    /// CPU production ops.
    pub cpu_ops: Vec<ScheduledOp>,
}

// ── Builder ──────────────────────────────────────────────────────────────

/// Build a region execution plan from a canonical model and the region catalogue.
///
/// For each decoder layer, emits the canonical op chain in execution order:
///
/// Attention block:
///   rms_norm → q_projection → k_projection → v_projection
///   → rotary_embedding → attention_score → softmax
///   → attention_value_aggregation → output_projection → residual_add
///
/// MLP block:
///   rms_norm → gate_projection → silu_activation
///   → up_projection → down_projection → residual_add
///
/// After all layers:
///   final_norm → logits_projection
pub fn build_region_plan(
    model: &CanonicalModel,
    catalogue: &RegionCatalogue,
) -> RegionExecutionPlan {
    let n_layers = model.architecture.num_hidden_layers;
    let mut ops: Vec<ScheduledOp> = Vec::new();
    let mut op_index: u32 = 0;

    for layer in 0..n_layers {
        // Attention block
        push_op(&mut ops, &mut op_index, layer, "rms_norm", "attn_norm", catalogue);
        push_op(&mut ops, &mut op_index, layer, "q_projection", "q", catalogue);
        push_op(&mut ops, &mut op_index, layer, "k_projection", "k", catalogue);
        push_op(&mut ops, &mut op_index, layer, "v_projection", "v", catalogue);
        push_op(&mut ops, &mut op_index, layer, "rotary_embedding", "rope", catalogue);
        push_op(&mut ops, &mut op_index, layer, "attention_score", "attn_score", catalogue);
        push_op(&mut ops, &mut op_index, layer, "softmax", "softmax", catalogue);
        push_op(&mut ops, &mut op_index, layer, "attention_value_aggregation", "attn_aggregate", catalogue);
        push_op(&mut ops, &mut op_index, layer, "output_projection", "o", catalogue);
        push_op(&mut ops, &mut op_index, layer, "residual_add", "residual_attn", catalogue);

        // MLP block
        push_op(&mut ops, &mut op_index, layer, "rms_norm", "mlp_norm", catalogue);
        push_op(&mut ops, &mut op_index, layer, "gate_projection", "gate", catalogue);
        push_op(&mut ops, &mut op_index, layer, "silu_activation", "silu", catalogue);
        push_op(&mut ops, &mut op_index, layer, "up_projection", "up", catalogue);
        push_op(&mut ops, &mut op_index, layer, "down_projection", "down", catalogue);
        push_op(&mut ops, &mut op_index, layer, "residual_add", "residual_mlp", catalogue);
    }

    // Post-layer ops
    push_op(&mut ops, &mut op_index, n_layers, "final_norm", "final_norm", catalogue);
    push_op(&mut ops, &mut op_index, n_layers, "logits_projection", "lm_head", catalogue);

    // Partition into Core ML islands
    let coreml_islands = partition_islands(&ops);

    let metal_ops: Vec<ScheduledOp> = ops
        .iter()
        .filter(|o| matches!(o.admission, RegionAdmission::MetalProduction))
        .cloned()
        .collect();
    let cpu_ops: Vec<ScheduledOp> = ops
        .iter()
        .filter(|o| matches!(o.admission, RegionAdmission::CpuProduction))
        .cloned()
        .collect();

    RegionExecutionPlan {
        n_layers,
        ops,
        coreml_islands,
        metal_ops,
        cpu_ops,
    }
}

fn push_op(
    ops: &mut Vec<ScheduledOp>,
    op_index: &mut u32,
    layer: u32,
    family: &str,
    role: &str,
    cat: &RegionCatalogue,
) {
    let entry = cat.find(family).unwrap_or_else(|| {
        panic!("catalogue missing required op: {family}")
    });
    ops.push(ScheduledOp {
        op_index: *op_index,
        layer,
        operator_family: family.to_string(),
        role: role.to_string(),
        admission: entry.primary_admission,
        island_id: None,
    });
    *op_index += 1;
}

fn partition_islands(ops: &[ScheduledOp]) -> Vec<CoreMlIsland> {
    let mut islands = Vec::new();
    let mut island_id = 0u32;
    let mut i = 0;
    while i < ops.len() {
        if matches!(ops[i].admission, RegionAdmission::CoreMlProduction) {
            let start = i;
            while i < ops.len() && matches!(ops[i].admission, RegionAdmission::CoreMlProduction) {
                i += 1;
            }
            let island_ops: Vec<ScheduledOp> = ops[start..i]
                .iter()
                .map(|op| {
                    let mut cloned = op.clone();
                    cloned.island_id = Some(island_id);
                    cloned
                })
                .collect();
            islands.push(CoreMlIsland {
                island_id,
                ops: island_ops,
                input_slots: vec![],
                output_slots: vec![],
            });
            island_id += 1;
        } else {
            i += 1;
        }
    }
    islands
}

/// Run the ANE eligibility pass on a set of operations and produce PhaseRegions.
/// Each region is a contiguous group of ops with homogeneous eligibility.
pub fn build_phase_regions(
    plan: &RegionExecutionPlan,
) -> Vec<PhaseRegion> {
    // For now: return one PhaseRegion per CoreML island, marked as MetalOnly.
    // Full eligibility integration will run AneEligibility::classify_ane_eligibility().
    let mut regions = Vec::new();
    for island in &plan.coreml_islands {
        // Map CoreMlIsland to PhaseRegion
        let region_id = island.island_id as RegionId;
        let ops: Vec<crate::compilation::phase_ir::CompilePhaseDescriptor> = Vec::new(); // stub
        let eligibility = AneEligibility {
            status: crate::compilation::ane_eligibility::AneEligibilityStatus::Deferred,
            shape_class: crate::compilation::ane_eligibility::AneShapeClass::MetalOnly,
            rejection_reason: None,
            qualified_buckets: vec![],
            input_layout_contract: crate::compilation::region_catalogue::LayoutContract {
                layout: String::new(),
                preferred: false,
                stride_constraints: vec![],
            },
            output_layout_contract: crate::compilation::region_catalogue::LayoutContract {
                layout: String::new(),
                preferred: false,
                stride_constraints: vec![],
            },
            evidence_requirements: vec![],
        };
        regions.push(PhaseRegion {
            region_id,
            operations: ops,
            placement_candidates: vec![],
            ane_eligibility: eligibility,
            input_contract: None,
            output_contract: None,
        });
    }
    regions
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_adapter::CanonicalModel;
    use crate::config::{AttentionKind, RopeSpec, TextArchitecture};

    fn make_test_canonical(n_layers: u32) -> CanonicalModel {
        let arch = TextArchitecture {
            hidden_size: 896,
            intermediate_size: 4864,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 128,
            global_head_dim: None,
            num_global_key_value_heads: None,
            num_hidden_layers: n_layers,
            vocab_size: 151936,
            sliding_window: 0,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_k_eq_v: true,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 0,
            model_type: "qwen2".into(),
            layer_types: vec![AttentionKind::FullAttention; n_layers as usize],
            rope_local: RopeSpec {
                theta: 1000000.0,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            moe_config: None,
            diffusion_config: None,
        };
        CanonicalModel {
            architecture: arch,
            tensors: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_build_plan_has_expected_op_count() {
        let model = make_test_canonical(2);
        let cat = RegionCatalogue::fp16_alpha();
        let plan = build_region_plan(&model, &cat);

        // 2 layers * (10 attention + 6 mlp) + 2 post-layer = 34 ops
        assert_eq!(plan.ops.len(), 34, "2-layer model should produce 34 ops");
        assert_eq!(plan.n_layers, 2);
    }

    #[test]
    fn test_every_op_has_catalogue_match() {
        let model = make_test_canonical(1);
        let cat = RegionCatalogue::fp16_alpha();
        let plan = build_region_plan(&model, &cat);

        for op in &plan.ops {
            // If any op is missing, find() would have panicked in push_op
            assert!(cat.find(&op.operator_family).is_some());
        }
    }

    #[test]
    fn test_coreml_islands_partitioned_correctly() {
        let model = make_test_canonical(1);
        let cat = RegionCatalogue::fp16_alpha();
        let plan = build_region_plan(&model, &cat);

        // Core ML ops: q, k, v, output_projection, gate, up, down, logits
        // These should appear as islands when adjacent
        for island in &plan.coreml_islands {
            for op in &island.ops {
                assert!(matches!(op.admission, RegionAdmission::CoreMlProduction));
                assert_eq!(op.island_id, Some(island.island_id));
            }
        }
    }

    #[test]
    fn test_metal_ops_are_marked() {
        let model = make_test_canonical(1);
        let cat = RegionCatalogue::fp16_alpha();
        let plan = build_region_plan(&model, &cat);

        for op in &plan.metal_ops {
            assert!(matches!(op.admission, RegionAdmission::MetalProduction));
        }
    }

    #[test]
    fn test_all_ops_accounted_in_partitions() {
        let model = make_test_canonical(1);
        let cat = RegionCatalogue::fp16_alpha();
        let plan = build_region_plan(&model, &cat);

        let partitioned: u32 = plan.coreml_islands.iter()
            .map(|i| i.ops.len() as u32)
            .sum::<u32>()
            + plan.metal_ops.len() as u32
            + plan.cpu_ops.len() as u32;
        // Each op belongs to exactly one partition (island or metal or cpu)
        assert_eq!(partitioned, plan.ops.len() as u32,
            "all {} ops must be in a partition", plan.ops.len());
    }
}
