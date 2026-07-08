//! Fusion-aware planner — bridges the dataflow graph and fusion scheduler
//! into the execution plan pipeline.

use serde::{Deserialize, Serialize};

use crate::execution_plan::backend_capability::{BackendCapabilityRegistry, BackendRole};
use crate::execution_plan::fusion::{DataflowGraph, DataflowNode, DataflowOp, FusedGroup};
use crate::execution_plan::fusion_scheduler::{
    FusionPolicy, FusionSchedule, FusionScheduler, FusionSelectionPolicy,
};
use crate::execution_plan::{
    ActivationArenaPlan, CodecFamily, CommandBufferPolicy, DispatchShape,
    EstimatedKernelCost, ExecutionMode, ExecutionPhase, ExecutionRegion,
    ExecutionRegionKind, HardwareProfileId, KernelOpKind, KernelSpecializationKey,
    ModelExecutionPlan, ModelExecutionPlanReceipt, ScheduledKernelOp, TileShape,
};

// ── PipelineGraphBuilder ──────────────────────────────────────────────────

/// Thin convenience wrapper around the canonical `DataflowGraphBuilder`
/// in `execution_plan::fusion`.
pub struct PipelineGraphBuilder;

impl PipelineGraphBuilder {
    /// Build a canonical Gemma decoder MLP dataflow graph.
    pub fn build_mlp() -> DataflowGraph {
        use crate::execution_plan::fusion::DataflowGraphBuilder;
        DataflowGraphBuilder::build_mlp()
    }
}

/// Map a `DataflowOp` variant to the corresponding `KernelOpKind`.
fn map_op_to_kernel_kind(op: &DataflowOp) -> KernelOpKind {
    match op {
        DataflowOp::RmsNorm { .. } => KernelOpKind::RmsNorm,
        DataflowOp::MatMul { .. } => KernelOpKind::MlpGateUp,
        DataflowOp::SiLU { .. } | DataflowOp::Gelu { .. } => KernelOpKind::MlpActivation,
        DataflowOp::Add { .. } | DataflowOp::ResidualAdd { .. } => {
            KernelOpKind::OProjectionResidual
        }
        DataflowOp::Mul { .. } => KernelOpKind::MlpActivation,
        DataflowOp::KvRead { .. } => KernelOpKind::AttentionScore,
        DataflowOp::KvWrite { .. } => KernelOpKind::AttentionApply,
        DataflowOp::LoadWeight { .. } | DataflowOp::Dequantize { .. } => {
            KernelOpKind::RmsNorm
        }
        DataflowOp::StoreActivation { .. } => KernelOpKind::RmsNorm,
    }
}

/// Build a placeholder specialization key for planner-level scheduling.
fn placeholder_specialization(
    hardware_profile: HardwareProfileId,
) -> KernelSpecializationKey {
    KernelSpecializationKey {
        template_id: crate::execution_plan::KernelTemplateId::Nf4Tile640Gemv,
        execution_phase: ExecutionPhase::Decode,
        codec: CodecFamily::RawF32,
        tile_shape: TileShape::tile640_decode(),
        group_size: 0,
        group_axis: crate::execution_plan::Axis::PackedContiguous,
        affine_mode: crate::execution_plan::AffineMode::ScaleOnly,
        metadata_layout: crate::execution_plan::MetadataLayout::AdjacentTile,
        input_dtype: crate::execution_plan::DType::F32,
        output_dtype: crate::execution_plan::DType::F16,
        hardware_profile,
        mode_flags: 0,
    }
}

// ── ExecutionPlanner ──────────────────────────────────────────────────────

/// Converts a resolved `DataflowGraph` into a `ModelExecutionPlan`.
pub struct ExecutionPlanner;

impl ExecutionPlanner {
    /// Produce a complete `ModelExecutionPlan` from a resolved dataflow graph.
    pub fn plan_with_fusion(
        graph: DataflowGraph,
        registry: Option<BackendCapabilityRegistry>,
        target: HardwareProfileId,
        execution_mode: ExecutionMode,
    ) -> (ModelExecutionPlan, ModelExecutionPlanReceipt) {
        // Build the effective registry based on execution mode.
        let registry = match execution_mode {
            ExecutionMode::OpByOp => BackendCapabilityRegistry::new(),
            _ => registry.unwrap_or_else(BackendCapabilityRegistry::new),
        };

        let scheduler = FusionScheduler::new(registry);

        let policy = FusionPolicy {
            max_group_size: match execution_mode {
                ExecutionMode::MegakernelExperimental => 16,
                _ => 8,
            },
            allow_materialization: true,
            forbid_cross_lane: true,
            allow_research_fusions: matches!(
                execution_mode,
                ExecutionMode::MegakernelExperimental
            ),
        };

        let selection_policy = FusionSelectionPolicy {
            prefer_lower_latency: true,
            prefer_memory_efficient: true,
            avoid_materialization: true,
        };

        // Schedule fused groups; fall back to singletons on error.
        let schedule: FusionSchedule = scheduler
            .schedule(
                &graph,
                &policy,
                &selection_policy,
                BackendRole::ProductionHotPath,
            )
            .unwrap_or_else(|_| FusionSchedule {
                groups: graph
                    .nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| FusedGroup {
                        id: format!("singleton_{i}"),
                        body: vec![node.clone()],
                        inputs: node.inputs.clone(),
                        outputs: node.outputs.clone(),
                        internal_values: vec![],
                        codec_family: CodecFamily::RawF32,
                    })
                    .collect(),
                receipts: vec![],
            });

        // Lower groups to execution regions.
        let (regions, pso_keys, scratch_bytes) =
            Self::lower_groups(&schedule.groups, &target);

        // Plan identity.
        let plan_id = format!("plan_{:016x}", millis_since_epoch());

        let plan = ModelExecutionPlan {
            plan_id,
            model_family: String::new(),
            cimage_digest: String::new(),
            policy_digest: String::new(),
            layout_profile: target,
            regions,
            pso_keys,
            total_scratch_budget_bytes: scratch_bytes,
            validation_digest: None,
            execution_mode,
        };

        // Receipt.
        let op_count: usize = schedule.groups.iter().map(|g| g.body.len()).sum();
        let fused_count = schedule.groups.iter().filter(|g| g.body.len() > 1).count();
        let mut fallbacks: Vec<String> = Vec::new();
        if fused_count == 0 && !matches!(execution_mode, ExecutionMode::OpByOp) {
            fallbacks.push("all_groups_fell_back_to_singletons".into());
        }

        let receipt = ModelExecutionPlanReceipt {
            plan_id: plan.plan_id.clone(),
            cimage_digest: plan.cimage_digest.clone(),
            policy_digest: plan.policy_digest.clone(),
            layout_profile: target,
            region_count: plan.regions.len(),
            scheduled_op_count: op_count,
            pso_count: plan.pso_keys.len(),
            peak_scratch_bytes: scratch_bytes,
            unsupported_ops: vec![],
            fallbacks,
            warnings: vec![],
        };

        (plan, receipt)
    }

    /// Lower `FusedGroup`s into `ExecutionRegion`s.
    fn lower_groups(
        groups: &[FusedGroup],
        target: &HardwareProfileId,
    ) -> (Vec<ExecutionRegion>, Vec<KernelSpecializationKey>, u64) {
        let mut regions = Vec::with_capacity(groups.len());
        let mut all_keys = Vec::new();
        let mut total_scratch: u64 = 0;

        for group in groups {
            let region_kind = if group.body.len() > 1 {
                ExecutionRegionKind::Fused
            } else {
                ExecutionRegionKind::DecoderLayerDecode
            };

            let ops: Vec<ScheduledKernelOp> = group
                .body
                .iter()
                .map(|node| {
                    let skey = placeholder_specialization(*target);
                    all_keys.push(skey.clone());
                    ScheduledKernelOp {
                        op_id: format!("df_node_{}", node.id),
                        op_kind: map_op_to_kernel_kind(&node.op),
                        tensor_key: None,
                        tensor_class: None,
                        specialization: skey,
                        bindings: vec![],
                        dependencies: vec![],
                        buffer_uses: vec![],
                        dispatch_shape: DispatchShape {
                            grid_x: 1,
                            grid_y: 1,
                            grid_z: 1,
                            threadgroup_m: 1,
                            threadgroup_n: 1,
                            threadgroup_p: 1,
                        },
                        estimated_cost: EstimatedKernelCost {
                            compute_us: 0.0,
                            memory_bytes_read: 0,
                            memory_bytes_written: 0,
                        },
                        validation_requirements: Default::default(),
                    }
                })
                .collect();

            total_scratch += (ops.len() as u64) * 4096;

            regions.push(ExecutionRegion {
                region_id: group.id.clone(),
                region_kind,
                layer_index: None,
                phase: ExecutionPhase::Decode,
                ops,
                command_buffer_policy: CommandBufferPolicy::decode_default(),
                hazard_policy: crate::execution_plan::HazardPolicy::Conservative,
                arena_plan: ActivationArenaPlan {
                    arena_id: format!("arena_{}", group.id),
                    total_bytes: 0,
                    allocations: vec![],
                    alias_groups: vec![],
                    peak_live_bytes: 0,
                },
                timing_policy: crate::execution_plan::TimingPolicy::Disabled,
            });
        }

        (regions, all_keys, total_scratch)
    }
}

/// Milliseconds since UNIX epoch for plan identity generation.
fn millis_since_epoch() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}
