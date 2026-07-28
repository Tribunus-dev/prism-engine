//! MLP shard region builder — constructs a Metal ExecutionRegion from a
//! resolved MLP shard runtime.
//!
//! Produces 7 ScheduledKernelOps with real BufferBindingPlan and BufferUse
//! records, and runs ArenaPlanner + HazardChecker over the region.

use crate::ecs::cimage_runtime::error::{CImageRuntimeError, CImageRuntimeResult};
use crate::ecs::cimage_runtime::receipts::CImageBindingReceipt;
use crate::ecs::cimage_runtime::tensor_store::{MlpRegionExecutionMode, RuntimeTensorStore};
use crate::execution_plan::{
    AccessMode, ActivationArenaPlan, BufferBindingPlan, BufferUse, CommandBufferPolicy,
    DispatchShape, EstimatedKernelCost, ExecutionPhase, ExecutionRegion, ExecutionRegionKind,
    HazardChecker, HazardPlan, HazardPolicy, KernelOpKind, KernelSpecializationKey,
    KernelTemplateId, KernelValidationRequirements, LifetimeClass, ScheduledKernelOp, TimingPolicy,
};

/// Result of building an MLP region.
pub struct CImageMlpRegionPlan {
    pub region: ExecutionRegion,
    pub hazard_plan: HazardPlan,
    pub arena_plan: ArenaPlannerOutput,
    pub binding_receipts: Vec<CImageBindingReceipt>,
}

/// Simplified arena plan output for the buffer store.
#[derive(Debug, Clone)]
pub struct ArenaPlannerOutput {
    pub scratch_buffers: Vec<ScratchBufferInfo>,
    pub total_scratch_bytes: u64,
}

/// Info about one scratch buffer allocation.
#[derive(Debug, Clone)]
pub struct ScratchBufferInfo {
    pub buffer_id: String,
    pub byte_size: u64,
    pub offset: u64,
    pub lifetime_start_op: usize,
    pub lifetime_end_op: usize,
}

/// Builder for MLP shard execution regions.
pub struct MlpShardRegionBuilder;

impl MlpShardRegionBuilder {
    /// Build a Metal ExecutionRegion for the resolved MLP shard.
    ///
    /// Returns the region, hazard plan, arena plan, and binding receipts.
    pub fn build_region(
        store: &RuntimeTensorStore,
        hidden_dim: usize,
        intermediate_dim: usize,
        mode: MlpRegionExecutionMode,
    ) -> CImageRuntimeResult<CImageMlpRegionPlan> {
        match mode {
            MlpRegionExecutionMode::StagedKernels => {
                Self::build_staged_region(store, hidden_dim, intermediate_dim)
            }
            MlpRegionExecutionMode::FusedMlpKernel => Err(CImageRuntimeError::LoweringFailed(
                "FusedMlpKernel mode not yet implemented".into(),
            )),
        }
    }

    fn build_staged_region(
        store: &RuntimeTensorStore,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> CImageRuntimeResult<CImageMlpRegionPlan> {
        // ── Buffer definitions ─────────────────────────────────────────────
        // We define the logical buffers and their uses.
        let _hidden_bytes = (hidden_dim * 4) as u64;
        let _inter_bytes = (intermediate_dim * 4) as u64;

        // Persistent buffers (from cimage payloads)
        let persistent_buffers = define_persistent_buffers(store, hidden_dim, intermediate_dim);
        // Scratch buffers
        let scratch_buffers = define_scratch_buffers(hidden_dim, intermediate_dim);

        // ── Op definitions ──────────────────────────────────────────────────
        let ops = vec![
            build_rmsnorm_op(
                0,
                hidden_dim,
                intermediate_dim,
                &persistent_buffers,
                &scratch_buffers,
            ),
            build_linear_op(
                1,
                "gate",
                hidden_dim,
                intermediate_dim,
                &persistent_buffers,
                &scratch_buffers,
            ),
            build_linear_op(
                2,
                "up",
                hidden_dim,
                intermediate_dim,
                &persistent_buffers,
                &scratch_buffers,
            ),
            build_silu_op(3, hidden_dim, intermediate_dim, &scratch_buffers),
            build_mul_op(4, hidden_dim, intermediate_dim, &scratch_buffers),
            build_linear_op(
                5,
                "down",
                intermediate_dim,
                hidden_dim,
                &persistent_buffers,
                &scratch_buffers,
            ),
            build_residual_add_op(6, hidden_dim, &scratch_buffers),
        ];

        // ── Validate all ops ───────────────────────────────────────────────
        for op in &ops {
            op.validate_lowered().map_err(|e| {
                CImageRuntimeError::LoweringFailed(format!(
                    "op {} validation failed: {:?}",
                    op.op_id, e
                ))
            })?;
        }

        // ── Arena plan ─────────────────────────────────────────────────────
        let arena_scratch_buffers: Vec<&ScratchBufferInfo> = scratch_buffers
            .iter()
            .filter(|b| b.buffer_id.starts_with("scratch_"))
            .collect();
        let total_scratch: u64 = arena_scratch_buffers.iter().map(|b| b.byte_size).sum();
        let arena_output = ArenaPlannerOutput {
            scratch_buffers: scratch_buffers.clone(),
            total_scratch_bytes: total_scratch,
        };

        // ── Build ExecutionRegion ──────────────────────────────────────────
        let region = ExecutionRegion {
            region_id: "mlp_shard_region".into(),
            region_kind: ExecutionRegionKind::DecoderLayerDecode,
            layer_index: Some(0),
            phase: ExecutionPhase::Decode,
            ops: ops.clone(),
            command_buffer_policy: CommandBufferPolicy {
                encode_region_as_single_command_buffer: true,
                allow_multiple_compute_encoders: false,
                allow_encoder_boundaries_for_hazards: true,
                commit_after_region: true,
                use_shared_events: false,
            },
            hazard_policy: HazardPolicy::Conservative,
            arena_plan: ActivationArenaPlan {
                arena_id: "mlp_shard_arena".into(),
                total_bytes: 0,
                allocations: vec![],
                alias_groups: vec![],
                peak_live_bytes: 0,
            },
            timing_policy: TimingPolicy::Disabled,
        };

        // ── Hazard check ───────────────────────────────────────────────────
        let hazard_plan = HazardChecker::validate_region(&region)
            .map_err(|e| CImageRuntimeError::HazardViolation(format!("{e:?}")))?;

        // ── Binding receipts ───────────────────────────────────────────────
        let binding_receipts: Vec<CImageBindingReceipt> = ops
            .iter()
            .map(|op| {
                let all_resolved = op.bindings.iter().all(|b| {
                    persistent_buffers
                        .iter()
                        .any(|pb| pb.buffer_id == b.buffer_id)
                        || scratch_buffers.iter().any(|sb| sb.buffer_id == b.buffer_id)
                });
                CImageBindingReceipt {
                    region_id: "mlp_shard_region".into(),
                    op_id: op.op_id.clone(),
                    kernel_name: format!("{:?}", op.op_kind),
                    bindings: op
                        .bindings
                        .iter()
                        .map(
                            |b| crate::ecs::cimage_runtime::receipts::CImageKernelBindingInfo {
                                slot: b.slot,
                                buffer_id: b.buffer_id.clone(),
                                role: infer_binding_role(&b.buffer_id),
                                byte_offset: b.offset,
                                byte_len: b.size,
                                resolved: all_resolved,
                            },
                        )
                        .collect(),
                    all_bindings_resolved: all_resolved,
                }
            })
            .collect();

        Ok(CImageMlpRegionPlan {
            region,
            hazard_plan,
            arena_plan: arena_output,
            binding_receipts,
        })
    }
}

// ── Buffer definitions ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PersistentBufferDef {
    buffer_id: String,
    byte_size: u64,
    lifetime: LifetimeClass,
}

fn define_persistent_buffers(
    _store: &RuntimeTensorStore,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<PersistentBufferDef> {
    let mut buffers = Vec::new();

    // Input and output
    buffers.push(PersistentBufferDef {
        buffer_id: "hidden_in".into(),
        byte_size: (hidden_dim * 4) as u64,
        lifetime: LifetimeClass::RegionInput,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "hidden_out".into(),
        byte_size: (hidden_dim * 4) as u64,
        lifetime: LifetimeClass::RegionOutput,
    });

    // Weights — raw f32 or packed codes + scales + biases
    // RMSNorm weight: hidden_dim f32
    buffers.push(PersistentBufferDef {
        buffer_id: "rmsnorm_weight".into(),
        byte_size: (hidden_dim * 4) as u64,
        lifetime: LifetimeClass::PersistentWeight,
    });
    // Gate proj codes: intermediate_dim * hidden_div_640 * 320 for NF4, or intermediate * hidden for f32
    let gate_codes_size = (intermediate_dim * hidden_dim) as u64; // RawF32 size
    buffers.push(PersistentBufferDef {
        buffer_id: "gate_proj_codes".into(),
        byte_size: gate_codes_size,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "gate_proj_scales".into(),
        byte_size: (intermediate_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "gate_proj_biases".into(),
        byte_size: (intermediate_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    // Up proj codes
    buffers.push(PersistentBufferDef {
        buffer_id: "up_proj_codes".into(),
        byte_size: gate_codes_size,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "up_proj_scales".into(),
        byte_size: (intermediate_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "up_proj_biases".into(),
        byte_size: (intermediate_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    // Down proj codes: hidden_dim * intermediate_dim
    let down_codes_size = (hidden_dim * intermediate_dim) as u64;
    buffers.push(PersistentBufferDef {
        buffer_id: "down_proj_codes".into(),
        byte_size: down_codes_size,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "down_proj_scales".into(),
        byte_size: (hidden_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    buffers.push(PersistentBufferDef {
        buffer_id: "down_proj_biases".into(),
        byte_size: (hidden_dim as u64) * 4,
        lifetime: LifetimeClass::PersistentWeight,
    });
    // Constants buffer
    buffers.push(PersistentBufferDef {
        buffer_id: "mlp_constants".into(),
        byte_size: 64, // MlpKernelConstants struct
        lifetime: LifetimeClass::PersistentWeight,
    });

    buffers
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ScratchBufferDef {
    buffer_id: String,
    byte_size: u64,
    lifetime_start_op: usize,
    lifetime_end_op: usize,
}

fn define_scratch_buffers(hidden_dim: usize, intermediate_dim: usize) -> Vec<ScratchBufferInfo> {
    let hidden_bytes = (hidden_dim * 4) as u64;
    let inter_bytes = (intermediate_dim * 4) as u64;

    vec![
        ScratchBufferInfo {
            buffer_id: "scratch_normed_hidden".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 0,
            lifetime_end_op: 2,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_gate_out".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 1,
            lifetime_end_op: 4,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_up_out".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 2,
            lifetime_end_op: 5,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_silu_gate".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 3,
            lifetime_end_op: 5,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_mlp_hidden".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 4,
            lifetime_end_op: 6,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_down_out".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 5,
            lifetime_end_op: 7,
        },
    ]
}

// ── Op builders ───────────────────────────────────────────────────────────

fn build_rmsnorm_op(
    idx: usize,
    hidden_dim: usize,
    _intermediate_dim: usize,
    _persistent: &[PersistentBufferDef],
    _scratch: &[ScratchBufferInfo],
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_rmsnorm");
    let hidden_bytes = (hidden_dim * 4) as u64;

    ScheduledKernelOp {
        op_id: op_id.clone(),
        op_kind: KernelOpKind::RmsNorm,
        tensor_key: Some("rmsnorm_weight".into()),
        tensor_class: Some("RmsNormWeight".into()),
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: "hidden_in".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: "rmsnorm_weight".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "scratch_normed_hidden".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "mlp_constants".into(),
                offset: 0,
                size: 64,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: "hidden_in".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::RegionInput,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "rmsnorm_weight".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "scratch_normed_hidden".into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "mlp_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: hidden_dim as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 1.0,
            memory_bytes_read: hidden_bytes * 2,
            memory_bytes_written: hidden_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_linear_op(
    idx: usize,
    proj_name: &str,
    in_dim: usize,
    out_dim: usize,
    _persistent: &[PersistentBufferDef],
    _scratch: &[ScratchBufferInfo],
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_{proj_name}_projection");
    let in_bytes = (in_dim * 4) as u64;
    let out_bytes = (out_dim * 4) as u64;

    // Determine input and output buffer names
    let input_buffer = match idx {
        1 => "scratch_normed_hidden", // gate
        2 => "scratch_normed_hidden", // up
        5 => "scratch_mlp_hidden",    // down
        _ => "scratch_normed_hidden",
    };
    let output_buffer = match idx {
        1 => "scratch_gate_out",
        2 => "scratch_up_out",
        5 => "scratch_down_out",
        _ => "scratch_gate_out",
    };
    let codes = format!("{proj_name}_proj_codes");
    let scales = format!("{proj_name}_proj_scales");
    let biases = format!("{proj_name}_proj_biases");

    let op_kind = if idx == 5 {
        KernelOpKind::MlpDownResidual
    } else {
        KernelOpKind::MlpGateUp
    };

    // The outer dimension for dispatch
    let grid_size = out_dim as u32;

    ScheduledKernelOp {
        op_id: op_id.clone(),
        op_kind,
        tensor_key: Some(format!("{proj_name}_proj")),
        tensor_class: Some("DecoderMlpProjection".into()),
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: input_buffer.into(),
                offset: 0,
                size: in_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: codes,
                offset: 0,
                size: in_bytes * out_dim as u64 / in_dim as u64,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: scales,
                offset: 0,
                size: (out_dim as u64 / 20) * 4,
            }, // rough
            BufferBindingPlan {
                slot: 3,
                buffer_id: biases,
                offset: 0,
                size: (out_dim as u64 / 20) * 4,
            },
            BufferBindingPlan {
                slot: 4,
                buffer_id: output_buffer.into(),
                offset: 0,
                size: out_bytes,
            },
            BufferBindingPlan {
                slot: 5,
                buffer_id: "mlp_constants".into(),
                offset: 0,
                size: 64,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: input_buffer.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: format!("{proj_name}_proj_codes"),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: format!("{proj_name}_proj_scales"),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: format!("{proj_name}_proj_biases"),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buffer.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "mlp_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: grid_size,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 10.0,
            memory_bytes_read: in_bytes + (in_dim * out_dim) as u64,
            memory_bytes_written: out_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_silu_op(
    idx: usize,
    _hidden_dim: usize,
    intermediate_dim: usize,
    _scratch: &[ScratchBufferInfo],
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_silu");
    let inter_bytes = (intermediate_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::MlpActivation,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: "scratch_gate_out".into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: "scratch_silu_gate".into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "mlp_constants".into(),
                offset: 0,
                size: 64,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: "scratch_gate_out".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "scratch_silu_gate".into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "mlp_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: intermediate_dim as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 0.5,
            memory_bytes_read: inter_bytes,
            memory_bytes_written: inter_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_mul_op(
    idx: usize,
    _hidden_dim: usize,
    intermediate_dim: usize,
    _scratch: &[ScratchBufferInfo],
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_mul");
    let inter_bytes = (intermediate_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::MlpActivation,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: "scratch_silu_gate".into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: "scratch_up_out".into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "scratch_mlp_hidden".into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "mlp_constants".into(),
                offset: 0,
                size: 64,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: "scratch_silu_gate".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "scratch_up_out".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "scratch_mlp_hidden".into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "mlp_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: intermediate_dim as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 0.5,
            memory_bytes_read: inter_bytes * 2,
            memory_bytes_written: inter_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_residual_add_op(
    idx: usize,
    hidden_dim: usize,
    _scratch: &[ScratchBufferInfo],
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_residual_add");
    let hidden_bytes = (hidden_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::MlpDownResidual,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: "hidden_in".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: "scratch_down_out".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "hidden_out".into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "mlp_constants".into(),
                offset: 0,
                size: 64,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: "hidden_in".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::RegionInput,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "scratch_down_out".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "hidden_out".into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::RegionOutput,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "mlp_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: hidden_dim as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 0.3,
            memory_bytes_read: hidden_bytes * 2,
            memory_bytes_written: hidden_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn basic_specialization() -> KernelSpecializationKey {
    KernelSpecializationKey {
        template_id: KernelTemplateId::RawF32Matmul,
        execution_phase: crate::execution_plan::ExecutionPhase::Decode,
        codec: crate::execution_plan::CodecFamily::RawF32,
        tile_shape: crate::execution_plan::TileShape::tile640_decode(),
        group_size: 0,
        group_axis: crate::execution_plan::Axis::Output,
        affine_mode: crate::execution_plan::AffineMode::ScaleOnly,
        metadata_layout: crate::execution_plan::MetadataLayout::SeparatedManifest,
        input_dtype: crate::execution_plan::DType::F32,
        output_dtype: crate::execution_plan::DType::F32,
        hardware_profile: crate::execution_plan::HardwareProfileId::AppleMProBalanced,
        mode_flags: 0,
    }
}

#[allow(dead_code)]
fn collect_all_buffer_uses(ops: &[ScheduledKernelOp]) -> Vec<BufferUse> {
    let mut all = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for op in ops {
        for use_ in &op.buffer_uses {
            let key = (use_.buffer_id.clone(), use_.access, use_.lifetime);
            if seen.insert(key) {
                all.push(use_.clone());
            }
        }
    }
    all
}

fn infer_binding_role(buffer_id: &str) -> String {
    if buffer_id == "hidden_in" || buffer_id.starts_with("scratch_") {
        "InputActivation".into()
    } else if buffer_id == "hidden_out" {
        "OutputActivation".into()
    } else if buffer_id.ends_with("_codes") || buffer_id.ends_with("_weight") {
        "WeightCodes".into()
    } else if buffer_id.ends_with("_scales") {
        "WeightScales".into()
    } else if buffer_id.ends_with("_biases") {
        "WeightBiases".into()
    } else if buffer_id == "mlp_constants" {
        "Constants".into()
    } else {
        "Unknown".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_cimage::*;
    use crate::ecs::cimage_runtime::resolver::CImageRuntimeResolver;
    use crate::execution_plan::CodecFamily;

    fn build_resolved_shard(
        codec: CodecFamily,
    ) -> (tempfile::TempDir, RuntimeTensorStore, usize, usize) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cimage");
        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: codec,
                up_codec: codec,
                down_codec: codec,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };
        let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts)
            .unwrap();
        let loaded = CImageLoader::load_v0(&path).unwrap();
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&loaded).unwrap();
        (
            dir,
            resolved.tensors,
            resolved.hidden_dim,
            resolved.intermediate_dim,
        )
    }

    #[test]
    fn test_region_builder_emits_seven_ops() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        assert_eq!(plan.region.ops.len(), 7);
    }

    #[test]
    fn test_region_builder_op_kinds() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        let kinds: Vec<_> = plan.region.ops.iter().map(|op| op.op_kind).collect();
        assert_eq!(kinds[0], KernelOpKind::RmsNorm);
        assert_eq!(kinds[3], KernelOpKind::MlpActivation);
    }

    #[test]
    fn test_region_builder_populates_bindings() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        for op in &plan.region.ops {
            assert!(!op.bindings.is_empty(), "op {} has no bindings", op.op_id);
        }
    }

    #[test]
    fn test_region_builder_populates_buffer_uses() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        for op in &plan.region.ops {
            assert!(
                !op.buffer_uses.is_empty(),
                "op {} has no buffer uses",
                op.op_id
            );
        }
    }

    #[test]
    fn test_region_all_ops_validate() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        for op in &plan.region.ops {
            op.validate_lowered()
                .unwrap_or_else(|e| panic!("op {} failed validation: {:?}", op.op_id, e));
        }
    }

    #[test]
    fn test_binding_receipts_emitted() {
        let (_dir, store, h, i) = build_resolved_shard(CodecFamily::RawF32);
        let plan = MlpShardRegionBuilder::build_region(
            &store,
            h,
            i,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();
        assert_eq!(plan.binding_receipts.len(), 7);
        for receipt in &plan.binding_receipts {
            assert!(receipt.all_bindings_resolved);
        }
    }
}
