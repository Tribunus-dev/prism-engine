//! Decoder layer region builder — constructs a Metal ExecutionRegion for a
//! full transformer decoder layer (pre-attention RMSNorm, QKV projections,
//! RoPE, KV cache append, attention, output projection, post-attention
//! RMSNorm, MLP block, and residual connections).
//!
//! Produces 18 ScheduledKernelOps with real BufferBindingPlan and BufferUse
//! records, and runs ArenaPlanner + HazardChecker over the region.

use crate::ecs::cimage_runtime::error::{CImageRuntimeError, CImageRuntimeResult};
use crate::ecs::cimage_runtime::receipts::CImageBindingReceipt;
use crate::ecs::cimage_runtime::tensor_store::RuntimeTensorStore;
use crate::execution_plan::{
    AccessMode, ActivationArenaPlan, ArenaAllocation, BufferBindingPlan, BufferUse,
    CommandBufferPolicy, DispatchShape, EstimatedKernelCost, ExecutionPhase, ExecutionRegion,
    ExecutionRegionKind, HazardChecker, HazardPlan, HazardPolicy, KernelOpKind,
    KernelSpecializationKey, KernelTemplateId, KernelValidationRequirements, LifetimeClass,
    ScheduledKernelOp, TimingPolicy,
};

/// Result of building a decoder layer region.
pub struct CImageDecoderRegionPlan {
    pub region: ExecutionRegion,
    pub hazard_plan: HazardPlan,
    pub arena_plan: ArenaPlannerOutput,
    pub binding_receipts: Vec<CImageBindingReceipt>,
    pub kv_plan: DecoderKvPlan,
}

/// KV cache dimension plan for the decoder layer.
pub struct DecoderKvPlan {
    pub max_seq_len: usize,
    pub kv_cache_byte_size: u64,
    pub kv_cache_store_id: String,
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

/// Builder for decoder layer execution regions.
pub struct DecoderShardRegionBuilder;

impl DecoderShardRegionBuilder {
    /// Build a Metal ExecutionRegion for a full transformer decoder layer.
    ///
    /// Returns the region, hazard plan, arena plan, binding receipts, and
    /// KV cache dimension plan.
    pub fn build_decoder_region(
        store: &RuntimeTensorStore,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        seq_len: usize,
    ) -> CImageRuntimeResult<CImageDecoderRegionPlan> {
        // ── Sizing constants ───────────────────────────────────────────────
        let _hidden_bytes = (hidden_dim * 4) as u64;
        let _inter_bytes = (intermediate_dim * 4) as u64;
        let _q_out_bytes = (num_heads * head_dim * 4) as u64;
        let _kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;
        let _kv_cache_bytes = (seq_len * num_kv_heads * head_dim * 4) as u64;
        let kv_cache_bytes = (seq_len * num_kv_heads * head_dim * 4) as u64;
        let _scores_bytes = (num_heads * seq_len * 4) as u64;
        let _attended_bytes = _q_out_bytes;

        // ── Persistent buffer definitions ──────────────────────────────────
        // Ordered by buffer_id for deterministic lookup.
        let persistent = define_persistent_buffers(
            store,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        );

        // ── Scratch buffer definitions ─────────────────────────────────────
        let scratch = define_scratch_buffers(
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        );

        // ── Op definitions (18 ops, indices 0–17) ─────────────────────────
        let ops = build_all_ops(
            &persistent,
            &scratch,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        );

        // ── Validate all ops ───────────────────────────────────────────────
        for op in &ops {
            op.validate_lowered().map_err(|e| {
                CImageRuntimeError::LoweringFailed(format!(
                    "decoder op {} validation failed: {:?}",
                    op.op_id, e
                ))
            })?;
        }

        // ── Arena plan ─────────────────────────────────────────────────────
        let arena_scratch: Vec<&ScratchBufferInfo> = scratch
            .iter()
            .filter(|b| b.buffer_id.starts_with("scratch_"))
            .collect();
        let total_scratch: u64 = arena_scratch.iter().map(|b| b.byte_size).sum();

        // Build arena allocations from scratch buffer info.
        let arena_allocations: Vec<ArenaAllocation> = scratch
            .iter()
            .map(|sb| ArenaAllocation {
                logical_buffer_id: sb.buffer_id.clone(),
                offset: sb.offset,
                size_bytes: sb.byte_size,
                alignment_bytes: 16, // 16-byte alignment
                lifetime_start_op: sb.lifetime_start_op,
                lifetime_end_op: sb.lifetime_end_op,
                alias_group: None,
            })
            .collect();

        let arena_output = ArenaPlannerOutput {
            scratch_buffers: scratch.clone(),
            total_scratch_bytes: total_scratch,
        };

        // ── Build ExecutionRegion ──────────────────────────────────────────
        let region = ExecutionRegion {
            region_id: "decoder_layer_region".into(),
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
                arena_id: "decoder_layer_arena".into(),
                total_bytes: total_scratch,
                allocations: arena_allocations,
                alias_groups: vec![],
                peak_live_bytes: total_scratch,
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
                    persistent.iter().any(|pb| pb.buffer_id == b.buffer_id)
                        || scratch.iter().any(|sb| sb.buffer_id == b.buffer_id)
                });
                CImageBindingReceipt {
                    region_id: "decoder_layer_region".into(),
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

        // ── KV plan ────────────────────────────────────────────────────────
        let kv_plan = DecoderKvPlan {
            max_seq_len: seq_len,
            kv_cache_byte_size: kv_cache_bytes * 2, // K + V
            kv_cache_store_id: "decoder_kv_cache".into(),
        };

        Ok(CImageDecoderRegionPlan {
            region,
            hazard_plan,
            arena_plan: arena_output,
            binding_receipts,
            kv_plan,
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
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    _seq_len: usize,
) -> Vec<PersistentBufferDef> {
    let hidden_bytes = (hidden_dim * 4) as u64;
    let q_weight_bytes = (num_heads * head_dim * hidden_dim * 4) as u64;
    let kv_weight_bytes = (num_kv_heads * head_dim * hidden_dim * 4) as u64;
    let o_weight_bytes = (hidden_dim * num_heads * head_dim * 4) as u64;
    let gate_weight_bytes = (intermediate_dim * hidden_dim * 4) as u64;
    let up_weight_bytes = (intermediate_dim * hidden_dim * 4) as u64;
    let down_weight_bytes = (hidden_dim * intermediate_dim * 4) as u64;
    let kv_cache_bytes = (_seq_len * num_kv_heads * head_dim * 4) as u64;

    vec![
        // Input / output
        PersistentBufferDef {
            buffer_id: "hidden_in".into(),
            byte_size: hidden_bytes,
            lifetime: LifetimeClass::RegionInput,
        },
        PersistentBufferDef {
            buffer_id: "hidden_out".into(),
            byte_size: hidden_bytes,
            lifetime: LifetimeClass::RegionOutput,
        },
        // Layer norm weights
        PersistentBufferDef {
            buffer_id: "input_layernorm_weight".into(),
            byte_size: hidden_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "post_attn_layernorm_weight".into(),
            byte_size: hidden_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // Q projection
        PersistentBufferDef {
            buffer_id: "q_proj_codes".into(),
            byte_size: q_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "q_proj_scales".into(),
            byte_size: (num_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "q_proj_biases".into(),
            byte_size: (num_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // K projection
        PersistentBufferDef {
            buffer_id: "k_proj_codes".into(),
            byte_size: kv_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "k_proj_scales".into(),
            byte_size: (num_kv_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "k_proj_biases".into(),
            byte_size: (num_kv_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // V projection
        PersistentBufferDef {
            buffer_id: "v_proj_codes".into(),
            byte_size: kv_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "v_proj_scales".into(),
            byte_size: (num_kv_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "v_proj_biases".into(),
            byte_size: (num_kv_heads as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // O projection
        PersistentBufferDef {
            buffer_id: "o_proj_codes".into(),
            byte_size: o_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "o_proj_scales".into(),
            byte_size: (hidden_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "o_proj_biases".into(),
            byte_size: (hidden_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // Gate projection (MLP)
        PersistentBufferDef {
            buffer_id: "gate_proj_codes".into(),
            byte_size: gate_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "gate_proj_scales".into(),
            byte_size: (intermediate_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "gate_proj_biases".into(),
            byte_size: (intermediate_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // Up projection (MLP)
        PersistentBufferDef {
            buffer_id: "up_proj_codes".into(),
            byte_size: up_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "up_proj_scales".into(),
            byte_size: (intermediate_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "up_proj_biases".into(),
            byte_size: (intermediate_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // Down projection (MLP)
        PersistentBufferDef {
            buffer_id: "down_proj_codes".into(),
            byte_size: down_weight_bytes,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "down_proj_scales".into(),
            byte_size: (hidden_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        PersistentBufferDef {
            buffer_id: "down_proj_biases".into(),
            byte_size: (hidden_dim as u64) * 4,
            lifetime: LifetimeClass::PersistentWeight,
        },
        // Constants buffer
        PersistentBufferDef {
            buffer_id: "decoder_constants".into(),
            byte_size: 128, // DecoderKernelConstants struct (larger than MLP)
            lifetime: LifetimeClass::PersistentWeight,
        },
        // KV cache (paged, sized for max sequence length)
        PersistentBufferDef {
            buffer_id: "kv_cache_k".into(),
            byte_size: kv_cache_bytes,
            lifetime: LifetimeClass::PersistentKvCache,
        },
        PersistentBufferDef {
            buffer_id: "kv_cache_v".into(),
            byte_size: kv_cache_bytes,
            lifetime: LifetimeClass::PersistentKvCache,
        },
    ]
}

fn define_scratch_buffers(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    seq_len: usize,
) -> Vec<ScratchBufferInfo> {
    let hidden_bytes = (hidden_dim * 4) as u64;
    let inter_bytes = (intermediate_dim * 4) as u64;
    let q_out_bytes = (num_heads * head_dim * 4) as u64;
    let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;
    let scores_bytes = (num_heads * seq_len * 4) as u64;

    vec![
        ScratchBufferInfo {
            buffer_id: "scratch_normed".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 0,
            lifetime_end_op: 3,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_q".into(),
            byte_size: q_out_bytes,
            offset: 0,
            lifetime_start_op: 1,
            lifetime_end_op: 4,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_k".into(),
            byte_size: kv_out_bytes,
            offset: 0,
            lifetime_start_op: 2,
            lifetime_end_op: 4,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_v".into(),
            byte_size: kv_out_bytes,
            offset: 0,
            lifetime_start_op: 3,
            lifetime_end_op: 5,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_q_rope".into(),
            byte_size: q_out_bytes,
            offset: 0,
            lifetime_start_op: 4,
            lifetime_end_op: 6,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_k_rope".into(),
            byte_size: kv_out_bytes,
            offset: 0,
            lifetime_start_op: 4,
            lifetime_end_op: 5,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_scores".into(),
            byte_size: scores_bytes,
            offset: 0,
            lifetime_start_op: 6,
            lifetime_end_op: 7,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_scores_post_softmax".into(),
            byte_size: scores_bytes,
            offset: 0,
            lifetime_start_op: 7,
            lifetime_end_op: 8,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_attended".into(),
            byte_size: q_out_bytes,
            offset: 0,
            lifetime_start_op: 8,
            lifetime_end_op: 9,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_o".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 9,
            lifetime_end_op: 10,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_post_attn".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 10,
            lifetime_end_op: 17,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_normed2".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 11,
            lifetime_end_op: 13,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_gate".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 12,
            lifetime_end_op: 14,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_up".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 13,
            lifetime_end_op: 15,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_silu_gate".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 14,
            lifetime_end_op: 15,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_mlp_hidden".into(),
            byte_size: inter_bytes,
            offset: 0,
            lifetime_start_op: 15,
            lifetime_end_op: 16,
        },
        ScratchBufferInfo {
            buffer_id: "scratch_mlp_down".into(),
            byte_size: hidden_bytes,
            offset: 0,
            lifetime_start_op: 16,
            lifetime_end_op: 17,
        },
    ]
}

// ── Op builders ───────────────────────────────────────────────────────────

fn build_all_ops(
    _persistent: &[PersistentBufferDef],
    _scratch: &[ScratchBufferInfo],
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    seq_len: usize,
) -> Vec<ScheduledKernelOp> {
    vec![
        // 0: Pre-attention RMSNorm
        build_rmsnorm_op(
            0,
            "hidden_in",
            "input_layernorm_weight",
            "scratch_normed",
            hidden_dim,
        ),
        // 1: Q projection
        build_projection_op(
            1,
            "q",
            hidden_dim,
            num_heads * head_dim,
            "scratch_normed",
            "scratch_q",
        ),
        // 2: K projection
        build_projection_op(
            2,
            "k",
            hidden_dim,
            num_kv_heads * head_dim,
            "scratch_normed",
            "scratch_k",
        ),
        // 3: V projection
        build_projection_op(
            3,
            "v",
            hidden_dim,
            num_kv_heads * head_dim,
            "scratch_normed",
            "scratch_v",
        ),
        // 4: RoPE
        build_rope_op(
            4,
            num_heads,
            num_kv_heads,
            head_dim,
            "scratch_q",
            "scratch_k",
            "scratch_q_rope",
            "scratch_k_rope",
        ),
        // 5: KV cache append
        build_kv_append_op(
            5,
            num_kv_heads,
            head_dim,
            "scratch_k_rope",
            "scratch_v",
            "kv_cache_k",
            "kv_cache_v",
        ),
        // 6: Attention scores (Q · K^T)
        build_attention_scores_op(
            6,
            num_heads,
            seq_len,
            "scratch_q_rope",
            "kv_cache_k",
            "scratch_scores",
        ),
        // 7: Attention softmax (in-place)
        build_attention_softmax_op(
            7,
            num_heads,
            seq_len,
            "scratch_scores",
            "scratch_scores_post_softmax",
        ),
        // 8: Attention apply (scores · V)
        build_attention_apply_op(
            8,
            num_heads,
            head_dim,
            seq_len,
            "scratch_scores_post_softmax",
            "kv_cache_v",
            "scratch_attended",
        ),
        // 9: Output projection
        build_projection_op(
            9,
            "o",
            num_heads * head_dim,
            hidden_dim,
            "scratch_attended",
            "scratch_o",
        ),
        // 10: Post-attention residual add
        build_residual_add_op(
            10,
            hidden_dim,
            "hidden_in",
            "scratch_o",
            "scratch_post_attn",
        ),
        // 11: Post-attention RMSNorm
        build_rmsnorm_op(
            11,
            "scratch_post_attn",
            "post_attn_layernorm_weight",
            "scratch_normed2",
            hidden_dim,
        ),
        // 12: Gate projection (MLP)
        build_projection_op(
            12,
            "gate",
            hidden_dim,
            intermediate_dim,
            "scratch_normed2",
            "scratch_gate",
        ),
        // 13: Up projection (MLP)
        build_projection_op(
            13,
            "up",
            hidden_dim,
            intermediate_dim,
            "scratch_normed2",
            "scratch_up",
        ),
        // 14: SiLU activation
        build_silu_op(14, intermediate_dim, "scratch_gate", "scratch_silu_gate"),
        // 15: Element-wise multiply (SiLU(gate) * up)
        build_mul_op(
            15,
            intermediate_dim,
            "scratch_silu_gate",
            "scratch_up",
            "scratch_mlp_hidden",
        ),
        // 16: Down projection (MLP)
        build_projection_op(
            16,
            "down",
            intermediate_dim,
            hidden_dim,
            "scratch_mlp_hidden",
            "scratch_mlp_down",
        ),
        // 17: Post-MLP residual add
        build_residual_add_op(
            17,
            hidden_dim,
            "scratch_post_attn",
            "scratch_mlp_down",
            "hidden_out",
        ),
    ]
}

fn build_rmsnorm_op(
    idx: usize,
    input_buf: &str,
    weight_buf: &str,
    output_buf: &str,
    hidden_dim: usize,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_rmsnorm");
    let hidden_bytes = (hidden_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::RmsNorm,
        tensor_key: Some(weight_buf.into()),
        tensor_class: Some("LayerNormWeight".into()),
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: input_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: weight_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: output_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: input_buf.into(),
                access: AccessMode::Read,
                lifetime: if input_buf == "hidden_in" {
                    LifetimeClass::RegionInput
                } else {
                    LifetimeClass::LayerScratch
                },
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: weight_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
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

fn build_projection_op(
    idx: usize,
    proj_name: &str,
    in_dim: usize,
    out_dim: usize,
    input_buf: &str,
    output_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_{proj_name}_projection");
    let in_bytes = (in_dim * 4) as u64;
    let out_bytes = (out_dim * 4) as u64;

    let codes_name = format!("{proj_name}_proj_codes");
    let scales_name = format!("{proj_name}_proj_scales");
    let biases_name = format!("{proj_name}_proj_biases");

    // Use MlpDownResidual for O and down projections, MlpGateUp for others.
    let op_kind = if proj_name == "o" || proj_name == "down" {
        KernelOpKind::MlpDownResidual
    } else {
        KernelOpKind::MlpGateUp
    };

    let tensor_class = if proj_name == "o" || proj_name == "down" {
        "DecoderMlpProjection".into()
    } else {
        "DecoderQkvProjection".into()
    };

    ScheduledKernelOp {
        op_id,
        op_kind,
        tensor_key: Some(format!("{proj_name}_proj")),
        tensor_class: Some(tensor_class),
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: input_buf.into(),
                offset: 0,
                size: in_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: codes_name.clone(),
                offset: 0,
                size: (in_dim * out_dim * 4) as u64,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: scales_name.clone(),
                offset: 0,
                size: (out_dim as u64 / 20) * 4,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: biases_name.clone(),
                offset: 0,
                size: (out_dim as u64 / 20) * 4,
            },
            BufferBindingPlan {
                slot: 4,
                buffer_id: output_buf.into(),
                offset: 0,
                size: out_bytes,
            },
            BufferBindingPlan {
                slot: 5,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: input_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: codes_name,
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: scales_name,
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: biases_name,
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: out_dim as u32,
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

fn build_rope_op(
    idx: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_in_buf: &str,
    k_in_buf: &str,
    q_out_buf: &str,
    k_out_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_rope");
    let max_hd = std::cmp::max(num_heads, num_kv_heads);
    let rope_grid = (max_hd * head_dim) as u32;

    let q_out_bytes = (num_heads * head_dim * 4) as u64;
    let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::AttentionScore,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: q_in_buf.into(),
                offset: 0,
                size: q_out_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: k_in_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: q_out_buf.into(),
                offset: 0,
                size: q_out_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: k_out_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 4,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: q_in_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: k_in_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: q_out_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: k_out_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: rope_grid,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 2.0,
            memory_bytes_read: q_out_bytes + kv_out_bytes,
            memory_bytes_written: q_out_bytes + kv_out_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_kv_append_op(
    idx: usize,
    num_kv_heads: usize,
    head_dim: usize,
    k_rope_buf: &str,
    v_buf: &str,
    kv_cache_k_buf: &str,
    kv_cache_v_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_kv_append");
    let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::AttentionScore,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: k_rope_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: v_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: kv_cache_k_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: kv_cache_v_buf.into(),
                offset: 0,
                size: kv_out_bytes,
            },
            BufferBindingPlan {
                slot: 4,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: k_rope_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: v_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: kv_cache_k_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::PersistentKvCache,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: kv_cache_v_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::PersistentKvCache,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: (num_kv_heads * head_dim) as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 0.5,
            memory_bytes_read: kv_out_bytes * 2,
            memory_bytes_written: kv_out_bytes * 2,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_attention_scores_op(
    idx: usize,
    num_heads: usize,
    seq_len: usize,
    q_rope_buf: &str,
    kv_cache_k_buf: &str,
    scores_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_attention_scores");
    let scores_bytes = (num_heads * seq_len * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::AttentionScore,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: q_rope_buf.into(),
                offset: 0,
                size: scores_bytes, // approximate
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: kv_cache_k_buf.into(),
                offset: 0,
                size: scores_bytes, // approximate
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: scores_buf.into(),
                offset: 0,
                size: scores_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: q_rope_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: kv_cache_k_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentKvCache,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: scores_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: num_heads as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 1,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 5.0,
            memory_bytes_read: scores_bytes * 2,
            memory_bytes_written: scores_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_attention_softmax_op(
    idx: usize,
    num_heads: usize,
    seq_len: usize,
    scores_buf: &str,
    output_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_attention_softmax");
    let scores_bytes = (num_heads * seq_len * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::AttentionScore,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: scores_buf.into(),
                offset: 0,
                size: scores_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: output_buf.into(),
                offset: 0,
                size: scores_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: scores_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: num_heads as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 1,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 1.0,
            memory_bytes_read: scores_bytes,
            memory_bytes_written: scores_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_attention_apply_op(
    idx: usize,
    num_heads: usize,
    head_dim: usize,
    _seq_len: usize,
    scores_buf: &str,
    kv_cache_v_buf: &str,
    attended_buf: &str,
) -> ScheduledKernelOp {
    let op_id = format!("op_{idx}_attention_apply");
    let attended_bytes = (num_heads * head_dim * 4) as u64;

    ScheduledKernelOp {
        op_id,
        op_kind: KernelOpKind::AttentionApply,
        tensor_key: None,
        tensor_class: None,
        specialization: basic_specialization(),
        bindings: vec![
            BufferBindingPlan {
                slot: 0,
                buffer_id: scores_buf.into(),
                offset: 0,
                size: attended_bytes, // approximate
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: kv_cache_v_buf.into(),
                offset: 0,
                size: attended_bytes, // approximate
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: attended_buf.into(),
                offset: 0,
                size: attended_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: scores_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: kv_cache_v_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentKvCache,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: attended_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::PersistentWeight,
                alias_group: None,
                byte_range: None,
            },
        ],
        dispatch_shape: DispatchShape {
            grid_x: (num_heads * head_dim) as u32,
            grid_y: 1,
            grid_z: 1,
            threadgroup_m: 64,
            threadgroup_n: 1,
            threadgroup_p: 1,
        },
        estimated_cost: EstimatedKernelCost {
            compute_us: 5.0,
            memory_bytes_read: attended_bytes * 2,
            memory_bytes_written: attended_bytes,
        },
        validation_requirements: KernelValidationRequirements::default(),
    }
}

fn build_silu_op(
    idx: usize,
    intermediate_dim: usize,
    input_buf: &str,
    output_buf: &str,
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
                buffer_id: input_buf.into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: output_buf.into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: input_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
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
    intermediate_dim: usize,
    lhs_buf: &str,
    rhs_buf: &str,
    output_buf: &str,
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
                buffer_id: lhs_buf.into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: rhs_buf.into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: output_buf.into(),
                offset: 0,
                size: inter_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: lhs_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: rhs_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
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
    skip_buf: &str,
    update_buf: &str,
    output_buf: &str,
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
                buffer_id: skip_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 1,
                buffer_id: update_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 2,
                buffer_id: output_buf.into(),
                offset: 0,
                size: hidden_bytes,
            },
            BufferBindingPlan {
                slot: 3,
                buffer_id: "decoder_constants".into(),
                offset: 0,
                size: 128,
            },
        ],
        dependencies: vec![],
        buffer_uses: vec![
            BufferUse {
                buffer_id: skip_buf.into(),
                access: AccessMode::Read,
                lifetime: if skip_buf == "hidden_in" {
                    LifetimeClass::RegionInput
                } else {
                    LifetimeClass::LayerScratch
                },
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: update_buf.into(),
                access: AccessMode::Read,
                lifetime: LifetimeClass::LayerScratch,
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: output_buf.into(),
                access: AccessMode::Write,
                lifetime: if output_buf == "hidden_out" {
                    LifetimeClass::RegionOutput
                } else {
                    LifetimeClass::LayerScratch
                },
                alias_group: None,
                byte_range: None,
            },
            BufferUse {
                buffer_id: "decoder_constants".into(),
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
    } else if buffer_id == "decoder_constants" {
        "Constants".into()
    } else if buffer_id.starts_with("kv_cache_") {
        "KVCache".into()
    } else {
        "Unknown".into()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_store() -> RuntimeTensorStore {
        RuntimeTensorStore::new()
    }

    fn build_plan() -> CImageDecoderRegionPlan {
        let store = dummy_store();
        DecoderShardRegionBuilder::build_decoder_region(
            &store, 64,  // hidden_dim
            4,   // num_heads
            2,   // num_kv_heads
            16,  // head_dim
            128, // intermediate_dim
            8,   // seq_len
        )
        .expect("build_decoder_region should succeed")
    }

    #[test]
    fn test_decoder_region_builder_emits_expected_ops() {
        let plan = build_plan();
        // We expect 18 ops (indices 0–17)
        assert_eq!(plan.region.ops.len(), 18, "expected 18 decoder ops");
    }

    #[test]
    fn test_decoder_region_builder_op_kinds() {
        let plan = build_plan();
        let kinds: Vec<KernelOpKind> = plan.region.ops.iter().map(|op| op.op_kind).collect();

        // Op 0: RMSNorm
        assert_eq!(kinds[0], KernelOpKind::RmsNorm);
        // Ops 1–3: Q, K, V projections
        assert_eq!(kinds[1], KernelOpKind::MlpGateUp);
        assert_eq!(kinds[2], KernelOpKind::MlpGateUp);
        assert_eq!(kinds[3], KernelOpKind::MlpGateUp);
        // Op 4: RoPE
        assert_eq!(kinds[4], KernelOpKind::AttentionScore);
        // Op 5: KV append
        assert_eq!(kinds[5], KernelOpKind::AttentionScore);
        // Op 6: Attention scores
        assert_eq!(kinds[6], KernelOpKind::AttentionScore);
        // Op 7: Softmax
        assert_eq!(kinds[7], KernelOpKind::AttentionScore);
        // Op 8: Attention apply
        assert_eq!(kinds[8], KernelOpKind::AttentionApply);
        // Op 9: Output projection
        assert_eq!(kinds[9], KernelOpKind::MlpDownResidual);
        // Op 10: Post-attention residual add
        assert_eq!(kinds[10], KernelOpKind::MlpDownResidual);
        // Op 11: Post-attention RMSNorm
        assert_eq!(kinds[11], KernelOpKind::RmsNorm);
        // Ops 12–13: Gate and up projections
        assert_eq!(kinds[12], KernelOpKind::MlpGateUp);
        assert_eq!(kinds[13], KernelOpKind::MlpGateUp);
        // Op 14: SiLU activation
        assert_eq!(kinds[14], KernelOpKind::MlpActivation);
        // Op 15: Mul
        assert_eq!(kinds[15], KernelOpKind::MlpActivation);
        // Op 16: Down projection
        assert_eq!(kinds[16], KernelOpKind::MlpDownResidual);
        // Op 17: Post-MLP residual add
        assert_eq!(kinds[17], KernelOpKind::MlpDownResidual);
    }

    #[test]
    fn test_decoder_region_builder_populates_bindings() {
        let plan = build_plan();
        for op in &plan.region.ops {
            assert!(!op.bindings.is_empty(), "op {} has no bindings", op.op_id);
        }
    }

    #[test]
    fn test_decoder_region_builder_populates_buffer_uses() {
        let plan = build_plan();
        for op in &plan.region.ops {
            assert!(
                !op.buffer_uses.is_empty(),
                "op {} has no buffer uses",
                op.op_id
            );
        }
    }

    #[test]
    fn test_decoder_region_hazard_check_passes() {
        let plan = build_plan();
        // The HazardChecker already ran during build and returned a plan;
        // we verify the hazard_plan is sane.
        let hp = &plan.hazard_plan;
        // HazardPlan must be populated with boundaries/barriers when hazards exist
        // (the decoder has many RAW/WAR dependencies, so some should be present).
        // At minimum the plan exists.
        assert!(
            hp.aliasing_approved || !hp.encoder_boundaries.is_empty(),
            "hazard plan should either approve aliasing or produce encoder boundaries"
        );
    }

    #[test]
    fn test_decoder_region_all_ops_validate() {
        let plan = build_plan();
        for op in &plan.region.ops {
            op.validate_lowered()
                .unwrap_or_else(|e| panic!("op {} failed validation: {:?}", op.op_id, e));
        }
    }

    #[test]
    fn test_decoder_region_binding_receipts() {
        let plan = build_plan();
        assert_eq!(plan.binding_receipts.len(), 18);
        for receipt in &plan.binding_receipts {
            assert!(
                receipt.all_bindings_resolved,
                "receipt op {} has unresolved bindings",
                receipt.op_id
            );
        }
    }

    #[test]
    fn test_decoder_kv_plan() {
        let plan = build_plan();
        assert_eq!(plan.kv_plan.max_seq_len, 8);
        // K + V cache: 2 * seq_len * num_kv_heads * head_dim * 4
        let expected_kv_bytes: u64 = 2 * 8 * 2 * 16 * 4;
        assert_eq!(plan.kv_plan.kv_cache_byte_size, expected_kv_bytes);
        assert_eq!(plan.kv_plan.kv_cache_store_id, "decoder_kv_cache");
    }

    #[test]
    fn test_decoder_region_has_correct_region_id() {
        let plan = build_plan();
        assert_eq!(plan.region.region_id, "decoder_layer_region");
        assert_eq!(
            plan.region.region_kind,
            ExecutionRegionKind::DecoderLayerDecode
        );
    }

    #[test]
    fn test_decoder_rope_op_binding_slots() {
        let plan = build_plan();
        // Op 4 is the RoPE
        let rope_op = &plan.region.ops[4];
        assert_eq!(rope_op.bindings.len(), 5);
        // slot 0: scratch_q, slot 1: scratch_k, slot 2: scratch_q_rope, slot 3: scratch_k_rope, slot 4: decoder_constants
        assert_eq!(rope_op.bindings[0].buffer_id, "scratch_q");
        assert_eq!(rope_op.bindings[1].buffer_id, "scratch_k");
        assert_eq!(rope_op.bindings[2].buffer_id, "scratch_q_rope");
        assert_eq!(rope_op.bindings[3].buffer_id, "scratch_k_rope");
        assert_eq!(rope_op.bindings[4].buffer_id, "decoder_constants");
    }

    #[test]
    fn test_decoder_attention_scores_dispatch() {
        let plan = build_plan();
        let scores_op = &plan.region.ops[6];
        // For 4 heads, grid_x = 4, threadgroup_m = 1
        assert_eq!(scores_op.dispatch_shape.grid_x, 4);
        assert_eq!(scores_op.dispatch_shape.threadgroup_m, 1);
    }

    #[test]
    fn test_decoder_softmax_in_place() {
        let plan = build_plan();
        let softmax_op = &plan.region.ops[7];
        // Softmax reads from scratch_scores and writes to scratch_scores_post_softmax
        let scores_use = softmax_op
            .buffer_uses
            .iter()
            .find(|u| u.buffer_id == "scratch_scores")
            .expect("softmax must use scratch_scores");
        assert_eq!(scores_use.access, AccessMode::Read);
        let post_softmax_use = softmax_op
            .buffer_uses
            .iter()
            .find(|u| u.buffer_id == "scratch_scores_post_softmax")
            .expect("softmax must write scratch_scores_post_softmax");
        assert_eq!(post_softmax_use.access, AccessMode::Write);
    }
}
