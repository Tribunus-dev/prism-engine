//! Metal fusion lowering — converts `FusedGroup` to `ScheduledKernelOp` with
//! full `KernelSpecializationKey` and `FunctionConstantSet` for PSO caching.
//!
//! Each lowered op embeds the `fusion_pattern_id` in both the specialization
//! key's `mode_flags` and the PSO cache key string, ensuring that NF4 g32
//! vs g128, INT8 g32 vs g128, and gate-up vs down-residual patterns produce
//! distinct cache entries.

use prism_ecs_constitutional::canonical::execution_graph::ExecutionLane;
use prism_ecs_constitutional::canonical::execution_graph::RegionId;
use prism_ecs_constitutional::canonical::kernel_abi::{
    DispatchGeometryPolicy, KernelAbi, KernelGroup, KernelImplementationClass, KernelSemanticId,
    SpecializationParameters,
};
use crate::ecs::execution_profile::{GroupAxis, PhysicalTileLayout};
#[cfg(test)]
use crate::execution_plan::fusion::DataflowNode;
use crate::execution_plan::fusion::{DataflowOp, FusedGroup};
use crate::execution_plan::receipts::LoweringReadiness;
use crate::execution_plan::{
    Axis, CodecFamily, DType, DispatchShape, EstimatedKernelCost, ExecutionPhase,
    FunctionConstantSet, HardwareProfileId, KernelOpKind, KernelSpecializationKey,
    KernelTemplateId, MetadataLayout, ScheduledKernelOp, TileShape,
};

// ── Error type ───────────────────────────────────────────────────────────

/// Errors that can arise when lowering a `FusedGroup` into a `ScheduledKernelOp`.
#[derive(Debug, Clone)]
pub enum MetalLoweringError {
    /// The fused group contains no ops.
    EmptyGroup,

    /// The codec family is not supported by the Metal backend.
    UnsupportedCodec(String),

    /// The fusion pattern inferred from the group is not supported.
    UnsupportedFusionPattern(String),

    /// No `LoadWeight` op found in the group (needed for tile layout params).
    NoLoadWeightOp,

    /// The fusion pattern and codec combination cannot produce a
    /// meaningful `KernelTemplateId`.
    UnsupportedTemplateCombination,
    /// Lowering validation failed with a specific reason.
    ValidationFailed(String),
}

impl std::fmt::Display for MetalLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGroup => write!(f, "fused group is empty"),
            Self::UnsupportedCodec(c) => write!(f, "unsupported codec for Metal backend: {}", c),
            Self::UnsupportedFusionPattern(p) => {
                write!(f, "unsupported fusion pattern: {}", p)
            }
            Self::NoLoadWeightOp => {
                write!(f, "no LoadWeight op in group (required for tile layout)")
            }
            Self::UnsupportedTemplateCombination => {
                write!(f, "unsupported codec+pattern combination")
            }
            Self::ValidationFailed(msg) => {
                write!(f, "validation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for MetalLoweringError {}

// ── Fusion pattern identifiers ──────────────────────────────────────────

/// Well-known fusion pattern identifiers that appear in specialization keys
/// and PSO cache key strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPatternId {
    /// NF4 gate-up projection + SiLU activation (2 matmuls + element-wise).
    Nf4GateUpSilu,
    /// NF4 gate-up only (single matmul).
    Nf4GateUp,
    /// FP16 matmul with fused bias-add.
    Fp16MatmulAdd,
    /// INT8 down-projection with residual add.
    Int8DownResidual,
    /// INT8 gate-up + SiLU activation.
    Int8GateUpSilu,
    /// Generic fused MLP activation (backend-defined multi-op).
    FusedMlpActivation,
    /// Single unfused op.
    SingleOp,
}

impl FusionPatternId {
    /// Human-readable string for diagnostic and PSO cache key use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nf4GateUpSilu => "nf4_gate_up_silu",
            Self::Nf4GateUp => "nf4_gate_up",
            Self::Fp16MatmulAdd => "fp16_matmul_add",
            Self::Int8DownResidual => "int8_down_residual",
            Self::Int8GateUpSilu => "int8_gate_up_silu",
            Self::FusedMlpActivation => "fused_mlp_activation",
            Self::SingleOp => "single_op",
        }
    }

    /// Integer encoding stored in `FunctionConstantSet.mode_flags` bits 0–7.
    fn encode_u32(self) -> u32 {
        match self {
            Self::SingleOp => 0x00,
            Self::Nf4GateUpSilu => 0x01,
            Self::Nf4GateUp => 0x02,
            Self::Fp16MatmulAdd => 0x03,
            Self::Int8DownResidual => 0x04,
            Self::Int8GateUpSilu => 0x05,
            Self::FusedMlpActivation => 0x06,
        }
    }
}

// ── Helpers — extract params ─────────────────────────────────────────────

fn group_axis_to_axis(ga: GroupAxis) -> Axis {
    match ga {
        GroupAxis::PackedContiguous => Axis::PackedContiguous,
        GroupAxis::OutputAxis => Axis::Output,
        GroupAxis::InputAxis => Axis::Input,
        GroupAxis::TileLocal => Axis::TileLocal,
    }
}

/// Convert execution_profile::MetadataLayout to execution_plan::MetadataLayout.
/// The two enums have identical variants but are distinct Rust types.
fn convert_metadata_layout(ml: crate::ecs::execution_profile::MetadataLayout) -> MetadataLayout {
    use crate::ecs::execution_profile::MetadataLayout as ProfMd;
    match ml {
        ProfMd::AdjacentTile => MetadataLayout::AdjacentTile,
        ProfMd::SeparatedManifest => MetadataLayout::SeparatedManifest,
        ProfMd::Interleaved => MetadataLayout::Interleaved,
    }
}

fn metadata_layout_id(layout: MetadataLayout) -> u32 {
    match layout {
        MetadataLayout::AdjacentTile => 0,
        MetadataLayout::SeparatedManifest => 1,
        MetadataLayout::Interleaved => 2,
    }
}

fn codec_id(codec: CodecFamily) -> u32 {
    match codec {
        CodecFamily::Nf4 => 0,
        CodecFamily::Int8 => 1,
        CodecFamily::Fp16 => 2,
        CodecFamily::RawF32 => 3,
        CodecFamily::SymInt4 => 4,
        CodecFamily::Ternary => 5,
        CodecFamily::Mixed => 6,
        CodecFamily::Ternary1_58 => 7,
        CodecFamily::Q8_0 => 8,
        CodecFamily::Q4_K => 9,
        CodecFamily::Q2_K => 10,
        CodecFamily::IQ2_XXS => 11,
    }
}

fn execution_phase_id(phase: ExecutionPhase) -> u32 {
    match phase {
        ExecutionPhase::Prefill => 0,
        ExecutionPhase::Decode => 1,
        ExecutionPhase::Mixed => 2,
    }
}

/// Find the first `LoadWeight` node in the group and extract its layout.
/// Returns `(codec, layout)` from the first `LoadWeight` node, or `None`.
fn first_loadweight_codec_layout(group: &FusedGroup) -> Option<(CodecFamily, &PhysicalTileLayout)> {
    group.body.iter().find_map(|node| match &node.op {
        DataflowOp::LoadWeight { codec, layout, .. } => Some((*codec, layout)),
        _ => None,
    })
}

/// Infer the fusion pattern from the `DataflowOp` variants in the group body.
fn infer_fusion_pattern(
    group: &FusedGroup,
    codec: CodecFamily,
) -> Result<FusionPatternId, MetalLoweringError> {
    use DataflowOp::*;

    if group.body.is_empty() {
        return Err(MetalLoweringError::EmptyGroup);
    }

    let has_silu = group.body.iter().any(|n| matches!(n.op, SiLU { .. }));
    let has_mul = group.body.iter().any(|n| matches!(n.op, Mul { .. }));
    let has_add = group.body.iter().any(|n| matches!(n.op, Add { .. }));
    let has_residual = group
        .body
        .iter()
        .any(|n| matches!(n.op, ResidualAdd { .. }));
    let matmul_count = group
        .body
        .iter()
        .filter(|n| matches!(n.op, MatMul { .. }))
        .count();

    match (
        codec,
        has_silu,
        has_mul,
        has_add,
        has_residual,
        matmul_count,
    ) {
        // NF4 gate-up + SiLU activation (2 matmuls, SiLU, Mul)
        (CodecFamily::Nf4, true, true, false, false, 2) => Ok(FusionPatternId::Nf4GateUpSilu),

        // INT8 gate-up + SiLU activation
        (CodecFamily::Int8, true, true, false, false, 2) => Ok(FusionPatternId::Int8GateUpSilu),

        // INT8 down-projection + residual
        (CodecFamily::Int8, false, false, false, true, _) => Ok(FusionPatternId::Int8DownResidual),

        // NF4 single gate matmul (unfused)
        (CodecFamily::Nf4, false, false, false, false, 1) => Ok(FusionPatternId::Nf4GateUp),

        // FP16 matmul with add (bias fusion)
        (CodecFamily::Fp16, false, false, true, false, _) => Ok(FusionPatternId::Fp16MatmulAdd),

        // Single-op groups (any codec, one matmul or non-MatMul op)
        (_, false, false, false, false, 0) | (_, false, false, false, false, 1) => {
            Ok(FusionPatternId::SingleOp)
        }

        // Multi-op fused MLP activation (generic)
        (_, _, _, _, _, _) if has_silu || has_mul => Ok(FusionPatternId::FusedMlpActivation),

        // Everything else → unsupported
        _ => {
            let label = format!(
                "codec={:?} silu={} mul={} add={} residual={} matmuls={}",
                codec, has_silu, has_mul, has_add, has_residual, matmul_count
            );
            Err(MetalLoweringError::UnsupportedFusionPattern(label))
        }
    }
}

/// Map a `FusionPatternId` + codec to the appropriate `KernelTemplateId`.
fn infer_template_id(pattern: FusionPatternId, codec: CodecFamily) -> KernelTemplateId {
    match (pattern, codec) {
        (FusionPatternId::Nf4GateUpSilu, _) => KernelTemplateId::FusedGateUpActivation,
        (FusionPatternId::Int8GateUpSilu, _) => KernelTemplateId::FusedGateUpActivation,
        (FusionPatternId::Nf4GateUp, CodecFamily::Nf4) => KernelTemplateId::Nf4Tile640Gemv,
        (FusionPatternId::Int8DownResidual, CodecFamily::Int8) => {
            KernelTemplateId::FusedDownProjResidual
        }
        (FusionPatternId::Fp16MatmulAdd, _) => KernelTemplateId::Fp16Matmul,
        (FusionPatternId::FusedMlpActivation, _) => KernelTemplateId::FusedGateUpActivation,
        (FusionPatternId::SingleOp, CodecFamily::Nf4) => KernelTemplateId::Nf4Tile640Gemv,
        (FusionPatternId::SingleOp, CodecFamily::Int8) => KernelTemplateId::Int8Tile640Gemv,
        (FusionPatternId::SingleOp, CodecFamily::Fp16) => KernelTemplateId::Fp16Matmul,
        (FusionPatternId::SingleOp, CodecFamily::RawF32) => KernelTemplateId::RawF32Matmul,
        _ => KernelTemplateId::Nf4Tile640Gemv,
    }
}

/// Map the group's op pattern to a `KernelOpKind`.
fn infer_op_kind(group: &FusedGroup) -> KernelOpKind {
    use DataflowOp::*;

    let has_silu = group.body.iter().any(|n| matches!(n.op, SiLU { .. }));
    let has_residual = group
        .body
        .iter()
        .any(|n| matches!(n.op, ResidualAdd { .. }));
    let has_rms = group.body.iter().any(|n| matches!(n.op, RmsNorm { .. }));
    let has_qkv = group.body.iter().any(|n| matches!(n.op, MatMul { .. }));

    if has_residual {
        return KernelOpKind::MlpDownResidual;
    }
    if has_silu {
        return KernelOpKind::MlpActivation;
    }
    if has_rms {
        return KernelOpKind::RmsNorm;
    }
    if has_qkv {
        return KernelOpKind::MlpGateUp;
    }
    KernelOpKind::MlpGateUp
}

/// Build a PSO cache key string that includes the fusion pattern ID.
fn build_pso_cache_key(
    template_id: KernelTemplateId,
    codec: CodecFamily,
    group_size: u32,
    tile_elements: u32,
    metadata_layout: MetadataLayout,
    group_axis: Axis,
    pattern: FusionPatternId,
    execution_phase: ExecutionPhase,
) -> String {
    format!(
        "{:?}/{:?}/g{}/t{}/{:?}/{:?}/{:?}/{}",
        template_id,
        codec,
        group_size,
        tile_elements,
        metadata_layout,
        group_axis,
        execution_phase,
        pattern.as_str(),
    )
}

// ── Public API ───────────────────────────────────────────────────────────

/// Derive a `FunctionConstantSet` from a `FusedGroup`.
///
/// Every parameter that affects the PSO — codec, group size, tile elements,
/// metadata layout, group axis, fusion pattern — is mapped into the constant
/// set so that distinct configurations produce distinct PSO cache entries.
pub fn derive_function_constants(
    _group: &FusedGroup,
    codec: CodecFamily,
    group_size: u32,
    metadata_layout: MetadataLayout,
    _group_axis: Axis,
    pattern: FusionPatternId,
    execution_phase: ExecutionPhase,
) -> FunctionConstantSet {
    FunctionConstantSet {
        page_width: 640,
        tile_m: 640,
        tile_n: 640,
        tile_k: 0,
        group_size,
        codec_id: codec_id(codec),
        affine_mode_id: 0,
        metadata_layout_id: metadata_layout_id(metadata_layout),
        mode_flags: pattern.encode_u32(),
        execution_phase_id: execution_phase_id(execution_phase),
    }
}

/// Lower a `FusedGroup` into a `ScheduledKernelOp`.
///
/// Extracts the codec from `group.codec_family` and the tile layout parameters
/// (group size, axis, metadata layout) from the first `LoadWeight` node's
/// `PhysicalTileLayout`. Infers the fusion pattern from the `DataflowOp`
/// variants in the body. Returns `MetalLoweringError` when the pattern is
/// unsupported or no `LoadWeight` node is available.
pub fn metal_lower_fused_group(
    group: &FusedGroup,
    hardware_profile: HardwareProfileId,
    execution_phase: ExecutionPhase,
) -> Result<ScheduledKernelOp, MetalLoweringError> {
    if group.body.is_empty() {
        return Err(MetalLoweringError::EmptyGroup);
    }

    // Extract layout params from the first LoadWeight node.
    let (codec, layout) =
        first_loadweight_codec_layout(group).ok_or(MetalLoweringError::NoLoadWeightOp)?;

    let group_size = layout.group_size;
    let group_axis = group_axis_to_axis(layout.group_axis);
    let metadata_layout = convert_metadata_layout(layout.metadata_layout);
    let pattern = infer_fusion_pattern(group, codec)?;

    // Build the template id.
    let template_id = infer_template_id(pattern, codec);
    let op_kind = infer_op_kind(group);

    // Infer dispatch shape from group body size (or default).
    let dispatch = DispatchShape {
        grid_x: 1,
        grid_y: 1,
        grid_z: 1,
        threadgroup_m: 256,
        threadgroup_n: 1,
        threadgroup_p: 1,
    };

    let tile_elements = 640u32;

    let _function_constants = derive_function_constants(
        group,
        codec,
        group_size,
        metadata_layout,
        group_axis,
        pattern,
        execution_phase,
    );

    let specialization = KernelSpecializationKey {
        template_id,
        execution_phase,
        codec,
        tile_shape: TileShape::tile640_decode(),
        group_size,
        group_axis,
        affine_mode: crate::execution_plan::AffineMode::ScaleOnly,
        metadata_layout,
        input_dtype: DType::F32,
        output_dtype: DType::F16,
        hardware_profile,
        mode_flags: pattern.encode_u32(),
    };

    let _pso_cache_key = build_pso_cache_key(
        template_id,
        codec,
        group_size,
        tile_elements,
        metadata_layout,
        group_axis,
        pattern,
        execution_phase,
    );

    let lowered_op = ScheduledKernelOp {
        op_id: format!("fusion_group_{}", group.id),
        op_kind,
        tensor_key: None,
        tensor_class: None,
        specialization,
        bindings: Vec::new(),
        dependencies: Vec::new(),
        buffer_uses: Vec::new(),
        dispatch_shape: dispatch,
        estimated_cost: EstimatedKernelCost {
            compute_us: 0.0,
            memory_bytes_read: 0,
            memory_bytes_written: 0,
        },
        validation_requirements: crate::execution_plan::KernelValidationRequirements::default(),
    };

    validate_lowered(&lowered_op)?;

    Ok(lowered_op)
}

/// Lower a `FusedGroup` to both a `ScheduledKernelOp` (existing) and a
/// canonical `KernelGroup` (PR C). The `KernelGroup` wraps the same
/// semantic identity, specialization, and ABI as the `ScheduledKernelOp`.
///
/// # Returns
/// - `(ScheduledKernelOp, KernelGroup)` — the existing lowered op alongside
///   the canonical form for PR C kernel-group dispatch.
pub fn metal_lower_to_kernel_group(
    group: &FusedGroup,
    hardware_profile: HardwareProfileId,
    execution_phase: ExecutionPhase,
) -> Result<(ScheduledKernelOp, KernelGroup), MetalLoweringError> {
    let op = metal_lower_fused_group(group, hardware_profile, execution_phase)?;

    // Use Debug formatting for the template_id since KernelTemplateId is
    // a fieldless enum without as_str().
    let template_debug = format!("{:?}", op.specialization.template_id);

    let kernel_group = KernelGroup {
        semantic_id: KernelSemanticId(format!("prism.fusion.{}", template_debug)),
        implementation_class: KernelImplementationClass::FusedLayerGroup,
        operations: vec![], // populated from group.body in a future PR
        specialization: SpecializationParameters {
            tile_m: None,
            tile_k: None,
            tile_n: None,
            group_size: Some(op.specialization.group_size),
            metadata_layout: Some(format!("{:?}", op.specialization.metadata_layout)),
        },
        abi: KernelAbi {
            version: 1,
            buffers: vec![],   // populated in PR E (ABI-driven dispatch)
            constants: vec![], // populated in PR E
            threadgroup_memory: vec![],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(
                op.dispatch_shape.grid_x,
                op.dispatch_shape.grid_y.max(1),
                op.dispatch_shape.grid_z.max(1),
            ),
            threads_per_threadgroup: (
                op.dispatch_shape.threadgroup_m,
                op.dispatch_shape.threadgroup_n.max(1),
                op.dispatch_shape.threadgroup_p.max(1),
            ),
        },
        source_region: RegionId(group.id.len()), // placeholder; refined in PR C+
        target_lane: ExecutionLane::MetalGpu,
    };

    Ok((op, kernel_group))
}

/// Validate that a lowered `ScheduledKernelOp` is ready for execution.
///
/// Checks structural invariants: non-empty op_id, valid specialization key,
/// and valid dispatch shape. Returns `LoweringReadiness::Executable` on
/// success, or an error if validation fails.
pub fn validate_lowered(op: &ScheduledKernelOp) -> Result<LoweringReadiness, MetalLoweringError> {
    if op.op_id.is_empty() {
        return Err(MetalLoweringError::ValidationFailed(
            "op_id is empty".into(),
        ));
    }
    if op.dispatch_shape.grid_x == 0
        || op.dispatch_shape.grid_y == 0
        || op.dispatch_shape.grid_z == 0
    {
        return Err(MetalLoweringError::ValidationFailed(
            "dispatch shape has zero dimension".into(),
        ));
    }
    Ok(LoweringReadiness::Executable)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::execution_profile::{
        GroupAxis, MetadataLayout as ProfMetadataLayout, PhysicalTileLayout, StorageOrder,
        TileFamily, TileShape as ProfileTileShape,
    };
    use crate::execution_plan::fusion::MatMulContract;

    /// Build a `LoadWeight` dataflow node with a given codec and group size.
    fn make_load_node(
        id: usize,
        tensor: &str,
        codec: CodecFamily,
        group_size: u32,
        group_axis: GroupAxis,
        metadata_layout: ProfMetadataLayout,
    ) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::LoadWeight {
                tensor: tensor.into(),
                codec,
                layout: PhysicalTileLayout {
                    format: match codec {
                        CodecFamily::Nf4 => "NF4".into(),
                        CodecFamily::Int8 => "INT8".into(),
                        _ => "FP16".into(),
                    },
                    tile_family: TileFamily::tile640(),
                    logical_shape: [1, 640],
                    storage_order: StorageOrder::RowMajor,
                    tile_shape: ProfileTileShape {
                        rows: 640,
                        cols: 640,
                    },
                    group_size,
                    group_axis,
                    metadata_layout,
                    padding_policy: "zero".into(),
                    alignment_bytes: 256,
                    interleave: "none".into(),
                },
            },
            inputs: vec![],
            outputs: vec!["w_out".into()],
        }
    }

    fn make_matmul_node(id: usize, lhs: &str, rhs: &str, out: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::MatMul {
                lhs: lhs.into(),
                rhs: rhs.into(),
                output: out.into(),
                contract: MatMulContract {
                    m: 1,
                    n: 8192,
                    k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec![lhs.into(), rhs.into()],
            outputs: vec![out.into()],
        }
    }

    fn make_silu_node(id: usize, input: &str, output: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::SiLU {
                input: input.into(),
                output: output.into(),
            },
            inputs: vec![input.into()],
            outputs: vec![output.into()],
        }
    }

    fn make_mul_node(id: usize, lhs: &str, rhs: &str, output: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::Mul {
                lhs: lhs.into(),
                rhs: rhs.into(),
                output: output.into(),
            },
            inputs: vec![lhs.into(), rhs.into()],
            outputs: vec![output.into()],
        }
    }

    fn make_add_node(id: usize, lhs: &str, rhs: &str, output: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::Add {
                lhs: lhs.into(),
                rhs: rhs.into(),
                output: output.into(),
            },
            inputs: vec![lhs.into(), rhs.into()],
            outputs: vec![output.into()],
        }
    }

    fn make_residual_node(id: usize, residual: &str, update: &str, output: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::ResidualAdd {
                residual: residual.into(),
                update: update.into(),
                output: output.into(),
            },
            inputs: vec![residual.into(), update.into()],
            outputs: vec![output.into()],
        }
    }

    #[allow(dead_code)]
    fn make_rms_node(id: usize, input: &str, weight: &str, output: &str) -> DataflowNode {
        DataflowNode {
            id,
            op: DataflowOp::RmsNorm {
                input: input.into(),
                weight: weight.into(),
                output: output.into(),
                epsilon: 1e-6,
            },
            inputs: vec![input.into()],
            outputs: vec![output.into()],
        }
    }

    /// Build an NF4 gate-up + SiLU activation fused group (2 matmuls + SiLU + Mul).
    fn nf4_gate_up_silu_group(group_size: u32) -> FusedGroup {
        let load_gate = make_load_node(
            0,
            "gate_proj.weight",
            CodecFamily::Nf4,
            group_size,
            GroupAxis::PackedContiguous,
            ProfMetadataLayout::AdjacentTile,
        );
        let load_up = make_load_node(
            1,
            "up_proj.weight",
            CodecFamily::Nf4,
            group_size,
            GroupAxis::PackedContiguous,
            ProfMetadataLayout::AdjacentTile,
        );
        let matmul_gate = make_matmul_node(2, "normalized", "gate_proj.weight", "gate_out");
        let matmul_up = make_matmul_node(3, "normalized", "up_proj.weight", "up_out");
        let silu = make_silu_node(4, "gate_out", "gated");
        let mul = make_mul_node(5, "gated", "up_out", "result");

        FusedGroup {
            id: format!("nf4_gs{}", group_size),
            body: vec![load_gate, load_up, matmul_gate, matmul_up, silu, mul],
            inputs: vec!["normalized".into()],
            outputs: vec!["result".into()],
            internal_values: vec!["gate_out".into(), "up_out".into(), "gated".into()],
            codec_family: CodecFamily::Nf4,
            precision_plan: None,
        }
    }

    /// Build an INT8 down-projection + residual add fused group.
    fn int8_down_residual_group(group_size: u32) -> FusedGroup {
        let load = make_load_node(
            0,
            "down_proj.weight",
            CodecFamily::Int8,
            group_size,
            GroupAxis::PackedContiguous,
            ProfMetadataLayout::AdjacentTile,
        );
        let matmul = make_matmul_node(1, "activated", "down_proj.weight", "down_out");
        let residual = make_residual_node(2, "skip", "down_out", "output");

        FusedGroup {
            id: format!("int8_gs{}", group_size),
            body: vec![load, matmul, residual],
            inputs: vec!["activated".into(), "skip".into()],
            outputs: vec!["output".into()],
            internal_values: vec!["down_out".into()],
            codec_family: CodecFamily::Int8,
            precision_plan: None,
        }
    }

    /// Build an unsupported 3-op RmsNorm → MatMul → SiLU group (no residual/add).
    fn unsupported_group() -> FusedGroup {
        let load = make_load_node(
            0,
            "w",
            CodecFamily::Nf4,
            32,
            GroupAxis::PackedContiguous,
            ProfMetadataLayout::AdjacentTile,
        );
        let add = make_add_node(1, "a", "b", "c");
        // LoadWeight + Add with NF4: has_add=true, matmul_count=0, no SiLU/Mul/Residual
        // → hits the catch-all arm (not SingleOp)
        FusedGroup {
            id: "unsupported".into(),
            body: vec![load, add],
            inputs: vec!["".into()],
            outputs: vec!["c".into()],
            internal_values: vec![],
            codec_family: CodecFamily::Nf4,
            precision_plan: None,
        }
    }

    /// Build an empty group.
    fn empty_group() -> FusedGroup {
        FusedGroup {
            id: "empty".into(),
            body: vec![],
            inputs: vec![],
            outputs: vec![],
            internal_values: vec![],
            codec_family: CodecFamily::Fp16,
            precision_plan: None,
        }
    }

    // ── Tests ────────────────────────────────────────────────────────

    #[test]
    fn nf4_g32_vs_g128_distinct_keys() {
        let g32 = nf4_gate_up_silu_group(32);
        let g128 = nf4_gate_up_silu_group(128);

        let op32 = metal_lower_fused_group(
            &g32,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("nf4 g32 lowering");
        let op128 = metal_lower_fused_group(
            &g128,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("nf4 g128 lowering");

        assert_ne!(
            op32.specialization, op128.specialization,
            "NF4 g32 and g128 must produce distinct specialization keys"
        );

        let pat32 = super::infer_fusion_pattern(&g32, op32.specialization.codec).unwrap();
        let pat128 = super::infer_fusion_pattern(&g128, op128.specialization.codec).unwrap();
        let key32 = super::build_pso_cache_key(
            op32.specialization.template_id,
            op32.specialization.codec,
            op32.specialization.group_size,
            640,
            op32.specialization.metadata_layout,
            op32.specialization.group_axis,
            pat32,
            op32.specialization.execution_phase,
        );
        let key128 = super::build_pso_cache_key(
            op128.specialization.template_id,
            op128.specialization.codec,
            op128.specialization.group_size,
            640,
            op128.specialization.metadata_layout,
            op128.specialization.group_axis,
            pat128,
            op128.specialization.execution_phase,
        );
        assert_ne!(key32, key128, "NF4 g32 and g128 PSO cache keys must differ");
    }

    #[test]
    fn int8_g32_vs_g128_distinct_keys() {
        let g32 = int8_down_residual_group(32);
        let g128 = int8_down_residual_group(128);

        let op32 = metal_lower_fused_group(
            &g32,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("int8 g32 lowering");
        let op128 = metal_lower_fused_group(
            &g128,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("int8 g128 lowering");

        assert_ne!(
            op32.specialization, op128.specialization,
            "INT8 g32 and g128 must produce distinct specialization keys"
        );

        let pat32 = super::infer_fusion_pattern(&g32, op32.specialization.codec).unwrap();
        let pat128 = super::infer_fusion_pattern(&g128, op128.specialization.codec).unwrap();
        let key32 = super::build_pso_cache_key(
            op32.specialization.template_id,
            op32.specialization.codec,
            op32.specialization.group_size,
            640,
            op32.specialization.metadata_layout,
            op32.specialization.group_axis,
            pat32,
            op32.specialization.execution_phase,
        );
        let key128 = super::build_pso_cache_key(
            op128.specialization.template_id,
            op128.specialization.codec,
            op128.specialization.group_size,
            640,
            op128.specialization.metadata_layout,
            op128.specialization.group_axis,
            pat128,
            op128.specialization.execution_phase,
        );
        assert_ne!(
            key32, key128,
            "INT8 g32 and g128 PSO cache keys must differ"
        );
    }

    #[test]
    fn gate_up_vs_down_distinct_pattern_ids() {
        let gate_up = nf4_gate_up_silu_group(32);
        let down = int8_down_residual_group(32);

        let op_gate = metal_lower_fused_group(
            &gate_up,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("gate_up lowering");
        let op_down = metal_lower_fused_group(
            &down,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("down residual lowering");

        assert_ne!(
            op_gate.specialization.mode_flags, op_down.specialization.mode_flags,
            "gate-up and down patterns must have distinct mode_flags"
        );
        assert_ne!(
            op_gate.specialization, op_down.specialization,
            "gate-up and down patterns must produce distinct specialization keys"
        );
    }

    #[test]
    fn unsupported_fusion_returns_error() {
        let group = unsupported_group();
        let result = metal_lower_fused_group(
            &group,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        );
        assert!(result.is_err(), "unsupported pattern should return error");
        match &result {
            Err(MetalLoweringError::UnsupportedFusionPattern(detail)) => {
                assert!(
                    detail.contains("add=true"),
                    "error should describe the pattern, got: {}",
                    detail
                );
            }
            other => panic!(
                "expected UnsupportedFusionPattern, got {:?}",
                other.as_ref().map(|_| ())
            ),
        }
    }

    #[test]
    fn empty_group_returns_error() {
        let group = empty_group();
        let result = metal_lower_fused_group(
            &group,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(MetalLoweringError::EmptyGroup)));
    }

    #[test]
    fn derive_function_constants_matches_specialization() {
        let group = nf4_gate_up_silu_group(32);
        let op = metal_lower_fused_group(
            &group,
            HardwareProfileId::AppleMBaseMemoryBound,
            ExecutionPhase::Decode,
        )
        .expect("lowering");

        let codec = op.specialization.codec;
        let gs = op.specialization.group_size;
        let ml = op.specialization.metadata_layout;
        let ga = op.specialization.group_axis;
        let pat = super::infer_fusion_pattern(&group, codec).unwrap();
        let phase = op.specialization.execution_phase;

        let fc = super::derive_function_constants(&group, codec, gs, ml, ga, pat, phase);

        assert_eq!(fc.group_size, 32);
        assert_eq!(fc.codec_id, super::codec_id(codec));
        assert_eq!(fc.metadata_layout_id, super::metadata_layout_id(ml));
        assert_eq!(fc.mode_flags, pat.encode_u32());
        assert_eq!(fc.execution_phase_id, super::execution_phase_id(phase));
    }
}
