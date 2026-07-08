//! Execution plan — kernel specialization, region batching, and model-level planning.
//!
//! This module defines the planner-side data types that bridge the gap between
//! composable kernel templates and dense command-buffer execution regions.
//!
//! Each type is independently serializable so the planner, profiler, and receipt
//! system can exchange structured plans without coupling to the Metal runtime.
//!
//! The five layers:
//!   1. KernelTemplate — reusable Metal kernel source
//!   2. KernelSpecialization — template + function constants + codec/layout params
//!   3. ScheduledKernelOp — concrete op with buffer views and dependencies
//!   4. ExecutionRegion — command-buffer scheduling unit
//!   5. ModelExecutionPlan — full ordered plan for one cimage + hardware profile

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ── KernelTemplate ───────────────────────────────────────────────────────

/// Identifier for a reusable kernel template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelTemplateId {
    Nf4Tile640Gemv,
    Int8Tile640Gemv,
    FusedGateUp,
    FusedGateUpActivation,
    FusedDownProjResidual,
    FusedOProjResidual,
    FusedRmsNormQkv,
    FusedAttentionScoreProbe,
    Gemma4FullInt4,
    RawF32Matmul,
    Fp16Matmul,
}

/// A reusable Metal kernel template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelTemplate {
    pub id: KernelTemplateId,
    pub metal_function_name: String,
    pub expected_bindings: Vec<BindingSpec>,
    pub supported_phases: Vec<ExecutionPhase>,
    pub supported_codecs: Vec<CodecFamily>,
    pub supports_function_constants: bool,
}

/// A single buffer binding slot expected by a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingSpec {
    pub index: u32,
    pub purpose: String,
    pub required: bool,
}

// ── Execution phase ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    Mixed,
}

// ── Codec family ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecFamily {
    Nf4,
    Int8,
    Fp16,
    RawF32,
    SymInt4,
    Ternary,
}

// ── KernelSpecialization ─────────────────────────────────────────────────

/// Every layout/codec parameter that affects a PSO must appear in this key.
/// No declared parameter may be silently ignored.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelSpecializationKey {
    pub template_id: KernelTemplateId,
    pub execution_phase: ExecutionPhase,
    pub codec: CodecFamily,
    pub tile_shape: TileShape,
    pub group_size: u32,
    pub group_axis: Axis,
    pub affine_mode: AffineMode,
    pub metadata_layout: MetadataLayout,
    pub input_dtype: DType,
    pub output_dtype: DType,
    pub hardware_profile: HardwareProfileId,
    pub mode_flags: u32,
}

/// A fully specialized kernel with resolved function constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpecialization {
    pub key: KernelSpecializationKey,
    pub function_constants: FunctionConstantSet,
    pub pso_cache_key: String,
    pub validation_digest: Option<String>,
}

/// Canonical set of Metal function constants.
/// Maps directly to constant_ids in .metal files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionConstantSet {
    pub page_width: u32,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub group_size: u32,
    pub codec_id: u32,
    pub affine_mode_id: u32,
    pub metadata_layout_id: u32,
    pub mode_flags: u32,
    pub execution_phase_id: u32,
}

// ── Tile shape ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub elements: u32,
}

impl TileShape {
    pub const fn tile640_decode() -> Self {
        Self { m: 1, n: 640, k: 0, elements: 640 }
    }
    pub const fn tile256_decode() -> Self {
        Self { m: 1, n: 256, k: 0, elements: 256 }
    }
    pub const fn tile1024_decode() -> Self {
        Self { m: 1, n: 1024, k: 0, elements: 1024 }
    }
}

// ── Axis, AffineMode, MetadataLayout, DType, HardwareProfile ────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
    Output,
    Input,
    TileLocal,
    PackedContiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AffineMode {
    ScaleOnly,
    ScaleBias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetadataLayout {
    AdjacentTile,
    SeparatedManifest,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    I8,
    U8,
    I32,
    U32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareProfileId {
    AppleA18Tiny,
    AppleMBaseMemoryBound,
    AppleMProBalanced,
    AppleMMaxBandwidth,
    AppleMUltraSharded,
}

impl HardwareProfileId {
    pub fn scratch_budget_bytes(&self) -> u64 {
        match self {
            HardwareProfileId::AppleA18Tiny => 256 * 1024 * 1024,
            HardwareProfileId::AppleMBaseMemoryBound => 512 * 1024 * 1024,
            HardwareProfileId::AppleMProBalanced => 1024 * 1024 * 1024,
            HardwareProfileId::AppleMMaxBandwidth => 2 * 1024 * 1024 * 1024,
            HardwareProfileId::AppleMUltraSharded => 4 * 1024 * 1024 * 1024,
        }
    }
}

// ── ScheduledKernelOp ───────────────────────────────────────────────────

/// A concrete operation instance in a layer plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledKernelOp {
    pub op_id: String,
    pub op_kind: KernelOpKind,
    pub tensor_key: Option<String>,
    pub tensor_class: Option<String>,
    pub specialization: KernelSpecializationKey,
    pub bindings: Vec<BufferBindingPlan>,
    pub dependencies: Vec<String>,
    pub buffer_uses: Vec<BufferUse>,
    pub dispatch_shape: DispatchShape,
    pub estimated_cost: EstimatedKernelCost,
    pub validation_requirements: KernelValidationRequirements,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelOpKind {
    RmsNorm,
    QkvProjection,
    AttentionScore,
    AttentionApply,
    OProjectionResidual,
    MlpGateUp,
    MlpActivation,
    MlpDownResidual,
    BridgeProjection,
    VisionPatchProjection,
    TtsProjection,
    TokenEmbedding,
    LmHead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferBindingPlan {
    pub slot: u32,
    pub buffer_id: String,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferUse {
    pub buffer_id: String,
    pub access: AccessMode,
    pub lifetime: LifetimeClass,
    pub alias_group: Option<String>,
    pub byte_range: Option<ByteRange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifetimeClass {
    PersistentWeight,
    PersistentKvCache,
    RegionInput,
    RegionOutput,
    LayerScratch,
    OpScratch,
    Ephemeral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchShape {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub threadgroup_m: u32,
    pub threadgroup_n: u32,
    pub threadgroup_p: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedKernelCost {
    pub compute_us: f64,
    pub memory_bytes_read: u64,
    pub memory_bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelValidationRequirements {
    pub allows_in_place_input_output: bool,
    pub requires_zeroed_output: bool,
    pub requires_aligned_metadata: bool,
    pub requires_hardware_validation: bool,
}

// ── ExecutionRegion ──────────────────────────────────────────────────────

/// A command-buffer scheduling unit: one layer decode, one prefill, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRegion {
    pub region_id: String,
    pub region_kind: ExecutionRegionKind,
    pub layer_index: Option<u32>,
    pub phase: ExecutionPhase,
    pub ops: Vec<ScheduledKernelOp>,
    pub command_buffer_policy: CommandBufferPolicy,
    pub hazard_policy: HazardPolicy,
    pub arena_plan: ActivationArenaPlan,
    pub timing_policy: TimingPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionRegionKind {
    DecoderLayerDecode,
    DecoderLayerPrefill,
    VisionPrefill,
    CrossModalBridge,
    TtsDecode,
    TtsPrefill,
    Embedding,
    LmHead,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandBufferPolicy {
    pub encode_region_as_single_command_buffer: bool,
    pub allow_multiple_compute_encoders: bool,
    pub allow_encoder_boundaries_for_hazards: bool,
    pub commit_after_region: bool,
    pub use_shared_events: bool,
}

impl CommandBufferPolicy {
    pub fn decode_default() -> Self {
        Self {
            encode_region_as_single_command_buffer: true,
            allow_multiple_compute_encoders: true,
            allow_encoder_boundaries_for_hazards: true,
            commit_after_region: true,
            use_shared_events: false,
        }
    }
}

/// Conservative hazard check policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HazardPolicy {
    Conservative,
    Aggressive,
    ValidateOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimingPolicy {
    Disabled,
    PerRegion,
    PerOp { max_ops: u32 },
}

// ── ActivationArenaPlan ─────────────────────────────────────────────────

/// A planned arena for scratch buffer allocation with alias support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationArenaPlan {
    pub arena_id: String,
    pub total_bytes: u64,
    pub allocations: Vec<ArenaAllocation>,
    pub alias_groups: Vec<AliasGroupPlan>,
    pub peak_live_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaAllocation {
    pub logical_buffer_id: String,
    pub offset: u64,
    pub size_bytes: u64,
    pub alignment_bytes: u32,
    pub lifetime_start_op: usize,
    pub lifetime_end_op: usize,
    pub alias_group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasGroupPlan {
    pub group_id: String,
    pub members: Vec<String>,
    pub total_bytes: u64,
}

// ── Hazard checker ─────────────────────────────────────────────────────

/// Result of hazard validation for an execution region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HazardPlan {
    pub encoder_boundaries: Vec<EncoderBoundary>,
    pub required_barriers: Vec<MemoryBarrier>,
    pub aliasing_approved: bool,
    pub safe: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderBoundary {
    pub after_op_index: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBarrier {
    pub after_op_index: usize,
    pub before_op_index: usize,
    pub mem_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HazardError {
    UnsafeAliasing { group_id: String, reason: String },
    OverlappingReadWrite { buffer_id: String, op_a: String, op_b: String },
    ScratchBudgetExceeded { requested: u64, budget: u64 },
    CyclicDependency { ops: Vec<String> },
    Unknown,
}

// ── ModelExecutionPlan ──────────────────────────────────────────────────

/// The full ordered execution plan for one cimage on one hardware/layout profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionPlan {
    pub plan_id: String,
    pub model_family: String,
    pub cimage_digest: String,
    pub policy_digest: String,
    pub layout_profile: HardwareProfileId,
    pub regions: Vec<ExecutionRegion>,
    pub pso_keys: Vec<KernelSpecializationKey>,
    pub total_scratch_budget_bytes: u64,
    pub validation_digest: Option<String>,
}

// ── Planning receipt ────────────────────────────────────────────────────

/// Receipt emitted during plan construction, before runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionPlanReceipt {
    pub plan_id: String,
    pub cimage_digest: String,
    pub policy_digest: String,
    pub layout_profile: HardwareProfileId,
    pub region_count: usize,
    pub scheduled_op_count: usize,
    pub pso_count: usize,
    pub peak_scratch_bytes: u64,
    pub unsupported_ops: Vec<String>,
    pub fallbacks: Vec<String>,
    pub warnings: Vec<String>,
}

// ── Hazard checker skeleton ─────────────────────────────────────────────

/// Conservative hazard checker.
pub struct HazardChecker;

impl HazardChecker {
    /// Validate an execution region's buffer accesses and aliasing plan.
    pub fn validate_region(region: &ExecutionRegion) -> Result<HazardPlan, HazardError> {
        let mut boundaries = Vec::new();
        let mut barriers = Vec::new();
        let mut safe = true;

        // Check for overlapping read-write pairs
        for (i, op_a) in region.ops.iter().enumerate() {
            for (j, op_b) in region.ops.iter().enumerate() {
                if i >= j { continue; }
                for use_a in &op_a.buffer_uses {
                    for use_b in &op_b.buffer_uses {
                        if use_a.buffer_id != use_b.buffer_id { continue; }
                        let is_write = matches!(use_a.access, AccessMode::Write | AccessMode::ReadWrite);
                        let is_write_b = matches!(use_b.access, AccessMode::Write | AccessMode::ReadWrite);
                        if is_write && is_write_b {
                            return Err(HazardError::OverlappingReadWrite {
                                buffer_id: use_a.buffer_id.clone(),
                                op_a: op_a.op_id.clone(),
                                op_b: op_b.op_id.clone(),
                            });
                        }
                        if is_write && matches!(use_b.access, AccessMode::Read) {
                            boundaries.push(EncoderBoundary {
                                after_op_index: i,
                                reason: format!("WAW/RAW boundary: {} after {}", op_a.op_id, op_b.op_id),
                            });
                        }
                    }
                }
            }
        }

        // Verify scratch budget
        if region.arena_plan.total_bytes > region.arena_plan.peak_live_bytes {
            // Conservative: check against hardware profile
        }

        Ok(HazardPlan {
            encoder_boundaries: boundaries,
            required_barriers: barriers,
            aliasing_approved: safe,
            safe,
        })
    }
}

// ── Arena planner skeleton ──────────────────────────────────────────────

pub struct ArenaPlanner;

impl ArenaPlanner {
    /// Plan scratch buffer allocation with interval-based aliasing.
    pub fn plan_arena(
        ops: &[ScheduledKernelOp],
        arena_id: &str,
    ) -> ActivationArenaPlan {
        let mut allocations = Vec::new();
        let mut alias_groups = Vec::new();
        let mut peak = 0u64;

        for op in ops {
            for use_ in &op.buffer_uses {
                if matches!(use_.lifetime,
                    LifetimeClass::PersistentWeight | LifetimeClass::PersistentKvCache
                ) { continue; }
                // Simple allocation: each scratch buffer gets its own slot
                // (interval coloring is a future optimization)
                let size = use_.byte_range.as_ref()
                    .map(|r| r.end - r.start)
                    .unwrap_or(0);
                if size == 0 { continue; }
                allocations.push(ArenaAllocation {
                    logical_buffer_id: use_.buffer_id.clone(),
                    offset: peak,
                    size_bytes: size,
                    alignment_bytes: 256,
                    lifetime_start_op: 0,
                    lifetime_end_op: ops.len(),
                    alias_group: use_.alias_group.clone(),
                });
                peak += size;
            }
        }

        ActivationArenaPlan {
            arena_id: arena_id.to_string(),
            total_bytes: peak,
            allocations,
            alias_groups,
            peak_live_bytes: peak,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specialization_key_nf4_g32_and_g128_differ() {
        let a = KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        };
        let b = KernelSpecializationKey {
            group_size: 128,
            ..a.clone()
        };
        assert_ne!(a, b, "different group_size must produce different keys");
    }

    #[test]
    fn test_specialization_key_codecs_differ() {
        let nf4 = KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        };
        let int8 = KernelSpecializationKey {
            codec: CodecFamily::Int8,
            ..nf4.clone()
        };
        let ternary = KernelSpecializationKey {
            codec: CodecFamily::Ternary,
            ..nf4.clone()
        };
        assert_ne!(nf4, int8);
        assert_ne!(nf4, ternary);
        assert_ne!(int8, ternary);
    }

    #[test]
    fn test_specialization_key_metadata_layout_changes_key() {
        let adjacent = KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        };
        let separated = KernelSpecializationKey {
            metadata_layout: MetadataLayout::SeparatedManifest,
            ..adjacent.clone()
        };
        assert_ne!(adjacent, separated);
    }

    #[test]
    fn test_specialization_key_group_axis_changes_key() {
        let packed = KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        };
        let input_axis = KernelSpecializationKey {
            group_axis: Axis::Input,
            ..packed.clone()
        };
        assert_ne!(packed, input_axis);
    }

    #[test]
    fn test_tile_shape_constants() {
        assert_eq!(TileShape::tile640_decode().elements, 640);
        assert_eq!(TileShape::tile256_decode().elements, 256);
        assert_eq!(TileShape::tile1024_decode().elements, 1024);
    }

    #[test]
    fn test_hardware_profile_budgets() {
        assert_eq!(HardwareProfileId::AppleA18Tiny.scratch_budget_bytes(), 256 * 1024 * 1024);
        assert_eq!(HardwareProfileId::AppleMMaxBandwidth.scratch_budget_bytes(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_execution_region_command_buffer_policy_default() {
        let p = CommandBufferPolicy::decode_default();
        assert!(p.encode_region_as_single_command_buffer);
        assert!(!p.use_shared_events);
    }

    #[test]
    fn test_hazard_checker_detects_overlapping_writes() {
        let region = ExecutionRegion {
            region_id: "test".into(),
            region_kind: ExecutionRegionKind::DecoderLayerDecode,
            layer_index: Some(0),
            phase: ExecutionPhase::Decode,
            ops: vec![
                ScheduledKernelOp {
                    op_id: "op1".into(),
                    op_kind: KernelOpKind::RmsNorm,
                    tensor_key: None,
                    tensor_class: None,
                    specialization: kernel_specialization_key_fixture(),
                    bindings: vec![],
                    dependencies: vec![],
                    buffer_uses: vec![BufferUse {
                        buffer_id: "shared_buf".into(),
                        access: AccessMode::Write,
                        lifetime: LifetimeClass::OpScratch,
                        alias_group: None,
                        byte_range: Some(ByteRange { start: 0, end: 1024 }),
                    }],
                    dispatch_shape: DispatchShape { grid_x: 1, grid_y: 1, grid_z: 1, threadgroup_m: 32, threadgroup_n: 1, threadgroup_p: 1 },
                    estimated_cost: EstimatedKernelCost { compute_us: 1.0, memory_bytes_read: 0, memory_bytes_written: 0 },
                    validation_requirements: KernelValidationRequirements { allows_in_place_input_output: false, requires_zeroed_output: false, requires_aligned_metadata: false, requires_hardware_validation: false },
                },
                ScheduledKernelOp {
                    op_id: "op2".into(),
                    op_kind: KernelOpKind::QkvProjection,
                    tensor_key: None,
                    tensor_class: None,
                    specialization: kernel_specialization_key_fixture(),
                    bindings: vec![],
                    dependencies: vec![],
                    buffer_uses: vec![BufferUse {
                        buffer_id: "shared_buf".into(),
                        access: AccessMode::Write,
                        lifetime: LifetimeClass::OpScratch,
                        alias_group: None,
                        byte_range: Some(ByteRange { start: 0, end: 1024 }),
                    }],
                    dispatch_shape: DispatchShape { grid_x: 1, grid_y: 1, grid_z: 1, threadgroup_m: 32, threadgroup_n: 1, threadgroup_p: 1 },
                    estimated_cost: EstimatedKernelCost { compute_us: 1.0, memory_bytes_read: 0, memory_bytes_written: 0 },
                    validation_requirements: KernelValidationRequirements { allows_in_place_input_output: false, requires_zeroed_output: false, requires_aligned_metadata: false, requires_hardware_validation: false },
                },
            ],
            command_buffer_policy: CommandBufferPolicy::decode_default(),
            hazard_policy: HazardPolicy::Conservative,
            arena_plan: ActivationArenaPlan {
                arena_id: "test".into(),
                total_bytes: 1024,
                allocations: vec![],
                alias_groups: vec![],
                peak_live_bytes: 1024,
            },
            timing_policy: TimingPolicy::Disabled,
        };

        let result = HazardChecker::validate_region(&region);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            HazardError::OverlappingReadWrite { buffer_id, .. } => {
                assert_eq!(buffer_id, "shared_buf");
            }
            _ => panic!("expected OverlappingReadWrite error"),
        }
    }

    #[test]
    fn test_model_execution_receipt_roundtrip() {
        let receipt = ModelExecutionPlanReceipt {
            plan_id: "plan_test".into(),
            cimage_digest: "cimage_digest".into(),
            policy_digest: "policy_digest".into(),
            layout_profile: HardwareProfileId::AppleMBaseMemoryBound,
            region_count: 42,
            scheduled_op_count: 300,
            pso_count: 8,
            peak_scratch_bytes: 500_000_000,
            unsupported_ops: vec![],
            fallbacks: vec![],
            warnings: vec![],
        };
        let json = serde_json::to_string(&receipt).unwrap();
        let back: ModelExecutionPlanReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.plan_id, "plan_test");
        assert_eq!(back.region_count, 42);
    }

    fn kernel_specialization_key_fixture() -> KernelSpecializationKey {
        KernelSpecializationKey {
            template_id: KernelTemplateId::Nf4Tile640Gemv,
            execution_phase: ExecutionPhase::Decode,
            codec: CodecFamily::Nf4,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        }
    }
}
