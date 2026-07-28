//! Constitutional compile surface for the engine's
//! `compute_image::{compile,orchestrator}/` directories.
//!
//! ## Sub-modules (single authority per file)
//!
//! | Sub-module | Authority |
//! |---|---|
//! | [`cimage_format`] | CImage wire format — header, segment directory, segment-kind taxonomy. |
//! | [`cimage_layout`] | CImage layout metadata — `CimageLayoutMeta`, `TensorRecord`, `verify_cimage`. |
//! | [`matrix_binding`] | `MatrixWeightBindingV1` and its LE (de)serialisation. |
//! | [`execution_graph`] | Self-describing execution graph descriptor (segment 24). |
//! | [`ternary_pipeline_quant`] | tile640 ternary quantisation pipeline (v7) — std-only. |
//! | [`int4_pack`] | CPU-side ternary repacker (5-trits/byte `TernaryBlock32`). |
//! | [`kernel_types`] | Per-page kernel receipts and projection metadata. |
//! | [`fp16`] | IEEE 754 FP16 ↔ f32 conversion helpers. |
//! | [`swizzled`] | Block-swizzled ternary repack helpers. |
//! | [`ternary_block_quant`] | 256-element block ternary quantiser + ANE swizzle. |
//! | [`cimage_compile`] | `ModelConfig`, `CompiledTensor`, `build_cimage`. |
//! | [`orchestrator_types`] | Orchestrator data types (phases, MTP, decode policy). |
//!
//! ## Migration status
//!
//! This surface absorbed the engine's
//! `compute-core/src/ecs/compute_image/compile/` (24 files, 20,356 LOC)
//! and `compute-core/src/ecs/compute_image/orchestrator/` (8 files,
//! 4,011 LOC) directories on 2026-07-27 as part of the `ci-compile`
//! migration. Data-only files and pure algorithms are re-implemented
//! here. Engine-coupled implementations (Metal/MLX/ROCm/ANE dispatch,
//! kernel registry, ML pipeline, file-system writers, GPU packers, etc.)
//! remain in the engine's `legacy_compute_image_compile/` directory
//! and engine callers retargeted to `crate::ecs::legacy_compute_image_compile::X`.
//!
//! ## Cross-crate authority
//!
//! The constitutional `Orchestrator` runtime (which owns MLX / Metal
//! pipeline state) lives in `prism_ecs_runtime::orchestrator`; this
//! module owns only the data types and pure helpers that engine callers
//! can share without taking on Metal / MLX dependencies.

pub mod cimage_compile;
pub mod cimage_format;
pub mod cimage_layout;
pub mod execution_graph;
pub mod fp16;
pub mod int4_pack;
pub mod kernel_types;
pub mod matrix_binding;
pub mod orchestrator_types;
pub mod swizzled;
pub mod ternary_block_quant;
pub mod ternary_pipeline_quant;

// Re-exports for ergonomic `prism_ecs_compile::compute_image_compile::X` paths.
pub use cimage_compile::{build_cimage, CompiledTensor, ModelConfig};
pub use cimage_format::{
    model_artifact_tag, read_cimage_header_le, write_cimage_header_le, AneModelDescriptor,
    AneModelRole, CimageHeader, LayerDirectoryEntry, ModelArtifactEntry, ModelArtifactIter,
    PrismCimageHeader, SegmentEntry, SegmentKind, CIMAGE_HEADER_WIRE_SIZE, CIMAGE_PAGE_SIZE,
    CIMAGE_SEGMENT_CAPACITY, PRISM_MAGIC, QUANT_SCHEMA_NF4_TILE640, QUANT_SCHEMA_TERNARY_TILE640,
};
pub use cimage_layout::{
    verify_cimage, verify_prism_cimage, CimageLayoutMeta, PrismCimageLayoutMeta, TensorRecord,
    CIMAGE_LAYOUT_META_WIRE_SIZE,
};
pub use execution_graph::{
    sidecar_byte_len, AttentionKind, CompactionEpoch, DeviceCapability, DraftSubGraph,
    ExecutionGraphDescriptor, LayerExecutionNode, MatrixWeightBinding, NodeKind,
    SidecarElementFormat, SidecarKind, EXECUTION_GRAPH_MAGIC,
};
pub use fp16::{f32_from_half, fp16_to_f32, half_from_f32, half_to_f32};
pub use int4_pack::{
    interleave_fused_ternary_layer, pack_5_trits, quantize_to_ternary_block32,
    repack_ternary_tensor, unpack_byte_5_trits, AlignedTernaryBlock32, TernaryBlock32,
};
pub use kernel_types::{
    buffer_slot, ActivationView, AttentionProbe, ErrorPartial, KernelReceipt, PageHeader,
    PageSidecarHeader, PackedTernaryPage640, PageScore, ProjectionParams,
};
pub use matrix_binding::{
    read_matrix_weight_binding_v1_le, write_matrix_weight_binding_v1_le, MatrixWeightBindingV1,
    MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH,
};
pub use orchestrator_types::{
    generate_speculative_candidates, sample_argmax, sample_argmax_f32, AppleDecodePolicy,
    AppleMemoryPressureClass, ComputePhaseKind, GpuWeightCacheMode, MtpDecodeRequest,
    MtpDecodeResult, MtpKvState, MtpProposal, MtpTreeSurface, GLOBAL_HEAD_DIM, LAYERS,
    MAX_CONTEXT, MAX_SURVIVORS, NUM_KV_HEADS, NUM_SLOTS,
};
pub use swizzled::{
    decode_ternary_u32, repack_ternary_to_swizzled_u8, swizzled_byte_offset,
    swizzled_buffer_size,
};
pub use ternary_block_quant::{
    generate_ane_swizzled_weights, requantize_kv_to_swizzled_u8, ternary_quantize_block,
};
pub use ternary_pipeline_quant::{
    bf16_bits_to_f32, dequantize, f32_to_bf16_bits, quantize_tensor, QuantConfig,
    QuantizedTensor, Rounding, LANE, PAGE,
};
