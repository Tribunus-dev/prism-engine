//! Shared vocabulary for the heterogeneous cimage schema.
//!
//! This module owns the **canonical type identities** the heterogeneous
//! cimage section references throughout its sub-modules: the execution
//! lane enum, the content-hash integrity tag, the activation ABI
//! contract, the tensor dtype enum, the physical layout enum, and the
//! operation family enum.
//!
//! These types are **pure data shapes** — no `unsafe`, no hardware
//! handles, no FFI, no process-local state. They are reimplemented
//! locally rather than imported from the engine because:
//!
//! 1. The destination crate does not (and must not) depend on the
//!    engine, which is mid-absorption and has pre-existing build
//!    errors.
//! 2. The constitutional crate is the **source of truth for state**;
//!    re-exporting engine types would keep the engine as a parallel
//!    authority. The engine counterparts are absorption targets, not
//!    dependencies.
//!
//! The shapes here are byte-for-byte compatible with the engine's
//! `serde_json` representation (same variant names, same field
//! layouts), so a cimage emitted by either side round-trips through
//! the other during the migration window.

use serde::{Deserialize, Serialize};

// ── Execution lane ────────────────────────────────────────────────────────

/// Which physical execution lane a phase, slot, or variant targets.
///
/// The engine uses `MlxGpu` (Metal GPU), `CoreAiAne` (Apple Neural
/// Engine via Core ML), and `AccelerateCpu` (Apple's vDSP/Accelerate
/// CPU). These labels are configuration descriptors — they identify
/// the lane a phase is *eligible* to run on, not live hardware
/// handles. The actual device, command queue, or model handle is
/// execution-boundary state owned by the per-backend runtime crate.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExecutionLane {
    /// Metal GPU (MLX-backed, the primary decode/attention lane).
    MlxGpu,
    /// Apple Neural Engine via Core ML (primary prefill/compaction lane).
    CoreAiAne,
    /// Apple Accelerate (vDSP) on CPU (fallback / small-shape lane).
    AccelerateCpu,
}

// ── Content hash ──────────────────────────────────────────────────────────

/// Opaque content hash used to integrity-tag cimage sections, programs,
/// and admission rules.
///
/// This is a deterministic 64-bit content fingerprint — a value, not
/// a handle. The runtime uses it to detect drift, partial
/// invalidation, and re-emission. The hashing function (BLAKE3 in the
/// current implementation) is an execution-plane concern and lives in
/// the runtime crate.
#[derive(
    Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct ContentHash(pub u64);

impl ContentHash {
    /// The zero hash — the identity element for content comparison.
    pub const ZERO: ContentHash = ContentHash(0);
}

impl Default for ContentHash {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContentHash({})", self.0)
    }
}

// ── Tensor dtype ──────────────────────────────────────────────────────────

/// Element dtype of a tensor flowing through the heterogeneous image.
///
/// Used as a sub-field of [`ActivationAbi`] variants. The runtime
/// matches on this value when deciding which lane-eligible
/// implementation to dispatch to.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum TensorDtype {
    Float16,
    Float32,
    BFloat16,
    Int8,
    UInt8,
    UInt16,
    Int32,
    Unknown,
}

// ── Physical layout ───────────────────────────────────────────────────────

/// How a tensor's logical dimensions map to physical memory.
///
/// Part of the activation ABI contract — a producer and consumer must
/// agree on the layout or materialize a copy at the boundary.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhysicalLayout {
    /// Row-major contiguous (C-order).
    ContiguousRowMajor,
    /// NCHW channel-first.
    NCHW,
    /// NHWC channel-last.
    NHWC,
    /// Custom stride-defined layout.
    Custom(Vec<u64>),
}

// ── Activation ABI ────────────────────────────────────────────────────────

/// Per-variant ABI for an activation tensor crossing a lane boundary.
///
/// This is the **producer/consumer contract** for an activation: the
/// physical layout, dtype, alignment, and shape constraints the
/// producer emits and the consumer expects. A `LaneCapability`
/// records which ABI a given phase requires on a given lane; a
/// `MaterializationPlan` describes what to insert if producer and
/// consumer do not match.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ActivationAbi {
    /// Decode-step activation (KV-cache projections, MLP intermediates).
    DecodeActivationV1(DecodeActivationV1Params),
    /// MHA / GQA attention heads.
    AttentionHeads(AttentionHeadsParams),
    /// Vision encoder/decoder image tensors.
    VisionImage(VisionImageParams),
    /// Opaque metal-only buffer (no tensor semantics).
    MetalOnly(MetalOnlyParams),
}

/// Parameters for a decode-step activation V1.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct DecodeActivationV1Params {
    pub dtype: TensorDtype,
    pub seq_bucket: u32,
    pub hidden_dim: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
    pub stride_constraint: Option<Vec<u64>>,
}

/// Parameters for attention head projections.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttentionHeadsParams {
    pub dtype: TensorDtype,
    pub num_heads: u32,
    pub seq_bucket: u32,
    pub head_dim: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
}

/// Parameters for vision image tensors.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct VisionImageParams {
    pub dtype: TensorDtype,
    pub channel_count: u32,
    pub height: u32,
    pub width: u32,
    pub physical_layout: PhysicalLayout,
    pub alignment: u32,
}

/// Parameters for an opaque metal-only buffer.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetalOnlyParams {
    pub name: String,
    pub dtype: TensorDtype,
    pub byte_count: u64,
}

// ── Operation family ──────────────────────────────────────────────────────

/// Coarse classification of a compiled phase node, used by the
/// executor for route eligibility and by the admission gates for
/// qualification.
///
/// The variants are intentionally a flat enum — they identify the
/// *kind* of work (a Q-projection, an attention block, a softmax)
/// without carrying the per-call parameters. The phase node's
/// [`crate::cimage_manifest::heterogeneous::lane_programs::ProgramBinding`]
/// carries the binding details.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum OperationFamily {
    /// Matrix multiplication.
    Matmul,
    /// Quantized matrix multiplication.
    QuantizedMatmul,
    /// Element-wise add.
    Add,
    /// Element-wise multiply.
    Multiply,
    /// SiLU activation.
    Silu,
    /// Softmax.
    Softmax,
    /// Reduction (sum, mean, max, …).
    Reduction,
    /// Reshape (view-only).
    Reshape,
    /// Transpose.
    Transpose,
    /// Index select / gather.
    IndexSelect,
    /// RMS normalization.
    RmsNorm,
    /// Rotary position embedding.
    RoPE,
    /// Sampling (top-k / top-p / temperature).
    Sampling,
    /// Layout transform (NCHW↔NHWC etc.).
    LayoutTransform,
    /// Checksum / integrity.
    Checksum,
    /// Attention block (Q/K/V/O projections as a unit).
    AttentionBlock,
    /// MLP block (gate/up/down as a unit).
    MlpBlock,
    /// Decoder layer (attention + MLP as a unit).
    DecoderLayer,
    /// Prefill fragment (one of several in a long-context prefill).
    PrefillFragment,
    /// Q projection in attention.
    QProj,
    /// K projection in attention.
    KProj,
    /// V projection in attention.
    VProj,
    /// O projection in attention.
    OProj,
    /// MLP gate projection.
    GateProj,
    /// MLP up projection.
    UpProj,
    /// MLP down projection.
    DownProj,
    /// Vision encoder pass.
    VisionEncode,
    /// Audio encoder pass.
    AudioEncode,
    /// Multimodal projection (vision/audio → text embedding space).
    MultimodalProject,
}
