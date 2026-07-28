//! Shared Metal/Rust kernel struct types for the distill-compiler.
//!
//! All structs are `#[repr(C)]` to match Metal Shading Language layout exactly.
//! Field order, type size, and alignment must match the corresponding MSL
//! `struct` definitions in the `.metal` template files.

use serde::{Deserialize, Serialize};

// ── Page format ─────────────────────────────────────────────────────────────

/// A single packed ternary page (640 weights) in the deployable `.cimage` format.
///
/// Contains only the packed ternary payload and a compact page header — no
/// implicit compiler-only metadata. Every page is independently addressable,
/// independently verifiable, and consumable without dynamic allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackedTernaryPage640 {
    /// Packed 2-bit trits: 640 weights × 2 bits = 1280 bits = 160 bytes.
    /// For the tile640 format using base-3 encoding: 20 trits × log₂(3) bits
    /// per word, with 32 words per page (32 × 20 trits = 640).
    pub payload: [u32; 40],
    /// Page header: scale index, sidecar offset range, valid tail length, flags.
    pub header: PageHeader,
}

/// Compact page header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageHeader {
    /// Index into the page-scales buffer.
    pub scale_index: u32,
    /// Start offset in the sidecar entries buffer.
    pub sidecar_start: u32,
    /// End offset (exclusive) in the sidecar entries buffer.
    pub sidecar_end: u32,
    /// Number of valid weight positions in this page (≤ 640).
    pub valid_tail_length: u16,
    /// Flags: bit 0 = sidecar_present, bit 1 = tail_padding, bit 2–15 reserved.
    pub flags: u16,
}

// ── Sidecar header ──────────────────────────────────────────────────────────

/// Per-page sidecar span header.
///
/// Each threadgroup processing a page can load the small sidecar span into
/// threadgroup memory or registers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageSidecarHeader {
    pub start_index: u32,
    pub count: u16,
    pub encoding: u16,
    pub residual_scale: f32, // half-prec in shader, f32 for C alignment
    pub flags: u32,
}

// ── Projection parameters ───────────────────────────────────────────────────

/// Per-dispatch projection parameters.
///
/// Mode bits: sidecar disabled/enabled, instrumentation enabled/disabled,
/// candidate-layout mode.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProjectionParams {
    /// Input dimension.
    pub in_dim: u32,
    /// Output dimension.
    pub out_dim: u32,
    /// Number of pages in this projection.
    pub page_count: u32,
    /// Page width (typically 640).
    pub page_width: u32,
    /// Mode bits: bit0 = sidecar_enabled, bit1 = instrumentation_enabled,
    /// bit2 = candidate_layout_mode (alternate page buffer for scoring).
    pub mode_flags: u32,
    /// Seed for deterministic probe sampling.
    pub probe_seed: u32,
    /// Reserved for future use; pad to 16-byte alignment.
    pub reserved: [u32; 5],
}

// ── Kernel receipt (instrumentation counters) ───────────────────────────────

/// Compact instrumentation counters written by the kernel.
///
/// The shader writes logical counters; the host records dispatch start/end,
/// allocation state, command-buffer completion, and peak arena bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KernelReceipt {
    pub kernel_id: u32,
    pub phase_id: u32,
    pub page_count: u32,
    pub sidecar_hits: u32,
    pub sidecar_entries_read: u32,
    pub threadgroups: u32,
    pub threads_per_threadgroup: u32,
    pub output_elements: u32,
    pub flags: u32,
    pub logical_weight_bytes: u64,
    pub logical_sidecar_bytes: u64,
    pub logical_activation_bytes: u64,
}

// ── Activation view (arena slot descriptor) ─────────────────────────────────

/// Compact slot descriptor passed to kernels via indirection table.
///
/// The host binds a large arena buffer, and each kernel receives offsets plus
/// descriptors. This lets you keep one bounded activation pool and reuse slots
/// without repeated buffer allocation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ActivationView {
    pub byte_offset: u64,
    pub row_stride: u32,
    pub col_stride: u32,
    pub dtype: u32,
    pub layout: u32,
    pub token_count: u32,
    pub hidden_dim: u32,
}

// ── Error partial (streaming comparison) ────────────────────────────────────

/// Fixed-order partial reduction record written by comparison kernels.
///
/// CPU or Accelerate lane reduces these records deterministically in canonical
/// threadgroup order. Avoids nondeterministic atomic ordering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ErrorPartial {
    pub sum_sq_error: f32,
    pub sum_abs_error: f32,
    pub dot_teacher_student: f32,
    pub sum_teacher_sq: f32,
    pub sum_student_sq: f32,
    pub max_abs_error: f32,
    pub element_count: u32,
    _pad: u32,
}

// ── Page score (candidate scoring) ──────────────────────────────────────────

/// Compact candidate scoring record.
///
/// The candidate scoring kernel estimates output impact for a candidate page's
/// changed entries without fully executing the whole block.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PageScore {
    pub page_id: u32,
    _pad: u32,
    pub local_weighted_error: f32,
    pub predicted_activation_delta: f32,
    pub sidecar_cost: f32,
    pub estimated_bytes: f32,
    pub estimated_loads: f32,
    pub accepted_score: f32,
    pub challenger_score: f32,
    pub flags: u32,
    _pad2: u32,
}

// ── Attention probe ─────────────────────────────────────────────────────────

/// Sampled attention probe output.
///
/// Store sampled rows, softmax entropy summaries, per-head maximum logits,
/// and attention mass concentration rather than full attention matrices.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionProbe {
    pub head_id: u32,
    pub token_index: u32,
    pub teacher_max_logit: f32,
    pub student_max_logit: f32,
    pub teacher_entropy: f32,
    pub student_entropy: f32,
    pub sampled_probability_l1: f32,
    pub sampled_probability_kl: f32,
}

// ── Buffer binding indices ──────────────────────────────────────────────────

/// Named constants for `[[buffer(N)]]` bindings shared across kernels.
pub mod buffer_slot {
    pub const WEIGHTS: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const PAGE_SCALES: u32 = 2;
    pub const CHANNEL_SCALES: u32 = 3;
    pub const SIDECAR: u32 = 4;
    pub const SIDECAR_OFFSETS: u32 = 5;
    pub const OUTPUT: u32 = 6;
    pub const PARAMS: u32 = 7;
    pub const RECEIPT: u32 = 8;
    pub const ERROR_PARTIALS: u32 = 9;
    pub const PROBE_OUTPUT: u32 = 10;
    pub const PAGE_SCORES: u32 = 11;
    pub const ACTIVATION_ARENA: u32 = 12;
    pub const ARENA_DESCRIPTORS: u32 = 13;
}
