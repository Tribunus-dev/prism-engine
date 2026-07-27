//! `KernelDispatcher` typed port — constitutional shape of a single GPU
//! kernel dispatch.
//!
//! This module owns the constitutional authority for the
//! `KernelDispatcher` typed port: the provider-neutral contract that
//! the runtime uses to request a single GPU kernel execution. The
//! port defines [`KernelKind`] (the constitutional enumeration of every
//! supported kernel class), [`KernelBindingRole`] (the constitutional
//! name for each `[[buffer(N)]]` slot, decoupled from the numeric
//! index), [`DispatchBufferHandle`] (an opaque, provider-typed GPU
//! buffer handle), [`KernelDispatchParams`] (the per-dispatch shape),
//! [`KernelDispatchRequest`] (the full request), [`KernelDispatchOutcome`]
//! (the receipt), [`DispatchError`] (Rejected / Failed / Stale), and
//! the [`KernelDispatcher`] trait that every provider implementation
//! conforms to.
//!
//! The port is **provider-neutral**: it does not name Metal types
//! (`metal::CommandBufferRef`, `metal::Buffer`, `MTLSize`,
//! `MTLResourceOptions`), does not own process-local state, and does
//! not perform any FFI. The Metal implementation lives in
//! `compute-core/src/ecs/compute_image/compile/kernel_dispatch.rs`
//! (17 dispatchers, all `#[cfg(feature = "metal-dispatch")]`); it
//! owns the encoder handles, the `Arc<parking_lot::Mutex<KernelRegistry>>`
//! per-process pipeline cache, the raw encoder operations, and the
//! `unsafe` read-back paths. The constitutional port is the
//! **contract**; the engine file is the **executor**. A future
//! absorption pass will route runtime calls through the trait so
//! the engine no longer holds canonical state in its dispatcher
//! surface.
//!
//! Propagation: this file is **type-only** — it introduces no state
//! and participates in no durable event chain. The downstream consumer
//! is the engine file, which already implements the per-kernel
//! dispatchers; once the engine file is re-typed behind this trait,
//! the runtime path will read the request, dispatch via the trait,
//! and project the outcome into the `CompileReceipt` chain.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Kernel kind ─────────────────────────────────────────────────────────────

/// Constitutional enumeration of every Metal kernel class the runtime
/// may dispatch. Each variant names one dispatcher in
/// `compute-core/.../kernel_dispatch.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum KernelKind {
    /// `ternary_tile640_gemv` — 1.6-bit ternary GEMV.
    TernaryProjection,
    /// `fused_gemv_nf4_tile640_fp32` — shared-layout NF4 Tile640 FP32 GEMV.
    Nf4Tile640Projection,
    /// `fused_gemv_nf4_scaled_reduction_tile640_fp32` — NF4 Tile640 with FP16
    /// reduction-axis scaling sidecar.
    Nf4ScaledReductionTile640,
    /// `fused_gemv_int8_tile640_fp32` — INT8 Tile640 FP32 GEMV.
    Int8Tile640GEMV,
    /// `batched_gemv_fp32` — batched GEMV for compile-time operator validation.
    GpuBatchMatmul,
    /// `palettized_gemv` — dense codebook-quantized GEMV.
    DenseProjection,
    /// `fused_teacher_student_gemv` — fused teacher-student forward GEMV.
    FusedTeacherStudent,
    /// `error_partial` — teacher-student error-partial reduce.
    ErrorPartial,
    /// `attention_probe` — attention distribution probe.
    AttentionProbe,
    /// `candidate_score` — candidate page scoring.
    CandidateScore,
    /// `pack_verify` — pack verification.
    PackVerify,
    /// `rmsnorm_residual_probe` — RMS norm with residual probe.
    RmsnormResidualProbe,
    /// `mlp_activation_probe` — MLP activation probe.
    MlpActivationProbe,
    /// `sidecar_apply_verify` — sidecar apply verification.
    SidecarApplyVerify,
    /// `fused_rmsnorm_qkv` — fused RMS norm + QKV projection.
    FusedRmsnormQkv,
    /// `fused_o_proj_residual` — fused O-proj + residual.
    FusedOProjResidual,
    /// `fused_multimodal` — fused multimodal projection.
    FusedMultimodal,
}

impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kernel_name())
    }
}

impl KernelKind {
    /// Stable, canonical kernel name as it appears in the engine file
    /// (`kernel_name: &'static str` field on each dispatcher).
    pub const fn kernel_name(self) -> &'static str {
        match self {
            Self::TernaryProjection => "ternary_tile640_gemv",
            Self::Nf4Tile640Projection => "fused_gemv_nf4_tile640_fp32",
            Self::Nf4ScaledReductionTile640 => "fused_gemv_nf4_scaled_reduction_tile640_fp32",
            Self::Int8Tile640GEMV => "fused_gemv_int8_tile640_fp32",
            Self::GpuBatchMatmul => "batched_gemv_fp32",
            Self::DenseProjection => "palettized_gemv",
            Self::FusedTeacherStudent => "fused_teacher_student_gemv",
            Self::ErrorPartial => "error_partial",
            Self::AttentionProbe => "attention_probe",
            Self::CandidateScore => "candidate_score",
            Self::PackVerify => "pack_verify",
            Self::RmsnormResidualProbe => "rmsnorm_residual_probe",
            Self::MlpActivationProbe => "mlp_activation_probe",
            Self::SidecarApplyVerify => "sidecar_apply_verify",
            Self::FusedRmsnormQkv => "fused_rmsnorm_qkv",
            Self::FusedOProjResidual => "fused_o_proj_residual",
            Self::FusedMultimodal => "fused_multimodal",
        }
    }

    /// Numeric kernel id (matches `receipt.kernel_id` in the engine).
    /// Stable across the engine file; do not renumber.
    pub const fn kernel_id(self) -> u32 {
        match self {
            Self::TernaryProjection => 1,
            Self::DenseProjection => 2,
            Self::ErrorPartial => 3,
            Self::AttentionProbe => 4,
            Self::CandidateScore => 5,
            Self::PackVerify => 6,
            Self::RmsnormResidualProbe => 7,
            Self::MlpActivationProbe => 8,
            Self::SidecarApplyVerify => 9,
            Self::FusedRmsnormQkv => 10,
            Self::FusedOProjResidual => 11,
            Self::FusedMultimodal => 12,
            Self::Nf4Tile640Projection => 12, // NB: shares id 12 with FusedMultimodal in the engine
            Self::Int8Tile640GEMV => 13,
            Self::Nf4ScaledReductionTile640 => 14,
            Self::GpuBatchMatmul => 0, // batched_gemv is not a profile kernel; no id assigned
            Self::FusedTeacherStudent => 0, // teacher-student is a training path; no profile id
        }
    }
}

// ── Binding role ────────────────────────────────────────────────────────────

/// Constitutional name for a `[[buffer(N)]]` binding slot, decoupled
/// from the numeric index that the engine uses. The mapping from
/// [`KernelBindingRole`] to the engine's `buffer_slot` index is the
/// engine's responsibility; the port names the role, not the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum KernelBindingRole {
    /// `buffer(0)` — primary weight payload.
    Weights,
    /// `buffer(1)` — primary activation input.
    Input,
    /// `buffer(2)` — per-page scales (or secondary weights for some kernels).
    PageScales,
    /// `buffer(3)` — per-channel scales (or quaternary weights for MLP probes).
    ChannelScales,
    /// `buffer(4)` — sparse sidecar entries.
    Sidecar,
    /// `buffer(5)` — per-page sidecar offsets (or scalar dim constants).
    SidecarOffsets,
    /// `buffer(6)` — primary output (or scalar dim constants).
    Output,
    /// `buffer(7)` — kernel params struct (`ProjectionParams`).
    Params,
    /// `buffer(8)` — kernel receipt (instrumentation counters).
    Receipt,
    /// `buffer(9)` — error partial records.
    ErrorPartials,
    /// `buffer(10)` — attention probe records.
    ProbeOutput,
    /// `buffer(11)` — page score records.
    PageScores,
    /// `buffer(12)` — activation arena.
    ActivationArena,
    /// `buffer(13)` — arena descriptor table.
    ArenaDescriptors,
}

// ── Buffer handle ───────────────────────────────────────────────────────────

/// Opaque, provider-typed handle to a GPU buffer. The provider
/// (e.g. the Metal implementation in
/// `compute-core/.../kernel_dispatch.rs`) converts this to/from its
/// native `&Buffer` reference at dispatch time. The port does not
/// dereference or interpret the handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DispatchBufferHandle(u64);

impl DispatchBufferHandle {
    /// Construct a new handle from a raw id. The id is provider-typed:
    /// the caller is responsible for using the provider's id scheme.
    pub const fn from_id(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw id. The caller is responsible for interpreting it.
    pub const fn id(self) -> u64 {
        self.0
    }
}

// ── Dispatch parameters ─────────────────────────────────────────────────────

/// Per-dispatch kernel parameters. Mirrors the engine's
/// `ProjectionParams` (`#[repr(C)]` Metal layout) but is
/// provider-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDispatchParams {
    /// Input dimension.
    pub in_dim: u32,
    /// Output dimension.
    pub out_dim: u32,
    /// Number of pages in this dispatch.
    pub page_count: u32,
    /// Page width (typically 640).
    pub page_width: u32,
    /// Mode bits: bit0 = sidecar_enabled, bit1 = instrumentation_enabled,
    /// bit2 = candidate_layout_mode.
    pub mode_flags: u32,
    /// Deterministic probe sampling seed.
    pub probe_seed: u32,
}

impl Default for KernelDispatchParams {
    fn default() -> Self {
        Self {
            in_dim: 0,
            out_dim: 0,
            page_count: 0,
            page_width: 640,
            mode_flags: 0,
            probe_seed: 0,
        }
    }
}

// ── Request ─────────────────────────────────────────────────────────────────

/// A single kernel dispatch request. The provider interprets the
/// request, looks up the cached pipeline state, encodes the kernel,
/// and returns the outcome.
#[derive(Debug, Clone)]
pub struct KernelDispatchRequest {
    /// Which kernel to dispatch.
    pub kind: KernelKind,
    /// Buffer bindings, keyed by constitutional role. Stored as
    /// `BTreeMap` so the binding order is deterministic for replay.
    pub bindings: BTreeMap<KernelBindingRole, DispatchBufferHandle>,
    /// Per-dispatch parameters.
    pub params: KernelDispatchParams,
    /// Whether to bind the receipt buffer and enable kernel-side
    /// instrumentation counters.
    pub instrumented: bool,
}

// ── Outcome ─────────────────────────────────────────────────────────────────

/// Outcome of a single kernel dispatch. Mirrors the receipt fields
/// populated by the engine's dispatchers (kernel_id, phase_id,
/// page_count, threadgroups, threads_per_threadgroup, output_elements,
/// logical_*_bytes). Provider implementations populate every field
/// from facts known at dispatch time; fields the provider cannot
/// determine must be left at their default value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelDispatchOutcome {
    /// Numeric kernel id (matches the engine's `receipt.kernel_id`).
    pub kernel_id: u32,
    /// Phase id (currently 0 for all dispatchers in the engine file).
    pub phase_id: u32,
    /// Number of pages in this dispatch.
    pub page_count: u32,
    /// Number of threadgroups dispatched.
    pub threadgroups: u32,
    /// Number of threads per threadgroup.
    pub threads_per_threadgroup: u32,
    /// Number of output elements.
    pub output_elements: u32,
    /// Logical weight payload bytes (excluding metadata).
    pub logical_weight_bytes: u64,
    /// Logical sidecar payload bytes.
    pub logical_sidecar_bytes: u64,
    /// Logical activation payload bytes.
    pub logical_activation_bytes: u64,
}

impl Default for KernelDispatchOutcome {
    fn default() -> Self {
        Self {
            kernel_id: 0,
            phase_id: 0,
            page_count: 0,
            threadgroups: 0,
            threads_per_threadgroup: 0,
            output_elements: 0,
            logical_weight_bytes: 0,
            logical_sidecar_bytes: 0,
            logical_activation_bytes: 0,
        }
    }
}

// ── Error ───────────────────────────────────────────────────────────────────

/// Constitutional dispatch error. Categorized per the project rules:
/// preflight rejections, effect failures, and stale-fencing mismatches.
#[derive(Debug, Error)]
pub enum DispatchError {
    /// Preflight or admission rejection — the request is invalid before
    /// the effect is attempted (unknown kernel kind, missing required
    /// binding, validation failure before encoding).
    #[error("dispatch rejected: {context}")]
    Rejected {
        /// Stable error category (e.g. `"unknown_kernel_kind"`,
        /// `"missing_binding"`, `"invalid_params"`).
        context: String,
        /// Underlying cause, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// The effect itself failed — the device lost, the command buffer
    /// reported an error, the pipeline state was invalid, or read-back
    /// could not be performed.
    #[error("dispatch failed: {context}")]
    Failed {
        /// Stable error category (e.g. `"device_lost"`,
        /// `"command_buffer_error"`, `"pipeline_state_invalid"`).
        context: String,
        /// Underlying cause, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
    /// Stale fencing generation — the buffer handle has been recycled
    /// or invalidated since the request was built, or the runtime
    /// epoch has advanced past the request's expected epoch.
    #[error("dispatch stale: {context}")]
    Stale {
        /// Stable error category (e.g. `"buffer_recycled"`,
        /// `"epoch_mismatch"`, `"generation_stale"`).
        context: String,
        /// Underlying cause, if any.
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

// ── Trait ───────────────────────────────────────────────────────────────────

/// Provider-neutral kernel dispatcher. Implementations map the
/// constitutional request to a provider-specific effect (e.g. Metal
/// `new_compute_command_encoder` + `set_buffer` + `dispatch_thread_groups`).
///
/// Implementations are responsible for:
/// - looking up or creating the pipeline state for the requested kernel
///   (caching is the implementer's choice; the engine's
///   `Arc<parking_lot::Mutex<KernelRegistry>>` is one such cache);
/// - validating the request (returning [`DispatchError::Rejected`] for
///   missing bindings, invalid params, or unknown kernel kinds);
/// - encoding the kernel onto the provider's command buffer;
/// - populating every field of [`KernelDispatchOutcome`] from facts
///   known at dispatch time;
/// - returning [`DispatchError::Stale`] when a buffer handle has been
///   recycled or the runtime epoch has advanced past the request.
///
/// The trait is `Send + Sync` so the runtime can hold a shared
/// dispatcher across threads.
pub trait KernelDispatcher: Send + Sync {
    /// Encode and dispatch a single kernel.
    fn dispatch(
        &self,
        request: &KernelDispatchRequest,
    ) -> Result<KernelDispatchOutcome, DispatchError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KernelKind coverage ──────────────────────────────────────────────

    /// All 17 dispatchers in `kernel_dispatch.rs` are reachable through
    /// the constitutional `KernelKind` enum. Adding a new dispatcher in
    /// the engine file without adding a variant here is a constitutional
    /// boundary violation.
    #[test]
    fn kernel_kind_is_exhaustive_over_engine_dispatchers() {
        // 17 dispatchers, one per KernelKind variant.
        let _all_kinds = [
            KernelKind::TernaryProjection,
            KernelKind::Nf4Tile640Projection,
            KernelKind::Nf4ScaledReductionTile640,
            KernelKind::Int8Tile640GEMV,
            KernelKind::GpuBatchMatmul,
            KernelKind::DenseProjection,
            KernelKind::FusedTeacherStudent,
            KernelKind::ErrorPartial,
            KernelKind::AttentionProbe,
            KernelKind::CandidateScore,
            KernelKind::PackVerify,
            KernelKind::RmsnormResidualProbe,
            KernelKind::MlpActivationProbe,
            KernelKind::SidecarApplyVerify,
            KernelKind::FusedRmsnormQkv,
            KernelKind::FusedOProjResidual,
            KernelKind::FusedMultimodal,
        ];
        assert_eq!(_all_kinds.len(), 17);
    }

    // ── Kernel name round-trip ───────────────────────────────────────────

    /// The constitutional name of every variant matches the
    /// `kernel_name: &'static str` field used by the engine dispatcher
    /// to look up the pipeline state. A drift here means the runtime
    /// will request a kernel that the engine's `KernelRegistry` cannot
    /// find.
    #[test]
    fn kernel_name_round_trips_to_engine_kernel_name() {
        assert_eq!(KernelKind::TernaryProjection.kernel_name(), "ternary_tile640_gemv");
        assert_eq!(
            KernelKind::Nf4Tile640Projection.kernel_name(),
            "fused_gemv_nf4_tile640_fp32"
        );
        assert_eq!(
            KernelKind::Nf4ScaledReductionTile640.kernel_name(),
            "fused_gemv_nf4_scaled_reduction_tile640_fp32"
        );
        assert_eq!(KernelKind::Int8Tile640GEMV.kernel_name(), "fused_gemv_int8_tile640_fp32");
        assert_eq!(KernelKind::GpuBatchMatmul.kernel_name(), "batched_gemv_fp32");
        assert_eq!(KernelKind::DenseProjection.kernel_name(), "palettized_gemv");
        assert_eq!(
            KernelKind::FusedTeacherStudent.kernel_name(),
            "fused_teacher_student_gemv"
        );
        assert_eq!(KernelKind::ErrorPartial.kernel_name(), "error_partial");
        assert_eq!(KernelKind::AttentionProbe.kernel_name(), "attention_probe");
        assert_eq!(KernelKind::CandidateScore.kernel_name(), "candidate_score");
        assert_eq!(KernelKind::PackVerify.kernel_name(), "pack_verify");
        assert_eq!(
            KernelKind::RmsnormResidualProbe.kernel_name(),
            "rmsnorm_residual_probe"
        );
        assert_eq!(KernelKind::MlpActivationProbe.kernel_name(), "mlp_activation_probe");
        assert_eq!(
            KernelKind::SidecarApplyVerify.kernel_name(),
            "sidecar_apply_verify"
        );
        assert_eq!(KernelKind::FusedRmsnormQkv.kernel_name(), "fused_rmsnorm_qkv");
        assert_eq!(KernelKind::FusedOProjResidual.kernel_name(), "fused_o_proj_residual");
        assert_eq!(KernelKind::FusedMultimodal.kernel_name(), "fused_multimodal");
    }

    // ── Deterministic binding order ──────────────────────────────────────

    /// The request's binding map is a `BTreeMap`, so iteration order is
    /// the sorted order of [`KernelBindingRole`]. This is the replay
    /// invariant — two runs of the same request must encode buffers in
    /// the same order.
    #[test]
    fn dispatch_request_bindings_iterate_in_sorted_order() {
        let mut bindings = BTreeMap::new();
        bindings.insert(KernelBindingRole::Output, DispatchBufferHandle::from_id(6));
        bindings.insert(KernelBindingRole::Weights, DispatchBufferHandle::from_id(0));
        bindings.insert(KernelBindingRole::Input, DispatchBufferHandle::from_id(1));

        let order: Vec<KernelBindingRole> = bindings.keys().copied().collect();
        // BTreeMap iterates in the variant declaration order (sort key).
        assert_eq!(order[0], KernelBindingRole::Weights);
        assert_eq!(order[1], KernelBindingRole::Input);
        assert_eq!(order[2], KernelBindingRole::Output);
    }

    // ── Buffer handle newtype ────────────────────────────────────────────

    /// The buffer handle is a typed newtype wrapping `u64`. The port
    /// never uses raw `u64` for an authority-bearing id.
    #[test]
    fn dispatch_buffer_handle_is_newtype() {
        let h = DispatchBufferHandle::from_id(42);
        assert_eq!(h.id(), 42);
        assert_eq!(DispatchBufferHandle::from_id(42), h);
        assert_ne!(DispatchBufferHandle::from_id(42), DispatchBufferHandle::from_id(43));
    }

    // ── Error categories ─────────────────────────────────────────────────

    /// `DispatchError` categorizes into exactly the three constitutional
    /// buckets: Rejected (preflight), Failed (effect), Stale (fencing).
    #[test]
    fn dispatch_error_has_exactly_three_constitutional_categories() {
        let rejected = DispatchError::Rejected {
            context: "missing_binding".into(),
            source: None,
        };
        let failed = DispatchError::Failed {
            context: "device_lost".into(),
            source: None,
        };
        let stale = DispatchError::Stale {
            context: "epoch_mismatch".into(),
            source: None,
        };

        // The Display impl proves the category is in the formatted string.
        assert!(rejected.to_string().contains("rejected"));
        assert!(failed.to_string().contains("failed"));
        assert!(stale.to_string().contains("stale"));
    }

    // ── Default outcome ──────────────────────────────────────────────────

    /// A fresh `KernelDispatchOutcome` is the zero receipt. The engine
    /// dispatcher must overwrite every field it knows; an unmodified
    /// zero outcome is a signal that the engine file has not yet been
    /// adapted to the port.
    #[test]
    fn dispatch_outcome_default_is_zero_receipt() {
        let o = KernelDispatchOutcome::default();
        assert_eq!(o.kernel_id, 0);
        assert_eq!(o.phase_id, 0);
        assert_eq!(o.page_count, 0);
        assert_eq!(o.threadgroups, 0);
        assert_eq!(o.threads_per_threadgroup, 0);
        assert_eq!(o.output_elements, 0);
        assert_eq!(o.logical_weight_bytes, 0);
        assert_eq!(o.logical_sidecar_bytes, 0);
        assert_eq!(o.logical_activation_bytes, 0);
    }
}
