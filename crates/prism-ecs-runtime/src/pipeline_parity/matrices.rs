//! Per-backend support matrices and the
//! [`BackendPhaseSupportMatrix`] type that holds them.
//!
//! Each matrix maps every [`PipelinePhase`](super::phase::PipelinePhase)
//! to a [`PhaseSupportStatus`](super::support::PhaseSupportStatus).
//! The runtime kernel reads these matrices when admitting a dispatch;
//! the matrices themselves are evidence, not authority.
//!
//! The 4 constructors ([`coreai_support_matrix`], [`mlx_support_matrix`],
//! [`accelerate_support_matrix`], [`reference_support_matrix`]) name
//! the four backends catalogued by the parity contract. New backends
//! must be added as new constructors — they do not mutate the existing
//! matrices.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use super::phase::PipelinePhase;
use super::support::{PendingCode, PhaseSupportStatus, UnsupportedCode};
use super::BackendId;

/// Per-backend support matrix mapping every pipeline phase to a support status.
#[derive(Debug, Clone)]
pub struct BackendPhaseSupportMatrix {
    /// Backend identifier.
    pub backend: BackendId,
    /// Per-phase support status, one entry per phase in discriminant order.
    pub phases: Vec<(PipelinePhase, PhaseSupportStatus)>,
}

impl BackendPhaseSupportMatrix {
    /// Return the support status for a specific phase.
    pub fn support_for(&self, phase: PipelinePhase) -> Option<&PhaseSupportStatus> {
        self.phases
            .iter()
            .find(|(p, _)| *p == phase)
            .map(|(_, s)| s)
    }

    /// True if every phase in the matrix is either `Native` or
    /// `Composed` (no `Unsupported`/`Pending` gaps).
    pub fn is_fully_covered(&self) -> bool {
        self.phases.iter().all(|(_, s)| {
            matches!(
                s,
                PhaseSupportStatus::Native | PhaseSupportStatus::Composed
            )
        })
    }
}

/// Core ML backend support matrix.
///
/// Core ML is an opaque compiled runtime. Static-shape matmul/projection
/// phases compile cleanly (Native). Dynamic-shape phases (KvRead, MaskApply)
/// are unsupported without a static subgraph boundary. Attention phases are
/// pending until the MIL compile path is stable and the predict bridge is
/// fully qualified. Sampling is a host-runtime responsibility.
pub fn coreai_support_matrix() -> BackendPhaseSupportMatrix {
    use PhaseSupportStatus::*;
    use UnsupportedCode::*;
    use PendingCode::*;
    BackendPhaseSupportMatrix {
        backend: BackendId::CoreAi,
        phases: vec![
            (PipelinePhase::TokenEmbedding, Pending { code: MilOpNotWired, reason: "embedding lookup not yet wired through MIL path" }),
            (PipelinePhase::PositionEncodingOrRope, Pending { code: MilOpNotWired, reason: "RoPE not yet compiled via MIL" }),
            (PipelinePhase::QkvProjection, Native),
            (PipelinePhase::KvRead, Unsupported { code: StatefulBoundary, reason: "KV cache is dynamic/stateful; Core ML static model boundary" }),
            (PipelinePhase::KvWrite, Unsupported { code: StatefulBoundary, reason: "Core ML static model cannot own cache mutation" }),
            (PipelinePhase::KvAppend, Unsupported { code: StatefulBoundary, reason: "Core ML static model cannot extend cache" }),
            (PipelinePhase::KvView, Unsupported { code: StatefulBoundary, reason: "Core ML cannot produce mutable view of runtime cache" }),
            (PipelinePhase::AttentionScores, Pending { code: BridgeNotQualified, reason: "reshape→transpose→matmul works in MIL but predict bridge not fully qualified" }),
            (PipelinePhase::MaskApply, Unsupported { code: DynamicShapeIncompatible, reason: "causal mask requires dynamic sequence dimension in compiled model" }),
            (PipelinePhase::Softmax, Composed),
            (PipelinePhase::AttentionWeightedSum, Pending { code: BridgeNotQualified, reason: "attention weighted sum not yet wired through compiled MIL path" }),
            (PipelinePhase::AttentionOutputProjection, Native),
            (PipelinePhase::ResidualAdd1, Composed),
            (PipelinePhase::Norm1, Pending { code: MilOpNotWired, reason: "RMS norm not yet wired through MIL path" }),
            (PipelinePhase::MlpGateUp, Native),
            (PipelinePhase::Activation, Composed),
            (PipelinePhase::MlpDown, Native),
            (PipelinePhase::ResidualAdd2, Composed),
            (PipelinePhase::Norm2, Pending { code: MilOpNotWired, reason: "RMS norm not yet wired through MIL path" }),
            (PipelinePhase::LmHead, Pending { code: BridgeNotQualified, reason: "LM head projection pending full 48-layer pipeline" }),
            (PipelinePhase::SamplingOrLogitsPostprocess, Unsupported { code: HostRuntimeResponsibility, reason: "sampling is host-runtime responsibility, not Core ML model output" }),
        ],
    }
}

/// MLX backend support matrix.
///
/// MLX is the broadest dynamic tensor runtime. Nearly all phases are
/// native or composed from MLX primitives. KV cache operations are
/// host-runtime-level (array reads/writes). Sampling is composed from
/// host-level operations.
pub fn mlx_support_matrix() -> BackendPhaseSupportMatrix {
    use PhaseSupportStatus::*;
    BackendPhaseSupportMatrix {
        backend: BackendId::Mlx,
        phases: vec![
            (PipelinePhase::TokenEmbedding, Composed),
            (PipelinePhase::PositionEncodingOrRope, Native),
            (PipelinePhase::QkvProjection, Native),
            (PipelinePhase::KvRead, Composed),
            (PipelinePhase::KvWrite, Composed),
            (PipelinePhase::KvAppend, Composed),
            (PipelinePhase::KvView, Composed),
            (PipelinePhase::AttentionScores, Native),
            (PipelinePhase::MaskApply, Composed),
            (PipelinePhase::Softmax, Native),
            (PipelinePhase::AttentionWeightedSum, Native),
            (PipelinePhase::AttentionOutputProjection, Native),
            (PipelinePhase::ResidualAdd1, Composed),
            (PipelinePhase::Norm1, Native),
            (PipelinePhase::MlpGateUp, Native),
            (PipelinePhase::Activation, Native),
            (PipelinePhase::MlpDown, Native),
            (PipelinePhase::ResidualAdd2, Composed),
            (PipelinePhase::Norm2, Native),
            (PipelinePhase::LmHead, Native),
            (PipelinePhase::SamplingOrLogitsPostprocess, Composed),
        ],
    }
}

/// Accelerate backend support matrix.
///
/// Accelerate provides CPU BLAS/vDSP/vForce kernels. Matmul-based phases
/// where Tribunus owns the dispatch are `Composed` (not Native), because
/// Accelerate is a kernel library, not a graph runtime. Direct kernel-
/// equivalent phases (QkvProjection, AttentionOutputProjection, MlpGateUp,
/// MlpDown, LmHead) are `Composed` — each is a single GEMM with
/// Tribunus-owned parameter setup and output marshalling.
/// Elementwise phases (Activation, ResidualAdd) use vDSP/vForce via
/// Tribunus domain adapter, classified as `Composed`.
/// Graph-level phases (attention scores, mask apply, softmax scheduling,
/// KV cache) are `Unsupported` because they need Tribunus-owned graph
/// scheduling above the BLAS layer.
pub fn accelerate_support_matrix() -> BackendPhaseSupportMatrix {
    use PhaseSupportStatus::*;
    use UnsupportedCode::*;
    BackendPhaseSupportMatrix {
        backend: BackendId::Accelerate,
        phases: vec![
            (PipelinePhase::TokenEmbedding, Unsupported { code: MissingPrimitive, reason: "no embedding primitive; needs host CPU or graph-runtime scheduler" }),
            (PipelinePhase::PositionEncodingOrRope, Unsupported { code: MissingPrimitive, reason: "RoPE not available as single Accelerate primitive" }),
            (PipelinePhase::QkvProjection, Composed),
            (PipelinePhase::KvRead, Unsupported { code: StatefulBoundary, reason: "KV cache is dynamic state; Accelerate is stateless kernel library" }),
            (PipelinePhase::KvWrite, Unsupported { code: StatefulBoundary, reason: "KV cache mutation above Accelerate kernel library surface" }),
            (PipelinePhase::KvAppend, Unsupported { code: StatefulBoundary, reason: "KV cache append is stateful; Accelerate is stateless" }),
            (PipelinePhase::KvView, Unsupported { code: StatefulBoundary, reason: "Cache view assembly requires graph-level scheduler" }),
            (PipelinePhase::AttentionScores, Unsupported { code: NeedsGraphScheduling, reason: "needs graph scheduling above BLAS to manage Q/K/V setup and batching" }),
            (PipelinePhase::MaskApply, Unsupported { code: NeedsGraphScheduling, reason: "mask broadcast needs composite graph awareness" }),
            (PipelinePhase::Softmax, Unsupported { code: NeedsGraphScheduling, reason: "softmax requires elementwise+broadcast scheduling above Accelerate surface" }),
            (PipelinePhase::AttentionWeightedSum, Unsupported { code: NeedsGraphScheduling, reason: "needs graph scheduling (QK^T result @ V) above BLAS" }),
            (PipelinePhase::AttentionOutputProjection, Composed),
            (PipelinePhase::ResidualAdd1, Composed),
            (PipelinePhase::Norm1, Unsupported { code: MissingPrimitive, reason: "RMS norm not available as single Accelerate primitive" }),
            (PipelinePhase::MlpGateUp, Composed),
            (PipelinePhase::Activation, Composed),
            (PipelinePhase::MlpDown, Composed),
            (PipelinePhase::ResidualAdd2, Composed),
            (PipelinePhase::Norm2, Unsupported { code: MissingPrimitive, reason: "RMS norm not available as single Accelerate primitive" }),
            (PipelinePhase::LmHead, Composed),
            (PipelinePhase::SamplingOrLogitsPostprocess, Unsupported { code: HostRuntimeResponsibility, reason: "sampling is host-runtime operation" }),
        ],
    }
}

/// Reference backend support matrix.
///
/// The reference evaluator supports all phases via pure-Rust
/// implementations. Every phase is `Composed` (i.e. implemented as a
/// composition of reference primitives) — there are no native kernels
/// because the reference backend is the ground truth, not a tuned
/// implementation.
pub fn reference_support_matrix() -> BackendPhaseSupportMatrix {
    use PhaseSupportStatus::*;
    BackendPhaseSupportMatrix {
        backend: BackendId::Reference,
        phases: PipelinePhase::all()
            .iter()
            .map(|&p| (p, Composed))
            .collect(),
    }
}

/// Return the support matrix for a given backend identifier.
pub fn support_matrix_for(backend: BackendId) -> BackendPhaseSupportMatrix {
    match backend {
        BackendId::CoreAi => coreai_support_matrix(),
        BackendId::Mlx => mlx_support_matrix(),
        BackendId::Accelerate => accelerate_support_matrix(),
        BackendId::Reference => reference_support_matrix(),
    }
}

/// Return KV phase support for a given backend.
///
/// - Core ML → all `Unsupported` (`StatefulBoundary`).
/// - MLX → all `Composed`.
/// - Accelerate → all `Unsupported` (`StatefulBoundary`).
/// - Reference → all `Composed`.
///
/// Returns a [`BTreeMap`] (not a `HashMap`) so iteration order is
/// observable — the constitutional rule for canonical collections.
pub fn kv_phase_support_for(backend: BackendId) -> BTreeMap<PipelinePhase, PhaseSupportStatus> {
    use PhaseSupportStatus::*;
    use UnsupportedCode::*;
    let kv_phases = [
        PipelinePhase::KvRead,
        PipelinePhase::KvWrite,
        PipelinePhase::KvAppend,
        PipelinePhase::KvView,
    ];
    let status: PhaseSupportStatus = match backend {
        BackendId::CoreAi => Unsupported {
            code: StatefulBoundary,
            reason: "KV cache is dynamic/stateful beyond Core ML static model boundary",
        },
        BackendId::Mlx => Composed,
        BackendId::Accelerate => Unsupported {
            code: StatefulBoundary,
            reason: "KV cache mutation is stateful; Accelerate is stateless kernel library",
        },
        BackendId::Reference => Composed,
    };
    kv_phases.iter().map(|&p| (p, status.clone())).collect()
}
