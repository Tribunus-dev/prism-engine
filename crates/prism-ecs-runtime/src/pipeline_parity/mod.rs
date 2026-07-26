//! Pipeline parity — the canonical authority for the 21 inference
//! pipeline phases, per-phase tensor contracts, and per-backend
//! support matrices used to compare compute backends.
//!
//! This module owns the design idea formerly in
//! `compute-core/src/ecs/core/pipeline_parity.rs` (1,930 LOC) — the
//! 21 canonical phases that every backend must either implement or
//! explicitly reject, the per-phase tensor shape contract catalog, the
//! structured support-status enum, and the graph-family-to-phase
//! mapping used for apples-to-apples backend comparison.
//!
//! # Position in the constitutional stack
//!
//! The phase catalog and per-backend support matrices are **evidence**,
//! not authority. They describe what a backend *can* execute; they do
//! not decide canonical lifecycle transitions. A `PhaseSupportStatus`
//! of `Native` does not promote a backend to the canonical authority
//! for a phase — that authority remains with the runtime kernel, which
//! reads the matrix when admitting a dispatch.
//!
//! # Phase semantics vs. graph families
//!
//! Graph families are **test artifacts**; pipeline phases are
//! **inference semantics**. A family may map to a phase for
//! qualification (see [`grouping::graph_family_to_phase`]), but phases
//! are not labels — they carry a typed contract that defines what the
//! phase consumes, produces, tolerates, and how comparison grouping is
//! legal.
//!
//! Unmapped graph families fail closed (return `Err`). The parity
//! contract never silently assigns a fallback phase.
//!
//! # Module layout
//!
//! - [`phase`] — the [`phase::PipelinePhase`] enum (21 canonical phases)
//!   and the [`phase::ALL_PHASES`] discriminant-order list.
//! - [`contract`] — the [`contract::PhaseContract`] type and the
//!   [`contract::PHASE_CONTRACTS`] static catalog that ties every phase
//!   to its tensor inputs/outputs/tolerance.
//! - [`dim`] — the [`dim::Dim`], [`dim::TensorRole`], and
//!   [`dim::TensorContract`] types that the catalog is built from.
//! - [`support`] — the [`support::PhaseSupportStatus`] enum and its
//!   [`support::UnsupportedCode`] / [`support::PendingCode`] structured
//!   reason codes.
//! - [`matrices`] — the [`matrices::BackendPhaseSupportMatrix`] type,
//!   the per-backend constructors
//!   ([`matrices::coreai_support_matrix`], [`matrices::mlx_support_matrix`],
//!   [`matrices::accelerate_support_matrix`], [`matrices::reference_support_matrix`]),
//!   and [`matrices::kv_phase_support_for`].
//! - [`grouping`] — the [`grouping::PipelineParityError`],
//!   [`grouping::PhaseComparisonGroup`], [`grouping::PhaseComparisonRow`],
//!   [`grouping::ComparisonReceiptView`] types, the graph-family
//!   mapping functions, and the
//!   [`grouping::graph_family_to_phase`] entry point.
//! - [`BackendId`] — the parity identifier for backends (CoreAi,
//!   Accelerate, Mlx, Reference).
//!
//! # Hard rules
//!
//! - **No `unsafe`**: this module is `forbid(unsafe_code)`. It is a
//!   pure-Rust catalog.
//! - **No `HashMap`**: comparison grouping uses [`std::collections::BTreeMap`]
//!   so the iteration order is observable.
//! - **No `unwrap`/`expect` in production paths**: the few `expect`
//!   calls are confined to the `tests` module.
//! - **No `anyhow::Error`**: errors are `PipelineParityError` (a plain
//!   `Debug + Display` struct) — a typed error carrying the family
//!   name and the rejection reason.

#![forbid(unsafe_code)]

pub mod contract;
pub mod dim;
pub mod grouping;
pub mod matrices;
pub mod phase;
pub mod support;

#[cfg(test)]
mod tests;

pub use contract::{PhaseContract, PHASE_CONTRACTS};
pub use dim::{Dim, TensorContract, TensorRole};
pub use grouping::{
    graph_family_phase_variant, graph_family_semantic_contract_id, graph_family_to_phase,
    group_for_comparison, ComparisonReceiptView, PhaseComparisonGroup, PhaseComparisonRow,
    PipelineParityError,
};
pub use matrices::{
    accelerate_support_matrix, coreai_support_matrix, kv_phase_support_for, mlx_support_matrix,
    reference_support_matrix, support_matrix_for, BackendPhaseSupportMatrix,
};
pub use phase::PipelinePhase;
pub use support::{PendingCode, PhaseSupportStatus, UnsupportedCode};

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Backend identifier for parity comparison. Distinct from
/// `prism_ecs_kernel::BackendKind` (which is for kernel dispatch) —
/// this enum names the four backends whose support matrices are
/// catalogued by the parity contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    /// Apple Core ML (ANE/GPU/CPU).
    CoreAi,
    /// Apple Accelerate (vDSP/vForce/BLAS).
    Accelerate,
    /// MLX (Apple's dynamic tensor framework).
    Mlx,
    /// Reference / pure-Rust evaluator.
    Reference,
}

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendId::CoreAi => write!(f, "coreai"),
            BackendId::Accelerate => write!(f, "accelerate"),
            BackendId::Mlx => write!(f, "mlx"),
            BackendId::Reference => write!(f, "reference"),
        }
    }
}

impl FromStr for BackendId {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "coreai" => Ok(BackendId::CoreAi),
            "accelerate" => Ok(BackendId::Accelerate),
            "mlx" => Ok(BackendId::Mlx),
            "reference" => Ok(BackendId::Reference),
            other => Err(format!("unknown BackendId variant: '{other}'")),
        }
    }
}
