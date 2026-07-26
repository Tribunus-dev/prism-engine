//! Per-backend support status — the structured enum and reason
//! codes used to declare whether a backend can execute a given
//! [`PipelinePhase`](super::phase::PipelinePhase).
//!
//! The runtime kernel reads these statuses when admitting a
//! dispatch; the statuses themselves are evidence, not authority.
//! They do not decide canonical lifecycle transitions.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// Structured code for `PhaseSupportStatus::Unsupported` variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnsupportedCode {
    /// The required primitive operation does not exist on this backend.
    MissingPrimitive,
    /// The phase involves dynamic shapes the backend cannot compile.
    DynamicShapeIncompatible,
    /// The phase needs graph-level scheduling the backend cannot own.
    NeedsGraphScheduling,
    /// The operation is a host-runtime responsibility (e.g. sampling, cache read).
    HostRuntimeResponsibility,
    /// Stateful boundary incompatible with backend's static model contract.
    StatefulBoundary,
}

impl fmt::Display for UnsupportedCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedCode::MissingPrimitive => write!(f, "missing_primitive"),
            UnsupportedCode::DynamicShapeIncompatible => {
                write!(f, "dynamic_shape_incompatible")
            }
            UnsupportedCode::NeedsGraphScheduling => write!(f, "needs_graph_scheduling"),
            UnsupportedCode::HostRuntimeResponsibility => {
                write!(f, "host_runtime_responsibility")
            }
            UnsupportedCode::StatefulBoundary => write!(f, "stateful_boundary"),
        }
    }
}

/// Structured code for `PhaseSupportStatus::Pending` variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PendingCode {
    /// MIL operation not yet wired into the builder.
    MilOpNotWired,
    /// Native bridge compiles but is not runtime-qualified.
    BridgeNotQualified,
    /// Fence validation (eval/materialization proof) not yet integrated.
    FenceValidationPending,
    /// Graph builder adapter not yet implemented.
    GraphBuilderNotImplemented,
}

impl fmt::Display for PendingCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PendingCode::MilOpNotWired => write!(f, "mil_op_not_wired"),
            PendingCode::BridgeNotQualified => write!(f, "bridge_not_qualified"),
            PendingCode::FenceValidationPending => write!(f, "fence_validation_pending"),
            PendingCode::GraphBuilderNotImplemented => write!(f, "graph_builder_not_implemented"),
        }
    }
}

/// Backend support status for a single canonical pipeline phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseSupportStatus {
    /// Backend has a direct native kernel for this phase.
    Native,
    /// Built from supported primitives (e.g. activation = mul + sigmoid,
    /// or Tribunus-owned graph schedule above BLAS/vDSP/vForce).
    Composed,
    /// Not supported due to fundamental backend capability gap.
    Unsupported {
        code: UnsupportedCode,
        reason: &'static str,
    },
    /// Not yet implemented but primitives exist and implementation is planned.
    Pending {
        code: PendingCode,
        reason: &'static str,
    },
}

impl fmt::Display for PhaseSupportStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhaseSupportStatus::Native => write!(f, "native"),
            PhaseSupportStatus::Composed => write!(f, "composed"),
            PhaseSupportStatus::Unsupported { code, reason } => {
                write!(f, "unsupported/{code}: {reason}")
            }
            PhaseSupportStatus::Pending { code, reason } => {
                write!(f, "pending/{code}: {reason}")
            }
        }
    }
}
