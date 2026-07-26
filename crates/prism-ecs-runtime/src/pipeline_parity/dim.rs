//! Tensor dimension and contract types — the building blocks of every
//! [`PhaseContract`](super::contract::PhaseContract).
//!
//! [`Dim`] distinguishes concrete sizes from named symbolic
//! dimensions; two `Dim::Symbol`s with the same name refer to the
//! same dimension across the phase catalog. [`TensorRole`] labels
//! each tensor as primary input, secondary input, or output.
//! [`TensorContract`] pairs the role with a name, shape pattern, and
//! element type.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

/// A dimension in a shape pattern — either a concrete size, a named
/// symbol whose binding is shared across phases in the same model, or
/// a wildcard.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Concrete dimension size (e.g. `Dim::Known(1)` for batch dim).
    Known(i64),
    /// Named symbolic dimension. Two `Dim::Symbol`s with the same name
    /// refer to the same dimension (e.g. `"hidden_dim"` in input and
    /// output patterns of QkvProjection).
    Symbol(&'static str),
    /// Unconstrained — matches any size.
    Any,
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Known(n) => write!(f, "{n}"),
            Dim::Symbol(name) => write!(f, "{{{name}}}"),
            Dim::Any => write!(f, "*"),
        }
    }
}

/// The role a tensor plays in a phase — where it enters the compute
/// graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorRole {
    /// Primary input (the main activation tensor for this phase).
    PrimaryInput,
    /// Secondary input (residual, mask, cache, bias).
    SecondaryInput,
    /// Output tensor produced by this phase.
    Output,
}

impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorRole::PrimaryInput => write!(f, "primary_input"),
            TensorRole::SecondaryInput => write!(f, "secondary_input"),
            TensorRole::Output => write!(f, "output"),
        }
    }
}

/// Description of one tensor entering or leaving a pipeline phase.
#[derive(Debug, Clone, Copy)]
pub struct TensorContract {
    /// Canonical name (e.g. "q", "k", "v", "hidden", "residual", "mask", "logits").
    pub name: &'static str,
    /// Role in the phase.
    pub role: TensorRole,
    /// Shape pattern (e.g. `[Dim::Known(1), Dim::Symbol("hidden_dim")]`).
    pub shape_pattern: &'static [Dim],
    /// Element type (e.g. "float32").
    pub dtype: &'static str,
}

impl fmt::Display for TensorContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape_pattern.iter().map(|d| d.to_string()).collect();
        write!(
            f,
            "{}: {}[{}] ({})",
            self.name,
            self.role,
            dims.join(","),
            self.dtype
        )
    }
}
