//! CImage kernel validation matrix — numerical equivalence, layout
//! equivalence, determinism, and bounds-safety tests for distilled
//! Metal kernels.
//!
//! This module owns the canonical authority for the post-emission
//! validation matrix. The pattern is the same as the engine's
//! `compile/validation_matrix.rs`: for each kernel, run a targeted
//! set of tests — numerical equivalence against a CPU reference,
//! layout equivalence, determinism, bounds safety, sidecar modes,
//! and memory admissibility. The matrix is the *primary verification
//! record* the runtime uses to decide whether a kernel is safe to
//! dispatch.
//!
//! # Module layout
//!
//! The validation matrix surface is split by authority:
//!
//! - [`result`] owns the [`ValidationResult`] and [`ValidationMatrix`]
//!   types and the `KernelName` / `TestName` newtypes.
//! - [`validators`] owns the per-kernel validator functions
//!   (`validate_ternary_projection`, `validate_dense_projection`,
//!   etc.) and the per-validator test runners.
//! - [`run`] owns the top-level entry points
//!   ([`run_validation_matrix`], [`run_validation_results`]).
//!
//! This file is the directory index and re-exports the public
//! surface.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod result;
pub mod run;
pub mod validators;

pub use result::{TestName, ValidationMatrix, ValidationResult};
pub use run::{run_validation_matrix, run_validation_results};

/// Kernel name newtype — the stable identity of a kernel under
/// validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KernelName(pub String);

impl KernelName {
    /// Construct a new [`KernelName`].
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The string form of the kernel name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for KernelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Per-crate error type for the cimage validation matrix.
#[derive(Debug, Error)]
pub enum CImageValidationError {
    #[error("rejected: {0}")]
    Rejected(String),

    #[error("failed: {0}")]
    Failed(String),
}

impl CImageValidationError {
    /// Construct a `Rejected` variant.
    pub fn rejected(message: impl Into<String>) -> Self {
        Self::Rejected(message.into())
    }

    /// Construct a `Failed` variant.
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(message.into())
    }
}

/// Result alias for the cimage validation matrix.
pub type CImageValidationResult<T> = Result<T, CImageValidationError>;

#[cfg(test)]
mod tests;
