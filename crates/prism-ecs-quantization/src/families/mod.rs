//! Quantization algorithm families — AWQ, GPTQ, SmoothQuant.
//!
//! Each module defines the codec-family struct (parameters, grid), a saliency
//! type capturing the importance/per-channel statistics the algorithm uses,
//! and (where applicable) migration or transformation descriptors.

pub mod awq;
pub mod gptq;
pub mod smoothquant;

pub use awq::*;
pub use gptq::*;
pub use smoothquant::*;

use serde::{Deserialize, Serialize};

/// Common quantization parameter shared by grouped codec families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupedQuantParams {
    /// Columns per quantization group (128, 64, 32, etc.).
    pub group_size: usize,
    /// Symmetric quantization (no zero-point).
    pub sym: bool,
}

impl Default for GroupedQuantParams {
    fn default() -> Self {
        Self {
            group_size: 128,
            sym: false,
        }
    }
}

/// Per-channel importance metric produced by calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceMetric {
    /// Per-channel activation magnitude (L1 mean abs).
    ActivationMagnitude,
    /// Hessian-trace or Hessian-diagonal importance.
    HessianDiag,
    /// Full Hessian matrix (GPTQ-style).
    HessianMatrix,
    /// Composite importance (imatrix-style).
    IMatrix,
}
