//! Calibration trait and result types for quantization-aware weight analysis.
//!
//! Defines the `Calibrator` trait — the core abstraction for running
//! calibration data through a model and collecting per-channel statistics
//! that guide quantization decisions.
//!
//! Two concrete implementations are provided:
//! - `AwqCalibrator` — activation magnitude profiling + transformation search
//! - `GptqCalibrator` — Hessian accumulation + Cholesky-based rounding

use crate::families::{AwqSaliency, GptqSaliency, SmoothQuantScale};
use serde::{Deserialize, Serialize};

// ── CalibrationResult ─────────────────────────────────────────────────────

/// Full result of calibrating a single model module.
///
/// Contains both AWQ-style activation scales and GPTQ-style Hessian
/// information.  The calibrator fills whichever representation(s) its
/// algorithm requires; unused fields are `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Optional per-channel activation scales (AWQ).
    pub act_scales: Option<AwqSaliency>,
    /// Optional Hessian-based saliency (GPTQ).
    pub hessian: Option<GptqSaliency>,
    /// Optional per-channel min/max activation range (SmoothQuant).
    pub act_range: Option<SmoothQuantScale>,
    /// Number of calibration samples consumed.
    pub num_samples: usize,
    /// Name of the module this result was computed for.
    pub module_name: String,
    /// Input feature count (columns of weight matrix).
    pub in_features: usize,
    /// Output feature count (rows of weight matrix).
    pub out_features: usize,
}

// ── CalibratedLayer ───────────────────────────────────────────────────────

/// A single calibrated layer with its quantization statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedLayer {
    /// Calibration data for this layer.
    pub result: CalibrationResult,
    /// The weight matrix as a flat float slice (reference, not owned).
    /// Owned clones are produced during calibration.
    pub weight: Vec<f32>,
}

// ── CalibrationSummary ────────────────────────────────────────────────────

/// Summary of calibration across all layers of a model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationSummary {
    /// Per-layer calibration results, keyed by module name.
    pub layers: Vec<CalibratedLayer>,
    /// Total number of samples consumed across all layers.
    pub total_samples: usize,
}

// ── Calibrator trait ──────────────────────────────────────────────────────

/// A calibrator samples model activations to produce quantization-relevant
/// statistics.
///
/// Implementations differ in what statistics they collect:
///
/// | Calibrator      | Collects                                       | Used by |
/// |-----------------|-------------------------------------------------|---------|
/// | `AwqCalibrator` | Activation magnitude per channel                | AWQ     |
/// | `GptqCalibrator`| Hessian matrix via outer product                | GPTQ    |
///
/// The trait is designed to be implementation-agnostic: `calibrate_module`
/// receives raw activation vectors and returns whichever statistics the
/// implementation computes.
pub trait Calibrator: Send + Sync {
    /// Calibrate a single module given its name, weight shape, and
    /// a batch of input activations (shape: `[batch_size, in_features]`).
    ///
    /// Returns the calibration result containing per-channel statistics.
    fn calibrate_module(
        &self,
        module_name: &str,
        in_features: usize,
        out_features: usize,
        activations: &[f32],
    ) -> CalibrationResult;

    /// Accumulate a new batch of activations into an existing result
    /// (online / streaming calibration).
    ///
    /// The default implementation ignores prior results and recomputes;
    /// override to implement true incremental accumulation.
    fn accumulate(
        &self,
        module_name: &str,
        in_features: usize,
        out_features: usize,
        activations: &[f32],
        prior: Option<&CalibrationResult>,
    ) -> CalibrationResult {
        match prior {
            Some(_) => {
                // Recompute from scratch (override for incremental merge).
                self.calibrate_module(module_name, in_features, out_features, activations)
            }
            None => self.calibrate_module(module_name, in_features, out_features, activations),
        }
    }

    /// Reset any internal state accumulated across calibration runs.
    fn reset(&mut self);
}

// ── CalibrationConfig ─────────────────────────────────────────────────────

/// Configuration controlling calibration behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Number of calibration samples to process (None = all).
    pub num_samples: Option<usize>,
    /// Batch size for forward passes during calibration.
    pub batch_size: usize,
    /// Whether to collect AWQ-style activation scales.
    pub collect_act_scales: bool,
    /// Whether to collect GPTQ-style Hessians.
    pub collect_hessians: bool,
    /// Whether to collect SmoothQuant-style activation ranges.
    pub collect_act_range: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_samples: None,
            batch_size: 1,
            collect_act_scales: true,
            collect_hessians: false,
            collect_act_range: false,
        }
    }
}
