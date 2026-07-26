//! Calibration harness for ternary codec quantization admission.
//!
//! Provides:
//! - `TernaryScaleCalibrator` — sweeps group sizes, learning rates, and
//!   regularization penalties to find the optimal ternary quantization recipe.
//! - `TernaryCalibrationLoss` — composite loss metrics from calibration runs.
//! - `ProgressiveTernaryRecipe` — multi-stage distillation recipe for ternary
//!   codec candidates.
//! - `CalibrationExecutionEnvironment` — where calibration executes.
//! - `DistillationMode` — how distillation is applied during calibration.

use crate::execution_plan::CodecFamily;
use serde::{Deserialize, Serialize};

/// Environment where calibration execution takes place.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationExecutionEnvironment {
    /// Calibration runs on the local Mac (ANE/GPU via Metal).
    LocalMac,
    /// Calibration runs on a remote GPU server.
    RemoteGpu,
    /// Calibration runs on a compute cluster.
    Cluster,
}

/// Distillation strategy applied during ternary calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistillationMode {
    /// No distillation; calibrate directly from reference.
    None,
    /// Match block-level activations between teacher and student.
    BlockActivationMatching,
    /// Full logit-level distillation (unsupported on local Mac hardware).
    FullLogitDistillationUnsupportedOnLocal,
    /// Only purified ops-d metadata is used for distillation.
    PurifiedOpsdMetadataOnly,
}

/// Hyper-parameter sweep space for ternary scale calibration.
///
/// The calibrator explores combinations of group sizes, scale learning
/// rates, and regularization penalties to find the quantization recipe
/// that minimises calibration loss for a given tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryScaleCalibrator {
    /// Candidate group sizes to sweep (e.g., [32, 64, 128, 256]).
    pub group_sizes: Vec<usize>,
    /// Scale learning rates to try for each group size.
    pub scale_learning_rates: Vec<f32>,
    /// L1/L2 outlier penalty coefficients.
    pub outlier_penalties: Vec<f32>,
    /// Sparsity-inducing regularization coefficients.
    pub sparsity_penalties: Vec<f32>,
    /// Maximum calibration steps per candidate.
    pub max_steps: usize,
    /// Batch limit for the calibration data (0 = use all).
    pub calibration_batch_limit: usize,
    /// Where this calibrator's execution runs.
    pub execution_environment: CalibrationExecutionEnvironment,
}

/// Composite loss metrics from a single ternary calibration run.
///
/// Each field captures a different aspect of quantization quality,
/// enabling Pareto-based candidate selection in the admission pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCalibrationLoss {
    /// Normalised RMSE between operator outputs (reference vs quantised).
    pub operator_nrmse: f64,
    /// Cosine similarity loss in the output activation space [0, 1] where
    /// 0 = identical direction, 1 = orthogonal.
    pub output_cosine_loss: f64,
    /// L2 shift in activation distributions after quantisation.
    pub activation_shift_l2: f64,
    /// Maximum per-channel activation shift across all channels.
    pub max_channel_shift: f64,
    /// L1 proxy for sparsity achieved by the ternary encoding.
    pub sparsity_l1_proxy: f64,
    /// Regularisation penalty from scale learning (to prevent overfitting).
    pub scale_regularization: f64,
}

/// A progressive multi-stage recipe for ternary codec refinement.
///
/// The recipe transitions through codec families (e.g., NF4 → Int8 →
/// Ternary1_58) with optional distillation, scale tuning, and norm
/// correction, gated on admission receipt fulfilment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveTernaryRecipe {
    /// Unique identifier for this recipe (e.g., "ternary-gemma-2b-v1").
    pub recipe_id: String,
    /// Codec family used in stage 1 (coarse initial quantisation).
    pub stage_1_codec: CodecFamily,
    /// Codec family used in stage 2 (intermediate refinement).
    pub stage_2_codec: CodecFamily,
    /// Whether to run scale fine-tuning in stage 3.
    pub stage_3_scale_tuning: bool,
    /// Whether norm correction is applied after each stage.
    pub norm_correction_enabled: bool,
    /// Distillation strategy used throughout the recipe.
    pub distillation_mode: DistillationMode,
    /// Receipt ids that must be satisfied before promotion.
    pub promotion_required_receipts: Vec<String>,
}
