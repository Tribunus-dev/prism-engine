//! Training target gates, status, and methods.
//!
//! These types describe the acceptance criteria (gates) a training target
//! must pass, the status lifecycle, failure modes, loss terms, and the
//! supported quantization-aware training methods.

use serde::{Deserialize, Serialize};

use super::spec::ActivationWeightedObjective;

// ── WeightTrainingGates ────────────────────────────────────────────────

/// Acceptance gates for a weight training target.
///
/// Each gate is `Option<f64>` — when `None` the gate is not enforced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTrainingGates {
    /// Maximum weight-space NRMSE relative to the source.
    pub max_weight_nrmse: Option<f64>,
    /// Maximum fraction of weights collapsed to zero after quantisation.
    pub max_zero_collapse_ratio: Option<f64>,
    /// Maximum operator-level NRMSE (applied per-layer).
    pub max_operator_nrmse: Option<f64>,
    /// Minimum cosine similarity between operator outputs.
    pub min_operator_cosine: Option<f64>,
    /// Maximum absolute error at any single operator output position.
    pub max_operator_abs_error: Option<f64>,
    /// Minimum byte savings ratio vs. the source codec.
    pub min_byte_savings_ratio: Option<f64>,
    /// Minimum evidence level required before this gate set is satisfied.
    pub required_evidence_level: RequiredEvidenceLevel,
}

impl Default for WeightTrainingGates {
    fn default() -> Self {
        Self {
            max_weight_nrmse: None,
            max_zero_collapse_ratio: None,
            max_operator_nrmse: None,
            min_operator_cosine: None,
            max_operator_abs_error: None,
            min_byte_savings_ratio: None,
            required_evidence_level: RequiredEvidenceLevel::WeightSpace,
        }
    }
}

// ── RequiredEvidenceLevel ──────────────────────────────────────────────

/// How much evidence must be collected before a gate set is satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequiredEvidenceLevel {
    /// Weight-space metrics only (NRMSE, collapse ratio).
    WeightSpace,
    /// Synthetic operator evaluation (random inputs, no real data).
    SyntheticOperator,
    /// Hardware operator evaluation (real device inference).
    HardwareOperator,
    /// Full model-quality evaluation (PPL, KL, top-1 agreement).
    ModelQuality,
    /// Runtime-profiled evidence (latency, memory, thermals).
    RuntimeProfiled,
    /// Produced from a staged production rollout.
    ProductionPromoted,
}

// ── TrainingTargetStatus ───────────────────────────────────────────────

/// Lifecycle status of a training target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingTargetStatus {
    /// Spec created but training has not started.
    Draft,
    /// All required data is available and training is ready.
    ReadyForTraining,
    /// Not enough evidence has been collected to evaluate.
    EvidenceIncomplete,
    /// Some gates pass, others fail — target is partially satisfied.
    PartiallySatisfied,
    /// All active gates pass — target is fully satisfied.
    Satisfied,
    /// One or more mandatory gates have failed.
    Failed,
}

// ── TrainingFailureMode ────────────────────────────────────────────────

/// Specific reason a training target failed a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingFailureMode {
    /// Weight-space NRMSE exceeded the configured threshold.
    WeightNrmseTooHigh,
    /// Zero-collapse ratio exceeded the configured threshold.
    ZeroCollapseTooHigh,
    /// Operator NRMSE exceeded the configured threshold.
    OperatorNrmseTooHigh,
    /// Operator cosine similarity fell below the configured threshold.
    OperatorCosineTooLow,
    /// Operator absolute tail error exceeded the configured threshold.
    OperatorAbsTailTooHigh,
    /// Byte savings ratio fell below the configured threshold.
    ByteSavingsTooLow,
    /// Activation profile data was missing for activation-weighted training.
    ActivationProfileMissing,
    /// Hardware evidence (operator / runtime) was unavailable.
    HardwareEvidenceMissing,
    /// Rollout evidence (model-quality / production) was unavailable.
    RolloutEvidenceMissing,
    /// Quality drift (PPL, KL) exceeded acceptable bounds.
    QualityDriftTooHigh,
    /// Runtime health metrics (memory, thermals, latency) failed.
    RuntimeHealthFailed,
}

// ── TargetedLossTerm ───────────────────────────────────────────────────

/// A specific loss term to target during training to address a failure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetedLossTerm {
    /// Penalise weights collapsing to zero.
    ReduceZeroCollapse,
    /// Reduce reconstruction error at the weight level.
    ReduceWeightReconstructionError,
    /// Reduce error weighted by activation magnitudes.
    ReduceActivationWeightedError,
    /// Reduce tail error at individual operator output positions.
    ReduceOperatorTailError,
    /// Preserve the direction of hidden-state vectors.
    PreserveHiddenDirection,
    /// Preserve the top-K logit ranking.
    PreserveLogitTopK,
    /// Preserve attention-score distributions.
    PreserveAttentionScores,
    /// Increase the acceptance rate of speculative decoding drafts.
    IncreaseDraftAcceptance,
}

// ── QuantTrainingMethod ────────────────────────────────────────────────

/// Quantization-aware training method to apply to a weight target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantTrainingMethod {
    /// Straight-through estimator on shadow (full-precision) weights.
    ShadowWeightsSte,
    /// Gradually reduce bit-width over a schedule.
    GradualBitTransition {
        /// Starting effective bit-width (e.g. 16.0).
        start_bits: f32,
        /// Target effective bit-width (e.g. 4.0).
        target_bits: f32,
        /// Number of training steps over which the transition occurs.
        schedule_steps: usize,
    },
    /// Soft ternarisation with learnable modulation.
    SoftTernarization {
        /// Initial temperature for the soft-sign function.
        temperature_start: f32,
        /// Final temperature after annealing.
        temperature_end: f32,
        /// Whether a per-channel learnable modulation factor is used.
        learnable_modulation: bool,
    },
    /// Activation-weighted objective — requires a profile.
    ActivationWeighted {
        /// Whether runtime profiling data is required.
        profile_required: bool,
        /// The activation-weighted objective parameters.
        objective: ActivationWeightedObjective,
    },
}
