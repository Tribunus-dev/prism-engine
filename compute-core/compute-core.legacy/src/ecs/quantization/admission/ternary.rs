//! Ternary admission and execution receipt types.
//!
//! Structured receipts for the ternary codec admission pipeline: deadzone
//! analysis, activation shift detection, Metal kernel execution, sensitivity
//! scoring, and candidate tracking.

use crate::execution_plan::CodecFamily;
use serde::{Deserialize, Serialize};

// ── ID type aliases ────────────────────────────────────────────────────────

/// Opaque identifier for a receipt.
pub type ReceiptId = String;
/// Opaque identifier for a tensor.
pub type TensorKey = String;
/// Opaque identifier for a candidate record.
pub type CandidateId = String;
/// Digest string for a CImage artifact.
pub type CImageDigest = String;

// ── ReceiptEvidenceKind ────────────────────────────────────────────────────

/// Classifies the evidence backing a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReceiptEvidenceKind {
    /// Default / unclassified evidence.
    Default,
    /// Evidence derived from model output activations.
    ModelOutput,
    /// Evidence derived from loss derivatives / gradients.
    LossDerivative,
    /// Evidence derived from attention patterns.
    Attention,
    /// Evidence derived from layer activations.
    LayerActivation,
    /// Combined evidence from multiple sources.
    Combined,
}

// ── TernaryPromotionStatus ─────────────────────────────────────────────────

/// Promotion status of a ternary candidate through the qualification pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TernaryPromotionStatus {
    /// Research-only stage; not yet evaluated on real tensors.
    ResearchOnly,
    /// Passed synthetic tensor validation.
    SyntheticPassed,
    /// Passed validation on a real tensor.
    RealTensorPassed,
    /// Passed validation across an entire model region.
    RegionPassed,
    /// Eligible for production deployment.
    ProductionEligible,
    /// Rejected at the current stage.
    Rejected,
}

// ── ActivationCorrectionKind ───────────────────────────────────────────────

/// Recommended correction strategy when activation shift exceeds thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationCorrectionKind {
    /// Apply a per-channel mean-shift correction.
    MeanShiftCorrection,
    /// Tweak activation normalization statistics.
    NormTweak,
    /// Rescue via scale recalibration.
    ScaleRescue,
    /// Fall back to a higher-precision codec.
    HigherPrecisionFallback,
}

// ── SearchBudgetClass ──────────────────────────────────────────────────────

/// Recommended search budget for ternary exploration on a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchBudgetClass {
    /// Skip ternary entirely for this tensor.
    SkipTernary,
    /// Run a cheap, low-effort ternary probe.
    CheapTernaryProbe,
    /// Run a full ternary parameter sweep.
    FullTernarySweep,
    /// Use only mixed-precision (non-ternary) codecs.
    MixedPrecisionOnly,
}

// ── TernaryAdmissionReceipt ────────────────────────────────────────────────

/// Full admission receipt for a ternary-quantized tensor.
///
/// Records weight distribution statistics, scale properties, operator-space
/// validation metrics (NRMSE, cosine, error bounds), activation shift
/// diagnostics, and the promotion decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryAdmissionReceipt {
    /// Unique identifier for this receipt.
    pub receipt_id: ReceiptId,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Tensor class label (stored as `String` because `TensorClass` lacks
    /// serde derives in `contract.rs`).
    pub tensor_class: String,
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
    /// Group size for quantization.
    pub group_size: usize,
    /// Effective bits per weight (includes scale overhead).
    pub effective_bits_per_weight: f64,
    /// Fraction of weights quantized to zero.
    pub zero_fraction: f64,
    /// Fraction of weights quantized to -1.
    pub neg_fraction: f64,
    /// Fraction of weights quantized to +1.
    pub pos_fraction: f64,
    /// Mean of the per-group scale values.
    pub scale_mean: f64,
    /// Standard deviation of the per-group scale values.
    pub scale_std: f64,
    /// Maximum per-group scale value.
    pub scale_max: f64,
    /// Operator-space NRMSE (teacher vs student).
    pub operator_nrmse: f64,
    /// Operator-space cosine similarity (teacher vs student).
    pub output_cosine: f64,
    /// Maximum absolute error across all output elements.
    pub max_abs_error: f64,
    /// L2 norm of the activation mean shift.
    pub activation_shift_l2: f64,
    /// Maximum per-channel activation shift.
    pub max_channel_shift: f64,
    /// Fraction of groups exhibiting deadzone collapse.
    pub deadzone_collapse: f64,
    /// Fraction of groups with magnitude overflow.
    pub magnitude_overflow: f64,
    /// Aggregate activation shift risk score.
    pub activation_shift_risk: f64,
    /// Whether a rescue codec is required.
    pub rescue_required: bool,
    /// Recommended rescue codec family, if rescue is required.
    pub recommended_rescue_codec: Option<CodecFamily>,
    /// Promotion status through the qualification pipeline.
    pub promotion_status: TernaryPromotionStatus,
    /// Classification of the evidence backing this receipt.
    pub evidence_kind: ReceiptEvidenceKind,
}

// ── TernaryMetalExecutionReceipt ───────────────────────────────────────────

/// Receipt from executing a ternary kernel on Metal.
///
/// Records buffer sizes, timing, bandwidth, and validation metrics comparing
/// Metal GPU output against the CPU reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryMetalExecutionReceipt {
    /// Unique identifier for this receipt.
    pub receipt_id: ReceiptId,
    /// Digest of the CImage artifact used.
    pub cimage_digest: CImageDigest,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Name of the Metal kernel that was executed.
    pub kernel_name: String,
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
    /// Group size used for quantization.
    pub group_size: usize,
    /// Effective bits per weight (includes scale overhead).
    pub effective_bits_per_weight: f64,
    /// Total bytes of code/trit data read by the kernel.
    pub code_bytes_read: u64,
    /// Total bytes of scale data read by the kernel.
    pub scale_bytes_read: u64,
    /// Total bytes of activation input read by the kernel.
    pub activation_bytes_read: u64,
    /// Total bytes of output written by the kernel.
    pub output_bytes_written: u64,
    /// Command buffer execution time in milliseconds.
    pub command_buffer_ms: f64,
    /// Effective memory bandwidth in GB/s.
    pub effective_bandwidth_gbps: f64,
    /// NRMSE between Metal GPU output and CPU reference.
    pub metal_vs_cpu_nrmse: f64,
    /// Cosine similarity between Metal GPU output and CPU reference.
    pub metal_vs_cpu_cosine: f64,
    /// Whether all validation checks passed.
    pub validation_passed: bool,
}

// ── DeadzoneReceipt ────────────────────────────────────────────────────────

/// Receipt from deadzone analysis of ternary-quantized weights.
///
/// Captures the fraction of weights that fall into the deadzone or near its
/// threshold, and whether a dynamic bias correction is recommended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadzoneReceipt {
    /// Unique identifier for this receipt.
    pub receipt_id: ReceiptId,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Group size used for quantization.
    pub group_size: usize,
    /// Fraction of weights quantized to zero.
    pub zero_fraction: f64,
    /// Fraction of weights near the ternary threshold.
    pub near_threshold_fraction: f64,
    /// Fraction of weights trapped between thresholds.
    pub trapped_weight_fraction: f64,
    /// Whether a dynamic bias adjustment is recommended.
    pub dynamic_bias_recommended: bool,
    /// Whether the deadzone analysis passed all checks.
    pub passed: bool,
}

// ── ActivationShiftReceipt ─────────────────────────────────────────────────

/// Receipt from activation shift analysis (codec-agnostic).
///
/// Compares pre-quantization and post-quantization activation mean statistics
/// and recommends a correction strategy if the shift exceeds thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationShiftReceipt {
    /// Unique identifier for this receipt.
    pub receipt_id: ReceiptId,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Codec family used for quantization.
    pub codec: CodecFamily,
    /// L2 norm of the pre-quantization activation mean.
    pub pre_quant_mean_l2: f64,
    /// L2 norm of the post-quantization activation mean.
    pub post_quant_mean_l2: f64,
    /// L2 norm of the mean shift (pre → post).
    pub mean_shift_l2: f64,
    /// Maximum per-channel activation shift.
    pub max_channel_shift: f64,
    /// Whether a correction is recommended.
    pub correction_recommended: bool,
    /// Recommended correction strategy, if applicable.
    pub correction_kind: Option<ActivationCorrectionKind>,
}

// ── TensorSensitivityReceipt ───────────────────────────────────────────────

/// Pre-admission sensitivity receipt used to guide search budget allocation.
///
/// Captures sensitivity, salience, outlier mass, and activation shift priors
/// to determine the appropriate ternary search budget for a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSensitivityReceipt {
    /// Unique identifier for this receipt.
    pub receipt_id: ReceiptId,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Tensor class label (stored as `String` because `TensorClass` lacks
    /// serde derives in `contract.rs`).
    pub tensor_class: String,
    /// Sensitivity score for this tensor.
    pub sensitivity_score: f64,
    /// Salience score for this tensor.
    pub salience_score: f64,
    /// Fraction of weights identified as outliers.
    pub outlier_mass: f64,
    /// Prior activation shift risk estimate.
    pub activation_shift_prior: f64,
    /// Recommended search budget class.
    pub recommended_search_budget: SearchBudgetClass,
}

// ── TernaryCandidateRecord ─────────────────────────────────────────────────

/// Record of a single ternary candidate evaluated during sweep.
///
/// Tracks calibration parameters, penalty terms, and the associated receipt
/// and pass/fail status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCandidateRecord {
    /// Unique identifier for this candidate.
    pub candidate_id: CandidateId,
    /// Key identifying the tensor.
    pub tensor_key: TensorKey,
    /// Group size used for this candidate.
    pub group_size: usize,
    /// Scale initialization strategy (stored as a descriptive string).
    pub scale_init: String,
    /// Number of calibration steps performed.
    pub calibration_steps: u64,
    /// Outlier penalty term applied.
    pub outlier_penalty: f64,
    /// Sparsity penalty term applied.
    pub sparsity_penalty: f64,
    /// Receipt identifier linking to the full admission receipt.
    pub receipt_id: ReceiptId,
    /// Whether this candidate passed all validation gates.
    pub passed: bool,
}
