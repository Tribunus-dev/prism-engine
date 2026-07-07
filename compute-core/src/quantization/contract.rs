//! Quantization admission contract types.
//!
//! Defines the representation family, reconstruction contracts, validation
//! profiles, and admission pipeline types for the Nf4Tile640 codec family.
//!
//! A matrix either passes its declared validation contract or the compilation
//! fails before sealing. No degraded-quality artifact may be emitted.

/// NF4 tile640 representation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantizedMatrixFormat {
    /// Standard NF4 tile640 with tile-local reconstruction.
    Nf4Tile640Base = 1,
    /// NF4 tile640 with an FP16 reduction-axis scale sidecar.
    Nf4Tile640ScaledReductionAxis = 2,
    /// INT8 tile640 with per-tile symmetric quantization.
    Int8Tile640Base = 3,
    /// Ternary tile640: 256-element blocks, 2-bit codes, FP16 scale per block.
    TernaryTile640Base = 4,
    /// Ternary tile640 with FP16 reduction-axis scale sidecar.
    TernaryTile640ScaledReductionAxis = 5,
}

/// NF4 codebook version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Nf4CodebookVersion {
    Tile640Nf4V1 = 1,
}

/// Policy for computing reduction-axis scale vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReductionScalePolicy {
    /// S_j = max_i |W[i,j]|
    MaxAbs = 1,
}

/// Storage format for scale vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChannelScaleStorage {
    /// IEEE 754 binary16 little-endian
    F16 = 1,
}

/// Axis along which the scale vector applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScaleAxis {
    /// Scale applies to the input/reduction dimension of the matrix multiply.
    ReductionInputColumn = 1,
}

/// Tensor classification for validation profile selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorClass {
    DecoderAttentionProjection = 1,
    DecoderMlpProjection = 2,
    TokenEmbedding = 3,
    VisionPatchProjection = 4,
    CrossModalBridge = 5,
    OutputHead = 6,
    Unknown = 255,
}

/// Evidence level used during operator-space validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EvidenceLevel {
    /// Only deterministic stress vectors used (codec pathology detection).
    StressOnly = 1,
    /// Model-native prerendered activation bank used for promotion/holdout.
    PrerenderedReference = 2,
}

/// Admission class of a quantized artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArtifactAdmissionClass {
    /// Stress-only validation; suitable for development and kernel bringup.
    DiagnosticOnly = 1,
    /// Activation-bank validation with promotion/holdout; sealed production.
    ProductionQualified = 2,
}

/// Validation phase within the admission pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ProfilePhase {
    /// Used during candidate selection (StressBank or ActivationBank promotion).
    Promotion = 1,
    /// Used after a candidate passes promotion (ActivationBank holdout only).
    Holdout = 2,
}

/// Reconstruction contract for a quantized matrix.
#[derive(Debug, Clone)]
pub enum ReconstructionContract {
    /// Tile-local NF4 reconstruction with no sidecar.
    BaseNf4Tile640,
    /// NF4 reconstruction with a reduction-axis FP16 scale sidecar.
    ScaledReductionAxis {
        policy: ReductionScalePolicy,
        scale_storage: ChannelScaleStorage,
        scale_axis: ScaleAxis,
        scale_count: u32,
        epsilon_bits: u16,
    },
}

/// Validation gate thresholds per tensor class.
///
/// Each tensor class has a promotion profile (for candidate selection) and
/// optionally a stricter holdout profile (for anti-overfitting after selection).
#[derive(Debug, Clone)]
pub struct QuantizationValidationProfile {
    pub tensor_class: TensorClass,
    pub phase: ProfilePhase,
    /// Maximum allowed normalized weight RMSE (NRMSE).
    pub max_weight_nrmse: f64,
    /// Maximum allowed zero-collapse ratio.
    /// Investigation ceiling: below this, a near-miss candidate continues to
    /// operator validation with a warning. Above this, reject before operator
    /// validation. Must be >= max_weight_nrmse.
    pub investigation_nrmse_ceiling: f64,
    pub max_zero_collapse_ratio: f64,
    /// Maximum allowed operator-space normalized RMSE.
    pub max_operator_nrmse: f32,
    /// Minimum acceptable cosine similarity (mean across all test vectors).
    pub min_mean_cosine: f32,
    /// Minimum acceptable cosine similarity (worst individual vector).
    pub min_worst_cosine: f32,
    /// Maximum allowed deviation from 1.0 of |quant_norm / ref_norm|.
    pub max_norm_ratio_drift: f32,
}

/// Weight-space validation results.
#[derive(Debug, Clone)]
pub struct WeightValidationReport {
    pub rmse: f64,
    pub nrmse: f64,
    pub max_abs_error: f64,
    pub zero_collapse_ratio: f64,
}

impl WeightValidationReport {
    pub fn passes(&self, profile: &QuantizationValidationProfile) -> bool {
        self.nrmse <= profile.max_weight_nrmse
            && self.zero_collapse_ratio <= profile.max_zero_collapse_ratio
    }
    /// Three-tier admission status for weight-space validation.
    pub fn admission_status(&self, profile: &QuantizationValidationProfile, is_ternary: bool) -> WeightAdmission {
        // Ternary intentionally produces ~70% zero values — that is the expected
        // sparsity of the format, not a packer pathology.  Skip the zero-collapse
        // gate for ternary candidates; operator-space validation is the real check.
        if !is_ternary {
            if self.zero_collapse_ratio > profile.max_zero_collapse_ratio {
                return WeightAdmission::Rejected {
                    reason: format!(
                        "zeroCollapse={:.4} > max={:.4}",
                        self.zero_collapse_ratio, profile.max_zero_collapse_ratio
                    ),
                };
            }
}
        if self.nrmse > profile.investigation_nrmse_ceiling {
            return WeightAdmission::Rejected {
                reason: format!(
                    "wNRMSE={:.4} > ceiling={:.4}",
                    self.nrmse, profile.investigation_nrmse_ceiling
                ),
            };
        }
        if self.nrmse > profile.max_weight_nrmse {
            return WeightAdmission::InvestigationBand {
                warning: format!(
                    "wNRMSE={:.4} exceeds target {:.4}, within ceiling {:.4}",
                    self.nrmse, profile.max_weight_nrmse, profile.investigation_nrmse_ceiling
                ),
            };
        }
        WeightAdmission::Passed
    }
}

/// Weight-space admission status for the 3-tier system.
#[derive(Debug, Clone)]
pub enum WeightAdmission {
    /// All metrics within target thresholds.
    Passed,
    /// Weight metric exceeds target but within investigation ceiling.
    /// Candidate should continue to operator validation with a warning receipt.
    InvestigationBand { warning: String },
    /// Weight metric exceeds investigation ceiling; reject before operator validation.
    Rejected { reason: String },
}

/// Operator-space validation results.
#[derive(Debug, Clone)]
pub struct OperatorValidationReport {
    /// Absolute RMSE between reference and quantized matmul outputs.
    pub rmse: f32,
    /// Normalized RMSE: rmse / (reference_output_rms + epsilon).
    pub operator_nrmse: f32,
    /// Average cosine similarity across all test vectors.
    pub cosine_similarity: f32,
    /// Worst (minimum) cosine similarity across test vectors.
    pub worst_cosine: f32,
    /// RMS of the reference output (diagnostic context).
    pub ref_output_rms: f32,
    /// Norm ratio drift: |quant_norm / ref_norm - 1| (worst across vectors).
    pub norm_ratio_drift: f32,
    /// Fraction of output elements whose sign matches the reference.
    pub sign_agreement: f32,
}

impl OperatorValidationReport {
    pub fn passes(&self, profile: &QuantizationValidationProfile) -> bool {
        self.operator_nrmse <= profile.max_operator_nrmse
            && self.cosine_similarity >= profile.min_mean_cosine
            && self.worst_cosine >= profile.min_worst_cosine
            && self.norm_ratio_drift <= profile.max_norm_ratio_drift
    }
}

/// Batch size for deadline-checking inside validation loops.
/// Check deadline after every VALIDATION_BATCH_SIZE vectors.
pub const VALIDATION_BATCH_SIZE: usize = 8;

/// Outcome of a cancellation-aware validation pass.
pub enum ValidationOutcome {
    Completed(OperatorValidationReport),
    Interrupted(InterruptedValidationReport),
}

/// Partial metrics from an interrupted validation pass.
pub struct InterruptedValidationReport {
    pub phase: String,
    pub processed_vectors: u32,
    pub partial_rmse: f32,
    pub partial_nrmse: f32,
    pub partial_cosine: f32,
    pub partial_ref_rms: f32,
}

/// The best candidate seen so far.
#[derive(Debug, Clone)]
pub struct BestCandidateSnapshot {
    pub format: QuantizedMatrixFormat,
    pub weight_nrmse: f64,
    pub zero_collapse_ratio: f64,
    pub operator_rmse: f32,
    pub operator_nrmse: f32,
    pub cosine_similarity: f32,
    pub ref_output_rms: f32,
    pub hard_gates_passed: u32,
    pub payload_bytes: u64,
}

impl BestCandidateSnapshot {
    /// Policy-defined comparator. Higher-return value means `self` is better.
    /// Tiebreak: hard_gates_passed > cosine_similarity > operator_nrmse > payload_bytes.
    pub fn better_than(&self, other: &Self) -> bool {
        if self.hard_gates_passed != other.hard_gates_passed {
            return self.hard_gates_passed > other.hard_gates_passed;
        }
        if (self.cosine_similarity - other.cosine_similarity).abs() > 1e-6 {
            return self.cosine_similarity > other.cosine_similarity;
        }
        if (self.operator_nrmse - other.operator_nrmse).abs() > 1e-6 {
            return self.operator_nrmse < other.operator_nrmse;
        }
        self.payload_bytes < other.payload_bytes
    }
}

/// A matrix that passed admission for a specific representation.
#[derive(Debug, Clone)]
pub struct QualifiedTensor {
    pub format: QuantizedMatrixFormat,
    pub reconstruction_contract: ReconstructionContract,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    /// FP16 reduction-axis scale sidecar (None for base format).
    pub scale_vector: Option<Vec<f32>>,
    pub weight_report: WeightValidationReport,
    pub operator_report: OperatorValidationReport,
    /// How operator-space validation was performed.
    pub evidence_level: EvidenceLevel,
    /// Admission classification based on available evidence.
    pub admission_class: ArtifactAdmissionClass,
}

/// Quantization hint from the model adapter.
#[derive(Debug, Clone)]
pub struct QuantizationHint {
    pub tensor_class: TensorClass,
    /// Whether the compiler may attempt ScaledReductionAxis candidates.
    pub permit_scale_candidate: bool,
    /// Whether the compiler may attempt Int8Tile640Base candidates.
    pub permit_int8_candidate: bool,
}

/// Structured admission failure.
#[derive(Debug, Clone)]
pub enum QuantizationAdmissionFailure {
    NoCandidatePassed {
        candidates_attempted: Vec<String>,
        last_weight_nrmse: f64,
        last_zero_collapse_ratio: f64,
        last_operator_rmse: f32,
        /// Operator NRMSE of the last failing candidate (if available).
        last_operator_nrmse: f32,
        /// Cosine similarity of the last failing candidate (if available).
        last_cosine_similarity: f32,
        /// Average reference output RMS.
        last_ref_output_rms: f32,
    },
    PackerFailure(String),
    /// The per-tensor wall-clock deadline expired during validation.
    /// Carries the best candidate metrics seen so far and the phase in
    /// which time ran out, so the caller can distinguish "ternary
    /// fundamentally failed" from "this tensor needs a larger budget."
    TimeoutDeadline {
        candidates_attempted: Vec<String>,
        best_candidate: BestCandidateSnapshot,
        /// Number of validation vectors processed before timeout.
        vectors_processed: u32,
        /// Name of the phase where the deadline expired (e.g. "probe",
        /// "promotion", "holdout").
        expired_phase: String,
    },
}
