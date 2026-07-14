//! Quantization admission contract types.
//!
//! Defines the representation family, reconstruction contracts, validation
//! profiles, and admission pipeline types for the Nf4Tile640 codec family.
//!
//! A matrix either passes its declared validation contract or the compilation
//! fails before sealing. No degraded-quality artifact may be emitted.

// \u2500\u2500 Wire ABI versions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
/// Cimage major wire version for V1.
pub const CIMAGE_MAJOR_VERSION: u16 = 1;
/// MatrixContract wire version for V1.
pub const MATRIX_CONTRACT_WIRE_VERSION: u16 = 1;
/// ExecutionGraph wire version for V1.
pub const EXECUTION_GRAPH_WIRE_VERSION: u16 = 1;
/// Representation registry version for V1.
pub const REPRESENTATION_REGISTRY_VERSION: u16 = 1;

// \u2500\u2500 Tile geometry constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
pub const REDUCTION_TILE_SIZE: usize = 640;
pub const TERNARY_TILE640_CODE_BYTES: usize = 160;
pub const TERNARY_TILE640_METADATA_BYTES: usize = 4; // alpha F32
pub const NF4_TILE640_CODE_BYTES: usize = 320;
pub const NF4_TILE640_METADATA_BYTES: usize = 8; // alpha + beta F32
pub const INT8_TILE640_CODE_BYTES: usize = 640;
pub const INT8_TILE640_METADATA_BYTES: usize = 4; // alpha F32 only

// \u2500\u2500 V1 wire ABI representation discriminants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RuntimeRepresentationClass {
    TernaryTile640Base = 0,
    Nf4Tile640Base = 1,
    Int8Tile640Base = 2,
    RawF32 = 3,
}

// ── Source matrix layout declaration ────────────────────────────────────────
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceMatrixLayout {
    /// Prism canonical: W[in_features, out_features]
    PrismInByOut,
    /// Checkpoint convention: W_checkpoint[out_features, in_features]
    CheckpointOutByIn,
}

// \u2500\u2500 Canonical shape contract \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#[derive(Debug, Clone, Copy)]
pub struct CanonicalShape {
    pub in_features: u32,
    pub out_features: u32,
    pub rank: u16,
}

impl CanonicalShape {
    pub fn validate(&self) -> Result<(), String> {
        if self.in_features == 0 {
            return Err("CanonicalShape: in_features must be > 0".into());
        }
        if self.out_features == 0 {
            return Err("CanonicalShape: out_features must be > 0".into());
        }
        if self.rank != 2 {
            return Err(format!("CanonicalShape: rank must be 2, got {}", self.rank));
        }
        Ok(())
    }
    pub fn element_count(&self) -> Option<u64> {
        (self.in_features as u64).checked_mul(self.out_features as u64)
    }
}

pub fn validate_source_layout(
    source_rows: u32,
    source_cols: u32,
    source_element_count: u64,
    in_features: u32,
    out_features: u32,
    layout: SourceMatrixLayout,
) -> Result<CanonicalShape, String> {
    if source_rows == 0 || source_cols == 0 {
        return Err("validate_source_layout: source dimensions must be > 0".into());
    }
    let source_expected = (source_rows as u64)
        .checked_mul(source_cols as u64)
        .ok_or_else(|| "source dimensions overflow".to_string())?;
    if source_element_count != source_expected {
        return Err(format!(
            "validate_source_layout: count {} != {}x{}={}",
            source_element_count, source_rows, source_cols, source_expected
        ));
    }
    let (norm_in, norm_out) = match layout {
        SourceMatrixLayout::PrismInByOut => (source_rows, source_cols),
        SourceMatrixLayout::CheckpointOutByIn => (source_cols, source_rows),
    };
    if norm_in != in_features {
        return Err(format!(
            "validate_source_layout: normalized in {} != expected {}",
            norm_in, in_features
        ));
    }
    if norm_out != out_features {
        return Err(format!(
            "validate_source_layout: normalized out {} != expected {}",
            norm_out, out_features
        ));
    }
    let norm_count = (in_features as u64)
        .checked_mul(out_features as u64)
        .ok_or_else(|| "expected dimensions overflow".to_string())?;
    if norm_count != source_element_count {
        return Err(format!(
            "validate_source_layout: normalized count {} != source count {}",
            norm_count, source_element_count
        ));
    }
    Ok(CanonicalShape {
        in_features,
        out_features,
        rank: 2,
    })
}

// \u2500\u2500 Tail handling contract \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailHandlingContract {
    ActivationZeroPredicationV1 = 1,
}

// \u2500\u2500 Tile macro layout \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileMacroLayout {
    OutputChannelContiguous = 1,
    ReductionTileInterleaved = 2,
}

/// NF4 tile640 representation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantizedMatrixFormat {
    /// NF4 tile640 with tile-local reconstruction.
    /// Supports multiple packing policies: MaxAbsV1, AwlsV1, OutputScaledFoldedV1.
    /// Output-scaled folding emits standard Nf4Tile640Base \u2014 no runtime sidecar.
    Nf4Tile640Base = 1,
    /// INT8 tile640 with per-tile symmetric quantization.
    Int8Tile640Base = 2,
    /// Ternary tile640: 256-element blocks, 2-bit codes, FP16 scale per block.
    TernaryTile640Base = 3,
    /// Raw F16 passthrough for tensors that cannot meet compressed parity.
    RawF32 = 4,
}

/// Packing policy for Nf4Tile640Base \u2014 determines how tile alpha/beta are derived.
/// All policies produce the same runtime format; the policy is recorded in receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nf4PackPolicy {
    MaxAbsV1,
    AwlsV1,
    OutputScaledFoldedV1,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// NF4 output scale folded into tile alpha/beta at pack time.
    OutputScaledFolded {
        policy: ReductionScalePolicy,
        scale_storage: ChannelScaleStorage,
        scale_axis: ScaleAxis,
        scale_count: u32,
        epsilon_bits: u16,
    },
}

// ── Ternary tile640 contract ──────────────────────────────────────────────
// RuntimeRepresentationClass::TernaryTile640Base is the canonical production
// representation. CodecFamily remains the search-level family; the compiled
// TernaryCandidateRecipe records the complete physical format.

/// Complete physical contract for one ternary candidate.
///
/// Every field must affect either packing, reconstruction, or kernel selection.
/// Dense residual retention is rejected at construction for production candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryCandidateRecipe {
    pub codec: TernaryCodec,
    pub scale_policy: TernaryScalePolicy,
    pub threshold_policy: TernaryThresholdPolicy,
    pub group_size: u32,
    pub residual_policy: TernaryResidualPolicy,
    pub kernel_abi: TernaryKernelAbi,
    pub representation_version: u16,
    pub sparse_residual_capacity: Option<u32>,
}

impl TernaryCandidateRecipe {
    pub fn validate(&self) -> Result<(), String> {
        if self.group_size == 0 || self.group_size > 640 || 640 % self.group_size != 0 {
            return Err(format!("group_size {} must divide 640", self.group_size));
        }
        if matches!(self.residual_policy, TernaryResidualPolicy::Dense { .. }) {
            return Err("dense residual invalidates compression fitness".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryCodec {
    Tile640,
    BitNet158,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryScalePolicy {
    SymmetricPerGroup,
    AsymmetricPerGroup,
    Learned,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TernaryThresholdPolicy {
    Fixed(f32),
    Percentile(f32),
    Learned,
    ActivationAware,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TernaryResidualPolicy {
    None,
    Sparse {
        fraction: f32,
        fallback: ResidualFallbackPrecision,
    },
    LowRank {
        rank: u32,
    },
    Dense {
        fallback: ResidualFallbackPrecision,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidualFallbackPrecision {
    Fp16,
    Int8,
    Nf4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TernaryKernelAbi {
    pub threadgroup_size: u32,
    pub simdgroup_width: u32,
    pub unroll_factor: u32,
}

impl Default for TernaryKernelAbi {
    fn default() -> Self {
        Self {
            threadgroup_size: 32,
            simdgroup_width: 32,
            unroll_factor: 4,
        }
    }
}

impl Default for TernaryCandidateRecipe {
    fn default() -> Self {
        Self {
            codec: TernaryCodec::Tile640,
            scale_policy: TernaryScalePolicy::SymmetricPerGroup,
            threshold_policy: TernaryThresholdPolicy::Percentile(50.0),
            group_size: 32,
            residual_policy: TernaryResidualPolicy::None,
            kernel_abi: TernaryKernelAbi::default(),
            representation_version: REPRESENTATION_REGISTRY_VERSION,
            sparse_residual_capacity: None,
        }
    }
}

pub fn ternary_candidate_digest(
    tensor_digest: &str,
    recipe: &TernaryCandidateRecipe,
    policy_digest: &str,
) -> Result<String, String> {
    let bytes = bincode::serialize(recipe).map_err(|e| format!("serialize: {e}"))?;
    let mut hasher = Sha256::new();
    hasher.update(tensor_digest.as_bytes());
    hasher.update(b":recipe:");
    hasher.update(&bytes);
    hasher.update(b":policy:");
    hasher.update(policy_digest.as_bytes());
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect())
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
#[derive(Debug, Clone, Default)]
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
    pub fn admission_status(
        &self,
        profile: &QuantizationValidationProfile,
        is_ternary: bool,
    ) -> WeightAdmission {
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
#[derive(Debug, Clone, Default)]
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

/// Evidence collected for a candidate during admission.
#[derive(Debug, Clone)]
pub struct CandidateEvidence {
    pub representation: RuntimeRepresentationClass,
    pub representation_version: u16,
    pub pack_policy_id: u16,
    pub source_digest: [u8; 32],
    pub canonical_shape: Option<CanonicalShape>,
    pub structural_report: Option<StructuralReport>,
    pub reconstruction_report: Option<ReconstructionReport>,
    pub probe_report: Option<OperatorValidationReport>,
    pub promotion_report: Option<OperatorValidationReport>,
    pub holdout_report: Option<OperatorValidationReport>,
    pub runtime_conformance_report: Option<RuntimeConformanceReport>,
    pub completed_vectors: PhaseVectorCounts,
    pub payload_bytes: u64,
    pub metadata_bytes: u64,
    pub estimated_runtime_cost: f64,
    pub result: CandidateResult,
}

impl Default for CandidateEvidence {
    fn default() -> Self {
        CandidateEvidence {
            representation: RuntimeRepresentationClass::Nf4Tile640Base,
            representation_version: 0,
            pack_policy_id: 0,
            source_digest: [0u8; 32],
            canonical_shape: None,
            structural_report: None,
            reconstruction_report: None,
            probe_report: None,
            promotion_report: None,
            holdout_report: None,
            runtime_conformance_report: None,
            completed_vectors: PhaseVectorCounts::default(),
            payload_bytes: 0,
            metadata_bytes: 0,
            estimated_runtime_cost: 0.0,
            result: CandidateResult::DiagnosticOnly,
        }
    }
}

/// Counts of validation vectors completed in each phase.
#[derive(Debug, Clone, Default)]
pub struct PhaseVectorCounts {
    pub probe: u32,
    pub promotion: u32,
    pub holdout: u32,
    pub total: u32,
}

/// Receipt for a stratified-selection round of activation vectors.
#[derive(Debug, Clone)]
pub struct ActivationBankSelectionReceipt {
    pub bank_digest: [u8; 32],
    pub original_bank_size: usize,
    pub selected_indices: Vec<usize>,
    pub selected_count: usize,
    pub seed: u64,
    pub num_strata: usize,
    pub algorithm_version: u32,
    pub exclusion_from_parent: bool,
}

/// Classification of a cimage artifact by its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CimageArtifactClass {
    ExperimentalCimage,
    ProductionCimage,
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
    pub format: RuntimeRepresentationClass,
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
    pub format: RuntimeRepresentationClass,
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
    /// Whether the compiler may attempt Int8Tile640Base candidates.
    pub permit_int8_candidate: bool,
}

/// Target execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BackendKind {
    Metal = 0,
    CpuReference = 1,
    Ane = 2,
}

/// Target backend capability for a representation.
#[derive(Debug, Clone)]
pub struct RepresentationCapability {
    pub representation: RuntimeRepresentationClass,
    pub representation_version: u16,
    pub backend: BackendKind,
    pub kernel_abi_digest: [u8; 32],
    pub cpu_reference_ready: bool,
    pub parser_ready: bool,
    pub artifact_writer_ready: bool,
    pub loader_ready: bool,
    pub runtime_kernel_ready: bool,
    pub nonzero_offset_test_passed: bool,
    pub tail_mask_test_passed: bool,
    /// Tensor-mixing within a layer: different tensors in same operation use different codecs (MLP level)
    pub tensor_mixing_passed: bool,
    /// Layer-mixing: different layers in same decoder use different codecs
    pub layer_mixing_passed: bool,
    /// Decoder integration: full prefill+decode with mixed codecs
    pub decoder_integration_passed: bool,
    /// Serving integration: session lifecycle with mixed codecs
    pub serving_integration_passed: bool,
    /// Keep existing booleans for backward compat
    pub mixed_format_test_passed: bool,
    pub end_to_end_profile_test_passed: bool,
    pub production_ready: bool,
}

/// Qualification result for a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateResult {
    ProductionQualified,
    DiagnosticOnly,
    Failed,
    Skipped,
}

/// Structural validation results for a candidate.
#[derive(Debug, Clone)]
pub struct StructuralReport {
    pub bytes_valid: bool,
    pub segment_bounds_valid: bool,
    pub alignment_valid: bool,
    pub macro_layout_compatible: bool,
    pub tail_contract_compatible: bool,
    pub errors: Vec<String>,
}

impl StructuralReport {
    pub fn is_pass(&self) -> bool {
        self.bytes_valid
            && self.segment_bounds_valid
            && self.alignment_valid
            && self.macro_layout_compatible
            && self.tail_contract_compatible
            && self.errors.is_empty()
    }
}

/// Weight-space reconstruction quality.
#[derive(Debug, Clone)]
pub struct ReconstructionReport {
    pub weight_nrmse: f64,
    pub zero_collapse_ratio: f64,
    pub max_abs_error: f64,
    pub snr_db: f64,
    pub structural: StructuralReport,
}

/// Runtime kernel conformance results.
#[derive(Debug, Clone)]
pub struct RuntimeConformanceReport {
    pub cpu_reference_parity: bool,
    pub nonzero_offset_passed: bool,
    pub tail_mask_passed: bool,
    pub mixed_format_passed: bool,
    pub errors: Vec<String>,
}

/// Activation trace format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ActivationTraceFormat {
    RawF32 = 0,
    SyntheticStressOnly = 1,
}

/// Production activation trace with exact graph-boundary semantics.
#[derive(Debug, Clone)]
pub struct ActivationTrace {
    pub trace_version: u16,
    pub tensor_id: [u8; 16],
    pub source_model_digest: [u8; 32],
    pub logical_width: u32,
    pub sample_count: u32,
    pub profile_id: [u8; 16],
    pub trace_digest: [u8; 32],
    pub data_format: ActivationTraceFormat,
    pub storage_ref: ArtifactRef,
}

/// Reference to an artifact in the content store.
#[derive(Debug, Clone)]
pub struct ArtifactRef {
    pub segment: u8,
    pub offset: u64,
    pub length: u64,
}

/// Structured admission failure.
#[derive(Debug, Clone)]
pub enum QuantizationAdmissionFailure {
    NoCandidatePassed {
        candidates_attempted: Vec<String>,
        best_evidence: Option<CandidateEvidence>,
        completed_vectors: PhaseVectorCounts,
        bank_selections: Vec<ActivationBankSelectionReceipt>,
    },
    PackerFailure(String),
    /// The per-tensor wall-clock deadline expired during validation.
    /// Carries the best candidate metrics seen so far and the phase in
    /// which time ran out, so the caller can distinguish "ternary
    /// fundamentally failed" from "this tensor needs a larger budget."
    TimeoutDeadline {
        candidates_attempted: Vec<String>,
        best_evidence: Option<CandidateEvidence>,
        completed_vectors: PhaseVectorCounts,
        expired_phase: String,
        bank_selections: Vec<ActivationBankSelectionReceipt>,
    },
}
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
