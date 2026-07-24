//! Sweep spec types — parameter grids, enums, and the top-level
//! QuantSweepSpec that drives a sweep run.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::contract::TensorClass;

// ── PolicyMode ────────────────────────────────────────────────────────────────

/// Mode for candidate generation — exploratory (all candidates) or production-only.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PolicyMode {
    Exploratory,
    ProductionCandidateOnly,
}

// ── SweepFailureReason ──────────────────────────────────────────────────────────

/// Reason a sweep candidate was rejected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SweepFailureReason {
    /// Candidate passed all gates.
    #[default]
    None,
    /// Weight-space NRMSE exceeded the gate threshold.
    WeightNrmse,
    /// Weight-space zero-collapse ratio exceeded the gate threshold.
    ZeroCollapse,
    /// Operator NRMSE exceeded the gate threshold.
    OperatorNrmse,
    /// Operator max-absolute tail exceeded the gate threshold (e.g., patch_dense pattern).
    OperatorMaxAbsTail,
    /// Operator cosine dissimilarity below threshold.
    OperatorCosine,
    /// Operator norm drift too large.
    NormDrift,
    /// Byte savings below minimum useful threshold.
    InsufficientByteSavings,
    /// Hardware validation required but not available.
    HardwareEvidenceMissing,
    /// Rollout validation required but not available.
    RolloutEvidenceMissing,
    /// Candidate has unsupported parameters (wrong group_size, etc.).
    UnsupportedParameter,
    /// Candidate is disallowed by the current policy.
    DisallowedByPolicy,
    /// Candidate codec not implemented yet (e.g., SymInt4).
    NotImplemented,
    /// Quality drift exceeded acceptable threshold.
    QualityDrift,
    /// Health/stability failure.
    HealthOrStability,
    /// Catch-all for unexpected failures.
    Other(String),
}

// ── Tensor selection ────────────────────────────────────────────────────────────

/// How to select tensors for a sweep run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorSelector {
    /// Match an exact safetensor key.
    ExactKey(String),
    /// Match tensor keys by regex pattern.
    Regex(String),
    /// Select up to `max_tensors` tensors of a given class.
    TensorClass {
        #[serde(with = "tensor_class_serde")]
        class: TensorClass,
        max_tensors: usize,
    },
    /// Depth-stratified sampling for efficient family validation.
    DepthAware(DepthAwareSelector),
}

/// Select tensors by family, stratified by depth.
/// Instead of testing every tensor, test representative samples
/// from early, middle, and late layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthAwareSelector {
    /// Tensor class to filter by. If empty, apply to all.
    pub tensor_class: String,
    /// Depth ranges to sample: e.g. ["0-3", "20-25", "42-46"]
    pub depth_ranges: Vec<String>,
    /// Max total tensors to select.
    pub max_tensors: usize,
}

// ── Codec parameter enums ───────────────────────────────────────────────────────

/// NF4 codebook identity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Nf4CodebookId {
    /// Prism's current production NF4 codebook.
    PrismCurrent,
    /// BitsAndBytes NF4 codebook.
    BitsAndBytesNf4,
    /// Symmetric normal float codebook.
    SymmetricNormalFloat,
}

/// Affine quantization mode — whether bias is present alongside scale.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AffineMode {
    /// Scale only, no bias.
    ScaleOnly,
    /// Scale and bias both present.
    ScaleBias,
}

/// Policy for computing the quantization scale factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalePolicy {
    /// Max absolute value clip.
    MaxAbs,
    /// MSE-optimal grid search with given number of steps.
    MseOptimalGrid { steps: u16 },
    /// Activation-weighted least-squares scale.
    ActivationWeightedLs,
}

/// Policy for clipping outlier values before quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClippingPolicy {
    /// No clipping.
    None,
    /// Clip at the given percentile (0.0–100.0).
    Percentile(f32),
    /// Clip at the given number of standard deviations from the mean.
    StddevMultiple(f32),
    /// Clip at per-magnitude fractions of the max-abs value.
    GridFractionOfMaxAbs(Vec<f32>),
}

/// Group-wise optimizer for quantization.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum GroupOptimizer {
    /// No group-level optimization.
    None,
    /// Alternating affine optimization for up to `max_iters` iterations.
    AffineAlternating { max_iters: u8 },
    /// Activation-weighted optimization for up to `max_iters` iterations.
    ActivationWeighted { max_iters: u8 },
}

/// Signed INT4 range variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SignedInt4Range {
    /// Range [-7, 7] — symmetrical.
    Neg7ToPos7,
    /// Range [-8, 7] — one extra negative value.
    Neg8ToPos7,
}

// ── Sweep grid types ────────────────────────────────────────────────────────────

/// Parameter grid for NF4 codec variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nf4SweepGrid {
    pub codebooks: Vec<Nf4CodebookId>,
    pub group_sizes: Vec<usize>,
    pub affine_modes: Vec<AffineMode>,
    pub clip_policies: Vec<ClippingPolicy>,
    pub optimizers: Vec<GroupOptimizer>,
    /// If true, generate both unweighted and activation-weighted variants
    /// for each parameter combination.
    #[serde(default)]
    pub activation_weighted: bool,
}

/// Parameter grid for symmetric INT4 codec variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymInt4SweepGrid {
    pub group_sizes: Vec<usize>,
    pub signed_ranges: Vec<SignedInt4Range>,
    pub affine_modes: Vec<AffineMode>,
    pub clip_policies: Vec<ClippingPolicy>,
    pub scale_policies: Vec<ScalePolicy>,
}

/// Parameter grid for INT8 codec variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int8SweepGrid {
    pub group_sizes: Vec<usize>,
    pub clipping_policies: Vec<ClippingPolicy>,
    pub scale_policies: Vec<ScalePolicy>,
    pub per_channel: bool,
}

/// Parameter grid for ternary codec variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernarySweepGrid {
    pub group_sizes: Vec<usize>,
    pub sparsity_targets: Vec<f32>,
    pub scale_policies: Vec<String>,
    pub residual_policies: Vec<String>,
}

/// Parameter grid for mixed-tile rescue variants.
///
/// Granularity of rescue units for mixed-tile quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueGranularity {
    Group,
    Tile640,
    OutputChannel,
}

/// Criterion for selecting rescue units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueSelector {
    WeightError,
    NormalizedWeightError,
    ActivationWeightedError,
    OutlierMagnitude,
}

/// Schedule for iterative rescue rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RescueSchedule {
    OneShot { fraction: f64 },
    FixedPerRound { fraction_per_round: f64, rounds: u8 },
    Geometric { fractions: Vec<f64> },
}

/// How rescue values are combined with the base representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlayMode {
    FullReplacement,
    DeltaCorrection,
}

/// Simplified quantization policy identifier for mixed-tile base policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantPolicy {
    pub family: String,
    pub parameters: serde_json::Value,
}

/// Parameter grid for mixed-tile rescue variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedTileSweepGrid {
    pub base_policies: Vec<QuantPolicy>,
    pub rescue_formats: Vec<String>,
    pub rescue_granularities: Vec<RescueGranularity>,
    pub selectors: Vec<RescueSelector>,
    pub schedules: Vec<RescueSchedule>,
    pub overlay_modes: Vec<OverlayMode>,
    pub max_rounds: u8,
    pub recompute_after_each_round: bool,
}

/// Parameter grid for layout parameter sweep alongside a codec grid.
///
/// Sweeps tile shape, group axis, metadata placement, and execution lane
/// so performance characteristics can be compared across hardware configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutSweepGrid {
    pub tile_shapes: Vec<String>,
    pub group_axes: Vec<String>,
    pub metadata_layouts: Vec<String>,
    pub execution_lanes: Vec<String>,
}

/// Top-level family sweep selector — each variant carries one grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantFamilySweep {
    Nf4(Nf4SweepGrid),
    SymInt4(SymInt4SweepGrid),
    Int8(Int8SweepGrid),
    Ternary(TernarySweepGrid),
    MixedTile(MixedTileSweepGrid),
    /// Layout parameter sweep alongside the codec grid.
    Layout(LayoutSweepGrid),
}

// ── Sweep configuration types ───────────────────────────────────────────────────

/// Validation configuration for a sweep run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepValidationConfig {
    pub run_weight_validation: bool,
    /// Max candidates per tensor (deprecated — use max_candidates_per_tensor).
    #[deprecated(note = "use max_candidates_per_tensor instead")]
    pub max_candidates: Option<usize>,
    pub max_candidates_per_tensor: usize,
    pub max_total_candidates: Option<usize>,
    pub policy_mode: PolicyMode,
}

/// Scoring configuration — controls how candidates are ranked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepScoringConfig {
    /// Per-family maximum acceptable weight NRMSE (hard gate).
    pub max_weight_nrmse_by_family: HashMap<String, f64>,
    /// Maximum acceptable zero-collapse ratio (hard gate).
    pub max_zero_collapse: f64,
    /// Weight applied to the byte-size term in the scoring formula.
    pub byte_weight: f64,
}

/// Resource limits for a sweep run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResourceLimits {
    pub max_workers: usize,
}

// ── Top-level sweep spec ────────────────────────────────────────────────────────

/// Complete specification for a single sweep run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantSweepSpec {
    pub spec_version: u16,
    pub tensor_selectors: Vec<TensorSelector>,
    pub families: Vec<QuantFamilySweep>,
    pub validation: SweepValidationConfig,
    pub scoring: SweepScoringConfig,
    pub resource_limits: SweepResourceLimits,
    pub output_dir: PathBuf,
}

// ── Serde helpers for contract types ────────────────────────────────────────────

/// Serialize/deserialize `TensorClass` as its debug name string.
pub(crate) mod tensor_class_serde {
    use serde::{
        de::{self, Unexpected},
        Deserialize, Deserializer, Serialize, Serializer,
    };

    use crate::contract::TensorClass;

    pub fn serialize<S>(tc: &TensorClass, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let name = match tc {
            TensorClass::DecoderAttentionProjection => "DecoderAttentionProjection",
            TensorClass::DecoderMlpProjection => "DecoderMlpProjection",
            TensorClass::TokenEmbedding => "TokenEmbedding",
            TensorClass::VisionPatchProjection => "VisionPatchProjection",
            TensorClass::CrossModalBridge => "CrossModalBridge",
            TensorClass::OutputHead => "OutputHead",
            TensorClass::Unknown => "Unknown",
        };
        name.serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<TensorClass, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "DecoderAttentionProjection" => Ok(TensorClass::DecoderAttentionProjection),
            "DecoderMlpProjection" => Ok(TensorClass::DecoderMlpProjection),
            "TokenEmbedding" => Ok(TensorClass::TokenEmbedding),
            "VisionPatchProjection" => Ok(TensorClass::VisionPatchProjection),
            "CrossModalBridge" => Ok(TensorClass::CrossModalBridge),
            "OutputHead" => Ok(TensorClass::OutputHead),
            "Unknown" => Ok(TensorClass::Unknown),
            other => Err(de::Error::invalid_value(
                Unexpected::Str(other),
                &"a valid TensorClass variant name",
            )),
        }
    }
}
