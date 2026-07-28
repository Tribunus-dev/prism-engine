//! Training target specification types.
//!
//! These types describe what kernels and tensors should be trained or
//! calibrated, with what methods, acceptance gates, and priority levels.

use serde::{Deserialize, Serialize};

use prism_ecs_constitutional::canonical::identity::{
    CorpusId, EngramArtifactId, EngramId, PhysicalSegmentId, ReceiptId, RegionId, TensorShape,
};
use crate::ecs::execution_profile::PhysicalTileLayout;
use crate::execution_plan::CodecFamily;

use super::gates::TargetedLossTerm;
use super::gates::{QuantTrainingMethod, WeightTrainingGates};

// ── TrainingTargetSpec ─────────────────────────────────────────────────

/// Top-level specification for a training-aware compilation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTargetSpec {
    /// Schema version for forward-compatibility.
    pub spec_version: u32,
    /// Model family identifier (e.g. "gemma-2-9b", "llama-3-8b").
    pub model_family: String,
    /// Optional digest of the source model for change detection.
    pub model_digest: Option<String>,
    /// Required digest of the compiler policy that generated this spec.
    pub source_policy_digest: String,
    /// Which cimage profile this spec targets (e.g. "production", "research").
    pub target_cimage_profile: String,
    /// Weight-level training targets, one per distinct tensor class / codec.
    pub weight_targets: Vec<WeightTrainingTarget>,
    /// Optional KV cache compression target.
    pub kv_cache_target: Option<KvCacheTrainingTarget>,
    /// Speculative decoding draft-model targets.
    pub speculative_targets: Vec<SpeculativeTrainingTarget>,
    /// Engram (long-term memory) training targets.
    pub engram_targets: Vec<EngramTrainingTarget>,
    /// Attention-shape-level compression / training targets.
    pub attention_shape_targets: Vec<AttentionShapeTrainingTarget>,
    /// Evidence gates that apply to all targets in this spec.
    pub evidence_gates: Vec<TrainingEvidenceGate>,
}

impl TrainingTargetSpec {
    /// Compute a deterministic BLAKE3 digest of this spec.
    pub fn digest(&self) -> String {
        let json = serde_json::to_string(self).expect("TrainingTargetSpec serialization");
        let hash = blake3::hash(json.as_bytes());
        hash.to_hex().to_string()
    }

    /// Number of weight-level training targets.
    pub fn weight_target_count(&self) -> usize {
        self.weight_targets.len()
    }

    /// Total target count across all categories.
    pub fn total_target_count(&self) -> usize {
        let mut n = self.weight_targets.len();
        n += self.speculative_targets.len();
        n += self.engram_targets.len();
        n += self.attention_shape_targets.len();
        if self.kv_cache_target.is_some() {
            n += 1;
        }
        n
    }

    /// Validate the spec's internal consistency.
    ///
    /// Returns `Ok(())` if valid, or an error string describing the first
    /// inconsistency found.
    pub fn check_consistency(&self) -> Result<(), String> {
        if self.model_family.is_empty() {
            return Err("model_family must not be empty".into());
        }
        if self.source_policy_digest.is_empty() {
            return Err("source_policy_digest must not be empty".into());
        }
        if self.target_cimage_profile.is_empty() {
            return Err("target_cimage_profile must not be empty".into());
        }
        for (i, wt) in self.weight_targets.iter().enumerate() {
            if wt.target_id.is_empty() {
                return Err(format!("weight_targets[{}].target_id must not be empty", i));
            }
            if wt.tensor_class.is_empty() {
                return Err(format!(
                    "weight_targets[{}].tensor_class must not be empty",
                    i
                ));
            }
            // Reject negative thresholds.
            let gates = &wt.gates;
            if let Some(v) = gates.max_weight_nrmse {
                if v < 0.0 {
                    return Err(format!(
                        "weight_targets[{}].gates.max_weight_nrmse must be >= 0, got {}",
                        i, v
                    ));
                }
            }
            if let Some(v) = gates.max_zero_collapse_ratio {
                if v < 0.0 || v > 1.0 {
                    return Err(format!(
                        "weight_targets[{}].gates.max_zero_collapse_ratio must be in [0,1], got {}",
                        i, v
                    ));
                }
            }
            if let Some(v) = gates.max_operator_nrmse {
                if v < 0.0 {
                    return Err(format!(
                        "weight_targets[{}].gates.max_operator_nrmse must be >= 0, got {}",
                        i, v
                    ));
                }
            }
            if let Some(v) = gates.min_operator_cosine {
                if !(-1.0..=1.0).contains(&v) {
                    return Err(format!(
                        "weight_targets[{}].gates.min_operator_cosine must be in [-1,1], got {}",
                        i, v
                    ));
                }
            }
            if let Some(v) = gates.max_operator_abs_error {
                if v < 0.0 {
                    return Err(format!(
                        "weight_targets[{}].gates.max_operator_abs_error must be >= 0, got {}",
                        i, v
                    ));
                }
            }
            if let Some(v) = gates.min_byte_savings_ratio {
                if v < 0.0 || v > 1.0 {
                    return Err(format!(
                        "weight_targets[{}].gates.min_byte_savings_ratio must be in [0,1], got {}",
                        i, v
                    ));
                }
            }
        }
        Ok(())
    }
}

// ── TrainingTargetPriority ─────────────────────────────────────────────

/// Priority level for a training target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrainingTargetPriority {
    /// Must be trained before deployment.
    Required,
    /// Strongly recommended for production quality.
    Recommended,
    /// Experimental — training is optional, for evaluation.
    Experimental,
    /// Research-only — not required for any deployment gate.
    Research,
}

// ── WeightTrainingTarget ───────────────────────────────────────────────

/// Describes one weight-tensor class to train/calibrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTrainingTarget {
    /// Unique identifier within the spec.
    pub target_id: String,
    /// Tensor class label (e.g. "attention.q_proj", "mlp.gate_proj").
    pub tensor_class: String,
    /// Tensor key glob patterns this target matches.
    pub tensor_key_match: Vec<String>,
    /// Target codec family (NF4, Int8, Ternary, etc.).
    pub target_codec: CodecFamily,
    /// Physical tile layout for the target codec.
    pub physical_layout: PhysicalTileLayout,
    /// Quantization-aware training method to use.
    pub training_method: QuantTrainingMethod,
    /// Acceptance gates for this target.
    pub gates: WeightTrainingGates,
    /// Priority of this target.
    pub priority: TrainingTargetPriority,
    /// Optional evolutionary search configuration for this weight target.
    /// When set, the training pipeline may run an evolutionary search over
    /// decomposition variants instead of using the default kernel.
    pub search_config: Option<String>,
}

// ── KvCacheTrainingTarget ──────────────────────────────────────────────

/// Training target for KV cache compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheTrainingTarget {
    /// Unique identifier.
    pub target_id: String,
    /// Policy identifier that governs this KV cache region.
    pub policy_id: String,
    /// Target compression ratio (e.g. 2.0 = 2× compression).
    pub target_compression_ratio: f64,
    /// Target codec for the compressed cache.
    pub target_codec: CodecFamily,
    /// Acceptance gates for KV cache quality.
    pub gates: WeightTrainingGates,
    /// Priority of this target.
    pub priority: TrainingTargetPriority,
}

// ── SpeculativeTrainingTarget ──────────────────────────────────────────

/// Training target for a speculative decoding draft model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeTrainingTarget {
    /// Unique identifier.
    pub target_id: String,
    /// Kind of draft model (e.g. "mlp_head", "small_transformer").
    pub draft_kind: String,
    /// Number of hidden layers to extract from the source model.
    pub source_hidden_layers: Vec<usize>,
    /// Minimum target acceptance rate (0.0–1.0).
    pub target_acceptance_rate: f64,
    /// Preferred backend target identifiers for the draft model.
    pub backend_preferences: Vec<String>,
    /// Priority of this target.
    pub priority: TrainingTargetPriority,
}

// ── EngramTrainingTarget ───────────────────────────────────────────────

/// Training target for an engram (long-term memory) codec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramTrainingTarget {
    /// Unique identifier.
    pub target_id: String,
    /// Memory kind (e.g. "episodic", "semantic", "procedural").
    pub memory_kind: String,
    /// Codec to use for stored values.
    pub value_codec: CodecFamily,
    /// Lookup policy identifier.
    pub lookup_policy: String,
    /// Residency mode for engram storage.
    pub residency: String,
    /// Priority of this target.
    pub priority: TrainingTargetPriority,
}
// ── EngramMemoryKind ───────────────────────────────────────────────────

/// Classification of engram memory type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngramMemoryKind {
    /// Episodic — records of specific events/occurrences.
    Episodic,
    /// Semantic — general knowledge and concepts.
    Semantic,
    /// Procedural — how-to knowledge / skill patterns.
    Procedural,
    /// Working — temporary task context.
    Working,
    /// Custom memory kind.
    Custom(String),
}

// ── EngramCodec ─────────────────────────────────────────────────────────

/// Codec used to encode an engram payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngramCodec {
    Nf4,
    Ternary,
    Int8,
    F32,
}

// ── EngramOperation ─────────────────────────────────────────────────────

/// Type of operation an engram performs at its insertion point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EngramOperation {
    Adapter,
    Modulation,
    Projection,
    Prefix,
    Custom(String),
}

// ── EngramApplication ───────────────────────────────────────────────────

/// How an engram's payload is applied to the target tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EngramApplication {
    AdditiveResidual,
    MultiplicativeModulation,
    LowRankProjection,
    LatentPrefix,
    AdapterActivation,
}

// ── EngramRoutingPolicy ─────────────────────────────────────────────────

/// Policy for when an engram is activated during inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EngramRoutingPolicy {
    AlwaysOn,
    ThresholdedSimilarity(f64),
    TopK(usize),
    Learned,
    PolicyControlled,
}

// ── EngramParameterSchema ───────────────────────────────────────────────

/// Schema describing the parameters of an engram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramParameterSchema {
    pub parameter_count: usize,
    pub bytes_per_parameter: usize,
    pub layout: String,
}

// ── PrivacyContract ─────────────────────────────────────────────────────

/// Privacy contract governing an engram's usage and disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyContract {
    pub purpose: String,
    pub retention: String,
    pub disclosure_class: String,
    pub assimilation_permitted: bool,
}

// ── EngramInsertionContract ─────────────────────────────────────────────

/// Contract specifying where and how an engram is applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramInsertionContract {
    pub region: RegionId,
    pub operation: EngramOperation,
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    pub application: EngramApplication,
    pub routing: EngramRoutingPolicy,
    pub maximum_latency_ns: Option<u64>,
}

// ── EngramArtifact ────────────────────────────────────────────────────

/// A trained engram artifact — a pattern applied to a tensor insertion point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramArtifact {
    pub artifact_id: EngramArtifactId,
    pub logical_id: EngramId,
    pub format_version: u32,
    pub memory_kind: EngramMemoryKind,
    pub codec: EngramCodec,
    pub insertion_contract: EngramInsertionContract,
    pub index_segment: Option<PhysicalSegmentId>,
    pub payload_segment: PhysicalSegmentId,
    pub routing_segment: Option<PhysicalSegmentId>,
    pub parameter_schema: EngramParameterSchema,
    pub training_corpus: CorpusId,
    pub training_receipt: ReceiptId,
    pub privacy_contract: PrivacyContract,
}

// ── EngramLookupParams ─────────────────────────────────────────────────

/// Parameters for engram lookup during inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramLookupParams {
    /// Target engram identifier.
    pub engram_id: String,
    /// How to retrieve and apply the engram.
    pub lookup_policy: EngramLookupPolicy,
    /// Optional similarity threshold (for ThresholdGate policy).
    pub retrieval_threshold: Option<f64>,
}

// ── EngramLookupPolicy ─────────────────────────────────────────────────

/// How to retrieve and apply an engram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngramLookupPolicy {
    /// Always apply the engram (no threshold check).
    AlwaysApply,
    /// Apply only if similarity exceeds threshold.
    ThresholdGate,
    /// Apply with learned scaling factor.
    Scaled,
}

// ── EngramLookupReceipt ────────────────────────────────────────────────

/// Receipt from an engram lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramLookupReceipt {
    /// The engram that was looked up.
    pub engram_id: String,
    /// Tensor class of the engram.
    pub tensor_class: String,
    /// Whether the engram was actually applied.
    pub looked_up: bool,
    /// ISO-8601 timestamp of the lookup.
    pub looked_up_at: String,
    /// Retrieval latency in nanoseconds, if measured.
    pub retrieval_latency_ns: Option<u64>,
    /// Digest of the applied payload, if looked_up.
    pub payload_digest: Option<String>,
}

// ── AttentionShapeTrainingTarget ───────────────────────────────────────

/// Training target for attention-head shape / compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionShapeTrainingTarget {
    /// Unique identifier.
    pub target_id: String,
    /// Optional layer range for the target (start, end) inclusive.
    pub layer_range: Option<(usize, usize)>,
    /// Axis along which to compress (e.g. "head", "kv_dim").
    pub compression_axis: String,
    /// Target compression ratio.
    pub compression_ratio: f64,
    /// Priority of this target.
    pub priority: TrainingTargetPriority,
}

// ── ActivationWeightedObjective ────────────────────────────────────────

/// Describes an activation-weighted training objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationWeightedObjective {
    /// Source of the activation profile (e.g. "calibration", "runtime").
    pub profile_source: String,
    /// Norm type for weighting (e.g. "l2", "l1", "max").
    pub activation_norm: String,
    /// Percentile threshold for high-activation channels (0.0–100.0).
    pub percentile: f64,
    /// Fraction of top-activation channels to weight preferentially.
    pub top_k_fraction: f64,
}

// ── TrainingEvidenceGate ───────────────────────────────────────────────

/// An evidence gate that a training target's results must pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingEvidenceGate {
    /// Unique gate identifier within the spec.
    pub gate_id: String,
    /// Human-readable gate type description.
    pub gate_type: String,
    /// Pass/fail threshold value.
    pub threshold: f64,
    /// Relative weight of this gate in aggregate scoring.
    pub weight: f64,
    /// If true, this gate is mandatory; failure marks the target as Failed.
    pub required: bool,
}

// ── MixedPrecisionTrainingTarget ────────────────────────────────────────

/// Describes a mixed-precision training target for a tensor class.
///
/// Mixed-precision targets specify a base codec and a set of allowed
/// override codecs that can be selectively promoted during training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionTrainingTarget {
    /// Tensor class label (e.g. "attention.q_proj", "mlp.gate_proj").
    pub tensor_class: String,
    /// The base codec used for non-promoted groups.
    pub base_codec: CodecFamily,
    /// Set of codecs that may be promoted to (rescue codecs).
    pub allowed_override_codecs: Vec<CodecFamily>,
    /// Maximum fraction of units that may be promoted (0.0 to 1.0).
    pub max_override_fraction: f64,
    /// Target fraction of units to promote — the planner aims for this.
    pub target_override_fraction: f64,
    /// Loss terms that apply to this target during training.
    pub loss_terms: Vec<TargetedLossTerm>,
}
// ── Identity types ──────────────────────────────────────────────────────
//
// Identity types (EngramArtifactId, EngramId, PhysicalSegmentId, CorpusId,
// RegionId, TensorShape, ReceiptId) are now imported from
// prism_ecs_constitutional::canonical::identity.

// ── EngramMemoryKind ───────────────────────────────────────────────────
