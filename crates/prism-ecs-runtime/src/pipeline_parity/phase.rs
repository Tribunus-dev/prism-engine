//! The 21 canonical inference pipeline phases.
//!
//! Every backend MUST implement or explicitly reject each phase (see
//! [`super::support::PhaseSupportStatus`]). Backends are compared
//! only on phases they both support.
//!
//! This enum encodes inference pipeline semantics only. Harness
//! control families (e.g. `identity_passthrough`) are excluded —
//! they map to no phase and never enter comparison groups
//! (see [`super::grouping::graph_family_to_phase`]).

#![forbid(unsafe_code)]

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The 21 canonical inference pipeline phases.
///
/// Every backend MUST implement or explicitly reject each phase.
/// Backends are compared only on phases they both support.
///
/// This enum encodes inference pipeline semantics only. Harness control
/// families (e.g. `identity_passthrough`) are excluded — they map to no
/// phase and never enter comparison groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PipelinePhase {
    /// Embed token IDs into dense vectors.
    TokenEmbedding,
    /// Apply positional encoding (RoPE or learned).
    PositionEncodingOrRope,
    /// Compute Q, K, V projections from input hidden states.
    QkvProjection,
    /// Read KV from cache.
    KvRead,
    /// Write KV key-value at position.
    KvWrite,
    /// Append KV key-value to cache.
    KvAppend,
    /// View slice from cache without mutation.
    KvView,
    /// Compute attention scores: Q @ K^T.
    AttentionScores,
    /// Apply causal mask to attention scores.
    MaskApply,
    /// Softmax over attention scores.
    Softmax,
    /// Weighted sum of V: softmax(QK^T) @ V.
    AttentionWeightedSum,
    /// Project attention output back to model dimension.
    AttentionOutputProjection,
    /// First residual add: attention_output + input.
    ResidualAdd1,
    /// First layer normalization (pre-MLP).
    Norm1,
    /// MLP gate + up projection (e.g., gate_proj, up_proj).
    MlpGateUp,
    /// Activation function (SiLU, ReLU, GELU, etc.).
    Activation,
    /// MLP down projection.
    MlpDown,
    /// Second residual add: mlp_output + residual.
    ResidualAdd2,
    /// Second layer normalization (pre-LM head).
    Norm2,
    /// Language model head projection (hidden → logits).
    LmHead,
    /// Sampling or logits post-processing.
    SamplingOrLogitsPostprocess,
}

impl fmt::Display for PipelinePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PipelinePhase::TokenEmbedding => "token_embedding",
            PipelinePhase::PositionEncodingOrRope => "position_encoding_or_rope",
            PipelinePhase::QkvProjection => "qkv_projection",
            PipelinePhase::KvRead => "kv_read",
            PipelinePhase::KvWrite => "kv_write",
            PipelinePhase::KvAppend => "kv_append",
            PipelinePhase::KvView => "kv_view",
            PipelinePhase::AttentionScores => "attention_scores",
            PipelinePhase::MaskApply => "mask_apply",
            PipelinePhase::Softmax => "softmax",
            PipelinePhase::AttentionWeightedSum => "attention_weighted_sum",
            PipelinePhase::AttentionOutputProjection => "attention_output_projection",
            PipelinePhase::ResidualAdd1 => "residual_add_1",
            PipelinePhase::Norm1 => "norm_1",
            PipelinePhase::MlpGateUp => "mlp_gate_up",
            PipelinePhase::Activation => "activation",
            PipelinePhase::MlpDown => "mlp_down",
            PipelinePhase::ResidualAdd2 => "residual_add_2",
            PipelinePhase::Norm2 => "norm_2",
            PipelinePhase::LmHead => "lm_head",
            PipelinePhase::SamplingOrLogitsPostprocess => "sampling_or_logits_postprocess",
        };
        write!(f, "{s}")
    }
}

impl FromStr for PipelinePhase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "token_embedding" => Ok(PipelinePhase::TokenEmbedding),
            "position_encoding_or_rope" => Ok(PipelinePhase::PositionEncodingOrRope),
            "qkv_projection" => Ok(PipelinePhase::QkvProjection),
            "kv_read" => Ok(PipelinePhase::KvRead),
            "kv_write" => Ok(PipelinePhase::KvWrite),
            "kv_append" => Ok(PipelinePhase::KvAppend),
            "kv_view" => Ok(PipelinePhase::KvView),
            "attention_scores" => Ok(PipelinePhase::AttentionScores),
            "mask_apply" => Ok(PipelinePhase::MaskApply),
            "softmax" => Ok(PipelinePhase::Softmax),
            "attention_weighted_sum" => Ok(PipelinePhase::AttentionWeightedSum),
            "attention_output_projection" => Ok(PipelinePhase::AttentionOutputProjection),
            "residual_add_1" => Ok(PipelinePhase::ResidualAdd1),
            "norm_1" => Ok(PipelinePhase::Norm1),
            "mlp_gate_up" => Ok(PipelinePhase::MlpGateUp),
            "activation" => Ok(PipelinePhase::Activation),
            "mlp_down" => Ok(PipelinePhase::MlpDown),
            "residual_add_2" => Ok(PipelinePhase::ResidualAdd2),
            "norm_2" => Ok(PipelinePhase::Norm2),
            "lm_head" => Ok(PipelinePhase::LmHead),
            "sampling_or_logits_postprocess" => Ok(PipelinePhase::SamplingOrLogitsPostprocess),
            other => Err(format!("unknown PipelinePhase variant: '{other}'")),
        }
    }
}

impl PipelinePhase {
    /// Return all phase variants in discriminant order.
    pub fn all() -> &'static [PipelinePhase] {
        &ALL_PHASES
    }
}

/// All 21 phase variants in discriminant order. Iteration order is
/// observable — [`super::matrices`] and the comparison grouping use
/// it to keep `BTreeMap` keys stable.
pub const ALL_PHASES: [PipelinePhase; 21] = [
    PipelinePhase::TokenEmbedding,
    PipelinePhase::PositionEncodingOrRope,
    PipelinePhase::QkvProjection,
    PipelinePhase::KvRead,
    PipelinePhase::KvWrite,
    PipelinePhase::KvAppend,
    PipelinePhase::KvView,
    PipelinePhase::AttentionScores,
    PipelinePhase::MaskApply,
    PipelinePhase::Softmax,
    PipelinePhase::AttentionWeightedSum,
    PipelinePhase::AttentionOutputProjection,
    PipelinePhase::ResidualAdd1,
    PipelinePhase::Norm1,
    PipelinePhase::MlpGateUp,
    PipelinePhase::Activation,
    PipelinePhase::MlpDown,
    PipelinePhase::ResidualAdd2,
    PipelinePhase::Norm2,
    PipelinePhase::LmHead,
    PipelinePhase::SamplingOrLogitsPostprocess,
];
