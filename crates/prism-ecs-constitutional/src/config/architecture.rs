//! Layer 2: Normalized architecture types and the supporting diffusion /
//! attention / quantization types they reference.
//!
//! Authority: the strict Rust types representing a model's normalized
//! architecture (text, vision, audio), the attention-kind / rope / MoE
//! / diffusion configuration, and the quantization metadata that
//! accompany them. The companion [`super::parser`] consumes
//! `config.json` and produces values of these types.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ── Layer 2: Normalized Architecture ───────────────────────────────────────

/// Fully resolved text model architecture from config.json.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextArchitecture {
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub global_head_dim: Option<u32>,
    pub num_global_key_value_heads: Option<u32>,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub sliding_window: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f64>,
    pub hidden_size_per_layer_input: u32,
    pub layer_types: Vec<AttentionKind>,
    pub rope_local: RopeSpec,
    pub rope_global: Option<RopeSpec>,
    pub model_type: String,

    /// Mixture-of-Experts configuration, if applicable.
    #[serde(default)]
    pub moe_config: Option<MoEConfig>,

    /// Diffusion model configuration, if applicable.
    #[serde(default)]
    pub diffusion_config: Option<DiffusionConfig>,

    /// Whether the model supports thinking/reasoning mode (e.g. Qwen3 dual-mode).
    #[serde(default)]
    pub thinking_mode: bool,
}

impl TextArchitecture {
    /// Compute the total number of weight elements that will be quantized
    /// via TernaryTile640.  This determines the exact .cimage weights segment size.
    ///
    /// Includes: embedding, per-layer Q/K/V/O/Gate/Up/Down, LM head (if untied).
    pub fn total_ternary_weight_elements(&self) -> u64 {
        let h = self.hidden_size as u64;
        let im = self.intermediate_size as u64;
        let v = self.vocab_size as u64;
        let n = self.num_hidden_layers as u64;
        let hd = self.head_dim as u64;
        let nq = self.num_attention_heads as u64;
        let nk = self.num_key_value_heads as u64;

        // Embedding: vocab x hidden
        let mut total = v * h;

        // Per layer projections:
        let per_layer = n
            * (
                h * (nq * hd)      // Q
            + h * (nk * hd)     // K
            + h * (nk * hd)     // V
            + (nq * hd) * h     // O
            + h * im             // Gate
            + h * im             // Up
            + im * h
                // Down
            );
        total += per_layer;

        // LM head (if not tied with embeddings)
        if !self.tie_word_embeddings {
            total += h * v;
        }

        total
    }
}

/// Vision encoder configuration from a model's vision_config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionArchitecture {
    #[serde(alias = "hiddenSize", alias = "mm_embed_dim")]
    pub hidden_size: u32,
    #[serde(alias = "num_heads")]
    pub num_attention_heads: u32,
    #[serde(alias = "depth")]
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    #[serde(default)]
    pub image_size: u32,
    #[serde(default)]
    pub patch_size: u32,
    #[serde(default)]
    pub num_channels: u32,
    #[serde(default, alias = "output_proj_dims")]
    pub projection_dim: u32,
    /// Cimage model family identifier (e.g. "clip-vit-b32").
    #[serde(default)]
    pub model_family: String,
    /// Whether a pre-compiled ANE program is embedded in the cimage.
    #[serde(default)]
    pub has_ane_program: bool,
}

impl Default for VisionArchitecture {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            intermediate_size: 0,
            image_size: 896,
            patch_size: 16,
            num_channels: 3,
            projection_dim: 0,
            model_family: String::new(),
            has_ane_program: false,
        }
    }
}

/// Audio encoder configuration from a model's audio_config.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioArchitecture {
    #[serde(alias = "audio_embed_dim")]
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub sample_rate: u32,        // e.g. 16000
    pub num_mel_bins: u32,       // e.g. 80
    pub hop_length: u32,         // e.g. 160
    pub max_audio_length_s: u32, // e.g. 30 (seconds)
    #[serde(default)]
    pub projection_dim: u32, // audio_features -> text hidden dim
}

impl Default for AudioArchitecture {
    fn default() -> Self {
        Self {
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            intermediate_size: 0,
            sample_rate: 16_000,
            num_mel_bins: 80,
            hop_length: 160,
            max_audio_length_s: 30,
            projection_dim: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    SlidingAttention,
    FullAttention,
    GeminiSparseAttention,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
}

/// Quantization metadata from the converted model.
///
/// The per-layer overrides map is ordered (BTreeMap) so serialization is
/// stable across runs and across hash implementations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationMeta {
    pub bits: u32,
    pub group_size: u32,
    pub mode: QuantizationMode,
    /// Per-layer overrides (if any layer has non-default group size or bits).
    pub overrides: BTreeMap<String, QuantizationMeta>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    None,
    Affine,
    Symmetric,
}

/// Mixture-of-Experts (MoE) routing configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoEConfig {
    /// Total number of experts in the MoE layer.
    pub num_experts: u32,
    /// Number of experts activated per token (top-K routing).
    pub top_k_experts: u32,
    /// FFN intermediate size within each expert.
    pub intermediate_size: u32,
    /// Whether shared (always-active) experts are used alongside routed experts.
    pub shared_experts: bool,
}

/// Diffusion model configuration from the model's config.json.
/// Used by DiffusionGemma for parallel denoising text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DiffusionConfig {
    /// Maximum number of diffusion tokens per batch (default 256).
    pub max_diffusion_tokens: u32,
    /// Default number of denoising steps (4-8 for text).
    pub default_denoising_steps: u32,
    /// Noise schedule type (cosine, sqrt, linear).
    pub noise_schedule: NoiseScheduleType,
    /// Number of tokens generated per forward pass (15-20).
    pub parallel_token_generation: u32,
    /// Whether the model supports image inputs natively.
    pub supports_images: bool,
    /// Whether the model supports video inputs natively.
    pub supports_video: bool,
    /// Input image size in pixels (e.g. 896).
    pub image_size: u32,
    /// Patch size for image/video processing.
    pub patch_size: u32,
    /// Maximum context length (e.g. 262144).
    pub max_context_length: u32,
    /// Token ID used for masking in diffusion (default 0).
    pub mask_token_id: u32,
    /// Padding token ID (default 0).
    pub pad_token_id: u32,
    /// End-of-sequence token ID (default 0).
    pub eos_token_id: u32,
    /// Maximum canvas tokens for diffusion generation (default 256).
    pub max_canvas_tokens: u32,
    /// Dimension of the timestep embedding (default 4096).
    pub timestep_embedding_dim: u32,
    /// Confidence type for token selection (default LogProb).
    pub confidence_type: ConfidenceType,
    /// Default confidence threshold for commit decisions (default 0.7).
    pub default_confidence_threshold: f32,
    /// Whether EOS collapse is enabled (default true).
    pub eos_collapse_enabled: bool,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            max_diffusion_tokens: 256,
            default_denoising_steps: 6,
            noise_schedule: NoiseScheduleType::Cosine,
            parallel_token_generation: 18,
            supports_images: true,
            supports_video: true,
            image_size: 896,
            patch_size: 16,
            max_context_length: 262_144,
            mask_token_id: 0,
            pad_token_id: 0,
            eos_token_id: 0,
            max_canvas_tokens: 256,
            timestep_embedding_dim: 4096,
            confidence_type: ConfidenceType::LogProb,
            default_confidence_threshold: 0.7,
            eos_collapse_enabled: true,
        }
    }
}

/// Confidence type for token selection during diffusion decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceType {
    /// Use log-probability of token.
    LogProb,
    /// Use softmax margin (top - second).
    SoftmaxMargin,
    /// Use normalized entropy.
    NormalizedEntropy,
}

/// Mask selection strategy for discrete diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MaskSelection {
    /// Mask tokens below a confidence threshold.
    Threshold { confidence_threshold: f32 },
    /// Mask a fixed ratio of tokens.
    Ratio { mask_ratio: f32 },
    /// Adaptively schedule masking.
    AdaptiveSchedule,
    /// Mask the lowest-confidence tokens.
    LowestConfidence,
}

/// Sampling policy for discrete diffusion decoding steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerPolicy {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub mask_selection: MaskSelection,
}

/// Policy for committing tokens during a diffusion step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitPolicy {
    pub min_confidence: f32,
    pub max_commits_per_step: Option<u32>,
}

/// Condition under which diffusion decoding stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopCondition {
    /// Stop after a fixed number of steps.
    MaxSteps(u32),
    /// Stop after N steps with no new commits.
    ConvergedAfter(u32),
    /// Stop when all tokens are committed.
    AllCommitted,
    /// Stop on EOS collapse.
    EosCollapse,
    /// Hard ceiling on total steps.
    HardStepCeiling(u32),
}

/// Forward pass strategy for a diffusion route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionForwardRoute {
    /// Full transformer forward pass.
    FullTransformer,
    /// Cached transformer forward pass with a KV cache strategy.
    CachedTransformer { cache_strategy: KvCacheMode },
}

/// A single stage in the diffusion execution pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStage {
    pub stage_index: u32,
    pub timestep: u32,
    pub forward_route: DiffusionForwardRoute,
    pub sampler_policy: SamplerPolicy,
    pub commit_policy: CommitPolicy,
    pub stop_conditions: Vec<StopCondition>,
}

/// Complete execution plan for a diffusion model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionExecutionPlan {
    pub stages: Vec<DiffusionStage>,
    pub total_denoising_steps: u32,
    pub kv_cache_mode: KvCacheMode,
    pub max_canvas_tokens: u32,
    pub final_logit_softcapping: Option<f64>,
}

/// Generation regime: autoregressive (token-by-token) or discrete diffusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationRegime {
    /// Standard autoregressive generation (token-by-token).
    Autoregressive,
    /// Discrete diffusion / parallel decoding.
    DiscreteDiffusion,
}

impl Default for GenerationRegime {
    fn default() -> Self {
        Self::Autoregressive
    }
}

/// KV cache strategy for diffusion or autoregressive decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheMode {
    /// Append new tokens to KV cache only.
    AppendOnly,
    /// Recompute the full KV cache at each step.
    FullRecompute,
    /// Block-wise KV cache with fixed-size blocks.
    BlockCache,
}

impl Default for KvCacheMode {
    fn default() -> Self {
        Self::AppendOnly
    }
}

/// Attention masking strategy for diffusion decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffusionAttentionKind {
    /// Full bidirectional attention.
    BidirectionalFull,
    /// Sliding window bidirectional attention.
    BidirectionalSliding,
}

/// Noise schedule type for diffusion denoising.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseScheduleType {
    /// Cosine schedule (cosine-based noise weighting).
    Cosine,
    /// Square-root schedule (sqrt-based noise weighting).
    Sqrt,
    /// Linear schedule (linear noise weighting).
    Linear,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_weight_elements_excludes_lm_head_when_tied() {
        let arch = TextArchitecture {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            num_hidden_layers: 2,
            vocab_size: 1000,
            tie_word_embeddings: true,
            ..Default::default()
        };
        let n = arch.total_ternary_weight_elements();
        // Embedding (v*h) + 2 layers of (h*nh*hd + h*nk*hd + h*nk*hd + nh*hd*h + h*im + h*im + im*h)
        let per_layer: u64 = 64 * (2 * 32)        // Q
            + 64 * (1 * 32)        // K
            + 64 * (1 * 32)        // V
            + (2 * 32) * 64        // O
            + 64 * 128             // Gate
            + 64 * 128             // Up
            + 128 * 64;            // Down
        let expected = 1000 * 64 + 2 * per_layer;
        assert_eq!(n, expected);
    }

    #[test]
    fn ternary_weight_elements_includes_lm_head_when_untied() {
        let arch = TextArchitecture {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            num_hidden_layers: 1,
            vocab_size: 1000,
            tie_word_embeddings: false,
            ..Default::default()
        };
        let n = arch.total_ternary_weight_elements();
        let per_layer: u64 = 64 * (2 * 32)
            + 64 * (1 * 32)
            + 64 * (1 * 32)
            + (2 * 32) * 64
            + 64 * 128
            + 64 * 128
            + 128 * 64;
        let expected = 1000 * 64 + per_layer + 64 * 1000;
        assert_eq!(n, expected);
    }

    #[test]
    fn attention_kind_round_trip() {
        let j = serde_json::to_string(&AttentionKind::FullAttention).unwrap();
        let back: AttentionKind = serde_json::from_str(&j).unwrap();
        assert_eq!(back, AttentionKind::FullAttention);
    }

    #[test]
    fn quantization_meta_btreemap_serialization_is_stable() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "layer.0".to_string(),
            QuantizationMeta {
                bits: 4,
                group_size: 64,
                mode: QuantizationMode::Affine,
                overrides: BTreeMap::new(),
            },
        );
        let q = QuantizationMeta {
            bits: 4,
            group_size: 64,
            mode: QuantizationMode::Affine,
            overrides,
        };
        let j1 = serde_json::to_string(&q).unwrap();
        let j2 = serde_json::to_string(&q).unwrap();
        assert_eq!(j1, j2);
    }
}
