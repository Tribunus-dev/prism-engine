//! Layer 2: Normalized architecture types and compile-related hardware targets.
//!
//! This module contains the strict Rust types representing model architecture
//! (TextArchitecture, VisionArchitecture, etc.), hardware targets, quantization
//! modes, diffusion configuration, and the compiled execution specification
//! (Layer 3).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::config_namespace::NamespaceBinding;

use super::operation_route;

// ── Layer 2: Normalized Architecture ───────────────────────────────────────

/// Fully resolved text model architecture from config.json.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
        let per_layer = n * (
            h * (nq * hd)      // Q
            + h * (nk * hd)     // K
            + h * (nk * hd)     // V
            + (nq * hd) * h     // O
            + h * im             // Gate
            + h * im             // Up
            + im * h              // Down
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
pub struct VisionArchitecture {
    #[serde(alias = "hiddenSize")]
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
    #[serde(default)]
    pub projection_dim: u32,
}

/// Audio encoder configuration from a model's audio_config.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioArchitecture {
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub sample_rate: u32,        // e.g. 16000
    pub num_mel_bins: u32,       // e.g. 80
    pub hop_length: u32,         // e.g. 160
    pub max_audio_length_s: u32, // e.g. 30 (seconds)
    pub projection_dim: u32,     // audio_features -> text hidden dim
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    SlidingAttention,
    FullAttention,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
}

/// Quantization metadata from the converted model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizationMeta {
    pub bits: u32,
    pub group_size: u32,
    pub mode: QuantizationMode,
    /// Per-layer overrides (if any layer has non-default group size or bits).
    pub overrides: HashMap<String, QuantizationMeta>,
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

// ── Compile-time Quantization & Hardware ───────────────────────────────────

/// Compile-time quantization mode for the ComputeImage compiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompileQuantMode {
    /// 4-bit NormalFloat (NF4) block quantization.
    Nf4 { group_size: u32 },
    /// 8-bit affine quantization.
    Af8 { group_size: u32 },
    /// Ternary 1.58-bit quantization (2-bit nibble encoding, 4 per byte).
    Ternary { group_size: u32 },
    /// Ternary 1.58-bit quantization with 640-weight SIMD-aligned tiles.
    TernaryTile640 { group_size: u32 },
}

impl CompileQuantMode {
    /// Parse a quant mode name into a CompileQuantMode.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "nf4" => Some(Self::Nf4 { group_size: 64 }),
            "nf4-128" => Some(Self::Nf4 { group_size: 128 }),
            "8bit" => Some(Self::Af8 { group_size: 64 }),
            "ternary" | "1.58" => Some(Self::Ternary { group_size: 32 }),
            "ternary_tile640" | "tile640" => Some(Self::TernaryTile640 { group_size: 640 }),
            "none" => None,
            _ => None,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nf4 { group_size: 64 } => "nf4",
            Self::Nf4 { group_size: 128 } => "nf4-128",
            Self::Af8 { .. } => "8bit",
            Self::Ternary { .. } => "ternary",
            Self::TernaryTile640 { .. } => "ternary_tile640",
            _ => "nf4-64",
        }
    }
}

/// Target hardware for a ComputeImage compilation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Apple M1 (16GB baseline) — max compression, streaming-friendly segments
    M1,
    /// Apple M1 Pro/Max (32-64GB) — moderate compression
    M1Pro,
    /// Apple M2/M2 Pro/Max (24-96GB) — balanced
    M2,
    /// Apple M2 Ultra/M3 Max (96-192GB) — high precision
    M2Ultra,
    /// Apple M3 Ultra (256-512GB) — maximum precision, batched layout
    M3Ultra,
}

impl HardwareTarget {
    /// Auto-detect the current machine's target.
    pub fn detect() -> Self {
        let ram_mb = crate::gpu_memory::total_physical_ram_mb();
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);

        match (ram_mb, cpu_count) {
            (r, c) if r >= 393_216 && c >= 24 => Self::M3Ultra,
            (r, c) if r >= 131_072 && c >= 20 => Self::M2Ultra,
            (r, c) if r >= 65_536 && c >= 12 => Self::M2,
            (r, _c) if r >= 32_768 => Self::M1Pro,
            _ => Self::M1,
        }
    }

    /// Optimal quantization for this hardware.
    pub fn recommended_quant(&self) -> &'static str {
        match self {
            Self::M1 => "nf4-128",
            Self::M1Pro => "nf4-64",
            Self::M2 => "nf4-64",
            Self::M2Ultra => "8bit",
            Self::M3Ultra => "none",
        }
    }

    /// Whether weight streaming is beneficial (low RAM systems).
    pub fn needs_weight_streaming(&self) -> bool {
        matches!(self, Self::M1 | Self::M1Pro)
    }

    /// Recommended batch size for prefill+decode.
    pub fn recommended_batch(&self) -> u32 {
        match self {
            Self::M1 => 4,
            Self::M1Pro => 8,
            Self::M2 => 12,
            Self::M2Ultra => 20,
            Self::M3Ultra => 32,
        }
    }

    /// Number of ANE cores available for speculation.
    pub fn ane_cores(&self) -> u32 {
        match self {
            Self::M1 | Self::M1Pro | Self::M2 => 16,
            Self::M2Ultra | Self::M3Ultra => 32,
        }
    }

    /// Segment layout: small + many for streaming, large + few for batched.
    pub fn segment_target_size_mb(&self) -> u32 {
        match self {
            Self::M1 => 64,
            Self::M1Pro => 128,
            Self::M2 => 256,
            Self::M2Ultra => 512,
            Self::M3Ultra => 1024,
        }
    }
}

// ── Layer 3: Compiled Execution Specification ──────────────────────────────

/// Full execution plan: one spec per layer, plus global tensors.
#[derive(Debug, Serialize, Clone)]
pub struct ExecutionSpec {
    pub architecture: TextArchitecture,
    pub namespace: NamespaceBinding,
    pub global_tensors: Vec<TensorBinding>,
    pub layers: Vec<LayerSpec>,
    pub quantization: Option<QuantizationMeta>,
}

/// A layer's complete specification.
#[derive(Clone, Debug, Serialize)]
pub struct LayerSpec {
    pub index: u32,
    pub attention_kind: AttentionKind,
    pub q_out: u32,
    pub kv_out: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub global_kv_out: Option<u32>,
    pub n_global_kv_heads: Option<u32>,
    pub global_head_dim: Option<u32>,
    pub rope_theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
    pub sliding_window: Option<u32>,
    pub tensors: Vec<TensorBinding>,
}

/// A single tensor's expected identity in the safetensors file.
#[derive(Clone, Debug, Serialize)]
pub struct TensorBinding {
    pub name: String,
    pub role: TensorRole,
    pub logical_shape: Vec<u32>,
    /// If quantized: the packed weight shape (i8→u32 packing).
    pub packed_shape: Option<PackedLinearShapes>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PackedLinearShapes {
    pub weight: Vec<u32>,
    pub scales: Vec<u32>,
    pub biases: Vec<u32>,
    pub bits: u32,
    pub group_size: u32,
    pub groups: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum TensorRole {
    Embedding,
    FinalNorm,
    LmHead,
    AttentionNorm,
    FfnNorm,
    QProj,
    KProj,
    VProj,
    OProj,
    GlobalKProj,
    GlobalVProj,
    GateProj,
    UpProj,
    DownProj,
    QNorm,
    KNorm,
}

// ── Compiler

/// Complete model execution plan emitted by the compiler.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelExecutionPlan {
    pub prologue: ProloguePlan,
    pub layers: Vec<LayerPlan>,
    pub epilogue: EpiloguePlan,
    /// Fused ANE regions compiled to .mlmodelc artifacts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fused_ane_islands: Vec<AneFusedIsland>,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub sliding_window: u32,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: bool,
    pub rms_norm_eps: f64,
    /// Speculative decoding config when this image is a paired draft+target compile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_config: Option<SpeculativeModelConfig>,
    /// Generation regime (autoregressive or discrete diffusion).
    #[serde(default)]
    pub generation_regime: GenerationRegime,
    /// Diffusion configuration, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_config: Option<DiffusionConfig>,
    /// Diffusion execution plan, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_execution_plan: Option<DiffusionExecutionPlan>,
    /// KV cache mode for generation.
    #[serde(default)]
    pub kv_cache_mode: KvCacheMode,
}

/// Segment ID containing the embedding table.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProloguePlan {
    /// Segment ID containing the embedding table.
    pub segment_id: String,
    /// Tensor entry ID for the embedding weights.
    pub embedding_tensor_id: u32,
    /// Name used for ARRAY_REGISTRY lookup (e.g. "model.embed_tokens.weight").
    pub embedding_name: String,
    /// Expected embedding shape [vocab_size, hidden_size].
    pub embedding_shape: Vec<u32>,
    pub embedding_dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPlan {
    pub layer_index: u32,
    pub attention_kind: String,
    pub segment_id: String,
    pub hidden_size: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// For global layers only.
    pub global_head_dim: Option<u32>,
    pub n_global_kv_heads: Option<u32>,
    pub sliding_window: u32,
    pub rope_theta: f32,
    pub partial_rotary_factor: Option<f32>,
    pub attention_k_eq_v: bool,
    pub q_norm_enabled: bool,
    pub k_norm_enabled: bool,
    /// Tensor IDs for this layer's weights in the tensor_table.
    pub q_proj_tensor_id: u32,
    pub k_proj_tensor_id: u32,
    pub v_proj_tensor_id: u32,
    pub o_proj_tensor_id: u32,
    pub q_norm_tensor_id: Option<u32>,
    pub k_norm_tensor_id: Option<u32>,
    pub gate_proj_tensor_id: u32,
    pub up_proj_tensor_id: u32,
    pub down_proj_tensor_id: u32,
    pub input_layernorm_tensor_id: u32,
    pub post_attention_layernorm_tensor_id: u32,
    pub pre_ffw_layernorm_tensor_id: Option<u32>,
    pub post_ffw_layernorm_tensor_id: Option<u32>,
    /// Layer scalars and other optional tensors.
    pub layer_scalar_ids: Vec<u32>,
    /// Quantization descriptor IDs for packed weight groups.
    pub quantization_ids: Vec<String>,
    /// Per-operation backend routing for heterogeneous dispatch.
    #[serde(default)]
    pub route: operation_route::OperationRoute,
    /// Fused operations detected at compile time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fused_operations: Vec<FusedOperation>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpiloguePlan {
    pub segment_id: String,
    pub final_norm_tensor_id: u32,
    pub final_norm_name: String,
    pub output_projection_tensor_id: Option<u32>,
    pub output_projection_name: Option<String>,
    pub final_logit_softcapping: Option<f64>,
    pub vocab_size: u32,
}

/// A fused ANE region compiled to a single .mlmodelc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneFusedIsland {
    pub island_id: String,
    pub modelc_relpath: String,
    pub layer_indices: Vec<u32>,
    pub compute_units: String,
    pub function_name: String,
    /// Semantic subgraph kind for this fused island.
    #[serde(default)]
    pub subgraph_kind: String,
}

/// A fused operation composed of multiple atomic operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusedOperation {
    FusedNormQProj,
    FusedNormKProj,
    FusedNormVProj,
    FusedFfnActivation,
    FusedResidualNorm,
    FusedFlashAttention,
    FusedMoERoute,
    Custom(String),
}

impl FusedOperation {
    /// Return the name of the precompiled Metal kernel.
    pub fn kernel_name(&self) -> &str {
        match self {
            Self::FusedNormQProj => "fused_norm_q_proj",
            Self::FusedNormKProj => "fused_norm_k_proj",
            Self::FusedNormVProj => "fused_norm_v_proj",
            Self::FusedFfnActivation => "fused_ffn_activation",
            Self::FusedResidualNorm => "fused_residual_norm",
            Self::FusedFlashAttention => "fused_flash_attention",
            Self::FusedMoERoute => "fused_moe_route",
            Self::Custom(name) => name.as_str(),
        }
    }
}

impl LayerPlan {
    /// Return the logical operation names for this layer in execution order.
    pub fn operation_names(&self) -> Vec<&'static str> {
        let mut ops = Vec::with_capacity(16);

        if self.input_layernorm_tensor_id != 0 {
            ops.push("rms_norm");
        }
        if self.q_proj_tensor_id != 0 {
            ops.push("q_proj");
        }
        if self.k_proj_tensor_id != 0 {
            ops.push("k_proj");
        }
        if self.v_proj_tensor_id != 0 {
            ops.push("v_proj");
        }

        if self.q_proj_tensor_id != 0 && self.k_proj_tensor_id != 0 {
            ops.push("matmul");
            ops.push("softmax");
        }
        if self.q_proj_tensor_id != 0 && self.v_proj_tensor_id != 0 {
            ops.push("matmul");
        }

        if self.o_proj_tensor_id != 0 {
            ops.push("add");
        }
        if self.post_attention_layernorm_tensor_id != 0 {
            ops.push("rms_norm");
        }

        if self.gate_proj_tensor_id != 0 {
            ops.push("gate_proj");
            ops.push("silu");
        }
        if self.up_proj_tensor_id != 0 {
            ops.push("multiply");
        }
        if self.down_proj_tensor_id != 0 {
            ops.push("down_proj");
        }

        ops
    }
}

/// Check if `ops` contains the contiguous sequence of operation names in
/// `pattern`.  Returns `true` when every element of `pattern` appears in
/// order as a subsequence of `ops`.
fn has_pattern(ops: &[&str], pattern: &[&str]) -> bool {
    if pattern.is_empty() {
        return true;
    }
    let mut pi = 0;
    for &op in ops {
        if op == pattern[pi] {
            pi += 1;
            if pi == pattern.len() {
                return true;
            }
        }
    }
    false
}

impl ModelExecutionPlan {
    /// Scan adjacent layers with ANE-routed ops and populate fused_ane_islands.
    pub fn build_ane_fusion_plan(&mut self) {
        let mut islands: Vec<AneFusedIsland> = Vec::new();
        let mut i = 0;
        while i < self.layers.len() {
            let is_ane = self.layers[i].route.has_ane_backend();
            if !is_ane {
                i += 1;
                continue;
            }
            let mut layer_indices = vec![self.layers[i].layer_index];
            i += 1;
            while i < self.layers.len() && self.layers[i].route.has_ane_backend() {
                layer_indices.push(self.layers[i].layer_index);
                i += 1;
            }
            if layer_indices.len() >= 2 {
                let island_id = format!(
                    "ane_fused_layer{}-{}",
                    layer_indices[0],
                    layer_indices.last().unwrap()
                );
                let modelc_path = format!("{}.modelc", island_id);
                let first_idx = layer_indices[0] as usize;
                let first_ops = self.layers[first_idx].operation_names();
                let subgraph_kind =
                    if first_ops.contains(&"gate_proj") && first_ops.contains(&"down_proj") {
                        "mlp_block".to_string()
                    } else if first_ops.contains(&"q_proj")
                        && first_ops.contains(&"k_proj")
                        && first_ops.contains(&"v_proj")
                        && !first_ops.contains(&"rms_norm")
                    {
                        "qkv_bundle".to_string()
                    } else if first_ops.contains(&"rms_norm") && first_ops.contains(&"q_proj") {
                        "rmsnorm_qkv".to_string()
                    } else if first_ops.contains(&"lm_head") {
                        "output_proj".to_string()
                    } else {
                        "mlp_block".to_string()
                    };
                islands.push(AneFusedIsland {
                    island_id,
                    modelc_relpath: modelc_path,
                    layer_indices,
                    compute_units: "cpuAndNeuralEngine".to_string(),
                    function_name: "main".to_string(),
                    subgraph_kind,
                });
            }
        }
        self.fused_ane_islands = islands;
    }
}

/// Config for a model pair compiled for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeModelConfig {
    /// Draft model architecture config
    pub draft_architecture: TextArchitecture,
    /// Target model architecture config
    pub target_architecture: TextArchitecture,
    /// Shared components
    pub shared_embedding: bool,
    pub shared_lm_head: bool,
    /// Segment ordering: draft layers come first for fast startup
    pub draft_first_segments: bool,
    /// Maximum draft speculation length
    pub speculation_length: u32,
}

impl ModelExecutionPlan {
    /// Post-plan fusion pass: detect common operation patterns and
    /// annotate layers with fused operations.
    pub fn apply_fusion_pass(&mut self) {
        let pattern_norm_q = &["rms_norm", "q_proj"];
        let pattern_norm_k = &["rms_norm", "k_proj"];
        let pattern_norm_v = &["rms_norm", "v_proj"];
        let pattern_silu_mul = &["silu", "multiply"];
        let pattern_add_norm = &["add", "rms_norm"];
        let pattern_mm_soft_mm = &["matmul", "softmax", "matmul"];

        for layer in &mut self.layers {
            let mut fused = Vec::new();
            let ops = layer.operation_names();

            if has_pattern(&ops, pattern_norm_q) {
                fused.push(FusedOperation::FusedNormQProj);
            }
            if has_pattern(&ops, pattern_norm_k) {
                fused.push(FusedOperation::FusedNormKProj);
            }
            if has_pattern(&ops, pattern_norm_v) {
                fused.push(FusedOperation::FusedNormVProj);
            }
            if has_pattern(&ops, pattern_silu_mul) {
                fused.push(FusedOperation::FusedFfnActivation);
            }
            if has_pattern(&ops, pattern_add_norm) {
                fused.push(FusedOperation::FusedResidualNorm);
            }
            if has_pattern(&ops, pattern_mm_soft_mm) {
                fused.push(FusedOperation::FusedFlashAttention);
            }

            layer.fused_operations = fused;
        }
    }

    /// Validate the execution plan consistency.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.layers.is_empty() {
            errors.push("execution plan has zero layers".into());
        }
        for (i, plan) in self.layers.iter().enumerate() {
            if plan.layer_index != i as u32 {
                errors.push(format!("layer {} has index {}", i, plan.layer_index));
            }
            if plan.hidden_size != self.hidden_size {
                errors.push(format!(
                    "layer {} hidden_size {} != model {}",
                    i, plan.hidden_size, self.hidden_size
                ));
            }
            if plan.q_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero q_proj_tensor_id", i));
            }
            if plan.k_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero k_proj_tensor_id", i));
            }
            if plan.o_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero o_proj_tensor_id", i));
            }
            if plan.gate_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero gate_proj_tensor_id", i));
            }
            if plan.up_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero up_proj_tensor_id", i));
            }
            if plan.down_proj_tensor_id == 0 {
                errors.push(format!("layer {} has zero down_proj_tensor_id", i));
            }
            if plan.input_layernorm_tensor_id == 0 {
                errors.push(format!("layer {} has zero input_layernorm_tensor_id", i));
            }
            if plan.post_attention_layernorm_tensor_id == 0 {
                errors.push(format!(
                    "layer {} has zero post_attention_layernorm_tensor_id",
                    i
                ));
            }
            match plan.attention_kind.as_str() {
                "sliding_attention" => {
                    if plan.v_proj_tensor_id == 0 {
                        errors.push(format!("sliding layer {} has zero v_proj_tensor_id", i));
                    }
                }
                "full_attention" => {
                    if plan.global_head_dim.is_none() {
                        errors.push(format!(
                            "full-attention layer {} missing global_head_dim",
                            i
                        ));
                    }
                }
                other => {
                    errors.push(format!("layer {} has unknown attention_kind: {}", i, other));
                }
            }
            let expected_seg = format!("layer_{}", i);
            if plan.segment_id != expected_seg {
                errors.push(format!(
                    "layer {} segment_id '{}' != expected '{}'",
                    i, plan.segment_id, expected_seg
                ));
            }
        }
        if self.epilogue.final_norm_tensor_id == 0 {
            errors.push("epilogue has zero final_norm_tensor_id".into());
        }
        if self.epilogue.vocab_size == 0 {
            errors.push("epilogue has zero vocab_size".into());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Build a ModelExecutionPlan from the TextArchitecture, namespace, and emitted tensor IDs.
pub fn build_execution_plan(
    arch: &TextArchitecture,
    namespace: &NamespaceBinding,
    emitted_ids: &std::collections::HashMap<String, u32>,
) -> ModelExecutionPlan {
    let root = &namespace.root;
    let mut layers = Vec::with_capacity(arch.layer_types.len());

    for (i, kind) in arch.layer_types.iter().enumerate() {
        let layer = i as u32;
        let base = format!("{}.layers.{}", root, layer);
        let is_full = *kind == AttentionKind::FullAttention;

        let get = |suffix: &str| -> u32 {
            let name = format!("{}.{}", base, suffix);
            emitted_ids.get(&name).copied().unwrap_or(0)
        };
        let get_opt = |suffix: &str| -> Option<u32> {
            let name = format!("{}.{}", base, suffix);
            emitted_ids.get(&name).copied()
        };

        let rope = if is_full {
            arch.rope_global.as_ref().unwrap_or(&arch.rope_local)
        } else {
            &arch.rope_local
        };

        let hdim = if is_full {
            arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.head_dim
        };
        let n_kv = if is_full {
            arch.num_global_key_value_heads
                .unwrap_or(arch.num_key_value_heads)
        } else {
            arch.num_key_value_heads
        };

        layers.push(LayerPlan {
            layer_index: layer,
            attention_kind: if is_full {
                "full_attention".into()
            } else {
                "sliding_attention".into()
            },
            segment_id: format!("layer_{}", layer),
            hidden_size: arch.hidden_size,
            n_heads: arch.num_attention_heads,
            n_kv_heads: n_kv,
            head_dim: hdim,
            global_head_dim: if is_full { arch.global_head_dim } else { None },
            n_global_kv_heads: if is_full {
                arch.num_global_key_value_heads
            } else {
                None
            },
            sliding_window: arch.sliding_window,
            rope_theta: rope.theta as f32,
            partial_rotary_factor: rope.partial_rotary_factor.map(|f| f as f32),
            attention_k_eq_v: arch.attention_k_eq_v && is_full,
            q_norm_enabled: true,
            k_norm_enabled: true,
            q_proj_tensor_id: get("self_attn.q_proj.weight"),
            k_proj_tensor_id: get("self_attn.k_proj.weight"),
            v_proj_tensor_id: if is_full {
                get("self_attn.k_proj.weight")
            } else {
                get("self_attn.v_proj.weight")
            },
            o_proj_tensor_id: get("self_attn.o_proj.weight"),
            q_norm_tensor_id: get_opt("self_attn.q_norm.weight"),
            k_norm_tensor_id: get_opt("self_attn.k_norm.weight"),
            gate_proj_tensor_id: get("mlp.gate_proj.weight"),
            up_proj_tensor_id: get("mlp.up_proj.weight"),
            down_proj_tensor_id: get("mlp.down_proj.weight"),
            input_layernorm_tensor_id: get("input_layernorm.weight"),
            post_attention_layernorm_tensor_id: get("post_attention_layernorm.weight"),
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: Vec::new(),
            quantization_ids: Vec::new(),
            route: Default::default(),
            fused_operations: Vec::new(),
        });
    }

    let embed_name = format!("{}.embed_tokens.weight", root);
    let fn_name = format!("{}.norm.weight", root);
    let lm_head_name = namespace.lm_head_key.clone();

    for layer in &mut layers {
        let is_full = layer.attention_kind == "full_attention";
        layer.route = operation_route::OperationRoute {
            rms_norm: 1,
            silu: 0,
            matmul: 0,
            attention: if is_full { 3 } else { 0 },
            softmax: 0,
            rope: 0,
            add: 1,
            multiply: 1,
            transpose: 0,
            reshape: 1,
        };
    }

    ModelExecutionPlan {
        prologue: ProloguePlan {
            segment_id: "persistent".into(),
            embedding_tensor_id: emitted_ids.get(&embed_name).copied().unwrap_or(0),
            embedding_name: embed_name,
            embedding_shape: vec![arch.vocab_size, arch.hidden_size],
            embedding_dtype: if arch.model_type == "qwen2" {
                "BF16".into()
            } else {
                "F32".into()
            },
        },
        layers,
        epilogue: EpiloguePlan {
            segment_id: "persistent".into(),
            final_norm_tensor_id: emitted_ids.get(&fn_name).copied().unwrap_or(0),
            final_norm_name: fn_name,
            output_projection_tensor_id: emitted_ids.get(&lm_head_name).copied(),
            output_projection_name: Some(lm_head_name),
            final_logit_softcapping: arch.final_logit_softcapping,
            vocab_size: arch.vocab_size,
        },
        fused_ane_islands: vec![],
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
        sliding_window: arch.sliding_window,
        final_logit_softcapping: arch.final_logit_softcapping,
        tie_word_embeddings: arch.tie_word_embeddings,
        rms_norm_eps: arch.rms_norm_eps,
        ..Default::default()
    }
}

/// Compile a TextArchitecture into an ExecutionSpec.
pub fn compile(
    arch: &TextArchitecture,
    namespace: &NamespaceBinding,
    q: Option<&QuantizationMeta>,
) -> ExecutionSpec {
    let mut spec = ExecutionSpec {
        architecture: arch.clone(),
        namespace: NamespaceBinding {
            root: namespace.root.clone(),
            discovery: namespace.discovery.clone(),
            lm_head_key: namespace.lm_head_key.clone(),
            lm_head_aliased: namespace.lm_head_aliased,
        },
        global_tensors: Vec::new(),
        layers: Vec::new(),
        quantization: q.cloned(),
    };

    let root = &namespace.root;
    let bits = q.as_ref().map(|m| m.bits).unwrap_or(0);
    let gs = q.map(|m| m.group_size).unwrap_or(64);

    // Embedding
    spec.global_tensors.push(TensorBinding {
        name: format!("{}.embed_tokens.weight", root),
        role: TensorRole::Embedding,
        logical_shape: vec![arch.vocab_size, arch.hidden_size],
        packed_shape: if q.is_some() {
            let gs = q.as_ref().map(|m| m.group_size).unwrap_or(64);
            let bits = q.as_ref().map(|m| m.bits).unwrap_or(16);
            let pack = 32 / bits;
            let packed_in = arch.hidden_size / pack;
            let n_groups = arch.hidden_size / gs;
            Some(PackedLinearShapes {
                weight: vec![arch.vocab_size, packed_in],
                scales: vec![arch.vocab_size, n_groups],
                biases: vec![arch.vocab_size, n_groups],
                bits,
                group_size: gs,
                groups: n_groups,
            })
        } else {
            None
        },
    });

    // Final norm
    spec.global_tensors.push(TensorBinding {
        name: format!("{}.norm.weight", root),
        role: TensorRole::FinalNorm,
        logical_shape: vec![arch.hidden_size],
        packed_shape: None,
    });

    // LM head
    if !arch.tie_word_embeddings {
        spec.global_tensors.push(TensorBinding {
            name: format!("{}.lm_head.weight", root),
            role: TensorRole::LmHead,
            logical_shape: vec![arch.vocab_size, arch.hidden_size],
            packed_shape: None,
        });
    }

    // Per-layer compilation
    for (i, kind) in arch.layer_types.iter().enumerate() {
        let layer = i as u32;
        let is_full = *kind == AttentionKind::FullAttention;

        let rope = if is_full {
            arch.rope_global.as_ref().unwrap_or(&arch.rope_local)
        } else {
            &arch.rope_local
        };

        let mut tensors = Vec::new();

        // Attention norms
        tensors.push(norm_binding(
            root,
            layer,
            "input_layernorm",
            TensorRole::AttentionNorm,
            arch.hidden_size,
        ));
        tensors.push(norm_binding(
            root,
            layer,
            "post_attention_layernorm",
            TensorRole::FfnNorm,
            arch.hidden_size,
        ));

        // QK norms
        let norm_dim = if is_full {
            arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.head_dim
        };
        tensors.push(TensorBinding {
            name: format!("{}.layers.{}.self_attn.q_norm.weight", root, layer),
            role: TensorRole::QNorm,
            logical_shape: vec![norm_dim],
            packed_shape: None,
        });
        tensors.push(TensorBinding {
            name: format!("{}.layers.{}.self_attn.k_norm.weight", root, layer),
            role: TensorRole::KNorm,
            logical_shape: vec![norm_dim],
            packed_shape: None,
        });

        // QKV projections
        let actual_kv_out = if is_full {
            arch.num_global_key_value_heads.unwrap_or(1)
                * arch.global_head_dim.unwrap_or(arch.head_dim)
        } else {
            arch.num_key_value_heads * arch.head_dim
        };
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.q_proj",
            TensorRole::QProj,
            if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.k_proj",
            TensorRole::KProj,
            actual_kv_out,
            arch.hidden_size,
            gs,
            bits,
        ));
        if !is_full {
            tensors.push(quantized_linear(
                root,
                layer,
                "self_attn.v_proj",
                TensorRole::VProj,
                arch.num_key_value_heads * arch.head_dim,
                arch.hidden_size,
                gs,
                bits,
            ));
        }
        tensors.push(quantized_linear(
            root,
            layer,
            "self_attn.o_proj",
            TensorRole::OProj,
            arch.hidden_size,
            if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            gs,
            bits,
        ));

        // MLP
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.gate_proj",
            TensorRole::GateProj,
            arch.intermediate_size,
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.up_proj",
            TensorRole::UpProj,
            arch.intermediate_size,
            arch.hidden_size,
            gs,
            bits,
        ));
        tensors.push(quantized_linear(
            root,
            layer,
            "mlp.down_proj",
            TensorRole::DownProj,
            arch.hidden_size,
            arch.intermediate_size,
            gs,
            bits,
        ));

        let sliding_window = if is_full {
            None
        } else {
            Some(arch.sliding_window)
        };

        spec.layers.push(LayerSpec {
            index: layer,
            attention_kind: kind.clone(),
            q_out: if is_full {
                arch.num_attention_heads * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_attention_heads * arch.head_dim
            },
            kv_out: if is_full {
                arch.num_global_key_value_heads.unwrap_or(1)
                    * arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.num_key_value_heads * arch.head_dim
            },
            n_heads: arch.num_attention_heads,
            n_kv_heads: arch.num_key_value_heads,
            head_dim: if is_full {
                arch.global_head_dim.unwrap_or(arch.head_dim)
            } else {
                arch.head_dim
            },
            global_kv_out: if is_full {
                Some(
                    arch.num_global_key_value_heads.unwrap_or(1)
                        * arch.global_head_dim.unwrap_or(arch.head_dim),
                )
            } else {
                None
            },
            n_global_kv_heads: arch.num_global_key_value_heads,
            global_head_dim: arch.global_head_dim,
            rope_theta: rope.theta,
            rope_type: rope.rope_type.clone(),
            partial_rotary_factor: rope.partial_rotary_factor,
            sliding_window,
            tensors,
        });
    }

    spec
}

/// Filter the compiled spec to only include bindings for tensors that exist
/// in the source model's tensor map.
pub fn filter_spec_to_existing(spec: &mut ExecutionSpec, existing_tensor_names: &HashSet<String>) {
    spec.global_tensors.retain(|b| {
        if existing_tensor_names.contains(&b.name) {
            true
        } else {
            eprintln!(
                "[dynamic-compile] skipping missing global tensor: {}",
                b.name
            );
            false
        }
    });

    for layer in spec.layers.iter_mut() {
        layer.tensors.retain(|b| {
            if existing_tensor_names.contains(&b.name) {
                true
            } else {
                eprintln!(
                    "[dynamic-compile] skipping missing layer tensor: {}",
                    b.name
                );
                false
            }
        });
    }
}

fn norm_binding(root: &str, layer: u32, name: &str, role: TensorRole, dim: u32) -> TensorBinding {
    TensorBinding {
        name: format!("{}.layers.{}.{}.weight", root, layer, name),
        role,
        logical_shape: vec![dim],
        packed_shape: None,
    }
}

fn quantized_linear(
    root: &str,
    layer: u32,
    proj_name: &str,
    role: TensorRole,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    bits: u32,
) -> TensorBinding {
    let packed_shape = if bits > 0 && bits <= 16 {
        let pack = 32 / bits;
        let packed_in = in_dim / pack;
        let n_groups = in_dim / group_size;

        Some(PackedLinearShapes {
            weight: vec![out_dim, packed_in],
            scales: vec![out_dim, n_groups],
            biases: vec![out_dim, n_groups],
            bits,
            group_size,
            groups: n_groups,
        })
    } else {
        None
    };

    TensorBinding {
        name: format!("{}.layers.{}.{}.weight", root, layer, proj_name),
        role,
        logical_shape: vec![out_dim, in_dim],
        packed_shape,
    }
}
