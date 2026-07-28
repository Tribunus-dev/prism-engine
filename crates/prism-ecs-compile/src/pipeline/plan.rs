//! `pipeline::plan` — model and layer plan types for the compile pipeline.
//!
//! This file owns the canonical authority for the plan types consumed by
//! the constitutional pipeline optimizers and schedulers. The engine's
//! `config::hardware` module had parallel plan types; those are being
//! replaced with these constitutional types in a follow-up migration.

use serde::{Deserialize, Serialize};

/// Attention kind for a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    /// Full attention over the full sequence.
    FullAttention,
    /// Sliding-window attention over a local window.
    SlidingAttention,
}

impl Default for AttentionKind {
    fn default() -> Self {
        Self::SlidingAttention
    }
}

/// RoPE configuration for one attention layer family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    /// Base theta for the rotary embedding.
    pub theta: f64,
    /// RoPE type identifier (e.g. "default", "linear").
    pub rope_type: String,
    /// Partial rotary factor, if any.
    pub partial_rotary_factor: Option<f32>,
}

impl Default for RopeSpec {
    fn default() -> Self {
        Self {
            theta: 500_000.0,
            rope_type: "default".into(),
            partial_rotary_factor: None,
        }
    }
}

/// Prologue plan (embedding, optional pre-norm).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProloguePlan {
    /// Identifier of the embedding tensor; 0 means no embedding.
    pub embedding_tensor_id: u64,
}

/// Epilogue plan (final norm, output projection).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EpiloguePlan {
    /// Identifier of the final norm tensor; 0 means absent.
    pub final_norm_tensor_id: u64,
}

/// Operation route — where each op in the layer prefers to execute.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OperationRoute {
    /// Dominant backend id (0 = Metal, 1 = Accelerate, 2 = ANE, 3 = MLX).
    pub dominant_backend: u32,
    /// Per-op preferred backend ids.
    pub per_op_backend: Vec<u32>,
}

impl OperationRoute {
    /// Return the dominant backend id.
    pub fn dominant_backend(&self) -> u32 {
        self.dominant_backend
    }
}

/// Per-layer execution plan — one transformer layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerPlan {
    /// Layer index (0-based).
    pub layer_index: u32,
    /// Attention kind for this layer.
    pub attention_kind: String,
    /// Segment identifier (e.g. "weights", "shared").
    pub segment_id: String,
    /// Hidden size for this layer.
    pub hidden_size: u32,
    /// Number of attention heads.
    pub n_heads: u32,
    /// Number of key-value heads.
    pub n_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Global head dimension (for global attention layers).
    pub global_head_dim: Option<u32>,
    /// Number of global key-value heads.
    pub n_global_kv_heads: Option<u32>,
    /// Sliding window size in tokens.
    pub sliding_window: u32,
    /// RoPE theta for this layer.
    pub rope_theta: f64,
    /// Partial rotary factor for this layer.
    pub partial_rotary_factor: Option<f32>,
    /// Whether attention K equals V (no K/V split).
    pub attention_k_eq_v: bool,
    /// Whether Q is normed.
    pub q_norm_enabled: bool,
    /// Whether K is normed.
    pub k_norm_enabled: bool,
    /// Tensor id of the Q projection.
    pub q_proj_tensor_id: u64,
    /// Tensor id of the K projection.
    pub k_proj_tensor_id: u64,
    /// Tensor id of the V projection.
    pub v_proj_tensor_id: u64,
    /// Tensor id of the O projection.
    pub o_proj_tensor_id: u64,
    /// Tensor id of the Q norm (if any).
    pub q_norm_tensor_id: Option<u64>,
    /// Tensor id of the K norm (if any).
    pub k_norm_tensor_id: Option<u64>,
    /// Tensor id of the gate projection.
    pub gate_proj_tensor_id: u64,
    /// Tensor id of the up projection.
    pub up_proj_tensor_id: u64,
    /// Tensor id of the down projection.
    pub down_proj_tensor_id: u64,
    /// Tensor id of the input layer norm.
    pub input_layernorm_tensor_id: u64,
    /// Tensor id of the post-attention layer norm.
    pub post_attention_layernorm_tensor_id: u64,
    /// Tensor id of the pre-FFW layer norm (if any).
    pub pre_ffw_layernorm_tensor_id: Option<u64>,
    /// Tensor id of the post-FFW layer norm (if any).
    pub post_ffw_layernorm_tensor_id: Option<u64>,
    /// Layer scalar ids (e.g. residual scale, soft-cap).
    pub layer_scalar_ids: Vec<u64>,
    /// Per-tensor quantization ids.
    pub quantization_ids: Vec<u64>,
    /// Operation route for this layer.
    pub route: OperationRoute,
    /// Fused operations in this layer.
    pub fused_operations: Vec<String>,
}

impl LayerPlan {
    /// Return the dominant backend id (0=Metal, 1=Accelerate, 2=ANE, 3=MLX).
    pub fn dominant_backend(&self) -> u32 {
        self.route.dominant_backend()
    }
}

/// Fused ANE island descriptor.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AneFusedIsland {
    /// Island name.
    pub name: String,
    /// Operations in the island.
    pub operations: Vec<String>,
}

/// Generation regime for the model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerationRegime {
    /// KV cache mode identifier.
    pub kv_cache_mode: String,
}

/// Diffusion configuration for diffusion models.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DiffusionConfig {
    /// Whether the model is a diffusion model.
    pub enabled: bool,
}

/// Speculative decoding configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Whether speculative decoding is enabled.
    pub enabled: bool,
}

/// Full model execution plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelExecutionPlan {
    /// Prologue plan.
    pub prologue: ProloguePlan,
    /// Per-layer plans.
    pub layers: Vec<LayerPlan>,
    /// Epilogue plan.
    pub epilogue: EpiloguePlan,
    /// Fused ANE islands.
    pub fused_ane_islands: Vec<AneFusedIsland>,
    /// Hidden size of the model.
    pub hidden_size: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Sliding window in tokens.
    pub sliding_window: u32,
    /// Final logit soft-capping value, if any.
    pub final_logit_softcapping: Option<f32>,
    /// Whether input and output embeddings are tied.
    pub tie_word_embeddings: bool,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Speculative decoding configuration.
    pub speculative_config: Option<SpeculativeConfig>,
    /// Generation regime.
    pub generation_regime: GenerationRegime,
    /// Diffusion configuration.
    pub diffusion_config: DiffusionConfig,
    /// Diffusion execution plan placeholder.
    pub diffusion_execution_plan: DiffusionConfig,
    /// KV cache mode.
    pub kv_cache_mode: String,
}

impl Default for ModelExecutionPlan {
    fn default() -> Self {
        Self {
            prologue: ProloguePlan::default(),
            layers: Vec::new(),
            epilogue: EpiloguePlan::default(),
            fused_ane_islands: Vec::new(),
            hidden_size: 0,
            vocab_size: 0,
            sliding_window: 0,
            final_logit_softcapping: None,
            tie_word_embeddings: false,
            rms_norm_eps: 1e-6,
            speculative_config: None,
            generation_regime: GenerationRegime::default(),
            diffusion_config: DiffusionConfig::default(),
            diffusion_execution_plan: DiffusionConfig::default(),
            kv_cache_mode: String::new(),
        }
    }
}

/// Text architecture description (used by the schedule compiler).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextArchitecture {
    /// Diffusion configuration.
    pub diffusion_config: Option<DiffusionConfig>,
    /// MoE configuration.
    pub moe_config: MoEConfig,
    /// Hidden size.
    pub hidden_size: u32,
    /// Intermediate (FFN) size.
    pub intermediate_size: u32,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Number of key-value heads.
    pub num_key_value_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// Global head dimension (for global attention).
    pub global_head_dim: Option<u32>,
    /// Number of global key-value heads.
    pub num_global_key_value_heads: Option<u32>,
    /// Number of hidden layers.
    pub num_hidden_layers: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Sliding window in tokens.
    pub sliding_window: u32,
    /// Maximum position embeddings.
    pub max_position_embeddings: u32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Whether input/output embeddings are tied.
    pub tie_word_embeddings: bool,
    /// Whether attention K equals V.
    pub attention_k_eq_v: bool,
    /// Final logit soft-capping value.
    pub final_logit_softcapping: Option<f32>,
    /// Hidden size per layer input (for layer-specific dimensions).
    pub hidden_size_per_layer_input: u32,
    /// Per-layer attention kinds.
    pub layer_types: Vec<AttentionKind>,
    /// Local RoPE spec.
    pub rope_local: RopeSpec,
    /// Global RoPE spec.
    pub rope_global: Option<RopeSpec>,
    /// Model type identifier.
    pub model_type: String,
    /// Whether thinking mode is enabled.
    pub thinking_mode: bool,
}

/// MoE configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MoEConfig {
    /// Number of experts.
    pub num_experts: u32,
    /// Number of experts per token.
    pub num_experts_per_tok: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_execution_plan_default_is_empty() {
        let plan = ModelExecutionPlan::default();
        assert!(plan.layers.is_empty());
        assert_eq!(plan.hidden_size, 0);
    }

    #[test]
    fn layer_plan_dominant_backend() {
        let layer = LayerPlan {
            layer_index: 0,
            attention_kind: "sliding_attention".into(),
            segment_id: "weights".into(),
            hidden_size: 3840,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 120,
            global_head_dim: None,
            n_global_kv_heads: None,
            sliding_window: 8192,
            rope_theta: 500_000.0,
            partial_rotary_factor: None,
            attention_k_eq_v: false,
            q_norm_enabled: false,
            k_norm_enabled: false,
            q_proj_tensor_id: 0,
            k_proj_tensor_id: 0,
            v_proj_tensor_id: 0,
            o_proj_tensor_id: 0,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 0,
            up_proj_tensor_id: 0,
            down_proj_tensor_id: 0,
            input_layernorm_tensor_id: 0,
            post_attention_layernorm_tensor_id: 0,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: vec![],
            quantization_ids: vec![],
            route: OperationRoute {
                dominant_backend: 3,
                per_op_backend: vec![],
            },
            fused_operations: vec![],
        };
        assert_eq!(layer.dominant_backend(), 3);
    }

    #[test]
    fn text_architecture_default_is_zeroed() {
        let arch = TextArchitecture::default();
        assert_eq!(arch.hidden_size, 0);
        assert_eq!(arch.num_hidden_layers, 0);
    }
}
