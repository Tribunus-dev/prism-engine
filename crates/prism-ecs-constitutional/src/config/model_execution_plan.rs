//! Complete model execution plan emitted by the compiler.
//!
//! Authority: the canonical [`ModelExecutionPlan`], [`ProloguePlan`],
//! [`LayerPlan`], [`EpiloguePlan`], [`AneFusedIsland`],
//! [`SpeculativeModelConfig`], and [`FusedOperation`] types plus the
//! post-plan fusion pass and ANE island detection routines. The
//! data here is the input to dispatch, scheduling, and projection
//! rebuild — there is no canonical authority for these shapes
//! anywhere else in the engine.

use serde::{Deserialize, Serialize};

use super::architecture::{DiffusionConfig, DiffusionExecutionPlan, GenerationRegime, KvCacheMode, TextArchitecture};
use super::operation_route::OperationRoute;

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
    pub route: OperationRoute,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl ModelExecutionPlan {
    /// Scan adjacent layers with ANE-routed ops and populate
    /// `fused_ane_islands`.  Layers with `route.has_ane_backend() == true`
    /// are grouped into a single island when they appear consecutively.
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
                let first_idx = *layer_indices.first().unwrap_or(&0) as usize;
                let last_idx = *layer_indices.last().unwrap_or(&0);
                let island_id = format!("ane_fused_layer{}-{}", first_idx, last_idx);
                let modelc_path = format!("{}.modelc", island_id);
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

    /// Post-plan fusion pass: detect common operation patterns and
    /// annotate layers with fused operations.
    pub fn apply_fusion_pass(&mut self) {
        const PATTERN_NORM_Q: &[&str] = &["rms_norm", "q_proj"];
        const PATTERN_NORM_K: &[&str] = &["rms_norm", "k_proj"];
        const PATTERN_NORM_V: &[&str] = &["rms_norm", "v_proj"];
        const PATTERN_SILU_MUL: &[&str] = &["silu", "multiply"];
        const PATTERN_ADD_NORM: &[&str] = &["add", "rms_norm"];
        const PATTERN_MM_SOFT_MM: &[&str] = &["matmul", "softmax", "matmul"];

        for layer in &mut self.layers {
            let mut fused = Vec::new();
            let ops = layer.operation_names();

            if has_pattern(&ops, PATTERN_NORM_Q) {
                fused.push(FusedOperation::FusedNormQProj);
            }
            if has_pattern(&ops, PATTERN_NORM_K) {
                fused.push(FusedOperation::FusedNormKProj);
            }
            if has_pattern(&ops, PATTERN_NORM_V) {
                fused.push(FusedOperation::FusedNormVProj);
            }
            if has_pattern(&ops, PATTERN_SILU_MUL) {
                fused.push(FusedOperation::FusedFfnActivation);
            }
            if has_pattern(&ops, PATTERN_ADD_NORM) {
                fused.push(FusedOperation::FusedResidualNorm);
            }
            if has_pattern(&ops, PATTERN_MM_SOFT_MM) {
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

/// Check if `ops` contains the contiguous sequence of operation names
/// in `pattern`.  Returns `true` when every element of `pattern`
/// appears in order as a subsequence of `ops`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::operation_route::OperationRoute;

    fn sample_plan() -> ModelExecutionPlan {
        let route_ane = OperationRoute {
            matmul: 3,
            attention: 3,
            ..Default::default()
        };
        let route_mlx = OperationRoute::default();
        ModelExecutionPlan {
            hidden_size: 64,
            vocab_size: 128,
            sliding_window: 64,
            tie_word_embeddings: true,
            rms_norm_eps: 1e-6,
            layers: vec![
                LayerPlan {
                    layer_index: 0,
                    attention_kind: "sliding_attention".into(),
                    segment_id: "layer_0".into(),
                    hidden_size: 64,
                    n_heads: 2,
                    n_kv_heads: 1,
                    head_dim: 32,
                    global_head_dim: None,
                    n_global_kv_heads: None,
                    sliding_window: 64,
                    rope_theta: 10_000.0,
                    partial_rotary_factor: None,
                    attention_k_eq_v: true,
                    q_norm_enabled: true,
                    k_norm_enabled: true,
                    q_proj_tensor_id: 10,
                    k_proj_tensor_id: 11,
                    v_proj_tensor_id: 12,
                    o_proj_tensor_id: 13,
                    q_norm_tensor_id: None,
                    k_norm_tensor_id: None,
                    gate_proj_tensor_id: 14,
                    up_proj_tensor_id: 15,
                    down_proj_tensor_id: 16,
                    input_layernorm_tensor_id: 17,
                    post_attention_layernorm_tensor_id: 18,
                    pre_ffw_layernorm_tensor_id: None,
                    post_ffw_layernorm_tensor_id: None,
                    layer_scalar_ids: vec![],
                    quantization_ids: vec![],
                    route: route_mlx.clone(),
                    fused_operations: vec![],
                },
                LayerPlan {
                    layer_index: 1,
                    attention_kind: "full_attention".into(),
                    segment_id: "layer_1".into(),
                    hidden_size: 64,
                    n_heads: 2,
                    n_kv_heads: 1,
                    head_dim: 32,
                    global_head_dim: Some(32),
                    n_global_kv_heads: Some(1),
                    sliding_window: 0,
                    rope_theta: 1_000_000.0,
                    partial_rotary_factor: None,
                    attention_k_eq_v: true,
                    q_norm_enabled: true,
                    k_norm_enabled: true,
                    q_proj_tensor_id: 20,
                    k_proj_tensor_id: 21,
                    v_proj_tensor_id: 0,
                    o_proj_tensor_id: 22,
                    q_norm_tensor_id: None,
                    k_norm_tensor_id: None,
                    gate_proj_tensor_id: 23,
                    up_proj_tensor_id: 24,
                    down_proj_tensor_id: 25,
                    input_layernorm_tensor_id: 26,
                    post_attention_layernorm_tensor_id: 27,
                    pre_ffw_layernorm_tensor_id: None,
                    post_ffw_layernorm_tensor_id: None,
                    layer_scalar_ids: vec![],
                    quantization_ids: vec![],
                    route: route_ane,
                    fused_operations: vec![],
                },
            ],
            epilogue: EpiloguePlan {
                segment_id: "persistent".into(),
                final_norm_tensor_id: 100,
                final_norm_name: "model.language_model.norm.weight".into(),
                output_projection_tensor_id: None,
                output_projection_name: None,
                final_logit_softcapping: None,
                vocab_size: 128,
            },
            prologue: ProloguePlan {
                segment_id: "persistent".into(),
                embedding_tensor_id: 1,
                embedding_name: "model.language_model.embed_tokens.weight".into(),
                embedding_shape: vec![128, 64],
                embedding_dtype: "F32".into(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn build_ane_fusion_plan_groups_consecutive_ane_layers() {
        let mut plan = sample_plan();
        plan.build_ane_fusion_plan();
        // Only layer 1 is ANE, so no island (need >= 2).
        assert!(plan.fused_ane_islands.is_empty());

        // Make layer 0 also ANE.
        plan.layers[0].route.matmul = 3;
        plan.layers[0].route.attention = 3;
        plan.build_ane_fusion_plan();
        assert_eq!(plan.fused_ane_islands.len(), 1);
        assert_eq!(plan.fused_ane_islands[0].layer_indices, vec![0, 1]);
    }

    #[test]
    fn apply_fusion_pass_detects_norm_q_pattern() {
        let mut plan = sample_plan();
        plan.apply_fusion_pass();
        // Layer 0 has rms_norm + q_proj so it gets FusedNormQProj.
        assert!(plan.layers[0]
            .fused_operations
            .contains(&FusedOperation::FusedNormQProj));
    }

    #[test]
    fn validate_rejects_zero_layers() {
        let mut plan = sample_plan();
        plan.layers.clear();
        let result = plan.validate();
        assert!(result.is_err());
    }

    #[test]
    fn fused_operation_kernel_name_is_stable() {
        assert_eq!(FusedOperation::FusedNormQProj.kernel_name(), "fused_norm_q_proj");
        assert_eq!(
            FusedOperation::Custom("foo".into()).kernel_name(),
            "foo"
        );
    }
}
