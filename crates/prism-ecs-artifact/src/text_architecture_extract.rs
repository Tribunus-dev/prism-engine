//! Text architecture extraction from HuggingFace-style config JSON.
//!
//! This module owns the canonical authority for translating a model's
//! `config.json` (and any `text_config` sub-section it may have) into a
//! `TextArchitecture` value that downstream compile-time systems can
//! attach to the model entity.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The model's tensor layout (owned by the spatial IR / phase graph).
//! - The `Model` entity lifecycle (owned by the model deployment
//!   subsystem).
//! - Numerical precision policy (owned by quantization).
//!
//! The extractor is a pure function: `config -> TextArchitecture`. The
//! type itself is a `Component` so it can be attached to a `Model`
//! entity, but the module never touches the world directly. The caller
//! is responsible for staging the result through a `WorldTxn`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Diffusion-model configuration block, if the model has one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionConfig {
    pub max_diffusion_tokens: u32,
    pub default_denoising_steps: u32,
    pub noise_schedule: NoiseScheduleType,
    pub parallel_token_generation: u32,
    pub supports_images: bool,
    pub supports_video: bool,
    pub image_size: u32,
    pub patch_size: u32,
    pub max_context_length: u32,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub max_canvas_tokens: u32,
    pub timestep_embedding_dim: u32,
    pub confidence_type: ConfidenceType,
    pub default_confidence_threshold: f32,
    pub eos_collapse_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseScheduleType {
    Cosine,
    Sqrt,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceType {
    LogProb,
    SoftmaxMargin,
    NormalizedEntropy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoEConfig {
    pub num_experts: u32,
    pub top_k_experts: u32,
    pub intermediate_size: u32,
    pub shared_experts: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f64,
    pub rope_type: String,
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    pub max_position_embeddings: u64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f64>,
    pub hidden_size_per_layer_input: u32,
    pub layer_types: Vec<AttentionKind>,
    pub rope_local: RopeSpec,
    pub rope_global: Option<RopeSpec>,
    pub model_type: String,
    pub moe_config: Option<MoEConfig>,
    pub diffusion_config: Option<DiffusionConfig>,
    pub thinking_mode: bool,
}

impl prism_ecs_core::Component for TextArchitecture {}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TextArchitectureError {
    #[error("config is not a JSON object")]
    NotAnObject,
    #[error("missing required field `{field}` for model_type `{model_type}`")]
    MissingField { field: String, model_type: String },
}

/// Extract a `TextArchitecture` from a HuggingFace-style config JSON
/// value and a model-type tag.
pub fn extract(config: &serde_json::Value, model_type: &str) -> TextArchitecture {
    let cfg = if let Some(text_cfg) = config.get("text_config") {
        text_cfg
    } else {
        config
    };

    let hs = num(cfg, "hidden_size");
    let im = num(cfg, "intermediate_size");
    let n_heads = num(cfg, "num_attention_heads");
    let n_kv = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
    let n_layers = num(cfg, "num_hidden_layers");
    let vocab = num(cfg, "vocab_size");
    let max_pos = num_opt_u64(cfg, "max_position_embeddings").unwrap_or(2048);
    let sliding = num(cfg, "sliding_window");
    let eps = f64_val(cfg, "rms_norm_eps").unwrap_or(1e-6);
    let tied = bool_val(cfg, "tie_word_embeddings").unwrap_or(true);
    let final_softcap = f64_val(cfg, "final_logit_softcapping");

    let head_dim = num_opt(cfg, "head_dim").unwrap_or_else(|| {
        if hs > 0 && n_heads > 0 {
            hs / n_heads
        } else {
            0
        }
    });

    let global_hd = num_opt(cfg, "global_head_dim");
    let n_global_kv = num_opt(cfg, "num_global_key_value_heads");

    let attention_k_eq_v = bool_val(cfg, "attention_k_eq_v")
        .unwrap_or(matches!(model_type, "deepseek2" | "ds4" | "ds3"));

    let rope_theta = f64_val(cfg, "rope_theta").unwrap_or(10000.0);
    let rope_type = cfg
        .get("rope_type")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();
    let partial_rotary = f64_val(cfg, "partial_rotary_factor");

    let layer_types = if n_layers > 0 {
        let kind = if model_type == "deepseek2"
            || model_type == "ds4"
            || bool_val(cfg, "use_full_attention").unwrap_or(false)
        {
            AttentionKind::FullAttention
        } else {
            AttentionKind::SlidingAttention
        };
        vec![kind; n_layers as usize]
    } else {
        vec![]
    };

    let moe = extract_moe(cfg, model_type, im);
    let diffusion_config = extract_diffusion(cfg, hs);
    let hidden_size_per_layer_input = num_opt(cfg, "hidden_size_per_layer_input").unwrap_or(hs);

    TextArchitecture {
        hidden_size: hs,
        intermediate_size: im,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        head_dim,
        global_head_dim: global_hd,
        num_global_key_value_heads: n_global_kv,
        num_hidden_layers: n_layers,
        vocab_size: vocab,
        sliding_window: sliding,
        max_position_embeddings: max_pos,
        rms_norm_eps: eps,
        tie_word_embeddings: tied,
        attention_k_eq_v,
        final_logit_softcapping: final_softcap,
        hidden_size_per_layer_input,
        layer_types,
        rope_local: RopeSpec {
            theta: rope_theta,
            rope_type,
            partial_rotary_factor: partial_rotary,
        },
        rope_global: None,
        model_type: model_type.to_string(),
        moe_config: moe,
        diffusion_config,
        thinking_mode: false,
    }
}

fn num(v: &serde_json::Value, key: &str) -> u32 {
    num_opt(v, key).unwrap_or(0)
}

fn num_opt(v: &serde_json::Value, key: &str) -> Option<u32> {
    v.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn num_opt_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|v| v.as_u64())
}

fn f64_val(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|v| v.as_f64())
}

fn bool_val(v: &serde_json::Value, key: &str) -> Option<bool> {
    v.get(key).and_then(|v| v.as_bool())
}

fn extract_moe(
    cfg: &serde_json::Value,
    model_type: &str,
    intermediate_fallback: u32,
) -> Option<MoEConfig> {
    if let Some(moe_val) = cfg.get("moe_config").or_else(|| cfg.get("moe")) {
        return Some(MoEConfig {
            num_experts: num_opt(moe_val, "num_experts").unwrap_or(0),
            top_k_experts: num_opt(moe_val, "top_k_experts").unwrap_or(0),
            intermediate_size: num_opt(moe_val, "intermediate_size")
                .unwrap_or(intermediate_fallback),
            shared_experts: bool_val(moe_val, "shared_experts").unwrap_or(false),
        });
    }

    if model_type.starts_with("deepseek") || model_type == "ds4" {
        let num_experts = num_opt(cfg, "num_experts").unwrap_or(0);
        let top_k = num_opt(cfg, "top_k_experts")
            .or_else(|| num_opt(cfg, "num_routed_experts"))
            .or_else(|| num_opt(cfg, "num_experts").map(|n| n / 2))
            .unwrap_or(0);
        if num_experts > 0 {
            return Some(MoEConfig {
                num_experts,
                top_k_experts: top_k,
                intermediate_size: intermediate_fallback,
                shared_experts: bool_val(cfg, "shared_experts").unwrap_or(true),
            });
        }
    }

    None
}

fn extract_diffusion(cfg: &serde_json::Value, hidden_size_fallback: u32) -> Option<DiffusionConfig> {
    let d = cfg.get("diffusion_config").or_else(|| cfg.get("_diffusion"))?;
    let max_diffusion_tokens = d
        .get("max_diffusion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as u32;
    let default_denoising_steps = d
        .get("default_denoising_steps")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as u32;
    let noise_schedule = match d.get("noise_schedule").and_then(|v| v.as_str()) {
        Some("cosine") => NoiseScheduleType::Cosine,
        Some("sqrt") => NoiseScheduleType::Sqrt,
        Some("linear") => NoiseScheduleType::Linear,
        _ => NoiseScheduleType::Cosine,
    };
    Some(DiffusionConfig {
        max_diffusion_tokens,
        default_denoising_steps,
        noise_schedule,
        parallel_token_generation: d
            .get("parallel_token_generation")
            .and_then(|v| v.as_u64())
            .unwrap_or(18) as u32,
        supports_images: d
            .get("supports_images")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        supports_video: d
            .get("supports_video")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        image_size: d.get("image_size").and_then(|v| v.as_u64()).unwrap_or(896) as u32,
        patch_size: d.get("patch_size").and_then(|v| v.as_u64()).unwrap_or(16) as u32,
        max_context_length: d
            .get("max_context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(262_144) as u32,
        mask_token_id: d.get("mask_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        pad_token_id: d.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        eos_token_id: d.get("eos_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        max_canvas_tokens: d
            .get("max_canvas_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as u32,
        timestep_embedding_dim: d
            .get("timestep_embedding_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_size_fallback as u64) as u32,
        confidence_type: match d.get("confidence_type").and_then(|v| v.as_str()) {
            Some("softmax_margin") => ConfidenceType::SoftmaxMargin,
            Some("normalized_entropy") => ConfidenceType::NormalizedEntropy,
            _ => ConfidenceType::LogProb,
        },
        default_confidence_threshold: d
            .get("default_confidence_threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7) as f32,
        eos_collapse_enabled: d
            .get("eos_collapse_enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_full_llama_style_config() {
        let cfg = json!({
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "rope_theta": 10000.0,
            "rope_type": "llama3",
        });
        let a = extract(&cfg, "llama");
        assert_eq!(a.hidden_size, 4096);
        assert_eq!(a.num_attention_heads, 32);
        assert_eq!(a.num_key_value_heads, 32);
        assert_eq!(a.head_dim, 128);
        assert_eq!(a.num_hidden_layers, 32);
        assert_eq!(a.vocab_size, 32000);
        assert_eq!(a.max_position_embeddings, 4096);
        assert!((a.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert!(!a.tie_word_embeddings);
        assert_eq!(a.rope_local.rope_type, "llama3");
        assert_eq!(a.model_type, "llama");
        assert!(a.moe_config.is_none());
    }

    #[test]
    fn extract_text_config_nested_under_root() {
        let cfg = json!({
            "model_type": "qwen2_vl",
            "text_config": {
                "hidden_size": 5120,
                "intermediate_size": 13824,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "num_hidden_layers": 48,
                "vocab_size": 152064,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
            }
        });
        let a = extract(&cfg.get("text_config").unwrap(), "qwen2_vl_text");
        assert_eq!(a.hidden_size, 5120);
        assert_eq!(a.num_key_value_heads, 8);
        assert_eq!(a.num_attention_heads, 40);
        assert_eq!(a.rope_local.theta, 1_000_000.0);
    }

    #[test]
    fn extract_deepseek_uses_full_attention() {
        let cfg = json!({
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 30,
            "vocab_size": 100000,
        });
        let a = extract(&cfg, "deepseek2");
        assert!(a.attention_k_eq_v);
        assert_eq!(a.layer_types.len(), 30);
        for kind in &a.layer_types {
            assert_eq!(*kind, AttentionKind::FullAttention);
        }
    }

    #[test]
    fn extract_standard_uses_sliding_attention() {
        let cfg = json!({
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_hidden_layers": 12,
            "vocab_size": 50000,
        });
        let a = extract(&cfg, "gemma");
        assert!(!a.attention_k_eq_v);
        assert_eq!(a.layer_types.len(), 12);
        for kind in &a.layer_types {
            assert_eq!(*kind, AttentionKind::SlidingAttention);
        }
    }

    #[test]
    fn extract_moe_from_top_level() {
        let cfg = json!({
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 12,
            "vocab_size": 32000,
            "moe_config": {
                "num_experts": 8,
                "top_k_experts": 2,
                "intermediate_size": 2048,
                "shared_experts": true,
            }
        });
        let a = extract(&cfg, "mixtral");
        let moe = a.moe_config.expect("moe_config present");
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k_experts, 2);
        assert_eq!(moe.intermediate_size, 2048);
        assert!(moe.shared_experts);
    }

    #[test]
    fn extract_moe_from_deepseek_layout() {
        let cfg = json!({
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 30,
            "vocab_size": 100000,
            "num_experts": 64,
            "num_routed_experts": 6,
            "shared_experts": true,
        });
        let a = extract(&cfg, "deepseek2");
        let moe = a.moe_config.expect("moe_config present");
        assert_eq!(moe.num_experts, 64);
        assert_eq!(moe.top_k_experts, 6);
        assert!(moe.shared_experts);
    }

    #[test]
    fn extract_diffusion_block() {
        let cfg = json!({
            "hidden_size": 2560,
            "intermediate_size": 7168,
            "num_attention_heads": 20,
            "num_hidden_layers": 26,
            "vocab_size": 256000,
            "diffusion_config": {
                "max_diffusion_tokens": 512,
                "default_denoising_steps": 8,
                "noise_schedule": "linear",
                "parallel_token_generation": 32,
                "image_size": 1024,
                "patch_size": 14,
            }
        });
        let a = extract(&cfg, "diffusion_gemma");
        let d = a.diffusion_config.expect("diffusion_config present");
        assert_eq!(d.max_diffusion_tokens, 512);
        assert_eq!(d.default_denoising_steps, 8);
        assert_eq!(d.noise_schedule, NoiseScheduleType::Linear);
        assert_eq!(d.parallel_token_generation, 32);
        assert_eq!(d.image_size, 1024);
        assert_eq!(d.patch_size, 14);
    }

    #[test]
    fn extract_defaults_to_safe_zero_architecture() {
        let cfg = json!({});
        let a = extract(&cfg, "unknown");
        assert_eq!(a.hidden_size, 0);
        assert_eq!(a.num_attention_heads, 0);
        assert_eq!(a.max_position_embeddings, 2048);
        assert!((a.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert_eq!(a.rope_local.theta, 10000.0);
        assert_eq!(a.rope_local.rope_type, "default");
        assert!(a.layer_types.is_empty());
    }

    #[test]
    fn missing_sliding_window_defaults_to_zero() {
        let cfg = json!({"num_hidden_layers": 4});
        let a = extract(&cfg, "test");
        assert_eq!(a.sliding_window, 0);
    }

    #[test]
    fn head_dim_derives_from_hidden_size() {
        let cfg = json!({
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
        });
        let a = extract(&cfg, "test");
        assert_eq!(a.head_dim, 128);
    }

    #[test]
    fn head_dim_uses_explicit_value() {
        let cfg = json!({
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "head_dim": 64,
            "num_hidden_layers": 2,
        });
        let a = extract(&cfg, "test");
        assert_eq!(a.head_dim, 64);
    }

    #[test]
    fn text_architecture_serializes_round_trip() {
        let cfg = json!({
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_hidden_layers": 8,
            "vocab_size": 50000,
        });
        let a = extract(&cfg, "test");
        let s = serde_json::to_string(&a).expect("serialize");
        let back: TextArchitecture = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(a, back);
    }
}
