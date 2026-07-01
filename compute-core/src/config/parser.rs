//! Layer 1: Raw manifest types and config.json parsing.
//!
//! Raw model manifest types plus the `parse_config` function that reads
//! config.json and produces a normalized TextArchitecture + QuantizationMeta.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use super::hardware::{
    AttentionKind, AudioArchitecture, MoEConfig, QuantizationMeta,
    QuantizationMode, RopeSpec, TextArchitecture, VisionArchitecture,
};

// ── Layer 1: Raw Manifest ──────────────────────────────────────────────────

/// Raw model manifest read from config.json.
#[derive(Serialize, Clone)]
pub struct ModelManifest {
    pub config_path: String,
    pub config_hash: String,
    pub model_type: String,
    pub has_text_config: bool,
    pub has_vision_config: bool,
    pub has_audio_config: bool,
    pub has_quantization_metadata: bool,
    pub quantization_bits: Option<u32>,
    pub quantization_group_size: Option<u32>,
    pub quantization_mode: Option<String>,
    pub vision_config: Option<VisionArchitecture>,
    pub audio_config: Option<AudioArchitecture>,
    pub safetensors_shards: Vec<ShardManifest>,
}

#[derive(Serialize, Clone)]
pub struct ShardManifest {
    pub path: String,
    pub sha256: String,
    pub tensor_count: usize,
}

// ── Raw JSON parsing to normalized types ───────────────────────────────────

#[derive(Deserialize)]
#[allow(dead_code)]
struct RawConfig {
    #[serde(default)]
    model_type: Option<String>,
    // Fallback fields for flat configs (no nested text_config)
    #[serde(default)]
    hidden_size: Option<u32>,
    #[serde(default)]
    intermediate_size: Option<u32>,
    #[serde(default)]
    num_attention_heads: Option<u32>,
    #[serde(default)]
    num_key_value_heads: Option<u32>,
    #[serde(default)]
    head_dim: Option<u32>,
    #[serde(default)]
    global_head_dim: Option<u32>,
    #[serde(default)]
    num_global_key_value_heads: Option<u32>,
    #[serde(default)]
    num_hidden_layers: Option<u32>,
    #[serde(default)]
    vocab_size: Option<u32>,
    #[serde(default)]
    sliding_window: Option<u32>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    attention_k_eq_v: Option<bool>,
    #[serde(default)]
    final_logit_softcapping: Option<f64>,
    #[serde(default)]
    hidden_size_per_layer_input: Option<u32>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    enable_moe_block: Option<bool>,
    #[serde(default)]
    moe_intermediate_size: Option<u32>,
    #[serde(default)]
    num_experts: Option<u32>,
    #[serde(default)]
    top_k_experts: Option<u32>,
    #[serde(default)]
    num_kv_shared_layers: Option<u32>,
    #[serde(alias = "text_config")]
    text_config: Option<RawTextConfig>,
    #[serde(default)]
    #[serde(alias = "vision_config")]
    vision_config: Option<VisionArchitecture>,
    #[serde(default)]
    #[serde(alias = "audio_config")]
    audio_config: Option<AudioArchitecture>,
    #[serde(default)]
    #[serde(alias = "quantization_config")]
    quantization: Option<RawQuantization>,
    #[serde(default)]
    max_position_embeddings: Option<u32>,
    #[serde(default)]
    dtype: Option<String>,
}
impl RawConfig {
    fn to_text_config_fallback(&self) -> RawTextConfig {
        RawTextConfig {
            hidden_size: self.hidden_size.unwrap_or(2048),
            intermediate_size: self.intermediate_size.unwrap_or(8192),
            num_attention_heads: self.num_attention_heads.unwrap_or(16),
            num_key_value_heads: self.num_key_value_heads.unwrap_or(4),
            head_dim: self.head_dim.unwrap_or_else(|| {
                self.hidden_size.unwrap_or(2048) / self.num_attention_heads.unwrap_or(16)
            }),
            global_head_dim: self.global_head_dim,
            num_global_key_value_heads: self.num_global_key_value_heads,
            num_hidden_layers: self.num_hidden_layers.unwrap_or(24),
            vocab_size: self.vocab_size.unwrap_or(32768),
            sliding_window: self.sliding_window,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-6),
            tie_word_embeddings: self.tie_word_embeddings,
            attention_k_eq_v: self.attention_k_eq_v,
            final_logit_softcapping: self.final_logit_softcapping,
            hidden_size_per_layer_input: self.hidden_size_per_layer_input,
            layer_types: self.layer_types.clone().unwrap_or_default(),
            rope_parameters: None,
            model_type: self.model_type.clone(),
        }
    }
}

#[derive(Deserialize, Clone)]
struct RawTextConfig {
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    global_head_dim: Option<u32>,
    num_global_key_value_heads: Option<u32>,
    num_hidden_layers: u32,
    vocab_size: u32,
    sliding_window: Option<u32>,
    max_position_embeddings: Option<u32>,
    rms_norm_eps: f64,
    tie_word_embeddings: Option<bool>,
    attention_k_eq_v: Option<bool>,
    final_logit_softcapping: Option<f64>,
    hidden_size_per_layer_input: Option<u32>,
    layer_types: Vec<String>,
    rope_parameters: Option<RawRopeParams>,
    model_type: Option<String>,
}

#[derive(Deserialize, Clone)]
struct RawRopeParams {
    sliding_attention: Option<RawRopeSpec>,
    full_attention: Option<RawRopeSpec>,
}

#[derive(Deserialize, Clone)]
struct RawRopeSpec {
    rope_theta: f64,
    rope_type: Option<String>,
    partial_rotary_factor: Option<f64>,
}

#[derive(Deserialize, Clone)]
struct RawQuantization {
    group_size: Option<u32>,
    bits: Option<u32>,
    mode: Option<String>,
}

/// Parse config.json and produce a normalized TextArchitecture + QuantizationMeta.
pub fn parse_config(
    config_path: &str,
) -> crate::Result<(TextArchitecture, Option<QuantizationMeta>, ModelManifest)> {
    let config_json = std::fs::read_to_string(config_path)
        .map_err(|e| crate::Error::from_reason(format!("Cannot read config: {}", e)))?;

    // Hash the raw config for provenance
    let mut hasher = Sha256::new();
    hasher.update(config_json.as_bytes());
    let config_hash = format!("{:x}", hasher.finalize());

    let raw: RawConfig = serde_json::from_str(&config_json)
        .map_err(|e| crate::Error::from_reason(format!("Invalid config JSON: {}", e)))?;

    let text = raw
        .text_config
        .clone()
        .unwrap_or_else(|| raw.to_text_config_fallback());

    let max_pos = text
        .max_position_embeddings
        .or(raw.max_position_embeddings)
        .unwrap_or(131072);

    let mut layer_types: Vec<AttentionKind> = text
        .layer_types
        .iter()
        .map(|s| match s.as_str() {
            "full_attention" | "full" => AttentionKind::FullAttention,
            _ => AttentionKind::SlidingAttention,
        })
        .collect();

    // If layer_types is empty (flat configs like Qwen, Llama), default to all sliding.
    if layer_types.is_empty() {
        for _ in 0..text.num_hidden_layers {
            layer_types.push(AttentionKind::SlidingAttention);
        }
    } else if layer_types.len() != text.num_hidden_layers as usize {
        return Err(crate::Error::from_reason(format!(
            "layer_types count ({}) != num_hidden_layers ({})",
            layer_types.len(),
            text.num_hidden_layers
        )));
    }

    let rope_local = {
        let raw_rope = text
            .rope_parameters
            .as_ref()
            .and_then(|r| r.sliding_attention.as_ref())
            .map(|s| RopeSpec {
                theta: s.rope_theta,
                rope_type: s.rope_type.clone().unwrap_or_else(|| "default".into()),
                partial_rotary_factor: s.partial_rotary_factor,
            })
            .unwrap_or_else(|| RopeSpec {
                theta: 10000.0,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            });
        raw_rope
    };

    let rope_global = text
        .rope_parameters
        .as_ref()
        .and_then(|r| r.full_attention.as_ref())
        .map(|s| RopeSpec {
            theta: s.rope_theta,
            rope_type: s.rope_type.clone().unwrap_or_else(|| "proportional".into()),
            partial_rotary_factor: s.partial_rotary_factor,
        });

    let moe_config = if raw.enable_moe_block.unwrap_or(false) {
        let num_experts = raw.num_experts.unwrap_or(0);
        let top_k = raw.top_k_experts.unwrap_or(1);
        let inter_size = raw
            .moe_intermediate_size
            .or_else(|| Some(text.intermediate_size))
            .unwrap_or(0);
        if num_experts > 0 && top_k > 0 {
            Some(MoEConfig {
                num_experts,
                top_k_experts: top_k,
                intermediate_size: inter_size,
                shared_experts: false,
            })
        } else {
            None
        }
    } else {
        None
    };

    let arch = TextArchitecture {
        diffusion_config: None,
        hidden_size: text.hidden_size,
        intermediate_size: text.intermediate_size,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        head_dim: text.head_dim,
        global_head_dim: text.global_head_dim,
        num_global_key_value_heads: text.num_global_key_value_heads,
        num_hidden_layers: text.num_hidden_layers,
        vocab_size: text.vocab_size,
        sliding_window: text.sliding_window.unwrap_or(4096),
        max_position_embeddings: max_pos,
        rms_norm_eps: text.rms_norm_eps,
        tie_word_embeddings: text.tie_word_embeddings.unwrap_or(true),
        attention_k_eq_v: text.attention_k_eq_v.unwrap_or(true),
        final_logit_softcapping: text.final_logit_softcapping,
        hidden_size_per_layer_input: text.hidden_size_per_layer_input.unwrap_or(0),
        layer_types,
        rope_local,
        rope_global,
        model_type: text
            .model_type
            .clone()
            .unwrap_or_else(|| "gemma4_unified_text".into()),
        moe_config,
    };

    let q_bits = raw.quantization.as_ref().and_then(|q| q.bits);
    let q_group_size = raw.quantization.as_ref().and_then(|q| q.group_size);
    let has_explicit_quant = raw.quantization.is_some();
    let explicit_quant = raw.quantization.map(|q| QuantizationMeta {
        bits: q.bits.unwrap_or(16),
        group_size: q.group_size.unwrap_or(64),
        mode: match q.mode.as_deref() {
            Some("affine") => QuantizationMode::Affine,
            _ => QuantizationMode::None,
        },
        overrides: HashMap::new(),
    });

    // For models with a nested text_config (e.g. Gemma4 Unified), the
    // conversion process may not have written an explicit quantization
    // section into config.json.  Detect this case by checking whether
    // the top-level model_type contains known unified/conversion patterns
    // and default to 8-bit block quantization if no explicit metadata.
    let quant = explicit_quant.or_else(|| {
        if raw.text_config.is_some() {
            let mt = raw.model_type.as_deref().unwrap_or("");
            if mt.contains("unified") || mt.starts_with("gemma4") {
                Some(QuantizationMeta {
                    bits: 8,
                    group_size: 64,
                    mode: QuantizationMode::Affine,
                    overrides: HashMap::new(),
                })
            } else {
                None
            }
        } else {
            None
        }
    });

    let manifest = ModelManifest {
        config_path: config_path.into(),
        config_hash,
        model_type: raw.model_type.unwrap_or_default(),
        has_text_config: true, // we already checked text_config exists
        has_vision_config: raw.vision_config.is_some(),
        has_audio_config: raw.audio_config.is_some(),
        has_quantization_metadata: has_explicit_quant,
        quantization_bits: q_bits,
        quantization_group_size: q_group_size,
        quantization_mode: quant.as_ref().map(|q| format!("{:?}", q.mode)),
        vision_config: raw.vision_config.clone(),
        audio_config: raw.audio_config.clone(),
        safetensors_shards: Vec::new(),
    };

    Ok((arch, quant, manifest))
}
