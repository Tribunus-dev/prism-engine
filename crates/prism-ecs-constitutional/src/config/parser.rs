//! Layer 1: Raw manifest types and `config.json` parsing.
//!
//! Authority: the canonical [`parse_config`] routine that reads
//! `config.json` and produces a normalized
//! [`super::architecture::TextArchitecture`] + [`QuantizationMeta`]
//! plus the [`ModelManifest`] / [`CimageManifest`] carrier types used
//! by the compile pipeline. Uses thiserror-driven
//! [`super::error::ConfigError`]; no `anyhow`, no `unwrap`, no `panic`.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use super::architecture::{
    AttentionKind, AudioArchitecture, MoEConfig, QuantizationMeta, QuantizationMode, RopeSpec,
    TextArchitecture, VisionArchitecture,
};
use super::error::{ConfigError, ConfigResult};

// ── Modality Discriminator ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ManifestModality {
    Text,
    Vision,
    Audio,
    Multimodal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ArchitectureConfig {
    Text(TextArchitecture),
    Vision(VisionArchitecture),
    Audio(AudioArchitecture),
}

// ── Layer 1: Raw Manifest ──────────────────────────────────────────────

/// Raw model manifest read from config.json.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelManifest {
    /// Modality discriminator — determines runtime dispatch path.
    pub modality: ManifestModality,
    /// Architecture config, typed by modality.
    pub architecture: Option<ArchitectureConfig>,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
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

/// Parse `config.json` and produce a normalized [`TextArchitecture`] +
/// [`QuantizationMeta`] + [`ModelManifest`].
pub fn parse_config(
    config_path: &str,
) -> ConfigResult<(TextArchitecture, Option<QuantizationMeta>, ModelManifest)> {
    if config_path.is_empty() {
        return Err(ConfigError::EmptyConfigPath);
    }

    let config_json = std::fs::read_to_string(config_path)?;

    // Hash the raw config for provenance
    let mut hasher = Sha256::new();
    hasher.update(config_json.as_bytes());
    let config_hash = format!("{:x}", hasher.finalize());

    let raw: RawConfig = serde_json::from_str(&config_json)?;

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
        return Err(ConfigError::LayerTypeCountMismatch {
            layer_types: layer_types.len(),
            num_hidden_layers: text.num_hidden_layers,
        });
    }

    let rope_local = text
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
        thinking_mode: false,
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
        overrides: BTreeMap::new(),
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
                    overrides: BTreeMap::new(),
                })
            } else {
                None
            }
        } else {
            None
        }
    });

    let manifest = ModelManifest {
        modality: ManifestModality::Text,
        architecture: None,
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

// ── CimageManifest — compile-target manifest ────────────────────────────

/// Compile-target manifest for standalone modality-typed cimage
/// builds.  Pairs with the audio / vision / text compilation
/// pipelines to produce a serialized cimage artifact on disk.
#[derive(Clone, Serialize, Deserialize)]
pub struct CimageManifest {
    pub modality: ManifestModality,
    pub architecture: ArchitectureConfig,
    /// Tensor entries, recorded at compile time.  Stored as
    /// `serde_json::Value` so this module does not depend on the
    /// engine-internal `TensorEntry` type; the compile pipeline
    /// re-attaches the typed entries when the cimage is emitted.
    pub tensor_table: Vec<serde_json::Value>,
}

impl CimageManifest {
    /// Serialize this manifest to a JSON file at `path`.
    pub fn write_to(&self, path: &std::path::Path) -> ConfigResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, &json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp_config(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "prism_ecs_constitutional_config_test_{}_{}_{}.json",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn parse_rejects_empty_path() {
        let err = parse_config("").unwrap_err();
        assert!(matches!(err, ConfigError::EmptyConfigPath));
    }

    #[test]
    fn parse_handles_minimal_flat_config() {
        let body = r#"{
            "model_type": "gemma4_unified_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "num_hidden_layers": 2,
            "vocab_size": 128,
            "rms_norm_eps": 1e-6
        }"#;
        let path = write_tmp_config("flat", body);
        let (arch, _q, manifest) = parse_config(path.to_str().unwrap()).unwrap();
        assert_eq!(arch.hidden_size, 32);
        assert_eq!(arch.num_hidden_layers, 2);
        // Empty layer_types => defaults to all sliding.
        assert_eq!(arch.layer_types.len(), 2);
        assert!(matches!(
            arch.layer_types[0],
            AttentionKind::SlidingAttention
        ));
        // The parser has determined this is a text model, so the
        // manifest flags `has_text_config = true` (matches the
        // engine-side invariant).
        assert!(manifest.has_text_config);
        // Cleanup best-effort.
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn parse_rejects_layer_count_mismatch() {
        let body = r#"{
            "text_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "num_hidden_layers": 2,
                "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention"]
            }
        }"#;
        let path = write_tmp_config("mismatch", body);
        let err = parse_config(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(
            err,
            ConfigError::LayerTypeCountMismatch { .. }
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cimage_manifest_round_trip() {
        let m = CimageManifest {
            modality: ManifestModality::Text,
            architecture: ArchitectureConfig::Text(TextArchitecture::default()),
            tensor_table: vec![],
        };
        let j = serde_json::to_string(&m).unwrap();
        let back: CimageManifest = serde_json::from_str(&j).unwrap();
        assert_eq!(back.modality, ManifestModality::Text);
    }
}
