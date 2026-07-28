//! ModelGraph — architecture-agnostic transformer descriptor.
//!
//! This module owns the canonical authority for the
//! backend-neutral model graph used by LUT-based AOT
//! compilation and runtime execution. It reads a HuggingFace
//! `config.json`, normalizes across Llama / Mistral / Qwen2 /
//! Qwen3.5 / Gemma2 / Gemma4 families, handles nested
//! `text_config` / `language_config`, hybrid attention types,
//! and produces a deterministic `Vec<ComputeNode>`.
//!
//! All types are backend-neutral. Hardware-specific lowering
//! and execution live in the respective backend crates and
//! consume this descriptor.

use serde::{Deserialize, Serialize};
use std::path::Path;

// ── Architecture families ─────────────────────────────────────────────

/// The architecture family inferred from the HuggingFace
/// `architectures[0]` field. The full set of supported
/// families is enumerated here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureFamily {
    Llama,
    Mistral,
    Qwen2,
    Qwen3_5,
    Gemma2,
    Gemma4,
}

/// Which nonlinear activation the model applies between the
/// gate and up projections in the MLP block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    Silu,
    Gelu,
}

/// Semantic role of a tensor in the transformer block. Used
/// by codegen and execution lanes to pick the right kernel
/// for the right projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorRole {
    QProj,
    KProj,
    VProj,
    OProj,
    GateProj,
    UpProj,
    DownProj,
    /// Shared K/V projection (attention_k_eq_v).
    KvProj,
    /// Single linear attention QKV projection.
    FusedQkvProj,
    /// attn_output_gate.
    OutputGate,
    /// Multi-token prediction.
    MtpHead,
}

/// A tensor reference inside a [`ModelGraph`]: the lookup
/// key into the weight table plus the matmul dimensions
/// (output dim M, input dim N).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBlueprint {
    pub key: String,
    pub dim_m: u32,
    pub dim_n: u32,
}

// ── Compute nodes (the execution graph) ───────────────────────────────

/// A single node in the deterministic execution graph built
/// from a [`UnifiedConfig`]. Hardware backends pattern-match
/// on this enum to lower each variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeNode {
    TokenEmbedding {
        key: String,
        vocab_size: u32,
        hidden_dim: u32,
    },
    Norm {
        key: String,
        hidden_dim: u32,
        eps: f32,
        is_rms: bool,
    },
    PalettizedMatmul {
        role: TensorRole,
        tensor: TensorBlueprint,
    },
    RotaryEmbedding {
        head_dim: u32,
        rope_theta: f32,
    },
    /// Full softmax attention (standard transformer).
    ScaledDotProductAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },
    /// Gated DeltaNet linear attention (SSM hybrid).
    LinearAttention {
        num_heads: u32,
        state_dim: u32,
        head_dim: u32,
        decay_factor: f32,
    },
    /// Attention output gate (SiLU-gated O projection).
    AttentionOutputGate {
        key: String,
        dim: u32,
    },
    /// Multi-resolution RoPE for vision-language tokens.
    MRoPE {
        head_dim: u32,
        rope_theta: f32,
        mrope_section: Vec<u32>,
    },
    /// Shared K/V projection (attention_k_eq_v).
    SharedKVProjection {
        tensor: TensorBlueprint,
    },
    /// Multi-token prediction head.
    MultiTokenPredictionHead {
        tensor: TensorBlueprint,
        depth: u32,
    },
    Activation {
        func: ActivationFunction,
    },
    LanguageModelHead {
        tensor: TensorBlueprint,
    },
}

// ── Raw HF config ─────────────────────────────────────────────────────

/// One block of HuggingFace config fields, with all fields
/// optional so partial / nested configs (e.g. Gemma4's
/// `text_config`) parse cleanly.
#[derive(Debug, Deserialize, Clone)]
pub struct HfConfigBlock {
    pub vocab_size: Option<u32>,
    pub hidden_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub rms_norm_eps: Option<f32>,
    pub layer_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    pub hidden_act: Option<String>,
    pub head_dim: Option<u32>,
    pub attention_k_eq_v: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub layer_types: Option<Vec<String>>,
    pub mrope_section: Option<Vec<u32>>,
    pub attn_output_gate: Option<bool>,
    pub mtp_num_hidden_layers: Option<u32>,
}

/// Raw HuggingFace config — supports the nested
/// `text_config` / `language_config` pattern with field
/// fallthrough to the root block.
#[derive(Debug, Deserialize)]
pub struct RawHfConfig {
    pub architectures: Option<Vec<String>>,
    #[serde(flatten)]
    pub root: HfConfigBlock,
    pub text_config: Option<HfConfigBlock>,
    pub language_config: Option<HfConfigBlock>,
}

/// Resolve a config field by cascading
/// `text_config` → `language_config` → root.
macro_rules! resolve_cfg {
    ($raw:expr, $field:ident) => {
        $raw.text_config
            .as_ref()
            .and_then(|c| c.$field.clone())
            .or_else(|| $raw.language_config.as_ref().and_then(|c| c.$field.clone()))
            .or_else(|| $raw.root.$field.clone())
    };
}

// ── Unified config ────────────────────────────────────────────────────

/// Architecture-agnostic model descriptor produced by
/// [`UnifiedConfig::from_file`]. Every downstream
/// `ModelGraph` builder, AOT compiler, and runtime
/// dispatcher consumes a `UnifiedConfig` rather than the raw
/// HF block.
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    pub family: ArchitectureFamily,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub linear_num_key_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_num_value_heads: u32,
    pub linear_value_head_dim: u32,
    pub head_dim: u32,
    pub norm_eps: f32,
    pub is_rms_norm: bool,
    pub rope_theta: f32,
    pub activation: ActivationFunction,
    pub attention_k_eq_v: bool,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    pub mrope_section: Vec<u32>,
    pub mtp_depth: u32,
    /// Weight key prefix (varies: "model" for Qwen,
    /// "language_model.model" for Gemma4/Qwen3_5).
    pub key_prefix: String,
}

impl UnifiedConfig {
    /// Extract `rope_theta` from `rope_parameters.rope_theta`
    /// (Qwen2.5+) or from the flat `rope_theta` field.
    fn resolve_rope_theta(raw: &RawHfConfig) -> f32 {
        for block in [
            &raw.text_config,
            &raw.language_config,
            &Some(raw.root.clone()),
        ]
        .iter()
        {
            if let Some(ref c) = block {
                if let Some(ref rp) = c.rope_parameters {
                    if let Some(theta) = rp.get("rope_theta").and_then(|v| v.as_f64()) {
                        return theta as f32;
                    }
                }
                if let Some(theta) = c.rope_theta {
                    return theta;
                }
            }
        }
        10_000.0
    }

    /// Read a HuggingFace `config.json` and produce a
    /// [`UnifiedConfig`]. Defaults are chosen so missing
    /// fields (a common occurrence in sharded configs) still
    /// yield a graph the codegen can consume.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let json_str =
            std::fs::read_to_string(path).map_err(|e| format!("read config.json: {e}"))?;
        let raw: RawHfConfig =
            serde_json::from_str(&json_str).map_err(|e| format!("parse config.json: {e}"))?;

        let arch_str = raw
            .architectures
            .as_ref()
            .and_then(|a| a.first().cloned())
            .unwrap_or_else(|| "LlamaForCausalLM".to_string());

        let family = match arch_str.as_str() {
            "MistralForCausalLM" => ArchitectureFamily::Mistral,
            "Qwen2ForCausalLM" => ArchitectureFamily::Qwen2,
            "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" => {
                ArchitectureFamily::Qwen3_5
            }
            "Gemma4UnifiedForConditionalGeneration" | "Gemma4ForCausalLM" => {
                ArchitectureFamily::Gemma4
            }
            _ => ArchitectureFamily::Llama,
        };

        // Cascading resolution (text_config > language_config > root)
        let rope_theta = Self::resolve_rope_theta(&raw);
        let hidden_size = resolve_cfg!(raw, hidden_size).unwrap_or(4096);
        let num_heads = resolve_cfg!(raw, num_attention_heads).unwrap_or(32);
        let num_kv = resolve_cfg!(raw, num_key_value_heads).unwrap_or(num_heads);
        let head_dim = resolve_cfg!(raw, head_dim).unwrap_or(hidden_size / num_heads);
        let norm_eps = resolve_cfg!(raw, rms_norm_eps)
            .or_else(|| resolve_cfg!(raw, layer_norm_eps))
            .unwrap_or(1e-5);
        let linear_num_key_heads = resolve_cfg!(raw, linear_num_key_heads).unwrap_or(num_kv);
        let linear_key_head_dim = resolve_cfg!(raw, linear_key_head_dim).unwrap_or(head_dim);
        let linear_num_value_heads = resolve_cfg!(raw, linear_num_value_heads).unwrap_or(num_kv);
        let linear_value_head_dim =
            resolve_cfg!(raw, linear_value_head_dim).unwrap_or(head_dim);
        let hidden_act = resolve_cfg!(raw, hidden_act);
        let tie_emb = resolve_cfg!(raw, tie_word_embeddings).unwrap_or(false);
        let layer_types = resolve_cfg!(raw, layer_types).unwrap_or_default();

        let activation = match hidden_act.as_deref() {
            Some("gelu") | Some("gelu_pytorch_tanh") => ActivationFunction::Gelu,
            _ => ActivationFunction::Silu,
        };

        let key_prefix = match family {
            ArchitectureFamily::Gemma4 => "language_model.model".to_string(),
            ArchitectureFamily::Qwen3_5 => "model.language_model".to_string(),
            _ => "model".to_string(),
        };

        Ok(UnifiedConfig {
            family,
            vocab_size: resolve_cfg!(raw, vocab_size).unwrap_or(151936),
            hidden_size,
            intermediate_size: resolve_cfg!(raw, intermediate_size).unwrap_or(11008),
            num_layers: resolve_cfg!(raw, num_hidden_layers).unwrap_or(32),
            num_heads,
            num_kv_heads: num_kv,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_num_value_heads,
            linear_value_head_dim,
            head_dim,
            norm_eps,
            is_rms_norm: raw
                .text_config
                .as_ref()
                .or_else(|| raw.language_config.as_ref())
                .map(|c| c.rms_norm_eps.is_some())
                .unwrap_or(true),
            rope_theta,
            activation,
            attention_k_eq_v: resolve_cfg!(raw, attention_k_eq_v).unwrap_or(false),
            tie_word_embeddings: tie_emb,
            layer_types,
            mrope_section: resolve_cfg!(raw, mrope_section).unwrap_or_default(),
            mtp_depth: resolve_cfg!(raw, mtp_num_hidden_layers).unwrap_or(0),
            key_prefix,
        })
    }
}

// ── ModelGraph builder ────────────────────────────────────────────────

/// The deterministic execution graph built from a
/// [`UnifiedConfig`]. Backend dispatchers pattern-match on
/// [`ComputeNode`] variants to lower each step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGraph {
    pub nodes: Vec<ComputeNode>,
    pub num_layers: u32,
}

impl ModelGraph {
    /// Build the deterministic graph for `config`. The order
    /// of nodes is stable: token embedding, then a fixed
    /// sequence per layer (input norm → Q/K/V → RoPE → SDPA →
    /// O → post-norm → MLP), then the final norm and LM head.
    pub fn build(config: &UnifiedConfig) -> Self {
        let mut nodes = Vec::new();
        let p = &config.key_prefix;

        // 1. Token embedding
        nodes.push(ComputeNode::TokenEmbedding {
            key: format!("{p}.embed_tokens.weight"),
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_size,
        });

        // 2. Transformer layers
        for layer_idx in 0..config.num_layers {
            let prefix = format!("{p}.layers.{layer_idx}");

            // Input norm
            nodes.push(ComputeNode::Norm {
                key: format!("{prefix}.input_layernorm.weight"),
                hidden_dim: config.hidden_size,
                eps: config.norm_eps,
                is_rms: config.is_rms_norm,
            });

            // Determine layer type for hybrid architectures
            let is_linear = config
                .layer_types
                .get(layer_idx as usize)
                .map(|t| t.contains("linear"))
                .unwrap_or(false);

            if is_linear {
                // Fused QKV dim for linear attention layers.
                let fused_qkv_dim = (config.num_heads * config.head_dim)
                    + (config.linear_num_key_heads * config.linear_key_head_dim)
                    + (config.linear_num_value_heads * config.linear_value_head_dim);

                // Linear attention layer (SSM hybrid)
                // QKV fused projection for Gated DeltaNet
                nodes.push(ComputeNode::PalettizedMatmul {
                    role: TensorRole::FusedQkvProj,
                    tensor: TensorBlueprint {
                        key: format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        dim_m: fused_qkv_dim,
                        dim_n: config.hidden_size,
                    },
                });
                nodes.push(ComputeNode::LinearAttention {
                    num_heads: config.num_heads,
                    head_dim: config.head_dim,
                    state_dim: 128, // typical linear attention state
                    decay_factor: 0.95,
                });

                // Attention output gate (Qwen 3.5)
                nodes.push(ComputeNode::PalettizedMatmul {
                    role: TensorRole::OutputGate,
                    tensor: TensorBlueprint {
                        key: format!("{prefix}.linear_attn.out_proj.weight"),
                        dim_m: config.hidden_size,
                        dim_n: config.hidden_size * 2,
                    },
                });
            } else {
                // Standard full attention layer
                let q_dim = config.num_heads * config.head_dim;
                let kv_dim = config.num_kv_heads * config.head_dim;
                // Qwen3.5 full attention doubles Q projection for QK-norm
                let q_proj_dim = if config.family == ArchitectureFamily::Qwen3_5 {
                    q_dim * 2
                } else {
                    q_dim
                };

                // Q/K/V projections (or shared K/V)
                nodes.push(ComputeNode::PalettizedMatmul {
                    role: TensorRole::QProj,
                    tensor: TensorBlueprint {
                        key: format!("{prefix}.self_attn.q_proj.weight"),
                        dim_m: q_proj_dim,
                        dim_n: config.hidden_size,
                    },
                });

                if config.attention_k_eq_v {
                    nodes.push(ComputeNode::SharedKVProjection {
                        tensor: TensorBlueprint {
                            key: format!("{prefix}.self_attn.k_proj.weight"),
                            dim_m: kv_dim,
                            dim_n: config.hidden_size,
                        },
                    });
                } else {
                    nodes.push(ComputeNode::PalettizedMatmul {
                        role: TensorRole::KProj,
                        tensor: TensorBlueprint {
                            key: format!("{prefix}.self_attn.k_proj.weight"),
                            dim_m: kv_dim,
                            dim_n: config.hidden_size,
                        },
                    });
                    nodes.push(ComputeNode::PalettizedMatmul {
                        role: TensorRole::VProj,
                        tensor: TensorBlueprint {
                            key: format!("{prefix}.self_attn.v_proj.weight"),
                            dim_m: kv_dim,
                            dim_n: config.hidden_size,
                        },
                    });
                }

                // RoPE or MRoPE
                if config.mrope_section.is_empty() {
                    nodes.push(ComputeNode::RotaryEmbedding {
                        head_dim: config.head_dim,
                        rope_theta: config.rope_theta,
                    });
                } else {
                    nodes.push(ComputeNode::MRoPE {
                        head_dim: config.head_dim,
                        rope_theta: config.rope_theta,
                        mrope_section: config.mrope_section.clone(),
                    });
                }

                nodes.push(ComputeNode::ScaledDotProductAttention {
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                });

                // Output projection
                nodes.push(ComputeNode::PalettizedMatmul {
                    role: TensorRole::OProj,
                    tensor: TensorBlueprint {
                        key: format!("{prefix}.self_attn.o_proj.weight"),
                        dim_m: config.hidden_size,
                        dim_n: q_dim,
                    },
                });
            }

            // Post-attention norm
            nodes.push(ComputeNode::Norm {
                key: format!("{prefix}.post_attention_layernorm.weight"),
                hidden_dim: config.hidden_size,
                eps: config.norm_eps,
                is_rms: config.is_rms_norm,
            });

            // MLP (same for all architectures)
            nodes.push(ComputeNode::PalettizedMatmul {
                role: TensorRole::GateProj,
                tensor: TensorBlueprint {
                    key: format!("{prefix}.mlp.gate_proj.weight"),
                    dim_m: config.intermediate_size,
                    dim_n: config.hidden_size,
                },
            });
            nodes.push(ComputeNode::PalettizedMatmul {
                role: TensorRole::UpProj,
                tensor: TensorBlueprint {
                    key: format!("{prefix}.mlp.up_proj.weight"),
                    dim_m: config.intermediate_size,
                    dim_n: config.hidden_size,
                },
            });
            nodes.push(ComputeNode::Activation {
                func: config.activation,
            });
            nodes.push(ComputeNode::PalettizedMatmul {
                role: TensorRole::DownProj,
                tensor: TensorBlueprint {
                    key: format!("{prefix}.mlp.down_proj.weight"),
                    dim_m: config.hidden_size,
                    dim_n: config.intermediate_size,
                },
            });
        }

        // 3. Final norm
        nodes.push(ComputeNode::Norm {
            key: format!("{p}.norm.weight"),
            hidden_dim: config.hidden_size,
            eps: config.norm_eps,
            is_rms: config.is_rms_norm,
        });

        // 4. LM head (or MTP head if multi-token prediction is enabled)
        if config.mtp_depth > 0 {
            nodes.push(ComputeNode::MultiTokenPredictionHead {
                tensor: TensorBlueprint {
                    // MTP head shares weights with embedding when tied
                    key: if config.tie_word_embeddings {
                        format!("{p}.embed_tokens.weight")
                    } else {
                        format!("{p}.lm_head.weight")
                    },
                    dim_m: config.vocab_size,
                    dim_n: config.hidden_size,
                },
                depth: config.mtp_depth,
            });
        } else if !config.tie_word_embeddings {
            nodes.push(ComputeNode::LanguageModelHead {
                tensor: TensorBlueprint {
                    key: format!("{p}.lm_head.weight"),
                    dim_m: config.vocab_size,
                    dim_n: config.hidden_size,
                },
            });
        }
        // If tie_word_embeddings and no MTP: LM head uses
        // embedding table weights.

        ModelGraph {
            nodes,
            num_layers: config.num_layers,
        }
    }

    /// All tensors that should be palettized at AOT compile
    /// time. Includes the language model head, multi-token
    /// prediction head, and shared K/V projection in
    /// addition to the per-block projections.
    pub fn palettized_tensors(&self) -> Vec<&TensorBlueprint> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                ComputeNode::PalettizedMatmul { tensor, .. } => Some(tensor),
                ComputeNode::SharedKVProjection { tensor, .. } => Some(tensor),
                ComputeNode::MultiTokenPredictionHead { tensor, .. } => Some(tensor),
                ComputeNode::LanguageModelHead { tensor, .. } => Some(tensor),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_layer_types(layer_types: Vec<String>) -> UnifiedConfig {
        UnifiedConfig {
            family: ArchitectureFamily::Qwen3_5,
            vocab_size: 248320,
            hidden_size: 1024,
            intermediate_size: 3584,
            num_layers: layer_types.len() as u32,
            num_heads: 8,
            num_kv_heads: 2,
            linear_num_key_heads: 2,
            linear_key_head_dim: 256,
            linear_num_value_heads: 2,
            linear_value_head_dim: 256,
            head_dim: 256,
            norm_eps: 1e-5,
            is_rms_norm: true,
            rope_theta: 10_000_000.0,
            activation: ActivationFunction::Silu,
            attention_k_eq_v: false,
            tie_word_embeddings: true,
            layer_types,
            mrope_section: Vec::new(),
            mtp_depth: 0,
            key_prefix: "model.language_model".to_string(),
        }
    }

    #[test]
    fn unified_config_qwen2_fields_resolve() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 896,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true
        }"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, json).unwrap();
        let cfg = UnifiedConfig::from_file(&path).expect("parse Qwen2 config");
        assert_eq!(cfg.family, ArchitectureFamily::Qwen2);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.intermediate_size, 4864);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.num_heads, 14);
        assert_eq!(cfg.num_kv_heads, 2);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.rope_theta as u32, 1_000_000);
        assert_eq!(cfg.activation, ActivationFunction::Silu);
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.key_prefix, "model");
        assert!(cfg.layer_types.is_empty());
    }

    #[test]
    fn unified_config_resolves_text_config_first() {
        // text_config values should take precedence over root
        let json = r#"{
            "architectures": ["Gemma4ForCausalLM"],
            "hidden_size": 9999,
            "text_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12
            }
        }"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, json).unwrap();
        let cfg = UnifiedConfig::from_file(&path).expect("parse");
        assert_eq!(cfg.family, ArchitectureFamily::Gemma4);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_layers, 12);
        assert_eq!(cfg.key_prefix, "language_model.model");
    }

    #[test]
    fn unified_config_resolves_rope_theta_from_rope_parameters() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "rope_parameters": { "rope_theta": 500000.0 }
        }"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, json).unwrap();
        let cfg = UnifiedConfig::from_file(&path).expect("parse");
        assert_eq!(cfg.rope_theta as u32, 500_000);
    }

    #[test]
    fn unified_config_defaults_when_fields_missing() {
        let json = r#"{"architectures": ["LlamaForCausalLM"]}"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, json).unwrap();
        let cfg = UnifiedConfig::from_file(&path).expect("parse");
        assert_eq!(cfg.family, ArchitectureFamily::Llama);
        assert_eq!(cfg.key_prefix, "model");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.rope_theta, 10_000.0);
    }

    #[test]
    fn model_graph_palettized_tensors_filters_correctly() {
        // Build a 2-layer Qwen3.5 (mixed) config: 1 full + 1 linear.
        // Full attention with separate K/V emits 4 matmul tensors
        // (Q, K, V, O) plus the MLP's 3 (Gate, Up, Down) = 7.
        // Linear attention emits FusedQkv + OutputGate + the MLP's
        // 3 (Gate, Up, Down) = 5. No LM head (tied embeddings, no MTP).
        let cfg = config_with_layer_types(vec![
            "full_attention".to_string(),
            "linear_attention".to_string(),
        ]);
        let graph = ModelGraph::build(&cfg);
        let pal = graph.palettized_tensors();
        assert_eq!(pal.len(), 12);
    }

    #[test]
    fn model_graph_qwen3_5_hybrid_node_count() {
        // 18 linear + 6 full = 24 total layers, matching the
        // Qwen3.5-7B hybrid schedule (1 full every 4 layers).
        // Per full layer (no shared KV): 12 nodes.
        // Per linear layer: 9 nodes (input_norm, FusedQkv,
        // LinearAttention, OutputGate, post_norm, Gate, Up,
        // Activation, Down).
        // + 1 embed + 1 final norm = 2
        // = 6*12 + 18*9 + 2 = 236 (matches engine baseline).
        let mut types = Vec::new();
        for i in 0..24 {
            if i % 4 == 3 {
                types.push("full_attention".to_string());
            } else {
                types.push("linear_attention".to_string());
            }
        }
        let cfg = config_with_layer_types(types);
        let graph = ModelGraph::build(&cfg);
        let n_full = 6;
        let n_lin = 18;
        let expected = 1 + n_full * 12 + n_lin * 9 + 1;
        assert_eq!(graph.nodes.len(), expected);
    }

    #[test]
    fn model_graph_lm_head_skipped_when_tied_and_no_mtp() {
        let cfg = config_with_layer_types(vec!["full_attention".to_string()]);
        let graph = ModelGraph::build(&cfg);
        let has_lm = graph
            .nodes
            .iter()
            .any(|n| matches!(n, ComputeNode::LanguageModelHead { .. }));
        assert!(!has_lm, "tied embeddings without MTP should skip LM head");
    }

    #[test]
    fn model_graph_mtp_head_uses_embedding_when_tied() {
        let mut cfg = config_with_layer_types(vec!["full_attention".to_string()]);
        cfg.tie_word_embeddings = true;
        cfg.mtp_depth = 1;
        let graph = ModelGraph::build(&cfg);
        let mtp = graph
            .nodes
            .iter()
            .find_map(|n| match n {
                ComputeNode::MultiTokenPredictionHead { tensor, .. } => Some(tensor),
                _ => None,
            })
            .expect("mtp head present");
        assert!(mtp.key.ends_with("embed_tokens.weight"));
    }
}
