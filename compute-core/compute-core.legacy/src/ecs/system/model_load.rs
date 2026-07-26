use crate::ecs::adapter::{architecture_patterns, match_tensor_pattern, SourceModel};
use crate::ecs::component::tensor::{CanonicalRoleComp, DType, DataType, LayerIndex, Shape};
use crate::ecs::config::{
    AttentionKind, ConfidenceType, DiffusionConfig, MoEConfig as ArchMoEConfig, NoiseScheduleType,
    RopeSpec, TextArchitecture,
};

use crate::ecs::{CompilerSystem, Component, EntityKind, SchedulePhase, World};
use serde_json::Value;
use std::collections::BTreeSet;

// ────────────────────────────────────────────────────────────────────────
// Component wrapper — lets us store a SourceModel in the ECS world
// ────────────────────────────────────────────────────────────────────────

/// ECS component wrapping the raw model source (config + tensors) so that
/// model-loading systems can reference it without a side-channel.
#[derive(Debug, Clone)]
pub struct ModelSourceComp(pub SourceModel);
impl Component for ModelSourceComp {}

// ────────────────────────────────────────────────────────────────────────
// Config → TextArchitecture extraction
// ────────────────────────────────────────────────────────────────────────

/// Extract a `TextArchitecture` from a HuggingFace-style `config.json` value.
///
/// Handles both top-level and `text_config`-nested layouts (common in
/// multi-modal models).
fn build_architecture_from_config(config: &Value, model_type: &str) -> TextArchitecture {
    // Some multi-modal configs nest text config under `text_config`.
    let cfg = config
        .get("text_config")
        .or_else(|| Some(config))
        .unwrap_or(config);

    let hs = num(cfg, "hidden_size");
    let im = num(cfg, "intermediate_size");
    let n_heads = num(cfg, "num_attention_heads");
    let n_kv = num_opt(cfg, "num_key_value_heads").unwrap_or(n_heads);
    let n_layers = num(cfg, "num_hidden_layers");
    let vocab = num(cfg, "vocab_size");
    let max_pos = num_opt(cfg, "max_position_embeddings").unwrap_or(2048);
    let sliding = num_opt(cfg, "sliding_window").unwrap_or(0);
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

    // RoPE
    let rope_theta = f64_val(cfg, "rope_theta").unwrap_or(10000.0);
    let rope_type = cfg
        .get("rope_type")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();
    let partial_rotary = f64_val(cfg, "partial_rotary_factor");

    // Layer types (default all sliding for standard, all full for deepseek)
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

    // MoE config (optional)
    let moe = if let Some(moe_val) = cfg.get("moe_config").or_else(|| cfg.get("moe")) {
        Some(ArchMoEConfig {
            num_experts: num_opt(moe_val, "num_experts").unwrap_or(0),
            top_k_experts: num_opt(moe_val, "top_k_experts").unwrap_or(0),
            intermediate_size: num_opt(moe_val, "intermediate_size").unwrap_or(im),
            shared_experts: bool_val(moe_val, "shared_experts").unwrap_or(false),
        })
    } else if model_type.starts_with("deepseek") || model_type == "ds4" {
        let num_experts = num_opt(cfg, "num_experts").unwrap_or(0);
        let top_k = num_opt(cfg, "top_k_experts")
            .or_else(|| num_opt(cfg, "num_routed_experts"))
            .or_else(|| num_opt(cfg, "num_experts").map(|n| n / 2))
            .unwrap_or(0);
        if num_experts > 0 {
            Some(ArchMoEConfig {
                num_experts,
                top_k_experts: top_k,
                intermediate_size: im,
                shared_experts: bool_val(cfg, "shared_experts").unwrap_or(true),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Diffusion config (optional, for diffusion_gemma and similar)
    let dcfg = cfg
        .get("diffusion_config")
        .or_else(|| cfg.get("_diffusion"));
    let diffusion_config = dcfg.map(|d| DiffusionConfig {
        max_diffusion_tokens: d
            .get("max_diffusion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as u32,
        default_denoising_steps: d
            .get("default_denoising_steps")
            .and_then(|v| v.as_u64())
            .unwrap_or(6) as u32,
        noise_schedule: match d.get("noise_schedule").and_then(|v| v.as_str()) {
            Some("cosine") => NoiseScheduleType::Cosine,
            Some("sqrt") => NoiseScheduleType::Sqrt,
            Some("linear") => NoiseScheduleType::Linear,
            _ => NoiseScheduleType::Cosine,
        },
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
            .unwrap_or(hs as u64) as u32,
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
    });

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

// ── Config value helpers ────────────────────────────────────────────

fn num(v: &Value, key: &str) -> u32 {
    v.get(key).and_then(|v| v.as_u64()).unwrap_or(0) as u32
}

fn num_opt(v: &Value, key: &str) -> Option<u32> {
    v.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
}

fn f64_val(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|v| v.as_f64())
}

fn bool_val(v: &Value, key: &str) -> Option<bool> {
    v.get(key).and_then(|v| v.as_bool())
}

// ────────────────────────────────────────────────────────────────────────
// DType conversion
// ────────────────────────────────────────────────────────────────────────

fn parse_dtype(dtype_str: &str) -> DType {
    match dtype_str {
        "f32" | "float32" | "F32" | "Float32" => DType::F32,
        "f16" | "float16" | "F16" | "Float16" | "half" | "Half" => DType::F16,
        "bf16" | "bfloat16" | "BF16" | "BFloat16" => DType::BF16,
        "i8" | "int8" | "I8" | "Int8" => DType::I8,
        "i4" | "int4" | "I4" | "Int4" => DType::I4,
        "i2" | "int2" | "I2" | "Int2" => DType::I2,
        other => {
            tracing::warn!(
                "ModelAdapterSystem: unknown dtype '{}', defaulting to F32",
                other
            );
            DType::F32
        }
    }
}

// TextArchitecture is used as an ECS component throughout the pipeline.
impl Component for TextArchitecture {}

// ────────────────────────────────────────────────────────────────────────
// ModelAdapterSystem — ECS-native implementation
// ────────────────────────────────────────────────────────────────────────

/// Parses model config JSON inline (without adapter dispatch), creates
/// Tensor + Layer entities from source tensor names, and stores a
/// `TextArchitecture` component on the model entity.
///
/// Uses the pattern-based tensor matching engine defined in
/// `ecs::adapter` so all adapter knowledge (tensor name → role mappings)
/// is captured in one place.
pub struct ModelAdapterSystem;

impl CompilerSystem for ModelAdapterSystem {
    fn name(&self) -> &str {
        "ModelAdapterSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::ModelLoading
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        for model_entity in world.entities_of_kind(EntityKind::Model) {
            // Read the raw source from the model entity
            let source = world
                .get_component::<ModelSourceComp>(model_entity)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ModelAdapterSystem: entity {:?} has no ModelSourceComp",
                        model_entity
                    )
                })?
                .0
                .clone();

            // ── 1. Determine model type and build architecture ──────────
            let model_type = &source.model_type;
            let architecture = build_architecture_from_config(&source.config, model_type);

            // ── 2. Parse tensor names into roles using pattern matching ─
            let patterns = architecture_patterns(model_type);
            let mut seen_layers: BTreeSet<u32> = BTreeSet::new();

            for tensor_name in &source.tensor_names {
                let mut matched = false;
                for tp in patterns {
                    if let Some((layer, expert)) = match_tensor_pattern(tensor_name, tp) {
                        let role = (tp.role)(layer, expert);

                        // Look up shape and dtype from source.tensors
                        let (dtype_str, shape, _data) = match source.tensors.get(tensor_name) {
                            Some(info) => info,
                            None => {
                                tracing::warn!(
                                    "ModelAdapterSystem: tensor '{}' not found in source data, skipping",
                                    tensor_name
                                );
                                continue;
                            }
                        };

                        let tensor_entity =
                            world.spawn(EntityKind::Tensor, Some(tensor_name.clone()))?;

                        let _ = world.add_component(tensor_entity, Shape(shape.clone()));
                        let _ =
                            world.add_component(tensor_entity, DataType(parse_dtype(dtype_str)));
                        let _ = world.add_component(tensor_entity, CanonicalRoleComp(role));

                        // Track layer membership
                        seen_layers.insert(layer);

                        matched = true;
                        break; // first matching pattern wins
                    }
                }
                if !matched {
                    tracing::warn!(
                        "ModelAdapterSystem: could not parse tensor name '{}', skipping",
                        tensor_name
                    );
                }
            }

            // ── 3. Create one Layer entity per unique layer index ──────
            for layer_idx in &seen_layers {
                let layer_entity =
                    world.spawn(EntityKind::Layer, Some(format!("layer_{}", layer_idx)))?;
                let _ = world.add_component(layer_entity, LayerIndex(*layer_idx));
            }

            // Store the architecture on the model entity (after all
            // tensor processing to avoid E0502 borrow conflicts).
            let _ = world.add_component(model_entity, architecture);
        }
        Ok(())
    }
}
