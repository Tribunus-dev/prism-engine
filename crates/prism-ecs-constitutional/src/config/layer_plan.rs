//! Layer 3: per-layer compile-time plan and the per-tensor binding
//! vocabulary that describes the model in the compiler's terms.
//!
//! Authority: the canonical [`ExecutionSpec`], [`LayerSpec`],
//! [`TensorBinding`], [`TensorRole`], and [`PackedLinearShapes`]
//! types plus the [`compile`] and [`build_execution_plan`] pure-Rust
//! routines that turn a [`super::architecture::TextArchitecture`] +
//! [`super::namespace_binding::NamespaceBinding`] into a layered,
//! backend-routed execution plan. All effects (tensor table
//! persistence, ANE compile, segment emission) are the caller's
//! responsibility; this module is data + pure transformation.

use serde::Serialize;
use std::collections::BTreeMap;

use super::architecture::{
    AttentionKind, QuantizationMeta, TextArchitecture,
};
use super::namespace_binding::NamespaceBinding;
use super::operation_route::OperationRoute;

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

/// Build a [`super::model_execution_plan::ModelExecutionPlan`] from a
/// [`TextArchitecture`], a [`NamespaceBinding`], and the caller's
/// `emitted_ids` table (tensor name → emitted tensor id).
///
/// The `emitted_ids` map is a transient lookup keyed by
/// `<root>.layers.<i>.self_attn.q_proj.weight` etc.; a `BTreeMap` is
/// required so the resulting plan is deterministic across runs and
/// across hash implementations.
pub fn build_execution_plan(
    arch: &TextArchitecture,
    namespace: &NamespaceBinding,
    emitted_ids: &BTreeMap<String, u32>,
) -> super::model_execution_plan::ModelExecutionPlan {
    use super::model_execution_plan::{
        EpiloguePlan, LayerPlan, ModelExecutionPlan, ProloguePlan,
    };

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
        layer.route = OperationRoute {
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

/// Compile a [`TextArchitecture`] into an [`ExecutionSpec`].
///
/// The resulting spec lists every expected tensor binding in
/// execution order. Callers should then [`filter_spec_to_existing`]
/// against the actual safetensors manifest before emission.
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

/// Filter the compiled spec to only include bindings for tensors that
/// exist in the source model's tensor map. Names not present are
/// dropped (with a `eprintln!` notice) so the compiled spec matches
/// the actual safetensors on disk.
pub fn filter_spec_to_existing(spec: &mut ExecutionSpec, existing_tensor_names: &BTreeSet<String>) {
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

use std::collections::BTreeSet;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::architecture::{
        AttentionKind, AudioArchitecture, QuantizationMeta, QuantizationMode, RopeSpec,
        TextArchitecture, VisionArchitecture,
    };
    use crate::config::namespace_binding::NamespaceBinding;
    use crate::config::operation_route::OperationRoute;
    use crate::config::model_execution_plan::{
        EpiloguePlan, LayerPlan, ModelExecutionPlan, ProloguePlan,
    };

    fn sample_arch() -> TextArchitecture {
        TextArchitecture {
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            num_hidden_layers: 2,
            vocab_size: 128,
            sliding_window: 64,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_k_eq_v: true,
            hidden_size_per_layer_input: 0,
            layer_types: vec![
                AttentionKind::SlidingAttention,
                AttentionKind::FullAttention,
            ],
            rope_local: RopeSpec {
                theta: 10_000.0,
                rope_type: "default".into(),
                partial_rotary_factor: None,
            },
            rope_global: Some(RopeSpec {
                theta: 1_000_000.0,
                rope_type: "proportional".into(),
                partial_rotary_factor: Some(0.5),
            }),
            model_type: "gemma4_unified_text".into(),
            ..Default::default()
        }
    }

    fn sample_namespace() -> NamespaceBinding {
        NamespaceBinding {
            root: "model.language_model".into(),
            discovery: "test".into(),
            lm_head_key: "model.language_model.lm_head.weight".into(),
            lm_head_aliased: false,
        }
    }

    #[test]
    fn compile_emits_one_layer_spec_per_arch_layer() {
        let arch = sample_arch();
        let ns = sample_namespace();
        let spec = compile(&arch, &ns, None);
        assert_eq!(spec.layers.len(), arch.layer_types.len());
        // Embedding + final norm; no LM head because tied.
        assert_eq!(spec.global_tensors.len(), 2);
    }

    #[test]
    fn compile_emits_lm_head_when_untied() {
        let mut arch = sample_arch();
        arch.tie_word_embeddings = false;
        let ns = sample_namespace();
        let spec = compile(&arch, &ns, None);
        assert_eq!(spec.global_tensors.len(), 3);
        assert!(spec
            .global_tensors
            .iter()
            .any(|b| b.role == TensorRole::LmHead));
    }

    #[test]
    fn build_execution_plan_assigns_routes_per_attention_kind() {
        let arch = sample_arch();
        let ns = sample_namespace();
        let mut emitted = BTreeMap::new();
        emitted.insert("model.language_model.embed_tokens.weight".into(), 1);
        emitted.insert("model.language_model.norm.weight".into(), 2);
        let plan = build_execution_plan(&arch, &ns, &emitted);
        assert_eq!(plan.layers.len(), 2);
        // Sliding (layer 0) routes attention to MLX (0); full (layer 1)
        // routes attention to ANE (3).
        assert_eq!(plan.layers[0].route.attention, 0);
        assert_eq!(plan.layers[1].route.attention, 3);
        assert_eq!(plan.prologue.embedding_tensor_id, 1);
    }

    #[test]
    fn filter_spec_to_existing_drops_unknown_tensors() {
        let arch = sample_arch();
        let ns = sample_namespace();
        let mut spec = compile(&arch, &ns, None);
        // Build a set containing only the embedding (drop the rest).
        let mut existing = BTreeSet::new();
        existing.insert("model.language_model.embed_tokens.weight".into());
        filter_spec_to_existing(&mut spec, &existing);
        // The final-norm tensor is gone from globals.
        assert!(spec
            .global_tensors
            .iter()
            .all(|b| b.name.ends_with("embed_tokens.weight")));
    }

    // Suppress unused imports warning for items used through the public
    // surface; tests reference them via the module's public re-exports.
    #[allow(dead_code)]
    fn _unused_imports(
        _vm: VisionArchitecture,
        _am: AudioArchitecture,
        _qm: QuantizationMeta,
        _mode: QuantizationMode,
        _plan: ModelExecutionPlan,
        _prologue: ProloguePlan,
        _epilogue: EpiloguePlan,
        _layer_plan: LayerPlan,
        _route: OperationRoute,
    ) {
    }
}
