//! GGUF frontend — reads GGUF model files into canonical ModelIr.
//!
//! This frontend parses GGUF file headers and tensor inventories, then
//! maps them to the canonical model IR types. It handles both inspection
//! (metadata-only) and full import (with tensor catalogue) workflows.

#[cfg(feature = "prism-backend")]
use crate::ecs::aot::prism_compiler::ModelFrontend;
#[cfg(feature = "prism-backend")]
use crate::ecs::canonical::compile_plan::{InspectRequest, ModelInspection};
#[cfg(feature = "prism-backend")]
use crate::ecs::canonical::model_ir::{
    ArchitectureId, LogicalGraph, ModelConfiguration, ModelIdentity, ModelIr, SourceProvenance,
    SourceType, TensorCatalogue, TensorDescriptor, TensorId, TokenizerDescriptor,
};
#[cfg(feature = "prism-backend")]
use crate::ecs::legacy_core::gguf;
#[cfg(feature = "prism-backend")]
use std::collections::HashMap;
#[cfg(feature = "prism-backend")]
use std::path::Path;

/// GGUF model frontend — reads `.gguf` files and produces canonical types.
pub struct GgufFrontend;

impl GgufFrontend {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "prism-backend")]
impl ModelFrontend for GgufFrontend {
    fn inspect(&self, source: &InspectRequest) -> Result<ModelInspection, String> {
        let path = Path::new(&source.source_path);
        let (metadata, tensors) = gguf::parse_gguf_header(path)?;

        let arch_config = gguf::extract_architecture(&metadata)?;

        let total_weight_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();

        Ok(ModelInspection {
            identity: ModelIdentity {
                name: path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                revision: None,
            },
            architecture: ArchitectureId(arch_config.model_type.clone()),
            configuration: build_model_configuration(&metadata, &arch_config),
            tensor_count: tensors.len(),
            total_weight_bytes,
        })
    }

    fn import(&self, source: &InspectRequest) -> Result<ModelIr, String> {
        let path = Path::new(&source.source_path);
        let (metadata, tensors) = gguf::parse_gguf_header(path)?;

        let arch_config = gguf::extract_architecture(&metadata)?;

        // Build tensor catalogue from the GGUF tensor inventory
        let mut by_id = Vec::with_capacity(tensors.len());
        let mut by_name = HashMap::with_capacity(tensors.len());
        for (idx, t) in tensors.iter().enumerate() {
            let id = TensorId(idx);
            by_name.insert(t.name.clone(), id);
            by_id.push(TensorDescriptor {
                id,
                name: t.name.clone(),
                shape: t.shape.iter().map(|&s| s as usize).collect(),
                byte_size: t.byte_size,
                is_lazy: true,
            });
        }

        let model_ir = ModelIr {
            identity: ModelIdentity {
                name: path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                revision: None,
            },
            architecture: ArchitectureId(arch_config.model_type.clone()),
            configuration: build_model_configuration(&metadata, &arch_config),
            tensors: TensorCatalogue { by_id, by_name },
            graph: LogicalGraph {
                ops: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            tokenizer: TokenizerDescriptor {
                tokenizer_type: "gguf".to_string(),
                vocab_size: arch_config.vocab_size as usize,
                bos_token_id: None,
                eos_token_id: None,
                pad_token_id: None,
            },
            source_provenance: SourceProvenance {
                source_type: SourceType::Gguf,
                source_path: source.source_path.clone(),
                file_digests: Vec::new(),
            },
        };

        Ok(model_ir)
    }
}

/// Build a canonical `ModelConfiguration` from GGUF metadata and extracted architecture.
#[cfg(feature = "prism-backend")]
fn build_model_configuration(
    metadata: &[(String, String)],
    arch: &crate::ecs::config::TextArchitecture,
) -> ModelConfiguration {
    let arch_prefix = &arch.model_type;

    // Read a u64 metadata value — try arch-prefixed key first, then generic.
    let meta_u64 = |generic_key: &str| -> Option<u64> {
        let with_prefix = format!("{}.{}", arch_prefix, generic_key);
        if let Some((_, v)) = metadata.iter().find(|(k, _)| *k == with_prefix) {
            return v.parse::<u64>().ok();
        }
        let (_, v) = metadata.iter().find(|(k, _)| *k == generic_key)?;
        v.parse::<u64>().ok()
    };

    // Read an f64 metadata value with the same flexible key resolution.
    let meta_f64 = |generic_key: &str| -> Option<f64> {
        let with_prefix = format!("{}.{}", arch_prefix, generic_key);
        if let Some((_, v)) = metadata.iter().find(|(k, _)| *k == with_prefix) {
            return v.parse::<f64>().ok();
        }
        let (_, v) = metadata.iter().find(|(k, _)| *k == generic_key)?;
        v.parse::<f64>().ok()
    };

    let hidden_size: usize =
        meta_u64("embedding_length").unwrap_or(arch.hidden_size as u64) as usize;
    let num_hidden_layers: usize =
        meta_u64("block_count").unwrap_or(arch.num_hidden_layers as u64) as usize;
    let num_attention_heads: usize =
        meta_u64("attention.head_count").unwrap_or(arch.num_attention_heads as u64) as usize;
    let num_kv_heads: usize =
        meta_u64("attention.head_count_kv").unwrap_or(arch.num_key_value_heads as u64) as usize;
    let head_dim: usize = meta_u64("attention.head_dim").unwrap_or_else(|| {
        if num_attention_heads > 0 {
            hidden_size as u64 / num_attention_heads as u64
        } else {
            arch.head_dim as u64
        }
    }) as usize;
    let vocab_size: usize = meta_u64("vocab_size").unwrap_or(arch.vocab_size as u64) as usize;
    let max_position_embeddings: usize = meta_u64("context_length")
        .or_else(|| meta_u64("max_position_embeddings"))
        .unwrap_or(arch.max_position_embeddings as u64)
        as usize;
    let intermediate_size: usize =
        meta_u64("feed_forward_length").unwrap_or(arch.intermediate_size as u64) as usize;
    let rms_norm_eps: f64 = meta_f64("attention.layer_norm_rms_epsilon")
        .or_else(|| meta_f64("rms_norm_eps"))
        .unwrap_or(arch.rms_norm_eps);
    let rope_theta: Option<f64> = meta_u64("rope.freq_base")
        .or_else(|| meta_u64("rope_theta"))
        .map(|v| v as f64);

    ModelConfiguration {
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_kv_heads,
        num_hidden_layers,
        head_dim,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        partial_rope_dim: None,
        tie_word_embeddings: arch.tie_word_embeddings,
        num_experts: arch.moe_config.as_ref().map(|m| m.num_experts as usize),
        num_experts_per_tok: arch.moe_config.as_ref().map(|m| m.top_k_experts as usize),
        moe_intermediate_size: None,
        num_mtp_heads: None,
        mtp_hidden_size: None,
        mtp_intermediate_size: None,
    }
}
