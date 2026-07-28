//! Gemma 4 safetensors frontend — wraps `inspect_gemma4_checkpoint` into
//! the PrismCompiler's `ModelFrontend` trait so the deployment compiler
//! can read real Gemma 4 unified checkpoints without a GGUF conversion step.

use std::path::Path;

use crate::ecs::aot::prism_compiler::ModelFrontend;
use crate::ecs::canonical::compile_plan::{InspectRequest, ModelInspection};
use crate::ecs::canonical::model_ir::{
    ArchitectureId, LogicalGraph, ModelConfiguration, ModelIdentity, ModelIr, SourceProvenance,
    SourceType, TensorCatalogue, TensorDescriptor, TensorId, TokenizerDescriptor,
};
use crate::ecs::legacy_compute_image_core::model_family::gemma4_inspect::inspect_gemma4_checkpoint;

pub struct Gemma4SafetensorsFrontend;

impl Gemma4SafetensorsFrontend {
    pub fn new() -> Self {
        Self
    }
}

impl ModelFrontend for Gemma4SafetensorsFrontend {
    fn inspect(&self, source: &InspectRequest) -> Result<ModelInspection, String> {
        let path = Path::new(&source.source_path);
        if !path.is_dir() {
            return Err(format!(
                "Gemma4SafetensorsFrontend: not a directory: {}",
                source.source_path
            ));
        }
        let inspection = inspect_gemma4_checkpoint(path)?;

        let hd = inspection.config.hidden_size as usize;
        let nh = inspection.config.num_attention_heads as usize;
        Ok(ModelInspection {
            identity: ModelIdentity {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                revision: None,
            },
            architecture: ArchitectureId("gemma-4-unified".into()),
            configuration: ModelConfiguration {
                hidden_size: hd,
                intermediate_size: inspection.config.intermediate_size as usize,
                num_attention_heads: nh,
                num_kv_heads: inspection.config.num_key_value_heads as usize,
                num_hidden_layers: inspection.config.num_layers as usize,
                head_dim: if nh > 0 { hd / nh } else { 64 },
                vocab_size: inspection.config.vocab_size as usize,
                max_position_embeddings: 262144,
                rms_norm_eps: 1e-6,
                rope_theta: Some(10000.0),
                partial_rope_dim: None,
                tie_word_embeddings: true,
                num_experts: None,
                num_experts_per_tok: None,
                moe_intermediate_size: None,
                num_mtp_heads: None,
                mtp_hidden_size: None,
                mtp_intermediate_size: None,
            },
            tensor_count: inspection.inventory.total_tensors,
            total_weight_bytes: inspection.inventory.total_tensors as u64 * 4,
        })
    }

    fn import(&self, source: &InspectRequest) -> Result<ModelIr, String> {
        let path = Path::new(&source.source_path);
        let inspection = inspect_gemma4_checkpoint(path)?;
        let hd = inspection.config.hidden_size as usize;
        let nh = inspection.config.num_attention_heads as usize;

        let mut by_id = Vec::new();
        let mut by_name = std::collections::HashMap::new();
        for (i, entry) in inspection.inventory.tensors.iter().enumerate() {
            let tid = TensorId(i);
            by_name.insert(entry.name.clone(), tid);
            by_id.push(TensorDescriptor {
                id: tid,
                name: entry.name.clone(),
                shape: entry.shape.clone(),
                byte_size: entry.param_count as u64 * 4,
                is_lazy: true,
            });
        }

        Ok(ModelIr {
            identity: ModelIdentity {
                name: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                revision: None,
            },
            architecture: ArchitectureId("gemma-4-unified".into()),
            configuration: ModelConfiguration {
                hidden_size: hd,
                intermediate_size: inspection.config.intermediate_size as usize,
                num_attention_heads: nh,
                num_kv_heads: inspection.config.num_key_value_heads as usize,
                num_hidden_layers: inspection.config.num_layers as usize,
                head_dim: if nh > 0 { hd / nh } else { 64 },
                vocab_size: inspection.config.vocab_size as usize,
                max_position_embeddings: 262144,
                rms_norm_eps: 1e-6,
                rope_theta: Some(10000.0),
                partial_rope_dim: None,
                tie_word_embeddings: true,
                num_experts: None,
                num_experts_per_tok: None,
                moe_intermediate_size: None,
                num_mtp_heads: None,
                mtp_hidden_size: None,
                mtp_intermediate_size: None,
            },
            tensors: TensorCatalogue { by_id, by_name },
            graph: LogicalGraph {
                ops: vec![],
                inputs: vec![],
                outputs: vec![],
            },
            tokenizer: TokenizerDescriptor {
                tokenizer_type: "gemma4".into(),
                vocab_size: inspection.config.vocab_size as usize,
                bos_token_id: Some(2),
                eos_token_id: Some(1),
                pad_token_id: Some(0),
            },
            source_provenance: SourceProvenance {
                source_type: SourceType::Safetensors,
                source_path: source.source_path.clone(),
                file_digests: Vec::new(),
            },
        })
    }
}
