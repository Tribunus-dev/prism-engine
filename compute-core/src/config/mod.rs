//! Config-driven architecture for Tribunus Compute Kernel.
//!
//! Layer 1: Raw model manifest — captures config.json hash and structure.
//! Layer 2: Normalized architecture — strict Rust types from JSON.
//! Layer 3: Compiled execution specification — per-layer dimensions, policies, tensor shapes.
//!
//! This module is decomposed into sub-modules:
//! - `hardware`: Architecture types + compile-related types
//! - `parser`: Raw JSON parsing and manifest types
//! - `network`: Server configuration
//! - `limits`: Compilation planning types
//! - `operation_route`: Per-operation backend routing

pub mod operation_route;
pub mod hardware;
pub mod parser;
pub mod network;
pub mod limits;

// Re-exports for backward compatibility — everything accessible
// at `crate::config::*` as before.
pub use hardware::*;
pub use parser::*;
pub use network::*;
pub use limits::*;

// Re-exported from config_namespace module.
pub use crate::config_namespace::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_layer(index: u32) -> LayerPlan {
        LayerPlan {
            layer_index: index,
            attention_kind: "sliding_attention".into(),
            segment_id: format!("layer_{}", index),
            hidden_size: 64,
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 16,
            global_head_dim: None,
            n_global_kv_heads: None,
            sliding_window: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: None,
            attention_k_eq_v: false,
            q_norm_enabled: false,
            k_norm_enabled: false,
            q_proj_tensor_id: 1,
            k_proj_tensor_id: 2,
            v_proj_tensor_id: 3,
            o_proj_tensor_id: 4,
            q_norm_tensor_id: None,
            k_norm_tensor_id: None,
            gate_proj_tensor_id: 5,
            up_proj_tensor_id: 6,
            down_proj_tensor_id: 7,
            input_layernorm_tensor_id: 8,
            post_attention_layernorm_tensor_id: 9,
            pre_ffw_layernorm_tensor_id: None,
            post_ffw_layernorm_tensor_id: None,
            layer_scalar_ids: Vec::new(),
            quantization_ids: Vec::new(),
            route: Default::default(),
            fused_operations: Default::default(),
        }
    }

    fn base_plan() -> ModelExecutionPlan {
        ModelExecutionPlan {
            prologue: ProloguePlan {
                segment_id: "persistent".into(),
                embedding_tensor_id: 10,
                embedding_name: "model.embed_tokens.weight".into(),
                embedding_shape: vec![64, 64],
                embedding_dtype: "U8".into(),
            },
            layers: vec![valid_layer(0)],
            epilogue: EpiloguePlan {
                segment_id: "persistent".into(),
                final_norm_tensor_id: 11,
                final_norm_name: "model.norm.weight".into(),
                output_projection_tensor_id: None,
                output_projection_name: None,
                final_logit_softcapping: None,
                vocab_size: 64,
            },
            hidden_size: 64,
            vocab_size: 64,
            sliding_window: 4096,
            final_logit_softcapping: None,
            tie_word_embeddings: true,
            rms_norm_eps: 1e-6,
            fused_ane_islands: vec![],
            speculative_config: None,
            generation_regime: Default::default(),
            diffusion_config: Default::default(),
            diffusion_execution_plan: Default::default(),
            kv_cache_mode: Default::default(),
        }
    }

    #[test]
    fn validate_rejects_malformed_plans() {
        // 1. Zero layers
        {
            let mut plan = base_plan();
            plan.layers.clear();
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("execution plan has zero layers")),
                "expected zero-layers error, got: {:?}",
                errs
            );
        }

        // 2. Layer index mismatch (layer at index 1 has layer_index=0)
        {
            let mut plan = base_plan();
            let mut l1 = valid_layer(1);
            l1.layer_index = 0;
            plan.layers.push(l1);
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("layer 1 has index 0")),
                "expected index mismatch error, got: {:?}",
                errs
            );
        }

        // 3. Layer hidden_size != model hidden_size
        {
            let mut plan = base_plan();
            plan.layers[0].hidden_size = 128;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("hidden_size") && e.contains("128") && e.contains("64")),
                "expected hidden_size mismatch error, got: {:?}",
                errs
            );
        }

        // 4. q_proj_tensor_id = 0
        {
            let mut plan = base_plan();
            plan.layers[0].q_proj_tensor_id = 0;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("zero q_proj_tensor_id")),
                "expected zero q_proj_tensor_id error, got: {:?}",
                errs
            );
        }

        // 5. full_attention layer missing global_head_dim
        {
            let mut plan = base_plan();
            plan.layers[0].attention_kind = "full_attention".into();
            plan.layers[0].global_head_dim = None;
            // full_attention branch checks global_head_dim, not v_proj
            plan.layers[0].v_proj_tensor_id = 99;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("missing global_head_dim")),
                "expected missing global_head_dim error, got: {:?}",
                errs
            );
        }

        // 6. Unknown attention_kind
        {
            let mut plan = base_plan();
            plan.layers[0].attention_kind = "bogus".into();
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter()
                    .any(|e| e.contains("unknown attention_kind: bogus")),
                "expected unknown attention_kind error, got: {:?}",
                errs
            );
        }

        // 8. Epilogue with zero final_norm_tensor_id
        {
            let mut plan = base_plan();
            plan.epilogue.final_norm_tensor_id = 0;
            let errs = plan.validate().unwrap_err();
            assert!(
                errs.iter().any(|e| e.contains("zero final_norm_tensor_id")),
                "expected zero final_norm_tensor_id error, got: {:?}",
                errs
            );
        }
    }
}
