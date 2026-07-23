//! Bonsai model integration — ternary/binary codecs and architecture config.
//!
//! Bonsai models use low-bit quantization formats (ternary 1.58-bit, binary 1-bit)
//! with specialized GEMM operations. This module provides:
//!
//! - BonsaiTensorConfig: per-tensor quantization and operation configuration
//! - BonsaiArchitectureConfig: model-wide architecture parameters
//! - BonsaiContextValidation: 262K context length validation
//!
//! The evolution.rs module already defines TensorFormat and TensorOperation
//! which cover the Bonsai search space. This module adds the Bonsai-specific
//! configuration structures.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType { FullAttention, LinearAttention }

pub struct Bonsai27B;
impl Bonsai27B {
    pub const LAYERS:u32=64; pub const HIDDEN_DIM:u32=5120; pub const INTERMEDIATE_DIM:u32=17408;
    pub const NUM_HEADS:u32=24; pub const NUM_KV_HEADS:u32=4; pub const KEY_LENGTH:u32=256;
    pub const VALUE_LENGTH:u32=256; pub const HEAD_DIM:u32=64; pub const VOCAB_SIZE:u32=248320;
    pub const CONTEXT_LENGTH:u32=262144; pub const NORM_EPS:f32=1e-6;
    pub fn layer_type(layer:u32)->LayerType { if layer < Self::LAYERS && layer % 4 == 0 { LayerType::FullAttention } else { LayerType::LinearAttention } }
}

/// Bonsai-specific quantization configuration for a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonsaiTensorConfig {
    /// Tensor name (e.g. "q_proj", "k_proj", "v_proj", "o_proj").
    pub name: String,
    /// The quantization format.
    pub format: crate::evolution::TensorFormat,
    /// The computation operation.
    pub operation: crate::evolution::TensorOperation,
}

impl BonsaiTensorConfig {
    pub fn new(name: impl Into<String>, format: crate::evolution::TensorFormat) -> Self {
        let operation = match format {
            crate::evolution::TensorFormat::Ternary158 => {
                crate::evolution::TensorOperation::TernaryGemm
            }
            crate::evolution::TensorFormat::Binary1 => {
                crate::evolution::TensorOperation::BinaryPopcountGemm
            }
            crate::evolution::TensorFormat::Int4 => {
                crate::evolution::TensorOperation::Int4DequantMatmul
            }
            _ => crate::evolution::TensorOperation::Matmul,
        };
        Self {
            name: name.into(),
            format,
            operation,
        }
    }
}

/// Bonsai model architecture configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BonsaiArchitectureConfig {
    /// Whether to use ternary weights (1.58-bit).
    pub use_ternary: bool,
    /// Whether to use binary weights (1-bit).
    pub use_binary: bool,
    /// Number of attention heads.
    pub num_attention_heads: u32,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// Maximum context length (Bonsai supports 262K).
    pub max_context_length: u32,
}

impl BonsaiArchitectureConfig {
    pub fn new(hidden_size: u32, num_attention_heads: u32) -> Self {
        Self {
            use_ternary: true,
            use_binary: false,
            num_attention_heads,
            hidden_size,
            max_context_length: 262_144, // 262K default
        }
    }

    /// Validate the context length is within Bonsai's supported range.
    pub fn validate_context_length(&self, context_length: u32) -> Result<(), String> {
        if context_length > self.max_context_length {
            return Err(format!(
                "context length {} exceeds Bonsai maximum of {}",
                context_length, self.max_context_length
            ));
        }
        if context_length == 0 {
            return Err("context length must be > 0".into());
        }
        Ok(())
    }

    /// Get the GEMM operation for this architecture.
    pub fn gemm_operation(&self) -> crate::evolution::TensorOperation {
        if self.use_ternary {
            crate::evolution::TensorOperation::TernaryGemm
        } else if self.use_binary {
            crate::evolution::TensorOperation::BinaryPopcountGemm
        } else {
            crate::evolution::TensorOperation::Matmul
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::{TensorFormat, TensorOperation};

    #[test]
    fn bonsai_tensor_config() {
        let config = BonsaiTensorConfig::new("q_proj", TensorFormat::Ternary158);
        assert_eq!(config.name, "q_proj");
        assert_eq!(config.format, TensorFormat::Ternary158);
        assert_eq!(config.operation, TensorOperation::TernaryGemm);
    }

    #[test]
    fn bonsai_architecture_config() {
        let config = BonsaiArchitectureConfig::new(4096, 32);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.max_context_length, 262_144);
        assert!(config.use_ternary);
    }

    #[test]
    fn bonsai_context_validation() {
        let config = BonsaiArchitectureConfig::new(4096, 32);

        // Valid context
        assert!(config.validate_context_length(262_144).is_ok());
        assert!(config.validate_context_length(128_000).is_ok());

        // Invalid: exceeds max
        assert!(config.validate_context_length(300_000).is_err());

        // Invalid: zero
        assert!(config.validate_context_length(0).is_err());
    }

    #[test]
    fn bonsai_tensor_config_auto_operation() {
        let t = BonsaiTensorConfig::new("o_proj", TensorFormat::Binary1);
        assert_eq!(t.operation, TensorOperation::BinaryPopcountGemm);

        let m = BonsaiTensorConfig::new("mlp", TensorFormat::Int4);
        assert_eq!(m.operation, TensorOperation::Int4DequantMatmul);

        let f = BonsaiTensorConfig::new("norm", TensorFormat::Fp16);
        assert_eq!(f.operation, TensorOperation::Matmul);
    }
}
