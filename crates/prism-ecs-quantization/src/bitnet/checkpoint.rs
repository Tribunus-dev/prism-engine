//! BitNet b1.58 2B4T real checkpoint loader.
//!
//! Loads packed ternary weights and BF16 metadata from a HuggingFace
//! safetensors checkpoint, and provides accessors for emitting cimage
//! shards.
//!
//! ## File format
//! - All weight tensors are `U8` dtype — already packed 2-bit ternary
//!   codes (4 weights per byte, encoding: 00=-1, 01=0, 10=+1).
//! - Each weight tensor has a companion `weight_scale` tensor — `BF16` [1].
//! - Layer norms are `BF16` [hidden_dim] or [intermediate_dim].
//! - `embed_tokens.weight` is `BF16` [vocab_size, hidden_dim].

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;
use thiserror::Error;

use super::ternary_codec::TernaryPackedTensor;

/// Extract the last (innermost) dimension from a tensor shape.
/// Handles both `[dim]` and `[1, dim]` shapes.
fn last_dim(shape: &[usize]) -> Option<&usize> {
    shape.last()
}

/// Errors that can occur during checkpoint loading and access.
#[derive(Debug, Error)]
pub enum BitNetCheckpointError {
    /// I/O error reading the file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Safetensors parsing or lookup failure.
    #[error("safetensors error: {0}")]
    SafeTensors(String),

    /// The named tensor was not found in the checkpoint.
    #[error("tensor '{0}' not found in checkpoint")]
    TensorNotFound(String),

    /// The tensor's dtype is not what the accessor expected.
    #[error("unexpected dtype for tensor '{tensor}': expected {expected:?}, got {got:?}")]
    DtypeMismatch {
        tensor: String,
        expected: String,
        got: String,
    },

    /// The tensor's shape is not what the accessor expected.
    #[error("shape mismatch for tensor '{tensor}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// The byte buffer for a BF16 value is too short.
    #[error("bf16 data too short for tensor '{tensor}': expected at least {expected} bytes, got {actual}")]
    Bf16DataTooShort {
        tensor: String,
        expected: usize,
        actual: usize,
    },
}

/// A loaded BitNet b1.58 2B4T checkpoint.
///
/// Holds the safetensors byte buffer and its parsed metadata/indices.
/// `SafeTensors` borrows from `_buffer` via an unsafe `'static`
/// transmute — this is sound because `_buffer` is never mutated or
/// dropped before `tensors`.
pub struct BitNetCheckpoint {
    /// Owned byte buffer for the safetensors file.
    _buffer: Vec<u8>,
    /// Parsed safetensors indexing into `_buffer`.
    tensors: SafeTensors<'static>,
    /// Number of transformer layers (inferred from tensor names).
    pub num_layers: usize,
    /// Hidden / model dimension.
    pub hidden_dim: usize,
    /// MLP intermediate (ffw) dimension.
    pub intermediate_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

const MODEL_PREFIX: &str = "model.layers.";

impl BitNetCheckpoint {
    /// Load a BitNet safetensors checkpoint from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, BitNetCheckpointError> {
        let buffer = std::fs::read(path.as_ref())?;
        Self::from_buffer(buffer)
    }

    /// Parse a safetensors byte buffer into a checkpoint.
    pub fn from_buffer(buffer: Vec<u8>) -> Result<Self, BitNetCheckpointError> {
        // SAFETY: `tensors` borrows from `_buffer`. We transmute the
        // lifetime to `'static` because `BitNetCheckpoint` owns the
        // buffer and never moves or drops it before this struct is
        // dropped (it is the last field). This is the standard
        // pattern used by the safetensors crate's own docs.
        let tensors = unsafe {
            std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(
                SafeTensors::deserialize(&buffer)
                    .map_err(|e| BitNetCheckpointError::SafeTensors(format!("{e:?}")))?,
            )
        };

        let embed_shape = tensors
            .tensor("model.embed_tokens.weight")
            .map_err(|_| BitNetCheckpointError::TensorNotFound("model.embed_tokens.weight".into()))?
            .shape()
            .to_vec();
        let vocab_size = embed_shape[0];

        let ln_shape = tensors
            .tensor("model.layers.0.input_layernorm.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound(
                    "model.layers.0.input_layernorm.weight".into(),
                )
            })?
            .shape()
            .to_vec();
        let hidden_dim =
            *last_dim(&ln_shape).ok_or_else(|| BitNetCheckpointError::ShapeMismatch {
                tensor: "model.layers.0.input_layernorm.weight".into(),
                expected: vec![2560],
                got: ln_shape.clone(),
            })?;

        let gate_shape = tensors
            .tensor("model.layers.0.mlp.gate_proj.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound(
                    "model.layers.0.mlp.gate_proj.weight".into(),
                )
            })?
            .shape()
            .to_vec();
        let intermediate_dim = gate_shape[0] * 4;

        let mut max_layer: usize = 0;
        for name in tensors.names() {
            if let Some(rest) = name.strip_prefix(MODEL_PREFIX) {
                if let Some(dot) = rest.find('.') {
                    if let Ok(n) = rest[..dot].parse::<usize>() {
                        max_layer = max_layer.max(n + 1);
                    }
                }
            }
        }
        let num_layers = max_layer;

        let k_shape = tensors
            .tensor("model.layers.0.self_attn.k_proj.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound(
                    "model.layers.0.self_attn.k_proj.weight".into(),
                )
            })?
            .shape()
            .to_vec();
        let kv_inner = k_shape[0] * 4;

        let head_dim: usize = 128;
        let num_heads = hidden_dim / head_dim;
        let num_kv_heads = kv_inner / head_dim;

        Ok(BitNetCheckpoint {
            _buffer: buffer,
            tensors,
            num_layers,
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
        })
    }

    /// Return the raw U8 code bytes for `model.layers.{layer}.{name}.weight`.
    pub fn layer_codes(&self, layer: usize, name: &str) -> Result<&[u8], BitNetCheckpointError> {
        let tensor_name = format!("{}{}.weight", self.layer_prefix(layer), name);
        let view = self
            .tensors
            .tensor(&tensor_name)
            .map_err(|_| BitNetCheckpointError::TensorNotFound(tensor_name.clone()))?;
        if view.dtype() != Dtype::U8 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: tensor_name,
                expected: "U8".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(view.data())
    }

    /// Extract the per-tensor scale for `model.layers.{layer}.{name}.weight_scale`.
    pub fn layer_scale(&self, layer: usize, name: &str) -> Result<f32, BitNetCheckpointError> {
        let tensor_name = format!("{}{}.weight_scale", self.layer_prefix(layer), name);
        let view = self
            .tensors
            .tensor(&tensor_name)
            .map_err(|_| BitNetCheckpointError::TensorNotFound(tensor_name.clone()))?;
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: tensor_name,
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        let data = view.data();
        if data.len() < 2 {
            return Err(BitNetCheckpointError::Bf16DataTooShort {
                tensor: tensor_name,
                expected: 2,
                actual: data.len(),
            });
        }
        let bf = bf16::from_le_bytes([data[0], data[1]]);
        Ok(bf.to_f32())
    }

    /// Return the BF16 norm weight bytes converted to f32 little-endian.
    pub fn layer_norm_weight(
        &self,
        layer: usize,
        name: &str,
    ) -> Result<Vec<u8>, BitNetCheckpointError> {
        let tensor_name = format!("{}{}.weight", self.layer_prefix(layer), name);
        let view = self
            .tensors
            .tensor(&tensor_name)
            .map_err(|_| BitNetCheckpointError::TensorNotFound(tensor_name.clone()))?;
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: tensor_name,
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(bf16_bytes_to_f32_bytes(view.data()))
    }

    /// Return the embed tokens weight as f32 little-endian bytes.
    pub fn embed_tokens(&self) -> Result<Vec<u8>, BitNetCheckpointError> {
        let view = self
            .tensors
            .tensor("model.embed_tokens.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound("model.embed_tokens.weight".into())
            })?;
        let shape = view.shape();
        if shape.len() != 2 {
            return Err(BitNetCheckpointError::ShapeMismatch {
                tensor: "model.embed_tokens.weight".into(),
                expected: vec![self.vocab_size, self.hidden_dim],
                got: shape.to_vec(),
            });
        }
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: "model.embed_tokens.weight".into(),
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(bf16_bytes_to_f32_bytes(view.data()))
    }

    /// Return the final layernorm weight as f32 little-endian bytes.
    pub fn final_layernorm(&self) -> Result<Vec<u8>, BitNetCheckpointError> {
        let view = self
            .tensors
            .tensor("model.final_layernorm.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound("model.final_layernorm.weight".into())
            })?;
        let shape = view.shape();
        if last_dim(shape) != Some(&self.hidden_dim) {
            return Err(BitNetCheckpointError::ShapeMismatch {
                tensor: "model.final_layernorm.weight".into(),
                expected: vec![self.hidden_dim],
                got: shape.to_vec(),
            });
        }
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: "model.final_layernorm.weight".into(),
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(bf16_bytes_to_f32_bytes(view.data()))
    }

    /// Return the attn_sub_norm weight (inside self_attn) as f32 bytes.
    pub fn layer_attn_sub_norm(&self, layer: usize) -> Result<Vec<u8>, BitNetCheckpointError> {
        let tensor_name = format!(
            "{}{}.weight",
            self.layer_prefix(layer),
            "self_attn.attn_sub_norm"
        );
        let view = self
            .tensors
            .tensor(&tensor_name)
            .map_err(|_| BitNetCheckpointError::TensorNotFound(tensor_name.clone()))?;
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: tensor_name,
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(bf16_bytes_to_f32_bytes(view.data()))
    }

    /// Return the ffn_sub_norm weight (inside mlp) as f32 bytes.
    pub fn layer_ffn_sub_norm(&self, layer: usize) -> Result<Vec<u8>, BitNetCheckpointError> {
        let tensor_name = format!("{}{}.weight", self.layer_prefix(layer), "mlp.ffn_sub_norm");
        let view = self
            .tensors
            .tensor(&tensor_name)
            .map_err(|_| BitNetCheckpointError::TensorNotFound(tensor_name.clone()))?;
        if view.dtype() != Dtype::BF16 {
            return Err(BitNetCheckpointError::DtypeMismatch {
                tensor: tensor_name,
                expected: "BF16".into(),
                got: format!("{:?}", view.dtype()),
            });
        }
        Ok(bf16_bytes_to_f32_bytes(view.data()))
    }

    /// Metadata-only access: get the shape of a tensor by name.
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors.tensor(name).ok().map(|v| v.shape().to_vec())
    }

    fn layer_prefix(&self, layer: usize) -> String {
        format!("{MODEL_PREFIX}{layer}.")
    }
}

/// Convert BF16 little-endian bytes to f32 little-endian bytes.
fn bf16_bytes_to_f32_bytes(bf16_bytes: &[u8]) -> Vec<u8> {
    let n = bf16_bytes.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for chunk in bf16_bytes.chunks_exact(2) {
        let bf = bf16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&bf.to_f32().to_le_bytes());
    }
    out
}

/// Build a `TernaryPackedTensor` from checkpoint weight data.
pub fn make_ternary_from_checkpoint(
    codes: &[u8],
    stored_rows: usize,
    stored_cols: usize,
    scale_bf16: f32,
    group_size: usize,
) -> TernaryPackedTensor {
    let rows = stored_cols;
    let cols = stored_rows * 4;
    let groups_per_row = cols.div_ceil(group_size);
    let bytes_per_group = (group_size + 3) / 4;
    let num_groups = rows * groups_per_row;
    TernaryPackedTensor {
        rows,
        cols,
        group_size,
        groups_per_row,
        bytes_per_group,
        codes: codes.to_vec(),
        scales: vec![f16::from_f32(scale_bf16); num_groups],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_to_f32_conversion() {
        let bf16_1_0 = [0x80u8, 0x3Fu8];
        let f32_bytes = bf16_bytes_to_f32_bytes(&bf16_1_0);
        assert_eq!(f32_bytes.len(), 4);
        let f32_val = f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
        assert!((f32_val - 1.0).abs() < 1e-3);

        let bf16_m0_5 = [0x00u8, 0xBFu8];
        let f32_bytes = bf16_bytes_to_f32_bytes(&bf16_m0_5);
        let f32_val = f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
        assert!((f32_val + 0.5).abs() < 1e-3);
    }

    #[test]
    fn make_ternary_from_checkpoint_basic() {
        let stored_rows = 512usize;
        let stored_cols = 2560usize;
        let num_codes = stored_rows * stored_cols;
        let codes = vec![0u8; num_codes];
        let scale = 0.5f32;

        let tensor = make_ternary_from_checkpoint(&codes, stored_rows, stored_cols, scale, 256);

        assert_eq!(tensor.rows, stored_cols);
        assert_eq!(tensor.cols, stored_rows * 4);
        assert_eq!(tensor.group_size, 256);
        assert_eq!(tensor.groups_per_row, 8);
        assert_eq!(tensor.bytes_per_group, 64);
        assert_eq!(tensor.scales.len(), tensor.rows * 8);
        for s in &tensor.scales {
            let f: f32 = (*s).into();
            assert!((f - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn make_ternary_from_checkpoint_down_proj() {
        let stored_rows = 1728usize;
        let stored_cols = 2560usize;
        let codes = vec![0xABu8; stored_rows * stored_cols];
        let scale = 0.25f32;

        let tensor = make_ternary_from_checkpoint(&codes, stored_rows, stored_cols, scale, 256);

        assert_eq!(tensor.rows, stored_cols);
        assert_eq!(tensor.cols, stored_rows * 4);
        assert_eq!(tensor.group_size, 256);
        assert_eq!(tensor.groups_per_row, 27);
        assert_eq!(tensor.bytes_per_group, 64);
        assert_eq!(tensor.codes.len(), stored_rows * stored_cols);
    }
}
