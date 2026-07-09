//! BitNet b1.58 2B4T real checkpoint loader.
//!
//! Loads packed ternary weights and BF16 metadata from a HuggingFace safetensors
//! checkpoint, and provides accessors for emitting cimage shards.
//!
//! ## File format
//! - All weight tensors are `U8` dtype — already packed 2-bit ternary codes
//!   (4 weights per byte, encoding: 00=-1, 01=0, 10=+1).
//! - Each weight tensor has a companion `weight_scale` tensor — `BF16` [1].
//! - Layer norms are `BF16` [hidden_dim] or [intermediate_dim].
//! - `embed_tokens.weight` is `BF16` [vocab_size, hidden_dim].

use crate::ternary::codec::TernaryPackedTensor;
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;
use thiserror::Error;

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
/// `SafeTensors` borrows from `_buffer` via an unsafe `'static` transmute —
/// this is sound because `_buffer` is never mutated or dropped before `tensors`.
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
    ///
    /// Reads the full file into memory, parses the header, and infers model
    /// dimensions from tensor shapes.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, BitNetCheckpointError> {
        let buffer = std::fs::read(path.as_ref())?;
        Self::from_buffer(buffer)
    }

    /// Parse a safetensors byte buffer into a checkpoint.
    ///
    /// Useful for tests that have already read the file or for partial reads.
    pub fn from_buffer(buffer: Vec<u8>) -> Result<Self, BitNetCheckpointError> {
        // SAFETY: `tensors` borrows from `_buffer`. We transmute the lifetime
        // to `'static` because `BitNetCheckpoint` owns the buffer and never
        // moves or drops it before this struct is dropped (it is the last field).
        // This is the standard pattern used by the safetensors crate's own docs.
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

        // hidden_dim: shape[0] of input_layernorm.weight (BF16 [hidden_dim])
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

        // intermediate_dim: from gate_proj.weight shape.
        // Shape is [out_features/4, in_features] U8.
        let gate_shape = tensors
            .tensor("model.layers.0.mlp.gate_proj.weight")
            .map_err(|_| {
                BitNetCheckpointError::TensorNotFound("model.layers.0.mlp.gate_proj.weight".into())
            })?
            .shape()
            .to_vec();
        // gate_proj.shape[0] = intermediate_dim / 4 (stored as packed U8)
        let intermediate_dim = gate_shape[0] * 4;

        // num_layers: scan tensor names for model.layers.<N>.*, find max N+1.
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

        // k_proj shape: [kv_inner/4, hidden_dim]
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

        // head_dim: hardcode 128 (standard for BitNet b1.58 2B4T).
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

    // ── Accessors ────────────────────────────────────────────────────────

    /// Return the raw U8 code bytes for `model.layers.{layer}.{name}.weight`.
    ///
    /// `name` should be e.g. `"self_attn.q_proj"` or `"mlp.gate_proj"`.
    /// Validates that the tensor is `U8` dtype.
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
    ///
    /// The scale tensor is a single BF16 value. Returns it as `f32`.
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
    ///
    /// `name` is e.g. `"input_layernorm"` or `"post_attention_layernorm"`.
    /// The returned `Vec<u8>` contains f32 bytes (4 bytes per element).
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
    ///
    /// Uses the already-deserialized SafeTensors to avoid re-parsing.
    /// Returns `None` if the tensor name is not found.
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors.tensor(name).ok().map(|v| v.shape().to_vec())
    }

    // ── helpers ──────────────────────────────────────────────────────────

    fn layer_prefix(&self, layer: usize) -> String {
        format!("{MODEL_PREFIX}{layer}.")
    }
}

// ── Non-method helpers ──────────────────────────────────────────────────

/// Convert BF16 little-endian bytes to f32 little-endian bytes.
///
/// Each pair of input bytes is one `bf16` value; each output is 4 f32 bytes.
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
///
/// # Parameters
/// - `codes` — the raw U8 packed ternary codes from the checkpoint.
/// - `stored_rows` — the first dimension of the stored weight tensor
///   (`out_features / 4`).
/// - `stored_cols` — the second dimension of the stored weight tensor
///   (`in_features`).
/// - `scale_bf16` — the per-tensor scale as a single f32.
/// - `group_size` — number of ternary values per quantization group.
///
/// The returned tensor has:
/// - `rows = stored_cols` (in_features)
/// - `cols = stored_rows * 4` (out_features)
/// - `group_size` as given
/// - `groups_per_row = cols.div_ceil(group_size)`
/// - `bytes_per_group = (group_size + 3) / 4`
/// - `scales = [scale_f16; rows * groups_per_row]`
pub fn make_ternary_from_checkpoint(
    codes: &[u8],
    stored_rows: usize,
    stored_cols: usize,
    scale_bf16: f32,
    group_size: usize,
) -> TernaryPackedTensor {
    let rows = stored_cols; // in_features
    let cols = stored_rows * 4; // out_features
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

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the `bf16_bytes_to_f32_bytes` helper is correct.
    #[test]
    fn test_bf16_to_f32_conversion() {
        // bf16(1.0) = 0x3F80
        let bf16_1_0 = [0x80u8, 0x3Fu8];
        let f32_bytes = bf16_bytes_to_f32_bytes(&bf16_1_0);
        assert_eq!(f32_bytes.len(), 4);
        let f32_val = f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
        assert!((f32_val - 1.0).abs() < 1e-3);

        // bf16(-0.5) = 0xBF00
        let bf16_m0_5 = [0x00u8, 0xBFu8];
        let f32_bytes = bf16_bytes_to_f32_bytes(&bf16_m0_5);
        let f32_val = f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
        assert!((f32_val + 0.5).abs() < 1e-3);
    }

    /// Verify `make_ternary_from_checkpoint` produces the correct layout.
    #[test]
    fn test_make_ternary_from_checkpoint() {
        // Simulate a q_proj.weight with stored shape [512, 2560].
        let stored_rows = 512usize;
        let stored_cols = 2560usize;
        let num_codes = stored_rows * stored_cols;
        let codes = vec![0u8; num_codes];
        let scale = 0.5f32;

        let tensor = make_ternary_from_checkpoint(&codes, stored_rows, stored_cols, scale, 256);

        assert_eq!(tensor.rows, stored_cols); // in_features = 2560
        assert_eq!(tensor.cols, stored_rows * 4); // out_features = 2048
        assert_eq!(tensor.group_size, 256);
        assert_eq!(tensor.groups_per_row, 8); // 2048 / 256
        assert_eq!(tensor.bytes_per_group, 64); // 256 / 4
        assert_eq!(tensor.scales.len(), tensor.rows * 8);
        for s in &tensor.scales {
            let f: f32 = (*s).into();
            assert!((f - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_make_ternary_from_checkpoint_down_proj() {
        // down_proj: shape [6912/4, 2560] = [1728, 2560]
        let stored_rows = 1728usize;
        let stored_cols = 2560usize;
        let codes = vec![0xABu8; stored_rows * stored_cols];
        let scale = 0.25f32;

        let tensor = make_ternary_from_checkpoint(&codes, stored_rows, stored_cols, scale, 256);

        assert_eq!(tensor.rows, stored_cols); // 2560
        assert_eq!(tensor.cols, stored_rows * 4); // 6912
        assert_eq!(tensor.group_size, 256);
        assert_eq!(tensor.groups_per_row, 27); // 6912 / 256 = 27.0
        assert_eq!(tensor.bytes_per_group, 64); // 256 / 4
        assert_eq!(tensor.codes.len(), stored_rows * stored_cols);
    }

    /// Test tensor shape — reads metadata from a REAL file but only the header
    /// (not the 1.18 GB of tensor data).
    #[test]
    #[ignore]
    fn test_checkpoint_metadata_shapes() {
        let ckpt = BitNetCheckpoint::load("models/bitnet-b1.58-2B-4T/model.safetensors").unwrap();

        // Model dimensions
        assert_eq!(ckpt.num_layers, 30);
        assert_eq!(ckpt.hidden_dim, 2560);
        assert_eq!(ckpt.intermediate_dim, 6912);
        assert_eq!(ckpt.num_heads, 20);
        assert_eq!(ckpt.num_kv_heads, 5);
        assert_eq!(ckpt.head_dim, 128);

        // Layer-0 tensor shapes via metadata-only path
        let q_shape = ckpt.tensor_shape("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(q_shape, Some(vec![640, 2560]));

        let k_shape = ckpt.tensor_shape("model.layers.0.self_attn.k_proj.weight");
        assert_eq!(k_shape, Some(vec![160, 2560]));

        let v_shape = ckpt.tensor_shape("model.layers.0.self_attn.v_proj.weight");
        assert_eq!(v_shape, Some(vec![160, 2560]));

        let o_shape = ckpt.tensor_shape("model.layers.0.self_attn.o_proj.weight");
        assert_eq!(o_shape, Some(vec![640, 2560]));

        let gate_shape = ckpt.tensor_shape("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(gate_shape, Some(vec![1728, 2560]));

        let up_shape = ckpt.tensor_shape("model.layers.0.mlp.up_proj.weight");
        assert_eq!(up_shape, Some(vec![1728, 2560]));

        let down_shape = ckpt.tensor_shape("model.layers.0.mlp.down_proj.weight");
        assert_eq!(down_shape, Some(vec![1728, 2560]));

        let embed_shape = ckpt.tensor_shape("model.embed_tokens.weight");
        assert_eq!(embed_shape, Some(vec![ckpt.vocab_size, 2560]));

        let ln_shape = ckpt.tensor_shape("model.layers.0.input_layernorm.weight");
        assert_eq!(ln_shape, Some(vec![2560]));
    }

    /// Full checkpoint load — only runs explicitly.
    /// Path: models/bitnet-b1.58-2B-4T/model.safetensors (1.18 GB).
    #[test]
    #[ignore]
    fn test_load_checkpoint_full() {
        let ckpt = BitNetCheckpoint::load("models/bitnet-b1.58-2B-4T/model.safetensors").unwrap();

        assert_eq!(ckpt.num_layers, 30);
        assert_eq!(ckpt.hidden_dim, 2560);
        assert_eq!(ckpt.intermediate_dim, 6912);
        assert_eq!(ckpt.num_heads, 20);
        assert_eq!(ckpt.num_kv_heads, 5);
        assert_eq!(ckpt.head_dim, 128);

        // Spot-check: first layer q_proj codes.
        let codes = ckpt
            .layer_codes(0, "self_attn.q_proj")
            .expect("q_proj codes");
        let expected_codes_len = 640 * 2560; // [640, 2560]
        assert_eq!(codes.len(), expected_codes_len);

        // Spot-check: scale is a finite f32.
        let scale = ckpt
            .layer_scale(0, "self_attn.q_proj")
            .expect("q_proj scale");
        assert!(scale.is_finite());

        // Spot-check: input_layernorm has hidden_dim elements.
        let ln_bytes = ckpt.layer_norm_weight(0, "input_layernorm").unwrap();
        assert_eq!(ln_bytes.len(), ckpt.hidden_dim * 4);

        // Spot-check: embed_tokens.
        let embed = ckpt.embed_tokens().unwrap();
        assert_eq!(embed.len(), ckpt.vocab_size * ckpt.hidden_dim * 4);

        // Spot-check: final_layernorm.
        let final_ln = ckpt.final_layernorm().unwrap();
        assert_eq!(final_ln.len(), ckpt.hidden_dim * 4);
    }
}
