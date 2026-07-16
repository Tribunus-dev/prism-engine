//! Model — a loaded `.cimage` model ready for inference.
//!
//! Reads the CImage header (zero-copy for tensor payload offsets) and
//! records per-tensor type, offset, and dimension metadata.

use prism_ecs_quantization::cimage::{CImageHeader, CImageReader};
use std::collections::HashMap;
use std::path::Path;

/// A single tensor in the loaded model.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Quantization / storage format.
    pub tensor_type: prism_ecs_quantization::cimage::TensorType,
    /// Byte offset from start of `.cimage` file (16 KB aligned).
    pub offset: u64,
    /// Payload size in bytes.
    pub size: u64,
    /// Output dimension (rows).
    pub dim_m: u32,
    /// Input dimension (columns).
    pub dim_n: u32,
}

/// Loaded `.cimage` model ready for inference dispatch.
#[derive(Debug, Clone)]
pub struct Model {
    /// Path the model was loaded from.
    pub path: std::path::PathBuf,
    /// Per-tensor metadata keyed by name (e.g. `"model.layers.0.self_attn.q_proj.weight"`).
    pub tensors: HashMap<String, TensorInfo>,
    /// Raw JSON metadata from the CImage header.
    pub metadata: serde_json::Value,
}

impl Model {
    /// Open a `.cimage` file and parse its header.
    ///
    /// Reads the magic, header size, JSON header, and populates the tensor
    /// map without loading any payload data (payloads are mmapped or read
    /// on demand by the inference engine).
    pub fn load(path: &Path) -> Result<Self, String> {
        let reader = CImageReader::open(path)?;
        let header: CImageHeader = reader.header;

        let mut tensors = HashMap::new();
        for (name, record) in &header.tensors {
            tensors.insert(
                name.clone(),
                TensorInfo {
                    tensor_type: record.tensor_type.clone(),
                    offset: record.offset,
                    size: record.size,
                    dim_m: record.dim_m,
                    dim_n: record.dim_n,
                },
            );
        }

        // Re-derive the JSON header from the reader's header for metadata.
        // The execution_plan field, if present, becomes part of metadata.
        let metadata = Self::header_to_json_value(&header);

        Ok(Model {
            path: path.to_path_buf(),
            tensors,
            metadata,
        })
    }

    /// Look up a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Number of loaded tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    // ── helpers ────────────────────────────────────────────────────────

    fn header_to_json_value(header: &CImageHeader) -> serde_json::Value {
        serde_json::to_value(header).unwrap_or(serde_json::Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_model_load_nonexistent_path() {
        let result = Model::load(Path::new("/tmp/nonexistent.cimage"));
        assert!(result.is_err(), "loading a missing file should fail");
    }
}
