//! SafeTensor provider — implements `TensorProvider` for safetensors model directories.
//!
//! Discovers `.safetensors` shards in a directory, indexes all tensor names
//! from their JSON header, and provides streaming access to f32 tensor data.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use prism_ecs_core::identity::{TensorInfo, TensorProvider, TensorReader};

/// Provide tensor access from a directory of `.safetensors` shard files.
///
/// Scans the directory for all `.safetensors` files and indexes every tensor
/// name across all shards. Reads and dequantizes tensor data on demand.
pub struct SafeTensorProvider {
    shards: Vec<PathBuf>,
    /// Name -> shard index in `shards`.
    tensor_index: HashMap<String, usize>,
    /// Lazily loaded shard file data, keyed by shard index.
    shard_cache: RefCell<HashMap<usize, Vec<u8>>>,
}

impl SafeTensorProvider {
    /// Discover all `.safetensors` shards in `dir` and index their tensors.
    pub fn new(dir: &Path) -> Result<Self, String> {
        let mut shards: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir).map_err(|e| format!("read dir {dir:?}: {e}"))? {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                shards.push(path);
            }
        }
        shards.sort();
        if shards.is_empty() {
            return Err(format!("No .safetensors files in {}", dir.display()));
        }

        // Build tensor name → shard index by reading each shard's JSON header.
        let mut tensor_index: HashMap<String, usize> = HashMap::new();
        for (shard_idx, shard_path) in shards.iter().enumerate() {
            let data = std::fs::read(shard_path)
                .map_err(|e| format!("read {}: {e}", shard_path.display()))?;
            // Parse the JSON header to get tensor names.
            let header_len =
                u64::from_le_bytes(data[..8].try_into().map_err(|_| "invalid header length")?);
            let header_bytes = &data[8..8 + header_len as usize];
            let header: serde_json::Value = serde_json::from_slice(header_bytes)
                .map_err(|e| format!("parse safetensors header: {e}"))?;
            let header_obj = header
                .as_object()
                .ok_or_else(|| "safetensors header is not a JSON object".to_string())?;
            for (tensor_name, _info) in header_obj {
                if tensor_name == "__metadata__" {
                    continue;
                }
                tensor_index.entry(tensor_name.clone()).or_insert(shard_idx);
            }
        }

        Ok(Self {
            shards,
            tensor_index,
            shard_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Get or load the shard data for a given shard index.
    fn load_shard(&self, shard_idx: usize) -> Result<Vec<u8>, String> {
        let mut cache = self.shard_cache.borrow_mut();
        if !cache.contains_key(&shard_idx) {
            let data = std::fs::read(&self.shards[shard_idx])
                .map_err(|e| format!("read {}: {e}", self.shards[shard_idx].display()))?;
            cache.insert(shard_idx, data);
        }
        Ok(cache[&shard_idx].clone())
    }
}

impl TensorProvider for SafeTensorProvider {
    fn list_tensors(&self) -> Result<Vec<TensorInfo>, String> {
        let mut infos = Vec::with_capacity(self.tensor_index.len());
        for (name, &shard_idx) in &self.tensor_index {
            let data = self.load_shard(shard_idx)?;
            let tensors = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| format!("deserialize: {e}"))?;
            let view = tensors
                .tensor(name)
                .map_err(|e| format!("tensor '{name}': {e}"))?;
            let shape: Vec<usize> = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            infos.push(TensorInfo {
                name: name.clone(),
                shape,
                dtype,
                size_bytes: view.data().len() as u64,
            });
        }
        Ok(infos)
    }

    fn open_tensor(&self, name: &str) -> Result<Box<dyn TensorReader>, String> {
        let &shard_idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' not found in safetensors shards"))?;
        let data = self.load_shard(shard_idx)?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| format!("deserialize shard: {e}"))?;
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("tensor '{name}': {e}"))?;

        // Convert to f32 using the shared dequantization helper in compiler.rs.
        let f32_vals = crate::compiler::tensor_to_f32(&tensors, &view, name)?;
        let raw_bytes: Vec<u8> = bytemuck::cast_slice(&f32_vals).to_vec();
        let shape: Vec<usize> = view.shape().to_vec();

        Ok(Box::new(SafeTensorReader {
            data: raw_bytes,
            shape,
            pos: 0,
        }))
    }
}

/// Reader that yields the dequantized f32 data of a single safetensors tensor.
struct SafeTensorReader {
    data: Vec<u8>,
    shape: Vec<usize>,
    pos: usize,
}

impl TensorReader for SafeTensorReader {
    fn read_chunk(&mut self, buffer: &mut [u8]) -> Result<usize, String> {
        let remaining = self.data.len() - self.pos;
        if remaining == 0 {
            return Ok(0);
        }
        let to_read = buffer.len().min(remaining);
        buffer[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
        self.pos += to_read;
        Ok(to_read)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn size_bytes(&self) -> u64 {
        self.data.len() as u64
    }
}
