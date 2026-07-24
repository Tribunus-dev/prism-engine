//! SafeTensor provider — implements `TensorProvider` for safetensors model directories.
//!
//! Discovers `.safetensors` shards in a directory, indexes all tensor names
//! from their JSON header, and provides streaming access to f32 tensor data.

use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use prism_ecs_core::identity::{TensorInfo, TensorProvider, TensorReader};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TernaryPreprocessRecord {
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub changed: bool,
    pub status: String,
    pub shard_digest: String,
    pub metal_packed_file: String,
    pub packed_file: String,
    pub scales_file: String,
    pub group_size: usize,
    pub physical_tile_width: usize,
}
#[derive(Debug, Clone)]
pub struct ShardSummary {
    pub tensor_names: Vec<String>,
}

pub struct MappedTensor {
    pub map: Mmap,
    pub shape: Vec<usize>,
    pub dtype: String,
}
impl MappedTensor {
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}

/// Provide tensor access from a directory of `.safetensors` shard files.
///
/// Scans the directory for all `.safetensors` files and indexes every tensor
/// name across all shards. Reads and dequantizes tensor data on demand.
pub struct SafeTensorProvider {
    shards: Vec<PathBuf>,
    /// Name -> shard index in `shards`.
    tensor_index: HashMap<String, usize>,
    /// Lazily loaded shard file data, keyed by shard index.
    shard_cache: Mutex<HashMap<usize, Vec<u8>>>,
}

impl SafeTensorProvider {
    pub fn map_tensor(&self, name: &str) -> Result<MappedTensor, String> {
        let shard_idx = *self
            .tensor_index
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' not found"))?;
        let path = &self.shards[shard_idx];
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(|e| e.to_string())?;
        let header_len = u64::from_le_bytes(len) as usize;
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| e.to_string())?;
        let header: serde_json::Value =
            serde_json::from_slice(&header_bytes).map_err(|e| e.to_string())?;
        let info = header
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' missing header"))?;
        let shape = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "tensor shape missing".to_string())?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect::<Vec<_>>();
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("U8")
            .to_string();
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "tensor offsets missing".to_string())?;
        let start = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "tensor start missing".to_string())?;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| "tensor end missing".to_string())?;
        let absolute = 8u64
            .checked_add(header_len as u64)
            .and_then(|v| v.checked_add(start))
            .ok_or_else(|| "tensor mapping overflow".to_string())?;
        let map = unsafe {
            MmapOptions::new()
                .offset(absolute)
                .len((end - start) as usize)
                .map(&file)
                .map_err(|e| format!("map tensor '{name}': {e}"))?
        };
        Ok(MappedTensor { map, shape, dtype })
    }
    pub fn open_streaming_tensor(&self, name: &str) -> Result<Box<dyn TensorReader>, String> {
        self.open_tensor(name)
    }
    pub fn shard_summaries(&self) -> Result<Vec<ShardSummary>, String> {
        self.shards
            .iter()
            .map(|path| {
                let mut file = std::fs::File::open(path)
                    .map_err(|e| format!("open {}: {e}", path.display()))?;
                let mut len = [0u8; 8];
                file.read_exact(&mut len)
                    .map_err(|e| format!("read header length {}: {e}", path.display()))?;
                let header_len = u64::from_le_bytes(len) as usize;
                let mut header_bytes = vec![0u8; header_len];
                file.read_exact(&mut header_bytes)
                    .map_err(|e| format!("read header {}: {e}", path.display()))?;
                let header: serde_json::Value = serde_json::from_slice(&header_bytes)
                    .map_err(|e| format!("parse safetensors header {}: {e}", path.display()))?;
                let tensor_names = header
                    .as_object()
                    .ok_or_else(|| "safetensors header is not a JSON object".to_string())?
                    .keys()
                    .filter(|name| name.as_str() != "__metadata__")
                    .cloned()
                    .collect();
                Ok(ShardSummary { tensor_names })
            })
            .collect()
    }
    pub fn write_preprocess_cache(
        &self,
        _dir: &Path,
    ) -> Result<Vec<TernaryPreprocessRecord>, String> {
        Ok(self
            .list_tensors()?
            .into_iter()
            .map(|t| TernaryPreprocessRecord {
                tensor_name: t.name,
                rows: t.shape.first().copied().unwrap_or(1),
                cols: t.shape.last().copied().unwrap_or(1),
                changed: false,
                status: "source".into(),
                shard_digest: String::new(),
                metal_packed_file: String::new(),
                packed_file: String::new(),
                scales_file: String::new(),
                group_size: 0,
                physical_tile_width: 0,
            })
            .collect())
    }
    pub fn write_ternary_preprocess_cache_from_records(
        &self,
        _dir: &Path,
        records: Vec<TernaryPreprocessRecord>,
    ) -> Result<Vec<TernaryPreprocessRecord>, String> {
        Ok(records)
    }
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
            let mut file = std::fs::File::open(shard_path)
                .map_err(|e| format!("open {}: {e}", shard_path.display()))?;
            let mut len = [0u8; 8];
            file.read_exact(&mut len)
                .map_err(|e| format!("read header length: {e}"))?;
            let header_len = u64::from_le_bytes(len) as usize;
            let mut header_bytes = vec![0u8; header_len];
            file.read_exact(&mut header_bytes)
                .map_err(|e| format!("read header: {e}"))?;
            // Parse only the JSON header to get tensor names; never materialize a full shard here.
            let header: serde_json::Value = serde_json::from_slice(&header_bytes)
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
            shard_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Get or load the shard data for a given shard index.
    fn load_shard(&self, shard_idx: usize) -> Result<Vec<u8>, String> {
        let mut cache = self
            .shard_cache
            .lock()
            .map_err(|_| "shard cache lock poisoned".to_string())?;
        if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(shard_idx) {
            let data = std::fs::read(&self.shards[shard_idx])
                .map_err(|e| format!("read {}: {e}", self.shards[shard_idx].display()))?;
            e.insert(data);
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
