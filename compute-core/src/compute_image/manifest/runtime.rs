//! ComputeImage runtime types — ImageRuntime, CompiledImageReader, tensor catalog.
//!
//! Contains the runtime struct definitions, segment activation helpers,
//! tensor catalog construction, and compiled image reader / verification.

use super::types::{
    compute_manifest_hash, is_valid_storage_abi, CompileReceipt, Manifest, ManifestVerification,
    QuantizationDesc, Segment, StorageBackend, TensorEntry, STORAGE_ABI_MAPPED_NO_COPY_V1,
};
use crate::mapped_image::MappedSegment;
pub(crate) use crate::quantized::QuantizedLinearBinding;
use mlx_rs::Array;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ── Tensor catalog & residency ─────────────────────────────────────────────

/// A resolved tensor binding — connects a manifest entry to its mapped segment
/// and provides the MLX array handle at runtime.
#[derive(Debug, Clone)]
pub struct ResolvedTensorBinding {
    pub tensor_id: u32,
    pub canonical_name: String,
    pub segment_id: String,
    pub offset: u64,
    pub byte_length: u64,
    pub physical_dtype: String,
    pub runtime_dtype: String,
    pub physical_shape: Vec<u32>,
    pub logical_shape: Vec<u32>,
    pub strides: Vec<u32>,
    pub quantization: Option<QuantizationDesc>,
    pub alias_of: Option<u32>,
    pub layout_version: u32,
}

/// Build a complete tensor binding catalog from a manifest.
///
/// Iterates `manifest.tensor_table` and `manifest.alias_table`, resolves aliases
/// (setting `alias_of` on the logical entry pointing to the physical tensor ID),
/// and returns a `HashMap` keyed by canonical tensor name.
///
/// Aliased entries share a single `ResolvedTensorBinding` with the alias entry
/// having `alias_of` set to the physical tensor's ID.
pub fn build_tensor_catalog(manifest: &Manifest) -> HashMap<String, ResolvedTensorBinding> {
    // First pass: build bindings from the tensor table.
    let mut catalog: HashMap<String, ResolvedTensorBinding> = HashMap::new();
    for entry in &manifest.tensor_table {
        catalog.insert(
            entry.name.clone(),
            ResolvedTensorBinding {
                tensor_id: entry.id,
                canonical_name: entry.name.clone(),
                segment_id: entry.segment.clone(),
                offset: entry.offset,
                byte_length: entry.byte_length,
                physical_dtype: entry.storage_dtype.clone(),
                runtime_dtype: entry.logical_dtype.clone(),
                physical_shape: entry.physical_shape.clone(),
                logical_shape: entry.logical_shape.clone(),
                strides: Vec::new(),
                quantization: entry.quantization.clone(),
                alias_of: None,
                layout_version: entry.layout_version,
            },
        );
    }

    // Second pass: resolve aliases.
    for alias in &manifest.alias_table {
        if let Some(phys_binding) = catalog.get(&resolve_tensor_name(
            alias.physical_tensor_id,
            &manifest.tensor_table,
        )) {
            let binding = ResolvedTensorBinding {
                tensor_id: alias.physical_tensor_id,
                canonical_name: alias.logical_name.clone(),
                segment_id: phys_binding.segment_id.clone(),
                offset: phys_binding.offset,
                byte_length: phys_binding.byte_length,
                physical_dtype: phys_binding.physical_dtype.clone(),
                runtime_dtype: phys_binding.runtime_dtype.clone(),
                physical_shape: phys_binding.physical_shape.clone(),
                logical_shape: phys_binding.logical_shape.clone(),
                strides: phys_binding.strides.clone(),
                quantization: phys_binding.quantization.clone(),
                alias_of: Some(alias.physical_tensor_id),
                layout_version: phys_binding.layout_version,
            };
            catalog.insert(alias.logical_name.clone(), binding);
        }
    }

    catalog
}

/// Helper: resolve a tensor ID to its canonical name from the tensor table.
pub fn resolve_tensor_name(id: u32, table: &[TensorEntry]) -> String {
    table
        .iter()
        .find(|entry| entry.id == id)
        .map(|entry| entry.name.clone())
        .unwrap_or_default()
}

// ── RAII layer lease ───────────────────────────────────────────────────────

/// RAII guard owning MLX array handles for a single layer segment.
/// Dropping this releases all arrays for that layer from ARRAY_REGISTRY.
/// The caller MUST call hidden.eval() before dropping to ensure the MLX
/// computation graph has consumed the weights.
pub struct LayerLease {
    pub layer_index: u32,
    pub segment_id: String,
    /// Bytes read from disk to materialise this layer.
    pub bytes_read: u64,
    pub(crate) handles: Vec<u64>,
}

impl Drop for LayerLease {
    fn drop(&mut self) {
        for h in &self.handles {
            let _ = crate::bridge::free_array(*h);
        }
    }
}

// ── Image Runtime ──────────────────────────────────────────────────────────

/// Image Runtime — holds opened manifest and handles for persistently
/// loaded tensors.
#[derive(Clone, Serialize, Deserialize)]
pub struct ImageRuntime {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
    pub backend: StorageBackend,
    /// Path to the image directory for on-demand segment reads.
    #[serde(skip)]
    pub(crate) image_dir: PathBuf,
    /// Handles for persistent tensors (embeddings, final norm). Always resident.
    #[serde(skip)]
    pub(crate) persistent_handles: HashMap<String, u64>,
    /// Quantized binding descriptors built from persistent tensors.
    #[serde(skip)]
    pub(crate) quantized_bindings: HashMap<String, QuantizedLinearBinding>,
    /// Monotonically accumulated bytes loaded across all activate_layer calls.
    #[serde(skip)]
    pub(crate) total_bytes_activated: u64,
    #[serde(skip)]
    pub(crate) released: bool,
}

// ── Compiled image reader ──────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct CompiledImageReader {
    pub manifest: Manifest,
    pub receipt: CompileReceipt,
    /// Path to the image directory; segment files are read on demand.
    #[serde(skip)]
    image_dir: PathBuf,
}

impl CompiledImageReader {
    pub fn open(image_dir: &Path) -> crate::Result<Self> {
        let manifest_path = image_dir.join("manifest.json");
        let receipt_path = image_dir.join("receipt.json");
        let manifest: Manifest =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| {
                crate::Error::from_reason(format!(
                    "read manifest {}: {}",
                    manifest_path.display(),
                    e
                ))
            })?)
            .map_err(|e| crate::Error::from_reason(format!("parse manifest: {}", e)))?;
        let receipt: CompileReceipt =
            match serde_json::from_str(&std::fs::read_to_string(&receipt_path).unwrap_or_default())
            {
                Ok(r) => r,
                Err(_) => CompileReceipt::default(),
            };

        let reader = Self {
            manifest,
            receipt,
            image_dir: image_dir.to_path_buf(),
        };
        reader.verify()?;
        Ok(reader)
    }

    /// Read a segment file from disk and return its bytes.
    fn read_segment_bytes(&self, filename: &str) -> crate::Result<Vec<u8>> {
        let path = self.image_dir.join(filename);
        std::fs::read(&path).map_err(|e| {
            crate::Error::from_reason(format!("read segment {}: {}", path.display(), e))
        })
    }

    pub fn verify(&self) -> crate::Result<ManifestVerification> {
        let skip = std::env::var("TRIBUNUS_SKIP_MANIFEST_HASH").is_ok();
        let manifest_hash_matches =
            self.manifest.image_hash == compute_manifest_hash(&self.manifest) || skip;
        let receipt_matches_manifest = self.receipt.complete_image_hash == self.manifest.image_hash
            && self.receipt.segment_hashes.len() == self.manifest.segments.len()
            && self
                .receipt
                .segment_hashes
                .iter()
                .zip(self.manifest.segments.iter())
                .all(|(receipt, segment)| {
                    receipt.id == segment.id
                        && receipt.filename == segment.filename
                        && receipt.sha256 == segment.sha256
                        && receipt.byte_size == segment.byte_size
                });

        let mut segment_hashes_match = true;
        let mut verified_segment_count = 0usize;
        let mut total_bytes = 0u64;

        for segment in &self.manifest.segments {
            let bytes = self.read_segment_bytes(&segment.filename).map_err(|e| {
                crate::Error::from_reason(format!("segment hash mismatch check - {}", e))
            })?;
            let actual_hash = sha256_bytes(&bytes);
            if actual_hash != segment.sha256 {
                segment_hashes_match = false;
            } else {
                verified_segment_count += 1;
            }
            total_bytes += bytes.len() as u64;
        }

        if self.receipt.complete_image_hash != self.manifest.image_hash {
            segment_hashes_match = false;
        }
        if !receipt_matches_manifest {
            segment_hashes_match = false;
        }

        if !manifest_hash_matches {
            return Err(crate::Error::from_reason(
                "compiled image manifest hash mismatch",
            ));
        }
        if !receipt_matches_manifest {
            return Err(crate::Error::from_reason(
                "compiled image receipt does not match manifest",
            ));
        }
        if !segment_hashes_match {
            return Err(crate::Error::from_reason(
                "compiled image segment hash mismatch",
            ));
        }
        // ── mapped-no-copy-v1 additional checks ──────────────────────
        if self.manifest.required_storage_abi == STORAGE_ABI_MAPPED_NO_COPY_V1 {
            for segment in &self.manifest.segments {
                let seg_path = self.image_dir.join(&segment.filename);
                if !seg_path.exists() {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment file does not exist: {}",
                        seg_path.display()
                    )));
                }
                let meta = seg_path.metadata().map_err(|e| {
                    crate::Error::from_reason(format!(
                        "mapped-no-copy: stat {}: {}",
                        seg_path.display(),
                        e
                    ))
                })?;
                let actual_len = meta.len();
                if actual_len != segment.byte_size {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment {} size mismatch: manifest says {} but file is {}",
                        segment.filename, segment.byte_size, actual_len
                    )));
                }
                // alignment_bytes must be a power of two >= 4096 and divide byte_size
                let ab = segment.alignment_bytes;
                if ab < 4096 || ab & (ab.wrapping_sub(1)) != 0 {
                    return Err(crate::Error::from_reason(format!(
                    "mapped-no-copy: segment {} alignment_bytes {} is not a power of two >= 4096",
                    segment.filename, ab
                )));
                }
                if segment.byte_size % ab != 0 {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: segment {} byte_size {} is not aligned to {}",
                        segment.filename, segment.byte_size, segment.alignment_bytes
                    )));
                }
            }
            let seg_map: HashMap<&str, &Segment> = self
                .manifest
                .segments
                .iter()
                .map(|s| (s.id.as_str(), s))
                .collect();
            for tensor in &self.manifest.tensor_table {
                let tab = if tensor.tensor_alignment_bytes != 0 {
                    tensor.tensor_alignment_bytes
                } else {
                    16u64
                };
                if tab == 0 || tensor.offset % tab != 0 {
                    return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: tensor {} offset {} not aligned to {}",
                        tensor.name, tensor.offset, tab
                    )));
                }
                if let Some(seg) = seg_map.get(tensor.segment.as_str()) {
                    let tensor_end = tensor.offset.saturating_add(tensor.byte_length);
                    if tensor_end > seg.byte_size {
                        return Err(crate::Error::from_reason(format!(
                        "mapped-no-copy: tensor {} offset {} + byte_length {} exceeds segment {} byte_size {}",
                        tensor.name, tensor.offset, tensor.byte_length, seg.id, seg.byte_size
                    )));
                    }
                }
            }
        } else if !is_valid_storage_abi(&self.manifest.required_storage_abi) {
            return Err(crate::Error::from_reason(format!(
                "unknown storage ABI: {}",
                self.manifest.required_storage_abi
            )));
        }

        Ok(ManifestVerification {
            manifest_hash_matches,
            segment_hashes_match,
            verified_segment_count,
            total_bytes,
        })
    }

    /// Read a single tensor's bytes from its segment file on disk.
    /// Used by fixture-test TensorLookup; not called during segment-scoped execution.
    pub fn tensor_bytes(&self, name: &str) -> crate::Result<(Vec<u8>, String, Vec<u32>)> {
        let entry = self
            .manifest
            .tensor_table
            .iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("tensor not found in manifest: {}", name))
            })?;

        let segment = self
            .manifest
            .segments
            .iter()
            .find(|segment| segment.id == entry.segment)
            .ok_or_else(|| {
                crate::Error::from_reason(format!("segment not found for tensor: {}", name))
            })?;

        let payload = self.read_segment_bytes(&segment.filename)?;

        let start = entry.offset as usize;
        let end = start + entry.byte_length as usize;
        if end > payload.len() {
            return Err(crate::Error::from_reason(format!(
                "tensor {} exceeds segment bounds",
                name
            )));
        }

        Ok((
            payload[start..end].to_vec(),
            entry.storage_dtype.clone(),
            entry.physical_shape.clone(),
        ))
    }

    pub fn open_runtime(&self, backend: StorageBackend) -> crate::Result<ImageRuntime> {
        if backend == StorageBackend::MappedNoCopy {
            // 1. Map all segment files via MappedSegment
            let segment_map: HashMap<String, Arc<MappedSegment>> = self
                .manifest
                .segments
                .iter()
                .map(|seg| {
                    let seg_path = self.image_dir.join(&seg.filename);
                    let mapped = MappedSegment::new(&seg_path, None).map_err(|e| {
                        crate::Error::from_reason(format!("mmap segment {}: {}", seg.filename, e))
                    })?;
                    Ok((seg.id.clone(), mapped))
                })
                .collect::<crate::Result<_>>()?;

            // 2. Build tensor catalog
            let catalog = build_tensor_catalog(&self.manifest);

            // 3. Populate persistent handles
            let mut persistent_handles: HashMap<String, u64> = HashMap::new();
            for (name, binding) in &catalog {
                if binding.segment_id == "persistent"
                    || binding.segment_id.starts_with("persistent_")
                {
                    if let Some(mapped) = segment_map.get(&binding.segment_id) {
                        if let Some(entry) =
                            self.manifest.tensor_table.iter().find(|e| e.name == *name)
                        {
                            let array =
                                crate::memory::compute_image_bridge::load_mlx_tensor(mapped, entry)
                                    .map_err(|e| {
                                        crate::Error::from_reason(format!(
                                            "load persistent tensor {}: {}",
                                            name, e
                                        ))
                                    })?;
                            let handle =
                                crate::bridge::ARRAY_REGISTRY.write().insert(array, None);
                            persistent_handles.insert(name.clone(), handle);
                        }
                    }
                }
            }

            // 4. Build and return the runtime
            let mut runtime = ImageRuntime {
                manifest: self.manifest.clone(),
                receipt: self.receipt.clone(),
                backend,
                image_dir: self.image_dir.clone(),
                persistent_handles,
                quantized_bindings: HashMap::new(),
                total_bytes_activated: 0,
                released: false,
            };
            runtime.rebuild_quantized_bindings_from_persistent()?;
            return Ok(runtime);
        }

        if !super::memory_override_enabled() {
            let total_memory = super::system_memory_bytes();
            let estimated_peak = super::estimate_open_runtime_peak_bytes(&self.manifest);
            if total_memory > 0
                && estimated_peak > total_memory.saturating_sub(2 * 1024 * 1024 * 1024)
            {
                return Err(crate::Error::from_reason(format!(
                    "refusing to open runtime: estimated peak {} exceeds safe budget on this machine (total memory {})",
                    estimated_peak,
                    total_memory,
                )));
            }
        }

        let _ = super::clear_mlx_cache();
        let _ = super::set_mlx_cache_limit(512 * 1024 * 1024);

        let mut runtime = ImageRuntime {
            manifest: self.manifest.clone(),
            receipt: self.receipt.clone(),
            backend,
            image_dir: self.image_dir.clone(),
            persistent_handles: HashMap::new(),
            quantized_bindings: HashMap::new(),
            total_bytes_activated: 0,
            released: false,
        };

        runtime.activate_persistent()?;
        Ok(runtime)
    }
}

impl crate::model::TensorLookup for CompiledImageReader {
    fn tensor(&self, name: &str) -> Option<Array> {
        let (bytes, dtype, shape) = self.tensor_bytes(name).ok()?;
        dtype_to_array(&bytes, &dtype, &shape).ok()
    }
}

// ── Utility functions ──────────────────────────────────────────────────────

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[allow(dead_code)]
fn compute_struct_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("struct hash serialization");
    sha256_bytes(&bytes)
}

fn dtype_to_array(bytes: &[u8], dtype: &str, shape: &[u32]) -> crate::Result<Array> {
    let dims = shape.iter().map(|&dim| dim as i32).collect::<Vec<_>>();
    match dtype {
        "U8" | "Uint8" => Ok(Array::from_slice(bytes, &dims)),
        "U32" | "Uint32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "u32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I8" | "Int8" => {
            let data = bytes.iter().map(|&byte| byte as i8).collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "I32" | "Int32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "i32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "F32" | "Float32" => {
            if bytes.len() % 4 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "f32 payload length is not a multiple of 4: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        "BF16" | "BFloat16" => {
            if bytes.len() % 2 != 0 {
                return Err(crate::Error::from_reason(format!(
                    "bf16 payload length is not a multiple of 2: {}",
                    bytes.len()
                )));
            }
            let data = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    f32::from_bits((bf as u32) << 16)
                })
                .collect::<Vec<_>>();
            Ok(Array::from_slice(&data, &dims))
        }
        other => Err(crate::Error::from_reason(format!(
            "unsupported tensor storage dtype: {}",
            other
        ))),
    }
}

/// Returns the process resident set size in bytes, or 0 if unavailable.
#[allow(dead_code)]
fn process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        #[allow(dead_code)]
        extern "C" {
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut u32,
                task_info_count: *mut u32,
            ) -> i32;
            fn mach_task_self() -> u32;
        }
        const TASK_BASIC_INFO: u32 = 5;
        const TASK_BASIC_INFO_COUNT: u32 = 10;
        let mut info = [0u32; 10];
        let mut count = TASK_BASIC_INFO_COUNT;
        let ret = unsafe {
            task_info(
                mach_task_self(),
                TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            )
        };
        if ret == 0 && count >= 2 {
            let lo = info[1] as u64;
            let hi = info[2] as u64;
            return (hi << 32) | lo;
        }
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Ok(kb) = rest.trim().trim_end_matches(" kB").parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
        0
    }
}
