//! Source loading — reads safetensors shards from disk, provides lazy mmap
//! access for deferred loading and streaming compilation.

use crate::compute_image::manifest::{Manifest, ShardHash, TensorDiff};
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

// ═══════════════════════════════════════════════════════════════════════════
// Source types
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) struct SourceTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
    pub source_filename: String,
    pub source_sha256: String,
    pub source_offset: u64,
}

/// Lightweight tensor metadata used for differential-compile hashing.
#[derive(Clone, Debug)]
pub struct SourceTensorInfo {
    pub name: String,
    pub sha256: String,
    pub byte_size: u64,
}

pub(crate) struct LoadedSource {
    pub arch: crate::config::TextArchitecture,
    pub manifest: crate::config::ModelManifest,
    pub namespace: crate::config::NamespaceBinding,
    pub spec: crate::config::ExecutionSpec,
    pub source_tensors: HashMap<String, SourceTensor>,
    pub mmap_bytes: Vec<Mmap>,
    pub shard_hashes: Vec<ShardHash>,
    pub tokenizer_hashes: Vec<ShardHash>,
    pub auxiliary_hashes: Vec<ShardHash>,
    pub validation: crate::validator::ValidationReport,
}

// ═══════════════════════════════════════════════════════════════════════════
// Hash helpers
// ═══════════════════════════════════════════════════════════════════════════

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_file(path: &Path) -> crate::Result<String> {
    let bytes = std::fs::read(path)
        .map_err(|e| crate::Error::from_reason(format!("read {}: {}", path.display(), e)))?;
    Ok(sha256_bytes(&bytes))
}

fn optional_hash(path: &Path) -> crate::Result<Option<ShardHash>> {
    if !path.exists() {
        return Ok(None);
    }
    let sha256 = hash_file(path)?;
    Ok(Some(ShardHash {
        filename: path.file_name().unwrap().to_string_lossy().into_owned(),
        sha256,
    }))
}

// ═══════════════════════════════════════════════════════════════════════════
// Lazy tensor loading from mmap
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn ensure_tensor_loaded(tensor: &mut SourceTensor, mmap: &[u8]) {
    if !tensor.data.is_empty() || (tensor.source_offset == 0 && tensor.shape.is_empty()) {
        return;
    }
    let elem_bytes: usize = match tensor.dtype.as_str() {
        "BF16" | "BFloat16" | "F16" | "Float16" => 2,
        "F32" | "Float32" | "I32" | "Int32" | "U32" | "Uint32" => 4,
        "I8" | "Int8" | "U8" | "Uint8" => 1,
        other => panic!("unknown dtype {} for tensor {}", other, tensor.name),
    };
    let n: usize = tensor.shape.iter().map(|d| *d as usize).product();
    let byte_len = n * elem_bytes;
    let offset = tensor.source_offset as usize;
    let end = (offset + byte_len).min(mmap.len());
    tensor.data = if offset < mmap.len() {
        mmap[offset..end].to_vec()
    } else {
        Vec::new()
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Source tensor table (lightweight metadata scan)
// ═══════════════════════════════════════════════════════════════════════════

/// Load per-tensor metadata (sha256, byte_size) from safetensors files in
/// `source_dir`.  This is a lightweight scan that reads headers but does
/// **not** extract the full tensor payloads, making it suitable for fast
/// diff computation.
pub fn load_source_tensor_table(
    source_dir: &Path,
) -> crate::Result<HashMap<String, SourceTensorInfo>> {
    let shard_paths = crate::validator::discover_shards(source_dir)?;
    let mut table = HashMap::new();

    for shard_path in &shard_paths {
        let bytes = std::fs::read(shard_path).map_err(|e| {
            crate::Error::from_reason(format!("read {}: {}", shard_path.display(), e))
        })?;
        let sha256 = sha256_bytes(&bytes);
        let (_metadata, tensor_meta) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| {
                crate::Error::from_reason(format!(
                    "bad safetensors header {}: {:?}",
                    shard_path.display(),
                    e
                ))
            })?;

        let mut entries: Vec<_> = tensor_meta.tensors().into_iter().collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (name, info) in &entries {
            let data_offsets = info.data_offsets;
            let byte_size = data_offsets.1 - data_offsets.0;
            table.insert(
                name.clone(),
                SourceTensorInfo {
                    name: name.clone(),
                    sha256: sha256.clone(),
                    byte_size: byte_size as u64,
                },
            );
        }
    }

    Ok(table)
}

// ═══════════════════════════════════════════════════════════════════════════
// Diff computation
// ═══════════════════════════════════════════════════════════════════════════

/// Compare the current source tensors (hashes) against a previous compilation
/// manifest and return a [`TensorDiff`] describing what has changed.
///
/// A tensor is considered **unchanged** when its source-file SHA-256 matches
/// the value recorded in the previous manifest.  New tensors, changed tensors,
/// and removed tensors are reported separately.
pub fn diff_tensors(source_dir: &Path, prev_manifest: &Manifest) -> crate::Result<TensorDiff> {
    let t0 = std::time::Instant::now();
    let current = load_source_tensor_table(source_dir)?;
    let mut diff = TensorDiff::default();

    for (name, info) in &current {
        match prev_manifest.tensor_table.iter().find(|t| t.name == *name) {
            Some(prev) if prev.source_sha256 == info.sha256 => {
                diff.unchanged.push(name.clone());
            }
            Some(_) => {
                diff.changed.push(name.clone());
            }
            None => {
                diff.new.push(name.clone());
            }
        }
    }

    // Find tensors present in previous manifest but absent from current source.
    for t in &prev_manifest.tensor_table {
        if !current.contains_key(&t.name) {
            diff.removed.push(t.name.clone());
        }
    }

    diff.elapsed_ms = t0.elapsed().as_millis() as u128;
    Ok(diff)
}

// ═══════════════════════════════════════════════════════════════════════════
// Full source loading (with mmap streaming)
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn load_source(
    source_dir: &Path,
    skip_validation: bool,
) -> crate::Result<LoadedSource> {
    use crate::{config, validator};

    let config_path = source_dir.join("config.json");
    let (arch, quant, manifest) = config::parse_config(
        config_path
            .to_str()
            .ok_or_else(|| crate::Error::from_reason("invalid config path"))?,
    )?;

    let shard_paths = validator::discover_shards(source_dir)?;
    let mut source_tensors = HashMap::new();
    let mut all_names = Vec::new();
    let mut shard_hashes = Vec::new();
    let mut mmaps: Vec<Mmap> = Vec::new();

    for shard_path in shard_paths {
        // Stream via mmap instead of reading entire file into memory
        let file = std::fs::File::open(&shard_path).map_err(|e| {
            crate::Error::from_reason(format!("open {}: {}", shard_path.display(), e))
        })?;
        // SAFETY: the mmap is read-only and the file handle outlives the mmap.
        // The mmaps are stored in LoadedSource which lives as long as needed.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            crate::Error::from_reason(format!("mmap {}: {}", shard_path.display(), e))
        })?;

        let source_sha256 = sha256_bytes(&mmap);

        // Parse metadata only — don't deserialize tensor data
        let (_, metadata) =
            safetensors::SafeTensors::read_metadata(&mmap).map_err(|e| {
                crate::Error::from_reason(format!(
                    "bad safetensors header {}: {:?}",
                    shard_path.display(),
                    e
                ))
            })?;

        let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        for (name, info) in entries {
            if source_tensors.contains_key(&name) {
                return Err(crate::Error::from_reason(format!(
                    "duplicate tensor name: {}",
                    name
                )));
            }

            source_tensors.insert(
                name.clone(),
                SourceTensor {
                    name: name.clone(),
                    dtype: format!("{:?}", info.dtype),
                    shape: info.shape.iter().map(|&d| d as u32).collect(),
                    // Data is loaded lazily from mmap — start empty
                    data: Vec::new(),
                    source_filename: shard_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    source_sha256: source_sha256.clone(),
                    source_offset: info.data_offsets.0 as u64,
                },
            );
            all_names.push(name);
        }

        shard_hashes.push(ShardHash {
            filename: shard_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            sha256: source_sha256,
        });

        mmaps.push(mmap);
    }

    let tokenizer_hashes = ["tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .filter_map(|name| {
            let path = source_dir.join(name);
            match optional_hash(&path) {
                Ok(Some(hash)) => Some(Ok(hash)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;

    let auxiliary_hashes = [
        "generation_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "README.md",
    ]
    .into_iter()
    .filter_map(|name| {
        let path = source_dir.join(name);
        match optional_hash(&path) {
            Ok(Some(hash)) => Some(Ok(hash)),
            Ok(None) => None,
            Err(err) => Some(Err(err)),
        }
    })
    .collect::<crate::Result<Vec<_>>>()?;

    let namespace = config::resolve_namespace(&all_names)
        .ok_or_else(|| crate::Error::from_reason("namespace not resolved"))?;
    let mut spec = config::compile(&arch, &namespace, quant.as_ref());

    // Dynamically filter out bindings for tensors that don't exist in the
    // source model (e.g., Q/K norms that Qwen2.5 lacks).
    let all_names_set: std::collections::HashSet<String> = all_names.into_iter().collect();
    config::filter_spec_to_existing(&mut spec, &all_names_set);

    let tensor_meta = source_tensors
        .iter()
        .map(|(name, tensor)| {
            (
                name.clone(),
                crate::validator::TensorMeta {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype.clone(),
                },
            )
        })
        .collect::<HashMap<_, _>>();

    let validation = validator::validate_bindings_from_map(&tensor_meta, &spec)?;
    if !skip_validation && !validation.verdict.executable {
        eprintln!("Missing tensors (first 20):");
        for (i, t) in validation.missing_tensors.iter().take(20).enumerate() {
            eprintln!("  {}. {}", i + 1, t);
        }
        eprintln!("Unexpected tensors (first 10):");
        for (i, t) in validation.unexpected_tensors.iter().take(10).enumerate() {
            eprintln!("  {}. {} (shape={:?})", i + 1, t.name, t.shape);
        }
        eprintln!(
            "Validation report keys: missing={}, unexpected={}, bindings={}",
            validation.missing_tensors.len(),
            validation.unexpected_tensors.len(),
            validation.bindings.len()
        );
        eprintln!("Failed bindings (first 10):");
        for (i, b) in validation
            .bindings
            .iter()
            .filter(|b| !matches!(b.status, crate::validator::BindingStatus::Ok))
            .take(10)
            .enumerate()
        {
            let pack_str = b
                .packed_detail
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("none");
            let st = match &b.status {
                crate::validator::BindingStatus::Ok => "ok".into(),
                crate::validator::BindingStatus::Missing => "missing".into(),
                crate::validator::BindingStatus::ShapeMismatch => "shape".into(),
                crate::validator::BindingStatus::DtypeMismatch { expected, actual } => {
                    format!("dtype: expected={} actual={}", expected, actual)
                }
                crate::validator::BindingStatus::UnexpectedDtype => "bad_dtype".into(),
                crate::validator::BindingStatus::PackedShapeError(s) => {
                    format!("packed: {}", s)
                }
            };
            eprintln!(
                "  {}. name={} exists={} logical={:?} actual={:?} pack={} status={}",
                i + 1,
                b.tensor_name,
                b.exists,
                b.logical_shape,
                b.actual_shape,
                pack_str,
                st
            );
        }
        return Err(crate::Error::from_reason(format!(
            "source checkpoint failed validation: {} errors across {} expected tensors",
            validation.verdict.errors, validation.verdict.total_expected,
        )));
    }

    // ── Model-family adapter validation ──
    {
        if skip_validation {
            // skip adapter check in dev/skip mode
        } else {
            let model_type = arch.model_type.as_str();
            let config_val: serde_json::Value =
                match std::fs::read_to_string(source_dir.join("config.json")) {
                    Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
                    Err(_) => serde_json::Value::Null,
                };
            let tnames: Vec<String> = source_tensors.keys().cloned().collect();
            let registry = crate::model_adapter::AdapterRegistry::new();
            let adapter = registry.select(&config_val, &tnames).map_err(|e| {
                crate::Error::from_reason(format!(
                    "unsupported model type '{}': no model adapter matched.\n\
                     Supported families: qwen2, llama, mistral, gemma, phi\n\
                     Detail: {}\n\
                     Tip: if the model is supported but adapter selection failed,\n\
                     run with `--skip-validation` to bypass adapter checks.",
                    model_type, e
                ))
            })?;
            let source_model = crate::model_adapter::SourceModel {
                config: config_val,
                config_path: source_dir.join("config.json"),
                model_type: model_type.to_string(),
                tensor_names: tnames.clone(),
                tensors: source_tensors
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            // Use empty data for streaming — weights loaded on demand
                            (v.dtype.clone(), v.shape.clone(), Vec::new()),
                        )
                    })
                    .collect(),
            };
            adapter.normalize(&source_model).map_err(|report| {
                crate::Error::from_reason(format!(
                    "model normalisation failed for '{}':\n{}\n\
                     This usually means one of:\n\
                     1. The model checkpoint has a different architecture than expected\n\
                     2. The safetensors files are missing some required tensors\n\
                     3. The model family adapter needs to be updated for new variants",
                    model_type, report
                ))
            })?;
            eprintln!("[adapter] {} validation passed", adapter.family_name());
        }
    }

    Ok(LoadedSource {
        arch,
        manifest,
        namespace,
        spec,
        source_tensors,
        mmap_bytes: mmaps,
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        validation,
    })
}
