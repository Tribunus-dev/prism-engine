//! Gemma 4 checkpoint inspector — reusable validation logic extracted
//! from the `gemma4_inspect` binary. The deployment compiler calls this
//! to build the tensor inventory before proceeding to quantization.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::gemma4_unified::{classify_tensor_name, Gemma4UnifiedSchema, TensorClassification};

/// Threshold for unknown tensor parameter count that triggers a hard error.
const UNKNOWN_PARAM_THRESHOLD: u64 = 1_000_000;

/// Result of inspecting a Gemma 4 checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gemma4Inspection {
    /// Raw configuration parsed from config.json.
    pub config: ModelConfig,
    /// Schema-derived from the configuration.
    pub schema: SerializedSchema,
    /// Tensor inventory — every tensor classified.
    pub inventory: TensorInventory,
    /// Processor contract — tokenizer, image, and audio metadata.
    pub processor_contract: serde_json::Value,
    /// Source identity — SHA-256 of config, tokenizer, and shard index.
    pub source_identity: SourceIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocab_size: u32,
    pub intermediate_size: u32,
    /// Multi-Token Prediction (MTP) depth, if configured.
    pub mtp_depth: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSchema {
    pub hidden_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub vocab_size: u32,
    pub intermediate_size: u32,
    pub head_dim: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventory {
    pub total_tensors: usize,
    pub classification: HashMap<String, TensorClassSummary>,
    pub tensors: Vec<TensorEntry>,
    pub unknown_large: Vec<TensorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorClassSummary {
    pub count: usize,
    pub total_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub classification: String,
    pub param_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub config_digest: String,
    pub tokenizer_digest: String,
    pub tensor_index_digest: String,
}

/// Stream safetensors shards one at a time, yielding raw bytes and
/// releasing each shard before the next is loaded.
pub struct SourceShardStream {
    shard_paths: Vec<PathBuf>,
    current_index: usize,
    current_data: Option<Vec<u8>>,
}

impl SourceShardStream {
    /// Advance to the next shard, loading it into `current_data`.
    /// Returns `Ok(Some(&[u8]))` with the shard bytes, `Ok(None)` when
    /// all shards have been consumed, or `Err(String)` on I/O failure.
    pub fn next_shard(&mut self) -> Result<Option<&[u8]>, String> {
        if self.current_index >= self.shard_paths.len() {
            self.current_data = None;
            return Ok(None);
        }
        let path = &self.shard_paths[self.current_index];
        let data = fs::read(path)
            .map_err(|e| format!("cannot read safetensors shard {}: {e}", path.display()))?;
        self.current_index += 1;
        self.current_data = Some(data);
        Ok(Some(self.current_data.as_ref().unwrap().as_slice()))
    }

    /// Number of shards discovered.
    pub fn shard_count(&self) -> usize {
        self.shard_paths.len()
    }

    /// Index of the shard that will be loaded next (0-based).
    pub fn current_index(&self) -> usize {
        self.current_index
    }
}

/// Stream tensor data from safetensors shards without loading everything
/// into memory at once. Discovers shards from a Gemma 4 model directory
/// by checking `model.safetensors.index.json` (multi-shard) or falling
/// back to `model.safetensors` (single file).
pub fn stream_source_shards(dir: &Path) -> Result<SourceShardStream, String> {
    if !dir.is_dir() {
        return Err(format!("model directory not found: {}", dir.display()));
    }

    // Try multi-shard index first
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value = serde_json::from_reader(BufReader::new(
            File::open(&index_path).map_err(|e| format!("cannot open safetensors index: {e}"))?,
        ))
        .map_err(|e| format!("invalid safetensors index: {e}"))?;

        // Collect unique shard filenames from the weight_map
        let mut shard_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
            for (_name, file) in weight_map {
                if let Some(fname) = file.as_str() {
                    shard_set.insert(fname.to_string());
                }
            }
        }

        let shard_paths: Vec<PathBuf> = shard_set.into_iter().map(|f| dir.join(&f)).collect();

        if shard_paths.is_empty() {
            return Err("safetensors index found but weight_map is empty or missing".into());
        }

        return Ok(SourceShardStream {
            shard_paths,
            current_index: 0,
            current_data: None,
        });
    }

    // Fall back to single shard
    let single_path = dir.join("model.safetensors");
    if single_path.exists() {
        return Ok(SourceShardStream {
            shard_paths: vec![single_path],
            current_index: 0,
            current_data: None,
        });
    }

    Err("no safetensors index or model.safetensors found".into())
}

/// Build a JSON receipt summarising the inspection for ingestion tracking.
pub fn build_ingestion_receipt(inspection: &Gemma4Inspection) -> serde_json::Value {
    serde_json::json!({
        "config_digest": inspection.source_identity.config_digest,
        "tokenizer_digest": inspection.source_identity.tokenizer_digest,
        "tensor_index_digest": inspection.source_identity.tensor_index_digest,
        "total_tensors": inspection.inventory.total_tensors,
        "total_params": inspection.inventory.classification.values()
            .map(|s| s.total_params).sum::<u64>(),
        "classification": inspection.inventory.classification,
        "mtp_detected": inspection.config.mtp_depth.unwrap_or(0) > 0,
        "mtp_depth": inspection.config.mtp_depth,
    })
}

/// Inspect a Gemma 4 checkpoint directory and return a structured inspection.
pub fn inspect_gemma4_checkpoint(dir: &Path) -> Result<Gemma4Inspection, String> {
    if !dir.is_dir() {
        return Err(format!("model directory not found: {}", dir.display()));
    }

    // ── Read config.json ──────────────────────────────────────────────
    let config_path = dir.join("config.json");
    let config_value: serde_json::Value = serde_json::from_reader(BufReader::new(
        fs::File::open(&config_path).map_err(|e| format!("cannot read config.json: {e}"))?,
    ))
    .map_err(|e| format!("invalid config.json: {e}"))?;

    // Gemma 4 unified format nests text config under "text_config" key.
    // Check top-level first, then fall back to text_config sub-object.
    let tc = config_value.get("text_config").and_then(|v| v.as_object());
    let val = |key: &str| -> u32 {
        config_value[key]
            .as_u64()
            .or_else(|| tc.and_then(|t| t[key].as_u64()))
            .unwrap_or(0) as u32
    };
    let hidden_size = val("hidden_size");
    let num_layers = val("num_hidden_layers");
    let num_heads = val("num_attention_heads");
    let num_kv_heads = val("num_key_value_heads");
    let vocab_size = val("vocab_size");
    let intermediate_size = val("intermediate_size");
    let mtp_depth = config_value["mtp_depth"]
        .as_u64()
        .or_else(|| tc.and_then(|t| t.get("mtp_depth").and_then(|v| v.as_u64())))
        .map(|d| d as u32);

    let config = ModelConfig {
        hidden_size,
        num_layers,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        vocab_size,
        intermediate_size,
        mtp_depth,
    };

    // ── Build schema and validate ─────────────────────────────────────
    let mut schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    schema.hidden_size = hidden_size;
    schema.num_layers = num_layers;
    schema.num_attention_heads = num_heads;
    schema.num_key_value_heads = num_kv_heads;
    schema.vocabulary_size = vocab_size;

    schema
        .validate_architecture()
        .map_err(|e| format!("architecture validation: {e}"))?;

    // ── Read Safetensors index ────────────────────────────────────────
    let st_paths = [
        dir.join("model.safetensors.index.json"),
        dir.join("model.safetensors"),
    ];

    let mut tensor_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for st_path in &st_paths {
        if !st_path.exists() {
            continue;
        }
        if st_path.extension().map_or(false, |e| e == "json") {
            let index: serde_json::Value = serde_json::from_reader(BufReader::new(
                fs::File::open(st_path)
                    .map_err(|e| format!("cannot open safetensors index: {e}"))?,
            ))
            .map_err(|e| format!("invalid safetensors index: {e}"))?;

            if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                for (name, _file) in weight_map {
                    tensor_shapes.entry(name.clone()).or_insert(Vec::new());
                }
            }
        } else {
            let file =
                fs::File::open(st_path).map_err(|e| format!("cannot open safetensors: {e}"))?;
            let mut reader = BufReader::new(file);
            let mut header_len_bytes = [0u8; 8];
            reader
                .read_exact(&mut header_len_bytes)
                .map_err(|e| format!("cannot read safetensors header: {e}"))?;
            let header_len = u64::from_le_bytes(header_len_bytes) as usize;
            let mut header_json = vec![0u8; header_len];
            reader
                .read_exact(&mut header_json)
                .map_err(|e| format!("cannot read safetensors header json: {e}"))?;
            let header: serde_json::Value =
                serde_json::from_slice(&header_json).map_err(|e| format!("invalid header: {e}"))?;
            if let Some(tensors) = header.as_object() {
                for (name, info) in tensors {
                    if let Some(shape) = info.get("shape").and_then(|s| s.as_array()) {
                        let dims: Vec<usize> = shape
                            .iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect();
                        tensor_shapes.insert(name.clone(), dims);
                    }
                }
            }
        }
        break;
    }

    // ── Classify tensors ──────────────────────────────────────────────
    let mut classification: HashMap<String, (usize, u64)> = HashMap::new();
    let mut tensor_list: Vec<TensorEntry> = Vec::new();
    let mut unknown_large: Vec<TensorEntry> = Vec::new();

    let names: Vec<String> = tensor_shapes.keys().cloned().collect();
    schema
        .reject_legacy_vision_tower(&names)
        .map_err(|e| format!("legacy vision tower rejection: {e}"))?;

    for (name, shape) in &tensor_shapes {
        let cls = classify_tensor_name(name);
        let param_count: u64 = shape.iter().map(|&d| d as u64).product();

        let entry = classification
            .entry(cls.as_str().to_string())
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += param_count;

        let tensor_entry = TensorEntry {
            name: name.clone(),
            shape: shape.clone(),
            classification: cls.as_str().to_string(),
            param_count,
        };
        tensor_list.push(tensor_entry.clone());

        if cls == TensorClassification::Unknown && param_count > UNKNOWN_PARAM_THRESHOLD {
            unknown_large.push(tensor_entry);
        }
    }

    if !unknown_large.is_empty() {
        let mut msg = format!(
            "{} unknown tensor(s) exceed {} parameter threshold:",
            unknown_large.len(),
            UNKNOWN_PARAM_THRESHOLD
        );
        for t in &unknown_large {
            msg.push_str(&format!("\n  {} ({:?})", t.name, t.shape));
        }
        return Err(msg);
    }

    let inventory = TensorInventory {
        total_tensors: tensor_shapes.len(),
        classification: classification
            .into_iter()
            .map(|(k, (count, total_params))| {
                (
                    k,
                    TensorClassSummary {
                        count,
                        total_params,
                    },
                )
            })
            .collect(),
        tensors: tensor_list,
        unknown_large,
    };

    // ── Build processor contract ──────────────────────────────────────
    let tokenizer_path = dir.join("tokenizer_config.json");
    let tokenizer_value: serde_json::Value = if tokenizer_path.exists() {
        serde_json::from_reader(BufReader::new(
            fs::File::open(&tokenizer_path).map_err(|e| format!("cannot read tokenizer: {e}"))?,
        ))
        .unwrap_or_default()
    } else {
        serde_json::Value::Null
    };

    let processor_path = dir.join("processor_config.json");
    let processor_value: serde_json::Value = if processor_path.exists() {
        serde_json::from_reader(BufReader::new(
            fs::File::open(&processor_path).map_err(|e| format!("cannot read processor: {e}"))?,
        ))
        .unwrap_or_default()
    } else {
        serde_json::Value::Null
    };

    let processor_contract = serde_json::json!({
        "text": {
            "vocabulary_size": vocab_size,
            "bos_token_id": tokenizer_value.get("bos_token_id"),
            "eos_token_id": tokenizer_value.get("eos_token_id"),
            "pad_token_id": tokenizer_value.get("pad_token_id"),
        },
        "image": {
            "patch_size": processor_value.get("patch_size").or(processor_value.get("image_patch_size")),
            "soft_token_default": processor_value.get("soft_tokens").or(processor_value.get("default_soft_tokens")),
        },
        "audio": {
            "sample_rate": processor_value.get("sampling_rate"),
        },
    });

    // ── Source identity ───────────────────────────────────────────────
    let source_identity = SourceIdentity {
        config_digest: sha256_file(&config_path).unwrap_or_default(),
        tokenizer_digest: if tokenizer_path.exists() {
            sha256_file(&tokenizer_path).unwrap_or_default()
        } else {
            String::new()
        },
        tensor_index_digest: if st_paths[0].exists() {
            sha256_file(&st_paths[0]).unwrap_or_default()
        } else if st_paths[1].exists() {
            sha256_file(&st_paths[1]).unwrap_or_default()
        } else {
            String::new()
        },
    };

    Ok(Gemma4Inspection {
        config,
        schema: SerializedSchema {
            hidden_size: schema.hidden_size,
            num_layers: schema.num_layers,
            num_attention_heads: schema.num_attention_heads,
            num_key_value_heads: schema.num_key_value_heads,
            vocab_size: schema.vocabulary_size,
            intermediate_size,
            head_dim: if num_heads > 0 {
                hidden_size / num_heads
            } else {
                0
            },
        },
        inventory,
        processor_contract,
        source_identity,
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::Digest;
    let data = fs::read(path).map_err(|e| format!("cannot read {path:?}: {e}"))?;
    let mut hasher = sha2::Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}
