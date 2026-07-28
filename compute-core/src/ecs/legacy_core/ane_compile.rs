#![cfg(feature = "mlx-backend")]
//! ANE compilation — standalone Core ML ANE subgraph compilation for
//! downloaded model directories.
//!
//! Reads config.json and safetensors from `model_dir/`, builds an execution
//! plan with ANE fusion, compiles each fused island via `coremlcompiler`,
//! and returns the paths of the generated .mlmodelc bundles.

use std::collections::HashMap;
use std::path::Path;

use crate::ecs::compute_image::legacy_compute_image_compile::coreai::compile_ane_islands;
use crate::ecs::compute_image::legacy_compute_image_compile::SourceTensor;
use prism_ecs_constitutional::config::{build_execution_plan, parse_config};
use crate::ecs::config_namespace::resolve_namespace;

/// Compile Core ML ANE subgraphs for a model at the given directory.
///
/// 1. Reads `config.json` from `model_dir/` → `TextArchitecture`
/// 2. Loads every `.safetensors` file from `model_dir/weights/` → `SourceTensor`s
/// 3. Discovers the tensor namespace from weight names
/// 4. Builds a `ModelExecutionPlan` with ANE fusion
/// 5. Compiles each fused island via `coremlcompiler` into `model_dir/ane/`
/// 6. Returns the list of generated `.mlmodelc` paths
pub fn compile_ane_artifacts(model_dir: &Path) -> Result<Vec<String>, String> {
    // ── 1. Read config.json ────────────────────────────────────────────
    let config_path = model_dir.join("config.json");
    let config_str = config_path
        .to_str()
        .ok_or_else(|| format!("non-UTF-8 config path: {}", config_path.display()))?;
    let (arch, _quant_meta, _manifest) =
        parse_config(config_str).map_err(|e| format!("config parse failed: {}", e))?;

    // ── 2. Load safetensors from model_dir/weights/ ────────────────────
    let weights_dir = model_dir.join("weights");
    if !weights_dir.is_dir() {
        return Err(format!(
            "weights directory not found: {}",
            weights_dir.display()
        ));
    }

    let mut source_tensors: HashMap<String, SourceTensor> = HashMap::new();
    let mut all_tensor_names: Vec<String> = Vec::new();

    // Collect and sort safetensors shards for deterministic ordering.
    let mut safetensors_files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&weights_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "safetensors") {
                safetensors_files.push(path);
            }
        }
    }
    safetensors_files.sort();

    if safetensors_files.is_empty() {
        return Err(format!(
            "no safetensors files found in {}",
            weights_dir.display()
        ));
    }

    for shard_path in &safetensors_files {
        let buffer = std::fs::read(shard_path)
            .map_err(|e| format!("read {}: {}", shard_path.display(), e))?;

        let (_, metadata) = safetensors::SafeTensors::read_metadata(&buffer)
            .map_err(|e| format!("bad safetensors header {}: {:?}", shard_path.display(), e))?;

        let safetensors = safetensors::SafeTensors::deserialize(&buffer)
            .map_err(|e| format!("bad safetensors file {}: {:?}", shard_path.display(), e))?;

        let mut entries: Vec<_> = metadata.tensors().into_iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        for (name, info) in entries {
            if source_tensors.contains_key(&name) {
                return Err(format!("duplicate tensor name: {}", name));
            }

            let view = safetensors
                .tensor(&name)
                .map_err(|e| format!("tensor {}: {:?}", name, e))?;

            source_tensors.insert(
                name.clone(),
                SourceTensor {
                    name: name.clone(),
                    dtype: format!("{:?}", info.dtype),
                    shape: info.shape.iter().map(|&d| d as u32).collect(),
                    data: view.data().to_vec(),
                    mmap_index: None,
                    source_filename: shard_path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    source_sha256: String::new(),
                    source_offset: info.data_offsets.0 as u64,
                    source_byte_size: 0,
                },
            );
            all_tensor_names.push(name);
        }
    }

    // ── 3. Discover namespace from tensor names ───────────────────────
    let namespace = resolve_namespace(&all_tensor_names).ok_or_else(|| {
        format!(
            "could not resolve model namespace from {} tensor names",
            all_tensor_names.len()
        )
    })?;

    // ── 4. Build emitted_ids and execution plan ───────────────────────
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();
    for (i, name) in source_tensors.keys().enumerate() {
        emitted_ids.insert(name.clone(), i as u32);
    }

    let execution_plan = build_execution_plan(&arch, &namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();

    // ── 5. Create output directory ────────────────────────────────────
    let output_dir = model_dir.join("ane");
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("create ane output dir: {}", e))?;

    // ── 6. Compile ANE islands with real weights ──────────────────────
    compile_ane_islands(&plan_with_fusion, &arch, &output_dir, true)
        .map_err(|e| format!("ANE compilation failed: {}", e))?;

    // ── 7. Collect generated .mlmodelc paths ──────────────────────────
    let mut mlmodelc_paths: Vec<String> = Vec::new();
    for island in &plan_with_fusion.fused_ane_islands {
        let full_path = output_dir.join(&island.modelc_relpath);
        mlmodelc_paths.push(full_path.to_string_lossy().into_owned());
    }

    Ok(mlmodelc_paths)
}

// ── Batch scheduler ──────────────────────────────────────────────────────

/// A group of same-shaped tensors compiled into a single batch-N MIL program.
///
/// Groups are formed by identical `(in_features, out_features)` pairs.
/// The `batch_size` field (1, 2, or 4) determines how many tensors are fused.
/// `tensor_indices` identifies the original tensor positions within the stage.
#[derive(Debug, Clone)]
pub struct BatchGroup {
    /// Number of input features (columns of the weight matrix viewed as [K, N] → K).
    pub in_features: u32,
    /// Number of output features (rows of the weight matrix viewed as [K, N] → N).
    pub out_features: u32,
    /// Batch size for this group: 1, 2, or 4 (power of 2).
    pub batch_size: u32,
    /// Indices of the original tensors in this group.
    pub tensor_indices: Vec<usize>,
}

/// Schedule tensors by `(in_features, out_features)` shape into batch groups.
///
/// Groups of 4+ same-shaped tensors get batch-4 programs. Smaller groups are
/// padded to the next power of 2 (1 → 1, 2 → 2, 3 → 4). Batches larger than 4
/// are split into multiple batch-4 groups plus a remainder.
///
/// This reduces ANE program count (and thus compilation + memory overhead)
/// while respecting the ANE's static-batching constraint of power-of-2 sizes.
///
/// # Example
///
/// ```ignore
/// let tensors = vec![(3840, 18432), (3840, 18432), (3840, 18432), (3840, 18432)];
/// let groups = schedule_batches(&tensors);
/// assert_eq!(groups.len(), 1);
/// assert_eq!(groups[0].batch_size, 4);
/// ```
pub fn schedule_batches(tensors: &[(u32, u32)]) -> Vec<BatchGroup> {
    // Group tensor indices by shape key
    let mut shape_groups: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, &(in_f, out_f)) in tensors.iter().enumerate() {
        shape_groups.entry((in_f, out_f)).or_default().push(i);
    }

    let mut groups = Vec::new();

    for ((in_features, out_features), indices) in shape_groups {
        let n = indices.len();
        let mut offset = 0;

        while offset < n {
            let remaining = n - offset;
            // Choose batch size: groups >= 4 get 4, 2-3 pad to next power of 2
            let batch_size = if remaining >= 4 {
                4
            } else if remaining >= 2 {
                // remaining = 3 → pad to 4; remaining = 2 → use 2
                if remaining == 3 {
                    4
                } else {
                    2
                }
            } else {
                1
            };

            let count = remaining.min(batch_size as usize);
            let slice = indices[offset..offset + count].to_vec();

            groups.push(BatchGroup {
                in_features,
                out_features,
                batch_size,
                tensor_indices: slice,
            });

            offset += count;
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_batches_groups_four_same_shaped() {
        let tensors = vec![(3840, 18432), (3840, 18432), (3840, 18432), (3840, 18432)];
        let groups = schedule_batches(&tensors);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].batch_size, 4);
        assert_eq!(groups[0].tensor_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn schedule_batches_single_tensor() {
        let tensors = vec![(4096, 4096)];
        let groups = schedule_batches(&tensors);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].batch_size, 1);
        assert_eq!(groups[0].tensor_indices, vec![0]);
    }

    #[test]
    fn schedule_batches_five_into_two_groups() {
        let tensors = vec![
            (3840, 18432),
            (3840, 18432),
            (3840, 18432),
            (3840, 18432),
            (3840, 18432),
        ];
        let groups = schedule_batches(&tensors);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].batch_size, 4);
        assert_eq!(groups[0].tensor_indices.len(), 4);
        assert_eq!(groups[1].batch_size, 1);
        assert_eq!(groups[1].tensor_indices.len(), 1);
    }

    #[test]
    fn schedule_batches_three_pads_to_four() {
        let tensors = vec![(768, 3072), (768, 3072), (768, 3072)];
        let groups = schedule_batches(&tensors);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].batch_size, 4);
    }

    #[test]
    fn schedule_batches_different_shapes_separate() {
        let tensors = vec![(3840, 18432), (768, 3072), (3840, 18432), (768, 3072)];
        let groups = schedule_batches(&tensors);
        assert_eq!(groups.len(), 2);
        // Both groups should have batch_size 2 (two tensors each)
        assert_eq!(groups[0].batch_size, 2);
        assert_eq!(groups[1].batch_size, 2);
    }

    #[test]
    fn schedule_batches_empty_input() {
        let tensors: Vec<(u32, u32)> = vec![];
        let groups = schedule_batches(&tensors);
        assert!(groups.is_empty());
    }
}
