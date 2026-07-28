//! Audio model compilation pipeline.
//!
//! Mirrors [`crate::ecs::compute_image::legacy_compute_image_compile::pipeline::compile_sequential`] but
//! for audio weights: extracts tensor data from safetensors, optionally packages
//! an ANE program archive, and writes a portable cimage manifest to disk.

use std::path::Path;

use crate::ecs::compute_image::manifest::TensorEntry;
use prism_ecs_constitutional::config::parser::{CimageManifest, ManifestModality};

/// Compile an audio model checkpoint into a standalone cimage artifact.
///
/// # Arguments
///
/// * `safetensors_path`  — Path to the `.safetensors` file containing audio weights.
/// * `coreai_modelc_path` — Optional path to a pre-compiled CoreML `.mlmodelc` bundle
///   for ANE offload. When `Some`, the bundle is tar-archived and embedded.
/// * `output_cimage`      — Destination path for the serialized cimage JSON manifest.
pub fn compile_audio_model(
    safetensors_path: &Path,
    coreai_modelc_path: Option<&Path>,
    output_cimage: &Path,
) -> anyhow::Result<()> {
    // 1. Extract audio weights into segments
    let tensor_table = extract_audio_segments(safetensors_path, output_cimage)?;

    // 2. Package ANE program
    let _has_ane = if let Some(mlmodelc) = coreai_modelc_path {
        crate::ecs::compile::pipeline::archive_ane_modelc(mlmodelc, output_cimage)?;
        true
    } else {
        false
    }; // Reserved: propagate into CimageManifest metadata

    // 3. Construct manifest
    // The constitutional `CimageManifest` carries its `tensor_table`
    // as `Vec<serde_json::Value>` (platform-neutral, no engine-coupling).
    // Convert the engine-internal `TensorEntry` list via the standard
    // serde pipeline; `serde_json::to_value` on a `Serialize` type
    // produces a faithful JSON view of every field.
    let tensor_table: Vec<serde_json::Value> = tensor_table
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("failed to convert tensor table to JSON: {e}"))?;

    let manifest = CimageManifest {
        modality: ManifestModality::Audio,
        architecture: prism_ecs_constitutional::config::ArchitectureConfig::Audio(
            prism_ecs_constitutional::config::hardware::AudioArchitecture {
                hidden_size: 384,
                num_attention_heads: 6,
                num_hidden_layers: 4,
                intermediate_size: 1536,
                sample_rate: 16000,
                num_mel_bins: 80,
                hop_length: 160,
                max_audio_length_s: 30,
                projection_dim: 384,
            },
        ),
        tensor_table,
    };

    manifest.write_to(output_cimage)?;
    Ok(())
}

/// Placeholder: extract audio weight tensors from safetensors into segments
/// and return the tensor table entries.
///
/// TODO: wire real safetensors reading.  For now returns an empty table.
fn extract_audio_segments(path: &Path, output: &Path) -> anyhow::Result<Vec<TensorEntry>> {
    let _ = (path, output);
    Ok(Vec::new())
}
