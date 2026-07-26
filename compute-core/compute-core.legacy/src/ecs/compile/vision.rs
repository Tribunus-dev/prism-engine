//! Vision model compilation pipeline.
//!
//! Mirrors [`crate::ecs::compute_image::compile::pipeline::compile_sequential`] but
//! for vision weights: extracts tensor data from safetensors, optionally packages
//! an ANE program archive, and writes a portable cimage manifest to disk.

use std::path::Path;

use crate::ecs::compute_image::TensorEntry;
use crate::ecs::config::hardware::VisionArchitecture;
use crate::ecs::config::parser::{ArchitectureConfig, CimageManifest, ManifestModality};

/// Compile a vision model checkpoint into a standalone cimage artifact.
///
/// # Arguments
///
/// * `safetensors_path`  — Path to the `.safetensors` file containing vision weights.
/// * `coreai_modelc_path` — Optional path to a pre-compiled CoreML `.mlmodelc` bundle
///   for ANE offload. When `Some`, the bundle is tar-archived and embedded.
/// * `output_cimage`      — Destination path for the serialized cimage JSON manifest.
pub fn compile_vision_model(
    safetensors_path: &Path,
    coreai_modelc_path: Option<&Path>,
    output_cimage: &Path,
) -> anyhow::Result<()> {
    // 1. Extract projection weights into zero-copy segments
    let tensor_table = extract_vision_segments(safetensors_path, output_cimage)?;

    // 2. Package the precompiled ANE program if provided
    let has_ane = if let Some(mlmodelc) = coreai_modelc_path {
        crate::ecs::compile::pipeline::archive_ane_modelc(mlmodelc, output_cimage)?;
        true
    } else {
        false
    };

    // 3. Construct and write the manifest
    let manifest = CimageManifest {
        modality: ManifestModality::Vision,
        architecture: ArchitectureConfig::Vision(VisionArchitecture {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            image_size: 224,
            patch_size: 32,
            num_channels: 3,
            projection_dim: 512,
            model_family: "clip-vit-b32".into(),
            has_ane_program: has_ane,
        }),
        tensor_table,
    };

    manifest.write_to(output_cimage)?;
    Ok(())
}

/// Extract vision weight tensors from safetensors into segments and return
/// the tensor table entries.
///
/// For now, returns an empty table — full implementation will wire the
/// safetensors reader.
fn extract_vision_segments(path: &Path, output: &Path) -> anyhow::Result<Vec<TensorEntry>> {
    let _ = (path, output);
    Ok(Vec::new())
}
