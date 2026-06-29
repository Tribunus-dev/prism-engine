//! Tensor emission helpers — writes source tensors into the ImageBuilder
//! segment pipeline, handles quantized weight triplets, and builds the
//! source identity / manifest hash for deterministic compilation.

use crate::compute_image::manifest::{
    Manifest, QuantizationDesc, SegmentKind, ShardHash, SourceIdentity,
};
use crate::compute_image::compile::source::SourceTensor;
use crate::config::PackedLinearShapes;
use mlx_rs::Array;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Tensor emission
// ═══════════════════════════════════════════════════════════════════════════

fn emit_tensor(
    builder: &mut crate::compute_image::manifest::ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    name: &str,
    role: String,
    layer: Option<u32>,
    logical_dtype: String,
    logical_shape: Vec<u32>,
    quantization: Option<QuantizationDesc>,
) -> crate::Result<u32> {
    let tensor = source_tensors
        .get(name)
        .ok_or_else(|| crate::Error::from_reason(format!("missing tensor: {}", name)))?;

    Ok(builder.add_tensor(
        name.to_string(),
        role,
        layer,
        &tensor.data,
        tensor.source_filename.clone(),
        tensor.source_sha256.clone(),
        tensor.source_offset,
        logical_dtype,
        &tensor.dtype,
        logical_shape,
        tensor.shape.clone(),
        quantization,
    ))
}

fn emit_quantized_binding(
    builder: &mut crate::compute_image::manifest::ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    weight_name: &str,
    role: String,
    layer: Option<u32>,
    logical_shape: Vec<u32>,
    packed: &PackedLinearShapes,
    logical_dtype: String,
) -> crate::Result<u32> {
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let scales_id = emit_tensor(
        builder,
        source_tensors,
        &scales_name,
        format!("{}::scales", role),
        layer,
        "F32".into(),
        packed.scales.clone(),
        None,
    )?;
    let biases_id = emit_tensor(
        builder,
        source_tensors,
        &biases_name,
        format!("{}::biases", role),
        layer,
        "F32".into(),
        packed.biases.clone(),
        None,
    )?;

    emit_tensor(
        builder,
        source_tensors,
        weight_name,
        role,
        layer,
        logical_dtype,
        logical_shape,
        Some(QuantizationDesc {
            bits: packed.bits,
            group_size: packed.group_size,
            groups: packed.groups,
            scale_tensor_id: scales_id,
            bias_tensor_id: biases_id,
        }),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Source identity
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn build_source_identity(
    manifest: &crate::config::ModelManifest,
    shard_hashes: Vec<ShardHash>,
    tokenizer_hashes: Vec<ShardHash>,
    auxiliary_hashes: Vec<ShardHash>,
) -> SourceIdentity {
    SourceIdentity {
        config_hash: manifest.config_hash.clone(),
        shard_hashes,
        tokenizer_hashes,
        auxiliary_hashes,
        model_type: manifest.model_type.clone(),
        quantization_bits: manifest.quantization_bits.unwrap_or(8),
        quantization_group_size: manifest.quantization_group_size.unwrap_or(64),
        quantization_mode: manifest
            .quantization_mode
            .clone()
            .unwrap_or_else(|| "affine".into()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Vision / audio encoder compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile vision encoder tensors from source into a dedicated segment.
pub(crate) fn compile_vision_encoder_tensors(
    builder: &mut crate::compute_image::manifest::ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
) -> crate::Result<()> {
    let mut vision_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| k.starts_with("vision_encoder."))
        .collect();
    vision_names.sort();

    if vision_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("vision_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("vision_encoder", SegmentKind::Persistent);

    for name in &vision_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!("vision tensor {} disappeared from source", name))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "VisionEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

/// Compile audio encoder tensors from source into a dedicated segment.
pub(crate) fn compile_audio_encoder_tensors(
    builder: &mut crate::compute_image::manifest::ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    emitted_ids: &mut HashMap<String, u32>,
    audio_config: Option<crate::config::AudioArchitecture>,
) -> crate::Result<()> {
    let mut audio_names: Vec<&String> = source_tensors
        .keys()
        .filter(|k| k.starts_with("audio_encoder.") || k.starts_with("embed_audio."))
        .collect();
    audio_names.sort();

    if audio_names.is_empty() {
        return Ok(());
    }

    if emitted_ids.keys().any(|k| k.starts_with("audio_encoder.")) {
        return Ok(());
    }

    builder.begin_segment("audio_encoder", SegmentKind::Persistent);
    if let Some(config) = audio_config {
        builder.set_audio_config(config);
    }

    for name in &audio_names {
        let tensor = source_tensors.get(*name).ok_or_else(|| {
            crate::Error::from_reason(format!("audio tensor {} disappeared from source", name))
        })?;

        let logical_shape: Vec<u32> = tensor.shape.iter().map(|&d| d as u32).collect();

        let id = emit_tensor(
            builder,
            source_tensors,
            name,
            "AudioEncoder".into(),
            None,
            tensor.dtype.clone(),
            logical_shape,
            None,
        )?;
        emitted_ids.insert((*name).clone(), id);
    }

    Ok(())
}

/// Emit a single tensor binding — either direct or quantized (with scales/biases).
pub(crate) fn emit_binding_set(
    builder: &mut crate::compute_image::manifest::ImageBuilder,
    source_tensors: &HashMap<String, SourceTensor>,
    binding: &crate::config::TensorBinding,
    layer: Option<u32>,
) -> crate::Result<u32> {
    let role = format!("{:?}", binding.role);
    match &binding.packed_shape {
        Some(packed) => emit_quantized_binding(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            binding.logical_shape.clone(),
            packed,
            "F32".into(),
        ),
        None => emit_tensor(
            builder,
            source_tensors,
            &binding.name,
            role,
            layer,
            "F32".into(),
            binding.logical_shape.clone(),
            None,
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Manifest hash computation
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic manifest fingerprint.  We hash only the semantic fields
/// (ignoring compiler timestamps and other transient metadata) so that two
/// compilations of identical inputs produce the same hash.
pub(crate) fn compute_manifest_hash(manifest: &Manifest) -> String {
    #[derive(Serialize)]
    struct Fingerprint<'a> {
        image_version: &'a str,
        compiler_version: &'a str,
        runtime_abi: &'a str,
        source: &'a SourceIdentity,
        architecture: &'a crate::config::TextArchitecture,
        segments: &'a [crate::compute_image::manifest::Segment],
        tensor_table: &'a [crate::compute_image::manifest::TensorEntry],
        alias_table: &'a [crate::compute_image::manifest::AliasEntry],
        residency_plan: &'a crate::compute_image::manifest::ResidencyPlan,
    }

    let fingerprint = Fingerprint {
        image_version: &manifest.image_version,
        compiler_version: &manifest.compiler_version,
        runtime_abi: &manifest.runtime_abi,
        source: &manifest.source,
        architecture: &manifest.architecture,
        segments: &manifest.segments,
        tensor_table: &manifest.tensor_table,
        alias_table: &manifest.alias_table,
        residency_plan: &manifest.residency_plan,
    };

    let bytes = serde_json::to_vec(&fingerprint).expect("manifest fingerprint serialization");
    sha256_bytes(&bytes)
}

#[allow(dead_code)]
fn compute_struct_hash<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("struct hash serialization");
    sha256_bytes(&bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

// ═══════════════════════════════════════════════════════════════════════════
// MLX dtype → array conversion
// ═══════════════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) fn dtype_to_array(bytes: &[u8], dtype: &str, shape: &[u32]) -> crate::Result<Array> {
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
            // Convert BF16 to F32 for MLX compute compatibility
            let data = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16 to F32: shift left 16, reinterpret as f32
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
