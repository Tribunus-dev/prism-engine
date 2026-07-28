//! Top-level orchestration: AOT plan → mmap → segments → header.

use super::archive::archive_mlmodelc_to_mmap;
use super::builder::AlignedMmapBuilder;
use super::layout::{predict_tar_size, CImageLayoutPlan, CImageTopologyTable};
use crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::{
    AttentionKind as GraphAttentionKind, CompactionEpoch, DeviceCapability, DraftSubGraph,
    ExecutionGraphDescriptor, LayerExecutionNode, NodeKind,
};
use crate::ecs::compute_image::legacy_compute_image_compile::source::LoadedSource;
use crate::ecs::compute_image::legacy_compute_image_compile::source::{source_tensor_byte_len, source_tensor_view};
use crate::ecs::compute_image::legacy_compute_image_compile::ternary::model_artifact_tag;
use crate::ecs::compute_image::legacy_compute_image_compile::ternary::{
    CimageHeader, LayerDirectoryEntry, ModelArtifactEntry, SegmentEntry, SegmentKind,
    CIMAGE_SEGMENT_CAPACITY,
};
use crate::ecs::compute_image::legacy_compute_image_compile::ternary::{
    QUANT_SCHEMA_NF4_TILE640, QUANT_SCHEMA_TERNARY_TILE640,
};
use crate::ecs::compute_image::legacy_compute_image_compile::tts_compile::pack_tts_weights;
use crate::ecs::compute_image::manifest::Manifest;
use crate::ecs::compute_image::manifest::SharedWeightLayout;
use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::descriptor::{
    MultimodalInputDescriptorV1, ProjectionRole, ProjectionTensorRecord,
    MULTIMODAL_DESCRIPTOR_MAGIC,
};
use prism_ecs_constitutional::config::CompileQuantMode;
use memmap2::MmapMut;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

fn is_draft_tensor_name(name: &str) -> bool {
    name.contains("draft") || name.contains("mtp")
}

fn ordered_weight_binding_names(loaded: &LoadedSource, include_draft: bool) -> Vec<String> {
    let mut names = Vec::new();
    for binding in &loaded.spec.global_tensors {
        if binding.name.ends_with(".weight") && is_draft_tensor_name(&binding.name) == include_draft
        {
            names.push(binding.name.clone());
        }
    }
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if binding.name.ends_with(".weight")
                && is_draft_tensor_name(&binding.name) == include_draft
            {
                names.push(binding.name.clone());
            }
        }
    }
    names
}

fn quantized_triplet_lengths(loaded: &LoadedSource, include_draft: bool) -> (u64, u64, u64) {
    let mut weight_len = 0u64;
    let mut scale_len = 0u64;
    let mut bias_len = 0u64;
    for name in ordered_weight_binding_names(loaded, include_draft) {
        let stem = name.strip_suffix(".weight").unwrap_or(&name);
        if let Some(t) = loaded.source_tensors.get(&name) {
            weight_len += source_tensor_byte_len(t) as u64;
        }
        if let Some(t) = loaded.source_tensors.get(&format!("{}.scales", stem)) {
            scale_len += source_tensor_byte_len(t) as u64;
        }
        if let Some(t) = loaded.source_tensors.get(&format!("{}.biases", stem)) {
            bias_len += source_tensor_byte_len(t) as u64;
        }
    }
    (weight_len, scale_len, bias_len)
}

#[allow(dead_code)]
fn copy_quantized_triplets_into_slices(
    loaded: &LoadedSource,
    include_draft: bool,
    weights_dst: &mut [u8],
    scales_dst: &mut [u8],
    biases_dst: &mut [u8],
) {
    let mut weights_off = 0usize;
    let mut scales_off = 0usize;
    let mut biases_off = 0usize;
    for name in ordered_weight_binding_names(loaded, include_draft) {
        let stem = name.strip_suffix(".weight").unwrap_or(&name);
        if let Some(t) = loaded.source_tensors.get(&name) {
            let end = weights_off + t.data.len();
            weights_dst[weights_off..end].copy_from_slice(&t.data);
            weights_off = end;
        }
        if let Some(t) = loaded.source_tensors.get(&format!("{}.scales", stem)) {
            let end = scales_off + t.data.len();
            scales_dst[scales_off..end].copy_from_slice(&t.data);
            scales_off = end;
        }
        if let Some(t) = loaded.source_tensors.get(&format!("{}.biases", stem)) {
            let end = biases_off + t.data.len();
            biases_dst[biases_off..end].copy_from_slice(&t.data);
            biases_off = end;
        }
    }
}

#[allow(dead_code)]
fn collect_quantized_triplet_bytes(
    loaded: &LoadedSource,
    include_draft: bool,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (weight_len, scale_len, bias_len) = quantized_triplet_lengths(loaded, include_draft);
    let mut weights = vec![0u8; weight_len as usize];
    let mut scales = vec![0u8; scale_len as usize];
    let mut biases = vec![0u8; bias_len as usize];
    copy_quantized_triplets_into_slices(
        loaded,
        include_draft,
        &mut weights,
        &mut scales,
        &mut biases,
    );
    (weights, scales, biases)
}

fn layer_directory_entries_for_loaded(
    loaded: &LoadedSource,
    include_draft: bool,
) -> Vec<LayerDirectoryEntry> {
    let mut entries = Vec::new();
    let mut weights_offset = 0u64;
    let mut scales_offset = 0u64;

    for layer in &loaded.spec.layers {
        let layer_is_draft = layer
            .tensors
            .iter()
            .any(|binding| is_draft_tensor_name(&binding.name));
        if layer_is_draft != include_draft {
            continue;
        }

        let mut weights_length = 0u64;
        let mut scales_length = 0u64;
        for binding in &layer.tensors {
            if !binding.name.ends_with(".weight") {
                continue;
            }
            if is_draft_tensor_name(&binding.name) != include_draft {
                continue;
            }
            let stem = binding
                .name
                .strip_suffix(".weight")
                .unwrap_or(&binding.name);
            weights_length += loaded
                .source_tensors
                .get(&binding.name)
                .map(|tensor| source_tensor_byte_len(tensor) as u64)
                .unwrap_or(0);
            scales_length += loaded
                .source_tensors
                .get(&format!("{}.scales", stem))
                .map(|tensor| source_tensor_byte_len(tensor) as u64)
                .unwrap_or(0);
        }

        entries.push(LayerDirectoryEntry {
            weights_offset,
            weights_length,
            scales_offset,
            scales_length,
            layer_kind: 0,
            flags: 0,
        });

        weights_offset += weights_length;
        scales_offset += scales_length;
    }

    entries
}

fn vocabulary_triplet_lengths(loaded: &LoadedSource, embed_key: &str) -> u64 {
    if embed_key.is_empty() {
        return 0;
    }
    let stem = embed_key.strip_suffix(".weight").unwrap_or(embed_key);
    let weight = loaded
        .source_tensors
        .get(embed_key)
        .map(|t| source_tensor_byte_len(t) as u64)
        .unwrap_or(0);
    let scales = loaded
        .source_tensors
        .get(&format!("{}.scales", stem))
        .map(|t| source_tensor_byte_len(t) as u64)
        .unwrap_or(0);
    let biases = loaded
        .source_tensors
        .get(&format!("{}.biases", stem))
        .map(|t| source_tensor_byte_len(t) as u64)
        .unwrap_or(0);
    weight + scales + biases
}

fn draft_projection_triplet_lengths(loaded: &LoadedSource) -> (u64, u64, u64, u64, u64, u64) {
    fn tensor_len(loaded: &LoadedSource, name: &str) -> u64 {
        loaded
            .source_tensors
            .get(name)
            .map(|tensor| source_tensor_byte_len(tensor) as u64)
            .unwrap_or(0)
    }

    let pre_names = ["pre_projection.weight", "model.mtp_projection.weight"];
    let post_names = ["post_projection.weight", "model.mtp_post_projection.weight"];

    let pre = pre_names
        .iter()
        .find(|name| loaded.source_tensors.contains_key(**name))
        .copied();
    let post = post_names
        .iter()
        .find(|name| loaded.source_tensors.contains_key(**name))
        .copied();

    let pre_weight = pre.map(|name| tensor_len(loaded, name)).unwrap_or(0);
    let pre_scales = pre
        .map(|name| {
            tensor_len(
                loaded,
                &format!("{}.scales", name.strip_suffix(".weight").unwrap_or(name)),
            )
        })
        .unwrap_or(0);
    let pre_biases = pre
        .map(|name| {
            tensor_len(
                loaded,
                &format!("{}.biases", name.strip_suffix(".weight").unwrap_or(name)),
            )
        })
        .unwrap_or(0);

    let post_weight = post.map(|name| tensor_len(loaded, name)).unwrap_or(0);
    let post_scales = post
        .map(|name| {
            tensor_len(
                loaded,
                &format!("{}.scales", name.strip_suffix(".weight").unwrap_or(name)),
            )
        })
        .unwrap_or(0);
    let post_biases = post
        .map(|name| {
            tensor_len(
                loaded,
                &format!("{}.biases", name.strip_suffix(".weight").unwrap_or(name)),
            )
        })
        .unwrap_or(0);

    (
        pre_weight,
        pre_scales,
        pre_biases,
        post_weight,
        post_scales,
        post_biases,
    )
}

fn synthesize_execution_graph_for_loaded(loaded: &LoadedSource) -> Option<Vec<u8>> {
    let text = &loaded.arch;
    let mut layers = Vec::new();

    if let Some(vision) = &loaded.manifest.vision_config {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::VisionPatchEmbed as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: vision.patch_size.min(u16::MAX as u32) as u16,
            num_heads: vision.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: vision.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::VisionFinalProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    if let Some(audio) = &loaded.manifest.audio_config {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::AudioFrameEmbed as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: audio.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: audio.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::AudioProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    if loaded.manifest.vision_config.is_some() || loaded.manifest.audio_config.is_some() {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::EmbeddingAssembly as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    let main_entries = layer_directory_entries_for_loaded(loaded, false);
    let draft_entries = layer_directory_entries_for_loaded(loaded, true);
    let (pre_weight, pre_scales, _pre_biases, post_weight, post_scales, _post_biases) =
        draft_projection_triplet_lengths(loaded);

    for (plan, entry) in loaded
        .spec
        .layers
        .iter()
        .filter(|layer| {
            !layer
                .tensors
                .iter()
                .any(|binding| is_draft_tensor_name(&binding.name))
        })
        .zip(main_entries.iter())
    {
        let is_sliding = matches!(
            plan.attention_kind,
            prism_ecs_constitutional::config::AttentionKind::SlidingAttention
        );
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DecoderLayer as u8,
            attention_kind: if is_sliding {
                GraphAttentionKind::SlidingWindow as u8
            } else {
                GraphAttentionKind::FullAttention as u8
            },
            device_capability: DeviceCapability::Both as u8,
            compaction_epoch: 0xFF,
            layer_index: plan.index,
            head_dim: plan.head_dim.min(u16::MAX as u32) as u16,
            num_heads: plan.n_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: entry.weights_offset,
            weight_length: entry.weights_length,
            scale_offset: entry.scales_offset,
            _reserved: [0u8; 8],
        });
    }

    let mut draft_sub_graph = None;
    if !draft_entries.is_empty() {
        let draft_hidden = text.hidden_size;
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DraftPreProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: pre_weight,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DraftPostProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: draft_hidden,
            weight_offset: pre_weight,
            weight_length: post_weight,
            scale_offset: pre_scales,
            _reserved: [0u8; 8],
        });
        for (layer_index, entry) in draft_entries.iter().enumerate() {
            layers.push(LayerExecutionNode {
                node_kind: NodeKind::DraftLayer as u8,
                attention_kind: GraphAttentionKind::FullAttention as u8,
                device_capability: DeviceCapability::Both as u8,
                compaction_epoch: 0xFF,
                layer_index: layer_index as u32,
                head_dim: text.head_dim.min(u16::MAX as u32) as u16,
                num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
                hidden_dim: draft_hidden,
                weight_offset: entry.weights_offset,
                weight_length: entry.weights_length,
                scale_offset: entry.scales_offset,
                _reserved: [0u8; 8],
            });
        }

        let draft_weight_offset = draft_entries.first().map(|e| e.weights_offset).unwrap_or(0);
        let draft_scale_offset = draft_entries.first().map(|e| e.scales_offset).unwrap_or(0);
        let draft_weight_length: u64 = draft_entries.iter().map(|e| e.weights_length).sum();
        let draft_scale_length: u64 = draft_entries.iter().map(|e| e.scales_length).sum();
        draft_sub_graph = Some(DraftSubGraph {
            num_layers: draft_entries.len().min(u32::MAX as usize) as u32,
            hidden_dim: draft_hidden,
            weight_offset: draft_weight_offset,
            weight_length: draft_weight_length,
            scale_offset: draft_scale_offset,
            scale_length: draft_scale_length,
            pre_proj_offset: 0,
            post_proj_offset: pre_weight,
        });
        let _ = post_scales;
    }

    if layers.is_empty() {
        return None;
    }

    Some(
        ExecutionGraphDescriptor {
            magic: crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers: main_entries.len().min(u16::MAX as usize) as u16,
            num_draft_layers: draft_entries.len().min(u16::MAX as usize) as u16,
            num_compaction_epochs: 0,
            node_count: layers.len().min(u32::MAX as usize) as u32,
            _pad: [0u8; 2],
            layers,
            compaction_epochs: Vec::new(),
            draft_sub_graph,
        }
        .to_bytes(),
    )
}

#[allow(dead_code)]
fn copy_vocabulary_triplet(loaded: &LoadedSource, embed_key: &str, dst: &mut [u8]) {
    if embed_key.is_empty() {
        return;
    }
    let stem = embed_key.strip_suffix(".weight").unwrap_or(embed_key);
    let mut off = 0usize;
    for key in [
        embed_key.to_string(),
        format!("{}.scales", stem),
        format!("{}.biases", stem),
    ] {
        if let Some(t) = loaded.source_tensors.get(&key) {
            let end = off + t.data.len();
            dst[off..end].copy_from_slice(&t.data);
            off = end;
        }
    }
}

/// Compile and pack the unified Gemma4_Unified.cimage.
///
/// 1. predict_tar_size scans the resulting .mlmodelc directories.
/// 2. CImageLayoutPlan::calculate() computes all offsets AOT.
/// 3. File is ftruncate'd + mmap'd at the exact total size.
/// 4. Metal lib + .mlmodelc are copied into the mmap.
/// 5. GPU writes quantized weights directly into the mmap via stream_weights_to_mmap_gpu.
/// 6. CImageHeader is written at offset 0.
///
/// GPU-accelerated TernaryTile640 quantization: streams weight tensors
/// from the loaded source into the cimage mmap, computing per-tensor
/// offsets within the weights segment and passing them to the Metal kernel
/// for direct-to-mmap write via `newBufferWithBytesNoCopy`.
#[cfg(feature = "metal-dispatch")]
pub(crate) fn stream_weights_to_mmap_gpu(
    loaded: &mut LoadedSource,
    plan: &CImageLayoutPlan,
    builder: &mut AlignedMmapBuilder,
    qmode: CompileQuantMode,
) -> crate::Result<()> {
    let mmap_base = builder.mmap_base();
    match qmode {
        CompileQuantMode::TernaryTile640 { .. } => {
            let total = stream_ternary_segment_to_mmap_gpu(
                loaded,
                &ordered_weight_binding_names(loaded, false),
                mmap_base,
                plan.main_weights.offset,
            )?;
            eprintln!(
                "[cimage] GPU ternary tile640: {} weights streamed into mmap at offset {:#X}, {} bytes total",
                if total > 0 { "all" } else { "no" },
                plan.main_weights.offset,
                total,
            );
            Ok(())
        }
        CompileQuantMode::Nf4Tile640 { .. } => {
            let main_total = stream_nf4_segment_to_mmap_gpu(
                loaded,
                &ordered_weight_binding_names(loaded, false),
                mmap_base,
                plan.main_weights.offset,
                plan.main_scales.offset,
                plan.main_biases.offset,
            )?;
            let mtp_total = stream_nf4_segment_to_mmap_gpu(
                loaded,
                &ordered_weight_binding_names(loaded, true),
                mmap_base,
                plan.mtp_weights.offset,
                plan.mtp_scales.offset,
                plan.mtp_biases.offset,
            )?;
            eprintln!(
                "[cimage] GPU nf4 tile640: main={}B mtp={}B streamed into resident triplet arenas",
                main_total, mtp_total,
            );
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(feature = "metal-dispatch")]
fn stream_ternary_segment_to_mmap_gpu(
    loaded: &mut LoadedSource,
    weight_names: &[String],
    mmap_base: *mut u8,
    segment_file_offset: u64,
) -> crate::Result<u64> {
    use crate::ecs::compute_image::legacy_compute_image_compile::try_ternary_tile640_pack_gpu;

    let mut tensor_cursor = 0u64;
    for binding_name in weight_names {
        let (out_dim, in_dim) = {
            let Some(entry) = loaded.source_tensors.get_mut(binding_name) else {
                continue;
            };
            for mmap in &loaded.mmap_bytes {
                crate::ecs::compute_image::legacy_compute_image_compile::source::ensure_tensor_loaded(entry, mmap);
                if !entry.data.is_empty() {
                    break;
                }
            }
            if entry.data.len() < 2 || (entry.dtype != "F16" && entry.dtype != "BF16") {
                continue;
            }
            if entry.shape.len() != 2 {
                continue;
            }
            (entry.shape[0], entry.shape[1])
        };

        let data = loaded
            .source_tensors
            .get_mut(binding_name)
            .map(|t| std::mem::take(&mut t.data))
            .unwrap_or_default();
        if data.is_empty() {
            continue;
        }

        let num_tiles = (in_dim as u64 + 639) / 640;
        let tensor_file_offset = segment_file_offset + tensor_cursor;
        try_ternary_tile640_pack_gpu(
            loaded,
            binding_name,
            &data,
            out_dim,
            in_dim,
            Some((mmap_base, tensor_file_offset)),
        )?;
        tensor_cursor += (out_dim as u64) * num_tiles * 32 * 4;
    }
    Ok(tensor_cursor)
}

#[cfg(feature = "metal-dispatch")]
fn stream_nf4_segment_to_mmap_gpu(
    loaded: &mut LoadedSource,
    weight_names: &[String],
    mmap_base: *mut u8,
    weights_segment_offset: u64,
    scales_segment_offset: u64,
    biases_segment_offset: u64,
) -> crate::Result<u64> {
    use crate::ecs::compute_image::legacy_compute_image_compile::{
        nf4_tile640_pack_layout, try_nf4_tile640_pack_gpu_to_output, Nf4Tile640MmapOutput,
    };

    let mut weights_cursor = 0u64;
    let mut scales_cursor = 0u64;
    let mut biases_cursor = 0u64;

    for binding_name in weight_names {
        let (out_dim, in_dim, dtype) = {
            let Some(entry) = loaded.source_tensors.get_mut(binding_name) else {
                continue;
            };
            for mmap in &loaded.mmap_bytes {
                crate::ecs::compute_image::legacy_compute_image_compile::source::ensure_tensor_loaded(entry, mmap);
                if !entry.data.is_empty() {
                    break;
                }
            }
            if entry.data.len() < 2 || (entry.dtype != "F16" && entry.dtype != "BF16") {
                continue;
            }
            if entry.shape.len() != 2 {
                continue;
            }
            (entry.shape[0], entry.shape[1], entry.dtype.clone())
        };

        let data = loaded
            .source_tensors
            .get_mut(binding_name)
            .map(|t| std::mem::take(&mut t.data))
            .unwrap_or_default();
        if data.is_empty() {
            continue;
        }

        let layout = nf4_tile640_pack_layout(out_dim, in_dim);
        let output = Nf4Tile640MmapOutput {
            mmap_base,
            weights_offset: weights_segment_offset + weights_cursor,
            scales_offset: scales_segment_offset + scales_cursor,
            biases_offset: biases_segment_offset + biases_cursor,
        };

        try_nf4_tile640_pack_gpu_to_output(
            loaded,
            binding_name,
            &data,
            &dtype,
            out_dim,
            in_dim,
            Some(output),
        )?;

        weights_cursor += layout.total_packed_bytes as u64;
        scales_cursor += layout.scales_len as u64;
        biases_cursor += layout.biases_len as u64;
    }

    Ok(weights_cursor + scales_cursor + biases_cursor)
}

pub(crate) fn compile_and_pack_god_binary(
    output_path: &str,
    metallib_path: &Path,
    main_mlmodelc_path: &Path,
    mtp_mlmodelc_path: &Path,
    main_weight_total_elements: u64,
    mtp_weight_total_elements: u64,
    loaded: &mut LoadedSource,
    qmode: CompileQuantMode,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_heads: u32,
    head_dim: u32,
) -> std::io::Result<()> {
    const SEG_MTP_GRAPH: usize = 3;
    const SEG_MTP_WEIGHTS: usize = 4;
    const SEG_MAIN_GRAPH: usize = 5;
    const SEG_MAIN_WEIGHTS: usize = 6;
    const SEG_LAYER_DIRECTORY: usize = 7;
    const SEG_MAIN_SCALES: usize = 8;
    const SEG_MAIN_BIASES: usize = 9;
    const SEG_MTP_SCALES: usize = 10;
    const SEG_MTP_BIASES: usize = 11;
    const SEG_EXECUTION_GRAPH: usize = 12;
    const SEG_MM_PROJ_WEIGHTS: usize = 13;
    const SEG_MM_PROJ_SCALES: usize = 14;
    const SEG_MM_DESCRIPTOR: usize = 15;
    const SEG_MM_POSITION: usize = 16;
    const SEG_MM_AUX: usize = 17;
    const SEG_MM_PROJ_BIASES: usize = 18;

    if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
        crate::ecs::compute_image::legacy_compute_image_compile::apply_quantize_to_loaded(loaded, qmode)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    let mut multimodal = synthesize_multimodal_segments_for_loaded(loaded)?;
    let mut execution_graph_bytes =
        synthesize_execution_graph_for_loaded(loaded).unwrap_or_default();

    let main_graph_len = predict_tar_size(main_mlmodelc_path)?;
    let mtp_graph_len = predict_tar_size(mtp_mlmodelc_path)?;
    let metal_lib_len = std::fs::metadata(metallib_path)?.len();
    let header_size = std::mem::size_of::<CimageHeader>() as u64;

    // ── Locate embed_tokens.weight for the vocabulary segment ──────────────
    // Try common HF key prefixes (Gemma4 / Llama / Qwen families).
    let embed_key = [
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ]
    .iter()
    .find(|&&k| loaded.source_tensors.contains_key(k))
    .copied()
    .unwrap_or("");

    let (
        main_weights_len,
        main_scales_len,
        main_biases_len,
        mtp_weights_len,
        mtp_scales_len,
        mtp_biases_len,
        vocab_len,
    ) = if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
        let (mw, ms, mb) = quantized_triplet_lengths(loaded, false);
        let (dw, ds, db) = quantized_triplet_lengths(loaded, true);
        let vocab_len = vocabulary_triplet_lengths(loaded, embed_key);
        (mw, ms, mb, dw, ds, db, vocab_len)
    } else {
        // TernaryTile640: 640 weights → 32 u32 lanes × 4 bytes = 128 bytes per tile.
        let mw = (main_weight_total_elements / 640) * 128;
        let dw = (mtp_weight_total_elements / 640) * 128;
        let vocab_weight_elements: u64 = if embed_key.is_empty() {
            eprintln!(
                "[cimage] ⚠️  embed_tokens.weight not found — Vocabulary segment will be empty"
            );
            0
        } else {
            let st = &loaded.source_tensors[embed_key];
            if st.shape.len() == 2 {
                st.shape[0] as u64 * st.shape[1] as u64
            } else {
                (st.source_byte_size / 2).max(st.data.len() as u64 / 2)
            }
        };
        let vocab_num_tiles = (vocab_weight_elements + 639) / 640;
        let vocab_packed_len = vocab_num_tiles * 32 * 4;
        let vocab_scales_len = vocab_num_tiles * 4;
        (mw, 0, 0, dw, 0, 0, vocab_packed_len + vocab_scales_len)
    };

    let plan = CImageLayoutPlan::calculate(
        header_size,
        metal_lib_len,
        main_graph_len,
        main_weights_len,
        main_scales_len,
        main_biases_len,
        mtp_graph_len,
        mtp_weights_len,
        mtp_scales_len,
        mtp_biases_len,
        vocab_len,
        num_layers,
        execution_graph_bytes.len() as u64,
        multimodal
            .as_ref()
            .map(|segments| segments.projection_weights.len() as u64),
        multimodal
            .as_ref()
            .map(|segments| segments.projection_scales.len() as u64),
        multimodal
            .as_ref()
            .map(|segments| segments.projection_biases.len() as u64),
        multimodal
            .as_ref()
            .map(|segments| segments.descriptor.len() as u64),
        multimodal
            .as_ref()
            .map(|segments| segments.position_embeddings.len() as u64),
        multimodal
            .as_ref()
            .map(|segments| segments.auxiliary_weights.len() as u64),
        qmode,
    );

    if let Some(multimodal) = &mut multimodal {
        if multimodal.descriptor.len() >= std::mem::size_of::<MultimodalInputDescriptorV1>() {
            let desc = unsafe {
                &mut *(multimodal.descriptor.as_mut_ptr() as *mut MultimodalInputDescriptorV1)
            };
            desc.projection_weight_segment_index = SEG_MM_PROJ_WEIGHTS as u16;
            desc.projection_scale_segment_index = if plan
                .multimodal_projection_scales
                .map(|segment| segment.length > 0)
                .unwrap_or(false)
            {
                SEG_MM_PROJ_SCALES as u16
            } else {
                u16::MAX
            };
            desc.projection_bias_segment_index = if plan
                .multimodal_projection_biases
                .map(|segment| segment.length > 0)
                .unwrap_or(false)
            {
                SEG_MM_PROJ_BIASES as u16
            } else {
                u16::MAX
            };
            desc.position_embedding_segment_index = if plan
                .multimodal_position_embeddings
                .map(|segment| segment.length > 0)
                .unwrap_or(false)
            {
                SEG_MM_POSITION as u16
            } else {
                u16::MAX
            };
            desc.auxiliary_weight_segment_index = if plan
                .multimodal_auxiliary_weights
                .map(|segment| segment.length > 0)
                .unwrap_or(false)
            {
                SEG_MM_AUX as u16
            } else {
                u16::MAX
            };
        }
        let _ = patch_execution_graph_multimodal_nodes(
            execution_graph_bytes.as_mut_slice(),
            &multimodal.descriptor,
        );
    }

    let topology_table = CImageTopologyTable::compute(
        hidden_size,
        intermediate_size,
        num_layers,
        num_heads,
        head_dim,
    );

    eprintln!(
        "[cimage] AOT layout: total={} metal_lib={} main_graph={} main_weights={} mtp_graph={} mtp_weights={} vocabulary={}",
        plan.total_file_size,
        plan.metal_lib.length, plan.main_graph.length,
        plan.main_weights.length, plan.mtp_graph.length, plan.mtp_weights.length,
        plan.vocabulary.length,
    );

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)?;
    file.set_len(plan.total_file_size)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    unsafe {
        std::ptr::write_bytes(mmap.as_mut_ptr(), 0u8, mmap.len());
    }
    let mut builder = AlignedMmapBuilder::new(mmap);

    // Segment: Metal megakernel
    let metallib_data = std::fs::read(metallib_path)?;
    builder.align_cursor();
    builder
        .allocate_slice(metallib_data.len())
        .copy_from_slice(&metallib_data);

    // Segment: Main .mlmodelc
    builder.align_cursor();
    let main_slice = builder.allocate_slice(plan.main_graph.length as usize);
    let written = archive_mlmodelc_to_mmap(main_mlmodelc_path, main_slice)?;
    eprintln!("[cimage] main .mlmodelc: {} bytes archived", written);

    // Segment: Main weights
    builder.align_cursor();
    if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
        let _main_weights_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.main_weights.length as usize) };
        builder.align_cursor();
        let _main_scales_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.main_scales.length as usize) };
        builder.align_cursor();
        let _main_biases_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.main_biases.length as usize) };
    } else {
        let _main_weights_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.main_weights.length as usize) };
    }

    // Segment: MTP .mlmodelc
    builder.align_cursor();
    let mtp_slice = builder.allocate_slice(plan.mtp_graph.length as usize);
    let written = archive_mlmodelc_to_mmap(mtp_mlmodelc_path, mtp_slice)?;
    eprintln!("[cimage] MTP .mlmodelc: {} bytes archived", written);

    // Segment: MTP weights
    builder.align_cursor();
    if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
        let _mtp_weights_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.mtp_weights.length as usize) };
        builder.align_cursor();
        let _mtp_scales_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.mtp_scales.length as usize) };
        builder.align_cursor();
        let _mtp_biases_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.mtp_biases.length as usize) };
    } else {
        let _mtp_weights_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.mtp_weights.length as usize) };
    }

    #[cfg(feature = "metal-dispatch")]
    {
        stream_weights_to_mmap_gpu(loaded, &plan, &mut builder, qmode)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }
    #[cfg(not(feature = "metal-dispatch"))]
    let _ = (loaded, qmode);

    // Segment: Topology table
    builder.align_cursor();
    let topology_bytes = unsafe {
        std::slice::from_raw_parts(
            &topology_table as *const CImageTopologyTable as *const u8,
            std::mem::size_of::<CImageTopologyTable>(),
        )
    };
    builder
        .allocate_slice(topology_bytes.len())
        .copy_from_slice(topology_bytes);

    // Segment: Vocabulary / embedding triplet
    builder.align_cursor();
    if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
        let mmap_capture = builder.mmap_base();
        let _vocab_ptr =
            unsafe { builder.allocate_hardware_pointer(plan.vocabulary.length as usize) };
        if !embed_key.is_empty() {
            let (vocab_out_dim, vocab_in_dim, vocab_dtype) = {
                let st = loaded.source_tensors.get(embed_key).ok_or_else(|| {
                    std::io::Error::other(format!("missing vocabulary tensor {}", embed_key))
                })?;
                if st.shape.len() == 2 {
                    (st.shape[0], st.shape[1], st.dtype.clone())
                } else {
                    let elems = source_tensor_byte_len(st) as u32 / 2;
                    (elems / hidden_size, hidden_size, st.dtype.clone())
                }
            };
            let stem = embed_key.strip_suffix(".weight").unwrap_or(embed_key);
            let weight_len = loaded
                .source_tensors
                .get(embed_key)
                .map(source_tensor_byte_len)
                .unwrap_or(0) as u64;
            let scales_len = loaded
                .source_tensors
                .get(&format!("{}.scales", stem))
                .map(source_tensor_byte_len)
                .unwrap_or(0) as u64;
            let raw_bytes = {
                let st = loaded.source_tensors.get_mut(embed_key).ok_or_else(|| {
                    std::io::Error::other(format!("missing vocabulary tensor {}", embed_key))
                })?;
                for mmap in &loaded.mmap_bytes {
                    crate::ecs::compute_image::legacy_compute_image_compile::source::ensure_tensor_loaded(st, mmap);
                    if !st.data.is_empty() {
                        break;
                    }
                }
                std::mem::take(&mut st.data)
            };
            if !raw_bytes.is_empty() {
                crate::ecs::compute_image::legacy_compute_image_compile::try_nf4_tile640_pack_gpu_to_output(
                    loaded,
                    embed_key,
                    &raw_bytes,
                    &vocab_dtype,
                    vocab_out_dim,
                    vocab_in_dim,
                    Some(crate::ecs::compute_image::legacy_compute_image_compile::Nf4Tile640MmapOutput {
                        mmap_base: mmap_capture,
                        weights_offset: plan.vocabulary.offset,
                        scales_offset: plan.vocabulary.offset + weight_len,
                        biases_offset: plan.vocabulary.offset + weight_len + scales_len,
                    }),
                )
                .map_err(|e| std::io::Error::other(e.to_string()))?;
                eprintln!(
                    "[cimage] Vocabulary {} → GPU nf4 tile640 done ({}×{})",
                    embed_key, vocab_out_dim, vocab_in_dim
                );
            }
        }
    } else if plan.vocabulary.length > 0 && !embed_key.is_empty() {
        // Collect the raw BF16/FP16 bytes; lazy-load from mmap if needed.
        let raw_bytes: Vec<u8> = {
            let st = loaded.source_tensors.get_mut(embed_key).unwrap();
            for mmap in &loaded.mmap_bytes {
                crate::ecs::compute_image::legacy_compute_image_compile::source::ensure_tensor_loaded(st, mmap);
                if !st.data.is_empty() {
                    break;
                }
            }
            st.data.clone()
        };
        let (vocab_out_dim, vocab_in_dim) = {
            let st = &loaded.source_tensors[embed_key];
            if st.shape.len() == 2 {
                (st.shape[0], st.shape[1])
            } else {
                let elems = raw_bytes.len() as u32 / 2;
                (elems / hidden_size, hidden_size)
            }
        };
        eprintln!(
            "[cimage] Vocabulary: {} ({}×{}) → tile640",
            embed_key, vocab_out_dim, vocab_in_dim
        );

        // Reserve the vocabulary slice in the mmap.
        // Capture mmap_base and vocab_file_offset before mutable borrow of builder.
        let mmap_capture = builder.mmap_base();
        let _vocab_file_offset = plan.vocabulary.offset;
        let vocab_slice = builder.allocate_slice(plan.vocabulary.length as usize);
        let num_tiles = (vocab_in_dim as u64 + 639) / 640;
        let packed_len = (num_tiles as usize) * 32 * 4 * vocab_out_dim as usize;
        let scales_len = plan.vocabulary.length as usize - packed_len;

        // Try GPU-accelerated path first.
        #[cfg(feature = "metal-dispatch")]
        let gpu_done = {
            let mmap_base = mmap_capture;
            let vocab_file_offset = plan.vocabulary.offset;
            let result = crate::ecs::compute_image::legacy_compute_image_compile::try_ternary_tile640_pack_gpu(
                loaded,
                embed_key,
                &raw_bytes,
                vocab_out_dim,
                vocab_in_dim,
                Some((mmap_base, vocab_file_offset)),
            );
            match result {
                Ok(true) => {
                    eprintln!("[cimage] Vocabulary → GPU tile640 done");
                    true
                }
                Ok(false) | Err(_) => false,
            }
        };
        #[cfg(not(feature = "metal-dispatch"))]
        let gpu_done = false;

        // CPU fallback: run apply_ternary_tile640_quantize and copy into mmap.
        if !gpu_done {
            eprintln!("[cimage] Vocabulary → CPU tile640 fallback");
            let f32_vals: Vec<f32> = raw_bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    // FP16 fast path (also handles BF16 with minor precision loss).
                    let st = &loaded.source_tensors[embed_key];
                    if st.dtype == "BF16" {
                        let bf_bits = (bits as u32) << 16;
                        f32::from_bits(bf_bits)
                    } else {
                        crate::ecs::compute_image::legacy_compute_image_compile::half_to_f32(bits)
                    }
                })
                .collect();
            let tile_size: usize = 640;
            let padded_cols = ((vocab_in_dim as usize + tile_size - 1) / tile_size) * tile_size;
            let tile_count = padded_cols / tile_size;
            let u32_per_tile = 32usize;
            let total_u32s = vocab_out_dim as usize * tile_count * u32_per_tile;
            let mut packed_u32 = vec![0u32; total_u32s];
            let mut scales_f32 = Vec::with_capacity(vocab_out_dim as usize * tile_count);

            for row in 0..vocab_out_dim as usize {
                let row_offset = row * vocab_in_dim as usize;
                let row_padded: Vec<f32> = {
                    let mut r = f32_vals[row_offset..row_offset + vocab_in_dim as usize].to_vec();
                    r.resize(padded_cols, 0.0);
                    r
                };
                for t in 0..tile_count {
                    let tile = &row_padded[t * tile_size..(t + 1) * tile_size];
                    let absmax = tile.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                    scales_f32.push(scale);
                    let inv = 1.0 / scale;
                    for lane in 0..32usize {
                        let mut v: u32 = 0;
                        for w in 0..20usize {
                            let idx = lane * 20 + w;
                            let q = if idx < tile.len() {
                                let x = tile[idx] * inv;
                                if x > 0.5 {
                                    1
                                } else if x < -0.5 {
                                    2
                                } else {
                                    0
                                }
                            } else {
                                0
                            };
                            v = v * 3 + q;
                        }
                        packed_u32[row * tile_count * u32_per_tile + t * u32_per_tile + lane] = v;
                    }
                }
            }
            let packed_bytes: Vec<u8> = packed_u32.iter().flat_map(|&w| w.to_le_bytes()).collect();
            let scales_bytes: Vec<u8> = scales_f32.iter().flat_map(|&s| s.to_le_bytes()).collect();
            let copy_len = packed_bytes.len().min(packed_len);
            vocab_slice[..copy_len].copy_from_slice(&packed_bytes[..copy_len]);
            let scale_dst =
                &mut vocab_slice[packed_len..packed_len + scales_len.min(scales_bytes.len())];
            scale_dst.copy_from_slice(&scales_bytes[..scale_dst.len()]);
        }
    } else if plan.vocabulary.length > 0 {
        // Vocabulary was requested but embed key not found — zero-fill the slot.
        builder.allocate_slice(plan.vocabulary.length as usize);
    }

    // Segment: LayerDirectory (per-layer weight/scale byte offsets)
    builder.align_cursor();
    if num_layers > 0 && plan.layer_directory.length > 0 {
        let layer_dir_slice = builder.allocate_slice(plan.layer_directory.length as usize);
        let entries_typed = layer_directory_entries_for_loaded(loaded, false);
        let mut entries: Vec<u8> = Vec::with_capacity(entries_typed.len() * 48);
        for e in entries_typed.iter().copied() {
            entries.extend_from_slice(unsafe {
                std::slice::from_raw_parts(&e as *const LayerDirectoryEntry as *const u8, 48)
            });
        }
        layer_dir_slice[..entries.len()].copy_from_slice(&entries);
        let per_layer_kb = entries_typed
            .first()
            .map(|entry| entry.weights_length as f64 / 1024.0)
            .unwrap_or(0.0);
        eprintln!(
            "[cimage] LayerDirectory: {} entries x 48B, {:.1} KB per layer",
            entries_typed.len(),
            per_layer_kb,
        );
    }

    builder.align_cursor();
    if !execution_graph_bytes.is_empty() {
        builder
            .allocate_slice(plan.execution_graph.length as usize)
            .copy_from_slice(&execution_graph_bytes);
        eprintln!(
            "[cimage] ExecutionGraph: {} bytes, {} nodes",
            execution_graph_bytes.len(),
            ExecutionGraphDescriptor::from_bytes(&execution_graph_bytes)
                .map(|graph| graph.node_count)
                .unwrap_or(0),
        );
    }

    if let Some(multimodal) = &multimodal {
        for (segment, bytes) in [
            (
                plan.multimodal_projection_weights,
                &multimodal.projection_weights,
            ),
            (
                plan.multimodal_projection_scales,
                &multimodal.projection_scales,
            ),
            (
                plan.multimodal_projection_biases,
                &multimodal.projection_biases,
            ),
            (plan.multimodal_input_descriptor, &multimodal.descriptor),
            (
                plan.multimodal_position_embeddings,
                &multimodal.position_embeddings,
            ),
            (
                plan.multimodal_auxiliary_weights,
                &multimodal.auxiliary_weights,
            ),
        ] {
            if let Some(segment) = segment {
                if segment.length > 0 {
                    builder.align_cursor();
                    builder
                        .allocate_slice(segment.length as usize)
                        .copy_from_slice(bytes);
                }
            }
        }
    }

    // Header at offset 0
    let mut segments = [SegmentEntry {
        kind: 0,
        offset: 0,
        length: 0,
    }; CIMAGE_SEGMENT_CAPACITY];
    segments[0] = SegmentEntry::new(
        SegmentKind::MetalLib,
        plan.metal_lib.offset,
        plan.metal_lib.length,
    );
    segments[1] = SegmentEntry::new(
        SegmentKind::TopologyTable,
        plan.topology_table.offset,
        plan.topology_table.length,
    );
    // Segment 2: Vocabulary (TernaryTile640-packed embed_tokens.weight)
    if plan.vocabulary.length > 0 {
        segments[2] = SegmentEntry::new(
            SegmentKind::Vocabulary,
            plan.vocabulary.offset,
            plan.vocabulary.length,
        );
    }
    segments[SEG_MAIN_GRAPH] = SegmentEntry::new(
        SegmentKind::AneArchive,
        plan.main_graph.offset,
        plan.main_graph.length,
    );
    segments[SEG_MAIN_WEIGHTS] = SegmentEntry::new(
        if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
            SegmentKind::Nf4Tile640Weights
        } else {
            SegmentKind::TernaryWeights
        },
        plan.main_weights.offset,
        plan.main_weights.length,
    );
    if plan.main_scales.length > 0 {
        segments[SEG_MAIN_SCALES] = SegmentEntry::new(
            SegmentKind::BlockScales,
            plan.main_scales.offset,
            plan.main_scales.length,
        );
    }
    if plan.main_biases.length > 0 {
        segments[SEG_MAIN_BIASES] = SegmentEntry::new(
            SegmentKind::BlockBiases,
            plan.main_biases.offset,
            plan.main_biases.length,
        );
    }
    if plan.mtp_graph.length > 0 {
        // If MTP present, insert as a second AneArchive or LayoutMeta
        segments[SEG_MTP_GRAPH] = SegmentEntry::new(
            SegmentKind::AneArchive,
            plan.mtp_graph.offset,
            plan.mtp_graph.length,
        );
        segments[SEG_MTP_WEIGHTS] = SegmentEntry::new(
            if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
                SegmentKind::Nf4Tile640Weights
            } else {
                SegmentKind::TernaryWeights
            },
            plan.mtp_weights.offset,
            plan.mtp_weights.length,
        );
        if plan.mtp_scales.length > 0 {
            segments[SEG_MTP_SCALES] = SegmentEntry::new(
                SegmentKind::BlockScales,
                plan.mtp_scales.offset,
                plan.mtp_scales.length,
            );
        }
        if plan.mtp_biases.length > 0 {
            segments[SEG_MTP_BIASES] = SegmentEntry::new(
                SegmentKind::BlockBiases,
                plan.mtp_biases.offset,
                plan.mtp_biases.length,
            );
        }
    }
    // Segment 7: LayerDirectory (per-layer weight/scale offset table)
    if num_layers > 0 {
        segments[SEG_LAYER_DIRECTORY] = SegmentEntry::new(
            SegmentKind::LayerDirectory,
            plan.layer_directory.offset,
            plan.layer_directory.length,
        );
    }
    if plan.execution_graph.length > 0 {
        segments[SEG_EXECUTION_GRAPH] = SegmentEntry::new(
            SegmentKind::ExecutionGraph,
            plan.execution_graph.offset,
            plan.execution_graph.length,
        );
    }
    if let Some(segment) = plan.multimodal_projection_weights {
        if segment.length > 0 {
            segments[SEG_MM_PROJ_WEIGHTS] = SegmentEntry::new(
                SegmentKind::MultimodalProjectionWeights,
                segment.offset,
                segment.length,
            );
        }
    }
    if let Some(segment) = plan.multimodal_projection_scales {
        if segment.length > 0 {
            segments[SEG_MM_PROJ_SCALES] = SegmentEntry::new(
                SegmentKind::MultimodalProjectionScales,
                segment.offset,
                segment.length,
            );
        }
    }
    if let Some(segment) = plan.multimodal_projection_biases {
        if segment.length > 0 {
            segments[SEG_MM_PROJ_BIASES] = SegmentEntry::new(
                SegmentKind::MultimodalProjectionBiases,
                segment.offset,
                segment.length,
            );
        }
    }
    if let Some(segment) = plan.multimodal_input_descriptor {
        if segment.length > 0 {
            segments[SEG_MM_DESCRIPTOR] = SegmentEntry::new(
                SegmentKind::MultimodalInputDescriptor,
                segment.offset,
                segment.length,
            );
        }
    }
    if let Some(segment) = plan.multimodal_position_embeddings {
        if segment.length > 0 {
            segments[SEG_MM_POSITION] = SegmentEntry::new(
                SegmentKind::MultimodalPositionEmbeddings,
                segment.offset,
                segment.length,
            );
        }
    }
    if let Some(segment) = plan.multimodal_auxiliary_weights {
        if segment.length > 0 {
            segments[SEG_MM_AUX] = SegmentEntry::new(
                SegmentKind::MultimodalAuxiliaryWeights,
                segment.offset,
                segment.length,
            );
        }
    }
    let seg_count = segments
        .iter()
        .rposition(|segment| segment.length > 0)
        .map(|index| (index + 1) as u32)
        .unwrap_or(0);
    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: if plan.execution_graph.length > 0
            || plan
                .multimodal_input_descriptor
                .map(|segment| segment.length > 0)
                .unwrap_or(false)
        {
            6
        } else {
            4
        },
        segment_count: seg_count,
        payload_hash: [0u8; 32],
        num_layers,
        num_heads: 0,
        head_dim: 0,
        hidden_dim: 0,
        intermediate_dim: 0,
        vocab_size: 0,
        quantization_schema: if matches!(qmode, CompileQuantMode::Nf4Tile640 { .. }) {
            QUANT_SCHEMA_NF4_TILE640
        } else {
            QUANT_SCHEMA_TERNARY_TILE640
        },
        draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };
    let saved = builder.current_offset();
    builder.cursor = 0;
    builder.write_header(&header);
    builder.cursor = saved as usize;

    let mmap = builder.into_mmap();
    mmap.flush()?;
    eprintln!(
        "✅ .cimage: {} bytes, {} segments",
        plan.total_file_size, header.segment_count
    );
    Ok(())
}

// ── Sequential fallback (legacy, kept for compatibility) ────────────

const APPLE_PAGE_SIZE: u64 = super::APPLE_PAGE_SIZE as u64;

fn align_to_page<W: std::io::Write + std::io::Seek>(writer: &mut W) -> std::io::Result<u64> {
    let current_pos = writer.stream_position()?;
    let remainder = current_pos % APPLE_PAGE_SIZE;
    if remainder != 0 {
        let padding = vec![0u8; (APPLE_PAGE_SIZE - remainder) as usize];
        writer.write_all(&padding)?;
    }
    writer.stream_position()
}

pub fn pack_unified_cimage(
    output_path: &str,
    metal_lib_bytes: &[u8],
    main_graph_bytes: &[u8],
    main_weights_bytes: &[u8],
    mtp_graph_bytes: &[u8],
    mtp_weights_bytes: &[u8],
) -> std::io::Result<()> {
    let file = File::create(output_path)?;
    let mut writer = std::io::BufWriter::new(file);
    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    // Reserve header space
    writer.write_all(&vec![0u8; header_size as usize])?;

    fn write_segment(
        writer: &mut std::io::BufWriter<File>,
        data: &[u8],
    ) -> std::io::Result<(u64, u64)> {
        let offset = align_to_page(writer)?;
        writer.write_all(data)?;
        Ok((offset, data.len() as u64))
    }

    let (metal_lib_offset, metal_lib_len) = write_segment(&mut writer, metal_lib_bytes)?;
    let (main_weights_offset, main_weights_len) = write_segment(&mut writer, main_weights_bytes)?;
    let (mtp_weights_offset, mtp_weights_len) = write_segment(&mut writer, mtp_weights_bytes)?;
    let (main_graph_offset, main_graph_len) = write_segment(&mut writer, main_graph_bytes)?;
    let (mtp_graph_offset, mtp_graph_len) = write_segment(&mut writer, mtp_graph_bytes)?;

    let mut segments = [SegmentEntry {
        kind: 0,
        offset: 0,
        length: 0,
    }; CIMAGE_SEGMENT_CAPACITY];
    segments[0] = SegmentEntry::new(SegmentKind::MetalLib, metal_lib_offset, metal_lib_len);
    segments[1] = SegmentEntry::new(
        SegmentKind::TernaryWeights,
        main_weights_offset,
        main_weights_len,
    );
    segments[2] = SegmentEntry::new(
        SegmentKind::TernaryWeights,
        mtp_weights_offset,
        mtp_weights_len,
    );
    segments[3] = SegmentEntry::new(SegmentKind::AneArchive, main_graph_offset, main_graph_len);
    segments[4] = SegmentEntry::new(SegmentKind::AneArchive, mtp_graph_offset, mtp_graph_len);

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: if mtp_graph_bytes.is_empty() { 3 } else { 5 },
        payload_hash: [0u8; 32],
        num_layers: 0,
        num_heads: 0,
        head_dim: 0,
        hidden_dim: 0,
        intermediate_dim: 0,
        vocab_size: 0,
        quantization_schema: 0,
        draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };

    writer.seek(std::io::SeekFrom::Start(0))?;
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const CimageHeader) as *const u8,
            header_size as usize,
        )
    };
    writer.write_all(header_bytes)?;
    writer.flush()?;
    Ok(())
}

/// Pack a directory of compiled cimage fragments into a single .cimage binary.
///
/// Reads: model.metallib, segment_*.bin (weights), *.ane.tar (ANE archives)
/// from `input_dir` and produces a V4 unified .cimage at `output_path`.
/// Uses `ftruncate` + `mmap` + `AlignedMmapBuilder` for zero-copy GPU
/// compatibility (16 KB page alignment).
pub fn pack_cimage_from_dir(input_dir: &Path, output_path: &Path) -> std::io::Result<()> {
    const MAX_CIMAGE_SEGMENTS: usize = CIMAGE_SEGMENT_CAPACITY;

    #[derive(Clone)]
    struct Slot {
        kind: SegmentKind,
        offset: u64,
        length: u64,
    }

    let manifest = load_manifest_if_present(input_dir);

    // 1. Discover all segments in the directory
    let kernel_patterns: &[(&str, SegmentKind)] = &[
        ("model.metallib", SegmentKind::MetalLib),
        ("model.cubin", SegmentKind::CudaLib),
        ("model.fatbin", SegmentKind::CudaLib),
        ("model.co", SegmentKind::RocmLib),
        ("model.hsaco", SegmentKind::RocmLib),
        ("model_l0.spv", SegmentKind::LevelZeroLib),
        ("model_vulkan.spv", SegmentKind::VulkanLib),
        ("model_wgsl.spv", SegmentKind::WebGpuLib),
    ];
    let npu_patterns: &[(&str, SegmentKind)] = &[
        ("npu_intel.bin", SegmentKind::IntelNpuBlob),
        ("npu_amdxdna.bin", SegmentKind::AmdNpuBlob),
        ("npu_qualcomm.bin", SegmentKind::QualcommNpuBlob),
        ("npu_google.bin", SegmentKind::GoogleTpuBlob),
        ("npu_ane.tar", SegmentKind::AneArchive),
        ("npu_huawei.bin", SegmentKind::HuaweiAscendBlob),
        ("npu_hailo.hef", SegmentKind::HailoBlob),
    ];
    let tts_patterns: &[(&str, SegmentKind)] = &[
        ("tts_talker_weight.bin", SegmentKind::TtsTalkerWeight),
        ("tts_talker_scale.bin", SegmentKind::TtsTalkerScale),
        ("tts_talker_bias.bin", SegmentKind::TtsTalkerBias),
        (
            "tts_code_predictor_weight.bin",
            SegmentKind::TtsCodePredictorWeight,
        ),
        (
            "tts_code_predictor_scale.bin",
            SegmentKind::TtsCodePredictorScale,
        ),
        (
            "tts_code_predictor_bias.bin",
            SegmentKind::TtsCodePredictorBias,
        ),
        ("tts_codec_weight.bin", SegmentKind::TtsCodecWeight),
        ("tts_codebook.bin", SegmentKind::TtsCodebook),
    ];

    // ── TTS model segment pre-packing ────────────────────────
    let tts_safetensors_path = input_dir.join("tts_model.safetensors");
    if tts_safetensors_path.exists() {
        if let Ok(_tts_entries) = pack_tts_weights(&tts_safetensors_path, input_dir) {
            eprintln!(
                "[cimage] TTS weights pre-packed from '{}'",
                tts_safetensors_path.display()
            );
        }
    }

    // Store only paths for large disk-backed segments (streaming read during write phase).
    // In-memory synthesized data (execution graph, model artifacts) kept as Vec<u8>.
    let mut weight_files: Vec<PathBuf> = Vec::new();
    let mut extra_files: Vec<(SegmentKind, PathBuf)> = Vec::new();
    let mut extra_data: Vec<(SegmentKind, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(input_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        if name_str.starts_with("segment_") && name_str.ends_with(".bin") {
            weight_files.push(entry.path());
            continue;
        }
        let mut matched = false;
        for (pat, kind) in kernel_patterns {
            if name_str == *pat {
                extra_files.push((*kind, entry.path()));
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        for (pat, kind) in npu_patterns {
            if name_str == *pat
                || (kind == &SegmentKind::AneArchive && name_str.ends_with(".ane.tar"))
            {
                extra_files.push((*kind, entry.path()));
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }
        for (pat, kind) in tts_patterns {
            if name_str == *pat {
                extra_files.push((*kind, entry.path()));
                break;
            }
        }
    }

    let mut multimodal = synthesize_multimodal_segments(input_dir, manifest.as_ref())?;
    if let Some(bytes) = load_or_synthesize_execution_graph(input_dir, manifest.as_ref())? {
        extra_data.push((SegmentKind::ExecutionGraph, bytes));
    }
    if let Some(bytes) = load_or_synthesize_model_artifacts(input_dir, manifest.as_ref())? {
        extra_data.push((SegmentKind::ModelArtifacts, bytes));
    }

    // ── Heterogeneous execution image ──────────────────────────
    let heterogeneous_path = input_dir.join("heterogeneous_image.json");
    if heterogeneous_path.exists() {
        if let Ok(bytes) = std::fs::read(&heterogeneous_path) {
            extra_data.push((SegmentKind::HeterogeneousImage, bytes));
        }
    }

    // Sum byte lengths for all segments of a given kind across both file and in-memory sources.
    let segment_total_len = |kind: SegmentKind| -> u64 {
        let file_sum: u64 = extra_files
            .iter()
            .filter(|(k, _)| *k == kind)
            .filter_map(|(_, p)| p.metadata().ok().map(|m| m.len()))
            .sum();
        let mem_sum: u64 = extra_data
            .iter()
            .filter(|(k, _)| *k == kind)
            .map(|(_, d)| d.len() as u64)
            .sum();
        file_sum + mem_sum
    };

    // 2. Compute layout
    let mut slots: Vec<Slot> = Vec::new();
    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    let weights_total: u64 = weight_files
        .iter()
        .filter_map(|p| p.metadata().ok().map(|m| m.len()))
        .sum();

    let mut cursor = header_size;
    let mut push_slot = |kind: SegmentKind, len: u64| {
        if len == 0 {
            return;
        }
        let r = cursor % APPLE_PAGE_SIZE;
        if r != 0 {
            cursor += APPLE_PAGE_SIZE - r;
        }
        slots.push(Slot {
            kind,
            offset: cursor,
            length: len,
        });
        cursor += len;
    };
    for kind in &[
        SegmentKind::MetalLib,
        SegmentKind::CudaLib,
        SegmentKind::RocmLib,
        SegmentKind::LevelZeroLib,
        SegmentKind::VulkanLib,
        SegmentKind::WebGpuLib,
    ] {
        push_slot(*kind, segment_total_len(*kind));
    }
    push_slot(SegmentKind::TernaryWeights, weights_total);
    if let Some(multimodal) = &multimodal {
        push_slot(
            SegmentKind::MultimodalProjectionWeights,
            multimodal.projection_weights.len() as u64,
        );
        push_slot(
            SegmentKind::MultimodalProjectionScales,
            multimodal.projection_scales.len() as u64,
        );
        push_slot(
            SegmentKind::MultimodalProjectionBiases,
            multimodal.projection_biases.len() as u64,
        );
        push_slot(
            SegmentKind::MultimodalInputDescriptor,
            multimodal.descriptor.len() as u64,
        );
        push_slot(
            SegmentKind::MultimodalPositionEmbeddings,
            multimodal.position_embeddings.len() as u64,
        );
        push_slot(
            SegmentKind::MultimodalAuxiliaryWeights,
            multimodal.auxiliary_weights.len() as u64,
        );
    }
    for kind in &[SegmentKind::ExecutionGraph, SegmentKind::ModelArtifacts] {
        push_slot(*kind, segment_total_len(*kind));
    }
    for kind in &[
        SegmentKind::AneArchive,
        SegmentKind::IntelNpuBlob,
        SegmentKind::AmdNpuBlob,
        SegmentKind::QualcommNpuBlob,
        SegmentKind::GoogleTpuBlob,
        SegmentKind::HuaweiAscendBlob,
        SegmentKind::HailoBlob,
    ] {
        push_slot(*kind, segment_total_len(*kind));
    }
    for kind in &[
        SegmentKind::TtsTalkerWeight,
        SegmentKind::TtsTalkerScale,
        SegmentKind::TtsTalkerBias,
        SegmentKind::TtsCodePredictorWeight,
        SegmentKind::TtsCodePredictorScale,
        SegmentKind::TtsCodePredictorBias,
        SegmentKind::TtsCodecWeight,
        SegmentKind::TtsCodebook,
    ] {
        push_slot(*kind, segment_total_len(*kind));
    }
    if let Some(multimodal) = &mut multimodal {
        if multimodal.descriptor.len() >= std::mem::size_of::<MultimodalInputDescriptorV1>() {
            let find_slot_index = |kind: SegmentKind| {
                slots
                    .iter()
                    .enumerate()
                    .find_map(|(index, slot)| (slot.kind == kind).then_some(index as u16))
                    .unwrap_or(u16::MAX)
            };
            let desc = unsafe {
                &mut *(multimodal.descriptor.as_mut_ptr() as *mut MultimodalInputDescriptorV1)
            };
            desc.projection_weight_segment_index =
                find_slot_index(SegmentKind::MultimodalProjectionWeights);
            desc.projection_scale_segment_index =
                find_slot_index(SegmentKind::MultimodalProjectionScales);
            desc.projection_bias_segment_index =
                find_slot_index(SegmentKind::MultimodalProjectionBiases);
            desc.position_embedding_segment_index =
                find_slot_index(SegmentKind::MultimodalPositionEmbeddings);
            desc.auxiliary_weight_segment_index =
                find_slot_index(SegmentKind::MultimodalAuxiliaryWeights);
        }
    }
    if let Some(multimodal) = &multimodal {
        if let Some((_, graph_bytes)) = extra_data
            .iter_mut()
            .find(|(kind, _)| *kind == SegmentKind::ExecutionGraph)
        {
            let _ = patch_execution_graph_multimodal_nodes(
                graph_bytes.as_mut_slice(),
                &multimodal.descriptor,
            );
        }
    }
    let total_file_size = cursor;

    // 3. Allocate and fill via ftruncate + mmap
    use std::fs::OpenOptions;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(output_path)?;
    file.set_len(total_file_size)?;
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };
    unsafe {
        std::ptr::write_bytes(mmap.as_mut_ptr(), 0u8, mmap.len());
    }
    let mut builder = AlignedMmapBuilder::new(mmap);
    builder.cursor = header_size as usize;
    let mut payload_hasher = Sha256::new();

    // Write all segments by iterating slots in their computed (header) order.
    // Snap cursor to the layout-computed slot offset (PATTERN A).
    // File-backed segments are read one-at-a-time and dropped immediately.
    for slot in &slots {
        if slot.length == 0 {
            continue;
        }
        builder.cursor = slot.offset as usize;
        let mut seg_slice = builder.allocate_slice(slot.length as usize);

        match slot.kind {
            SegmentKind::TernaryWeights => {
                for wpath in &weight_files {
                    let data = std::fs::read(wpath)?;
                    payload_hasher.update(&data);
                    let (head, tail) = seg_slice.split_at_mut(data.len());
                    head.copy_from_slice(&data);
                    seg_slice = tail;
                }
            }
            SegmentKind::MultimodalProjectionWeights
            | SegmentKind::MultimodalProjectionScales
            | SegmentKind::MultimodalProjectionBiases
            | SegmentKind::MultimodalInputDescriptor
            | SegmentKind::MultimodalPositionEmbeddings
            | SegmentKind::MultimodalAuxiliaryWeights => {
                let bytes = match slot.kind {
                    SegmentKind::MultimodalProjectionWeights => {
                        multimodal.as_ref().map(|m| &m.projection_weights[..])
                    }
                    SegmentKind::MultimodalProjectionScales => {
                        multimodal.as_ref().map(|m| &m.projection_scales[..])
                    }
                    SegmentKind::MultimodalProjectionBiases => {
                        multimodal.as_ref().map(|m| &m.projection_biases[..])
                    }
                    SegmentKind::MultimodalInputDescriptor => {
                        multimodal.as_ref().map(|m| &m.descriptor[..])
                    }
                    SegmentKind::MultimodalPositionEmbeddings => {
                        multimodal.as_ref().map(|m| &m.position_embeddings[..])
                    }
                    SegmentKind::MultimodalAuxiliaryWeights => {
                        multimodal.as_ref().map(|m| &m.auxiliary_weights[..])
                    }
                    _ => None,
                };
                if let Some(bytes) = bytes {
                    payload_hasher.update(bytes);
                    seg_slice.copy_from_slice(bytes);
                }
            }
            _ => {
                // File-backed segments of this kind (MetalLib, AneArchive, etc.)
                for (kind, path) in &extra_files {
                    if *kind == slot.kind {
                        let data = std::fs::read(path)?;
                        payload_hasher.update(&data);
                        let (head, tail) = seg_slice.split_at_mut(data.len());
                        head.copy_from_slice(&data);
                        seg_slice = tail;
                    }
                }
                // In-memory segments of this kind (ExecutionGraph, ModelArtifacts, etc.)
                for (kind, data) in &extra_data {
                    if *kind == slot.kind {
                        payload_hasher.update(data);
                        let (head, tail) = seg_slice.split_at_mut(data.len());
                        head.copy_from_slice(data);
                        seg_slice = tail;
                    }
                }
            }
        }
    }

    // 4. Build header
    let total_segments = slots.len().min(MAX_CIMAGE_SEGMENTS);
    let mut segments_dir = [SegmentEntry {
        kind: 0,
        offset: 0,
        length: 0,
    }; MAX_CIMAGE_SEGMENTS];
    for (i, slot) in slots.iter().enumerate().take(MAX_CIMAGE_SEGMENTS) {
        segments_dir[i] = SegmentEntry::new(slot.kind, slot.offset, slot.length);
    }
    if slots.len() > MAX_CIMAGE_SEGMENTS {
        eprintln!(
            "[cimage] warning: {} segments discovered, truncating to {} entries due to header limit",
            slots.len(),
            MAX_CIMAGE_SEGMENTS
        );
    }

    let payload_hash: [u8; 32] = payload_hasher.finalize().into();
    let (
        num_layers,
        num_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        draft_num_layers,
    ) = header_fields_from_manifest(manifest.as_ref());
    let header_version = if slots.iter().any(|slot| {
        matches!(
            slot.kind,
            SegmentKind::ExecutionGraph | SegmentKind::ModelArtifacts
        )
    }) {
        6
    } else {
        4
    };

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: header_version,
        segment_count: total_segments as u32,
        payload_hash,
        num_layers,
        num_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        quantization_schema: 0,
        draft_num_layers,
        segments: segments_dir,
        _pad: [0u8; 8],
    };

    let saved = builder.current_offset();
    builder.cursor = 0;
    builder.write_header(&header);
    builder.cursor = saved as usize;

    let mmap = builder.into_mmap();
    mmap.flush()?;
    eprintln!(
        "[cimage] packed {} bytes, {} segments → {}",
        total_file_size,
        total_segments,
        output_path.display()
    );
    Ok(())
}

fn load_manifest_if_present(input_dir: &Path) -> Option<Manifest> {
    let bytes = std::fs::read(input_dir.join("manifest.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn load_or_synthesize_execution_graph(
    input_dir: &Path,
    manifest: Option<&Manifest>,
) -> std::io::Result<Option<Vec<u8>>> {
    let path = input_dir.join("execution_graph.bin");
    if path.exists() {
        return std::fs::read(path).map(Some);
    }
    Ok(manifest.and_then(synthesize_execution_graph))
}

fn load_or_synthesize_model_artifacts(
    input_dir: &Path,
    manifest: Option<&Manifest>,
) -> std::io::Result<Option<Vec<u8>>> {
    let path = input_dir.join("model_artifacts.bin");
    if path.exists() {
        return std::fs::read(path).map(Some);
    }
    Ok(synthesize_model_artifacts(input_dir, manifest))
}

fn read_projection_records_from_descriptor_bytes(
    descriptor_bytes: &[u8],
) -> Option<Vec<ProjectionTensorRecord>> {
    if descriptor_bytes.len() < std::mem::size_of::<MultimodalInputDescriptorV1>() {
        return None;
    }
    let desc = unsafe {
        std::ptr::read_unaligned(descriptor_bytes.as_ptr() as *const MultimodalInputDescriptorV1)
    };
    let total_records = desc.image_projection_count as usize + desc.audio_projection_count as usize;
    let record_size = std::mem::size_of::<ProjectionTensorRecord>();
    let records_offset = std::mem::size_of::<MultimodalInputDescriptorV1>();
    let byte_len = total_records.checked_mul(record_size)?;
    let end = records_offset.checked_add(byte_len)?;
    if end > descriptor_bytes.len() {
        return None;
    }

    let mut records = Vec::with_capacity(total_records);
    let mut cursor = records_offset;
    for _ in 0..total_records {
        let record = unsafe {
            std::ptr::read_unaligned(
                descriptor_bytes[cursor..].as_ptr() as *const ProjectionTensorRecord
            )
        };
        records.push(record);
        cursor += record_size;
    }
    Some(records)
}

fn patch_execution_graph_multimodal_nodes(graph_bytes: &mut [u8], descriptor_bytes: &[u8]) -> bool {
    let Some(records) = read_projection_records_from_descriptor_bytes(descriptor_bytes) else {
        return false;
    };
    let Ok(mut graph) = ExecutionGraphDescriptor::from_bytes(graph_bytes) else {
        return false;
    };

    let role_for_node = |node_kind: u8| match node_kind {
        x if x == NodeKind::VisionPatchEmbed as u8 => Some(ProjectionRole::ImagePatchEmbedding),
        x if x == NodeKind::VisionFinalProjection as u8 => Some(ProjectionRole::ImageProjection),
        x if x == NodeKind::AudioFrameEmbed as u8 => Some(ProjectionRole::AudioFrameEmbedding),
        x if x == NodeKind::AudioProjection as u8 => Some(ProjectionRole::AudioProjection),
        _ => None,
    };

    let mut changed = false;
    for node in &mut graph.layers {
        let Some(role) = role_for_node(node.node_kind) else {
            continue;
        };
        let Some(record) = records.iter().find(|record| record.role == role as u16) else {
            continue;
        };
        node.weight_offset = record.weight_offset;
        node.weight_length = record.weight_length;
        node.scale_offset = record.scale_offset;
        node.hidden_dim = record.output_width;
        changed = true;
    }

    if changed {
        let bytes = graph.to_bytes();
        if bytes.len() == graph_bytes.len() {
            graph_bytes.copy_from_slice(&bytes);
            true
        } else {
            false
        }
    } else {
        false
    }
}

fn synthesize_execution_graph(manifest: &Manifest) -> Option<Vec<u8>> {
    let text = &manifest.architecture;
    let mut layers = Vec::new();

    if let Some(vision) = &manifest.vision_config {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::VisionPatchEmbed as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: vision.patch_size.min(u16::MAX as u32) as u16,
            num_heads: vision.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: vision.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::VisionFinalProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    if let Some(audio) = &manifest.audio_config {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::AudioFrameEmbed as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: audio.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: audio.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::AudioProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    if manifest.vision_config.is_some() || manifest.audio_config.is_some() {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::EmbeddingAssembly as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    let mut compaction_epochs = Vec::new();
    for layer in &manifest.execution_plan.layers {
        let is_sliding = layer
            .attention_kind
            .eq_ignore_ascii_case("slidingattention")
            || layer
                .attention_kind
                .eq_ignore_ascii_case("sliding_attention");
        let compaction_epoch = if is_sliding {
            let epoch_index = compaction_epochs.len() as u8;
            compaction_epochs.push(CompactionEpoch {
                epoch_index,
                trigger_layer: layer.layer_index.min(u8::MAX as u32) as u8,
                tier_count: 1,
                _pad: 0,
                compression_ratio: [0; 4],
                tier_boundaries: [0; 3],
                access_threshold: 0,
            });
            epoch_index
        } else {
            0xFF
        };
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DecoderLayer as u8,
            attention_kind: if is_sliding {
                GraphAttentionKind::SlidingWindow as u8
            } else {
                GraphAttentionKind::FullAttention as u8
            },
            device_capability: DeviceCapability::Both as u8,
            compaction_epoch,
            layer_index: layer.layer_index,
            head_dim: layer.head_dim.min(u16::MAX as u32) as u16,
            num_heads: layer.n_heads.min(u16::MAX as u32) as u16,
            hidden_dim: layer.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
    }

    let draft_sub_graph = manifest
        .execution_plan
        .speculative_config
        .as_ref()
        .map(|draft| DraftSubGraph {
            num_layers: draft.draft_architecture.num_hidden_layers,
            hidden_dim: draft.draft_architecture.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            scale_length: 0,
            pre_proj_offset: 0,
            post_proj_offset: 0,
        });

    if let Some(draft) = &manifest.execution_plan.speculative_config {
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DraftPreProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: draft.draft_architecture.head_dim.min(u16::MAX as u32) as u16,
            num_heads: draft
                .draft_architecture
                .num_attention_heads
                .min(u16::MAX as u32) as u16,
            hidden_dim: text.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        layers.push(LayerExecutionNode {
            node_kind: NodeKind::DraftPostProjection as u8,
            attention_kind: 2,
            device_capability: DeviceCapability::Gpu as u8,
            compaction_epoch: 0xFF,
            layer_index: 0,
            head_dim: text.head_dim.min(u16::MAX as u32) as u16,
            num_heads: text.num_attention_heads.min(u16::MAX as u32) as u16,
            hidden_dim: draft.draft_architecture.hidden_size,
            weight_offset: 0,
            weight_length: 0,
            scale_offset: 0,
            _reserved: [0u8; 8],
        });
        for layer_index in 0..draft.draft_architecture.num_hidden_layers {
            layers.push(LayerExecutionNode {
                node_kind: NodeKind::DraftLayer as u8,
                attention_kind: GraphAttentionKind::FullAttention as u8,
                device_capability: DeviceCapability::Both as u8,
                compaction_epoch: 0xFF,
                layer_index,
                head_dim: draft.draft_architecture.head_dim.min(u16::MAX as u32) as u16,
                num_heads: draft
                    .draft_architecture
                    .num_attention_heads
                    .min(u16::MAX as u32) as u16,
                hidden_dim: draft.draft_architecture.hidden_size,
                weight_offset: 0,
                weight_length: 0,
                scale_offset: 0,
                _reserved: [0u8; 8],
            });
        }
    }

    if layers.is_empty() {
        return None;
    }

    Some(
        ExecutionGraphDescriptor {
            magic: crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers: manifest.architecture.num_hidden_layers.min(u16::MAX as u32) as u16,
            num_draft_layers: manifest
                .execution_plan
                .speculative_config
                .as_ref()
                .map(|draft| draft.draft_architecture.num_hidden_layers)
                .unwrap_or(0)
                .min(u16::MAX as u32) as u16,
            num_compaction_epochs: compaction_epochs.len().min(u16::MAX as usize) as u16,
            node_count: layers.len().min(u32::MAX as usize) as u32,
            _pad: [0u8; 2],
            layers,
            compaction_epochs,
            draft_sub_graph,
        }
        .to_bytes(),
    )
}

fn synthesize_model_artifacts(input_dir: &Path, manifest: Option<&Manifest>) -> Option<Vec<u8>> {
    let mut artifacts = Vec::new();

    for name in ["tokenizer.model", "tokenizer.json"] {
        let path = input_dir.join(name);
        if path.exists() {
            let data = std::fs::read(path).ok()?;
            ModelArtifactEntry::encode(model_artifact_tag::TOKENIZER, &data, &mut artifacts);
            break;
        }
    }

    let tokenizer_config = read_json_if_present(&input_dir.join("tokenizer_config.json"));
    let generation_config = read_json_if_present(&input_dir.join("generation_config.json"));

    if let Some(chat_template) = tokenizer_config
        .as_ref()
        .and_then(|json| json.get("chat_template"))
        .and_then(|value| value.as_str())
    {
        ModelArtifactEntry::encode(
            model_artifact_tag::CHAT_TEMPLATE,
            chat_template.as_bytes(),
            &mut artifacts,
        );
    }

    if let Some(config) = generation_config.as_ref() {
        let bytes = serde_json::to_vec(config).ok()?;
        ModelArtifactEntry::encode(
            model_artifact_tag::GENERATION_CONFIG,
            &bytes,
            &mut artifacts,
        );
    }

    let mut token_map = serde_json::Map::new();
    if let Some(config) = tokenizer_config.as_ref() {
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
            if let Some(value) = config.get(key) {
                token_map.insert(key.to_string(), value.clone());
            }
        }
    }
    if let Some(manifest) = manifest {
        if let Some(vision) = &manifest.vision_config {
            token_map.insert(
                "image_start_token".into(),
                serde_json::Value::String("<start_of_image>".into()),
            );
            token_map.insert(
                "image_end_token".into(),
                serde_json::Value::String("<end_of_image>".into()),
            );
            token_map.insert(
                "image_token_count".into(),
                serde_json::Value::from(vision.hidden_size as u64),
            );
            token_map.insert(
                "vision_patch_size".into(),
                serde_json::Value::from(vision.patch_size as u64),
            );
        }
        if let Some(audio) = &manifest.audio_config {
            token_map.insert(
                "audio_start_token".into(),
                serde_json::Value::String("<start_of_audio>".into()),
            );
            token_map.insert(
                "audio_end_token".into(),
                serde_json::Value::String("<end_of_audio>".into()),
            );
            token_map.insert(
                "audio_sample_rate".into(),
                serde_json::Value::from(audio.sample_rate as u64),
            );
            let frame_ms = if audio.sample_rate > 0 {
                ((audio.hop_length as f64 / audio.sample_rate as f64) * 1000.0).round() as u64
            } else {
                0
            };
            token_map.insert("audio_frame_ms".into(), serde_json::Value::from(frame_ms));
        }
    }
    if !token_map.is_empty() {
        let bytes = serde_json::to_vec(&token_map).ok()?;
        ModelArtifactEntry::encode(model_artifact_tag::TOKEN_MAP, &bytes, &mut artifacts);
    }

    if artifacts.is_empty() {
        None
    } else {
        Some(artifacts)
    }
}

struct SynthesizedMultimodalSegments {
    projection_weights: Vec<u8>,
    projection_scales: Vec<u8>,
    /// Byte-parallel to `projection_scales` (kernels/MULTIMODAL_NF4_BIAS_ABI.md):
    /// filled in lockstep from `{stem}.biases` sidecar tensors. Empty when the
    /// quantizer emitted no biases — the v1-compat shape.
    projection_biases: Vec<u8>,
    descriptor: Vec<u8>,
    position_embeddings: Vec<u8>,
    auxiliary_weights: Vec<u8>,
}

fn logical_shape_for_tensor(loaded: &LoadedSource, name: &str) -> Vec<u32> {
    loaded
        .spec
        .global_tensors
        .iter()
        .chain(
            loaded
                .spec
                .layers
                .iter()
                .flat_map(|layer| layer.tensors.iter()),
        )
        .find(|binding| binding.name == name)
        .map(|binding| binding.logical_shape.clone())
        .unwrap_or_else(|| {
            loaded
                .source_tensors
                .get(name)
                .map(|tensor| tensor.shape.clone())
                .unwrap_or_default()
        })
}

fn synthesize_multimodal_segments_for_loaded(
    loaded: &LoadedSource,
) -> std::io::Result<Option<SynthesizedMultimodalSegments>> {
    if loaded.manifest.vision_config.is_none() && loaded.manifest.audio_config.is_none() {
        return Ok(None);
    }

    let mut projection_weights = Vec::new();
    let mut projection_scales = Vec::new();
    let mut projection_biases = Vec::new();
    let mut position_embeddings = Vec::new();
    let mut auxiliary_weights = Vec::new();
    let mut image_records = Vec::new();
    let mut audio_records = Vec::new();

    let mut tensor_names: Vec<&String> = loaded
        .source_tensors
        .keys()
        .filter(|name| classify_multimodal_tensor(name).is_some())
        .collect();
    tensor_names.sort();

    for name in tensor_names {
        let Some(class) = classify_multimodal_tensor(name) else {
            continue;
        };
        let logical_shape = logical_shape_for_tensor(loaded, name);
        let entry_kind = classify_multimodal_entry(name, &logical_shape);
        let Some(tensor) = loaded.source_tensors.get(name) else {
            continue;
        };
        let tensor_bytes = source_tensor_view(tensor, &loaded.mmap_bytes).to_vec();
        if tensor_bytes.is_empty() {
            continue;
        }

        let start_offset = match entry_kind {
            MultimodalEntryKind::ProjectionWeight => {
                let start = projection_weights.len() as u64;
                projection_weights.extend_from_slice(&tensor_bytes);
                start
            }
            MultimodalEntryKind::PositionEmbedding => {
                position_embeddings.extend_from_slice(&tensor_bytes);
                continue;
            }
            MultimodalEntryKind::Auxiliary => {
                auxiliary_weights.extend_from_slice(&tensor_bytes);
                continue;
            }
        };

        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let scale_name = format!("{}.scales", stem);
        let bias_name = format!("{}.biases", stem);
        let mut record_flags = 0u8;
        let (scale_offset, scale_length, layout_code, quantization_kind) =
            if let Some(scale_tensor) = loaded.source_tensors.get(&scale_name) {
                let scale_bytes = source_tensor_view(scale_tensor, &loaded.mmap_bytes).to_vec();
                if !scale_bytes.is_empty() {
                    let scale_offset = projection_scales.len() as u64;
                    projection_scales.extend_from_slice(&scale_bytes);
                    // Bias sidecar: captured byte-parallel to the scales. The
                    // parallelism contract is structural — the bias segment
                    // advances in lockstep with the scale segment, so a
                    // record's scale_offset/scale_length address both. That
                    // only holds if EVERY scaled record carries biases, which
                    // the all-or-none check after this loop enforces.
                    if let Some(bias_tensor) = loaded.source_tensors.get(&bias_name) {
                        let bias_bytes =
                            source_tensor_view(bias_tensor, &loaded.mmap_bytes).to_vec();
                        if !bias_bytes.is_empty() {
                            if bias_bytes.len() != scale_bytes.len() {
                                return Err(std::io::Error::other(format!(
                                    "multimodal bias sidecar {bias_name} is {} bytes but \
                                     {scale_name} is {} — biases must be scale-parallel \
                                     ([tiles × 5] f32 per row)",
                                    bias_bytes.len(),
                                    scale_bytes.len()
                                )));
                            }
                            let bias_offset = projection_biases.len() as u64;
                            if bias_offset != scale_offset {
                                return Err(std::io::Error::other(format!(
                                    "multimodal bias segment desynchronized at {bias_name}: \
                                     bias cursor {bias_offset} vs scale cursor {scale_offset} \
                                     — a preceding scaled record lacked its bias sidecar"
                                )));
                            }
                            projection_biases.extend_from_slice(&bias_bytes);
                            record_flags |= ProjectionTensorRecord::FLAG_HAS_BIAS;
                        }
                    }
                    (
                        scale_offset,
                        scale_bytes.len() as u64,
                        ProjectionTensorRecord::LAYOUT_NF4_TILE640,
                        ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
                    )
                } else {
                    (
                        0,
                        0,
                        ProjectionTensorRecord::LAYOUT_DENSE_ROW_MAJOR,
                        ProjectionTensorRecord::QUANTIZATION_NONE,
                    )
                }
            } else {
                (
                    0,
                    0,
                    ProjectionTensorRecord::LAYOUT_DENSE_ROW_MAJOR,
                    ProjectionTensorRecord::QUANTIZATION_NONE,
                )
            };

        let record = ProjectionTensorRecord {
            logical_name_hash: stable_name_hash(name),
            role: projection_role_for_name(name) as u16,
            dtype: dtype_code(&tensor.dtype),
            weight_offset: start_offset,
            weight_length: tensor_bytes.len() as u64,
            scale_offset,
            scale_length,
            input_width: logical_shape.get(1).copied().unwrap_or(0),
            output_width: logical_shape.first().copied().unwrap_or(0),
            rank: logical_shape.len() as u8,
            layout: layout_code,
            quantization_kind,
            flags: record_flags,
            dims: dims4(&logical_shape),
        };
        match class {
            MultimodalClass::Image => image_records.push(record),
            MultimodalClass::Audio => audio_records.push(record),
        }
    }

    if projection_weights.is_empty()
        && projection_scales.is_empty()
        && position_embeddings.is_empty()
        && auxiliary_weights.is_empty()
    {
        return Ok(None);
    }

    let desc_size = std::mem::size_of::<MultimodalInputDescriptorV1>() as u64;
    let image_offset = desc_size;
    let audio_offset =
        image_offset + (image_records.len() * std::mem::size_of::<ProjectionTensorRecord>()) as u64;
    let mut desc = MultimodalInputDescriptorV1::default();
    desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
    desc.version = 1;
    desc.modality_mask = 0b0001
        | if !image_records.is_empty() { 0b0010 } else { 0 }
        | if !audio_records.is_empty() { 0b0100 } else { 0 };
    desc.decoder_hidden_size = loaded.arch.hidden_size;
    desc.vocabulary_size = loaded.arch.vocab_size;
    if let Some(vision) = &loaded.manifest.vision_config {
        desc.image_patch_size = vision.patch_size.min(u16::MAX as u32) as u16;
        desc.image_channels = vision.num_channels.min(u16::MAX as u32) as u16;
        desc.image_position_embedding_width = vision.hidden_size;
    }
    desc.image_projection_table_offset = image_offset;
    desc.image_projection_count = image_records.len().min(u32::MAX as usize) as u32;
    desc.audio_projection_table_offset = audio_offset;
    desc.audio_projection_count = audio_records.len().min(u32::MAX as usize) as u32;
    desc.processor_contract_digest = Sha256::digest(&projection_weights).into();
    let mut layout_hasher = Sha256::new();
    for record in image_records.iter().chain(audio_records.iter()) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                record as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        };
        layout_hasher.update(bytes);
    }
    desc.tensor_layout_digest = layout_hasher.finalize().into();

    let mut descriptor = Vec::with_capacity(
        std::mem::size_of::<MultimodalInputDescriptorV1>()
            + (image_records.len() + audio_records.len())
                * std::mem::size_of::<ProjectionTensorRecord>(),
    );
    descriptor.extend_from_slice(unsafe {
        std::slice::from_raw_parts(
            &desc as *const MultimodalInputDescriptorV1 as *const u8,
            std::mem::size_of::<MultimodalInputDescriptorV1>(),
        )
    });
    for record in image_records.iter().chain(audio_records.iter()) {
        descriptor.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                record as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        });
    }

    // All-or-none bias policy: the parallel-layout contract (bias offsets ≡
    // scale offsets) is only sound when every scaled record carried a bias
    // sidecar. A partial set means the quantizer output is inconsistent —
    // fail the pack rather than seal an artifact whose bias views would
    // silently misalign. (The lockstep cursor check above catches orderings
    // where a scaled-but-biasless record precedes a biased one; this catches
    // the trailing case.)
    if !projection_biases.is_empty() && projection_biases.len() != projection_scales.len() {
        return Err(std::io::Error::other(format!(
            "multimodal bias segment is {} bytes but the scale segment is {} — \
             bias sidecars must be present for ALL scaled projections or none",
            projection_biases.len(),
            projection_scales.len()
        )));
    }

    Ok(Some(SynthesizedMultimodalSegments {
        projection_weights,
        projection_scales,
        projection_biases,
        descriptor,
        position_embeddings,
        auxiliary_weights,
    }))
}

fn synthesize_multimodal_segments(
    input_dir: &Path,
    manifest: Option<&Manifest>,
) -> std::io::Result<Option<SynthesizedMultimodalSegments>> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if manifest.vision_config.is_none() && manifest.audio_config.is_none() {
        return Ok(None);
    }

    let segment_files: HashMap<&str, std::path::PathBuf> = manifest
        .segments
        .iter()
        .map(|segment| (segment.id.as_str(), input_dir.join(&segment.filename)))
        .collect();

    let mut projection_weights = Vec::new();
    let mut projection_scales = Vec::new();
    let mut position_embeddings = Vec::new();
    let mut auxiliary_weights = Vec::new();
    let mut image_records = Vec::new();
    let mut audio_records = Vec::new();
    let tensor_by_id: HashMap<u32, &crate::ecs::compute_image::manifest::TensorEntry> = manifest
        .tensor_table
        .iter()
        .map(|tensor| (tensor.id, tensor))
        .collect();

    for tensor in &manifest.tensor_table {
        let Some(class) = classify_multimodal_tensor(&tensor.name) else {
            continue;
        };
        let Some(path) = segment_files.get(tensor.segment.as_str()) else {
            continue;
        };
        let bytes = read_tensor_payload(path, tensor.offset, tensor.byte_length)?;
        let entry_kind = classify_multimodal_entry(&tensor.name, &tensor.logical_shape);
        let start_offset = match entry_kind {
            MultimodalEntryKind::ProjectionWeight => {
                let start = projection_weights.len() as u64;
                projection_weights.extend_from_slice(&bytes);
                start
            }
            MultimodalEntryKind::PositionEmbedding => {
                position_embeddings.extend_from_slice(&bytes);
                continue;
            }
            MultimodalEntryKind::Auxiliary => {
                auxiliary_weights.extend_from_slice(&bytes);
                continue;
            }
        };
        let (scale_offset, scale_length, layout_code, quantization_kind) = if let Some(quant) =
            &tensor.quantization
        {
            match &quant.storage_layout {
                Some(SharedWeightLayout::Nf4Tile640(_layout)) => {
                    let Some(scale_tensor) = tensor_by_id.get(&quant.scale_tensor_id).copied()
                    else {
                        return Err(std::io::Error::other(format!(
                            "missing multimodal scale tensor id {} for {}",
                            quant.scale_tensor_id, tensor.name
                        )));
                    };
                    let Some(scale_path) = segment_files.get(scale_tensor.segment.as_str()) else {
                        return Err(std::io::Error::other(format!(
                            "missing segment file for multimodal scale tensor {}",
                            scale_tensor.name
                        )));
                    };
                    let scale_bytes = read_tensor_payload(
                        scale_path,
                        scale_tensor.offset,
                        scale_tensor.byte_length,
                    )?;
                    let scale_offset = projection_scales.len() as u64;
                    projection_scales.extend_from_slice(&scale_bytes);
                    (
                        scale_offset,
                        scale_bytes.len() as u64,
                        ProjectionTensorRecord::LAYOUT_NF4_TILE640,
                        ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
                    )
                }
                _ => (
                    0,
                    0,
                    ProjectionTensorRecord::LAYOUT_DENSE_ROW_MAJOR,
                    ProjectionTensorRecord::QUANTIZATION_NONE,
                ),
            }
        } else {
            (
                0,
                0,
                ProjectionTensorRecord::LAYOUT_DENSE_ROW_MAJOR,
                ProjectionTensorRecord::QUANTIZATION_NONE,
            )
        };
        let record = ProjectionTensorRecord {
            logical_name_hash: stable_name_hash(&tensor.name),
            role: projection_role_for_name(&tensor.name) as u16,
            dtype: dtype_code(&tensor.storage_dtype),
            weight_offset: start_offset,
            weight_length: bytes.len() as u64,
            scale_offset,
            scale_length,
            input_width: tensor.logical_shape.get(1).copied().unwrap_or(0),
            output_width: tensor.logical_shape.first().copied().unwrap_or(0),
            rank: tensor.logical_shape.len() as u8,
            layout: layout_code,
            quantization_kind,
            flags: 0,
            dims: dims4(&tensor.logical_shape),
        };
        match class {
            MultimodalClass::Image => image_records.push(record),
            MultimodalClass::Audio => audio_records.push(record),
        }
    }

    if projection_weights.is_empty()
        && position_embeddings.is_empty()
        && auxiliary_weights.is_empty()
    {
        return Ok(None);
    }

    let desc_size = std::mem::size_of::<MultimodalInputDescriptorV1>() as u64;
    let image_offset = desc_size;
    let audio_offset =
        image_offset + (image_records.len() * std::mem::size_of::<ProjectionTensorRecord>()) as u64;
    let mut desc = MultimodalInputDescriptorV1::default();
    desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
    desc.version = 1;
    desc.modality_mask = 0b0001
        | if !image_records.is_empty() { 0b0010 } else { 0 }
        | if !audio_records.is_empty() { 0b0100 } else { 0 };
    desc.decoder_hidden_size = manifest.architecture.hidden_size;
    desc.vocabulary_size = manifest.architecture.vocab_size;
    if let Some(vision) = &manifest.vision_config {
        desc.image_patch_size = vision.patch_size.min(u16::MAX as u32) as u16;
        desc.image_channels = vision.num_channels.min(u16::MAX as u32) as u16;
        desc.image_position_embedding_width = vision.hidden_size;
    }
    desc.image_projection_table_offset = image_offset;
    desc.image_projection_count = image_records.len().min(u32::MAX as usize) as u32;
    desc.audio_projection_table_offset = audio_offset;
    desc.audio_projection_count = audio_records.len().min(u32::MAX as usize) as u32;
    desc.processor_contract_digest = Sha256::digest(&projection_weights).into();
    let mut layout_hasher = Sha256::new();
    for record in image_records.iter().chain(audio_records.iter()) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                record as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        };
        layout_hasher.update(bytes);
    }
    desc.tensor_layout_digest = layout_hasher.finalize().into();

    let mut descriptor = Vec::with_capacity(
        std::mem::size_of::<MultimodalInputDescriptorV1>()
            + (image_records.len() + audio_records.len())
                * std::mem::size_of::<ProjectionTensorRecord>(),
    );
    descriptor.extend_from_slice(unsafe {
        std::slice::from_raw_parts(
            &desc as *const MultimodalInputDescriptorV1 as *const u8,
            std::mem::size_of::<MultimodalInputDescriptorV1>(),
        )
    });
    for record in image_records.iter().chain(audio_records.iter()) {
        descriptor.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                record as *const ProjectionTensorRecord as *const u8,
                std::mem::size_of::<ProjectionTensorRecord>(),
            )
        });
    }

    Ok(Some(SynthesizedMultimodalSegments {
        projection_weights,
        projection_scales,
        // The directory-manifest schema has no bias tensor ids (its quant
        // descriptor predates the bias ABI) — artifacts packed through this
        // path stay zero-bias v1-compat, records keep flags == 0.
        projection_biases: Vec::new(),
        descriptor,
        position_embeddings,
        auxiliary_weights,
    }))
}

#[derive(Clone, Copy)]
enum MultimodalClass {
    Image,
    Audio,
}

#[derive(Clone, Copy)]
enum MultimodalEntryKind {
    ProjectionWeight,
    PositionEmbedding,
    Auxiliary,
}

fn classify_multimodal_tensor(name: &str) -> Option<MultimodalClass> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("vision_encoder")
        || lower.contains("vision_embedder")
        || lower.contains("embed_vision")
    {
        return Some(MultimodalClass::Image);
    }
    if lower.contains("audio_encoder") || lower.contains("embed_audio") {
        return Some(MultimodalClass::Audio);
    }
    None
}

fn classify_multimodal_entry(name: &str, logical_shape: &[u32]) -> MultimodalEntryKind {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".scales") || lower.ends_with(".biases") {
        return MultimodalEntryKind::Auxiliary;
    }
    if lower.contains("pos_embedding") || lower.contains("position_embed") {
        return MultimodalEntryKind::PositionEmbedding;
    }
    if logical_shape.len() >= 2
        && (lower.contains("projection")
            || lower.contains("patch_dense")
            || lower.contains("patch_embed"))
    {
        return MultimodalEntryKind::ProjectionWeight;
    }
    MultimodalEntryKind::Auxiliary
}

fn projection_role_for_name(name: &str) -> ProjectionRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("patch_dense") || lower.contains("patch_embed") {
        ProjectionRole::ImagePatchEmbedding
    } else if lower.contains("embed_vision") || lower.contains("vision") {
        ProjectionRole::ImageProjection
    } else if lower.contains("embed_audio") || lower.contains("audio") {
        ProjectionRole::AudioProjection
    } else {
        ProjectionRole::ImageProjection
    }
}

fn dims4(shape: &[u32]) -> [u32; 4] {
    let mut dims = [0u32; 4];
    for (idx, dim) in shape.iter().take(4).enumerate() {
        dims[idx] = *dim;
    }
    dims
}

fn dtype_code(dtype: &str) -> u16 {
    match dtype {
        "F16" | "Float16" => 1,
        "BF16" | "BFloat16" => 2,
        "F32" | "Float32" => 3,
        "U8" | "Uint8" => 4,
        "U32" | "Uint32" => 5,
        _ => 0,
    }
}

fn stable_name_hash(name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

fn read_tensor_payload(path: &Path, offset: u64, byte_length: u64) -> std::io::Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let end = offset.saturating_add(byte_length);
    if end > file_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "tensor slice {}..{} exceeds segment {} length {}",
                offset,
                end,
                path.display(),
                file_len
            ),
        ));
    }
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; byte_length as usize];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_json_if_present(path: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn header_fields_from_manifest(manifest: Option<&Manifest>) -> (u32, u32, u32, u32, u32, u32, u32) {
    if let Some(manifest) = manifest {
        (
            manifest.architecture.num_hidden_layers,
            manifest.architecture.num_attention_heads,
            manifest.architecture.head_dim,
            manifest.architecture.hidden_size,
            manifest.architecture.intermediate_size,
            manifest.architecture.vocab_size,
            manifest
                .execution_plan
                .speculative_config
                .as_ref()
                .map(|draft| draft.draft_architecture.num_hidden_layers)
                .unwrap_or(0),
        )
    } else {
        (0, 0, 0, 0, 0, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::compute_image::manifest::{
        CompileReadiness, Nf4Tile640Layout, QuantizationDesc, QuantizationQualityStatus,
        ResidencyPlan, Segment, SegmentKind as ManifestSegmentKind, ShardHash, SharedWeightLayout,
        SourceIdentity, TensorEntry,
    };
    use prism_ecs_constitutional::config::{
        AudioArchitecture, GenerationRegime, LayerPlan, ModelExecutionPlan, RopeSpec,
        TextArchitecture, VisionArchitecture,
    };

    fn test_manifest() -> Manifest {
        Manifest {
            image_version: "0.1.0".into(),
            compiler_version: "test".into(),
            runtime_abi: "test".into(),
            hardware_target: None,
            readiness: Some(CompileReadiness::Ready),
            compile_date: String::new(),
            compile_host: String::new(),
            source: SourceIdentity {
                config_hash: String::new(),
                shard_hashes: Vec::<ShardHash>::new(),
                tokenizer_hashes: Vec::<ShardHash>::new(),
                auxiliary_hashes: Vec::<ShardHash>::new(),
                model_type: "gemma4".into(),
                quantization_bits: 2,
                quantization_group_size: 64,
                quantization_mode: "ternary".into(),
            },
            architecture: TextArchitecture {
                hidden_size: 3840,
                intermediate_size: 15360,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 256,
                global_head_dim: None,
                num_global_key_value_heads: None,
                num_hidden_layers: 48,
                vocab_size: 262144,
                sliding_window: 1024,
                max_position_embeddings: 131072,
                rms_norm_eps: 1e-6,
                final_logit_softcapping: None,
                hidden_size_per_layer_input: 0,
                layer_types: Vec::new(),
                rope_local: RopeSpec {
                    theta: 10000.0,
                    rope_type: "default".into(),
                    partial_rotary_factor: None,
                },
                rope_global: None,
                attention_k_eq_v: false,
                tie_word_embeddings: true,
                model_type: "gemma4".into(),
                moe_config: None,
                diffusion_config: None,
                thinking_mode: false,
            },
            vision_config: Some(VisionArchitecture {
                hidden_size: 1152,
                num_attention_heads: 16,
                num_hidden_layers: 27,
                intermediate_size: 4304,
                image_size: 896,
                patch_size: 14,
                num_channels: 3,
                projection_dim: 256,
                model_family: "gemma4_unified".into(),
                has_ane_program: false,
            }),
            audio_config: Some(AudioArchitecture {
                hidden_size: 1024,
                num_attention_heads: 8,
                num_hidden_layers: 12,
                intermediate_size: 4096,
                sample_rate: 16000,
                num_mel_bins: 80,
                hop_length: 160,
                max_audio_length_s: 30,
                projection_dim: 256,
            }),
            segments: Vec::new(),
            tensor_table: Vec::new(),
            alias_table: Vec::new(),
            residency_plan: ResidencyPlan {
                persistent_segments: Vec::new(),
                layer_segments: Vec::new(),
                layer_window_size: 2,
                total_bytes: 0,
            },
            image_hash: String::new(),
            required_storage_abi: "copied-v0".into(),
            required_capabilities: Vec::new(),
            prepacked_layout: "none".into(),
            metallib_hash: None,
            metallib_size: None,
            metal_kernel_artifacts: Vec::new(),
            execution_plan: ModelExecutionPlan {
                prologue: Default::default(),
                layers: vec![LayerPlan {
                    layer_index: 0,
                    attention_kind: "full_attention".into(),
                    segment_id: "layer_0".into(),
                    hidden_size: 3840,
                    n_heads: 16,
                    n_kv_heads: 8,
                    head_dim: 256,
                    global_head_dim: None,
                    n_global_kv_heads: None,
                    sliding_window: 0,
                    rope_theta: 10000.0,
                    partial_rotary_factor: None,
                    attention_k_eq_v: false,
                    q_norm_enabled: false,
                    k_norm_enabled: false,
                    q_proj_tensor_id: 0,
                    k_proj_tensor_id: 0,
                    v_proj_tensor_id: 0,
                    o_proj_tensor_id: 0,
                    q_norm_tensor_id: None,
                    k_norm_tensor_id: None,
                    gate_proj_tensor_id: 0,
                    up_proj_tensor_id: 0,
                    down_proj_tensor_id: 0,
                    input_layernorm_tensor_id: 0,
                    post_attention_layernorm_tensor_id: 0,
                    pre_ffw_layernorm_tensor_id: None,
                    post_ffw_layernorm_tensor_id: None,
                    layer_scalar_ids: Vec::new(),
                    quantization_ids: Vec::new(),
                    route: Default::default(),
                    fused_operations: Vec::new(),
                }],
                epilogue: Default::default(),
                fused_ane_islands: Vec::new(),
                hidden_size: 3840,
                vocab_size: 262144,
                sliding_window: 0,
                final_logit_softcapping: None,
                tie_word_embeddings: true,
                rms_norm_eps: 1e-6,
                speculative_config: None,
                generation_regime: GenerationRegime::Autoregressive,
                diffusion_config: None,
                diffusion_execution_plan: None,
                kv_cache_mode: Default::default(),
            },
            phase_dag: None,
            compatibility_receipt: None,
            quantization_profiles: Vec::new(),
            quantization_quality: Vec::new(),
            quantization_quality_status: QuantizationQualityStatus::Unknown,
        }
    }

    #[test]
    fn synthesized_execution_graph_reflects_multimodal_manifest() {
        let bytes = synthesize_execution_graph(&test_manifest()).expect("execution graph");
        let graph = ExecutionGraphDescriptor::from_bytes(&bytes).expect("decode graph");
        assert!(graph
            .layers
            .iter()
            .any(|node| node.node_kind == NodeKind::VisionPatchEmbed as u8));
        assert!(graph
            .layers
            .iter()
            .any(|node| node.node_kind == NodeKind::AudioProjection as u8));
        assert_eq!(graph.num_layers, 48);
    }

    #[test]
    fn synthesized_model_artifacts_include_multimodal_token_map() {
        let dir = tempfile::tempdir().expect("tempdir");
        let artifacts =
            synthesize_model_artifacts(dir.path(), Some(&test_manifest())).expect("artifacts");
        let entries: Vec<(u32, &[u8])> = ModelArtifactEntry::iter_entries(&artifacts).collect();
        let token_map = entries
            .iter()
            .find(|(tag, _)| *tag == model_artifact_tag::TOKEN_MAP)
            .map(|(_, bytes)| {
                serde_json::from_slice::<serde_json::Value>(bytes).expect("token map")
            })
            .expect("token map entry");
        assert!(token_map.get("image_start_token").is_some());
        assert!(token_map.get("audio_start_token").is_some());
        assert_eq!(
            token_map
                .get("audio_sample_rate")
                .and_then(|value| value.as_u64()),
            Some(16000)
        );
    }

    #[test]
    fn synthesized_multimodal_segments_preserve_nf4_tile640_scale_abi() {
        let dir = tempfile::tempdir().expect("tempdir");
        let layout = Nf4Tile640Layout::canonical();
        let out_dim = 2u32;
        let input_width = layout.tile_elements;
        let weight_len = u64::from(layout.packed_row_bytes(input_width)) * u64::from(out_dim);
        let scale_len = u64::from(layout.metadata_row_values(input_width)) * u64::from(out_dim) * 4;

        let weight_bytes = vec![0xABu8; weight_len as usize];
        let scale_bytes = vec![0xCDu8; scale_len as usize];
        let segment_path = dir.path().join("vision_segment.bin");
        let mut segment_bytes = weight_bytes.clone();
        segment_bytes.extend_from_slice(&scale_bytes);
        std::fs::write(&segment_path, &segment_bytes).expect("write segment");

        let mut manifest = test_manifest();
        manifest.segments = vec![Segment {
            id: "vision_encoder".into(),
            filename: "vision_segment.bin".into(),
            byte_size: segment_bytes.len() as u64,
            sha256: String::new(),
            tensor_ids: vec![1, 2],
            kind: ManifestSegmentKind::Persistent,
            alignment_bytes: 4096,
        }];
        manifest.tensor_table = vec![
            TensorEntry {
                id: 1,
                name: "vision_encoder.patch_dense.weight".into(),
                role: "weight".into(),
                layer: None,
                segment: "vision_encoder".into(),
                source_filename: String::new(),
                source_sha256: String::new(),
                source_offset: 0,
                offset: 0,
                byte_length: weight_len,
                logical_dtype: "F32".into(),
                storage_dtype: "U8".into(),
                logical_shape: vec![out_dim, input_width],
                physical_shape: vec![out_dim, layout.packed_row_bytes(input_width)],
                mutability: "immutable".into(),
                quantization: Some(QuantizationDesc {
                    bits: 4,
                    group_size: layout.quant_group_size,
                    groups: out_dim * layout.metadata_row_values(input_width),
                    scale_tensor_id: 2,
                    bias_tensor_id: 0,
                    storage_layout: Some(SharedWeightLayout::Nf4Tile640(layout.clone())),
                }),
                tensor_alignment_bytes: 16,
                layout_version: 1,
                artifact_bindings: HashMap::new(),
            },
            TensorEntry {
                id: 2,
                name: "vision_encoder.patch_dense.scales".into(),
                role: "weight::scales".into(),
                layer: None,
                segment: "vision_encoder".into(),
                source_filename: String::new(),
                source_sha256: String::new(),
                source_offset: 0,
                offset: weight_len,
                byte_length: scale_len,
                logical_dtype: "F32".into(),
                storage_dtype: "F32".into(),
                logical_shape: vec![out_dim, layout.metadata_row_values(input_width)],
                physical_shape: vec![out_dim, layout.metadata_row_values(input_width)],
                mutability: "immutable".into(),
                quantization: None,
                tensor_alignment_bytes: 16,
                layout_version: 1,
                artifact_bindings: HashMap::new(),
            },
        ];

        let synthesized = synthesize_multimodal_segments(dir.path(), Some(&manifest))
            .expect("synthesize")
            .expect("segments");
        assert_eq!(synthesized.projection_weights, weight_bytes);
        assert_eq!(synthesized.projection_scales, scale_bytes);

        let desc_size = std::mem::size_of::<MultimodalInputDescriptorV1>();
        let record_size = std::mem::size_of::<ProjectionTensorRecord>();
        assert!(synthesized.descriptor.len() >= desc_size + record_size);
        let record = unsafe {
            std::ptr::read_unaligned(
                synthesized.descriptor[desc_size..].as_ptr() as *const ProjectionTensorRecord
            )
        };
        assert!(record.is_nf4_tile640());
        assert_eq!(record.weight_length, weight_len);
        assert_eq!(record.scale_offset, 0);
        assert_eq!(record.scale_length, scale_len);
        record.validate_nf4_tile640().expect("valid nf4 record");
    }

    #[test]
    fn execution_graph_multimodal_nodes_pick_up_descriptor_offsets() {
        let mut graph = ExecutionGraphDescriptor {
            magic: crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers: 0,
            num_draft_layers: 0,
            num_compaction_epochs: 0,
            node_count: 2,
            _pad: [0; 2],
            layers: vec![
                LayerExecutionNode {
                    node_kind: NodeKind::VisionPatchEmbed as u8,
                    attention_kind: 2,
                    device_capability: DeviceCapability::Gpu as u8,
                    compaction_epoch: 0xFF,
                    layer_index: 0,
                    head_dim: 14,
                    num_heads: 8,
                    hidden_dim: 0,
                    weight_offset: 0,
                    weight_length: 0,
                    scale_offset: 0,
                    _reserved: [0; 8],
                },
                LayerExecutionNode {
                    node_kind: NodeKind::AudioProjection as u8,
                    attention_kind: 2,
                    device_capability: DeviceCapability::Gpu as u8,
                    compaction_epoch: 0xFF,
                    layer_index: 0,
                    head_dim: 256,
                    num_heads: 8,
                    hidden_dim: 0,
                    weight_offset: 0,
                    weight_length: 0,
                    scale_offset: 0,
                    _reserved: [0; 8],
                },
            ],
            compaction_epochs: Vec::new(),
            draft_sub_graph: None,
        };

        let mut desc = MultimodalInputDescriptorV1::default();
        desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
        desc.version = 1;
        desc.image_projection_count = 1;
        desc.audio_projection_count = 1;
        let image = ProjectionTensorRecord {
            role: ProjectionRole::ImagePatchEmbedding as u16,
            weight_offset: 128,
            weight_length: 640,
            scale_offset: 64,
            scale_length: 40,
            output_width: 1152,
            layout: ProjectionTensorRecord::LAYOUT_NF4_TILE640,
            quantization_kind: ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
            ..ProjectionTensorRecord::default()
        };
        let audio = ProjectionTensorRecord {
            role: ProjectionRole::AudioProjection as u16,
            weight_offset: 2048,
            weight_length: 4096,
            scale_offset: 512,
            scale_length: 160,
            output_width: 3840,
            layout: ProjectionTensorRecord::LAYOUT_NF4_TILE640,
            quantization_kind: ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
            ..ProjectionTensorRecord::default()
        };
        let mut descriptor = Vec::new();
        descriptor.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                &desc as *const MultimodalInputDescriptorV1 as *const u8,
                std::mem::size_of::<MultimodalInputDescriptorV1>(),
            )
        });
        for record in [image, audio] {
            descriptor.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &record as *const ProjectionTensorRecord as *const u8,
                    std::mem::size_of::<ProjectionTensorRecord>(),
                )
            });
        }

        let mut graph_bytes = graph.to_bytes();
        assert!(patch_execution_graph_multimodal_nodes(
            graph_bytes.as_mut_slice(),
            &descriptor
        ));
        graph = ExecutionGraphDescriptor::from_bytes(&graph_bytes).expect("patched graph");
        assert_eq!(graph.layers[0].weight_offset, 128);
        assert_eq!(graph.layers[0].weight_length, 640);
        assert_eq!(graph.layers[0].scale_offset, 64);
        assert_eq!(graph.layers[0].hidden_dim, 1152);
        assert_eq!(graph.layers[1].weight_offset, 2048);
        assert_eq!(graph.layers[1].weight_length, 4096);
        assert_eq!(graph.layers[1].scale_offset, 512);
        assert_eq!(graph.layers[1].hidden_dim, 3840);
    }
}
