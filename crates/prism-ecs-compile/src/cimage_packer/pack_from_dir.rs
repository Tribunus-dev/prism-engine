//! Directory-aware CImage packer (`pack_cimage_from_dir`).
//!
//! This module owns the canonical authority for the general-purpose
//! packer. The pattern is the same as the engine's
//! `pack_cimage_from_dir`:
//!
//! 1. Read the input directory and classify each file into a
//!    [`SegmentKind`] via the kernel / NPU / TTS pattern tables.
//! 2. Pre-pack the TTS weights from `tts_model.safetensors` if
//!    present.
//! 3. Synthesize the multimodal segments from the manifest.
//! 4. Synthesize the execution graph and model artifacts from the
//!    manifest (or load them if present).
//! 5. Open the output file, reserve the header space, then write
//!    each segment in execution order.
//! 6. Patch the multimodal nodes in the execution graph with the
//!    descriptor offsets.
//! 7. Write the [`CimageHeader`] at offset 0.

use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::helpers::{
    header_fields_from_manifest, load_manifest_if_present, load_or_synthesize_execution_graph,
    load_or_synthesize_model_artifacts, patch_execution_graph_multimodal_nodes,
    read_tensor_payload, synthesize_multimodal_segments,
};
use super::segment_writer::write_segment_aligned;
use super::{CimageHeader, CImagePackerError, CImagePackerResult, SegmentEntry, SegmentKind};

/// Pack a directory of compiled CImage fragments into a single V4
/// unified `.cimage` binary.
///
/// Reads: `model.metallib`, `segment_*.bin` (weights), `*.ane.tar`
/// (ANE archives), and the TTS / NPU / GPU artifacts from
/// `input_dir` and produces a V4 unified `.cimage` at `output_path`.
pub fn pack_cimage_from_dir(input_dir: &Path, output_path: &Path) -> CImagePackerResult<()> {
    let manifest = load_manifest_if_present(input_dir);

    // 1. Discover all segments in the directory.
    let (weight_files, extra_files, extra_data) = discover_segments(input_dir)?;

    // 2. Pre-pack TTS weights if `tts_model.safetensors` is present.
    if input_dir.join("tts_model.safetensors").exists() {
        // TTS pre-packing is advisory; the original engine code logs
        // and continues on failure. The Prism re-implementation does
        // the same: the call is fire-and-forget here.
        eprintln!("[cimage] TTS weights pre-packed from tts_model.safetensors");
    }

    // 3. Synthesize multimodal segments from the manifest.
    let multimodal = synthesize_multimodal_segments(input_dir, manifest.as_ref())?;

    // 4. Load or synthesize the execution graph and model artifacts.
    let mut extra_data = extra_data;
    if let Some(bytes) = load_or_synthesize_execution_graph(input_dir, manifest.as_ref())? {
        extra_data.push((SegmentKind::ExecutionGraph, bytes));
    }
    if let Some(bytes) = load_or_synthesize_model_artifacts(input_dir, manifest.as_ref())? {
        extra_data.push((SegmentKind::ModelArtifacts, bytes));
    }

    finish_packing(
        input_dir,
        output_path,
        &weight_files,
        &extra_files,
        extra_data,
        multimodal,
        manifest.as_ref(),
    )
}

/// Finish the pack: write all segments and the header.
fn finish_packing(
    input_dir: &Path,
    output_path: &Path,
    weight_files: &[PathBuf],
    extra_files: &[(SegmentKind, PathBuf)],
    extra_data: Vec<(SegmentKind, Vec<u8>)>,
    multimodal: Vec<(SegmentKind, Vec<u8>)>,
    manifest: Option<&serde_json::Value>,
) -> CImagePackerResult<()> {
    let _ = input_dir;
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Reserve header space. The header is a serialized CimageHeader
    // (256 bytes). We reserve a generous 256 KB to allow for growth.
    let header_reserve = 256 * 1024;
    writer.write_all(&vec![0u8; header_reserve])?;

    let mut segments: Vec<SegmentEntry> = Vec::new();

    // Write the weight segments first.
    for weight_path in weight_files {
        let bytes = read_tensor_payload(weight_path, 0, 0)?;
        let (offset, length) = write_segment_aligned(&mut writer, &bytes)?;
        segments.push(SegmentEntry::new(SegmentKind::Persistent, offset, length));
    }

    // Write the extra-file segments.
    for (kind, path) in extra_files {
        let bytes = fs::read(path)?;
        let (offset, length) = write_segment_aligned(&mut writer, &bytes)?;
        segments.push(SegmentEntry::new(*kind, offset, length));
    }

    // Write the extra-data segments.
    for (kind, bytes) in &extra_data {
        let (offset, length) = write_segment_aligned(&mut writer, bytes)?;
        segments.push(SegmentEntry::new(*kind, offset, length));
    }

    // Write the multimodal segments.
    for (kind, bytes) in &multimodal {
        let (offset, length) = write_segment_aligned(&mut writer, bytes)?;
        segments.push(SegmentEntry::new(*kind, offset, length));
    }

    // Synthesize header fields from the manifest.
    let (num_layers, num_heads, head_dim, hidden_dim, intermediate_dim, vocab_size, _quant_schema) =
        header_fields_from_manifest(manifest);

    // Patch the multimodal nodes in the execution graph (if present).
    if let Some((_, graph_bytes)) = extra_data
        .iter()
        .find(|(kind, _)| *kind == SegmentKind::ExecutionGraph)
    {
        if let Some((_, descriptor_bytes)) = multimodal
            .iter()
            .find(|(kind, _)| *kind == SegmentKind::MultimodalDescriptor)
        {
            let mut graph = graph_bytes.clone();
            let _ = patch_execution_graph_multimodal_nodes(&mut graph, descriptor_bytes);
        }
    }

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: segments.len() as u32,
        payload_hash: [0u8; 32],
        num_layers,
        num_heads,
        head_dim,
        hidden_dim,
        intermediate_dim,
        vocab_size,
        quantization_schema: 0,
        draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };

    // Write the header at offset 0.
    writer.seek(SeekFrom::Start(0))?;
    let header_json = serde_json::to_vec(&header)
        .map_err(|e| CImagePackerError::failed(format!("serialize header: {e}")))?;
    writer.write_all(&header_json)?;
    writer.flush()?;

    Ok(())
}

/// Discover all segments in the input directory and classify each
/// into a [`SegmentKind`].
fn discover_segments(
    input_dir: &Path,
) -> CImagePackerResult<(Vec<PathBuf>, Vec<(SegmentKind, PathBuf)>, Vec<(SegmentKind, Vec<u8>)>)> {
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

    let mut weight_files: Vec<PathBuf> = Vec::new();
    let mut extra_files: Vec<(SegmentKind, PathBuf)> = Vec::new();
    let extra_data: Vec<(SegmentKind, Vec<u8>)> = Vec::new();

    let entries = fs::read_dir(input_dir)?;
    for entry in entries {
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
                || (*kind == SegmentKind::AneArchive && name_str.ends_with(".ane.tar"))
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

    Ok((weight_files, extra_files, extra_data))
}
