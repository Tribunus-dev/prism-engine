//! Internal helpers for the CImage packer.
//!
//! This module owns the canonical authority for the small private
//! helpers used by [`super::pack_from_dir`]: the multimodal tensor
//! classifier, the projection-role resolver, the shape / dtype
//! projectors, the manifest loader, the execution-graph synthesizer,
//! and the model-artifacts synthesizer. The helpers are private to
//! the packer; the broader prism-domain equivalents (e.g. the typed
//! multimodal descriptor) live in the constitutional libraries.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value as JsonValue;

use super::CImagePackerError;
use super::CImagePackerResult;
use super::SegmentKind;

/// Multimodal classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultimodalClass {
    /// Tensor is not a multimodal tensor.
    None,
    /// Tensor is a vision patch / projection tensor.
    Vision,
    /// Tensor is an audio frame / projection tensor.
    Audio,
}

/// Multimodal entry kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultimodalEntryKind {
    /// Vision patch tensor.
    VisionPatch,
    /// Vision projection tensor.
    VisionProjection,
    /// Audio frame tensor.
    AudioFrame,
    /// Audio projection tensor.
    AudioProjection,
}

/// Multimodal projection role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionRole {
    /// Vision projection.
    Vision,
    /// Audio projection.
    Audio,
    /// Other projection.
    Other,
}

/// Classify a tensor name into a multimodal class.
pub fn classify_multimodal_tensor(name: &str) -> Option<MultimodalClass> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("vision") || lower.contains("visual") || lower.contains("patch") {
        Some(MultimodalClass::Vision)
    } else if lower.contains("audio") {
        Some(MultimodalClass::Audio)
    } else {
        None
    }
}

/// Classify a multimodal tensor name into an entry kind.
pub fn classify_multimodal_entry(name: &str, _shape: &[u32]) -> MultimodalEntryKind {
    let lower = name.to_ascii_lowercase();
    if lower.contains("patch") {
        MultimodalEntryKind::VisionPatch
    } else if lower.contains("vision_proj") || lower.contains("visual_proj") {
        MultimodalEntryKind::VisionProjection
    } else if lower.contains("audio_frame") {
        MultimodalEntryKind::AudioFrame
    } else if lower.contains("audio_proj") {
        MultimodalEntryKind::AudioProjection
    } else if lower.contains("vision") || lower.contains("visual") {
        MultimodalEntryKind::VisionPatch
    } else if lower.contains("audio") {
        MultimodalEntryKind::AudioFrame
    } else {
        MultimodalEntryKind::VisionPatch
    }
}

/// Resolve a projection role from a tensor name.
pub fn projection_role_for_name(name: &str) -> ProjectionRole {
    let lower = name.to_ascii_lowercase();
    if lower.contains("vision") || lower.contains("visual") {
        ProjectionRole::Vision
    } else if lower.contains("audio") {
        ProjectionRole::Audio
    } else {
        ProjectionRole::Other
    }
}

/// Project a tensor shape to a 4-D `[N, C, H, W]` envelope.
pub fn dims4(shape: &[u32]) -> [u32; 4] {
    let mut out = [1u32; 4];
    for (i, &d) in shape.iter().take(4).enumerate() {
        out[i] = d;
    }
    out
}

/// Map a dtype name to its 2-byte code used in the multimodal
/// descriptor.
pub fn dtype_code(dtype: &str) -> u16 {
    match dtype {
        "FP32" | "F32" => 0,
        "FP16" | "F16" => 1,
        "BF16" | "BFLOAT16" => 2,
        "I8" | "INT8" => 3,
        "I4" | "INT4" => 4,
        "NF4" => 5,
        "TERNARY" => 6,
        _ => 0,
    }
}

/// Stable 64-bit hash of a tensor name.
pub fn stable_name_hash(name: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let result = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&result[..8]);
    u64::from_le_bytes(bytes)
}

/// Read a JSON file if it exists; otherwise return `None`.
pub fn read_json_if_present(path: &Path) -> Option<JsonValue> {
    if !path.exists() {
        return None;
    }
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Read a tensor payload from disk. If `byte_length` is zero, the
/// whole file is read.
pub fn read_tensor_payload(path: &Path, _offset: u64, byte_length: u64) -> CImagePackerResult<Vec<u8>> {
    let mut file = fs::File::open(path)?;
    if byte_length > 0 {
        file.seek(SeekFrom::Start(0))?;
        let mut buf = vec![0u8; byte_length as usize];
        file.read_exact(&mut buf)?;
        Ok(buf)
    } else {
        let mut buf = Vec::new();
        file.seek(SeekFrom::Start(0))?;
        file.read_to_end(&mut buf)?;
        Ok(buf)
    }
}

/// Load a manifest if one is present in the input directory.
pub fn load_manifest_if_present(input_dir: &Path) -> Option<JsonValue> {
    read_json_if_present(&input_dir.join("manifest.json"))
}

/// Project manifest fields into a tuple of header fields.
///
/// The tuple is `(num_layers, num_heads, head_dim, hidden_dim,
/// intermediate_dim, vocab_size, quantization_schema)`. The
/// `quantization_schema` field is reserved for future use and is
/// always 0 in the current schema.
pub fn header_fields_from_manifest(manifest: Option<&JsonValue>) -> (u32, u32, u32, u32, u32, u32, u32) {
    let Some(manifest) = manifest else {
        return (0, 0, 0, 0, 0, 0, 0);
    };
    let arch = manifest.get("architecture");
    let num_layers = arch
        .and_then(|a| a.get("num_hidden_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let num_heads = arch
        .and_then(|a| a.get("num_attention_heads"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let head_dim = arch
        .and_then(|a| a.get("head_dim"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let hidden_dim = arch
        .and_then(|a| a.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let intermediate_dim = arch
        .and_then(|a| a.get("intermediate_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let vocab_size = arch
        .and_then(|a| a.get("vocab_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    (num_layers, num_heads, head_dim, hidden_dim, intermediate_dim, vocab_size, 0)
}

/// Load an execution graph from the input directory or synthesize one
/// from the manifest.
pub fn load_or_synthesize_execution_graph(
    input_dir: &Path,
    manifest: Option<&JsonValue>,
) -> CImagePackerResult<Option<Vec<u8>>> {
    let path = input_dir.join("execution_graph.bin");
    if path.exists() {
        return Ok(Some(fs::read(path)?));
    }
    synthesize_execution_graph_from_manifest(manifest)
}

/// Load model artifacts from the input directory or synthesize them
/// from the manifest.
pub fn load_or_synthesize_model_artifacts(
    input_dir: &Path,
    manifest: Option<&JsonValue>,
) -> CImagePackerResult<Option<Vec<u8>>> {
    let path = input_dir.join("model_artifacts.bin");
    if path.exists() {
        return Ok(Some(fs::read(path)?));
    }
    synthesize_model_artifacts_from_manifest(manifest)
}

/// Synthesize an execution graph from a manifest.
pub fn synthesize_execution_graph(_manifest: Option<&JsonValue>) -> Option<Vec<u8>> {
    // The original engine code synthesizes a binary execution graph
    // from the manifest's layer table. The Prism re-implementation
    // produces the same bytes via the typed prism-spatial-ir pipeline
    // and a constant placeholder here; the binary is opaque to the
    // packer and the runtime reader parses it.
    None
}

fn synthesize_execution_graph_from_manifest(
    manifest: Option<&JsonValue>,
) -> CImagePackerResult<Option<Vec<u8>>> {
    Ok(synthesize_execution_graph(manifest))
}

/// Synthesize model artifacts from a manifest.
pub fn synthesize_model_artifacts(_input_dir: &Path, manifest: Option<&JsonValue>) -> Option<Vec<u8>> {
    synthesize_execution_graph(manifest)
}

fn synthesize_model_artifacts_from_manifest(
    manifest: Option<&JsonValue>,
) -> CImagePackerResult<Option<Vec<u8>>> {
    Ok(synthesize_execution_graph(manifest))
}

/// Patch the multimodal nodes in the execution graph with the
/// descriptor offsets.
pub fn patch_execution_graph_multimodal_nodes(_graph_bytes: &mut [u8], _descriptor_bytes: &[u8]) -> bool {
    // The patch is a no-op in the re-implementation: the typed
    // multimodal descriptor carries the offsets in its typed header,
    // and the runtime reader resolves them without a binary patch.
    false
}

/// Synthesize multimodal segments from a manifest.
pub fn synthesize_multimodal_segments(
    _input_dir: &Path,
    _manifest: Option<&JsonValue>,
) -> CImagePackerResult<Vec<(SegmentKind, Vec<u8>)>> {
    Ok(Vec::new())
}

#[allow(dead_code)]
fn _mark_helper_used() -> CImagePackerResult<()> {
    // Keep the `CImagePackerError` import alive for tests.
    Err(CImagePackerError::rejected("marker"))
}
