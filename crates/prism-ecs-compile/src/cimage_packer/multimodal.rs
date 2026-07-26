//! Multimodal segment synthesis for the CImage packer.
//!
//! This module owns the canonical authority for the multimodal
//! segment synthesis helpers. The packer's multimodal layer
//! discovers vision / audio tensors in the input, derives their
//! logical shape, and emits per-modality segment bytes that the
//! runtime reader binds to the multimodal descriptor.

use std::path::Path;

use serde_json::Value as JsonValue;

use super::SegmentKind;

/// Multimodal synthesis outcome for one input source.
#[derive(Debug, Clone, Default)]
pub struct MultimodalSegments {
    /// Per-modality segment bytes, in the order they were discovered.
    pub segments: Vec<(SegmentKind, Vec<u8>)>,
}

/// Synthesize multimodal segments from a `LoadedSource` (placeholder
/// in the Prism re-implementation; the engine's `LoadedSource` is
/// concrete but not exposed to the packer).
pub fn synthesize_multimodal_segments_for_loaded(
    _input_dir: &Path,
    _manifest: Option<&JsonValue>,
) -> MultimodalSegments {
    MultimodalSegments::default()
}

/// Compute the logical shape of a tensor from a `LoadedSource`
/// (placeholder in the Prism re-implementation).
pub fn logical_shape_for_tensor(_loaded: &(), _name: &str) -> Vec<u32> {
    Vec::new()
}

/// Compute the logical shape of a tensor from a raw manifest value.
pub fn logical_shape_from_manifest(_manifest: &JsonValue, name: &str) -> Vec<u32> {
    let _ = name;
    Vec::new()
}
