//! Multimodal binding — pure data types for the sealed segment
//! binding surface.

use serde::{Deserialize, Serialize};

use super::descriptor::ProjectionTensorRecord;

/// A sealed segment binding between a multimodal projection tensor and
/// a content-store segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedSegmentBinding {
    /// Segment identifier.
    pub segment_id: String,
    /// Tensor record.
    pub tensor: ProjectionTensorRecord,
    /// Whether the binding is sealed (immutable).
    pub sealed: bool,
}
