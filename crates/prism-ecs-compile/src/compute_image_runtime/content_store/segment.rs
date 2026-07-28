//! Content segment descriptors — pure data types for the content
//! store's segment records.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;

/// Opaque identifier for a content segment.
pub type ContentSegmentId = String;

/// A content segment record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSegment {
    /// Segment identifier.
    pub segment_id: ContentSegmentId,
    /// Content hash of the segment bytes.
    pub content_hash: ContentHash,
    /// Byte size of the segment.
    pub byte_size: u64,
    /// Offset within the underlying file (0 for top-level segments).
    pub file_offset: u64,
}
