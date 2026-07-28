//! Content-addressed index — pure data types and pure algorithms for
//! the content store index.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;

/// Opaque identifier for a content object.
pub type ContentObjectId = String;

/// Opaque identifier for a segment.
pub type SegmentId = String;

/// Opaque identifier for a target layout.
pub type TargetLayoutId = String;

/// Tensor shape as a list of dimensions.
pub type TensorShape = Vec<i64>;

/// Tensor strides as a list of strides.
pub type TensorStrides = Vec<i64>;

/// Tensor data type as a string (e.g., "F32", "I8").
pub type TensorDType = String;

/// Content-addressed content store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContentAddressedContentStore {
    /// Format version of the store.
    pub version: ContentStoreVersion,
    /// Immutable segments in the store.
    pub segments: Vec<ImmutableSegment>,
    /// Index of content objects.
    pub objects: Vec<ContentObjectEntry>,
}

/// Version of the content store format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStoreVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl Default for ContentStoreVersion {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

/// An immutable segment in the content store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImmutableSegment {
    /// Segment identifier.
    pub segment_id: SegmentId,
    /// Byte size of the segment.
    pub byte_size: u64,
    /// Content hash of the segment.
    pub content_hash: ContentHash,
}

/// A content object entry in the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentObjectEntry {
    /// Object identifier.
    pub object_id: ContentObjectId,
    /// Segment that stores this object.
    pub segment_id: SegmentId,
    /// Kind of object.
    pub kind: ContentObjectKind,
}

/// Kind of content object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentObjectKind {
    /// A canonical weight tensor.
    CanonicalWeight,
    /// A tokenizer asset.
    Tokenizer,
    /// A configuration file.
    Config,
    /// A generic binary asset.
    Generic,
}

/// Reference to a consumer of an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConsumerRef {
    /// Consumer identifier (e.g., profile_id, layer_index).
    pub consumer_id: String,
    /// Object id consumed.
    pub object_id: ContentObjectId,
    /// Residency class for this consumer.
    pub residency_class: crate::compute_image_runtime::residency::plan::ResidencyClass,
}
