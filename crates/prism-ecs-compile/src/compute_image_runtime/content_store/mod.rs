//! Content-addressed store — pure data types and pure algorithms for
//! the runtime content store.

pub mod aliases;
pub mod index;
pub mod integrity;
pub mod layout;
pub mod mmap;
pub mod packing;
pub mod segment;

pub use aliases::AliasEntry;
pub use index::{
    ArtifactConsumerRef, ContentAddressedContentStore, ContentObjectEntry, ContentObjectId,
    ContentObjectKind, ContentStoreVersion, ImmutableSegment, SegmentId, TargetLayoutId,
    TensorDType, TensorShape, TensorStrides,
};
pub use integrity::IntegrityVerifier;
pub use mmap::{MappedSegment, MmapLoadError, MmapLoader, MmapRegion};
pub use packing::{InterleaveConfig, PackingPolicy, PackingResult, PaddingMode};
pub use segment::{ContentSegment, ContentSegmentId};
