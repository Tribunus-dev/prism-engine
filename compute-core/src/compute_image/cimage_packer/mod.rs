//! V3 .cimage packer — AOT layout, AlignedMmapBuilder, direct GPU write.
//!
//! Replaces the old `ImageBuilder` vector accumulation and
//! `DeterministicSegmentWriter` with a pre-allocated mmap engine.
//!
//! Pipeline:
//!   1. predict_tar_size() → exact .mlmodelc archive sizes
//!   2. CImageLayoutPlan::calculate() → all offsets known AOT
//!   3. ftruncate + mmap at total_file_size
//!   4. AlignedMmapBuilder slices the mmap — no Vec allocations
//!   5. GPU writes weights directly into mmap via newBufferWithBytesNoCopy
//!   6. archive_mlmodelc_to_mmap writes .mlmodelc into mmap
//!   7. CImageHeader written at offset 0

pub mod archive;
pub mod builder;
pub mod layout;
pub mod pipeline;

pub use archive::*;
pub use builder::*;
pub use layout::*;
pub use pipeline::*;

pub const APPLE_PAGE_SIZE: usize = 16_384;
