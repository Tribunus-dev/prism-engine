//! Memory-mapped segment loading — pure data types and pure algorithms.

use serde::{Deserialize, Serialize};

/// A memory-mapped segment.
#[derive(Debug, Clone)]
pub struct MappedSegment {
    /// Segment identifier.
    pub segment_id: String,
    /// Byte size of the mapped region.
    pub byte_size: u64,
    /// Whether the mapping is read-only.
    pub read_only: bool,
}

/// A memory-mapped region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmapRegion {
    /// Region identifier.
    pub region_id: String,
    /// Byte offset within the underlying file.
    pub file_offset: u64,
    /// Byte size of the region.
    pub byte_size: u64,
}

/// Error type for mmap loading failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MmapLoadError {
    /// File not found.
    FileNotFound {
        /// Path of the file.
        path: String,
    },
    /// Read failed.
    ReadFailed {
        /// Path of the file.
        path: String,
        /// Reason for the failure.
        reason: String,
    },
    /// Size mismatch.
    SizeMismatch {
        /// Expected size.
        expected: u64,
        /// Actual size.
        actual: u64,
    },
}

/// Memory-mapped loader.
#[derive(Debug, Clone, Default)]
pub struct MmapLoader;

impl MmapLoader {
    /// Create a new mmap loader.
    pub fn new() -> Self {
        Self
    }

    /// Map a file at the given path.
    pub fn map(&self, _path: &std::path::Path) -> Result<MappedSegment, MmapLoadError> {
        // Stub: the actual implementation depends on `memmap2` and
        // would platform-dispatch the right call. The data-only
        // contract lives in [`MappedSegment`] / [`MmapRegion`] /
        /// [`MmapLoadError`].
        Err(MmapLoadError::FileNotFound {
            path: "<stub>".to_string(),
        })
    }
}
