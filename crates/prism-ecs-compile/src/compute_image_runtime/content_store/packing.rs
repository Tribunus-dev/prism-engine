//! Packing policy — pure data types and pure algorithms for content
//! packing.

use serde::{Deserialize, Serialize};

/// Packing policy for content store entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingPolicy {
    /// Padding mode.
    pub padding: PaddingMode,
    /// Interleave configuration.
    pub interleave: InterleaveConfig,
    /// Alignment in bytes.
    pub alignment: u32,
}

/// Interleave configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterleaveConfig {
    /// Number of elements per interleave group.
    pub group_size: u32,
    /// Whether to interleave (false = contiguous).
    pub interleave: bool,
}

/// Padding mode for packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PaddingMode {
    /// No padding.
    None,
    /// Pad to the next alignment boundary.
    Align,
    /// Pad to the next page boundary.
    Page,
}

/// Result of a packing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackingResult {
    /// Byte size of the packed result.
    pub byte_size: u64,
    /// Number of elements packed.
    pub element_count: u64,
    /// Padding bytes added.
    pub padding_bytes: u64,
}
