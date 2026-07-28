//! Layout descriptors for the content store.

use serde::{Deserialize, Serialize};

/// Memory layout for content store entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayout {
    /// Row-major contiguous layout.
    RowMajor,
    /// Column-major contiguous layout.
    ColumnMajor,
    /// Packed / blocked layout (e.g., nf4tile640).
    Packed,
    /// Sparse layout.
    Sparse,
}
