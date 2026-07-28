//! Megakernel gather kernel — pure data types and pure algorithms.
//!
//! The Metal kernel source for the gather/scatter kernels lives
//! engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/megakernel/gather_kernel.rs`.

use serde::{Deserialize, Serialize};

/// Inputs to the gather kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherKernelInputs {
    /// Source segment identifier.
    pub source_segment_id: String,
    /// Number of bytes to gather.
    pub byte_count: u64,
    /// Destination offset in the staging buffer.
    pub dest_offset: u64,
}

/// Output of the gather kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatherKernelOutput {
    /// Number of bytes gathered.
    pub byte_count: u64,
    /// Content hash of the gathered bytes.
    pub content_hash: u64,
}

/// Statistics from the gather kernel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GatherKernelStats {
    /// Total bytes gathered.
    pub total_bytes: u64,
    /// Number of gather dispatches.
    pub num_dispatches: u64,
    /// Total elapsed time in nanoseconds.
    pub elapsed_ns: u64,
}
