//! Virtual hardware abstractions — Level 2 of the SpatialIR representation.
//!
//! Logical execution and memory resources abstracted from any specific target.
//! Placement is the process of mapping Level 1 (spatial graph) nodes to Level 2
//! (virtual hardware) units, producing a logical schedule.

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// VirtualComputeUnit
// ---------------------------------------------------------------------------

/// A logical execution unit abstracted from any specific hardware target.
///
/// Each variant represents an abstract execution domain that a compute node
/// can be placed on. The physical lowering step maps these to backend-specific
/// implementations (Metal pipelines, Core ML models, Accelerate calls, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VirtualComputeUnit {
    /// General-purpose CPU vector lane (e.g., Accelerate, AMX).
    CpuLane,
    /// GPU compute region (e.g., Metal compute pipeline on Apple Silicon).
    GpuComputeRegion,
    /// Apple Neural Engine subgraph (Core ML model execution).
    AnEngine,
    /// Accelerate framework primitive (vDSP, BLAS, vImage).
    AccelerateUnit,
    /// Transfer or synchronization unit for data movement between domains.
    TransferUnit,
}

// ---------------------------------------------------------------------------
// VirtualMemoryRegion
// ---------------------------------------------------------------------------

/// A logical memory region abstracted from any specific hardware memory.
///
/// Describes where data lives during execution, not the physical address space.
/// The legalizer validates that all data movement across region boundaries
/// has an explicit materialization boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VirtualMemoryRegion {
    /// System memory shared between CPU and GPU (unified memory on Apple Silicon).
    UnifiedMemory,
    /// GPU-dedicated VRAM (discrete GPU or carved-out region).
    DedicatedGpuVram,
    /// Shared on-chip cache (e.g., SLC on Apple Silicon).
    SharedCache,
    /// Core ML / ANE input/output memory.
    AnEngineMemory,
    /// Immutable weight storage (read-only mapped).
    MappedWeights,
}

// ---------------------------------------------------------------------------
// ExecutionBoundary
// ---------------------------------------------------------------------------

/// An explicit execution boundary between two regions in the spatial graph.
///
/// Boundaries carry measured costs from calibration (submission overhead,
/// domain transition latency, materialization bandwidth) and are used by
/// the cost model to estimate overall schedule cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionBoundary {
    /// Submission boundary between GPU regions.
    GpuSubmission {
        /// Region producing the output.
        source_region: String,
        /// Region consuming the input.
        destination_region: String,
        /// Measured or estimated submission latency.
        estimated_latency: Duration,
    },
    /// Synchronization point within a CPU execution graph.
    CpuSync {
        /// Reason for the synchronization.
        reason: String,
        /// Estimated synchronization cost.
        estimated_cost: Duration,
    },
    /// Synchronization point for ANE (Core ML) invocation boundaries.
    AnSync {
        /// Model artifact identifier.
        model_id: String,
        /// Whether this is the first invocation (cold start).
        is_cold_start: bool,
        /// Estimated latency for the invocation.
        estimated_latency: Duration,
    },
    /// Start of a domain materialization transfer.
    TransferStart {
        /// Tensor identifier being moved.
        tensor_id: String,
        /// Source memory region.
        from: VirtualMemoryRegion,
        /// Destination memory region.
        to: VirtualMemoryRegion,
        /// Total bytes to transfer.
        bytes: usize,
    },
    /// End of a domain materialization transfer.
    TransferEnd {
        /// Tensor identifier being moved.
        tensor_id: String,
        /// Source memory region.
        from: VirtualMemoryRegion,
        /// Destination memory region.
        to: VirtualMemoryRegion,
        /// Number of bytes actually transferred.
        bytes: usize,
    },
    /// Synchronization between L1 scratchpad regions (dataflow buffer handoff).
    L1Sync {
        source_region: String,
        dest_region: String,
        waits_on: Vec<String>,
    },
    /// Start of a collective communication operation.
    CollectiveStart {
        collective: String,
        device_ids: Vec<u32>,
    },
    /// End of a collective operation.
    CollectiveEnd { collective: String },
}

impl ExecutionBoundary {
    /// Returns a short human-readable label describing this boundary.
    pub fn label(&self) -> &str {
        match self {
            Self::GpuSubmission { .. } => "gpu_submission",
            Self::CpuSync { .. } => "cpu_sync",
            Self::AnSync { .. } => "ane_sync",
            Self::TransferStart { .. } => "transfer_start",
            Self::TransferEnd { .. } => "transfer_end",
            Self::L1Sync { .. } => "l1_sync",
            Self::CollectiveStart { .. } => "collective_start",
            Self::CollectiveEnd { .. } => "collective_end",
        }
    }

    /// Returns the estimated latency of this boundary.
    pub fn estimated_latency(&self) -> Duration {
        match self {
            Self::GpuSubmission {
                estimated_latency, ..
            } => *estimated_latency,
            Self::CpuSync { estimated_cost, .. } => *estimated_cost,
            Self::AnSync {
                estimated_latency, ..
            } => *estimated_latency,
            Self::TransferStart { bytes, .. } => Duration::from_micros((*bytes as u64) / 100),
            Self::TransferEnd { .. } => Duration::from_micros(1),
            Self::L1Sync { .. } => Duration::from_micros(5),
            Self::CollectiveStart { .. } => Duration::from_micros(10),
            Self::CollectiveEnd { .. } => Duration::from_micros(5),
        }
    }

    /// Returns the number of bytes transferred across this boundary, if applicable.
    pub fn transfer_bytes(&self) -> Option<usize> {
        match self {
            Self::TransferStart { bytes, .. } => Some(*bytes),
            Self::TransferEnd { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }
}

/// Kind of collective communication operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollectiveKind {
    AllGather {
        dim: usize,
        devices: u32,
    },
    ReduceScatter {
        dim: usize,
        devices: u32,
        reduction: ReduceKind,
    },
    AllReduce {
        devices: u32,
        reduction: ReduceKind,
    },
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceKind {
    Sum,
    Max,
    Mean,
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

/// Describes the placement of a spatial graph node onto virtual hardware.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Placement {
    /// The node being placed.
    pub node_id: String,
    /// The virtual compute unit assigned.
    pub compute_unit: VirtualComputeUnit,
    /// The memory region for this node's data.
    pub memory_region: VirtualMemoryRegion,
    /// Boundaries that must be crossed for this placement.
    pub boundaries: Vec<ExecutionBoundary>,
}
