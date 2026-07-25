//! This module owns the canonical authority for the executable-plan
//! envelope types (`BufferAllocation`, `MemoryPlan`, `ReplayPlan`,
//! `ExecutionReceipt`).
//! It does not own graph mutation, kernel lowering, or replay submission.

use serde::{Deserialize, Serialize};

use crate::phase_graph::kernel_op::LoweringTarget;
use crate::phase_graph::uop::UOpId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferAllocation {
    pub value: UOpId,
    pub slot: usize,
    /// Minimum number of f32 elements required by this value.
    #[serde(default)]
    pub elements: usize,
    pub first_command: usize,
    pub last_command: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub allocations: Vec<BufferAllocation>,
    pub slot_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayPlan {
    pub command_ids: Vec<u32>,
    pub synchronization_points: Vec<u32>,
    /// Whether the command sequence is intended for persistent replay.
    /// Persistent executors can submit the complete sequence in one call;
    /// the default hook below preserves correctness for simpler executors.
    #[serde(default)]
    pub persistent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub target: LoweringTarget,
    /// Digest of the complete validated capture, including graph, memory,
    /// replay, and kernel metadata.
    #[serde(default)]
    pub capture_digest: String,
    pub command_ids: Vec<u32>,
    pub kernel_digests: Vec<String>,
    #[serde(default)]
    pub persistent: bool,
    pub replayed: bool,
}
