//! Receipts emitted during CPU fusion lowering.
//!
//! A [`CpuLoweringReceipt`] is emitted every time a `FusedGroup` is
//! lowered into an `AccelerateRayonProgram`. Receipts are used for
//! auditing, telemetry, and debugging — they capture the decisions
//! made during lowering without retaining the full program.

use crate::cpu_runtime::rayon_strategy::RayonStrategy;
use serde::{Deserialize, Serialize};

/// Receipt produced by the CPU lowering pipeline for auditing.
///
/// One receipt per fused group that was lowered. Contains the summary
/// of the lowering decision — which strategy was chosen, how many ops
/// and Accelerate calls were involved, and whether the result is
/// deterministic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLoweringReceipt {
    /// Identifier of the lowered program (matches `AccelerateRayonProgram.program_id`).
    pub program_id: String,

    /// Group id from the original `FusedGroup`.
    pub group_id: usize,

    /// Number of ops fused into this program.
    pub op_count: usize,

    /// The Rayon strategy selected during lowering.
    pub parallel_strategy: RayonStrategy,

    /// How many Accelerate framework calls were emitted.
    pub accelerate_call_count: usize,

    /// Total scratch buffer bytes allocated.
    pub scratch_bytes: u64,

    /// Whether the program is deterministic.
    pub deterministic: bool,

    /// Warnings emitted during lowering (e.g. "fallback codec path used").
    pub warnings: Vec<String>,
}
