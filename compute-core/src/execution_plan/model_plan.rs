//! Model execution plan utilities — convenience functions for constructing
//! and inspecting `ModelExecutionPlan` instances.
//!
//! These helpers live in their own module to keep `mod.rs` focused on type
//! definitions while still offering plan-level API surface.

use crate::execution_plan::ExecutionMode;
use crate::execution_plan::ModelExecutionPlan;

/// Default execution mode for a legacy (non-fusion) plan.
pub const DEFAULT_EXECUTION_MODE: ExecutionMode = ExecutionMode::OpByOp;

impl ModelExecutionPlan {
    /// Returns `true` when the plan was constructed for fused execution.
    pub fn is_fused(&self) -> bool {
        matches!(
            self.execution_mode,
            ExecutionMode::RegionBatched | ExecutionMode::MegakernelExperimental
        )
    }

    /// Returns `true` when the plan uses megakernel fusion.
    pub fn is_megakernel(&self) -> bool {
        self.execution_mode == ExecutionMode::MegakernelExperimental
    }

    /// Count the total number of scheduled kernel ops across all regions.
    pub fn total_ops(&self) -> usize {
        self.regions.iter().map(|r| r.ops.len()).sum()
    }
}
