//! Metal lane executor (constitutional home).
//!
//! Per the inventory v2.1 row 27, this replaces the engine's
//! `metal_lane_executor.rs` (142 LOC). The full implementation
//! arrives with step 36 (heterogeneous_executor split).
//!
//! Placeholder: the constitutional executor wraps the dispatch
//! contract. The Metal-specific command queue, pipeline state,
//! and texture cache arrive when the engine migrates.

use crate::execution_lane::ExecutionLane;
use crate::backend::dispatcher::heterogeneous::Completion;

/// Metal lane executor. Placeholder constitutional-side executor.
#[derive(Debug, Default)]
pub struct MetalLaneExecutor {
    _placeholder: (),
}

impl MetalLaneExecutor {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }

    /// Submit work to the Metal lane. Placeholder: returns a
    /// successful completion immediately. The real implementation
    /// submits a Metal command buffer and waits for completion.
    pub fn submit(&self) -> Result<Completion, String> {
        Ok(Completion::stub(ExecutionLane::MlxGpu, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_lane_executor_constructs() {
        let _ = MetalLaneExecutor::new();
    }

    #[test]
    fn metal_lane_executor_submit_returns_completion() {
        let e = MetalLaneExecutor::new();
        let c = e.submit().expect("submit succeeds");
        assert_eq!(c.lane, ExecutionLane::MlxGpu);
        assert!(c.success);
    }
}
