//! Heterogeneous dispatcher (constitutional home, kernel half).
//!
//! Per the inventory v2.1 row 17, the engine's
//! `heterogeneous_executor.rs` (933 LOC) is split:
//! - Runtime half → `prism-ecs-runtime::scheduling::systems::heterogeneous_orchestration`
//! - Kernel half → this file
//!
//! The dispatcher is the kernel-side coordinator. It receives a
//! typed, non-authoritative dispatch request from the runtime,
//! routes it to the appropriate backend, and produces a typed,
//! non-authoritative completion value.
//!
//! Placeholder: the full implementation arrives with step 36.

use crate::execution_lane::ExecutionLane;
use std::time::Instant;

/// A typed, non-authoritative completion value.
///
/// The kernel produces a `Completion` after a backend submission
/// completes (or fails). The runtime reconciliation system
/// observes the completion and stages the resulting state
/// transition through `ConstitutionalWorldTxn`. Until the
/// transition is staged, the completion is non-authoritative —
/// the world does not yet reflect it.
#[derive(Debug, Clone)]
pub struct Completion {
    pub lane: ExecutionLane,
    pub success: bool,
    pub submit_time: Instant,
    pub completion_time: Instant,
}

impl Completion {
    /// Construct a completion for testing or stub purposes.
    /// Production code uses the real completion path through the
    /// backend executors.
    pub fn stub(lane: ExecutionLane, success: bool) -> Self {
        let now = Instant::now();
        Self {
            lane,
            success,
            submit_time: now,
            completion_time: now,
        }
    }
}

/// Heterogeneous dispatcher (kernel half).
#[derive(Debug, Default)]
pub struct HeterogeneousDispatcher {
    _placeholder: (),
}

impl HeterogeneousDispatcher {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_stub_carries_lane_and_success() {
        // Architectural invariant: a stub completion records the
        // lane and the success flag. The runtime reconciliation
        // system reads these fields.
        let c = Completion::stub(ExecutionLane::MlxGpu, true);
        assert_eq!(c.lane, ExecutionLane::MlxGpu);
        assert!(c.success);
    }

    #[test]
    fn heterogeneous_dispatcher_constructs() {
        let _ = HeterogeneousDispatcher::new();
    }
}
