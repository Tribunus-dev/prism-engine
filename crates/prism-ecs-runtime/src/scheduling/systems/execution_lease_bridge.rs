//! Execution-lease bridge (constitutional home).
//!
//! Thin API over the constitutional execution-lease commands
//! (`AcquireExecutionLeaseCommand`, `CompleteExecutionLeaseCommand`).
//! This is the constitutional-side counterpart to the engine's
//! `ExecutionLeaseBridge`; the engine file is the legacy duplicate
//! and is deleted in step 58.
//!
//! # Authority
//!
//! This is a system (S bucket). It wraps the lease command, which is
//! a `ConstitutionalWorldTxn` operation. Errors are discarded at the
//! call site (`let _ = …`) so a constitutional failure never stalls
//! the hot execution path.
//!
//! # Placeholder engine types
//!
//! The engine's bridge writes to the engine's `World`. The
//! constitutional side writes via `WorldTxn` (in
//! `prism-ecs-constitutional`). The constitutional-side bridge
//! is a placeholder; the full implementation is added when the
//! engine's heterogeneous_executor migrates in step 36.

/// Constitutional-side execution-lease bridge.
///
/// The full implementation is added when heterogeneous_executor
/// migrates. This placeholder defines the public surface so the
/// constitutional side compiles and can be referenced by future
/// systems.
pub struct ExecutionLeaseBridge {
    _placeholder: (),
}

impl ExecutionLeaseBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for ExecutionLeaseBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `execution_lease_bridge` system.

    use super::*;

    #[test]
    fn bridge_constructs() {
        // Architectural invariant: a fresh bridge has no world
        // attached. The full bridge API arrives with step 36.
        let _ = ExecutionLeaseBridge::new();
    }
}
