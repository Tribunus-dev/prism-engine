//! Work-lifecycle bridge (constitutional home).
//!
//! Records work-item lifecycle transitions in the constitutional
//! world. This is the constitutional-side counterpart to the
//! engine's `WorkLifecycleBridge`; the engine file is the legacy
//! duplicate and is deleted in step 58.
//!
//! # Placeholder
//!
//! The engine's bridge writes to the engine's `World`. The
//! constitutional side writes via `WorldTxn`. The constitutional-side
//! bridge is a placeholder; the full implementation is added when
//! the engine's heterogeneous_executor migrates in step 36.

/// Constitutional-side work-lifecycle bridge.
pub struct WorkLifecycleBridge {
    _placeholder: (),
}

impl WorkLifecycleBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for WorkLifecycleBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_constructs() {
        let _ = WorkLifecycleBridge::new();
    }
}
