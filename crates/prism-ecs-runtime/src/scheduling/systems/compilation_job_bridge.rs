//! Compilation-job bridge (constitutional home).
//!
//! Wraps the constitutional compilation commands behind a simple API.
//! This is the constitutional-side counterpart to the engine's
//! `CompilationJobBridge`; the engine file is the legacy duplicate
//! and is deleted in step 58.
//!
//! # Placeholder
//!
//! The engine's bridge writes to the engine's `World`. The
//! constitutional side writes via `WorldTxn`. The constitutional-side
//! bridge is a placeholder; the full implementation is added when
//! the engine's deployment_compiler migrates (in a future
//! compiler-side migration).

/// Constitutional-side compilation-job bridge.
pub struct CompilationJobBridge {
    _placeholder: (),
}

impl CompilationJobBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for CompilationJobBridge {
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
        let _ = CompilationJobBridge::new();
    }
}
