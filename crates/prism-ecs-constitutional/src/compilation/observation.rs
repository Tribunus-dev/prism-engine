//! Observation projection: `CompileProgress`.
//!
//! **Single authority:** owns the canonical, side-effect-free shape
//! of a `CompileProgress` snapshot — the projection that a watcher
//! reads to display pipeline progress. Absorbed from the engine's
//! `compute-core/src/ecs/core/compile_progress.rs`.
//!
//! ## Why this is canonical
//!
//! Per AGENTS.md criteria 1–4, this type owns no hardware handles,
//! uses no `unsafe`, owns no process-local state (no channels /
//! locks / `OnceLock`), and performs no FFI. `eprintln!` to stderr
//! is a process-local side effect, not a file-descriptor ownership
//! pattern, so the canonical projection can carry an `emit` method
//! that mirrors the engine's original helper. Constitutional
//! observers (downstream consumers, dashboards) that want a typed
//! progress stream should depend on this type.
//!
//! ## Engine boundary
//!
//! The engine's `compute-core/src/ecs/core/compile_state.rs` remains
//! **execution-boundary**: it owns the `CompileState::write` /
//! `read` methods that open files on disk (`std::fs::File` per
//! criterion 1), so it stays in the engine. The data types it
//! carries (`CompileStage`, `SegmentCompletion`, `SchedulerConfig`)
//! have no constitutional equivalent and are not absorbed; consumers
//! who need them keep using the engine's path. The mapping from
//! `compile_state::CompileStage` to [`super::job::JobLifecycle`] is
//! documented in [`super::job`].
//!
//! ## Migration note for engine consumers
//!
//! `use compute_core::compile_progress::CompileProgress;` continues
//! to compile because the engine module now re-exports this type
//! (see the `compilation/observation.rs` shim in the engine crate).

use serde::{Deserialize, Serialize};

/// A snapshot of long-running compilation progress.
///
/// `CompileProgress` is a side-effect-free value type: the `emit`
/// method only writes a single line to stderr and is provided for
/// parity with the engine's original helper. Constitutional
/// consumers that want to drive a UI or a tracing span should
/// read the fields directly and emit through `tracing` or their
/// own reporting channel.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompileProgress {
    /// Logical stage name (e.g. "planning", "emitting", "verifying").
    pub stage: String,
    /// Bytes processed so far in the current stage.
    pub bytes_processed: u64,
    /// Total bytes to process in the current stage. May be 0 if the
    /// stage is byte-agnostic.
    pub bytes_total: u64,
    /// Elapsed time since the pipeline started, in milliseconds.
    pub elapsed_ms: u64,
}

impl CompileProgress {
    /// Create a new progress snapshot.
    #[must_use]
    pub fn new(
        stage: impl Into<String>,
        bytes_processed: u64,
        bytes_total: u64,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            stage: stage.into(),
            bytes_processed,
            bytes_total,
            elapsed_ms,
        }
    }

    /// Emit a one-line summary to stderr. Side effect: writes to
    /// `stderr`. Mirrors the engine's original `emit` helper.
    pub fn emit(&self) {
        eprintln!(
            "[compile-progress] {} {}/{} bytes {}ms",
            self.stage, self.bytes_processed, self.bytes_total, self.elapsed_ms
        );
    }
}

// `Default` so call sites that want a zero-valued progress can
// construct one without naming every field.
impl Default for CompileProgress {
    fn default() -> Self {
        Self {
            stage: String::new(),
            bytes_processed: 0,
            bytes_total: 0,
            elapsed_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_construction_via_new() {
        let p = CompileProgress::new("planning", 100, 1000, 250);
        assert_eq!(p.stage, "planning");
        assert_eq!(p.bytes_processed, 100);
        assert_eq!(p.bytes_total, 1000);
        assert_eq!(p.elapsed_ms, 250);
    }

    #[test]
    fn progress_default_is_zero() {
        let p = CompileProgress::default();
        assert_eq!(p.stage, "");
        assert_eq!(p.bytes_processed, 0);
        assert_eq!(p.bytes_total, 0);
        assert_eq!(p.elapsed_ms, 0);
    }

    #[test]
    fn progress_serde_roundtrip() {
        let p = CompileProgress::new("emitting", 50, 100, 75);
        let json = serde_json::to_string(&p).expect("serialize");
        let back: CompileProgress = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p, back);
    }
}
