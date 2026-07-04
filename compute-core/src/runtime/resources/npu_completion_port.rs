//! Completion port bridging blocking NPU wait() to the ECS atomic observer.
//!
//! A background thread blocks on npu_wait() and writes the completed
//! submission sequence number here.  The ECS completion observer polls
//! with `Ordering::Acquire` — identical to the IOSurface pattern used
//! by the Metal backend's `StreamObservationSystem`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// NpuCompletionPort
// ---------------------------------------------------------------------------

/// Completion port bridging blocking NPU wait() to the ECS atomic observer.
///
/// A background thread blocks on npu_wait() and writes the completed
/// submission sequence number here.  The ECS StreamObservationSystem
/// polls with Ordering::Acquire — identical to the IOSurface pattern.
pub struct NpuCompletionPort {
    /// Written by the background thread (release), read by the observer (acquire).
    completed: Arc<AtomicU64>,
    /// Incremented by the submitter before npu_submit_execution.
    submission_seq: AtomicU64,
}

impl NpuCompletionPort {
    pub fn new() -> Self {
        Self {
            completed: Arc::new(AtomicU64::new(0)),
            submission_seq: AtomicU64::new(0),
        }
    }

    /// Allocate the next submission ID and return it.
    ///
    /// The caller submits this ID to the NPU, then the background
    /// thread stores it in `completed` when wait() returns.
    pub fn next_submission(&self) -> u64 {
        self.submission_seq.fetch_add(1, Ordering::AcqRel)
    }

    /// Non-blocking poll — the ECS observer calls this every tick.
    ///
    /// Returns the last completed submission ID, or 0 if none.
    pub fn poll_completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Returns a shared reference to the atomic for the background thread.
    pub fn completed_atomic(&self) -> Arc<AtomicU64> {
        self.completed.clone()
    }
}

// ---------------------------------------------------------------------------
// Resource impl
// ---------------------------------------------------------------------------

// `NpuCompletionPort` is `Send + Sync` because its fields (`Arc<AtomicU64>`,
// `AtomicU64`) atomics are natively thread-safe.  The blanket impl
// `impl<T: 'static + Send + Sync> Resource for T {}` covers this type.

impl Default for NpuCompletionPort {
    fn default() -> Self {
        Self::new()
    }
}
