//! Worker crash recovery — extracted from worker_supervisor.rs.
//!
//! Contains [`CrashRecoveryState`] for tracking crash retries and buffering
//! in-flight request payloads.

use parking_lot::Mutex;
use serde_json::Value;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

// ── CrashRecoveryState ──────────────────────────────────────────────────────

/// Shared state for tracking crash recovery across threads.
///
/// The watchdog thread writes to this before marking the runtime faulted;
/// the supervisor reads from it when attempting recovery.
pub struct CrashRecoveryState {
    /// Serialized request payloads captured on crash, to be re-queued.
    pub pending_payloads: Mutex<Vec<Value>>,
    /// When the most recent crash was detected (for backoff computation).
    pub last_crash_time: Mutex<Option<Instant>>,
    /// Number of consecutive recovery attempts made.
    pub retry_count: AtomicU32,
}

impl CrashRecoveryState {
    pub fn new() -> Self {
        Self {
            pending_payloads: Mutex::new(Vec::new()),
            last_crash_time: Mutex::new(None),
            retry_count: AtomicU32::new(0),
        }
    }

    /// Maximum number of automatic recovery attempts before giving up.
    pub const MAX_RETRIES: u32 = 3;

    /// Exponential backoff in seconds for the nth retry (0-indexed).
    /// 1st retry: 1s, 2nd: 2s, 3rd: 4s
    pub fn backoff_seconds(&self) -> u64 {
        1u64 << self.retry_count.load(Ordering::SeqCst)
    }

    /// Drain and return the captured payloads.
    pub fn drain_payloads(&self) -> Vec<Value> {
        self.pending_payloads.lock().drain(..).collect()
    }
}

