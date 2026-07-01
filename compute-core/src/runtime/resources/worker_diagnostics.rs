//! WorkerDiagnosticsResource — diagnostic counters for worker supervision.
//!
//! Each counter is an `AtomicU64` so multiple systems can increment
//! concurrently without locking.  Counters are monotonic — they only
//! increase within a World instance.

use std::sync::atomic::{AtomicU64, Ordering};

/// Diagnostics counters for worker supervision events.
///
/// Exported (public) fields allow direct reads for observability;
/// increment methods ensure relaxed ordering for performance.
#[derive(Debug)]
pub struct WorkerDiagnosticsResource {
    /// Events dropped because the event buffer was full or the worker was
    /// no longer tracked.
    pub stale_event_drops: AtomicU64,
    /// Events whose assignment_generation did not match the current value.
    pub generation_mismatches: AtomicU64,
    /// Requests rejected because the worker lifecycle forbade the transition.
    pub lifecycle_rejections: AtomicU64,
    /// Watchdog-triggered escalations (worker reported unhealthy).
    pub watchdog_escalations: AtomicU64,
    /// Worker restart requests issued.
    pub restart_requests: AtomicU64,
    /// Failures to deliver a response token or terminal to the bridge.
    pub response_delivery_failures: AtomicU64,
}

impl WorkerDiagnosticsResource {
    /// Create a new diagnostics resource with all counters zeroed.
    pub fn new() -> Self {
        Self {
            stale_event_drops: AtomicU64::new(0),
            generation_mismatches: AtomicU64::new(0),
            lifecycle_rejections: AtomicU64::new(0),
            watchdog_escalations: AtomicU64::new(0),
            restart_requests: AtomicU64::new(0),
            response_delivery_failures: AtomicU64::new(0),
        }
    }

    /// Increment the stale event drops counter.
    pub fn record_stale_event_drop(&self) {
        self.stale_event_drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the generation mismatches counter.
    pub fn record_generation_mismatch(&self) {
        self.generation_mismatches.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the lifecycle rejections counter.
    pub fn record_lifecycle_rejection(&self) {
        self.lifecycle_rejections.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the watchdog escalations counter.
    pub fn record_watchdog_escalation(&self) {
        self.watchdog_escalations.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the restart requests counter.
    pub fn record_restart_request(&self) {
        self.restart_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the response delivery failures counter.
    pub fn record_response_delivery_failure(&self) {
        self.response_delivery_failures.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for WorkerDiagnosticsResource {
    fn default() -> Self {
        Self::new()
    }
}
