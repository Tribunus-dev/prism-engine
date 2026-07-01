//! WorkerPoolResource — legacy worker supervisor abstraction.
//!
//! Wraps the concept of selecting healthy workers and dispatching requests
//! to them.  In Slice 2 this is a placeholder that delegates to the legacy
//! worker supervisor path.  All public APIs return owned values — mutex
//! guards are never leaked.

use std::sync::atomic::{AtomicU32, Ordering};

/// Health status reported for a given worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealthStatus {
    /// Worker is accepting and processing requests normally.
    Healthy,
    /// Worker is still alive but experiencing elevated latency or errors.
    Degraded,
    /// Worker cannot be reached; requests to it will fail.
    Unreachable,
}

/// Service-level resource for worker pool management.
///
/// In Slice 2 this is a placeholder: `next_worker_id` exists for future
/// worker lifecycle tracking, and the public methods return defaults or
/// placeholder values until wired to the legacy supervisor.
pub struct WorkerPoolResource {
    /// Monotonically increasing counter for assigning worker identifiers.
    next_worker_id: AtomicU32,
}

impl WorkerPoolResource {
    /// Create a new worker pool resource starting from ID 1.
    pub fn new() -> Self {
        Self {
            next_worker_id: AtomicU32::new(1),
        }
    }

    /// Select a healthy worker from the pool.
    ///
    /// Returns `None` when no worker is available.  Placeholder — returns
    /// a synthetic worker identifier.
    pub fn select_healthy_worker(&self) -> Option<String> {
        // Placeholder: mint a synthetic worker id.
        let id = self.next_worker_id.fetch_add(1, Ordering::Relaxed);
        Some(format!("worker-{id}"))
    }

    /// Send a request payload to a specific worker for execution.
    ///
    /// Placeholder implementation — always succeeds.
    pub fn send_request(
        &self,
        _worker_id: &str,
        _request_id: &str,
        _payload: &[u8],
    ) -> Result<(), String> {
        // Placeholder: no-op, always succeeds.
        Ok(())
    }

    /// Request cancellation of an in-flight request on a worker.
    ///
    /// Placeholder implementation — always succeeds.
    pub fn request_cancellation(
        &self,
        _worker_id: &str,
        _request_id: &str,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Inspect the health of a specific worker.
    ///
    /// Placeholder — always returns `Healthy`.
    pub fn inspect_health(&self, _worker_id: &str) -> WorkerHealthStatus {
        WorkerHealthStatus::Healthy
    }

    /// Request that a worker be recovered (restarted / drained).
    ///
    /// Placeholder implementation — always succeeds.
    pub fn request_recovery(&self, _worker_id: &str) -> Result<(), String> {
        Ok(())
    }
}

impl Default for WorkerPoolResource {
    fn default() -> Self {
        Self::new()
    }
}
