//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Accelerate CPU lane executor.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use crate::backend::placement::ExecutionLane;
use crate::compilation::tri_lane::NumericalStatus;
use crate::scheduling::lane_work::{
    BackendExecutionTiming, BackendStatus, LaneExecutionError, LaneExecutor, LaneWorkRequest,
    TimestampQuality, WorkCompletion, WorkSubmission,
};

/// Accelerate CPU lane executor with bounded worker pool.
///
/// Wraps a bounded CPU worker pool delivered through Tokio's
/// `spawn_blocking` to avoid blocking the async scheduler.  Per-session
/// concurrency is tracked so no session exceeds `max_workers` concurrent
/// CPU operations.
///
/// The CPU work is a placeholder that spins for a short time.  In
/// production it calls Accelerate framework kernels from
/// `crate::backend::accelerate_lane`.
pub struct AccelerateLaneExecutor {
    name: String,
    max_workers: usize,
    runtime_handle: Handle,
    /// Per-session concurrency tracking (Arc for shared access with
    /// spawned blocking tasks).
    active_work: Arc<Mutex<HashMap<String, usize>>>,
}

impl AccelerateLaneExecutor {
    /// Create a new Accelerate lane executor.
    ///
    /// `max_workers` sets the maximum concurrent CPU workers (should be
    /// min(physical_performance_cores - 1, configured_max)).
    pub fn new(name: &str, max_workers: usize, runtime_handle: Handle) -> Self {
        Self {
            name: name.to_string(),
            max_workers,
            runtime_handle,
            active_work: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl LaneExecutor for AccelerateLaneExecutor {
    fn submit(
        &mut self,
        request: LaneWorkRequest,
        completion_tx: mpsc::UnboundedSender<WorkCompletion>,
    ) -> Result<WorkSubmission, LaneExecutionError> {
        let submit_time = Instant::now();
        let submit_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Check and record per-session concurrency.
        let session_key = format!("{:?}", request.session_id);
        {
            let mut active = self.active_work.lock();
            let count = active.entry(session_key.clone()).or_insert(0);
            if *count >= self.max_workers {
                return Err(LaneExecutionError {
                    work_id: request.work_id,
                    lane: ExecutionLane::AccelerateCpu,
                    reason: format!(
                        "session {:?} at max workers ({})",
                        request.session_id, self.max_workers
                    ),
                });
            }
            *count += 1;
        }

        // Clone the shared state for the blocking task.
        let active_work = Arc::clone(&self.active_work);

        // Clone what the blocking task needs from the request.
        let work_id = request.work_id;
        let phase_id = request.phase_id;
        let variant_id = request.variant_id;
        let output_slot = request.output_slot;
        let handle = self.runtime_handle.clone();

        handle.spawn_blocking(move || {
            let backend_start_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            // Simulated CPU work — in production this calls
            // Accelerate framework kernels from
            // crate::backend::accelerate_lane.
            //
            // The spin consumes ~50 µs of CPU time, enough to be
            // measurable and exercise the blocking-thread dispatch.
            let spin_start = Instant::now();
            while spin_start.elapsed().as_micros() < 50 {
                std::hint::spin_loop();
            }

            let backend_end_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            let timing = BackendExecutionTiming {
                submit_ns,
                backend_start_ns,
                backend_end_ns,
                completion_callback_ns: backend_end_ns,
                timestamp_quality: TimestampQuality::WorkerThreadBoundary,
            };

            let _ = completion_tx.send(WorkCompletion {
                work_id,
                phase_id,
                variant_id,
                lane: ExecutionLane::AccelerateCpu,
                success: true,
                output_slot,
                backend_status: BackendStatus::Completed,
                numerical_status: NumericalStatus::Pass,
                timing,
            });

            // Decrement per-session count.
            let mut active = active_work.lock();
            if let Some(count) = active.get_mut(&session_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    active.remove(&session_key);
                }
            }
        });

        Ok(WorkSubmission {
            work_id: request.work_id,
            lane: ExecutionLane::AccelerateCpu,
            submission_time: submit_time,
        })
    }
}

impl std::fmt::Debug for AccelerateLaneExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let active_sessions = self.active_work.lock().len();
        f.debug_struct("AccelerateLaneExecutor")
            .field("name", &self.name)
            .field("max_workers", &self.max_workers)
            .field("active_sessions", &active_sessions)
            .finish()
    }
}
