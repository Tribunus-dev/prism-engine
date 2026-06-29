//! PRISM-REAL-CONCURRENT-EXECUTION-0001: Lane work types and executor trait.
//!
//! Defines the [`LaneExecutor`] trait and all transfer types for real
//! concurrent Metal + Core ML lane execution.  A lane executor returns
//! immediately after native submission; completion arrives later through
//! a Tokio channel.

use std::time::Instant;

use tokio::sync::mpsc;

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};
use crate::compilation::phase_ir::PhaseId;
use crate::compilation::tri_lane::NumericalStatus;
use crate::compute_image::compile::portfolio::CoreMlArtifactKey;
use crate::scheduling::tri_lane_orchestrator::{EpochId, VariantId};

// ── Work ID ─────────────────────────────────────────────────────────────────

/// Unique identifier for a submitted work item, returned by
/// [`LaneExecutor::submit`] and used to match completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkId(pub u64);

// ── Stream / session identifier ─────────────────────────────────────────────

/// Identifies a logical stream of work (branch A vs branch B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u64);

// ── Metal pipeline reference ────────────────────────────────────────────────

/// Opaque reference to a compiled Metal pipeline + its resource bindings.
#[derive(Debug, Clone)]
pub struct MetalPipelineRef {
    pub function_name: String,
    pub pipeline_digest: String,
}

// ── Completion clock ────────────────────────────────────────────────────────

/// Single-use clock that records submission and completion timestamps
/// for one work item.  The executor fills in `backend_start_ns` and
/// `backend_end_ns`; the orchestrator fills `submit_ns`.
#[derive(Debug, Clone)]
pub struct CompletionClock {
    pub submit_ns: u64,
    pub backend_start_ns: Option<u64>,
    pub backend_end_ns: Option<u64>,
    pub completion_callback_ns: Option<u64>,
}

impl CompletionClock {
    pub fn new(submit_ns: u64) -> Self {
        Self {
            submit_ns,
            backend_start_ns: None,
            backend_end_ns: None,
            completion_callback_ns: None,
        }
    }

    pub fn record_backend_start(&mut self, ns: u64) {
        self.backend_start_ns.get_or_insert(ns);
    }

    pub fn record_backend_end(&mut self, ns: u64) {
        self.backend_end_ns.get_or_insert(ns);
    }

    pub fn record_completion(&mut self, ns: u64) {
        self.completion_callback_ns.get_or_insert(ns);
    }
}

// ── Lane work request ──────────────────────────────────────────────────────

/// Everything a lane executor needs to begin executing one work item.
#[derive(Debug, Clone)]
pub struct LaneWorkRequest {
    pub work_id: WorkId,
    pub session_id: StreamId,
    pub epoch_id: EpochId,
    pub phase_id: PhaseId,
    pub variant_id: VariantId,
    pub lane: ExecutionLane,
    pub input_slots: Vec<SlotLeaseId>,
    pub output_slot: SlotLeaseId,
    pub input_abi: ActivationAbi,
    pub output_abi: ActivationAbi,
    pub artifact_key: Option<CoreMlArtifactKey>,
    pub metal_pipeline: Option<MetalPipelineRef>,
    pub completion_clock: CompletionClock,
}

// ── Work submission receipt ─────────────────────────────────────────────────

/// Returned by [`LaneExecutor::submit`] immediately — before the work finishes.
#[derive(Debug, Clone)]
pub struct WorkSubmission {
    pub work_id: WorkId,
    pub lane: ExecutionLane,
    pub submission_time: Instant,
}

// ── Work completion (backend-timed variant) ─────────────────────────────────

/// Produced by a lane executor's completion handler and sent through
/// the Tokio completion channel.
#[derive(Debug, Clone)]
pub struct WorkCompletion {
    pub work_id: WorkId,
    pub phase_id: PhaseId,
    pub variant_id: VariantId,
    pub lane: ExecutionLane,
    pub success: bool,
    pub output_slot: SlotLeaseId,
    pub backend_status: BackendStatus,
    pub numerical_status: NumericalStatus,
    pub timing: BackendExecutionTiming,
}

// ── Backend execution timing ────────────────────────────────────────────────

/// High-resolution timing for one backend execution, collected from
/// backend-specific instrumentation (Metal completion handler, ANE
/// worker thread boundary, etc.).
#[derive(Debug, Clone, Copy)]
pub struct BackendExecutionTiming {
    /// Monotonic timestamp just before native submission.
    pub submit_ns: u64,
    /// Monotonic timestamp when backend execution actually began
    /// (Metal GPU start, ANE prediction call entry).
    pub backend_start_ns: u64,
    /// Monotonic timestamp when backend execution completed
    /// (Metal GPU completion, ANE prediction return).
    pub backend_end_ns: u64,
    /// Monotonic timestamp when the completion callback was invoked.
    pub completion_callback_ns: u64,
    /// Quality indicator for the timing source.
    pub timestamp_quality: TimestampQuality,
}

/// Describes how the timing values were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampQuality {
    /// Timestamps from a backend completion handler or callback
    /// (most reliable — reflects real GPU/ANE execution).
    BackendCallback,
    /// Timestamps recorded at the worker thread boundary
    /// (ANE prediction entry/return on a dedicated thread).
    WorkerThreadBoundary,
    /// Timestamps from a Metal command-buffer completion handler.
    CommandBufferCompletion,
    /// Timestamps approximated from scheduler submission time
    /// (least reliable — only for stub implementations).
    SchedulerApproximation,
}

// ── Backend status ──────────────────────────────────────────────────────────

/// Execution result from the backend itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendStatus {
    Completed,
    Failed(String),
    Cancelled,
}

// ── Lane executor trait ─────────────────────────────────────────────────────

/// A lane executor owns one backend (Metal, Core ML ANE, or Accelerate/CPU)
/// and can submit work for asynchronous execution.
///
/// `submit()` must return immediately after native submission.  The actual
/// completion must arrive later through the provided `completion_tx`.
///
/// The executor must NOT mutate orchestrator state (readiness, leases,
/// cache) directly.  All side effects go through the completion channel.
pub trait LaneExecutor: Send {
    fn submit(
        &mut self,
        request: LaneWorkRequest,
        completion_tx: mpsc::UnboundedSender<WorkCompletion>,
    ) -> Result<WorkSubmission, LaneExecutionError>;
}

// ── Lane execution error ────────────────────────────────────────────────────

/// Non-retryable error from a lane executor during submission.
#[derive(Debug, Clone)]
pub struct LaneExecutionError {
    pub work_id: WorkId,
    pub lane: ExecutionLane,
    pub reason: String,
}

impl std::fmt::Display for LaneExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LaneExecutionError(work={:?}, lane={:?}): {}",
            self.work_id, self.lane, self.reason
        )
    }
}

impl std::error::Error for LaneExecutionError {}

// ── Work ID generator (simple atomic counter) ──────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_WORK_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_work_id() -> WorkId {
    WorkId(NEXT_WORK_ID.fetch_add(1, Ordering::Relaxed))
}

// ── Test helpers ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_work_id_increments() {
        let a = next_work_id();
        let b = next_work_id();
        assert!(a.0 < b.0);
    }

    #[test]
    fn test_completion_clock_records() {
        let mut clock = CompletionClock::new(100);
        clock.record_backend_start(200);
        clock.record_backend_end(500);
        clock.record_completion(600);
        assert_eq!(clock.backend_start_ns, Some(200));
        assert_eq!(clock.backend_end_ns, Some(500));
        assert_eq!(clock.completion_callback_ns, Some(600));
    }
}
