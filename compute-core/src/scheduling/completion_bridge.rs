//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Bounded completion event bridge.
//!
//! Provides a bounded [`mpsc`] channel that all lane executors send completions
//! through on their dedicated worker threads.  The orchestrator's completion
//! handler drains this channel on the async side.
//!
//! The [`CompletionEvent`] carries richer provenance than the minimal
//! [`WorkCompletion`] — full `Instant`-based timing plus the logical
//! [`WorkKey`] and [`TimingQuality`] annotation — so the orchestrator can
//! back-date timing histograms without referring back to registry state.

use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::SlotLeaseId;
use crate::compilation::tri_lane::NumericalStatus;
use crate::scheduling::lane_work::{BackendStatus, TimestampQuality, WorkCompletion, WorkId};
use crate::scheduling::work_registry::{WorkKey, WorkStatus};

// ---------------------------------------------------------------------------
// Completion queue capacity
// ---------------------------------------------------------------------------

/// Default capacity of the bounded completion channel.
///
/// Sized to absorb burst completions from all lane executors (Metal GPU,
/// Core ML ANE, Accelerate/CPU) without back-pressuring their completion
/// handlers.  4096 entries at ~200 bytes each ≈ 800 KiB, well within L2
/// cache on Apple Silicon.
pub const DEFAULT_COMPLETION_QUEUE_CAPACITY: usize = 4096;

// ---------------------------------------------------------------------------
// Timing quality
// ---------------------------------------------------------------------------

/// Quality of timing data in the completion.
///
/// Each variant describes how the `backend_started_at` / `backend_ended_at`
/// fields were obtained, so the orchestrator can weight timing metrics
/// appropriately when computing histograms and backpressure signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TimingQuality {
    /// Timestamps from a Metal command-buffer completion handler (GPU-level
    /// start/end, highest fidelity for Metal lanes).
    MetalCommandBufferCompletion,
    /// Timestamps from the Core ML worker thread entry/exit boundary
    /// (ANE prediction call wall time).
    CoreMlWorkerBoundary,
    /// Timestamps from an Accelerate/CPU worker thread boundary.
    AccelerateWorkerBoundary,
    /// Timestamps captured natively by the backend (callback handler,
    /// scheduler approximation fallback, or direct `Instant::now()`).
    BackendNativeTimestamp,
}

// ---------------------------------------------------------------------------
// Completion event
// ---------------------------------------------------------------------------

/// Production-grade work completion with full timing provenance.
///
/// Richer than [`WorkCompletion`] — carries the logical [`WorkKey`],
/// `Instant`-based timing reconstructed from the backend's raw monotonic
/// nanos, and explicit [`TimingQuality`] and [`WorkStatus`] so the
/// orchestrator can process completions without consulting the work
/// registry for every field.
pub struct CompletionEvent {
    /// Unique work identifier (physical submission).
    pub work_id: WorkId,
    /// Logical work identity across retries and fallback attempts.
    pub work_key: WorkKey,
    /// Execution lane that produced this completion.
    pub lane: ExecutionLane,
    /// Wall-clock submission time (reconstructed from backend timing).
    pub submitted_at: Instant,
    /// Wall-clock time when backend execution actually began.
    pub backend_started_at: Instant,
    /// Wall-clock time when backend execution ended.
    pub backend_ended_at: Instant,
    /// Wall-clock time when this event was received by the orchestrator.
    pub callback_received_at: Instant,
    /// Lifecycle status derived from the completion outcome.
    pub status: WorkStatus,
    /// Output slot lease produced by this execution.
    pub output_lease: SlotLeaseId,
    /// Execution result from the backend.
    pub backend_status: BackendStatus,
    /// Quality annotation for the timing fields.
    pub timing_quality: TimingQuality,
    /// Numerical status from verification (if available).
    pub numerical_status: NumericalStatus,
    /// Free-form diagnostics payload from the backend.
    pub backend_diagnostics: String,
}

// ---------------------------------------------------------------------------
// Channel type aliases
// ---------------------------------------------------------------------------

/// Completion sender type — bounded channel sender.
///
/// Lane executors hold a cloneable sender and send [`CompletionEvent`]s
/// from their completion handler (Metal command-buffer callback, Core ML
/// worker thread, etc.).
pub type CompletionSender = mpsc::Sender<CompletionEvent>;

/// Completion receiver type.
///
/// The orchestrator's completion handler owns the single receiver and
/// drains it on the async runtime.
pub type CompletionReceiver = mpsc::Receiver<CompletionEvent>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Create a bounded completion channel pair.
///
/// `capacity` controls the maximum number of pending [`CompletionEvent`]s
/// before senders begin to back-pressure.  Use
/// [`DEFAULT_COMPLETION_QUEUE_CAPACITY`] (4096) when no specific capacity
/// is required.
pub fn completion_channel(capacity: usize) -> (CompletionSender, CompletionReceiver) {
    mpsc::channel(capacity)
}

/// Convert a lane executor's [`WorkCompletion`] into a [`CompletionEvent`].
///
/// Uses the backend-provided monotonic nanosecond timestamps and the
/// wall-clock `received_at` to reconstruct all four `Instant` fields:
///
/// | CompletionEvent field | Source |
/// |---|---|
/// | `submitted_at` | `received_at - (completion_callback_ns - submit_ns)` |
/// | `backend_started_at` | `received_at - (completion_callback_ns - backend_start_ns)` |
/// | `backend_ended_at` | `received_at - (completion_callback_ns - backend_end_ns)` |
/// | `callback_received_at` | `received_at` |
///
/// This is valid because all four raw-ns timestamps are from the same
/// monotonic clock domain (`mach_continuous_time` / `clock_gettime_ns_np`
/// on Darwin).
pub fn work_completion_to_event(
    wc: WorkCompletion,
    work_key: WorkKey,
    received_at: Instant,
) -> CompletionEvent {
    let timing = wc.timing;
    let callback_ns = timing.completion_callback_ns;

    // Compute wall-clock instants by subtracting the delta between the
    // backend's callback time and each recorded event from the observed
    // callback instant.
    let submitted_at =
        received_at - Duration::from_nanos(callback_ns.saturating_sub(timing.submit_ns));
    let backend_started_at =
        received_at - Duration::from_nanos(callback_ns.saturating_sub(timing.backend_start_ns));
    let backend_ended_at =
        received_at - Duration::from_nanos(callback_ns.saturating_sub(timing.backend_end_ns));

    let status = derive_work_status(&wc).unwrap_or(WorkStatus::Completed);
    let timing_quality = map_timing_quality(timing.timestamp_quality, wc.lane);
    let backend_diagnostics = extract_diagnostics(&wc.backend_status);

    CompletionEvent {
        work_id: wc.work_id,
        work_key,
        lane: wc.lane,
        submitted_at,
        backend_started_at,
        backend_ended_at,
        callback_received_at: received_at,
        status,
        output_lease: wc.output_slot,
        backend_status: wc.backend_status,
        timing_quality,
        numerical_status: wc.numerical_status,
        backend_diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Derive [`WorkStatus`] from a [`WorkCompletion`].
fn derive_work_status(wc: &WorkCompletion) -> Option<WorkStatus> {
    match wc.success {
        true => Some(WorkStatus::Completed),
        false => match &wc.backend_status {
            BackendStatus::Completed => {
                // success == false but backend says Completed: numerical failure.
                Some(WorkStatus::Completed)
            }
            BackendStatus::Failed(_) => Some(WorkStatus::ExecutionFailed),
            BackendStatus::Cancelled => Some(WorkStatus::CancelledBeforeSubmit),
        },
    }
}

/// Map [`TimestampQuality`] from the backend to the richer [`TimingQuality`].
///
/// * `CommandBufferCompletion` → [`MetalCommandBufferCompletion`](TimingQuality::MetalCommandBufferCompletion)
/// * `WorkerThreadBoundary`   → [`CoreMlWorkerBoundary`](TimingQuality::CoreMlWorkerBoundary)
///                              when lane is [`CoreMlAne`](ExecutionLane::CoreMlAne),
///                              otherwise [`AccelerateWorkerBoundary`](TimingQuality::AccelerateWorkerBoundary)
/// * `BackendCallback`        → [`BackendNativeTimestamp`](TimingQuality::BackendNativeTimestamp)
/// * `SchedulerApproximation` → [`BackendNativeTimestamp`](TimingQuality::BackendNativeTimestamp)
fn map_timing_quality(src: TimestampQuality, lane: ExecutionLane) -> TimingQuality {
    match src {
        TimestampQuality::CommandBufferCompletion => TimingQuality::MetalCommandBufferCompletion,
        TimestampQuality::WorkerThreadBoundary if lane == ExecutionLane::CoreMlAne => {
            TimingQuality::CoreMlWorkerBoundary
        }
        TimestampQuality::WorkerThreadBoundary => TimingQuality::AccelerateWorkerBoundary,
        TimestampQuality::BackendCallback => TimingQuality::BackendNativeTimestamp,
        TimestampQuality::SchedulerApproximation => TimingQuality::BackendNativeTimestamp,
    }
}

/// Extract a human-readable diagnostics string from a [`BackendStatus`].
fn extract_diagnostics(status: &BackendStatus) -> String {
    match status {
        BackendStatus::Completed => String::new(),
        BackendStatus::Failed(msg) => msg.clone(),
        BackendStatus::Cancelled => "cancelled".into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::phase_ir::PhaseId;
    use crate::scheduling::lane_work::BackendExecutionTiming;

    fn sample_timing(quality: TimestampQuality) -> BackendExecutionTiming {
        let base = 1_000_000_000_000; // 1000 sec in ns
        BackendExecutionTiming {
            submit_ns: base,
            backend_start_ns: base + 100_000,       // +100 µs
            backend_end_ns: base + 500_000,         // +500 µs
            completion_callback_ns: base + 600_000, // +600 µs
            timestamp_quality: quality,
        }
    }

    fn sample_completion(quality: TimestampQuality) -> WorkCompletion {
        WorkCompletion {
            work_id: WorkId(42),
            phase_id: PhaseId(1),
            variant_id: 0,
            lane: ExecutionLane::MlxGpu,
            success: true,
            output_slot: SlotLeaseId(100),
            backend_status: BackendStatus::Completed,
            numerical_status: NumericalStatus::Pass,
            timing: sample_timing(quality),
        }
    }

    fn sample_work_key() -> WorkKey {
        WorkKey {
            session_id: "test-session".into(),
            request_id: "req-1".into(),
            sequence_id: 0,
            epoch_id: 0,
            phase_id: PhaseId(1),
            attempt: 0,
        }
    }

    // ── channel creation ──────────────────────────────────────────────────

    #[test]
    fn test_completion_channel_default_capacity() {
        let (tx, mut rx) = completion_channel(DEFAULT_COMPLETION_QUEUE_CAPACITY);
        let ev = CompletionEvent {
            work_id: WorkId(1),
            work_key: sample_work_key(),
            lane: ExecutionLane::MlxGpu,
            submitted_at: Instant::now(),
            backend_started_at: Instant::now(),
            backend_ended_at: Instant::now(),
            callback_received_at: Instant::now(),
            status: WorkStatus::Completed,
            output_lease: SlotLeaseId(0),
            backend_status: BackendStatus::Completed,
            timing_quality: TimingQuality::MetalCommandBufferCompletion,
            numerical_status: NumericalStatus::NotValidated,
            backend_diagnostics: String::new(),
        };

        // Send must succeed (channel has capacity)
        assert!(tx.try_send(ev).is_ok());
        assert!(rx.try_recv().is_ok());
    }

    // ── timing reconstruction ─────────────────────────────────────────────

    #[test]
    fn test_timing_reconstruction_is_plausible() {
        let now = Instant::now();
        let wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        let event = work_completion_to_event(wc, sample_work_key(), now);

        // submitted_at must be <= backend_started_at <= backend_ended_at <= callback
        assert!(
            event.submitted_at <= event.backend_started_at,
            "submitted_at must precede backend_started_at"
        );
        assert!(
            event.backend_started_at <= event.backend_ended_at,
            "backend_started_at must precede backend_ended_at"
        );
        assert!(
            event.backend_ended_at <= event.callback_received_at,
            "backend_ended_at must precede callback_received_at"
        );
        // callback_received_at must equal the received_at we passed in
        assert_eq!(event.callback_received_at, now);
    }

    #[test]
    fn test_timing_reconstruction_exact_deltas() {
        let now = Instant::now();
        let wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        let event = work_completion_to_event(wc, sample_work_key(), now);

        // submit_ns = base, completion_callback_ns = base + 600_000
        // elapsed = 600 µs
        let submit_delta = now.duration_since(event.submitted_at);
        assert_eq!(
            submit_delta.as_nanos() as u64,
            600_000,
            "submit → callback should be 600 µs"
        );

        // backend_started_at = base + 100_000, delta from callback = 500 µs
        let start_delta = now.duration_since(event.backend_started_at);
        assert_eq!(
            start_delta.as_nanos() as u64,
            500_000,
            "backend_start → callback should be 500 µs"
        );

        // backend_ended_at = base + 500_000, delta from callback = 100 µs
        let end_delta = now.duration_since(event.backend_ended_at);
        assert_eq!(
            end_delta.as_nanos() as u64,
            100_000,
            "backend_end → callback should be 100 µs"
        );
    }

    // ── TimingQuality mapping ──────────────────────────────────────────────

    #[test]
    fn test_map_command_buffer_completion() {
        assert_eq!(
            map_timing_quality(
                TimestampQuality::CommandBufferCompletion,
                ExecutionLane::MlxGpu
            ),
            TimingQuality::MetalCommandBufferCompletion
        );
    }

    #[test]
    fn test_map_worker_thread_boundary_coreml() {
        assert_eq!(
            map_timing_quality(
                TimestampQuality::WorkerThreadBoundary,
                ExecutionLane::CoreMlAne
            ),
            TimingQuality::CoreMlWorkerBoundary
        );
    }

    #[test]
    fn test_map_worker_thread_boundary_other() {
        assert_eq!(
            map_timing_quality(
                TimestampQuality::WorkerThreadBoundary,
                ExecutionLane::AccelerateCpu
            ),
            TimingQuality::AccelerateWorkerBoundary
        );
        assert_eq!(
            map_timing_quality(
                TimestampQuality::WorkerThreadBoundary,
                ExecutionLane::MlxGpu
            ),
            TimingQuality::AccelerateWorkerBoundary
        );
    }

    #[test]
    fn test_map_backend_callback_and_approximation() {
        assert_eq!(
            map_timing_quality(TimestampQuality::BackendCallback, ExecutionLane::MlxGpu),
            TimingQuality::BackendNativeTimestamp
        );
        assert_eq!(
            map_timing_quality(
                TimestampQuality::SchedulerApproximation,
                ExecutionLane::CoreMlAne
            ),
            TimingQuality::BackendNativeTimestamp
        );
    }

    // ── WorkStatus derivation ─────────────────────────────────────────────

    #[test]
    fn test_derive_completed_status() {
        let wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        assert_eq!(derive_work_status(&wc), Some(WorkStatus::Completed));
    }

    #[test]
    fn test_derive_execution_failed_status() {
        let mut wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        wc.success = false;
        wc.backend_status = BackendStatus::Failed("kernel crashed".into());
        assert_eq!(derive_work_status(&wc), Some(WorkStatus::ExecutionFailed));
    }

    #[test]
    fn test_derive_cancelled_status() {
        let mut wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        wc.success = false;
        wc.backend_status = BackendStatus::Cancelled;
        assert_eq!(
            derive_work_status(&wc),
            Some(WorkStatus::CancelledBeforeSubmit)
        );
    }

    #[test]
    fn test_derive_paradoxical_completed_backend() {
        // success == false but backend_status says Completed — treat as
        // completion (numerical failure case).
        let mut wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        wc.success = false;
        wc.backend_status = BackendStatus::Completed;
        assert_eq!(derive_work_status(&wc), Some(WorkStatus::Completed));
    }

    // ── Diagnostics extraction ────────────────────────────────────────────

    #[test]
    fn test_extract_diagnostics_empty() {
        assert_eq!(extract_diagnostics(&BackendStatus::Completed), "");
    }

    #[test]
    fn test_extract_diagnostics_failed() {
        assert_eq!(
            extract_diagnostics(&BackendStatus::Failed("OOM".into())),
            "OOM"
        );
    }

    #[test]
    fn test_extract_diagnostics_cancelled() {
        assert_eq!(extract_diagnostics(&BackendStatus::Cancelled), "cancelled");
    }

    // ── End-to-end conversion ──────────────────────────────────────────────

    #[test]
    fn test_work_completion_to_event_round_trip() {
        let now = Instant::now();
        let wc = sample_completion(TimestampQuality::CommandBufferCompletion);
        let key = sample_work_key();
        let event = work_completion_to_event(wc, key.clone(), now);

        assert_eq!(event.work_id, WorkId(42));
        assert_eq!(event.work_key, key);
        assert_eq!(event.lane, ExecutionLane::MlxGpu);
        assert_eq!(event.output_lease, SlotLeaseId(100));
        assert_eq!(event.backend_status, BackendStatus::Completed);
        assert_eq!(event.status, WorkStatus::Completed);
        assert_eq!(
            event.timing_quality,
            TimingQuality::MetalCommandBufferCompletion
        );
        assert_eq!(event.numerical_status, NumericalStatus::Pass);
        assert!(event.backend_diagnostics.is_empty());
    }
}
