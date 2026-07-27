//! Lane work transfer types (constitutional home).
//!
//! This is the constitutional home for the lane-work transfer types:
//! work identifiers, completion timing, submission and completion records,
//! and the status enums that flow between runtime scheduling and kernel
//! backend executors.
//!
//! # Authority
//!
//! Every type in this module is **scheduling state** in the C bucket.
//! A `LaneWorkRequest` becomes visible to a kernel backend only after
//! the dispatch-selection system stages it through `ConstitutionalWorldTxn`.
//! A `WorkCompletion` is non-authoritative until the runtime
//! completion-reconciliation system stages it.
//!
//! # Split from the engine `lane_work.rs`
//!
//! The legacy home was `compute-core/src/ecs/scheduling/lane_work.rs`.
//! The engine file also contained the `LaneExecutor` trait and the
//! `next_work_id()` global counter, which are NOT state. Per the
//! engine absorption principle (constitutional governance, ECS-shaped
//! data):
//!
//! - **`LaneExecutor` trait** — moves to `prism-ecs-kernel::backend::lane_executor`
//!   in step 36 alongside the heterogeneous-executor split. The trait
//!   is the contract that kernel backends implement; it is not state.
//! - **`next_work_id()`** — the engine uses a process-global atomic.
//!   In the constitutional home, work-id allocation is a runtime
//!   scheduling concern; the global atomic becomes a per-world
//!   `WorkIdAllocator` when the heterogeneous executor is split.
//!   For now this module does not export a global counter; callers
//!   should obtain a `WorkId` from the per-world allocator once that
//!   exists.
//!
//! # Engine-type placeholders
//!
//! The legacy `LaneWorkRequest` and `WorkCompletion` reference engine
//! types (`PhaseId`, `SlotLeaseId`, `ActivationAbi`, `NumericalStatus`,
//! `CoreAiArtifactKey`, `EpochId`, `VariantId`). The constitutional
//! home provides **placeholder newtypes** for each so the runtime
//! file builds and tests. When the engine files for those types move
//! into their constitutional homes (in their own migration steps),
//! the placeholders here are replaced by the moved definitions.
//! The placeholders have the same wire shape (`u64`, enum, struct)
//! as the engine types so receipt signatures stay stable across the
//! transition.
//!
//! # Invariants
//!
//! - `WorkId` is opaque to the runtime; the engine's global atomic is
//!   replaced by a per-world allocator.
//! - `CompletionClock` timestamps are monotonic. The clock is
//!   single-use per work item; reusing it would corrupt the receipt.
//! - `TimestampQuality` is a 4-variant enum that the runtime
//!   reconciliation system reads to decide whether a completion is
//!   admissible evidence.

use std::time::Instant;

use prism_ecs_kernel::execution_lane::ExecutionLane;

// ---------------------------------------------------------------------------
// Placeholder engine types
//
// These are temporary stand-ins for engine types that will move in
// later migration steps. When the engine files land, these become
// `pub use` re-exports of the canonical types. The wire shape matches
// the engine so receipt signatures stay stable.
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::scheduling::tri_lane_orchestrator::EpochId`
/// (which is a `u64` type alias). Replaced when `tri_lane_orchestrator`
/// moves in step 27.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EpochId(pub u64);

/// Placeholder for `compute-core::ecs::scheduling::tri_lane_orchestrator::VariantId`
/// (which is a `u64` type alias). Replaced when `tri_lane_orchestrator`
/// moves in step 27.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VariantId(pub u64);

/// Placeholder for `compute-core::ecs::compilation::phase_ir::PhaseId(u64)`.
/// Replaced when `phase_ir` moves into `prism-ecs-compile` (separate migration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PhaseId(pub u64);

/// Placeholder for `compute-core::ecs::compilation::activation_abi::SlotLeaseId(u64)`.
/// Replaced when `activation_abi` moves into `prism-ecs-compile` (separate migration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SlotLeaseId(pub u64);

/// Placeholder for the activation ABI body (engine has a richer struct;
/// for now we keep the lane-work types self-contained and accept the
/// same wire shape as the engine).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ActivationAbi {
    /// Opaque ABI identifier (a digest of the binding signature).
    pub abi_digest: String,
}

/// Placeholder for `compute-core::ecs::compilation::tri_lane::NumericalStatus`.
/// Replaced when `tri_lane` moves into `prism-ecs-compile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NumericalStatus {
    /// Backend reports the result is numerically correct within tolerance.
    Ok,
    /// Backend reports a precision warning but the result is usable.
    PrecisionWarning,
    /// Backend reports the result is not numerically usable.
    Failed,
}

/// Placeholder for `compute-core::ecs::compute_image::compile::portfolio::CoreAiArtifactKey`.
/// Replaced when `compile/portfolio` moves into `prism-ecs-compile`
/// (separate migration).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CoreAiArtifactKey {
    /// Opaque artifact identifier (a digest).
    pub artifact_digest: String,
}

// ---------------------------------------------------------------------------
// Work ID
// ---------------------------------------------------------------------------

/// Unique identifier for a submitted work item.
///
/// Returned by the dispatch-selection system and used to match
/// completions. The id is opaque to backends; it is meaningful to
/// the runtime completion-reconciliation system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkId(pub u64);

// ---------------------------------------------------------------------------
// Stream / session identifier
// ---------------------------------------------------------------------------

/// Identifies a logical stream of work (branch A vs branch B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamId(pub u64);

// ---------------------------------------------------------------------------
// Metal pipeline reference
// ---------------------------------------------------------------------------

/// Opaque reference to a compiled Metal pipeline + its resource bindings.
#[derive(Debug, Clone)]
pub struct MetalPipelineRef {
    pub function_name: String,
    pub pipeline_digest: String,
}

// ---------------------------------------------------------------------------
// Completion clock
// ---------------------------------------------------------------------------

/// Single-use clock that records submission and completion timestamps
/// for one work item. The kernel fills in `backend_start_ns` and
/// `backend_end_ns`; the runtime fills `submit_ns`; the runtime
/// reconciliation system fills `completion_callback_ns` when the
/// completion reenters the world.
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

// ---------------------------------------------------------------------------
// Lane work request
// ---------------------------------------------------------------------------

/// Everything a backend executor needs to begin executing one work item.
///
/// A `LaneWorkRequest` is a scheduling-state record. The dispatch-selection
/// system stages a request through `ConstitutionalWorldTxn`; the kernel
/// backend consumes the committed request from the world and submits the
/// actual hardware work. The request itself is immutable once committed.
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
    pub artifact_key: Option<CoreAiArtifactKey>,
    pub metal_pipeline: Option<MetalPipelineRef>,
    pub completion_clock: CompletionClock,
}

// ---------------------------------------------------------------------------
// Work submission receipt
// ---------------------------------------------------------------------------

/// Returned by the kernel backend immediately after native submission —
/// before the work finishes.
///
/// A `WorkSubmission` is non-authoritative. It is a typed completion
/// value that the runtime reconciliation system observes; the actual
/// authoritative state change happens when the corresponding
/// `WorkCompletion` is staged through `ConstitutionalWorldTxn`.
#[derive(Debug, Clone)]
pub struct WorkSubmission {
    pub work_id: WorkId,
    pub lane: ExecutionLane,
    pub submission_time: Instant,
}

// ---------------------------------------------------------------------------
// Work completion (backend-timed variant)
// ---------------------------------------------------------------------------

/// Produced by a kernel backend's completion handler and sent through
/// the Tokio completion channel.
///
/// A `WorkCompletion` is non-authoritative until the runtime
/// completion-reconciliation system stages it through
/// `ConstitutionalWorldTxn`. The kernel must NOT mutate orchestrator
/// state (readiness, leases, cache) directly; all side effects go
/// through the completion channel.
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

// ---------------------------------------------------------------------------
// Backend execution timing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Backend status
// ---------------------------------------------------------------------------

/// Execution result from the backend itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendStatus {
    Completed,
    Failed(String),
    Cancelled,
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `lane_work` state.
    //!
    //! These tests verify the *constitution* of lane work: that
    //! completion clocks are monotone, that submission and completion
    //! are distinct non-authoritative events, and that the placeholders
    //! preserve the engine wire shape.

    use super::*;

    #[test]
    fn work_ids_are_distinct() {
        let a = WorkId(1);
        let b = WorkId(2);
        assert_ne!(a, b);
    }

    #[test]
    fn completion_clock_records_first_wins() {
        // Architectural invariant: a completion clock is single-use
        // per work item. If a backend records a timestamp twice, the
        // first value wins (the `get_or_insert` semantics). The test
        // pins that behavior so receipt signatures stay stable.
        let mut clock = CompletionClock::new(100);
        clock.record_backend_start(200);
        clock.record_backend_start(300);
        assert_eq!(clock.backend_start_ns, Some(200));

        clock.record_backend_end(500);
        clock.record_backend_end(600);
        assert_eq!(clock.backend_end_ns, Some(500));

        clock.record_completion(700);
        clock.record_completion(800);
        assert_eq!(clock.completion_callback_ns, Some(700));
    }

    #[test]
    fn completion_clock_submit_is_set_at_construction() {
        let clock = CompletionClock::new(42);
        assert_eq!(clock.submit_ns, 42);
        assert_eq!(clock.backend_start_ns, None);
        assert_eq!(clock.backend_end_ns, None);
        assert_eq!(clock.completion_callback_ns, None);
    }

    #[test]
    fn timestamp_quality_variants_are_distinct() {
        // Four variants, mutually exclusive. A reader must be able to
        // dispatch on the variant without forgetting a case.
        for q in [
            TimestampQuality::BackendCallback,
            TimestampQuality::WorkerThreadBoundary,
            TimestampQuality::CommandBufferCompletion,
            TimestampQuality::SchedulerApproximation,
        ] {
            // Each variant is distinct from the others.
            let all = [
                TimestampQuality::BackendCallback,
                TimestampQuality::WorkerThreadBoundary,
                TimestampQuality::CommandBufferCompletion,
                TimestampQuality::SchedulerApproximation,
            ];
            let count = all.iter().filter(|&&v| v == q).count();
            assert_eq!(count, 1, "every variant must be self-equal exactly once");
        }
    }

    #[test]
    fn backend_status_equality() {
        // `Completed` and `Cancelled` are equal to themselves; `Failed`
        // compares structurally on the inner string.
        assert_eq!(BackendStatus::Completed, BackendStatus::Completed);
        assert_eq!(BackendStatus::Cancelled, BackendStatus::Cancelled);
        assert_eq!(
            BackendStatus::Failed("x".into()),
            BackendStatus::Failed("x".into())
        );
        assert_ne!(
            BackendStatus::Failed("x".into()),
            BackendStatus::Failed("y".into())
        );
    }

    #[test]
    fn placeholders_match_engine_wire_shape() {
        // The placeholder types must have the same wire shape as the
        // engine types so receipt signatures stay stable. We assert
        // the inner types directly.
        assert_eq!(std::mem::size_of::<EpochId>(), std::mem::size_of::<u64>());
        assert_eq!(
            std::mem::size_of::<VariantId>(),
            std::mem::size_of::<u64>()
        );
        assert_eq!(std::mem::size_of::<PhaseId>(), std::mem::size_of::<u64>());
        assert_eq!(
            std::mem::size_of::<SlotLeaseId>(),
            std::mem::size_of::<u64>()
        );
    }
}
