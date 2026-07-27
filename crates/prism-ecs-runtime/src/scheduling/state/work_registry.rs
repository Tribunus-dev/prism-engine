//! Work registry state (constitutional home).
//!
//! Per-work-item scheduling state: the work status state machine and
//! the cross-retry work key.
//!
//! # Authority
//!
//! `WorkStatus` is **scheduling state** in the C bucket. The
//! status transitions are staged through `ConstitutionalWorldTxn`;
//! the runtime completion-reconciliation system is the only
//! producer of transitions to terminal states.
//!
//! `WorkKey` is the identity record that survives retries and
//! fallback attempts. A `WorkKey` is associated with a work item
//! at creation; it remains stable across the work item's lifetime,
//! even when the underlying `WorkId` changes (e.g. fallback retry).
//!
//! # Placeholder engine types
//!
//! `WorkId` matches the engine's lane_work::WorkId type
//! (the engine's WorkId is `pub struct WorkId(pub u64)`). The
//! placeholder is defined here as a newtype; replaced when the
//! engine's lane_work types are unified with the constitutional
//! `lane_work` module (step 2).
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/work_registry/`
//! (registry.rs + scheduling.rs + mod.rs). The engine directory is
//! the legacy duplicate; step 58 deletes it.

use std::collections::BTreeMap;

use super::phase::PhaseId;

// ---------------------------------------------------------------------------
// WorkId
// ---------------------------------------------------------------------------

/// Unique work-item identifier. A new WorkId is allocated for each
/// physical submission; the corresponding WorkKey is stable across
/// retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkId(pub u64);

// ---------------------------------------------------------------------------
// WorkStatus
// ---------------------------------------------------------------------------

/// Complete state machine for a single work item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum WorkStatus {
    /// Initial state — work item created but not yet ready for selection.
    Created,
    /// Ready to be selected by a lane scheduler.
    Ready,
    /// Selected by a lane scheduler for execution.
    Selected,
    /// Backend capacity has been reserved for this work item.
    CapacityReserved,
    /// Activation slots have been reserved.
    SlotsReserved,
    /// Submitted to the backend for execution.
    Submitted,
    /// Currently executing on the backend.
    Running,
    /// Backend execution completed successfully.
    Completed,
    /// Output is ready for consumption by the next phase.
    OutputReady,
    /// Output has been consumed by the downstream phase.
    Consumed,
    /// Resources released — terminal success.
    Released,
    // Terminal failures
    /// Work was denied (e.g. capacity unavailable).
    Denied,
    /// Work was cancelled before submission.
    CancelledBeforeSubmit,
    /// Backend submission failed (non-retryable).
    SubmitFailed,
    /// Backend execution failed.
    ExecutionFailed,
    /// Backend execution timed out.
    TimedOut,
    /// Fallback execution is pending (alternative lane).
    FallbackPending,
    /// Fallback execution is running on an alternative lane.
    FallbackRunning,
    /// Terminal failure after all fallback attempts exhausted.
    FailedTerminal,
}

impl WorkStatus {
    /// Returns `true` if this status is a terminal (non-transitioning) state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            WorkStatus::Released
                | WorkStatus::Denied
                | WorkStatus::CancelledBeforeSubmit
                | WorkStatus::SubmitFailed
                | WorkStatus::FailedTerminal
        )
    }

    /// Returns `true` if this status represents a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, WorkStatus::Released)
    }

    /// Returns `true` if this status represents a terminal failure.
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            WorkStatus::Denied
                | WorkStatus::CancelledBeforeSubmit
                | WorkStatus::SubmitFailed
                | WorkStatus::FailedTerminal
        )
    }

    /// Returns the exhaustive set of legal transition targets from this state.
    pub fn legal_transitions(&self) -> &'static [WorkStatus] {
        match self {
            WorkStatus::Created => &[WorkStatus::Ready, WorkStatus::Denied],
            WorkStatus::Ready => &[WorkStatus::Selected, WorkStatus::CancelledBeforeSubmit],
            WorkStatus::Selected => &[WorkStatus::CapacityReserved],
            WorkStatus::CapacityReserved => &[WorkStatus::SlotsReserved, WorkStatus::Denied],
            WorkStatus::SlotsReserved => &[WorkStatus::Submitted, WorkStatus::FallbackPending],
            WorkStatus::Submitted => &[WorkStatus::Running, WorkStatus::SubmitFailed],
            WorkStatus::Running => &[
                WorkStatus::Completed,
                WorkStatus::ExecutionFailed,
                WorkStatus::TimedOut,
            ],
            WorkStatus::Completed => &[WorkStatus::OutputReady],
            WorkStatus::OutputReady => &[WorkStatus::Consumed, WorkStatus::FallbackPending],
            WorkStatus::Consumed => &[WorkStatus::Released],
            WorkStatus::Released => &[],
            WorkStatus::Denied => &[],
            WorkStatus::CancelledBeforeSubmit => &[],
            WorkStatus::SubmitFailed => &[],
            WorkStatus::ExecutionFailed => {
                &[WorkStatus::FallbackPending, WorkStatus::FailedTerminal]
            }
            WorkStatus::TimedOut => &[WorkStatus::FallbackPending, WorkStatus::FailedTerminal],
            WorkStatus::FallbackPending => &[WorkStatus::FallbackRunning],
            WorkStatus::FallbackRunning => &[
                WorkStatus::Completed,
                WorkStatus::ExecutionFailed,
                WorkStatus::TimedOut,
                WorkStatus::FailedTerminal,
            ],
            WorkStatus::FailedTerminal => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// WorkKey
// ---------------------------------------------------------------------------

/// Work identity across retries and fallback attempts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkKey {
    /// Logical session identifier.
    pub session_id: String,
    /// Request identifier within the session.
    pub request_id: String,
    /// Sequence number within the request.
    pub sequence_id: u64,
    /// Epoch identifier (for epoch-based scheduling).
    pub epoch_id: u64,
    /// Compilation phase.
    pub phase_id: PhaseId,
    /// Attempt number (0 = original, 1+ = fallback retries).
    pub attempt: u32,
}

// ---------------------------------------------------------------------------
// WorkRegistry
// ---------------------------------------------------------------------------

/// Per-work-item status map.
///
/// Uses `BTreeMap` (not `HashMap`) for stable iteration order — the
/// receipt-snapshot projection iterates the map and the result must
/// be deterministic across runs.
#[derive(Debug, Clone, Default)]
pub struct WorkRegistry {
    statuses: BTreeMap<WorkId, WorkStatus>,
}

impl WorkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: WorkId, status: WorkStatus) {
        self.statuses.insert(id, status);
    }

    pub fn get(&self, id: WorkId) -> Option<WorkStatus> {
        self.statuses.get(&id).copied()
    }

    pub fn status(&self, id: WorkId) -> WorkStatus {
        self.statuses.get(&id).copied().unwrap_or(WorkStatus::Created)
    }

    pub fn remove(&mut self, id: WorkId) -> Option<WorkStatus> {
        self.statuses.remove(&id)
    }

    pub fn len(&self) -> usize {
        self.statuses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.statuses.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `work_registry` state.

    use super::*;

    #[test]
    fn terminal_states_partition_into_success_and_failure() {
        // Architectural invariant: the five terminal states partition
        // into exactly one success (`Released`) and four failures.
        let success = [
            WorkStatus::Released,
        ];
        let failures = [
            WorkStatus::Denied,
            WorkStatus::CancelledBeforeSubmit,
            WorkStatus::SubmitFailed,
            WorkStatus::FailedTerminal,
        ];
        for s in success {
            assert!(s.is_terminal());
            assert!(s.is_success());
            assert!(!s.is_failure());
        }
        for s in failures {
            assert!(s.is_terminal());
            assert!(s.is_failure());
            assert!(!s.is_success());
        }
    }

    #[test]
    fn non_terminal_states_are_neither_success_nor_failure() {
        // Architectural invariant: non-terminal states are not
        // classified as success or failure.
        for s in [
            WorkStatus::Created,
            WorkStatus::Ready,
            WorkStatus::Selected,
            WorkStatus::CapacityReserved,
            WorkStatus::SlotsReserved,
            WorkStatus::Submitted,
            WorkStatus::Running,
            WorkStatus::Completed,
            WorkStatus::OutputReady,
            WorkStatus::Consumed,
            WorkStatus::ExecutionFailed,
            WorkStatus::TimedOut,
            WorkStatus::FallbackPending,
            WorkStatus::FallbackRunning,
        ] {
            assert!(!s.is_terminal());
            assert!(!s.is_success());
            assert!(!s.is_failure());
        }
    }

    #[test]
    fn terminal_states_have_no_outgoing_transitions() {
        // Architectural invariant: terminal states do not allow
        // further transitions. The legal_transitions() list is empty.
        for s in [
            WorkStatus::Released,
            WorkStatus::Denied,
            WorkStatus::CancelledBeforeSubmit,
            WorkStatus::SubmitFailed,
            WorkStatus::FailedTerminal,
        ] {
            assert!(
                s.legal_transitions().is_empty(),
                "{s:?} should have no outgoing transitions"
            );
        }
    }

    #[test]
    fn work_registry_insert_get_remove() {
        let mut reg = WorkRegistry::new();
        let id = WorkId(1);
        reg.insert(id, WorkStatus::Created);
        assert_eq!(reg.get(id), Some(WorkStatus::Created));
        assert_eq!(reg.status(id), WorkStatus::Created);
        reg.insert(id, WorkStatus::Ready);
        assert_eq!(reg.status(id), WorkStatus::Ready);
        assert_eq!(reg.remove(id), Some(WorkStatus::Ready));
        assert_eq!(reg.get(id), None);
        assert_eq!(reg.status(id), WorkStatus::Created); // default
    }

    #[test]
    fn work_key_distinguishes_attempts() {
        // Architectural invariant: a work key with attempt=0 and
        // the same key with attempt=1 are different (they represent
        // different physical submissions of the same logical work).
        let k1 = WorkKey {
            session_id: "s".into(),
            request_id: "r".into(),
            sequence_id: 0,
            epoch_id: 0,
            phase_id: PhaseId("p".into()),
            attempt: 0,
        };
        let mut k2 = k1.clone();
        k2.attempt = 1;
        assert_ne!(k1, k2);
    }
}
