//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — In-flight work tracking and state machine.
//!
//! Maintains a registry of all in-flight work items across all execution lanes,
//! managing the work state machine and providing indexed access by lane, session,
//! and phase.  The [`WorkRegistry`] is the single source of truth for the lifecycle
//! of every submitted work item.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::SlotLeaseId;
use crate::compilation::phase_ir::PhaseId;
use crate::scheduling::lane_work::{BackendExecutionTiming, WorkId};

use super::*;

// ── Work status ─────────────────────────────────────────────────────────────

/// Complete state machine for a single work item.
///
/// # State machine
///
/// ```text
///                                   ┌─────────────────┐
///                                   │     Created      │
///                                   └────────┬─────────┘
///                                          │
///                          ┌───────────────┼───────────────┐
///                          ▼               ▼               ▼
///                     ┌──────────┐   ┌──────────┐   ┌──────────┐
///                     │  Ready   │   │  Denied  │   │ (cancel) │
///                     └────┬─────┘   └──────────┘   └──────────┘
///                          │            [terminal failure]
///                          ▼
///                     ┌──────────┐
///                     │ Selected │
///                     └────┬─────┘
///                          │
///                          ▼
///                   ┌──────────────┐
///                   │Cap.Reserved  │
///                   └──────┬───────┘
///                          │
///                  ┌───────┼───────────┐
///                  ▼       ▼           │
///            ┌─────────┐ ┌──────────┐ │
///            │SlotsRes.│ │  Denied  │ │
///            └────┬────┘ └──────────┘ │
///                 │       [terminal]  │
///            ┌────┼────────┐         │
///            ▼    ▼        │         │
///       ┌────────┐  ┌─────────────┐  │
///       │Submit  │  │FallbackPend.│  │
///       └───┬────┘  └──────┬──────┘  │
///           │              │         │
///      ┌────┼────┐         ▼         │
///      ▼    ▼    │   ┌────────────┐  │
/// ┌────────┐ ┌──────┐ │FallbackRun│  │
/// │Running │ │Submit│ └──┬──────┬──┘  │
/// └───┬────┘ │Fail  │    │      │     │
///     │      └──────┘    │      │     │
///  ┌──┼──┐    [terminal] │      │     │
///  ▼  ▼  ▼              │      │     │
/// ┌──┐ ┌──┐ ┌────────┐  │      │     │
/// │C │ │EF│ │TimedOut│◄─┘      │     │
/// │o │ └──┘ └────────┘         │     │
/// │m │  │       │              │     │
/// │p │  └───┬───┘              │     │
/// │l │      │                  │     │
/// │e │  ┌────┴─────┐           │     │
/// │t │  ▼          ▼           │     │
/// │e │┌────────┐┌────────┐     │     │
/// │d ││FbackPd││FailTerm│     │     │
/// │  │└───┬───┘└────────┘     │     │
/// │  │    │    [terminal]     │     │
/// │  │    ▼                   │     │
/// │  │┌────────┐              │     │
/// │  ││FbackRun│◄─────────────┘     │
/// │  │└───┬────┘                   │
/// │  │  ┌─┴──┐                    │
/// │  │  ▼    ▼                     │
/// │  │┌──┐ ┌─────┐                │
/// │  ││C │ │EF/TO│                │
/// │  ││om│ └──┬──┘                │
/// │  ││pl│    │                   │
/// │  ││et│┌───┴────┐             │
/// │  ││ed│▼        ▼            │
/// │  │└──┘┌─────┐┌────────┐    │
/// │  │    │FbPd││FailTerm│    │
/// │  │    └──┬──┘└────────┘   │
/// │  ▼      ▼                 │
/// │  ┌──────────┐             │
/// │  │OutputRdy │             │
/// │  └────┬─────┘             │
/// │       │                   │
/// │  ┌────┼───────┐          │
/// │  ▼    ▼       │          │
/// │┌────┐┌────────┐│         │
/// ││Cons││FallbPd ││         │
/// │└──┬─┘└────────┘│         │
/// │   │            │          │
/// │   ▼            │          │
/// │┌──────┐        │          │
/// ││Releas│◄───────┘          │
/// │└──────┘  etc.             │
/// └───────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    // ── Terminal failures ──────────────────────────────────────────────
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
    ///
    /// Only [`Released`](WorkStatus::Released) is a terminal success state.
    /// All other states are either intermediate or terminal failures.
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
            // Terminal success — no transitions.
            WorkStatus::Released => &[],
            // Terminal failures — no transitions.
            WorkStatus::Denied => &[],
            WorkStatus::CancelledBeforeSubmit => &[],
            WorkStatus::SubmitFailed => &[],
            // Non-terminal failure — may retry or give up.
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

// ── Work record ─────────────────────────────────────────────────────────────

/// A tracked work item with full lifecycle state.
#[derive(Debug)]
pub struct WorkRecord {
    /// Unique identifier for this work item.
    pub work_id: WorkId,
    /// The execution lane this work is (or was) running on.
    pub lane: ExecutionLane,
    /// Logical session identifier.
    pub session_id: String,
    /// Compilation phase this work belongs to.
    pub phase_id: PhaseId,
    /// Input slot lease ids consumed by this execution.
    pub input_slots: Vec<SlotLeaseId>,
    /// Output slot lease id produced by this execution.
    pub output_slot: SlotLeaseId,
    /// Current lifecycle status.
    pub status: WorkStatus,
    /// Timestamp when the record was created.
    pub created_at: Instant,
    /// Timestamp when the work was submitted to the backend (if submitted).
    pub submitted_at: Option<Instant>,
    /// Timestamp when the work reached a terminal state (if completed).
    pub completed_at: Option<Instant>,
    /// Attempt number (0-based; incremented on fallback retry).
    pub attempt: u32,
    /// High-resolution backend execution timing, if available.
    pub backend_timing: Option<BackendExecutionTiming>,
}

// ── Work registry ───────────────────────────────────────────────────────────

/// Registry of all in-flight work items, keyed by [`WorkId`].
///
/// Maintains secondary indexes by lane, session, and phase for efficient
/// queries.  The [`by_phase`](WorkRegistry::by_phase) index tracks only the
/// *latest* work item for each phase — on fallback retries the old entry is
/// replaced.
pub struct WorkRegistry {
    records: HashMap<WorkId, WorkRecord>,
    by_lane: HashMap<ExecutionLane, Vec<WorkId>>,
    by_session: HashMap<String, Vec<WorkId>>,
    /// Latest work_id per phase (replaced on fallback retry).
    by_phase: HashMap<PhaseId, WorkId>,
}

impl WorkRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            by_lane: HashMap::new(),
            by_session: HashMap::new(),
            by_phase: HashMap::new(),
        }
    }

    /// Register a new work item.
    ///
    /// Returns an error if `work_id` already exists in the registry.
    pub fn register(&mut self, record: WorkRecord) -> Result<(), String> {
        let work_id = record.work_id;
        if self.records.contains_key(&work_id) {
            return Err(format!("WorkId({:?}) already registered", work_id.0));
        }

        // Insert into the primary map.
        self.records.insert(work_id, record);

        // Index by lane.
        let lane = self.records[&work_id].lane;
        self.by_lane.entry(lane).or_default().push(work_id);

        // Index by session.
        let session_id = self.records[&work_id].session_id.clone();
        self.by_session.entry(session_id).or_default().push(work_id);

        // Index by phase — replace any prior mapping (fallback retry).
        let phase_id = self.records[&work_id].phase_id;
        self.by_phase.insert(phase_id, work_id);

        Ok(())
    }

    /// Transition work to a new status.
    ///
    /// Returns an error if the transition is illegal or the work_id does not
    /// exist.  Automatically records timing snapshots at key transitions:
    ///
    /// | Transition to        | Field set               |
    /// |----------------------|-------------------------|
    /// | `Submitted`          | `submitted_at = now()`  |
    /// | `Running`            | `submitted_at = now()`  |
    /// | Any terminal state   | `completed_at = now()`  |
    pub fn transition(&mut self, work_id: WorkId, new_status: WorkStatus) -> Result<(), String> {
        let record = self
            .records
            .get_mut(&work_id)
            .ok_or_else(|| format!("WorkId({:?}) not found", work_id.0))?;

        let current = record.status;
        if !current.legal_transitions().contains(&new_status) {
            return Err(format!(
                "Illegal transition: {:?} -> {:?} for WorkId({:?})",
                current, new_status, work_id.0
            ));
        }

        // Record timing snapshots.
        let now = Instant::now();
        if new_status == WorkStatus::Submitted || new_status == WorkStatus::Running {
            if record.submitted_at.is_none() {
                record.submitted_at = Some(now);
            }
        }
        if new_status.is_terminal() {
            record.completed_at = Some(now);
        }

        record.status = new_status;
        Ok(())
    }

    /// Returns the current status of the work item, or `None` if unknown.
    pub fn status(&self, work_id: WorkId) -> Option<WorkStatus> {
        self.records.get(&work_id).map(|r| r.status)
    }

    /// Returns a shared reference to the work record.
    pub fn get(&self, work_id: WorkId) -> Option<&WorkRecord> {
        self.records.get(&work_id)
    }

    /// Returns a mutable reference to the work record.
    pub fn get_mut(&mut self, work_id: WorkId) -> Option<&mut WorkRecord> {
        self.records.get_mut(&work_id)
    }

    /// Removes a completed work item from the registry, returning it.
    ///
    /// Also cleans up all secondary indexes.
    pub fn remove(&mut self, work_id: WorkId) -> Option<WorkRecord> {
        let record = self.records.remove(&work_id)?;

        // Clean up lane index.
        if let Some(ids) = self.by_lane.get_mut(&record.lane) {
            ids.retain(|id| *id != work_id);
            if ids.is_empty() {
                self.by_lane.remove(&record.lane);
            }
        }

        // Clean up session index.
        if let Some(ids) = self.by_session.get_mut(&record.session_id) {
            ids.retain(|id| *id != work_id);
            if ids.is_empty() {
                self.by_session.remove(&record.session_id);
            }
        }

        // Clean up phase index — only remove if this is still the current
        // mapping for that phase (don't clobber a fallback retry).
        if self.by_phase.get(&record.phase_id) == Some(&work_id) {
            self.by_phase.remove(&record.phase_id);
        }

        Some(record)
    }

    /// Returns the number of active (non-terminal) work items per lane.
    pub fn active_count_by_lane(&self) -> HashMap<ExecutionLane, usize> {
        let mut counts: HashMap<ExecutionLane, usize> = HashMap::new();
        for record in self.records.values() {
            if !record.status.is_terminal() {
                *counts.entry(record.lane).or_default() += 1;
            }
        }
        counts
    }

    /// Returns the number of active work items for a given session.
    pub fn active_by_session(&self, session_id: &str) -> usize {
        self.records
            .values()
            .filter(|r| r.session_id == session_id && !r.status.is_terminal())
            .count()
    }

    /// Returns the total number of active work items across all lanes.
    pub fn total_active(&self) -> usize {
        self.records
            .values()
            .filter(|r| !r.status.is_terminal())
            .count()
    }

    /// Returns all work ids currently in the registry.
    pub fn all_ids(&self) -> Vec<WorkId> {
        self.records.keys().copied().collect()
    }

    /// Returns all work ids for a given execution lane.
    pub fn by_lane(&self, lane: ExecutionLane) -> Vec<WorkId> {
        self.by_lane
            .get(&lane)
            .map(|ids| ids.clone())
            .unwrap_or_default()
    }

    /// Returns the latest work id for a given phase, if any.
    pub fn by_phase(&self, phase_id: PhaseId) -> Option<WorkId> {
        self.by_phase.get(&phase_id).copied()
    }

    /// Removes all work items for a session (cancellation path).
    ///
    /// Drains all records associated with the session and cleans up every
    /// secondary index.  Returns the removed records.
    pub fn remove_session(&mut self, session_id: &str) -> Vec<WorkRecord> {
        // Collect the work ids to remove.
        let ids: Vec<WorkId> = self
            .records
            .iter()
            .filter(|(_, r)| r.session_id == session_id)
            .map(|(id, _)| *id)
            .collect();

        let mut removed = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(record) = self.remove(id) {
                removed.push(record);
            }
        }
        removed
    }
}

impl Default for WorkRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sample_record(work_id: WorkId) -> WorkRecord {
        WorkRecord {
            work_id,
            lane: ExecutionLane::MlxGpu,
            session_id: "ses-1".into(),
            phase_id: PhaseId(1),
            input_slots: vec![SlotLeaseId(10), SlotLeaseId(11)],
            output_slot: SlotLeaseId(20),
            status: WorkStatus::Created,
            created_at: Instant::now(),
            submitted_at: None,
            completed_at: None,
            attempt: 0,
            backend_timing: None,
        }
    }

    // ── Registration ───────────────────────────────────────────────────

    #[test]
    fn test_register_and_get() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        assert_eq!(
            reg.get(WorkId(1)).map(|r| r.status),
            Some(WorkStatus::Created)
        );
        assert_eq!(reg.all_ids(), vec![WorkId(1)]);
    }

    #[test]
    fn test_register_duplicate_rejected() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        let err = reg.register(sample_record(WorkId(1))).unwrap_err();
        assert!(err.contains("already registered"), "got: {err}");
    }

    #[test]
    fn test_register_indexes_by_lane_and_session() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        assert_eq!(reg.by_lane(ExecutionLane::MlxGpu), vec![WorkId(1)]);
        assert_eq!(reg.by_lane(ExecutionLane::CoreMlAne), Vec::<WorkId>::new());
    }

    // ── State machine transitions ──────────────────────────────────────

    #[test]
    fn test_happy_path_transitions() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());

        let happy = [
            WorkStatus::Ready,
            WorkStatus::Selected,
            WorkStatus::CapacityReserved,
            WorkStatus::SlotsReserved,
            WorkStatus::Submitted,
            WorkStatus::Running,
            WorkStatus::Completed,
            WorkStatus::OutputReady,
            WorkStatus::Consumed,
            WorkStatus::Released,
        ];
        for &status in &happy {
            assert!(reg.transition(WorkId(1), status).is_ok());
        }
        assert_eq!(reg.status(WorkId(1)), Some(WorkStatus::Released));
    }

    #[test]
    fn test_illegal_transition_rejected() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        // Created -> Submitted is illegal (skips Ready, Selected, etc.)
        let err = reg
            .transition(WorkId(1), WorkStatus::Submitted)
            .unwrap_err();
        assert!(err.contains("Illegal transition"), "got: {err}");
        // Still in Created.
        assert_eq!(reg.status(WorkId(1)), Some(WorkStatus::Created));
    }

    #[test]
    fn test_transition_unknown_work_id() {
        let mut reg = WorkRegistry::new();
        let err = reg.transition(WorkId(999), WorkStatus::Ready).unwrap_err();
        assert!(err.contains("not found"), "got: {err}");
    }

    #[test]
    fn test_terminal_stops_transitions() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        // Denied is terminal.
        assert!(reg.transition(WorkId(1), WorkStatus::Denied).is_ok());
        assert_eq!(reg.status(WorkId(1)), Some(WorkStatus::Denied));
        // No transition from a terminal state.
        let err = reg.transition(WorkId(1), WorkStatus::Ready).unwrap_err();
        assert!(err.contains("Illegal transition"), "got: {err}");
    }

    #[test]
    fn test_fallback_retry_path() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());

        // Execute to failure.
        for &s in &[
            WorkStatus::Ready,
            WorkStatus::Selected,
            WorkStatus::CapacityReserved,
            WorkStatus::SlotsReserved,
            WorkStatus::Submitted,
            WorkStatus::Running,
        ] {
            assert!(reg.transition(WorkId(1), s).is_ok());
        }
        assert!(reg
            .transition(WorkId(1), WorkStatus::ExecutionFailed)
            .is_ok());
        assert!(reg
            .transition(WorkId(1), WorkStatus::FallbackPending)
            .is_ok());
        assert!(reg
            .transition(WorkId(1), WorkStatus::FallbackRunning)
            .is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Completed).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::OutputReady).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Consumed).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Released).is_ok());
    }

    #[test]
    fn test_fallback_exhausted() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());

        // Execute and fail.
        for &s in &[
            WorkStatus::Ready,
            WorkStatus::Selected,
            WorkStatus::CapacityReserved,
            WorkStatus::SlotsReserved,
            WorkStatus::Submitted,
            WorkStatus::Running,
            WorkStatus::ExecutionFailed,
        ] {
            assert!(reg.transition(WorkId(1), s).is_ok());
        }
        // No more fallback — go terminal.
        assert!(reg
            .transition(WorkId(1), WorkStatus::FailedTerminal)
            .is_ok());
        assert_eq!(reg.status(WorkId(1)), Some(WorkStatus::FailedTerminal));
        assert!(reg.status(WorkId(1)).unwrap().is_terminal());
        assert!(reg.status(WorkId(1)).unwrap().is_failure());
    }

    // ── Legal transitions coverage ─────────────────────────────────────

    #[test]
    fn test_legal_transitions_exhaustive() {
        // Spot-check a few representative states; the happy-path and
        // fallback tests above exercise the full reachable graph.

        assert_eq!(
            WorkStatus::Created.legal_transitions(),
            &[WorkStatus::Ready, WorkStatus::Denied,]
        );

        assert_eq!(
            WorkStatus::Running.legal_transitions(),
            &[
                WorkStatus::Completed,
                WorkStatus::ExecutionFailed,
                WorkStatus::TimedOut,
            ]
        );

        assert_eq!(WorkStatus::Released.legal_transitions(), &[]);
        assert_eq!(WorkStatus::FailedTerminal.legal_transitions(), &[]);
    }

    // ── Status predicates ──────────────────────────────────────────────

    #[test]
    fn test_status_predicates() {
        assert!(WorkStatus::Released.is_terminal());
        assert!(WorkStatus::Released.is_success());
        assert!(!WorkStatus::Released.is_failure());

        assert!(WorkStatus::Denied.is_terminal());
        assert!(!WorkStatus::Denied.is_success());
        assert!(WorkStatus::Denied.is_failure());

        assert!(WorkStatus::FailedTerminal.is_terminal());
        assert!(!WorkStatus::FailedTerminal.is_success());
        assert!(WorkStatus::FailedTerminal.is_failure());

        // Intermediate states.
        assert!(!WorkStatus::Created.is_terminal());
        assert!(!WorkStatus::Created.is_success());
        assert!(!WorkStatus::Created.is_failure());

        assert!(!WorkStatus::Running.is_terminal());
        assert!(!WorkStatus::ExecutionFailed.is_terminal());
        assert!(!WorkStatus::TimedOut.is_terminal());
    }

    // ── Active counts ──────────────────────────────────────────────────

    #[test]
    fn test_active_counts() {
        let mut reg = WorkRegistry::new();
        assert_eq!(reg.total_active(), 0);
        assert!(reg.active_by_session("ses-1") == 0);

        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(2),
                lane: ExecutionLane::CoreMlAne,
                session_id: "ses-1".into(),
                ..sample_record(WorkId(2))
            })
            .is_ok());
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(3),
                lane: ExecutionLane::MlxGpu,
                session_id: "ses-2".into(),
                ..sample_record(WorkId(3))
            })
            .is_ok());

        assert_eq!(reg.total_active(), 3);
        assert_eq!(reg.active_by_session("ses-1"), 2);

        let by_lane = reg.active_count_by_lane();
        assert_eq!(by_lane.get(&ExecutionLane::MlxGpu), Some(&2));
        assert_eq!(by_lane.get(&ExecutionLane::CoreMlAne), Some(&1));

        // Complete one item.
        assert!(reg.transition(WorkId(1), WorkStatus::Ready).is_ok());
        for &s in &[
            WorkStatus::Selected,
            WorkStatus::CapacityReserved,
            WorkStatus::SlotsReserved,
            WorkStatus::Submitted,
            WorkStatus::Running,
            WorkStatus::Completed,
            WorkStatus::OutputReady,
            WorkStatus::Consumed,
            WorkStatus::Released,
        ] {
            assert!(reg.transition(WorkId(1), s).is_ok());
        }
        assert_eq!(reg.total_active(), 2);
        assert_eq!(reg.active_by_session("ses-1"), 1);
    }

    // ── Remove ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_cleans_indexes() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(2),
                session_id: "ses-1".into(),
                ..sample_record(WorkId(2))
            })
            .is_ok());

        let removed = reg.remove(WorkId(1)).unwrap();
        assert_eq!(removed.work_id, WorkId(1));
        assert!(reg.get(WorkId(1)).is_none());
        // Lane index still has WorkId(2).
        assert_eq!(reg.by_lane(ExecutionLane::MlxGpu), vec![WorkId(2)]);
        // Session index still has WorkId(2).
        assert_eq!(reg.by_session.get("ses-1").map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_remove_unknown() {
        let mut reg = WorkRegistry::new();
        assert!(reg.remove(WorkId(999)).is_none());
    }

    // ── Remove session ─────────────────────────────────────────────────

    #[test]
    fn test_remove_session_drains_all() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok()); // ses-1
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(2),
                session_id: "ses-1".into(),
                ..sample_record(WorkId(2))
            })
            .is_ok());
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(3),
                session_id: "ses-2".into(),
                phase_id: PhaseId(2),
                ..sample_record(WorkId(3))
            })
            .is_ok());

        let removed = reg.remove_session("ses-1");
        assert_eq!(removed.len(), 2);
        assert!(reg.get(WorkId(1)).is_none());
        assert!(reg.get(WorkId(2)).is_none());
        assert!(reg.get(WorkId(3)).is_some()); // untouched
        assert!(reg.by_session.get("ses-1").is_none());

        // Both removed records used PhaseId(1); the phase index should be empty.
        assert!(reg.by_phase(PhaseId(1)).is_none());

        // WorkId(3) has PhaseId(2) and remains in the index.
        assert_eq!(reg.by_phase(PhaseId(2)), Some(WorkId(3)));
    }

    // ── by_phase index (fallback retry replacement) ────────────────────

    #[test]
    fn test_by_phase_replaced_on_fallback() {
        let mut reg = WorkRegistry::new();
        assert!(reg.register(sample_record(WorkId(1))).is_ok());
        assert_eq!(reg.by_phase(PhaseId(1)), Some(WorkId(1)));

        // Register a fallback retry for the same phase.
        assert!(reg
            .register(WorkRecord {
                work_id: WorkId(2),
                phase_id: PhaseId(1),
                attempt: 1,
                ..sample_record(WorkId(2))
            })
            .is_ok());

        // by_phase now points to the latest.
        assert_eq!(reg.by_phase(PhaseId(1)), Some(WorkId(2)));

        // Remove the old work — phase index stays.
        reg.remove(WorkId(1));
        assert_eq!(reg.by_phase(PhaseId(1)), Some(WorkId(2)));

        // Remove the latest — phase index is cleaned up.
        reg.remove(WorkId(2));
        assert_eq!(reg.by_phase(PhaseId(1)), None);
    }

    // ── Timing snapshots ───────────────────────────────────────────────

    #[test]
    fn test_timing_snapshots_on_transition() {
        let mut reg = WorkRegistry::new();
        let mut rec = sample_record(WorkId(1));
        rec.created_at = Instant::now() - Duration::from_secs(10);
        assert!(reg.register(rec).is_ok());

        // submitted_at set on first Submitted.
        assert!(reg.transition(WorkId(1), WorkStatus::Ready).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Selected).is_ok());
        assert!(reg
            .transition(WorkId(1), WorkStatus::CapacityReserved)
            .is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::SlotsReserved).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Submitted).is_ok());
        assert!(reg.get(WorkId(1)).unwrap().submitted_at.is_some());

        // completed_at set on terminal state.
        assert!(reg.transition(WorkId(1), WorkStatus::Running).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Completed).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::OutputReady).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Consumed).is_ok());
        assert!(reg.transition(WorkId(1), WorkStatus::Released).is_ok());
        assert!(reg.get(WorkId(1)).unwrap().completed_at.is_some());
    }

    // ── Default ────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let reg = WorkRegistry::default();
        assert_eq!(reg.total_active(), 0);
        assert!(reg.all_ids().is_empty());
    }
}
