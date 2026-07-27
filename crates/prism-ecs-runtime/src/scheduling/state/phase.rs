//! Phase state (constitutional home).
//!
//! This is the constitutional home for the phase lifecycle state machine
//! and the per-phase state tracker.
//!
//! # Authority
//!
//! Every type in this module is **scheduling state** in the C bucket.
//! A `PhaseLifecycleState` transition is staged through
//! `ConstitutionalWorldTxn`; the runtime completion-reconciliation
//! system is the only producer of transitions to terminal states.
//!
//! Per the inventory v2.1 step 7, this file merges the engine's
//! `phase_engine.rs` and `phase_engine_state.rs` into a single
//! constitutional home. The legacy duplicates are deleted in step 58.
//!
//! # Placeholder engine types
//!
//! `PhaseId` is currently `compute-core::ecs::compute_image::phase_graph::PhaseId`
//! (a type alias for `String` in the engine, with another `PhaseId`
//! type alias for `u64` in `contracts/mod.rs`, and a `PhaseId(pub String)`
//! struct in `compute_image/phase_graph.rs`). The constitutional home
//! uses a placeholder newtype (`PhaseId(String)`) matching the
//! `compute_image::phase_graph` shape; when the engine's phase graph
//! types migrate, the placeholder is replaced.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::compute_image::phase_graph::PhaseId`
/// (which is `pub struct PhaseId(pub String)` in the engine).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhaseId(pub String);

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhase`.
#[derive(Debug, Clone, Default)]
pub struct EmittedPhasePlaceholder {
    pub phase_id: String,
}

/// Re-export of the placeholder under the engine's name.
pub type EmittedPhase = EmittedPhasePlaceholder;

// ---------------------------------------------------------------------------
// PhaseLifecycleState
// ---------------------------------------------------------------------------

/// Phase lifecycle state.
///
/// A phase is admitted to the scheduler in `Dormant`; it advances
/// through `Ready → ResidencyPending → LeasePending → Admitted →
/// Dispatched → AwaitingCompletion → Validating → Publishing →
/// Complete` on the happy path. Failure paths land in
/// `Rejected | Cancelled | TimedOut | FailedBeforePublication |
/// FailedAfterTentativeState | RolledBack | FallbackComplete |
/// Quarantined`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhaseLifecycleState {
    // Happy path
    Dormant,
    Ready,
    ResidencyPending,
    LeasePending,
    Admitted,
    Dispatched,
    AwaitingCompletion,
    Validating,
    Publishing,
    Complete,
    // Failure states
    Rejected,
    Cancelled,
    TimedOut,
    FailedBeforePublication,
    FailedAfterTentativeState,
    RolledBack,
    FallbackPending,
    FallbackComplete,
    Quarantined,
}

impl PhaseLifecycleState {
    /// Returns `true` if the state is terminal (no further transitions
    /// are expected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            PhaseLifecycleState::Complete
                | PhaseLifecycleState::Rejected
                | PhaseLifecycleState::Cancelled
                | PhaseLifecycleState::TimedOut
                | PhaseLifecycleState::FailedBeforePublication
                | PhaseLifecycleState::FailedAfterTentativeState
                | PhaseLifecycleState::RolledBack
                | PhaseLifecycleState::FallbackComplete
                | PhaseLifecycleState::Quarantined
        )
    }

    /// Returns `true` if the state represents a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            PhaseLifecycleState::Complete | PhaseLifecycleState::FallbackComplete
        )
    }

    /// Returns `true` if the phase may proceed past this state in
    /// the happy path.
    pub fn can_proceed(&self) -> bool {
        matches!(
            self,
            PhaseLifecycleState::Complete | PhaseLifecycleState::FallbackComplete
        )
    }
}

// ---------------------------------------------------------------------------
// RuntimeWorkItemHandle
// ---------------------------------------------------------------------------

/// A handle for a RuntimeWorkItem created by the engine for a phase.
#[derive(Debug, Clone)]
pub struct RuntimeWorkItemHandle {
    pub phase_id: PhaseId,
    pub request_id: u64,
    pub lane: String,
    pub layer_index: Option<usize>,
    pub artifact_id: Option<String>,
    pub required_weight_set: Option<String>,
    pub deadline: Option<std::time::Instant>,
}

impl RuntimeWorkItemHandle {
    pub fn new(phase_id: PhaseId, request_id: u64) -> Self {
        Self {
            phase_id,
            request_id,
            lane: String::new(),
            layer_index: None,
            artifact_id: None,
            required_weight_set: None,
            deadline: None,
        }
    }
}

// ---------------------------------------------------------------------------
// PhaseLifecycleTracker
// ---------------------------------------------------------------------------

/// Phase lifecycle tracker — maps phase IDs to their lifecycle states.
///
/// Uses `BTreeMap` (not `HashMap`) so the iteration order over
/// `states` is stable. The aggregate `all_complete` check visits
/// every state; stable order means the result is deterministic
/// across runs.
#[derive(Debug, Clone, Default)]
pub struct PhaseLifecycleTracker {
    states: BTreeMap<String, PhaseLifecycleState>,
    #[allow(dead_code)]
    activation_generations: BTreeMap<String, u64>,
}

impl PhaseLifecycleTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, phase_id: &str) {
        self.states
            .entry(phase_id.to_string())
            .or_insert(PhaseLifecycleState::Dormant);
    }

    /// Apply a state transition.
    ///
    /// Returns `Ok(())` if the transition is valid; otherwise an error
    /// describing the invalid transition. Terminal-failure targets
    /// (`is_terminal()`) are always allowed from any state.
    pub fn transition(
        &mut self,
        phase_id: &str,
        to: PhaseLifecycleState,
    ) -> Result<(), String> {
        let current = self
            .states
            .get(phase_id)
            .copied()
            .unwrap_or(PhaseLifecycleState::Dormant);
        if to.is_terminal() {
            self.states.insert(phase_id.to_string(), to);
            return Ok(());
        }
        match (current, to) {
            (PhaseLifecycleState::Dormant, PhaseLifecycleState::Ready)
            | (PhaseLifecycleState::Ready, PhaseLifecycleState::ResidencyPending)
            | (PhaseLifecycleState::Ready, PhaseLifecycleState::Admitted)
            | (PhaseLifecycleState::ResidencyPending, PhaseLifecycleState::LeasePending)
            | (PhaseLifecycleState::LeasePending, PhaseLifecycleState::Admitted)
            | (PhaseLifecycleState::Admitted, PhaseLifecycleState::Dispatched)
            | (PhaseLifecycleState::Dispatched, PhaseLifecycleState::AwaitingCompletion)
            | (PhaseLifecycleState::AwaitingCompletion, PhaseLifecycleState::Validating)
            | (PhaseLifecycleState::Validating, PhaseLifecycleState::Publishing)
            | (PhaseLifecycleState::Publishing, PhaseLifecycleState::Complete)
            | (PhaseLifecycleState::FallbackPending, PhaseLifecycleState::FallbackComplete) => {
                self.states.insert(phase_id.to_string(), to);
                Ok(())
            }
            _ => Err(format!(
                "invalid phase lifecycle transition: {current:?} -> {to:?}"
            )),
        }
    }

    /// Return the current state of a phase, or `Dormant` if not registered.
    pub fn state(&self, phase_id: &str) -> PhaseLifecycleState {
        self.states
            .get(phase_id)
            .copied()
            .unwrap_or(PhaseLifecycleState::Dormant)
    }

    /// Returns `true` if every registered phase is in a terminal state.
    ///
    /// Architectural invariant: an empty tracker is NOT "all complete"
    /// (the question is ill-posed when no phases are registered).
    /// Callers that need the empty case should check
    /// `states.is_empty()` first.
    pub fn all_complete(&self) -> bool {
        !self.states.is_empty() && self.states.values().all(|s| s.is_terminal())
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `phase` state.

    use super::*;

    #[test]
    fn happy_path_transitions() {
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        for to in [
            PhaseLifecycleState::Ready,
            PhaseLifecycleState::Admitted,
            PhaseLifecycleState::Dispatched,
            PhaseLifecycleState::AwaitingCompletion,
            PhaseLifecycleState::Validating,
            PhaseLifecycleState::Publishing,
            PhaseLifecycleState::Complete,
        ] {
            assert!(t.transition("p1", to).is_ok(), "happy transition to {to:?}");
        }
    }

    #[test]
    fn invalid_transition_is_rejected() {
        // Architectural invariant: a phase can only move along the
        // state machine. Skipping a step (e.g. Dormant → Dispatched)
        // is an error.
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        assert!(t.transition("p1", PhaseLifecycleState::Dispatched).is_err());
    }

    #[test]
    fn terminal_failure_is_always_allowed() {
        // Architectural invariant: a phase can always be moved to a
        // terminal-failure state, regardless of current state. This
        // lets the runtime cancel / reject any in-flight phase.
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        assert!(t.transition("p1", PhaseLifecycleState::Cancelled).is_ok());
        assert_eq!(t.state("p1"), PhaseLifecycleState::Cancelled);
    }

    #[test]
    fn all_complete_only_when_every_state_is_terminal() {
        // Architectural invariant: `all_complete` returns true iff
        // every registered phase is in a terminal state. An empty
        // tracker returns false (no phases means the question is
        // ill-posed; we conservatively answer "not complete").
        let mut t = PhaseLifecycleTracker::new();
        assert!(!t.all_complete());
        t.register("p1");
        t.register("p2");
        assert!(!t.all_complete());
        let _ = t.transition("p1", PhaseLifecycleState::Cancelled);
        let _ = t.transition("p2", PhaseLifecycleState::Complete);
        assert!(t.all_complete());
    }

    #[test]
    fn all_complete_false_when_one_phase_not_terminal() {
        // Architectural invariant: a single non-terminal phase makes
        // the whole tracker "not all complete", even if every other
        // phase is terminal.
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        t.register("p2");
        let _ = t.transition("p1", PhaseLifecycleState::Complete);
        let _ = t.transition("p2", PhaseLifecycleState::Cancelled);
        assert!(t.all_complete());
        t.register("p3");
        assert!(!t.all_complete());
    }

    #[test]
    fn terminal_states_partition_correctly() {
        // Architectural invariant: a terminal state is either a
        // success (`is_success()`) or a failure. The two are
        // mutually exclusive.
        for s in [
            PhaseLifecycleState::Complete,
            PhaseLifecycleState::Rejected,
            PhaseLifecycleState::Cancelled,
            PhaseLifecycleState::TimedOut,
            PhaseLifecycleState::FailedBeforePublication,
            PhaseLifecycleState::FailedAfterTentativeState,
            PhaseLifecycleState::RolledBack,
            PhaseLifecycleState::FallbackComplete,
            PhaseLifecycleState::Quarantined,
        ] {
            assert!(s.is_terminal());
            // Each terminal state is either success or failure, not
            // both. Complete and FallbackComplete are success; the
            // rest are failure.
            let is_success = s.is_success();
            let is_other_failure = matches!(
                s,
                PhaseLifecycleState::Rejected
                    | PhaseLifecycleState::Cancelled
                    | PhaseLifecycleState::TimedOut
                    | PhaseLifecycleState::FailedBeforePublication
                    | PhaseLifecycleState::FailedAfterTentativeState
                    | PhaseLifecycleState::RolledBack
                    | PhaseLifecycleState::Quarantined
            );
            assert!(is_success ^ is_other_failure);
        }
    }
}
