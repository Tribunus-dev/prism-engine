//! Phase advancement system (constitutional home).
//!
//! Per the inventory v2.1, this is the canonical home for both:
//! - The engine's `phase_engine.rs` system half (step 15)
//! - The engine's `phase_runner/execution.rs` system half (step 16)
//!
//! Both move to `systems::phase_advancement`. The state half
//! (phase_engine's data types) is already in `state::phase` (step 7).
//!
//! The phase-advancement system transitions a phase through its
//! lifecycle states: `Dormant → Ready → ... → Complete`, and reads
//! the phase DAG + completed set to determine which phases are
//! ready to advance.

use crate::scheduling::state::phase::{PhaseId, PhaseLifecycleState, PhaseLifecycleTracker};

/// Advance a phase from its current state to the next on the happy
/// path. Placeholder: returns a transition; the full state-machine
/// algorithm arrives with the engine migration.
pub fn advance_phase(
    tracker: &mut PhaseLifecycleTracker,
    phase_id: &str,
) -> Result<(), String> {
    let current = tracker.state(phase_id);
    let next = match current {
        PhaseLifecycleState::Dormant => PhaseLifecycleState::Ready,
        PhaseLifecycleState::Ready => PhaseLifecycleState::ResidencyPending,
        PhaseLifecycleState::ResidencyPending => PhaseLifecycleState::LeasePending,
        PhaseLifecycleState::LeasePending => PhaseLifecycleState::Admitted,
        PhaseLifecycleState::Admitted => PhaseLifecycleState::Dispatched,
        PhaseLifecycleState::Dispatched => PhaseLifecycleState::AwaitingCompletion,
        PhaseLifecycleState::AwaitingCompletion => PhaseLifecycleState::Validating,
        PhaseLifecycleState::Validating => PhaseLifecycleState::Publishing,
        PhaseLifecycleState::Publishing => PhaseLifecycleState::Complete,
        // Terminal states do not advance.
        _ => {
            return Err(format!(
                "phase {phase_id} in terminal state {current:?}, cannot advance"
            ))
        }
    };
    tracker.transition(phase_id, next)
}

/// Determine the set of phases ready to advance, given the
/// completed-phase set. The full DAG-aware algorithm arrives with
/// the engine migration. Placeholder returns the empty set.
pub fn ready_to_advance(_completed: &[PhaseId]) -> Vec<PhaseId> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_phase_walks_happy_path() {
        // Architectural invariant: advance_phase walks the happy
        // path: Dormant → Ready → ... → Complete.
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        for _ in 0..9 {
            assert!(advance_phase(&mut t, "p1").is_ok());
        }
        assert_eq!(t.state("p1"), PhaseLifecycleState::Complete);
    }

    #[test]
    fn advance_phase_terminal_state_errors() {
        // Architectural invariant: a phase in a terminal state
        // cannot advance. advance_phase returns an error.
        let mut t = PhaseLifecycleTracker::new();
        t.register("p1");
        let _ = t.transition("p1", PhaseLifecycleState::Cancelled);
        let result = advance_phase(&mut t, "p1");
        assert!(result.is_err());
    }

    #[test]
    fn ready_to_advance_placeholder_is_empty() {
        // Architectural invariant: the placeholder returns the
        // empty list. The full DAG-aware algorithm arrives with
        // the engine migration.
        let completed: Vec<PhaseId> = vec![];
        let ready = ready_to_advance(&completed);
        assert!(ready.is_empty());
    }
}
