//! Phase cancellation system (constitutional home, system half).
//!
//! Per the inventory v2.1 step 29, this is the system half of
//! `phase_cancellation`. The state half (CancellationEvidence,
// CancellationChecker, CancellationPolicy) is already in
//! `state::phase_cancellation` (step 9). The system half runs
//! the cancellation checks and signals phase-termination.
//!
//! Placeholder: the full engine migration arrives with step 29.

use crate::scheduling::state::phase_cancellation::{
    CancellationChecker, CancellationEvidence, CancellationPolicy,
};

/// Run the cancellation check for a session. The checker
/// encapsulates the shared flag + deadline; the system consults
/// it to decide whether the session should stop.
pub fn should_stop_session(checker: &CancellationChecker) -> bool {
    checker.should_stop()
}

/// Decide whether to cancel a specific phase. Returns the
/// cancellation evidence. Placeholder: delegates to the checker.
pub fn check_phase(
    checker: &CancellationChecker,
    phase_id: &str,
) -> CancellationEvidence {
    use crate::scheduling::state::phase::PhaseId;
    let pid = PhaseId(phase_id.to_string());
    checker.check_phase(&pid, crate::scheduling::state::phase_cancellation::CancellationClass::Preemptible)
}

/// Apply a default cancellation policy.
pub fn default_policy() -> CancellationPolicy {
    CancellationPolicy::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn fresh_checker_does_not_signal_stop() {
        let flag = Arc::new(AtomicBool::new(false));
        let checker = CancellationChecker::new(flag);
        assert!(!should_stop_session(&checker));
    }

    #[test]
    fn cancelled_flag_signals_stop() {
        let flag = Arc::new(AtomicBool::new(true));
        let checker = CancellationChecker::new(flag);
        assert!(should_stop_session(&checker));
    }

    #[test]
    fn check_phase_returns_evidence() {
        // Architectural invariant: check_phase returns a
        // CancellationEvidence whose any_cancelled() reflects the
        // shared flag's state.
        let flag = Arc::new(AtomicBool::new(true));
        let checker = CancellationChecker::new(flag);
        let ev = check_phase(&checker, "p1");
        assert!(ev.any_cancelled());
    }

    #[test]
    fn default_policy_is_permissive() {
        // Architectural invariant: the default policy allows
        // preemptible interrupts and waits for non-preemptible
        // completion.
        let p = default_policy();
        assert!(p.allow_preemptible_interrupt);
        assert!(p.wait_for_non_preemptible);
    }
}
