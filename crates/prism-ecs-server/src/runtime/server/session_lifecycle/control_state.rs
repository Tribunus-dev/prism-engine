//! Host-side session state machine — the control-side transitions for a
//! generation session (`Created` → `Admitted` → `Submitted` → `PrefillRunning`
//! → `Decoding` → `Completed` / `Cancelled`, plus the legacy `PrefillReady`
//! path and `Failed` from any non-terminal state).

/// Host-side session state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlSessionState {
    /// Session created, pending admission.
    Created,
    /// Session admitted by the engine, awaiting worker submission.
    Admitted,
    /// Session submitted to worker, pending prefill execution.
    Submitted,
    /// Prefill input is available and ready to start (legacy path — kept for
    /// compatibility with callers that bypass admission).
    PrefillReady,
    /// Prefill is actively running.
    PrefillRunning,
    /// Autoregressive decoding loop is running.
    Decoding,
    /// Generation completed normally (EOS or max_tokens reached).
    Completed,
    /// Generation was externally cancelled.
    Cancelled,
    /// Generation failed with an error.
    Failed,
}

impl ControlSessionState {
    /// Returns `true` if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }

    /// Returns `true` if transitioning to `next` is a legal forward move.
    ///
    /// Terminal states reject all transitions (including to `Failed`). Failed
    /// is reachable from any non-terminal state only.
    pub fn can_transition_to(&self, next: Self) -> bool {
        use ControlSessionState::*;

        // Identity (no-op) is always permitted.
        if *self == next {
            return true;
        }

        // Terminal states reject all non-identity transitions.
        if self.is_terminal() {
            return false;
        }

        match (*self, next) {
            // Mainline path.
            (Created, Admitted)
            | (Admitted, Submitted)
            | (Submitted, PrefillRunning)
            | (PrefillRunning, Decoding)
            | (Decoding, Completed) => true,
            // Cancellation paths.
            (Decoding, Cancelled)
            | (PrefillReady, Cancelled)
            | (PrefillRunning, Cancelled)
            | (Admitted, Cancelled)
            | (Submitted, Cancelled) => true,
            // Legacy: PrefillReady can jump into the mainline at PrefillRunning.
            (PrefillReady, PrefillRunning) => true,
            // Forward to PrefillReady from Created / Admitted.
            (Created, PrefillReady) | (Admitted, PrefillReady) => true,
            // Failed from any non-terminal.
            (_, Failed) => true,
            _ => false,
        }
    }

    /// Attempt a state transition. Returns `Ok(())` on success or `Err` with
    /// a description of the invalid transition.
    pub fn transition(&self, next: Self) -> Result<(), String> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(format!("Invalid state transition: {:?} → {:?}", self, next))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_state_initial_is_not_terminal() {
        assert!(!ControlSessionState::Created.is_terminal());
    }

    #[test]
    fn control_state_terminal_set() {
        assert!(ControlSessionState::Completed.is_terminal());
        assert!(ControlSessionState::Cancelled.is_terminal());
        assert!(ControlSessionState::Failed.is_terminal());
        assert!(!ControlSessionState::Decoding.is_terminal());
    }

    #[test]
    fn control_state_valid_transitions() {
        // Classic legacy path
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::PrefillReady)
            .is_ok());
        assert!(ControlSessionState::PrefillReady
            .transition(ControlSessionState::PrefillRunning)
            .is_ok());
        assert!(ControlSessionState::PrefillRunning
            .transition(ControlSessionState::Decoding)
            .is_ok());
        assert!(ControlSessionState::Decoding
            .transition(ControlSessionState::Completed)
            .is_ok());
        // Cancellation paths
        assert!(ControlSessionState::Decoding
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        assert!(ControlSessionState::PrefillReady
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        assert!(ControlSessionState::PrefillRunning
            .transition(ControlSessionState::Cancelled)
            .is_ok());
        // New admission path
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Admitted)
            .is_ok());
        assert!(ControlSessionState::Admitted
            .transition(ControlSessionState::Submitted)
            .is_ok());
        assert!(ControlSessionState::Submitted
            .transition(ControlSessionState::PrefillRunning)
            .is_ok());
    }

    #[test]
    fn control_state_failed_from_non_terminal() {
        let non_terminal = [
            ControlSessionState::Created,
            ControlSessionState::Admitted,
            ControlSessionState::Submitted,
            ControlSessionState::PrefillReady,
            ControlSessionState::PrefillRunning,
            ControlSessionState::Decoding,
        ];
        for s in non_terminal {
            assert!(
                s.transition(ControlSessionState::Failed).is_ok(),
                "Failed transition should be valid from {:?}",
                s,
            );
        }
    }

    #[test]
    fn control_state_terminal_rejects_failed() {
        assert!(ControlSessionState::Completed
            .transition(ControlSessionState::Failed)
            .is_err());
        assert!(ControlSessionState::Cancelled
            .transition(ControlSessionState::Failed)
            .is_err());
        assert!(ControlSessionState::Failed
            .transition(ControlSessionState::Failed)
            .is_ok());
    }

    #[test]
    fn control_state_invalid_transitions() {
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Decoding)
            .is_err());
        assert!(ControlSessionState::Created
            .transition(ControlSessionState::Completed)
            .is_err());
        assert!(ControlSessionState::Completed
            .transition(ControlSessionState::PrefillReady)
            .is_err());
        assert!(ControlSessionState::Cancelled
            .transition(ControlSessionState::PrefillReady)
            .is_err());
        assert!(ControlSessionState::Failed
            .transition(ControlSessionState::Created)
            .is_err());
    }
}
