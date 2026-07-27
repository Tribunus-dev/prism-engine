//! Worker-side inference phase state machine — the inference-side
//! transitions that drive the worker prefill/decode loop (`Created` →
//! `PrefillRunning` → `Decoding` → `Completed` / `Cancelled`, plus `Failed`
//! from any non-terminal phase).
//!
//! **Naming note:** Renamed from `InferenceSessionState` to
//! `WorkerInferencePhase` to avoid collision with the server-side
//! `InferenceSessionState` defined in
//! `crate::runtime::server_types::InferenceSessionState`, which is the
//! `SessionManager` lifecycle state used by `crate::runtime::session`.
//! The two state machines are at different layers: this one drives the
//! worker prefill/decode loop; the other drives the server's session
//! admission / ready / closed transitions.

/// Worker-side inference session state machine, absorbed from
/// `compute-core/src/ecs/core/session.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerInferencePhase {
    /// Session created, not yet started prefill.
    Created,
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

impl WorkerInferencePhase {
    /// Returns `true` if the session is in a terminal phase.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Cancelled | Self::Failed)
    }

    /// Returns `true` if transitioning to `next` is a legal forward move.
    ///
    /// Terminal phases reject all transitions (including to `Failed`). Failed
    /// is reachable from any non-terminal phase only.
    pub fn can_transition_to(&self, next: Self) -> bool {
        // Identity (no-op) is always permitted.
        if *self == next {
            return true;
        }

        // Terminal phases reject all non-identity transitions.
        if self.is_terminal() {
            return false;
        }

        match (*self, next) {
            (Self::Created, Self::PrefillRunning)
            | (Self::PrefillRunning, Self::Decoding)
            | (Self::Decoding, Self::Completed)
            | (Self::Decoding, Self::Cancelled)
            | (Self::PrefillRunning, Self::Cancelled)
            | (_, Self::Failed) => true,
            _ => false,
        }
    }

    /// Attempt a phase transition. Returns `Ok(())` on success or `Err`.
    pub fn transition(&self, next: Self) -> Result<(), String> {
        if self.can_transition_to(next) {
            Ok(())
        } else {
            Err(format!(
                "Invalid WorkerInferencePhase transition: {:?} → {:?}",
                self, next
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_state_initial_is_not_terminal() {
        assert!(!WorkerInferencePhase::Created.is_terminal());
    }

    #[test]
    fn inference_state_valid_transitions() {
        assert!(WorkerInferencePhase::Created
            .transition(WorkerInferencePhase::PrefillRunning)
            .is_ok());
        assert!(WorkerInferencePhase::PrefillRunning
            .transition(WorkerInferencePhase::Decoding)
            .is_ok());
        assert!(WorkerInferencePhase::Decoding
            .transition(WorkerInferencePhase::Completed)
            .is_ok());
        assert!(WorkerInferencePhase::Decoding
            .transition(WorkerInferencePhase::Cancelled)
            .is_ok());
    }

    #[test]
    fn inference_state_failed_from_non_terminal() {
        let non_terminal = [
            WorkerInferencePhase::Created,
            WorkerInferencePhase::PrefillRunning,
            WorkerInferencePhase::Decoding,
        ];
        for s in non_terminal {
            assert!(
                s.transition(WorkerInferencePhase::Failed).is_ok(),
                "Failed from {:?} should be valid",
                s,
            );
        }
    }

    #[test]
    fn inference_state_terminal_rejects_failed() {
        assert!(WorkerInferencePhase::Completed
            .transition(WorkerInferencePhase::Failed)
            .is_err());
        assert!(WorkerInferencePhase::Cancelled
            .transition(WorkerInferencePhase::Failed)
            .is_err());
    }
}
