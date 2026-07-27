//! Session outcome envelope and host-side session handle — the canonical
//! `SessionOutcome` (terminal result variants) and `GenerationControlSession`
//! (the canonical, no-MLX, no-KV-cache session record that holds identity,
//! policy state, lifecycle state, deadline tracking, and terminal outcome).

use super::control_state::ControlSessionState;

/// Outcome of a completed, cancelled, or failed generation session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionOutcome {
    /// Generation completed with the given number of tokens produced.
    Completed {
        /// Total tokens generated (excluding prompt prefix).
        token_count: u32,
    },
    /// Generation was externally cancelled.
    Cancelled {
        /// Human-readable reason for cancellation.
        reason: String,
    },
    /// Generation failed with an error.
    Failed {
        /// Machine-readable error code (e.g. `"OOM"`, `"TIMEOUT"`).
        error_code: String,
        /// Human-readable error message.
        message: String,
    },
}

/// Host-side control session — owns identity, policy state, lifecycle state,
/// deadline tracking, stream assignment, and terminal outcome.
///
/// Owns **no** MLX arrays and **no** KV cache — those belong to the worker.
#[derive(Debug)]
pub struct GenerationControlSession {
    /// Opaque session identifier.
    pub session_id: String,
    /// Hash of the model image used for this generation.
    pub model_image_hash: Option<String>,
    /// PID of the worker process executing this session.
    pub worker_pid: Option<u32>,
    /// JSON-serialised admission receipt from the engine.
    pub admission_receipt_json: Option<String>,
    /// Terminal outcome, set when the session reaches a terminal state.
    pub terminal_outcome: Option<SessionOutcome>,
    /// Current token position in the sequence (0-indexed).
    pub position: u32,
    /// Token ID that signals end-of-sequence generation.
    pub eos_token_id: u32,
    /// Maximum number of tokens to generate (inclusive of any prompt
    /// prefix length already consumed before this session).
    pub max_tokens: u32,
    /// Current session state.
    state: ControlSessionState,
}

impl GenerationControlSession {
    /// Create a new generation control session.
    pub fn new(session_id: String, eos_token_id: u32, max_tokens: u32) -> Self {
        Self {
            session_id,
            model_image_hash: None,
            worker_pid: None,
            admission_receipt_json: None,
            terminal_outcome: None,
            position: 0,
            eos_token_id,
            max_tokens,
            state: ControlSessionState::Created,
        }
    }

    /// Return the current state.
    pub fn state(&self) -> ControlSessionState {
        self.state
    }

    /// Attempt a state transition. Returns `Ok(())` or `Err` on invalid
    /// transition (the state is unchanged on error).
    pub fn transition(&mut self, next: ControlSessionState) -> Result<(), String> {
        self.state.transition(next).map(|()| self.state = next)
    }

    /// Returns `true` if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_session_initial_state() {
        let session = GenerationControlSession::new("test-1".into(), 2, 100);
        assert_eq!(session.session_id, "test-1");
        assert_eq!(session.eos_token_id, 2);
        assert_eq!(session.max_tokens, 100);
        assert_eq!(session.position, 0);
        assert!(session.model_image_hash.is_none());
        assert!(session.worker_pid.is_none());
        assert!(session.admission_receipt_json.is_none());
        assert!(session.terminal_outcome.is_none());
        assert_eq!(session.state(), ControlSessionState::Created);
        assert!(!session.is_terminal());
    }

    #[test]
    fn control_session_happy_path() {
        let mut session = GenerationControlSession::new("s1".into(), 2, 100);
        session
            .transition(ControlSessionState::PrefillReady)
            .expect("legacy path Created -> PrefillReady is valid");
        session
            .transition(ControlSessionState::PrefillRunning)
            .expect("legacy path PrefillReady -> PrefillRunning is valid");
        session
            .transition(ControlSessionState::Decoding)
            .expect("PrefillRunning -> Decoding is valid");
        session
            .transition(ControlSessionState::Completed)
            .expect("Decoding -> Completed is valid");
        assert_eq!(session.state(), ControlSessionState::Completed);
        assert!(session.is_terminal());
    }

    #[test]
    fn control_session_invalid_transition_preserves_state() {
        let mut session = GenerationControlSession::new("s6".into(), 2, 100);
        assert_eq!(session.state(), ControlSessionState::Created);
        assert!(session.transition(ControlSessionState::Decoding).is_err());
        assert_eq!(session.state(), ControlSessionState::Created);
    }

    #[test]
    fn control_session_identity_transition_is_noop() {
        let mut session = GenerationControlSession::new("s7".into(), 2, 100);
        session
            .transition(ControlSessionState::PrefillReady)
            .expect("Created -> PrefillReady is valid");
        assert!(session
            .transition(ControlSessionState::PrefillReady)
            .is_ok());
        assert_eq!(session.state(), ControlSessionState::PrefillReady);
    }

    #[test]
    fn session_outcome_completed() {
        let outcome = SessionOutcome::Completed { token_count: 42 };
        assert_eq!(outcome, SessionOutcome::Completed { token_count: 42 });
    }

    #[test]
    fn session_outcome_cancelled() {
        let outcome = SessionOutcome::Cancelled {
            reason: "user request".into(),
        };
        assert_eq!(
            outcome,
            SessionOutcome::Cancelled {
                reason: "user request".into()
            }
        );
    }

    #[test]
    fn session_outcome_failed() {
        let outcome = SessionOutcome::Failed {
            error_code: "OOM".into(),
            message: "out of memory".into(),
        };
        assert_eq!(
            outcome,
            SessionOutcome::Failed {
                error_code: "OOM".into(),
                message: "out of memory".into(),
            }
        );
    }
}
