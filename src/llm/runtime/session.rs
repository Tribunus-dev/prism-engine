// ── Prism LLM — Session Manager ────────────────────────────────────────
//
// Manages the lifecycle of LLM inference sessions: creation, state
// transitions, dispatch tracking, and teardown.

use crate::llm::manifest::SessionId;
use crate::llm::server::{
    CreateSessionRequest, DispatchId, InferenceAdmissionReceipt, InferenceSessionState,
    MemoryPressureLevel,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

// ── Constants for atomic memory-pressure encoding ────────────────────

const MP_NORMAL: u8 = 0;
const MP_ELEVATED: u8 = 1;
const MP_CRITICAL: u8 = 2;

// ── SessionHandle ────────────────────────────────────────────────────

/// Runtime handle wrapping all state for a single inference session.
#[derive(Debug, Clone)]
pub struct SessionHandle {
    /// The unique identifier for this session.
    pub session_id: SessionId,
    /// Timestamp at which the session was created.
    pub created_at: SystemTime,
    /// Current lifecycle state.
    pub state: InferenceSessionState,
    /// Ordered history of state transitions with timestamps.
    pub state_history: Vec<(SystemTime, InferenceSessionState)>,
    /// Dispatches still in-flight for this session.
    pub pending_dispatches: Vec<DispatchId>,
    /// Admission receipt, populated during `create_session`.
    pub receipt: Option<InferenceAdmissionReceipt>,
}

// ── SessionManager ───────────────────────────────────────────────────

/// Manages all active inference sessions and their lifecycle transitions.
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<SessionId, SessionHandle>>>,
    memory_pressure: AtomicU8,
}

impl SessionManager {
    /// Create a new empty `SessionManager` with normal memory pressure.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            memory_pressure: AtomicU8::new(MP_NORMAL),
        }
    }

    /// Update the current memory pressure level observed by this manager.
    pub fn set_memory_pressure(&self, level: MemoryPressureLevel) {
        let v = match level {
            MemoryPressureLevel::Normal => MP_NORMAL,
            MemoryPressureLevel::Elevated => MP_ELEVATED,
            MemoryPressureLevel::Critical => MP_CRITICAL,
        };
        self.memory_pressure.store(v, Ordering::Relaxed);
    }

    // ── Session lifecycle ──────────────────────────────────────────

    /// Validate a session creation request, admit the session, build the
    /// admission receipt, and transition it to `Ready`.
    ///
    /// Returns the newly allocated `SessionId` on success.
    pub fn create_session(&self, request: CreateSessionRequest) -> Result<SessionId, String> {
        // --- Validation ---

        // Reject empty context profiles as unsupported.
        if request.context_profile.0.is_empty() {
            return Err("context_profile is empty or unsupported".into());
        }

        // Reject empty CImage identifiers.
        if request.cimage_id.0.is_empty() {
            return Err("cimage_id is empty".into());
        }

        // Reject session creation when unified memory is critically
        // pressured.
        if self.memory_pressure.load(Ordering::Relaxed) == MP_CRITICAL {
            return Err("cannot admit session: memory pressure is critical".into());
        }

        // --- Admission ---

        let session_id = SessionId(uuid::Uuid::new_v4());
        let now = SystemTime::now();

        // Initial handle in the Admitting state with history tracing
        // back to Created.
        let mut handle = SessionHandle {
            session_id,
            created_at: now,
            state: InferenceSessionState::Admitting,
            state_history: vec![
                (now, InferenceSessionState::Created),
                (now, InferenceSessionState::Admitting),
            ],
            pending_dispatches: Vec::new(),
            receipt: None,
        };

        // Build the admission receipt (validation passed => admitted).
        let receipt = InferenceAdmissionReceipt {
            cimage_id: request.cimage_id,
            context_profile: request.context_profile,
            execution_policy: request.execution_policy,
            admitted: true,
            refusal_reason: None,
        };
        handle.receipt = Some(receipt);

        // Transition to Ready.
        handle.state = InferenceSessionState::Ready;
        handle
            .state_history
            .push((SystemTime::now(), InferenceSessionState::Ready));

        let mut sessions = self
            .sessions
            .lock()
            .map_err(|e| format!("session lock error: {}", e))?;
        sessions.insert(session_id, handle);
        Ok(session_id)
    }

    /// Return the current lifecycle state of a session, if it exists.
    pub fn get_state(&self, id: &SessionId) -> Option<InferenceSessionState> {
        let sessions = self.sessions.lock().ok()?;
        sessions.get(id).map(|h| h.state)
    }

    /// Transition a session to the next state, recording the change in its
    /// state history.
    ///
    /// Returns an error if the session does not exist.
    pub fn transition(&self, id: &SessionId, next: InferenceSessionState) -> Result<(), String> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|e| format!("session lock error: {}", e))?;
        let handle = sessions
            .get_mut(id)
            .ok_or_else(|| format!("session {:?} not found", id))?;
        let now = SystemTime::now();
        handle.state = next;
        handle.state_history.push((now, next));
        Ok(())
    }

    /// Record a dispatch as pending on the specified session.
    pub fn add_dispatch(&self, id: &SessionId, dispatch_id: DispatchId) {
        if let Ok(mut sessions) = self.sessions.lock() {
            if let Some(handle) = sessions.get_mut(id) {
                handle.pending_dispatches.push(dispatch_id);
            }
        }
    }

    /// Close a session and return its admission receipt.
    ///
    /// The session must exist and must have been admitted (i.e. the receipt
    /// must be present).  The session is marked `Closed` in the map but is
    /// not removed, preserving its state history for diagnostics.
    pub fn close_session(&self, id: &SessionId) -> Result<InferenceAdmissionReceipt, String> {
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|e| format!("session lock error: {}", e))?;
        let handle = sessions
            .get_mut(id)
            .ok_or_else(|| format!("session {:?} not found", id))?;

        handle.state = InferenceSessionState::Closed;
        handle
            .state_history
            .push((SystemTime::now(), InferenceSessionState::Closed));

        handle
            .receipt
            .clone()
            .ok_or_else(|| format!("session {:?} has no admission receipt", id))
    }
}

#[cfg(test)]
impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}
// ── Compute-core integration ────────────────────────────────────
///
/// When the `prism-backend` feature is enabled, delegates session
/// lifecycle to tribunal compute-core's `GenerationControlSession`.
///
#[cfg(feature = "prism-backend")]
mod compute_session {
    use super::*;
    use crate::llm::server::InferenceSessionState;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use tribunus_compute_core::session::{ControlSessionState, GenerationControlSession};

    /// Map a compute-core `ControlSessionState` to the engine's
    /// `InferenceSessionState`.
    #[allow(dead_code)]
    fn control_state_to_inference_state(state: ControlSessionState) -> InferenceSessionState {
        match state {
            ControlSessionState::Created => InferenceSessionState::Created,
            ControlSessionState::Admitted => InferenceSessionState::Admitting,
            ControlSessionState::Submitted => InferenceSessionState::LoadingCImage,
            ControlSessionState::PrefillReady => InferenceSessionState::Ready,
            ControlSessionState::PrefillRunning => InferenceSessionState::Prefilling,
            ControlSessionState::Decoding => InferenceSessionState::Decoding,
            ControlSessionState::Completed => InferenceSessionState::Completed,
            ControlSessionState::Cancelled => InferenceSessionState::Cancelled,
            ControlSessionState::Failed => InferenceSessionState::Failed,
        }
    }

    /// Reverse mapping: engine state → compute-core state.
    #[allow(dead_code)]
    fn inference_state_to_control_state(state: InferenceSessionState) -> ControlSessionState {
        match state {
            InferenceSessionState::Created => ControlSessionState::Created,
            InferenceSessionState::Admitting => ControlSessionState::Admitted,
            InferenceSessionState::LoadingCImage => ControlSessionState::Submitted,
            InferenceSessionState::EstablishingResidency
            | InferenceSessionState::Resident
            | InferenceSessionState::Prefilling => ControlSessionState::PrefillRunning,
            InferenceSessionState::PublishingKvEpoch
            | InferenceSessionState::Ready => ControlSessionState::PrefillReady,
            InferenceSessionState::Decoding => ControlSessionState::Decoding,
            InferenceSessionState::CompressingKv
            | InferenceSessionState::RefreshingContext => ControlSessionState::Decoding,
            InferenceSessionState::Cancelling | InferenceSessionState::Cancelled => {
                ControlSessionState::Cancelled
            }
            InferenceSessionState::Recovering
            | InferenceSessionState::Completed
            | InferenceSessionState::Closed => ControlSessionState::Completed,
            InferenceSessionState::Failed => ControlSessionState::Failed,
        }
    }

    /// Compute-backed session manager that wraps
    /// `GenerationControlSession` instances.
    #[allow(dead_code)]
    pub struct ComputeSessionManager {
        sessions: Mutex<HashMap<SessionId, GenerationControlSession>>,
    }

    impl ComputeSessionManager {
        /// Create a new, empty compute session manager.
        #[allow(dead_code)]
        pub fn new() -> Self {
            Self {
                sessions: Mutex::new(HashMap::new()),
            }
        }

        /// Create a new generation control session and store it.
        #[allow(dead_code)]
        pub fn create_session(
            &self,
            session_id: SessionId,
            eos_token_id: u32,
            max_tokens: u32,
        ) -> Result<(), String> {
            let session = GenerationControlSession::new(
                session_id.0.to_string(),
                eos_token_id,
                max_tokens,
            );
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|e| format!("session lock error: {}", e))?;
            sessions.insert(session_id, session);
            Ok(())
        }

        /// Transition an existing session to a new engine-level state.
        /// The state is converted to the compute-core equivalent before
        /// delegation.
        #[allow(dead_code)]
        pub fn transition(
            &self,
            id: &SessionId,
            next: InferenceSessionState,
        ) -> Result<(), String> {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|e| format!("session lock error: {}", e))?;
            let session = sessions
                .get_mut(id)
                .ok_or_else(|| format!("session {:?} not found", id))?;
            let control_next = inference_state_to_control_state(next);
            session.transition(control_next)
        }

        /// Return the current engine-level state of a session.
        #[allow(dead_code)]
        pub fn get_state(&self, id: &SessionId) -> Option<InferenceSessionState> {
            let sessions = self.sessions.lock().ok()?;
            sessions
                .get(id)
                .map(|s| control_state_to_inference_state(s.state()))
        }

        /// Close and remove a session, consuming its compute-core state.
        #[allow(dead_code)]
        pub fn close_session(&self, id: &SessionId) -> Result<(), String> {
            let mut sessions = self
                .sessions
                .lock()
                .map_err(|e| format!("session lock error: {}", e))?;
            sessions
                .remove(id)
                .map(|_| ())
                .ok_or_else(|| format!("session {:?} not found", id))
        }
    }
}
