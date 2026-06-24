// ── Prism LLM Inference — Cancellation Manager ────────────────────────────
//
// Cooperative cancellation for inference sessions. The CancellationManager
// holds a set of cancelled session IDs and produces CancellationHandle values
// for each registered session. Consumers check is_cancelled before proceeding
// with work, yielding control when a cancellation is detected.

use std::collections::HashSet;
use std::sync::Mutex;

use super::super::manifest::SessionId;
use super::super::server::{
    CancellationHandle, InferenceCancelledReceipt, InferenceSessionState, RequestId,
};

/// Manages cooperative cancellation for inference sessions.
///
/// Stores cancelled session IDs in a thread-safe set. Handles are lightweight
/// identifiers that carry the owning session's ID and a unique request ID.
pub struct CancellationManager {
    cancelled: Mutex<HashSet<SessionId>>,
}

impl CancellationManager {
    /// Creates a new, empty CancellationManager.
    pub fn new() -> Self {
        Self {
            cancelled: Mutex::new(HashSet::new()),
        }
    }

    /// Registers a new cancellation handle for the given session.
    ///
    /// The returned handle can be passed to [`Self::cancel`] to request
    /// cancellation of the associated session.
    pub fn register_handle(&self, session_id: SessionId) -> CancellationHandle {
        CancellationHandle {
            session_id,
            request_id: RequestId(uuid::Uuid::new_v4()),
        }
    }

    /// Requests cancellation for the session identified by `handle`.
    ///
    /// Adds the session to the cancelled set and returns an
    /// `InferenceCancelledReceipt` with `state_at_cancellation` set to
    /// `Cancelling`.
    pub fn cancel(
        &self,
        handle: &CancellationHandle,
    ) -> Result<InferenceCancelledReceipt, String> {
        let mut set = self
            .cancelled
            .lock()
            .map_err(|e| format!("cancellation lock poisoned: {e}"))?;
        set.insert(handle.session_id);
        Ok(InferenceCancelledReceipt {
            session_id: handle.session_id,
            request_id: handle.request_id.clone(),
            state_at_cancellation: InferenceSessionState::Cancelling,
            active_epoch: None,
            completed_decode_tokens: 0,
            cleanup_completed: false,
        })
    }

    /// Returns `true` if the given session has been cancelled.
    pub fn is_cancelled(&self, session_id: &SessionId) -> bool {
        self.cancelled
            .lock()
            .map(|set| set.contains(session_id))
            .unwrap_or(false)
    }
}

impl Default for CancellationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        assert!(!mgr.is_cancelled(&sid));
    }

    #[test]
    fn test_register_handle_creates_valid_handle() {
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        let handle = mgr.register_handle(sid);
        assert_eq!(handle.session_id, sid);
    }

    #[test]
    fn test_cancel_marks_session_and_provides_receipt() {
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        let handle = mgr.register_handle(sid);

        let receipt = mgr.cancel(&handle).expect("cancel should succeed");

        assert!(mgr.is_cancelled(&sid));
        assert_eq!(receipt.session_id, sid);
        assert_eq!(receipt.request_id, handle.request_id);
        assert_eq!(receipt.state_at_cancellation, InferenceSessionState::Cancelling);
    }

    #[test]
    fn test_cancel_without_register_implicitly_inserts() {
        // cancel works even if no explicit handle was registered; the handle
        // only carries the identifiers used for the receipt.
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        let handle = CancellationHandle {
            session_id: sid,
            request_id: RequestId(uuid::Uuid::new_v4()),
        };

        let receipt = mgr.cancel(&handle).expect("cancel should succeed");
        assert!(mgr.is_cancelled(&sid));
        assert_eq!(receipt.session_id, sid);
    }

    #[test]
    fn test_is_cancelled_returns_false_for_uncancelled_session() {
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        assert!(!mgr.is_cancelled(&sid));

        let sid2 = SessionId(uuid::Uuid::new_v4());
        let handle = mgr.register_handle(sid2);
        mgr.cancel(&handle).unwrap();
        assert!(mgr.is_cancelled(&sid2));
        assert!(!mgr.is_cancelled(&sid)); // sid is still not cancelled
    }

    #[test]
    fn test_cancel_idempotent() {
        let mgr = CancellationManager::new();
        let sid = SessionId(uuid::Uuid::new_v4());
        let handle = mgr.register_handle(sid);

        mgr.cancel(&handle).expect("first cancel");
        // second cancel is still a no-op error-wise; it stays in the set
        assert!(mgr.is_cancelled(&sid));

        let receipt2 = mgr.cancel(&handle).expect("second cancel should also succeed");
        assert_eq!(receipt2.session_id, sid);
    }

    #[test]
    fn test_default_creates_empty() {
        let mgr = CancellationManager::default();
        let sid = SessionId(uuid::Uuid::new_v4());
        assert!(!mgr.is_cancelled(&sid));
    }
}
