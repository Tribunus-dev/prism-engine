//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — First-class cancellation support.
//!
//! Provides a cooperative cancellation framework for heterogeneous execution
//! backends (Metal GPU, Core ML ANE, Accelerate CPU).  Backends that do not
//! support true preemption can still participate by checking the shared
//! [`CancellationToken`] at safe polling points (kernel boundaries, iteration
//! boundaries, etc.).
//!
//! # State machine
//!
//! Each work item transitions through the following cancellation states:
//!
//! ```text
//!                    request_cancel()
//!   Active ──────────────────────────► Requested
//!                                       │
//!                           ┌───────────┴───────────┐
//!                           │                       │
//!                      acknowledge()         completion arrives
//!                           │                       │
//!                           ▼                       ▼
//!                    Acknowledged        CompletedAfterCancellation
//!                           │                       │
//!                           └────────┬──────────────┘
//!                                    │
//!                               remove()
//!                                    │
//!                                    ▼
//!                               Released
//! ```
//!
//! - `Active` → forward progress continues normally.
//! - `Requested` → cancellation requested but not yet acknowledged by the
//!   backend.
//! - `Acknowledged` → backend noticed the cancellation and is winding down.
//! - `CompletedAfterCancellation` → execution finished after cancellation was
//!   requested (the scheduler must still free slots safely rather than jumping
//!   to `Released` in one step).
//! - `Released` → resources have been freed; the entry may be garbage-collected.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::scheduling::lane_work::WorkId;
use parking_lot::Mutex;

// ── Cancellation state ──────────────────────────────────────────────────────

/// Observability state of cancellation for one work item.
///
/// The scheduler uses this to decide whether a completion can release slots
/// immediately (`Acknowledged`) or must go through an additional accounting
/// step (`CompletedAfterCancellation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CancellationState {
    /// Forward progress continues normally.
    Active,
    /// Cancellation was requested but not yet acknowledged by the backend.
    Requested,
    /// Backend acknowledges cancellation (may still be executing, but a
    /// completion is expected).
    Acknowledged,
    /// Backend may continue executing after cancellation was requested
    /// (informational; completion will still be processed).
    BackendMayContinue,
    /// Execution completed after cancellation was requested — the scheduler
    /// must still free resources safely rather than skipping the slot release.
    CompletedAfterCancellation,
    /// Resources have been released; the entry is ready for garbage collection.
    Released,
}

// ── Cancellation token ─────────────────────────────────────────────────────

/// Token tied to one work item, checked by backends and the scheduler
/// without holding the [`CancellationManager`] lock.
///
/// The token wraps an `Arc<AtomicBool>` so that lock-free reads are possible
/// from backend completion handlers, Metal MTLCommandBuffer completion
/// callbacks, and hot scheduler paths.
#[derive(Debug)]
pub struct CancellationToken {
    /// Shared atomic flag: `true` means cancellation has been requested.
    inner: Arc<AtomicBool>,
    /// The work item this token governs.
    pub work_id: WorkId,
    /// Session identifier for bulk-cancellation operations.
    pub session_id: String,
    /// Optional human-readable reason for cancellation.
    pub reason: Mutex<Option<String>>,
}

impl Clone for CancellationToken {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            work_id: self.work_id,
            session_id: self.session_id.clone(),
            reason: Mutex::new(self.reason.lock().clone()),
        }
    }
}

impl CancellationToken {
    /// Create a new cancellation token for the given work and session.
    ///
    /// The initial state is non-cancelled (`false`).
    pub fn new(work_id: WorkId, session: &str) -> Self {
        Self {
            inner: Arc::new(AtomicBool::new(false)),
            work_id,
            session_id: session.to_string(),
            reason: Mutex::new(None),
        }
    }

    /// Check whether cancellation has been requested.
    ///
    /// This is a lock-free read suitable for hot scheduler paths and backend
    /// polling loops.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }

    /// Request cancellation of this work item.
    ///
    /// Sets the atomic flag to `true` and records the reason.  Idempotent:
    /// subsequent calls are no-ops.
    pub fn cancel(&self, reason: &str) {
        self.inner.store(true, Ordering::Release);
        // Only record the first reason.
        let mut guard = self.reason.lock();
        if guard.is_none() {
            *guard = Some(reason.to_string());
        }
    }
}

// ── Cancellation manager ────────────────────────────────────────────────────

/// Manages cancellation for all in-flight work across heterogeneous backends.
///
/// The manager holds:
/// - Tokens for lock-free cancellation signalling.
/// - Per-work-item state machine transitions.
/// - A session index for bulk-cancellation queries.
///
/// All mutation methods take `&mut self`; the caller is responsible for
/// providing mutual exclusion (typically via `parking_lot::Mutex` or
/// single-threaded ownership).
pub struct CancellationManager {
    tokens: HashMap<WorkId, CancellationToken>,
    states: HashMap<WorkId, CancellationState>,
    by_session: HashMap<String, Vec<WorkId>>,
}

impl CancellationManager {
    /// Create a new empty cancellation manager.
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            states: HashMap::new(),
            by_session: HashMap::new(),
        }
    }

    /// Register a cancellation token for new work.
    ///
    /// The initial state is [`CancellationState::Active`].
    ///
    /// # Panics
    ///
    /// Panics if the work ID is already registered (double-registration is
    /// a programming error).
    pub fn register(&mut self, token: CancellationToken) {
        let work_id = token.work_id;
        assert!(
            !self.tokens.contains_key(&work_id),
            "CancellationManager: work_id {:?} already registered",
            work_id,
        );

        self.by_session
            .entry(token.session_id.clone())
            .or_default()
            .push(work_id);
        self.states.insert(work_id, CancellationState::Active);
        self.tokens.insert(work_id, token);
    }

    /// Request cancellation of a single work item.
    ///
    /// Sets the cancellation flag on the token (visible lock-free to backends)
    /// and transitions the state to [`CancellationState::Requested`].
    ///
    /// Returns the *previous* state on success, or an `Err` if the work ID is
    /// unknown.
    pub fn request_cancel(
        &mut self,
        work_id: WorkId,
        reason: &str,
    ) -> Result<CancellationState, String> {
        let token = self
            .tokens
            .get(&work_id)
            .ok_or_else(|| format!("CancellationManager: unknown work_id {:?}", work_id))?;

        token.cancel(reason);
        let prev = self
            .states
            .insert(work_id, CancellationState::Requested)
            .unwrap_or(CancellationState::Active);
        Ok(prev)
    }

    /// Cancel all work items belonging to a session.
    ///
    /// Returns the list of work IDs that were actually cancelled (may be
    /// empty).  Already-cancelled items are re-requested (idempotent at the
    /// token level).
    pub fn cancel_session(&mut self, session_id: &str, reason: &str) -> Vec<WorkId> {
        let mut cancelled = Vec::new();
        if let Some(ids) = self.by_session.get(session_id).cloned() {
            for work_id in &ids {
                if let Some(token) = self.tokens.get(work_id) {
                    token.cancel(reason);
                }
                self.states.insert(*work_id, CancellationState::Requested);
                cancelled.push(*work_id);
            }
        }
        cancelled
    }

    /// Acknowledge cancellation from the backend.
    ///
    /// The backend calls this when it notices the cancellation flag and
    /// agrees to stop or is winding down.  Transitions from
    /// [`CancellationState::Requested`] to
    /// [`CancellationState::Acknowledged`].
    ///
    /// Returns an error if the work ID is unknown or if the current state
    /// is not `Requested`.
    pub fn acknowledge(&mut self, work_id: WorkId) -> Result<(), String> {
        let state = self
            .states
            .get(&work_id)
            .ok_or_else(|| format!("CancellationManager: unknown work_id {:?}", work_id))?;

        match *state {
            CancellationState::Requested => {
                self.states.insert(work_id, CancellationState::Acknowledged);
                Ok(())
            }
            CancellationState::Active => Err(format!(
                "CancellationManager: cannot acknowledge {:?} — still Active",
                work_id
            )),
            CancellationState::Acknowledged
            | CancellationState::BackendMayContinue
            | CancellationState::CompletedAfterCancellation
            | CancellationState::Released => Err(format!(
                "CancellationManager: cannot acknowledge {:?} — already in state {:?}",
                work_id, state
            )),
        }
    }

    /// Record that a backend completed execution of a work item that was
    /// previously requested for cancellation.
    ///
    /// This is called by the scheduler's completion handler when the backend
    /// finishes normally but cancellation was requested before completion
    /// arrived.  The state moves to
    /// [`CancellationState::CompletedAfterCancellation`] so the scheduler
    /// can safely free the slot.
    ///
    /// Returns an error if the work ID is unknown.
    pub fn record_completion_after_cancellation(&mut self, work_id: WorkId) -> Result<(), String> {
        let state = self
            .states
            .get(&work_id)
            .ok_or_else(|| format!("CancellationManager: unknown work_id {:?}", work_id))?;

        match *state {
            CancellationState::Requested | CancellationState::Acknowledged => {
                self.states
                    .insert(work_id, CancellationState::CompletedAfterCancellation);
                Ok(())
            }
            _ => Err(format!(
                "CancellationManager: cannot record completion for {:?} — state is {:?}",
                work_id, state
            )),
        }
    }

    /// Transition a work item to [`CancellationState::Released`].
    ///
    /// This is the terminal state, indicating the scheduler has freed all
    /// resources associated with the work.  The entry is **not** removed
    /// from the manager — call [`remove`](Self::remove) after releasing
    /// resources to garbage-collect the state entirely.
    ///
    /// Returns an error if the work ID is unknown.
    pub fn mark_released(&mut self, work_id: WorkId) -> Result<(), String> {
        if !self.states.contains_key(&work_id) {
            return Err(format!(
                "CancellationManager: unknown work_id {:?}",
                work_id
            ));
        }
        self.states.insert(work_id, CancellationState::Released);
        Ok(())
    }

    /// Get the current cancellation state of a work item.
    pub fn state(&self, work_id: WorkId) -> Option<CancellationState> {
        self.states.get(&work_id).copied()
    }

    /// Remove all state for a completed work item.
    ///
    /// This garbage-collects the token, state, and session index entry for the
    /// given work ID.  The caller should ensure resources have been released
    /// first (see [`mark_released`](Self::mark_released)).
    pub fn remove(&mut self, work_id: WorkId) {
        if let Some(token) = self.tokens.remove(&work_id) {
            // Clean up the session index.
            if let Some(ids) = self.by_session.get_mut(&token.session_id) {
                ids.retain(|id| *id != work_id);
                if ids.is_empty() {
                    self.by_session.remove(&token.session_id);
                }
            }
        }
        self.states.remove(&work_id);
    }

    /// Get a reference to the cancellation token for a work item, if
    /// registered.
    pub fn token(&self, work_id: WorkId) -> Option<CancellationToken> {
        self.tokens.get(&work_id).cloned()
    }

    /// Check whether a session has any work items in a cancelled state
    /// (`Requested`, `Acknowledged`, `BackendMayContinue`, or
    /// `CompletedAfterCancellation`).
    pub fn has_cancelled(&self, session_id: &str) -> bool {
        self.by_session
            .get(session_id)
            .map(|ids| {
                ids.iter().any(|id| {
                    self.states.get(id).is_some_and(|s| {
                        matches!(
                            s,
                            CancellationState::Requested
                                | CancellationState::Acknowledged
                                | CancellationState::BackendMayContinue
                                | CancellationState::CompletedAfterCancellation
                        )
                    })
                })
            })
            .unwrap_or(false)
    }

    /// Return the number of registered work items.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if no work items are registered.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Iterate over all registered work IDs.
    pub fn work_ids(&self) -> impl Iterator<Item = &WorkId> {
        self.tokens.keys()
    }

    /// Iterate over all registered `(WorkId, CancellationState)` pairs.
    pub fn states(&self) -> impl Iterator<Item = (&WorkId, &CancellationState)> {
        self.states.iter()
    }
}

impl Default for CancellationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(n: u64) -> WorkId {
        WorkId(n)
    }

    // ── CancellationToken ────────────────────────────────────────────────

    #[test]
    fn test_token_starts_not_cancelled() {
        let token = CancellationToken::new(make_id(1), "test-session");
        assert!(!token.is_cancelled());
        assert!(token.reason.lock().is_none());
    }

    #[test]
    fn test_token_cancel_sets_flag() {
        let token = CancellationToken::new(make_id(1), "test-session");
        token.cancel("timeout");
        assert!(token.is_cancelled());
        assert_eq!(token.reason.lock().as_deref(), Some("timeout"));
    }

    #[test]
    fn test_token_cancel_is_idempotent() {
        let token = CancellationToken::new(make_id(1), "test-session");
        token.cancel("reason-a");
        assert!(token.is_cancelled());
        // Subsequent cancel keeps first reason.
        token.cancel("reason-b");
        assert_eq!(token.reason.lock().as_deref(), Some("reason-a"));
    }

    #[test]
    fn test_token_lock_free_shared_across_clones() {
        let token_a = CancellationToken::new(make_id(1), "session");
        let token_b = token_a.clone();
        assert!(!token_b.is_cancelled());
        token_a.cancel("shared");
        // Clone shares the same inner AtomicBool.
        assert!(token_b.is_cancelled());
    }

    // ── CancellationManager registration ─────────────────────────────────

    #[test]
    fn test_register_sets_active_state() {
        let mut mgr = CancellationManager::new();
        let token = CancellationToken::new(make_id(1), "sess");
        mgr.register(token);
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Active));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_register_duplicate_panics() {
        let mut mgr = CancellationManager::new();
        let t1 = CancellationToken::new(make_id(42), "sess");
        let t2 = CancellationToken::new(make_id(42), "sess");
        mgr.register(t1);
        mgr.register(t2);
    }

    // ── request_cancel ───────────────────────────────────────────────────

    #[test]
    fn test_request_cancel_transitions_to_requested() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));

        let prev = mgr.request_cancel(make_id(1), "user-abort").unwrap();
        assert_eq!(prev, CancellationState::Active);
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Requested));
    }

    #[test]
    fn test_request_cancel_sets_token_flag() {
        let mut mgr = CancellationManager::new();
        let token = CancellationToken::new(make_id(1), "sess");
        mgr.register(token.clone());

        mgr.request_cancel(make_id(1), "timeout").unwrap();
        assert!(token.is_cancelled());
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Requested));
    }

    #[test]
    fn test_request_cancel_unknown_id() {
        let mut mgr = CancellationManager::new();
        let result = mgr.request_cancel(make_id(999), "who?");
        assert!(result.is_err());
    }

    // ── cancel_session ───────────────────────────────────────────────────

    #[test]
    fn test_cancel_session_cancels_all_in_session() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "alpha"));
        mgr.register(CancellationToken::new(make_id(2), "alpha"));
        mgr.register(CancellationToken::new(make_id(3), "beta"));

        let cancelled = mgr.cancel_session("alpha", "session-end");
        assert_eq!(cancelled.len(), 2);
        assert!(cancelled.contains(&make_id(1)));
        assert!(cancelled.contains(&make_id(2)));

        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Requested));
        assert_eq!(mgr.state(make_id(2)), Some(CancellationState::Requested));
        // Session "beta" untouched.
        assert_eq!(mgr.state(make_id(3)), Some(CancellationState::Active));
    }

    #[test]
    fn test_cancel_session_returns_empty_for_unknown_session() {
        let mut mgr = CancellationManager::new();
        let cancelled = mgr.cancel_session("ghost", "gone");
        assert!(cancelled.is_empty());
    }

    // ── acknowledge ──────────────────────────────────────────────────────

    #[test]
    fn test_acknowledge_transitions_from_requested() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.request_cancel(make_id(1), "stop").unwrap();

        mgr.acknowledge(make_id(1)).unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Acknowledged));
    }

    #[test]
    fn test_acknowledge_fails_from_active() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));

        let result = mgr.acknowledge(make_id(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_acknowledge_unknown_id() {
        let mut mgr = CancellationManager::new();
        assert!(mgr.acknowledge(make_id(999)).is_err());
    }

    // ── record_completion_after_cancellation ─────────────────────────────

    #[test]
    fn test_completion_after_cancellation_from_requested() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.request_cancel(make_id(1), "stop").unwrap();

        mgr.record_completion_after_cancellation(make_id(1))
            .unwrap();
        assert_eq!(
            mgr.state(make_id(1)),
            Some(CancellationState::CompletedAfterCancellation)
        );
    }

    #[test]
    fn test_completion_after_cancellation_from_acknowledged() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.request_cancel(make_id(1), "stop").unwrap();
        mgr.acknowledge(make_id(1)).unwrap();

        mgr.record_completion_after_cancellation(make_id(1))
            .unwrap();
        assert_eq!(
            mgr.state(make_id(1)),
            Some(CancellationState::CompletedAfterCancellation)
        );
    }

    #[test]
    fn test_completion_after_cancellation_fails_from_active() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));

        let result = mgr.record_completion_after_cancellation(make_id(1));
        assert!(result.is_err());
    }

    // ── mark_released / remove ───────────────────────────────────────────

    #[test]
    fn test_mark_released_transitions_state() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));

        mgr.mark_released(make_id(1)).unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Released));
    }

    #[test]
    fn test_remove_clears_all_state() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));

        mgr.remove(make_id(1));
        assert_eq!(mgr.state(make_id(1)), None);
        assert!(mgr.token(make_id(1)).is_none());
        assert!(!mgr.has_cancelled("sess"));
    }

    #[test]
    fn test_remove_unknown_id_is_noop() {
        let mut mgr = CancellationManager::new();
        mgr.remove(make_id(999)); // must not panic
    }

    // ── has_cancelled ────────────────────────────────────────────────────

    #[test]
    fn test_has_cancelled_false_when_all_active() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        assert!(!mgr.has_cancelled("sess"));
    }

    #[test]
    fn test_has_cancelled_true_after_request() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.request_cancel(make_id(1), "stop").unwrap();
        assert!(mgr.has_cancelled("sess"));
    }

    #[test]
    fn test_has_cancelled_true_for_acknowledged() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.request_cancel(make_id(1), "stop").unwrap();
        mgr.acknowledge(make_id(1)).unwrap();
        assert!(mgr.has_cancelled("sess"));
    }

    #[test]
    fn test_has_cancelled_false_for_unknown_session() {
        let mgr = CancellationManager::new();
        assert!(!mgr.has_cancelled("ghost"));
    }

    // ── len / is_empty / work_ids / states ───────────────────────────────

    #[test]
    fn test_len_and_is_empty() {
        let mut mgr = CancellationManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);

        mgr.register(CancellationToken::new(make_id(1), "sess"));
        assert!(!mgr.is_empty());
        assert_eq!(mgr.len(), 1);

        mgr.remove(make_id(1));
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_work_ids_iteration() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(10), "sess"));
        mgr.register(CancellationToken::new(make_id(20), "sess"));

        let ids: Vec<WorkId> = mgr.work_ids().copied().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&make_id(10)));
        assert!(ids.contains(&make_id(20)));
    }

    #[test]
    fn test_states_iteration() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "sess"));
        mgr.register(CancellationToken::new(make_id(2), "sess"));
        mgr.request_cancel(make_id(2), "stop").unwrap();

        let states: Vec<(WorkId, CancellationState)> =
            mgr.states().map(|(&id, &s)| (id, s)).collect();
        assert_eq!(states.len(), 2);
        assert!(states.contains(&(make_id(1), CancellationState::Active)));
        assert!(states.contains(&(make_id(2), CancellationState::Requested)));
    }

    // ── Full lifecycle ───────────────────────────────────────────────────

    #[test]
    fn test_full_lifecycle_cancelled_before_completion() {
        let mut mgr = CancellationManager::new();
        let token = CancellationToken::new(make_id(1), "session-A");
        mgr.register(token);

        // 1. Request cancellation.
        mgr.request_cancel(make_id(1), "timeout").unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Requested));

        // 2. Backend notices and acknowledges.
        mgr.acknowledge(make_id(1)).unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Acknowledged));

        // 3. Backend completes.
        mgr.record_completion_after_cancellation(make_id(1))
            .unwrap();
        assert_eq!(
            mgr.state(make_id(1)),
            Some(CancellationState::CompletedAfterCancellation)
        );

        // 4. Scheduler frees resources.
        mgr.mark_released(make_id(1)).unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Released));

        // 5. Garbage collection.
        mgr.remove(make_id(1));
        assert_eq!(mgr.state(make_id(1)), None);
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_full_lifecycle_completion_without_acknowledge() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "session-B"));

        // Completion arrives before the backend had a chance to acknowledge.
        mgr.request_cancel(make_id(1), "stop").unwrap();
        mgr.record_completion_after_cancellation(make_id(1))
            .unwrap();
        assert_eq!(
            mgr.state(make_id(1)),
            Some(CancellationState::CompletedAfterCancellation)
        );

        mgr.mark_released(make_id(1)).unwrap();
        mgr.remove(make_id(1));
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_full_lifecycle_normal_completion() {
        let mut mgr = CancellationManager::new();
        mgr.register(CancellationToken::new(make_id(1), "session-C"));

        // No cancellation — normal path.
        mgr.mark_released(make_id(1)).unwrap();
        assert_eq!(mgr.state(make_id(1)), Some(CancellationState::Released));

        mgr.remove(make_id(1));
        assert!(mgr.is_empty());
    }
}
