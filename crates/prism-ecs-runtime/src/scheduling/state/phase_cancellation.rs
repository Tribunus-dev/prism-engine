//! Phase cancellation state (constitutional home).
//!
//! Per-step cancellation evidence and the cancellation-checker record.
//!
//! # Authority
//!
//! The cancellation flag and deadline live in the canonical scheduling
//! state; the runtime completion-reconciliation system is the only
//! producer of cancellation evidence. The checker itself is a
//! borrowed view; the underlying `AtomicBool` flag is the canonical
//! state, replicated to per-phase checkers through a reference.
//!
//! # Placeholder engine types
//!
//! `PhaseId` is a placeholder matching the engine's
//! `phase_graph::PhaseId`. `CancellationClass` is a placeholder enum
//! matching the engine. Both move in their own migrations.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::phase::PhaseId;

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::compute_image::phase_graph::CancellationClass`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancellationClass {
    /// Phase may be interrupted at any point.
    Preemptible,
    /// Phase must complete or reach a safe checkpoint.
    NonPreemptible,
    /// Phase has observable side effects; cancellation must roll back.
    SideEffecting,
}

// ---------------------------------------------------------------------------
// CancellationEvidence
// ---------------------------------------------------------------------------

/// Evidence recorded about cancellation state during phase execution.
#[derive(Debug, Clone)]
pub struct CancellationEvidence {
    pub cancelled_at_dispatch: bool,
    pub cancelled_at_completion: bool,
    pub cancelled_at_publication: bool,
    pub deadline_expired: bool,
}

impl CancellationEvidence {
    pub fn new() -> Self {
        Self {
            cancelled_at_dispatch: false,
            cancelled_at_completion: false,
            cancelled_at_publication: false,
            deadline_expired: false,
        }
    }

    /// Returns `true` if any cancellation or deadline-expiry signal
    /// was observed at any of the three checkpoint points.
    pub fn any_cancelled(&self) -> bool {
        self.cancelled_at_dispatch
            || self.cancelled_at_completion
            || self.cancelled_at_publication
            || self.deadline_expired
    }
}

impl Default for CancellationEvidence {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CancellationChecker
// ---------------------------------------------------------------------------

/// Cancellation checker — checks the shared cancellation flag and deadline.
pub struct CancellationChecker {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancellationChecker {
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self {
            flag,
            deadline: None,
        }
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    /// Check if the deadline has expired.
    pub fn is_expired(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }

    /// Combined check — returns true if execution should stop.
    pub fn should_stop(&self) -> bool {
        self.is_cancelled() || self.is_expired()
    }

    /// Check three-point cancellation evidence for a phase.
    pub fn check_phase(
        &self,
        _phase_id: &PhaseId,
        _cancellation_class: CancellationClass,
    ) -> CancellationEvidence {
        CancellationEvidence {
            cancelled_at_dispatch: self.is_cancelled(),
            cancelled_at_completion: self.is_cancelled(),
            cancelled_at_publication: self.is_cancelled(),
            deadline_expired: self.is_expired(),
        }
    }
}

// ---------------------------------------------------------------------------
// CancellationPolicy
// ---------------------------------------------------------------------------

/// Cancellation policy for different phase classes.
#[derive(Debug, Clone)]
pub struct CancellationPolicy {
    /// Whether preemptible phases can be interrupted mid-execution.
    pub allow_preemptible_interrupt: bool,
    /// Whether non-preemptible phases are waited on before discarding.
    pub wait_for_non_preemptible: bool,
    /// Maximum grace period for non-preemptible completion after cancellation.
    pub non_preemptible_grace_ms: u64,
}

impl Default for CancellationPolicy {
    fn default() -> Self {
        Self {
            allow_preemptible_interrupt: true,
            wait_for_non_preemptible: true,
            non_preemptible_grace_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `phase_cancellation` state.

    use super::*;
    use std::time::Duration;

    #[test]
    fn fresh_evidence_is_clean() {
        // Architectural invariant: a fresh CancellationEvidence has
        // no cancellation signals. any_cancelled() returns false.
        let e = CancellationEvidence::new();
        assert!(!e.any_cancelled());
    }

    #[test]
    fn evidence_signals_partition_disjoint() {
        // Architectural invariant: the four signals are independent
        // and any_cancelled() is the OR of all four.
        let mut e = CancellationEvidence::new();
        assert!(!e.any_cancelled());
        e.cancelled_at_dispatch = true;
        assert!(e.any_cancelled());
        e.cancelled_at_dispatch = false;
        e.cancelled_at_completion = true;
        assert!(e.any_cancelled());
        e.cancelled_at_completion = false;
        e.cancelled_at_publication = true;
        assert!(e.any_cancelled());
        e.cancelled_at_publication = false;
        e.deadline_expired = true;
        assert!(e.any_cancelled());
    }

    #[test]
    fn checker_reads_shared_flag() {
        // Architectural invariant: two checkers sharing the same
        // Arc<AtomicBool> see the same flag state.
        let flag = Arc::new(AtomicBool::new(false));
        let a = CancellationChecker::new(flag.clone());
        let b = CancellationChecker::new(flag.clone());
        assert!(!a.is_cancelled());
        assert!(!b.is_cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(a.is_cancelled());
        assert!(b.is_cancelled());
    }

    #[test]
    fn checker_should_stop_is_disjunction() {
        // Architectural invariant: should_stop is the OR of
        // is_cancelled and is_expired. Both must be false for
        // should_stop to be false.
        let flag = Arc::new(AtomicBool::new(false));
        let c = CancellationChecker::new(flag.clone());
        assert!(!c.should_stop());
        flag.store(true, Ordering::Relaxed);
        assert!(c.should_stop());
        flag.store(false, Ordering::Relaxed);
        let future = Instant::now() + Duration::from_secs(60);
        let c2 = CancellationChecker::new(flag).with_deadline(future);
        assert!(!c2.should_stop());
    }

    #[test]
    fn checker_check_phase_propagates_flag() {
        // Architectural invariant: check_phase returns evidence
        // whose three dispatch/completion/publication flags all
        // reflect the shared flag's current state.
        let flag = Arc::new(AtomicBool::new(true));
        let c = CancellationChecker::new(flag);
        let ev = c.check_phase(
            &PhaseId("p1".into()),
            CancellationClass::Preemptible,
        );
        assert!(ev.cancelled_at_dispatch);
        assert!(ev.cancelled_at_completion);
        assert!(ev.cancelled_at_publication);
    }

    #[test]
    fn default_policy_allows_preemptible_interrupt() {
        // Architectural invariant: the default cancellation policy
        // is permissive — preemptible phases may be interrupted,
        // non-preemptible phases are waited on.
        let p = CancellationPolicy::default();
        assert!(p.allow_preemptible_interrupt);
        assert!(p.wait_for_non_preemptible);
        assert_eq!(p.non_preemptible_grace_ms, 5000);
    }
}
