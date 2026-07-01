//! WorkerLifecycle — phase-machine tracking a worker request from submission
//! through dispatch, streaming, completion, and terminal states.

use std::time::Instant;
use std::fmt;
use serde::{Deserialize, Serialize};

use crate::runtime::scheduling::component_id::SchedulableComponent;
use crate::runtime::components::WORKER_LIFECYCLE_COMPONENT;

// ---------------------------------------------------------------------------
// Phase enumeration
// ---------------------------------------------------------------------------

/// All phases a worker-bound request may pass through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerRequestPhase {
    /// Request queued, awaiting dispatch.
    Queued,
    /// Being dispatched to a worker process.
    Dispatching,
    /// Dispatched; expecting the first event from the worker.
    AwaitingFirstEvent,
    /// Actively streaming output tokens from the worker.
    Streaming,
    /// Final output received; completing the request.
    Completing,
    /// Request finished successfully. Terminal.
    Completed,
    /// Request finished with a failure. Terminal.
    Failed,
    /// Cancellation has been requested; waiting for acknowledgement.
    CancelRequested,
    /// Worker acknowledged cancellation. Terminal.
    Cancelled,
    /// Request abandoned (e.g. entity despawn while non-terminal). Terminal.
    Abandoned,
}

impl WorkerRequestPhase {
    /// Returns `true` for phases that represent a final state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled | Self::Abandoned)
    }

    /// Returns `true` for phases that are still in progress.
    pub fn is_active(self) -> bool {
        !self.is_terminal()
    }
}

impl fmt::Display for WorkerRequestPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Dispatching => write!(f, "Dispatching"),
            Self::AwaitingFirstEvent => write!(f, "AwaitingFirstEvent"),
            Self::Streaming => write!(f, "Streaming"),
            Self::Completing => write!(f, "Completing"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::CancelRequested => write!(f, "CancelRequested"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Abandoned => write!(f, "Abandoned"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transition error
// ---------------------------------------------------------------------------

/// Error returned when an invalid lifecycle transition is attempted.
#[derive(Debug, Clone)]
pub struct LifecycleTransitionError {
    /// Phase the request was in.
    pub from_phase: WorkerRequestPhase,
    /// Phase the request was being moved to.
    pub to_phase: WorkerRequestPhase,
    /// Human-readable explanation.
    pub reason: String,
}

impl fmt::Display for LifecycleTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid lifecycle transition: {} -> {}: {}",
            self.from_phase, self.to_phase, self.reason,
        )
    }
}

impl std::error::Error for LifecycleTransitionError {}

// ---------------------------------------------------------------------------
// Lifecycle component
// ---------------------------------------------------------------------------

/// Tracks the lifecycle phase of a worker request on its ECS entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLifecycle {
    /// Current phase.
    pub phase: WorkerRequestPhase,
    /// Number of times the request has been retried.
    pub retry_count: u32,
    /// Instant of the most recent phase transition.
    #[serde(skip, default = "instant_now")]
    pub last_transition_at: Instant,
}

fn instant_now() -> Instant {
    Instant::now()
}

impl WorkerLifecycle {
    /// Create a new lifecycle starting in [`WorkerRequestPhase::Queued`].
    pub fn new() -> Self {
        Self {
            phase: WorkerRequestPhase::Queued,
            retry_count: 0,
            last_transition_at: Instant::now(),
        }
    }

    /// Attempt a transition from the current phase to `target`.
    ///
    /// Returns `Ok(())` on success, or a [`LifecycleTransitionError`] with
    /// a descriptive reason when the transition is disallowed.
    pub fn transition_to(&mut self, target: WorkerRequestPhase) -> Result<(), LifecycleTransitionError> {
        Self::validate_transition(self.phase, target)?;
        self.phase = target;
        self.last_transition_at = Instant::now();
        Ok(())
    }

    /// Validate a transition between two phases without applying it.
    ///
    /// # Allowed transitions
    ///
    /// | From              | To                                                                 |
    /// |-------------------|--------------------------------------------------------------------|
    /// | `Queued`          | `Dispatching`, `CancelRequested`                                   |
    /// | `Dispatching`     | `AwaitingFirstEvent`, `Failed`, `CancelRequested`                  |
    /// | `AwaitingFirstEvent` | `Streaming`, `Failed`, `CancelRequested`                        |
    /// | `Streaming`       | `Completing`, `Failed`, `CancelRequested`                          |
    /// | `Completing`      | `Completed`, `Failed`, `CancelRequested`                           |
    /// | `CancelRequested` | `Cancelled`, `Failed`                                              |
    /// | *Terminal*        | *none*                                                             |
    /// | *Active*          | `Abandoned`                                                        |
    pub fn validate_transition(
        from: WorkerRequestPhase,
        to: WorkerRequestPhase,
    ) -> Result<(), LifecycleTransitionError> {
        // Terminal states have no outgoing transitions.
        if from.is_terminal() {
            return Err(LifecycleTransitionError {
                from_phase: from,
                to_phase: to,
                reason: format!(
                    "cannot transition from terminal phase '{}'",
                    from,
                ),
            });
        }

        // `Abandoned` is reachable from any non-terminal phase.
        if to == WorkerRequestPhase::Abandoned {
            return Ok(());
        }

        let allowed = match from {
            WorkerRequestPhase::Queued => {
                matches!(to, WorkerRequestPhase::Dispatching | WorkerRequestPhase::CancelRequested)
            }
            WorkerRequestPhase::Dispatching => {
                matches!(
                    to,
                    WorkerRequestPhase::AwaitingFirstEvent
                        | WorkerRequestPhase::Failed
                        | WorkerRequestPhase::CancelRequested
                )
            }
            WorkerRequestPhase::AwaitingFirstEvent => {
                matches!(
                    to,
                    WorkerRequestPhase::Streaming
                        | WorkerRequestPhase::Failed
                        | WorkerRequestPhase::CancelRequested
                )
            }
            WorkerRequestPhase::Streaming => {
                matches!(
                    to,
                    WorkerRequestPhase::Completing
                        | WorkerRequestPhase::Failed
                        | WorkerRequestPhase::CancelRequested
                )
            }
            WorkerRequestPhase::Completing => {
                matches!(
                    to,
                    WorkerRequestPhase::Completed
                        | WorkerRequestPhase::Failed
                        | WorkerRequestPhase::CancelRequested
                )
            }
            WorkerRequestPhase::CancelRequested => {
                matches!(to, WorkerRequestPhase::Cancelled | WorkerRequestPhase::Failed)
            }
            // Terminal phases already caught above.
            WorkerRequestPhase::Completed
            | WorkerRequestPhase::Failed
            | WorkerRequestPhase::Cancelled
            | WorkerRequestPhase::Abandoned => unreachable!(),
        };

        if allowed {
            Ok(())
        } else {
            Err(LifecycleTransitionError {
                from_phase: from,
                to_phase: to,
                reason: format!("transition from '{}' to '{}' is not allowed", from, to),
            })
        }
    }
}

impl Default for WorkerLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulableComponent for WorkerLifecycle {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_LIFECYCLE_COMPONENT;
    const NAME: &'static str = "WorkerLifecycle";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_to_dispatching() {
        let mut lc = WorkerLifecycle::new();
        assert_eq!(lc.phase, WorkerRequestPhase::Queued);
        assert!(lc.transition_to(WorkerRequestPhase::Dispatching).is_ok());
        assert_eq!(lc.phase, WorkerRequestPhase::Dispatching);
    }

    #[test]
    fn full_lifecycle() {
        let mut lc = WorkerLifecycle::new();
        // Happy path: Queued -> Dispatching -> AwaitingFirstEvent -> Streaming -> Completing -> Completed.
        lc.transition_to(WorkerRequestPhase::Dispatching).unwrap();
        lc.transition_to(WorkerRequestPhase::AwaitingFirstEvent).unwrap();
        lc.transition_to(WorkerRequestPhase::Streaming).unwrap();
        lc.transition_to(WorkerRequestPhase::Completing).unwrap();
        lc.transition_to(WorkerRequestPhase::Completed).unwrap();
        assert_eq!(lc.phase, WorkerRequestPhase::Completed);
    }

    #[test]
    fn illegal_transition() {
        let mut lc = WorkerLifecycle::new();
        // Queued -> Streaming is not allowed.
        let err = lc.transition_to(WorkerRequestPhase::Streaming).unwrap_err();
        assert_eq!(err.from_phase, WorkerRequestPhase::Queued);
        assert_eq!(err.to_phase, WorkerRequestPhase::Streaming);
        assert!(err.reason.contains("not allowed"));
        // Phase should remain unchanged.
        assert_eq!(lc.phase, WorkerRequestPhase::Queued);
    }

    #[test]
    fn terminal_to_terminal_rejected() {
        let mut lc = WorkerLifecycle::new();
        lc.transition_to(WorkerRequestPhase::Dispatching).unwrap();
        lc.transition_to(WorkerRequestPhase::Failed).unwrap();
        assert!(lc.phase.is_terminal());
        // Cannot transition from a terminal state.
        let err = lc.transition_to(WorkerRequestPhase::Abandoned).unwrap_err();
        assert_eq!(err.from_phase, WorkerRequestPhase::Failed);
        assert!(err.reason.contains("terminal"));
    }

    #[test]
    fn cancel_from_active() {
        // CancelRequested should be reachable from active states.
        let mut lc = WorkerLifecycle::new();
        // Queued -> CancelRequested
        assert!(lc.transition_to(WorkerRequestPhase::CancelRequested).is_ok());
        assert_eq!(lc.phase, WorkerRequestPhase::CancelRequested);
    }

    #[test]
    fn abandon_from_nonterminal() {
        let mut lc = WorkerLifecycle::new();
        // Queued -> Dispatching -> Abandoned
        lc.transition_to(WorkerRequestPhase::Dispatching).unwrap();
        assert!(lc.transition_to(WorkerRequestPhase::Abandoned).is_ok());
        assert_eq!(lc.phase, WorkerRequestPhase::Abandoned);
        assert!(lc.phase.is_terminal());
    }

    #[test]
    fn abandon_after_complete_rejected() {
        let mut lc = WorkerLifecycle::new();
        lc.transition_to(WorkerRequestPhase::Dispatching).unwrap();
        lc.transition_to(WorkerRequestPhase::AwaitingFirstEvent).unwrap();
        lc.transition_to(WorkerRequestPhase::Streaming).unwrap();
        lc.transition_to(WorkerRequestPhase::Completing).unwrap();
        lc.transition_to(WorkerRequestPhase::Completed).unwrap();
        let err = lc.transition_to(WorkerRequestPhase::Abandoned).unwrap_err();
        assert!(err.reason.contains("terminal"));
    }

    #[test]
    fn is_terminal_and_is_active() {
        let terminal = [
            WorkerRequestPhase::Completed,
            WorkerRequestPhase::Failed,
            WorkerRequestPhase::Cancelled,
            WorkerRequestPhase::Abandoned,
        ];
        let active = [
            WorkerRequestPhase::Queued,
            WorkerRequestPhase::Dispatching,
            WorkerRequestPhase::AwaitingFirstEvent,
            WorkerRequestPhase::Streaming,
            WorkerRequestPhase::Completing,
            WorkerRequestPhase::CancelRequested,
        ];
        for p in terminal {
            assert!(p.is_terminal(), "{p:?} should be terminal");
            assert!(!p.is_active(), "{p:?} should not be active");
        }
        for p in active {
            assert!(p.is_active(), "{p:?} should be active");
            assert!(!p.is_terminal(), "{p:?} should not be terminal");
        }
    }
}
