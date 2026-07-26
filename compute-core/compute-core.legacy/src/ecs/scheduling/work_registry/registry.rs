//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Work status enum (legacy).
//!
//! The [`WorkRegistry`] write path has been replaced by the constitutional
//! [`WorkLifecycleBridge`].  Only [`WorkStatus`] is retained for compatibility
//! in downstream consumers such as [`completion_bridge`] and [`receipt`].

use serde::{Deserialize, Serialize};

// ── Work status ─────────────────────────────────────────────────────────────

/// Complete state machine for a single work item.
///
/// # State machine
///
/// ```text
///                                   ┌─────────────────┐
///                                   │     Created      │
///                                   └────────┬─────────┘
///                                          │
///                          ┌───────────────┼───────────────┐
///                          ▼               ▼               ▼
///                     ┌──────────┐   ┌──────────┐   ┌──────────┐
///                     │  Ready   │   │  Denied  │   │ (cancel) │
///                     └────┬─────┘   └──────────┘   └──────────┘
///                          │            [terminal failure]
///                          ▼
///                     ┌──────────┐
///                     │ Selected │
///                     └────┬─────┘
///                          │
///                          ▼
///                   ┌──────────────┐
///                   │Cap.Reserved  │
///                   └──────┬───────┘
///                          │
///                  ┌───────┼───────────┐
///                  ▼       ▼           │
///            ┌─────────┐ ┌──────────┐ │
///            │SlotsRes.│ │  Denied  │ │
///            └────┬────┘ └──────────┘ │
///                 │       [terminal]  │
///            ┌────┼────────┐         │
///            ▼    ▼        │         │
///       ┌────────┐  ┌─────────────┐  │
///       │Submit  │  │FallbackPend.│  │
///       └───┬────┘  └──────┬──────┘  │
///           │              │         │
///      ┌────┼────┐         ▼         │
///      ▼    ▼    │   ┌────────────┐  │
/// ┌────────┐ ┌──────┐ │FallbackRun│  │
/// │Running │ │Submit│ └──┬──────┬──┘  │
/// └───┬────┘ │Fail  │    │      │     │
///     │      └──────┘    │      │     │
///  ┌──┼──┐    [terminal] │      │     │
///  ▼  ▼  ▼              │      │     │
/// ┌──┐ ┌──┐ ┌────────┐  │      │     │
/// │C │ │EF│ │TimedOut│◄─┘      │     │
/// │o │ └──┘ └────────┘         │     │
/// │m │  │       │              │     │
/// │p │  └───┬───┘              │     │
/// │l │      │                  │     │
/// │e │  ┌────┴─────┐           │     │
/// │t │  ▼          ▼           │     │
/// │e │┌────────┐┌────────┐     │     │
/// │d ││FbackPd││FailTerm│     │     │
/// │  │└───┬───┘└────────┘     │     │
/// │  │    │    [terminal]     │     │
/// │  │    ▼                   │     │
/// │  │┌────────┐              │     │
/// │  ││FbackRun│◄─────────────┘     │
/// │  │└───┬────┘                   │
/// │  │  ┌─┴──┐                    │
/// │  │  ▼    ▼                     │
/// │  │┌──┐ ┌─────┐                │
/// │  ││C │ │EF/TO│                │
/// │  ││om│ └──┬──┘                │
/// │  ││pl│    │                   │
/// │  ││et│┌───┴────┐             │
/// │  ││ed│▼        ▼            │
/// │  │└──┘┌─────┐┌────────┐    │
/// │  │    │FbPd││FailTerm│    │
/// │  │    └──┬──┘└────────┘   │
/// │  ▼      ▼                 │
/// │  ┌──────────┐             │
/// │  │OutputRdy │             │
/// │  └────┬─────┘             │
/// │       │                   │
/// │  ┌────┼───────┐          │
/// │  ▼    ▼       │          │
/// │┌────┐┌────────┐│         │
/// ││Cons││FallbPd ││         │
/// │└──┬─┘└────────┘│         │
/// │   │            │          │
/// │   ▼            │          │
/// │┌──────┐        │          │
/// ││Releas│◄───────┘          │
/// │└──────┘  etc.             │
/// └───────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkStatus {
    /// Initial state — work item created but not yet ready for selection.
    Created,
    /// Ready to be selected by a lane scheduler.
    Ready,
    /// Selected by a lane scheduler for execution.
    Selected,
    /// Backend capacity has been reserved for this work item.
    CapacityReserved,
    /// Activation slots have been reserved.
    SlotsReserved,
    /// Submitted to the backend for execution.
    Submitted,
    /// Currently executing on the backend.
    Running,
    /// Backend execution completed successfully.
    Completed,
    /// Output is ready for consumption by the next phase.
    OutputReady,
    /// Output has been consumed by the downstream phase.
    Consumed,
    /// Resources released — terminal success.
    Released,
    // ── Terminal failures ──────────────────────────────────────────────
    /// Work was denied (e.g. capacity unavailable).
    Denied,
    /// Work was cancelled before submission.
    CancelledBeforeSubmit,
    /// Backend submission failed (non-retryable).
    SubmitFailed,
    /// Backend execution failed.
    ExecutionFailed,
    /// Backend execution timed out.
    TimedOut,
    /// Fallback execution is pending (alternative lane).
    FallbackPending,
    /// Fallback execution is running on an alternative lane.
    FallbackRunning,
    /// Terminal failure after all fallback attempts exhausted.
    FailedTerminal,
}

impl WorkStatus {
    /// Returns `true` if this status is a terminal (non-transitioning) state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            WorkStatus::Released
                | WorkStatus::Denied
                | WorkStatus::CancelledBeforeSubmit
                | WorkStatus::SubmitFailed
                | WorkStatus::FailedTerminal
        )
    }

    /// Returns `true` if this status represents a successful outcome.
    ///
    /// Only [`Released`](WorkStatus::Released) is a terminal success state.
    /// All other states are either intermediate or terminal failures.
    pub fn is_success(&self) -> bool {
        matches!(self, WorkStatus::Released)
    }

    /// Returns `true` if this status represents a terminal failure.
    pub fn is_failure(&self) -> bool {
        matches!(
            self,
            WorkStatus::Denied
                | WorkStatus::CancelledBeforeSubmit
                | WorkStatus::SubmitFailed
                | WorkStatus::FailedTerminal
        )
    }

    /// Returns the exhaustive set of legal transition targets from this state.
    pub fn legal_transitions(&self) -> &'static [WorkStatus] {
        match self {
            WorkStatus::Created => &[WorkStatus::Ready, WorkStatus::Denied],
            WorkStatus::Ready => &[WorkStatus::Selected, WorkStatus::CancelledBeforeSubmit],
            WorkStatus::Selected => &[WorkStatus::CapacityReserved],
            WorkStatus::CapacityReserved => &[WorkStatus::SlotsReserved, WorkStatus::Denied],
            WorkStatus::SlotsReserved => &[WorkStatus::Submitted, WorkStatus::FallbackPending],
            WorkStatus::Submitted => &[WorkStatus::Running, WorkStatus::SubmitFailed],
            WorkStatus::Running => &[
                WorkStatus::Completed,
                WorkStatus::ExecutionFailed,
                WorkStatus::TimedOut,
            ],
            WorkStatus::Completed => &[WorkStatus::OutputReady],
            WorkStatus::OutputReady => &[WorkStatus::Consumed, WorkStatus::FallbackPending],
            WorkStatus::Consumed => &[WorkStatus::Released],
            // Terminal success — no transitions.
            WorkStatus::Released => &[],
            // Terminal failures — no transitions.
            WorkStatus::Denied => &[],
            WorkStatus::CancelledBeforeSubmit => &[],
            WorkStatus::SubmitFailed => &[],
            // Non-terminal failure — may retry or give up.
            WorkStatus::ExecutionFailed => {
                &[WorkStatus::FallbackPending, WorkStatus::FailedTerminal]
            }
            WorkStatus::TimedOut => &[WorkStatus::FallbackPending, WorkStatus::FailedTerminal],
            WorkStatus::FallbackPending => &[WorkStatus::FallbackRunning],
            WorkStatus::FallbackRunning => &[
                WorkStatus::Completed,
                WorkStatus::ExecutionFailed,
                WorkStatus::TimedOut,
                WorkStatus::FailedTerminal,
            ],
            WorkStatus::FailedTerminal => &[],
        }
    }
}
