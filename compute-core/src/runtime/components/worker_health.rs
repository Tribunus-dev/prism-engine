//! WorkerHeartbeat and WorkerOutcome — liveness tracking and terminal result
//! for worker-bound requests.

use std::fmt;
use std::time::Instant;
use serde::{Deserialize, Serialize};

use crate::runtime::scheduling::component_id::SchedulableComponent;
use crate::runtime::components::{WORKER_HEARTBEAT_COMPONENT, WORKER_OUTCOME_COMPONENT};

// ---------------------------------------------------------------------------
// Terminal status
// ---------------------------------------------------------------------------

/// Terminal disposition of a worker request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminalStatus {
    /// Request completed successfully.
    Success,
    /// Request failed with an error.
    Failed,
    /// Request was cancelled.
    Cancelled,
    /// Request was abandoned (entity despawned while active).
    Abandoned,
}

impl fmt::Display for TerminalStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Abandoned => write!(f, "Abandoned"),
        }
    }
}

// ---------------------------------------------------------------------------
// Error category
// ---------------------------------------------------------------------------

/// High-level categorisation of a worker failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerErrorCategory {
    /// No error (successful outcome).
    None,
    /// Worker process crashed unexpectedly.
    ProcessCrash,
    /// Internal engine error.
    Internal,
    /// Worker violated the request-response protocol.
    ProtocolViolation,
    /// Request timed out.
    Timeout,
    /// Worker ran out of memory, GPU VRAM, or other resource.
    ResourceExhaustion,
    /// Error that does not fit any other category.
    Unknown,
}

impl fmt::Display for WorkerErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::ProcessCrash => write!(f, "ProcessCrash"),
            Self::Internal => write!(f, "Internal"),
            Self::ProtocolViolation => write!(f, "ProtocolViolation"),
            Self::Timeout => write!(f, "Timeout"),
            Self::ResourceExhaustion => write!(f, "ResourceExhaustion"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Heartbeat component
// ---------------------------------------------------------------------------

/// Periodic heartbeat from a worker process for liveness detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    /// Unique worker identifier.
    pub worker_id: String,
    /// Assignment generation this heartbeat corresponds to.
    pub assignment_generation: u32,
    /// Most recent heartbeat timestamp.
    #[serde(skip, default = "instant_now")]
    pub last_heartbeat_at: Instant,
    /// Number of consecutive missed heartbeats.
    pub consecutive_misses: u32,
}

fn instant_now() -> Instant {
    Instant::now()
}

impl WorkerHeartbeat {
    /// Create a new heartbeat record.
    pub fn new(worker_id: impl Into<String>, assignment_generation: u32) -> Self {
        Self {
            worker_id: worker_id.into(),
            assignment_generation,
            last_heartbeat_at: Instant::now(),
            consecutive_misses: 0,
        }
    }

    /// Record a heartbeat reception.
    pub fn mark_received(&mut self) {
        self.last_heartbeat_at = Instant::now();
        self.consecutive_misses = 0;
    }

    /// Increment the miss counter (called when a heartbeat is expected but
    /// not received within the watchdog interval).
    pub fn mark_missed(&mut self) {
        self.consecutive_misses += 1;
    }
}

impl SchedulableComponent for WorkerHeartbeat {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_HEARTBEAT_COMPONENT;
    const NAME: &'static str = "WorkerHeartbeat";
}

/// Default value for `completed_at` when deserializing without the field.
fn now_instant() -> Instant {
    Instant::now()
}

// ---------------------------------------------------------------------------
// Outcome component
// ---------------------------------------------------------------------------

/// Terminal result of a worker request, written once at completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerOutcome {
    /// Terminal status of the request.
    pub status: TerminalStatus,
    /// High-level error category (meaningful only when status is `Failed`).
    pub error_category: WorkerErrorCategory,
    /// Optional protocol-level error code (e.g. gRPC status, HTTP code).
    pub protocol_error_code: Option<u32>,
    /// Timestamp of completion.
    #[serde(skip)]
    #[serde(default = "now_instant")]
    pub completed_at: Instant,
    /// Assignment generation at which the outcome was produced.
    pub assignment_generation: u32,
}

impl WorkerOutcome {
    /// Create a new outcome record.
    pub fn new(
        status: TerminalStatus,
        error_category: WorkerErrorCategory,
        protocol_error_code: Option<u32>,
        assignment_generation: u32,
    ) -> Self {
        Self {
            status,
            error_category,
            protocol_error_code,
            completed_at: Instant::now(),
            assignment_generation,
        }
    }

    /// Returns `true` when the outcome represents a successful completion.
    pub fn is_success(&self) -> bool {
        self.status == TerminalStatus::Success
    }
}

impl SchedulableComponent for WorkerOutcome {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_OUTCOME_COMPONENT;
    const NAME: &'static str = "WorkerOutcome";
}
