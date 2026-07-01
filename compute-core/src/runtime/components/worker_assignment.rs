//! WorkerAssignment — which worker is serving a request and at what generation.
//!
//! Written when a request is dispatched to a worker process and updated if
//! the worker is re-assigned (e.g. after a crash + retry).

use std::time::Instant;

use crate::runtime::scheduling::component_id::SchedulableComponent;
use crate::runtime::components::WORKER_ASSIGNMENT_COMPONENT;
use serde::{Deserialize, Serialize};

/// Records the worker process assigned to a request entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerAssignment {
    /// Unique worker identifier (e.g. `"worker-42"`).
    pub worker_id: String,
    /// Monotonic assignment generation — incremented each time this request
    /// is re-assigned to a (potentially same) worker after a failure.
    pub generation: u32,
    /// Whether the request has been dispatched to the worker.
    pub dispatched: bool,
    /// Instant at which this assignment was created.
    #[serde(skip, default = "instant_now")]
    pub assigned_at: Instant,
}

fn instant_now() -> Instant {
    Instant::now()
}

impl WorkerAssignment {
    /// Create a new assignment for `worker_id` at the given `generation`.
    pub fn new(worker_id: impl Into<String>, generation: u32) -> Self {
        Self {
            worker_id: worker_id.into(),
            generation,
            dispatched: false,
            assigned_at: Instant::now(),
        }
    }

    /// Mark the request as having been dispatched to the worker.
    pub fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    /// Returns `true` when both the worker identifier and generation match.
    pub fn matches(&self, worker_id: &str, generation: u32) -> bool {
        self.worker_id == worker_id && self.generation == generation
    }
}

impl SchedulableComponent for WorkerAssignment {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_ASSIGNMENT_COMPONENT;
    const NAME: &'static str = "WorkerAssignment";
}
