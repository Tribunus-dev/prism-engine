//! WorkerRequest — immutable payload for a worker-bound inference request.
//!
//! Immutable after entity creation except for explicit cancellation metadata.
//! Contains the request identifier, execution payload, request class,
//! creation timestamp, and optional cancellation token reference.

use std::time::Instant;

use crate::runtime::scheduling::component_id::SchedulableComponent;
use crate::runtime::components::WORKER_REQUEST_COMPONENT;

/// A submitted inference request awaiting or undergoing worker execution.
///
/// This component is written once at entity creation and read by systems.
/// It must not contain child-process handles, mutable response sinks,
/// or worker-owned state.
#[derive(Debug, Clone)]
pub struct WorkerRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Serialized request payload (e.g. generation params, token IDs).
    pub payload: Vec<u8>,
    /// Request class for scheduling and admission decisions.
    pub request_class: RequestClass,
    /// Monotonic creation timestamp.
    pub created_at: Instant,
    /// Optional cancellation state.
    pub cancellation: Option<CancellationState>,
}

/// Broad classification of a worker request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestClass {
    /// Text generation (prefill + decode).
    Generate,
    /// Embedding extraction.
    Embed,
    /// Image generation.
    ImageGen,
    /// Audio generation.
    AudioGen,
}

/// Optional cancellation metadata attached after the request is created.
#[derive(Debug, Clone)]
pub struct CancellationState {
    /// Whether cancellation has been requested.
    pub requested: bool,
    /// Reason for cancellation, if any.
    pub reason: Option<String>,
}

impl WorkerRequest {
    /// Create a new worker request.
    pub fn new(request_id: impl Into<String>, payload: Vec<u8>, request_class: RequestClass) -> Self {
        Self {
            request_id: request_id.into(),
            payload,
            request_class,
            created_at: Instant::now(),
            cancellation: None,
        }
    }

    /// Mark this request as cancelled.
    pub fn cancel(&mut self, reason: Option<String>) {
        self.cancellation = Some(CancellationState {
            requested: true,
            reason,
        });
    }

    /// Returns true when cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .map_or(false, |c| c.requested)
    }
}

impl SchedulableComponent for WorkerRequest {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_REQUEST_COMPONENT;
    const NAME: &'static str = "WorkerRequest";
}
