//! WorkerEventSource — event ingress from worker processes.
//!
//! Workers emit events (heartbeats, token generation, completions, etc.)
//! through an IPC channel.  In Slice 2 this resource wraps the legacy IPC
//! reader and exposes a batched drain API for the event processing system.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

/// Kind of event emitted by a worker process.
#[derive(Debug, Clone)]
pub enum EventKind {
    /// Periodic liveness signal from the worker.
    Heartbeat,
    /// A generated token identified by its position in the output sequence.
    Token {
        /// Monotonically increasing token position within the request.
        token_id: u32,
        /// Raw token bytes.
        bytes: Vec<u8>,
    },
    /// Request completed with a status string.
    Completion {
        /// Completion status (e.g. "ok", "cancelled").
        status: String,
    },
    /// Request failed with a machine-readable category and optional code.
    Failure {
        /// Failure category (e.g. "timeout", "oom", "internal").
        category: String,
        /// Optional numeric error code.
        code: Option<u32>,
    },
    /// Intermediate progress update.
    Progress {
        /// Number of tokens generated since the last progress event.
        tokens_generated: u32,
    },
}

/// An event envelope produced by a worker and consumed by the event system.
#[derive(Debug, Clone)]
pub struct WorkerEventEnvelope {
    /// Identifier of the worker that produced this event.
    pub worker_id: String,
    /// Request identifier this event pertains to.
    pub request_id: String,
    /// Assignment generation counter for ordering and staleness detection.
    pub assignment_generation: u32,
    /// Monotonically increasing sequence number within this worker.
    pub event_sequence: u64,
    /// Instant when this event was received by the runtime.
    pub receipt_timestamp: Instant,
    /// The event payload.
    pub kind: EventKind,
}

/// Buffered event source from worker IPC.
///
/// Events are pushed by the IPC reader (or placeholder injection) and
/// drained in configurable batches by the event drain system.  Thread-safe
/// via internal mutex.
#[derive(Debug)]
pub struct WorkerEventSource {
    events: Mutex<VecDeque<WorkerEventEnvelope>>,
}

impl WorkerEventSource {
    /// Create an empty event source.
    pub fn new() -> Self {
        Self {
            events: Mutex::new(VecDeque::new()),
        }
    }

    /// Drain up to `max` events from the buffer.
    ///
    /// Returns the drained events in FIFO order.  Remaining events stay in
    /// the buffer for a future drain.
    pub fn drain_batch(&self, max: usize) -> Vec<WorkerEventEnvelope> {
        let mut guard = self.events.lock().expect("WorkerEventSource lock poisoned");
        let count = guard.len().min(max);
        guard.drain(..count).collect()
    }
}

impl Default for WorkerEventSource {
    fn default() -> Self {
        Self::new()
    }
}
