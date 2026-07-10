//! WorkerIngressQueue — FIFO ingress queue for worker-bound requests.
//!
//! Systems that bridge external requests into the ECS world push entries
//! here.  The worker ingress system drains them in batches and creates
//! entities with the appropriate components.

use std::collections::VecDeque;

/// A single queued ingress entry carrying a request into the ECS world.
#[derive(Debug, Clone)]
pub struct IngressEntry {
    /// Entity ID assigned at drain time, or 0 before insertion.
    pub entity_id: u32,
    /// Unique request identifier.
    pub request_id: String,
    /// Serialized request payload.
    pub payload: Vec<u8>,
    /// Correlation key from the external bridge layer for response routing.
    pub bridge_correlation_key: String,
}

/// FIFO queue of incoming worker requests awaiting entity creation.
///
/// Systems push entries from bridge endpoints (HTTP, IPC, etc.) and the
/// ingress system drains them in order during its scheduling slot.  The
/// internal `VecDeque` provides amortised O(1) push and drain with
/// controllable batching.
#[derive(Debug)]
pub struct WorkerIngressQueue {
    queue: VecDeque<IngressEntry>,
}

impl WorkerIngressQueue {
    /// Create an empty ingress queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Push a new ingress entry onto the back of the queue.
    pub fn push(&mut self, entry: IngressEntry) {
        self.queue.push_back(entry);
    }

    /// Drain up to `max` entries from the front of the queue.
    ///
    /// Returns the drained entries.  The remaining entries stay in the queue
    /// for a future drain call.
    pub fn drain(&mut self, max: usize) -> Vec<IngressEntry> {
        let count = self.queue.len().min(max);
        self.queue.drain(..count).collect()
    }

    /// Returns `true` when the queue contains no entries.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

impl Default for WorkerIngressQueue {
    fn default() -> Self {
        Self::new()
    }
}
