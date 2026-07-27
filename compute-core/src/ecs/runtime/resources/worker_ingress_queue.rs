//! WorkerIngressQueue — FIFO ingress queue for worker-bound requests.
//!
//! Systems that bridge external requests into the ECS world push entries
//! here.  The worker ingress system drains them in batches and creates
//! entities with the appropriate components.

use prism_ecs_runtime::ports::ingress::IngressBridge;
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
    /// Optional constitutional bridge for submitting ingress requests
    /// through the canonical ECS path.
    ingress_bridge: Option<IngressBridge>,
}

impl WorkerIngressQueue {
    /// Create an empty ingress queue.
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            ingress_bridge: None,
        }
    }

    /// Attach an [`IngressBridge`] for routing submissions through the
    /// constitutional ingress path.
    pub fn set_ingress_bridge(&mut self, bridge: IngressBridge) {
        self.ingress_bridge = Some(bridge);
    }

    /// Submit an ingress request through the constitutional bridge, if one
    /// is attached.  Returns the allocated entity id on success, or `None`
    /// when no bridge is configured.
    pub fn submit_ingress_request(
        &self,
        transport: &str,
        method: &str,
        path: &str,
        body: Vec<u8>,
    ) -> Option<Result<u64, String>> {
        self.ingress_bridge
            .as_ref()
            .map(|b| b.submit_request(transport, method, path, body))
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
