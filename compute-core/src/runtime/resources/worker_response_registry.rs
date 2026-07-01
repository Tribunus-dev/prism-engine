//! WorkerResponseRegistry — typed response routing for worker completions.
//!
//! When a worker produces intermediate tokens or a terminal result, the
//! event drain system looks up the corresponding response sink in this
//! registry and delivers the data.  Sinks are opaque `Box<dyn Any + Send>`
//! so the bridge layer can use whatever channel/type suffices.
//!
//! In addition to the generic sink table, the registry supports oneshot
//! mpsc-based pending requests where the caller provides a `Sender<String>`
//! and the event drain delivers the response string when the worker completes.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Mutex;

/// A pending generation request registered with an mpsc response channel.
///
/// When the worker completes and the event drain system encounters the
/// corresponding Completion event, it sends the response payload through
/// `response_tx` and removes the entry.
pub struct PendingRequest {
    /// Unique request identifier.
    pub request_id: String,
    /// Optional oneshot sender for delivering the response payload.
    pub response_tx: Option<std::sync::mpsc::Sender<String>>,
    /// Instant when this request was registered.
    pub created_at: std::time::Instant,
}

/// Registry mapping `(request_id, entity_id)` pairs to external response
/// sinks.
///
/// The response sink is type-erased so any bridge channel (oneshot, mpsc,
/// custom callbacks) can be stored without the ECS core knowing the concrete
/// type.
pub struct WorkerResponseRegistry {
    /// Maps `"{request_id}:{entity_id}"` → opaque response sink.
    entries: Mutex<HashMap<String, Box<dyn Any + Send>>>,
    /// Maps `request_id` → pending request with an mpsc oneshot sender.
    pending: Mutex<HashMap<String, PendingRequest>>,
}

impl WorkerResponseRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Register a response sink for a `(request_id, entity_id)` pair.
    ///
    /// The sink is type-erased; the bridge layer that created it retains
    /// knowledge of its concrete type.
    pub fn register(
        &self,
        request_id: &str,
        entity_id: u32,
        sink: Box<dyn Any + Send>,
    ) {
        let key = Self::key(request_id, entity_id);
        let mut guard = self.entries.lock().expect("WorkerResponseRegistry lock poisoned");
        guard.insert(key, sink);
    }

    /// Remove and return the response sink for a `(request_id, entity_id)` pair.
    pub fn remove(
        &self,
        request_id: &str,
        entity_id: u32,
    ) -> Option<Box<dyn Any + Send>> {
        let key = Self::key(request_id, entity_id);
        let mut guard = self.entries.lock().expect("WorkerResponseRegistry lock poisoned");
        guard.remove(&key)
    }

    /// Deliver an intermediate token to the response sink.
    ///
    /// The caller provides the concrete sink type `T` and a closure that
    /// performs the delivery.  Returns `Ok(())` on success or an error
    /// string if the sink is missing or the type does not match.
    pub fn deliver_token<T: 'static>(
        &self,
        request_id: &str,
        entity_id: u32,
        deliver: impl FnOnce(&mut T) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut guard = self.entries.lock().expect("WorkerResponseRegistry lock poisoned");
        let key = Self::key(request_id, entity_id);
        let sink = guard.get_mut(&key).ok_or_else(|| {
            format!("no sink registered for ({request_id}, {entity_id})")
        })?;
        let typed = sink.downcast_mut::<T>().ok_or_else(|| {
            format!("sink type mismatch for ({request_id}, {entity_id})")
        })?;
        deliver(typed)
    }

    /// Deliver a terminal result and remove the sink.
    ///
    /// Like `deliver_token` but removes the entry after delivery.
    pub fn deliver_terminal<T: 'static>(
        &self,
        request_id: &str,
        entity_id: u32,
        deliver: impl FnOnce(T) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut guard = self.entries.lock().expect("WorkerResponseRegistry lock poisoned");
        let key = Self::key(request_id, entity_id);
        let sink = guard.remove(&key).ok_or_else(|| {
            format!("no sink registered for ({request_id}, {entity_id})")
        })?;
        let typed: Box<T> = sink.downcast::<T>().map_err(|_| {
            format!("sink type mismatch for ({request_id}, {entity_id})")
        })?;
        deliver(*typed)
    }

    // ── Oneshot pending-request API ────────────────────────────────────

    /// Register a pending request with an mpsc response channel.
    ///
    /// When the worker completes and the event drain system encounters the
    /// Completion event for `request_id`, it will send the response string
    /// through `tx` and remove the entry.
    pub fn register_pending(
        &self,
        request_id: &str,
        tx: std::sync::mpsc::Sender<String>,
    ) {
        let mut guard = self.pending.lock().expect("WorkerResponseRegistry lock poisoned");
        guard.insert(
            request_id.to_string(),
            PendingRequest {
                request_id: request_id.to_string(),
                response_tx: Some(tx),
                created_at: std::time::Instant::now(),
            },
        );
    }

    /// Remove and return a pending request by `request_id`.
    ///
    /// Returns `None` when no pending request is registered for that id.
    pub fn remove_pending(&self, request_id: &str) -> Option<PendingRequest> {
        let mut guard = self.pending.lock().expect("WorkerResponseRegistry lock poisoned");
        guard.remove(request_id)
    }

    /// Deliver a response string to the pending request's oneshot channel.
    ///
    /// Looks up the pending request by `request_id`, sends `response`
    /// through its channel, and removes the entry.  Returns `Ok(())` on
    /// success or an error string if the request is not pending or the
    /// channel receiver has dropped.
    pub fn deliver_response(
        &self,
        request_id: &str,
        response: String,
    ) -> Result<(), String> {
        let mut guard = self.pending.lock().expect("WorkerResponseRegistry lock poisoned");
        let pending = guard.remove(request_id).ok_or_else(|| {
            format!("no pending request registered for '{request_id}'")
        })?;
        match pending.response_tx {
            Some(tx) => tx.send(response).map_err(|e| {
                format!("failed to deliver response for '{request_id}': receiver dropped — {e}")
            }),
            None => Err(format!(
                "pending request '{request_id}' has no response channel"
            )),
        }
    }

    /// Build the lookup key from request_id and entity_id.
    fn key(request_id: &str, entity_id: u32) -> String {
        format!("{request_id}:{entity_id}")
    }
}

impl Default for WorkerResponseRegistry {
    fn default() -> Self {
        Self::new()
    }
}
