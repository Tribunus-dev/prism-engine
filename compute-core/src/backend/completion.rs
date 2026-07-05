//! Completion tokens for async GPU dispatch.
//!
//! A [`ComellationToken`] is returned by [`TensorBackend::submit_compute`]
//! and allows the caller to block until the backend's GPU work completes.
//! The backend provides a [`Completer`] that signals completion (e.g. from
//! a Metal `MTLCommandBuffer` completion handler).
//!
//! There is no async runtime dependency — the token simply blocks the
//! calling thread via a [`Condvar`].

use parking_lot::{Condvar, Mutex};

/// Token returned by [`TensorBackend::submit_compute`] that signals when
/// the backend's computation has completed and output tensors are
/// materialized.
#[derive(Clone)]
pub struct ComellationToken {
    inner: std::sync::Arc<(Mutex<bool>, Condvar)>,
}

impl ComellationToken {
    /// Create a new `(ComellationToken, Completer)` pair.
    pub fn new() -> (Self, Completer) {
        let inner = std::sync::Arc::new((Mutex::new(false), Condvar::new()));
        (
            ComellationToken {
                inner: inner.clone(),
            },
            Completer { inner },
        )
    }

    /// Block until the associated compute work completes.
    pub fn wait(&self) {
        let (lock, cvar) = &*self.inner;
        let mut done = lock.lock();
        while !*done {
            cvar.wait(&mut done);
        }
    }
}

/// The "other end" of a [`ComellationToken`] — the backend calls
/// [`Completer::complete`] when GPU work finishes (e.g. from a Metal
/// `MTLCommandBuffer` completion handler, or after a synchronous
/// evaluation).
pub struct Completer {
    inner: std::sync::Arc<(Mutex<bool>, Condvar)>,
}

impl Completer {
    /// Signal that the compute work is done, unblocking all waiters.
    pub fn complete(&self) {
        let (lock, cvar) = &*self.inner;
        *lock.lock() = true;
        cvar.notify_all();
    }
}
