//! Completion tokens for async GPU dispatch.
//!
//! A [`ComputationToken`] is returned by [`TensorBackend::submit_compute`]
//! and tracks GPU completion (e.g. via MTLCommandBuffer completion handler,
//! or a Condvar signal for generic backends). The token supports both
//! synchronous [`wait`](ComellationToken::wait) and asynchronous
//! [`then`](ComellationToken::then) completion notification.

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use block::ConcreteBlock;

use parking_lot::{Condvar, Mutex};

static NEXT_TOKEN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_token_id() -> u64 {
    NEXT_TOKEN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Token wrapper returned by [`TensorBackend::submit_compute`].
///
/// Backend-specific variants carry the appropriate completion tracking:
/// - `Metal` — backed by an `MTLCommandBuffer` completion handler
/// - `Generic` — backed by a Condvar signal (synchronous backends)
pub enum ComputationToken {
    /// Token backed by an MTLCommandBuffer (Metal backend).
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    Metal(ComellationToken),
    /// Generic token backed by Condvar signalling.
    Generic(ComellationToken),
}

impl std::fmt::Debug for ComputationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputationToken::Generic(t) => f.debug_tuple("Generic").field(t).finish(),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            ComputationToken::Metal(t) => f.debug_tuple("Metal").field(t).finish(),
        }
    }
}

impl Clone for ComputationToken {
    fn clone(&self) -> Self {
        match self {
            Self::Generic(t) => Self::Generic(t.clone()),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            Self::Metal(t) => Self::Metal(t.clone()),
        }
    }
}

impl PartialEq for ComputationToken {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl ComputationToken {
    /// Unique identifier for this token.
    pub fn id(&self) -> u64 {
        match self {
            ComputationToken::Generic(t) => t.id(),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            ComputationToken::Metal(t) => t.id(),
        }
    }

    /// Register a callback to fire when this token's work completes.
    ///
    /// If the work has already completed, `f` is invoked immediately.
    pub fn then(&self, f: impl FnOnce() + Send + 'static) {
        match self {
            ComputationToken::Generic(t) => t.then(f),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            ComputationToken::Metal(t) => t.then(f),
        }
    }

    /// Wait synchronously for completion.
    pub fn wait(&self) {
        match self {
            ComputationToken::Generic(t) => t.wait(),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            ComputationToken::Metal(t) => t.wait(),
        }
    }
}

/// A token tracking async GPU compute completion.
///
/// Created via [`ComellationToken::new`] (generic backends) or
/// [`ComellationToken::from_command_buffer`] (Metal backends).
///
/// The token can be used to register completion callbacks with [`then`](ComellationToken::then)
/// or to block synchronously with [`wait`](ComellationToken::wait).
pub struct ComellationToken {
    inner: std::sync::Arc<Inner>,
}

struct Inner {
    /// Unique identifier for this token.
    id: u64,
    /// Whether this token has completed on the GPU.
    completed: std::sync::atomic::AtomicBool,
    /// Callbacks registered to fire on completion.
    on_complete: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    /// Condvar pair for blocking `wait()`.
    condvar_pair: (Mutex<bool>, Condvar),
}

impl std::fmt::Debug for ComellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComellationToken")
            .field("id", &self.inner.id)
            .field(
                "completed",
                &self
                    .inner
                    .completed
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl Clone for ComellationToken {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl ComellationToken {
    /// Create a new `(ComellationToken, Completer)` pair for generic backends.
    ///
    /// The caller retains the [`Completer`] end and signals it when GPU work
    /// finishes via [`Completer::complete`].
    pub fn new() -> (Self, Completer) {
        let inner = std::sync::Arc::new(Inner {
            id: next_token_id(),
            completed: std::sync::atomic::AtomicBool::new(false),
            on_complete: Mutex::new(Vec::new()),
            condvar_pair: (Mutex::new(false), Condvar::new()),
        });
        let token = ComellationToken {
            inner: inner.clone(),
        };
        let completer = Completer {
            inner: std::sync::Arc::downgrade(&inner),
        };
        (token, completer)
    }

    /// Create a completion token from an MTLCommandBuffer.
    ///
    /// Registers a completion handler on the buffer that sets the token's
    /// completed state, fires all registered [`then`](ComellationToken::then)
    /// callbacks, and wakes any [`wait`](ComellationToken::wait) callers.
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub fn from_command_buffer(cb: &metal::CommandBufferRef) -> Self {
        let inner = std::sync::Arc::new(Inner {
            id: next_token_id(),
            completed: std::sync::atomic::AtomicBool::new(false),
            on_complete: Mutex::new(Vec::new()),
            condvar_pair: (Mutex::new(false), Condvar::new()),
        });
        let weak = std::sync::Arc::downgrade(&inner);
        // Safety: ConcreteBlock::copy creates a heap-allocated block.
        // The command buffer retains the block until it fires the handler.
        let handler = ConcreteBlock::new(move |_cmd_buf: &metal::CommandBufferRef| {
            if let Some(inner) = weak.upgrade() {
                inner
                    .completed
                    .store(true, std::sync::atomic::Ordering::Release);
                let callbacks = inner.on_complete.lock().drain(..).collect::<Vec<_>>();
                for cb in callbacks {
                    cb();
                }
                // Wake any condvar waiters.
                let (lock, cvar) = &inner.condvar_pair;
                *lock.lock() = true;
                cvar.notify_all();
            }
        });
        // Safety: copy() heap-allocates the block; the command buffer retains it.
        let handler = handler.copy();
        cb.add_completed_handler(&handler);
        ComellationToken { inner }
    }

    /// The unique identifier for this token.
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// Register a callback to fire when this token's work completes.
    ///
    /// If the work has already completed, `f` is invoked immediately.
    pub fn then(&self, f: impl FnOnce() + Send + 'static) {
        if self
            .inner
            .completed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            f();
        } else {
            self.inner.on_complete.lock().push(Box::new(f));
        }
    }

    /// Wait synchronously for completion.
    ///
    /// Blocks on a Condvar signalled by the Metal completion handler or
    /// the paired [`Completer`].
    pub fn wait(&self) {
        if self
            .inner
            .completed
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return;
        }
        let (lock, cvar) = &self.inner.condvar_pair;
        let mut done = lock.lock();
        while !*done {
            cvar.wait(&mut done);
        }
    }
}

/// The signaling end paired with [`ComellationToken::new`].
///
/// Backends call [`Completer::complete`] when GPU work finishes, which fires
/// all registered [`then`](ComellationToken::then) callbacks and wakes any
/// [`wait`](ComellationToken::wait) callers.
pub struct Completer {
    inner: std::sync::Weak<Inner>,
}

impl Completer {
    /// Signal that the compute work is done, unblocking all waiters.
    ///
    /// Fires all registered `then` callbacks and wakes any `wait` callers.
    pub fn complete(&self) {
        if let Some(inner) = self.inner.upgrade() {
            inner
                .completed
                .store(true, std::sync::atomic::Ordering::Release);
            let callbacks = inner.on_complete.lock().drain(..).collect::<Vec<_>>();
            for cb in callbacks {
                cb();
            }
            // Wake condvar waiters.
            let (lock, cvar) = &inner.condvar_pair;
            *lock.lock() = true;
            cvar.notify_all();
        }
    }
}
