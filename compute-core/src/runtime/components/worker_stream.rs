//! WorkerStream — token and byte accounting for streaming worker output.
//!
//! Tracks output progress via a capped token tail (used for duplication
//! detection) and cumulative byte/sequence counters.

use std::time::Instant;

use crate::runtime::scheduling::component_id::SchedulableComponent;
use crate::runtime::components::{
    WORKER_HARDWARE_STREAM_COMPONENT,
    WORKER_STREAM_COMPONENT,
};

/// Rolling window of recently produced token IDs for duplicate detection.
///
/// Capacity is fixed at construction.  Once full, new pushes overwrite the
/// oldest entry.
#[derive(Debug, Clone)]
pub struct TokenTail {
    capacity: usize,
    tokens: Vec<u32>,
    cursor: usize,
}

impl TokenTail {
    /// Create a new tail with the given `capacity`.
    ///
    /// A capacity of zero disables token tracking entirely.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tokens: Vec::with_capacity(capacity),
            cursor: 0,
        }
    }

    /// Push a token into the tail window.
    pub fn push(&mut self, token: u32) {
        if self.capacity == 0 {
            return;
        }
        if self.tokens.len() < self.capacity {
            self.tokens.push(token);
        } else {
            self.tokens[self.cursor] = token;
        }
        self.cursor = (self.cursor + 1) % self.capacity;
    }

    /// Return a slice of the currently tracked tokens in insertion order.
    pub fn tail(&self) -> &[u32] {
        &self.tokens
    }
}

/// Stream progress for a worker request.
#[derive(Debug, Clone)]
pub struct WorkerStream {
    /// Monotonic output event sequence number.
    pub sequence: u64,
    /// Total output bytes produced so far.
    pub output_bytes: u64,
    /// Instant the first output token was received.
    pub first_token_at: Option<Instant>,
    /// Instant of the most recent output.
    pub last_output_at: Option<Instant>,
    /// Rolling token tail for deduplication.
    pub token_tail: TokenTail,
}

impl WorkerStream {
    /// Create a new stream tracker with default capacity (16 tokens).
    pub fn new() -> Self {
        Self {
            sequence: 0,
            output_bytes: 0,
            first_token_at: None,
            last_output_at: None,
            token_tail: TokenTail::new(16),
        }
    }

    /// Record an output event, optionally pushing `token` into the tail and
    /// accumulating `bytes` into the total byte count.
    pub fn record_output(&mut self, token: Option<u32>, bytes: u64) {
        let now = Instant::now();
        self.sequence += 1;
        self.output_bytes += bytes;
        self.last_output_at = Some(now);
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
        }
        if let Some(t) = token {
            self.token_tail.push(t);
        }
    }
}

impl Default for WorkerStream {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulableComponent for WorkerStream {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_STREAM_COMPONENT;
    const NAME: &'static str = "WorkerStream";
}
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};


/// Zero-cost abstraction over a hardware-mapped memory pointer.
///
/// Points to an IOSurface-backed MTLBuffer that the GPU atomically
/// increments.  The read compiles to a single CPU load-acquire instruction.
pub struct HardwareAtomicPtr {
    ptr: NonNull<AtomicU32>,
}

// SAFETY: The GPU writes to this memory with release ordering;
// our read uses acquire ordering.  No data race.
unsafe impl Send for HardwareAtomicPtr {}
unsafe impl Sync for HardwareAtomicPtr {}

impl HardwareAtomicPtr {
    /// Create from a raw pointer returned by the Metal FFI.
    ///
    /// # Safety
    /// `raw_ptr` must point to valid, aligned, page-pinned memory
    /// that the GPU will atomically increment.
    pub unsafe fn new(raw_ptr: *mut u32) -> Self {
        Self {
            ptr: NonNull::new(raw_ptr as *mut AtomicU32)
                .expect("null pointer from Metal FFI"),
        }
    }

    /// Single CPU instruction read with acquire semantics.
    #[inline(always)]
    pub fn poll(&self) -> u32 {
        unsafe { self.ptr.as_ref().load(Ordering::Acquire) }
    }
}

/// Attached to an entity during the Streaming phase.
///
/// The Metal shader writes to the shared memory; the ECS observer
/// polls these atomics at the stage barrier.  No locks, no syscalls.
pub struct HardwareStreamHandle {
    pub tokens_generated: HardwareAtomicPtr,
    pub stream_closed: HardwareAtomicPtr,
    pub last_observed_count: u32,
}

impl HardwareStreamHandle {
    /// # Safety
    /// Both pointers must be valid, IOSurface-backed, GPU-writable memory.
    pub unsafe fn new(tok_ptr: *mut u32, closed_ptr: *mut u32) -> Self {
        Self {
            tokens_generated: HardwareAtomicPtr::new(tok_ptr),
            stream_closed: HardwareAtomicPtr::new(closed_ptr),
            last_observed_count: 0,
        }
    }

    /// Current token count from GPU shared memory.
    pub fn current_count(&self) -> u32 {
        self.tokens_generated.poll()
    }

    /// Whether the GPU has signaled end-of-stream.
    pub fn is_closed(&self) -> bool {
        self.stream_closed.poll() > 0
    }
}

impl SchedulableComponent for HardwareStreamHandle {
    const COMPONENT_ID: crate::runtime::scheduling::component_id::ComponentId =
        WORKER_HARDWARE_STREAM_COMPONENT;
    const NAME: &'static str = "HardwareStreamHandle";
}
