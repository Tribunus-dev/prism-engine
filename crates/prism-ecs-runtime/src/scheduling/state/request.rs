//! Request state (constitutional home).
//!
//! Per-request scheduling state. A `Request` is the canonical record
//! of an inference request: its prompt, its priority, its current
//! lifecycle state, and the slot (if any) it is leased to.
//!
//! # Authority
//!
//! The `Request` type is **scheduling state** in the C bucket. The
//! request's lifecycle state transitions are staged through
//! `ConstitutionalWorldTxn`; the runtime scheduling systems are the
//! only producers of those transitions.
//!
//! The `SavedRequest` type is a **preemption record**: when a
//! higher-priority request preempts a running one, the preempted
//! request's KV-cache state is captured into a `SavedRequest`. The
//! resume path restores the request from the saved record.
//!
//! # Placeholder engine types
//!
//! `RequestState` matches the engine's enum (re-exported from
//! `state::batch`). `CompressedKvSlot` is a placeholder for the
//! engine's KV cache type (moves with the kv_cache crate).
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/mod.rs` (the
//! `Request` type definition), `request.rs` (the impl block), and
//! `saved_request.rs` (the saved-request type). The engine files are
//! the legacy duplicates; step 58 deletes them.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// RequestState (canonical home)
// ---------------------------------------------------------------------------

/// Request lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RequestState {
    Queued,
    Prefilling,
    Decoding,
    Paused,
    Completed,
    Cancelled,
}

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::kv_cache::CompressedKvSlot`.
/// Replaced when the kv_cache crate migrates.
#[derive(Debug, Clone, Default)]
pub struct CompressedKvSlot {
    /// KV cache offset in tokens.
    pub kv_offset: usize,
    /// Number of tokens in this slot.
    pub num_tokens: usize,
    /// Compressed KV data (opaque bytes).
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Priority constants
// ---------------------------------------------------------------------------

/// Highest priority value — requests at this level are never preempted.
pub const PRIORITY_HIGHEST: u8 = 0;

/// Default priority for requests that don't specify otherwise.
pub const PRIORITY_DEFAULT: u8 = 128;

/// Lowest priority value — requests at this level are preempted first.
pub const PRIORITY_LOWEST: u8 = 255;

/// Maximum number of times a request can be preempted before it gets
/// a starvation boost that effectively exempts it from further preemption.
pub const MAX_PREEMPTIONS_BEFORE_BOOST: usize = 3;

/// Priority boost applied to a starved request on each preemption cycle.
/// The boost reduces the priority value (making it higher priority),
/// protecting it from being preempted again.
pub const STARVATION_PRIORITY_BOOST: u8 = 64;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A single inference request.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub priority: u8,
    pub state: RequestState,
    pub created_at: Instant,
    pub slot: Option<usize>,
}

impl Request {
    /// Create a new request for the given prompt.
    pub fn new(prompt: Vec<u32>, max_tokens: usize) -> Self {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            id,
            prompt,
            max_tokens,
            priority: PRIORITY_DEFAULT,
            state: RequestState::Queued,
            created_at: Instant::now(),
            slot: None,
        }
    }

    /// Transition the request to a new state.
    pub fn transition(&mut self, state: RequestState) {
        self.state = state;
    }
}

// ---------------------------------------------------------------------------
// SavedRequest
// ---------------------------------------------------------------------------

/// A request whose KV cache has been saved to allow preemption.
///
/// The `kv_cache_snapshot` holds the compressed KV page data that was
/// in GPU-accessible memory. On resume, these slots are re-assigned to
/// the request's new slot and the pages are re-attached.
#[derive(Debug, Clone)]
pub struct SavedRequest {
    /// Original request ID.
    pub request_id: u64,
    /// Compressed KV pages — each entry corresponds to one page's worth
    /// of tokens, identified by `kv_offset` and `num_tokens`.
    pub kv_cache_snapshot: Vec<CompressedKvSlot>,
    /// The original prompt tokens (needed to reconstruct the request
    /// on resume when the KV cache fully covers the prompt).
    pub prompt: Vec<u32>,
    /// Maximum tokens for the original request.
    pub max_tokens: usize,
    /// Tokens already generated before preemption.
    pub tokens_generated: usize,
    /// KV cache length at preemption time.
    pub kv_cache_length: usize,
    /// KV cache start position.
    pub kv_cache_start: usize,
    /// Priority at preemption time (may have been boosted by anti-starvation).
    pub priority: u8,
    /// Page IDs from the paged allocator — used to re-attach pages on resume.
    pub kv_cache_pages: Vec<usize>,
    /// Number of times this request has been preempted.
    pub preemption_count: usize,
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `request` state.

    use super::*;

    #[test]
    fn new_request_is_queued_with_default_priority() {
        // Architectural invariant: a fresh request is in the Queued
        // state with the default priority (128), no slot assignment,
        // and a unique id derived from the system clock.
        let r = Request::new(vec![1, 2, 3], 100);
        assert_eq!(r.prompt, vec![1, 2, 3]);
        assert_eq!(r.max_tokens, 100);
        assert_eq!(r.priority, PRIORITY_DEFAULT);
        assert_eq!(r.state, RequestState::Queued);
        assert!(r.slot.is_none());
        assert!(r.id > 0);
    }

    #[test]
    fn transition_updates_state() {
        // Architectural invariant: transition is a direct assignment
        // to the state field. The runtime may transition through any
        // state sequence; the only requirement is that the new state
        // is one of the RequestState variants.
        let mut r = Request::new(vec![], 10);
        r.transition(RequestState::Prefilling);
        assert_eq!(r.state, RequestState::Prefilling);
        r.transition(RequestState::Decoding);
        assert_eq!(r.state, RequestState::Decoding);
        r.transition(RequestState::Completed);
        assert_eq!(r.state, RequestState::Completed);
    }

    #[test]
    fn priority_constants_partition() {
        // Architectural invariant: PRIORITY_HIGHEST (0) < PRIORITY_DEFAULT
        // (128) < PRIORITY_LOWEST (255). The priority order matches
        // numerical order: lower value = higher priority.
        assert!(PRIORITY_HIGHEST < PRIORITY_DEFAULT);
        assert!(PRIORITY_DEFAULT < PRIORITY_LOWEST);
    }

    #[test]
    fn starvation_boost_reduces_priority() {
        // Architectural invariant: applying the starvation boost to a
        // priority reduces the value (making the request higher
        // priority). The boost is subtracted, not added.
        let original = PRIORITY_DEFAULT;
        let boosted = original.saturating_sub(STARVATION_PRIORITY_BOOST);
        assert!(boosted < original);
    }

    #[test]
    fn saved_request_carries_all_required_fields() {
        // Architectural invariant: a SavedRequest is a complete
        // preemption record — it carries enough state to resume
        // the request without consulting any other source.
        let saved = SavedRequest {
            request_id: 42,
            kv_cache_snapshot: vec![],
            prompt: vec![1, 2, 3],
            max_tokens: 100,
            tokens_generated: 10,
            kv_cache_length: 10,
            kv_cache_start: 0,
            priority: PRIORITY_DEFAULT,
            kv_cache_pages: vec![1, 2, 3],
            preemption_count: 0,
        };
        assert_eq!(saved.request_id, 42);
        assert_eq!(saved.tokens_generated, 10);
        assert_eq!(saved.kv_cache_pages.len(), 3);
    }
}
