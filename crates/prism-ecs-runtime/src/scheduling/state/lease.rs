//! Lease state (constitutional home).
//!
//! This is the constitutional home for the per-slot lease state: a
//! `Slot` is a leased execution unit (one model execution slot, owned
//! by the scheduler and leased to a request for the duration of
//! prefill + decode).
//!
//! # Authority
//!
//! Every type in this module is **scheduling state** in the C bucket.
//! A slot is allocated by the lane-lease-allocation system, assigned
//! to a request by the batching system, and released by the
//! completion-reconciliation system. All transitions are staged
//! through `ConstitutionalWorldTxn`.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/mod.rs` (the
//! `Slot` type definition) and `compute-core/src/ecs/scheduling/slot.rs`
//! (the impl block). The engine files are the legacy duplicate; step
//! 58 deletes them when no engine caller remains. No compatibility
//! facade.

/// A slot in the batch (one model execution unit).
///
/// A `Slot` is a scheduling-state record. The dispatch-selection system
/// allocates slots from the lane-lease state; the batching system
/// populates a slot with the request's prompt length and KV-cache
/// pages. Once a batch commits, every slot in it is in-flight; the
/// completion-reconciliation system releases slots on completion.
#[derive(Debug, Clone)]
pub struct Slot {
    pub id: usize,
    pub request_id: Option<u64>,
    pub tokens_generated: usize,
    pub kv_cache_start: usize,
    pub kv_cache_length: usize,
    /// Target execution backend for this slot.
    /// 0=MLX, 1=Accelerate, 2=CoreML, 3=ANE/Orion
    pub backend_id: u32,
    /// Page IDs allocated from the paged allocator for this slot's KV cache.
    pub kv_cache_pages: Vec<usize>,
}

impl Slot {
    /// Create a new empty slot with the given id and default backend.
    pub fn new(id: usize) -> Self {
        Slot {
            id,
            request_id: None,
            tokens_generated: 0,
            kv_cache_start: 0,
            kv_cache_length: 0,
            backend_id: 0,
            kv_cache_pages: vec![],
        }
    }

    /// Returns true if the slot is not assigned to any request.
    pub fn is_free(&self) -> bool {
        self.request_id.is_none()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `lease` state.
    //!
    //! Test names describe the constitutional rule, not the function.

    use super::*;

    #[test]
    fn new_slot_is_free() {
        // Architectural invariant: a freshly constructed slot has
        // no request assignment and zero counters. A reader can
        // rely on `is_free() == true` immediately after construction.
        let slot = Slot::new(7);
        assert!(slot.is_free());
        assert_eq!(slot.id, 7);
        assert_eq!(slot.request_id, None);
        assert_eq!(slot.tokens_generated, 0);
        assert_eq!(slot.kv_cache_start, 0);
        assert_eq!(slot.kv_cache_length, 0);
        assert_eq!(slot.backend_id, 0);
        assert!(slot.kv_cache_pages.is_empty());
    }

    #[test]
    fn slot_with_request_id_is_not_free() {
        // Architectural invariant: a slot is "leased" iff it has a
        // request_id. The free/leased distinction is the only state
        // machine on the slot itself.
        let mut slot = Slot::new(0);
        slot.request_id = Some(42);
        assert!(!slot.is_free());
    }
}
