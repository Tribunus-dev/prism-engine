//! Activation buffer allocation transaction guard (constitutional home).
//!
//! The state half of `activation_transaction`. The `ActivationTransaction`
//! is a guard that wraps a series of activation-arena allocations and
//! rolls back on drop if not committed.
//!
//! # Authority
//!
//! The transaction is **scheduling state** in the C bucket. The
//! underlying `ActivationArena` is an FFI-side arena (kernel concern,
//! step 44). The transaction's commit/rollback semantics are
//! state-level: the runtime reconciliation system observes whether
//! the transaction committed and stages the resulting allocation
//! through `ConstitutionalWorldTxn`.
//!
//! # Placeholder engine types
//!
//! `ActivationArena` and `ArenaBinding` are FFI-side types that move
//! to `prism-ecs-kernel::backend::metal` in step 44. The
//! constitutional home ships minimal placeholder types matching the
//! engine's wire shape.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/activation_transaction.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it.

// ---------------------------------------------------------------------------
// Placeholder engine types (FFI half; moves to kernel in step 44)
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::scheduling::activation_arena::ActivationArena`.
#[derive(Debug)]
pub struct ActivationArena {
    capacity: u64,
    allocated: u64,
}

impl ActivationArena {
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            allocated: 0,
        }
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated
    }

    pub fn set_allocated_bytes(&mut self, value: u64) {
        self.allocated = value.min(self.capacity);
    }

    /// Allocate `byte_size` bytes with the given alignment. The engine's
    /// `ActivationArena` is a bump allocator; the placeholder mirrors
    /// that behavior.
    pub fn allocate(&mut self, byte_size: u64, alignment: u64) -> Result<ArenaBinding, String> {
        // Naive alignment bump: round up `allocated` to the nearest multiple of `alignment`.
        let aligned = if alignment == 0 {
            self.allocated
        } else {
            ((self.allocated + alignment - 1) / alignment) * alignment
        };
        if aligned + byte_size > self.capacity {
            return Err(format!(
                "arena exhausted: requested {} bytes, capacity {}",
                byte_size,
                self.capacity - aligned
            ));
        }
        self.allocated = aligned + byte_size;
        Ok(ArenaBinding {
            offset: aligned,
            size: byte_size,
        })
    }
}

/// Placeholder for `compute-core::ecs::scheduling::activation_binding::ArenaBinding`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaBinding {
    pub offset: u64,
    pub size: u64,
}

// ---------------------------------------------------------------------------
// ActivationTransaction
// ---------------------------------------------------------------------------

/// Transaction guard for activation buffer allocations.
///
/// On drop, resets the arena to its pre-transaction state if not committed.
///
/// # Rollback semantics
///
/// The activation arena is a simple bump allocator. When the transaction
/// is rolled back, the arena's commit watermark is reset to the position
/// before the first allocation in this transaction, effectively reclaiming
/// all reserved bytes. This is safe because the arena is single-use per
/// scheduling epoch (reset at epoch boundaries).
#[derive(Debug)]
pub struct ActivationTransaction<'a> {
    arena: &'a mut ActivationArena,
    /// Saved allocated_bytes from before any allocation in this transaction.
    saved_watermark: u64,
    committed: bool,
}

impl<'a> ActivationTransaction<'a> {
    /// Begin a new activation transaction for the given arena.
    ///
    /// Snapshots the arena's current allocated_bytes so rollback can restore it.
    pub fn new(arena: &'a mut ActivationArena) -> Self {
        let saved_watermark = arena.allocated_bytes();
        Self {
            arena,
            saved_watermark,
            committed: false,
        }
    }

    /// Allocate a slot in the activation arena within this transaction.
    ///
    /// On failure, returns the error without rolling back prior successful
    /// allocations — the caller may retry or let the Drop impl roll back
    /// the entire transaction.
    pub fn allocate(&mut self, byte_size: u64, alignment: u64) -> Result<ArenaBinding, String> {
        self.arena.allocate(byte_size, alignment)
    }

    /// Commit the transaction — marks allocations as permanent.
    ///
    /// Consumes `self` so the Drop impl skips rollback.
    pub fn commit(mut self) -> Result<(), String> {
        self.committed = true;
        Ok(())
    }

    /// Returns true if the transaction has been committed.
    pub fn is_committed(&self) -> bool {
        self.committed
    }
}

impl<'a> Drop for ActivationTransaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            // Restore the arena to its pre-transaction watermark, effectively
            // releasing all allocations made within this transaction.
            self.arena.set_allocated_bytes(self.saved_watermark);
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `activation_transaction` state.
    //!
    //! Test names describe the constitutional rule, not the function.

    use super::*;

    fn make_arena() -> ActivationArena {
        ActivationArena::new(65536)
    }

    #[test]
    fn transaction_rolls_back_on_drop() {
        // Architectural invariant: a transaction that is dropped
        // without commit() reverts the arena to its pre-transaction
        // state. No allocations leak past the transaction.
        let mut arena = make_arena();
        assert_eq!(arena.allocated_bytes(), 0);
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(4096, 64).expect("allocate should succeed");
            tx.allocate(2048, 64).expect("allocate should succeed");
        }
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn commit_keeps_allocations() {
        // Architectural invariant: a transaction that commits
        // retains its allocations. The arena's allocated_bytes
        // reflects the sum of the transaction's allocations.
        let mut arena = make_arena();
        assert_eq!(arena.allocated_bytes(), 0);
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(4096, 64).expect("allocate should succeed");
            tx.allocate(2048, 64).expect("allocate should succeed");
            tx.commit().expect("commit should succeed");
        }
        assert_eq!(arena.allocated_bytes(), 6144);
    }

    #[test]
    fn commit_consumes_self() {
        // Architectural invariant: commit() takes `self` by value,
        // so the transaction cannot be used after commit. The
        // compiler enforces this; a runtime check is unnecessary.
        let mut arena = make_arena();
        let tx = ActivationTransaction::new(&mut arena);
        let _ = tx.commit();
        // No further use of `tx` is possible — the type system
        // prevents calling `tx.allocate` or `tx.commit` again.
    }

    #[test]
    fn empty_commit_is_a_no_op() {
        // Architectural invariant: a transaction with no allocations
        // that commits is a no-op for the arena.
        let mut arena = make_arena();
        {
            let tx = ActivationTransaction::new(&mut arena);
            tx.commit().expect("empty commit should succeed");
        }
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn rollback_after_partial_failure() {
        // Architectural invariant: when an allocation fails mid-
        // transaction, the Drop impl rolls back EVERY allocation
        // (including successful ones before the failure), because
        // the transaction is all-or-nothing.
        let mut arena = ActivationArena::new(100);
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(60, 1).expect("first allocation succeeds");
            let result = tx.allocate(60, 1);
            assert!(result.is_err());
            // tx drops — rolls back the 60 bytes too.
        }
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn multiple_transactions_isolate() {
        // Architectural invariant: two transactions on the same
        // arena do not interfere. A rollback of the second
        // transaction does not affect the first's allocations.
        let mut arena = make_arena();
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(1024, 64).expect("first allocation succeeds");
            tx.commit().expect("first commit succeeds");
        }
        let after_first = arena.allocated_bytes();
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(8192, 64).expect("second allocation succeeds");
            // drop without commit
        }
        assert_eq!(arena.allocated_bytes(), after_first);
    }

    #[test]
    fn arena_allocate_alignment_respected() {
        // Architectural invariant: allocations are aligned to the
        // requested alignment. A request for 7 bytes with alignment
        // 16 starts at the next 16-byte boundary.
        let mut arena = ActivationArena::new(1024);
        let b1 = arena.allocate(7, 16).expect("first allocation");
        assert_eq!(b1.offset, 0);
        let b2 = arena.allocate(7, 16).expect("second allocation");
        assert_eq!(b2.offset, 16);
    }

    #[test]
    fn arena_exhaustion_returns_error() {
        // Architectural invariant: when the arena cannot satisfy an
        // allocation, the request fails with an error and the
        // arena's allocated_bytes is unchanged.
        let mut arena = ActivationArena::new(10);
        let _ = arena.allocate(8, 1).expect("first allocation");
        let result = arena.allocate(8, 1);
        assert!(result.is_err());
        assert_eq!(arena.allocated_bytes(), 8);
    }
}
