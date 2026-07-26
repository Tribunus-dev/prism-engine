//! Activation buffer allocation transaction guard.
//!
//! [`ActivationTransaction`] wraps a series of activation arena allocations
//! and rolls back (releases) them on drop if not committed.  Use the
//! builder-style API:
//!
//! ```ignore
//! let mut tx = ActivationTransaction::new(&mut arena);
//! tx.allocate(4096, 64)?;
//! let binding = tx.allocate(2048, 64)?;
//! tx.commit()?;
//! ```
//!
//! This module is gated behind `mlx-backend` because it depends on
//! `activation_arena` and `activation_binding`.
#![cfg(feature = "mlx-backend")]

use crate::ecs::scheduling::activation_arena::ActivationArena;
use crate::ecs::scheduling::activation_binding::ArenaBinding;

/// Transaction guard for activation buffer allocations.
///
/// On drop, resets the arena to its pre-transaction state if not committed.
///
/// # Rollback semantics
///
/// The activation arena is a simple bump allocator.  When the transaction
/// is rolled back, the arena's commit watermark is reset to the position
/// before the first allocation in this transaction, effectively reclaiming
/// all reserved bytes.  This is safe because the arena is single-use per
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

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arena() -> ActivationArena {
        ActivationArena::new(65536)
    }

    #[test]
    fn test_activation_transaction_rolls_back_on_drop() {
        let mut arena = make_arena();
        assert_eq!(arena.allocated_bytes(), 0);

        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(4096, 64).expect("allocate should succeed");
            tx.allocate(2048, 64).expect("allocate should succeed");
            // tx drops without commit
        }

        // Arena should be reset to pre-transaction state
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn test_activation_transaction_commit_keeps_allocation() {
        let mut arena = make_arena();
        assert_eq!(arena.allocated_bytes(), 0);

        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(4096, 64).expect("allocate should succeed");
            tx.allocate(2048, 64).expect("allocate should succeed");
            tx.commit().expect("commit should succeed");
        }

        // Arena should retain the allocations
        assert_eq!(arena.allocated_bytes(), 6144); // 4096 + 2048
    }

    #[test]
    fn test_activation_transaction_multiple_scopes() {
        let mut arena = make_arena();

        // First transaction — committed
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(1024, 64).expect("allocate should succeed");
            tx.commit().expect("commit should succeed");
        }
        let after_first = arena.allocated_bytes();

        // Second transaction — rolled back
        {
            let mut tx = ActivationTransaction::new(&mut arena);
            tx.allocate(8192, 64).expect("allocate should succeed");
            // drop without commit
        }

        // Arena should be back to after-first level
        assert_eq!(arena.allocated_bytes(), after_first);
    }

    #[test]
    fn test_activation_transaction_empty_commit() {
        let mut arena = make_arena();
        {
            let tx = ActivationTransaction::new(&mut arena);
            tx.commit().expect("empty commit should succeed");
        }
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn test_activation_transaction_allocation_failure_does_not_corrupt() {
        let mut arena = ActivationArena::new(100);
        let mut tx = ActivationTransaction::new(&mut arena);
        // First allocation fits
        tx.allocate(60, 1).expect("allocate should succeed");
        // Second allocation doesn't fit
        let result = tx.allocate(60, 1);
        assert!(result.is_err(), "should fail on exhaustion");
        // tx drops — should roll back everything including the 60 bytes
        assert_eq!(arena.allocated_bytes(), 0);
    }
}
