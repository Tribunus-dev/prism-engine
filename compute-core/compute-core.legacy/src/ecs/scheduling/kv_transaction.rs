use crate::ecs::compute_image::phase_graph::PhaseId;
use serde::{Deserialize, Serialize};

/// A cache generation identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvGeneration(pub u64);

/// Publication state of a KV transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvPublicationState {
    Tentative,
    Committed,
    RolledBack,
}

/// Record of a single KV write operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWriteRecord {
    pub layer_index: usize,
    pub token_position: usize,
    pub num_new_tokens: usize,
    pub bytes_written: u64,
}

/// A tentative KV cache mutation that can be committed or rolled back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvMutationTransaction {
    pub phase_id: PhaseId,
    pub layer_index: usize,
    pub prior_generation: KvGeneration,
    pub tentative_generation: KvGeneration,
    pub writes: Vec<KvWriteRecord>,
    pub publication_state: KvPublicationState,
}

impl KvMutationTransaction {
    pub fn new(phase_id: PhaseId, layer_index: usize, prior_generation: KvGeneration) -> Self {
        Self {
            phase_id,
            layer_index,
            prior_generation,
            tentative_generation: KvGeneration(prior_generation.0 + 1),
            writes: Vec::new(),
            publication_state: KvPublicationState::Tentative,
        }
    }

    /// Commit the transaction — marks writes as visible.
    pub fn commit(&mut self) -> Result<(), String> {
        match self.publication_state {
            KvPublicationState::Tentative => {
                self.publication_state = KvPublicationState::Committed;
                Ok(())
            }
            KvPublicationState::Committed => Err("transaction already committed".to_string()),
            KvPublicationState::RolledBack => {
                Err("cannot commit a rolled-back transaction".to_string())
            }
        }
    }

    /// Roll back the transaction — discards tentative writes.
    pub fn rollback(&mut self) -> Result<(), String> {
        match self.publication_state {
            KvPublicationState::Tentative => {
                self.publication_state = KvPublicationState::RolledBack;
                self.writes.clear();
                Ok(())
            }
            KvPublicationState::Committed => {
                Err("cannot roll back a committed transaction".to_string())
            }
            KvPublicationState::RolledBack => Err("transaction already rolled back".to_string()),
        }
    }

    pub fn is_committed(&self) -> bool {
        self.publication_state == KvPublicationState::Committed
    }

    pub fn is_tentative(&self) -> bool {
        self.publication_state == KvPublicationState::Tentative
    }
}

/// Transaction receipt stored in the phase receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvTransactionReceipt {
    pub layer_index: usize,
    pub prior_generation: u64,
    pub new_generation: u64,
    pub num_writes: usize,
    pub committed: bool,
    pub bytes_written: u64,
}

impl From<&KvMutationTransaction> for KvTransactionReceipt {
    fn from(tx: &KvMutationTransaction) -> Self {
        let total_bytes: u64 = tx.writes.iter().map(|w| w.bytes_written).sum();
        KvTransactionReceipt {
            layer_index: tx.layer_index,
            prior_generation: tx.prior_generation.0,
            new_generation: tx.tentative_generation.0,
            num_writes: tx.writes.len(),
            committed: tx.is_committed(),
            bytes_written: total_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let phase_id = PhaseId("layer_0_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id.clone(), 0, KvGeneration(1));
        assert!(tx.is_tentative());
        tx.writes.push(KvWriteRecord {
            layer_index: 0,
            token_position: 0,
            num_new_tokens: 1,
            bytes_written: 4096,
        });
        assert!(tx.commit().is_ok());
        assert!(tx.is_committed());
        // Double commit fails
        assert!(tx.commit().is_err());
    }

    #[test]
    fn test_rollback() {
        let phase_id = PhaseId("layer_0_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id, 0, KvGeneration(5));
        assert!(tx.rollback().is_ok());
        assert!(!tx.is_committed());
        assert!(tx.commit().is_err());
    }

    #[test]
    fn test_receipt_conversion() {
        let phase_id = PhaseId("layer_1_attn".to_string());
        let mut tx = KvMutationTransaction::new(phase_id, 1, KvGeneration(3));
        tx.writes.push(KvWriteRecord {
            layer_index: 1,
            token_position: 10,
            num_new_tokens: 1,
            bytes_written: 2048,
        });
        tx.commit().unwrap();
        let receipt: KvTransactionReceipt = (&tx).into();
        assert_eq!(receipt.layer_index, 1);
        assert_eq!(receipt.new_generation, 4);
        assert!(receipt.committed);
        assert_eq!(receipt.bytes_written, 2048);
    }
}

// ── KvBlockTransaction ───────────────────────────────────────────────────
// Block-level allocation transaction guard for KV cache.

/// Transaction guard for KV block allocations.
///
/// On drop, rolls back (frees all allocated blocks) if not committed.
/// Use the builder-style API:
///
/// ```ignore
/// let tx = KvBlockTransaction::new_for("req-1", &mut coord)
///     .allocate(5, CacheGroupType::FullAttention)?
///     .commit()?;
/// ```
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub struct KvBlockTransaction {
    request_id: String,
    coord: *mut crate::ecs::kv_cache::layered_cache::KVCacheCoordinator,
    committed: bool,
}

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
impl KvBlockTransaction {
    /// Create a new transaction for the given request.
    ///
    /// No blocks are allocated yet; call [`allocate`](Self::allocate) to
    /// reserve blocks within the transaction scope.
    pub fn new_for(
        request_id: &str,
        coord: &mut crate::ecs::kv_cache::layered_cache::KVCacheCoordinator,
    ) -> Self {
        Self {
            request_id: request_id.to_string(),
            coord: coord as *mut crate::ecs::kv_cache::layered_cache::KVCacheCoordinator,
            committed: false,
        }
    }

    /// Allocate `num_tokens` worth of KV cache blocks for the request.
    ///
    /// Uses the raw-pointer coordinator captured at construction time.
    /// Returns `self` for chaining.
    pub fn allocate(
        &mut self,
        num_tokens: usize,
        group_type: crate::ecs::kv_cache::layered_cache::CacheGroupType,
    ) -> Result<&mut Self, String> {
        let blocks =
            unsafe { (*self.coord).allocate_slots(&self.request_id, num_tokens, group_type)? };
        // Blocks are registered with the coordinator — on drop, free_request
        // releases them all by request_id.
        let _ = blocks;
        Ok(self)
    }

    /// Commit the transaction — marks the allocation as permanent.
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

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
impl Drop for KvBlockTransaction {
    fn drop(&mut self) {
        if !self.committed {
            unsafe {
                (*self.coord).free_request(&self.request_id);
            }
        }
    }
}

// ── KvBlockTransaction tests ─────────────────────────────────────────────

#[cfg(all(test, any(feature = "mlx-backend", feature = "prism-backend")))]
mod kv_block_transaction_tests {
    use super::*;
    use crate::ecs::kv_cache::layered_cache::{CacheGroupType, KVCacheCoordinator};

    #[test]
    fn test_kv_block_transaction_rolls_back_on_drop() {
        let mut coord = KVCacheCoordinator::new(10);
        let initial_free = coord.pool.free_queue.len();

        {
            // Scope: allocate without commit → transaction drops → rollback
            let mut tx = KvBlockTransaction::new_for("req-drop", &mut coord);
            tx.allocate(3, CacheGroupType::FullAttention)
                .expect("allocate should succeed");
        }

        // All 3 blocks should be returned to the pool
        assert_eq!(coord.pool.free_queue.len(), initial_free);
        // req should not be tracked in any group
        let fa = coord.groups.get(&CacheGroupType::FullAttention);
        assert!(fa.is_none() || !fa.unwrap().req_to_blocks.contains_key("req-drop"));
    }

    #[test]
    fn test_kv_block_transaction_commit_keeps_blocks() {
        let mut coord = KVCacheCoordinator::new(10);
        let initial_free = coord.pool.free_queue.len();

        {
            // Scope: allocate and commit → blocks survive the drop
            let mut tx = KvBlockTransaction::new_for("req-keep", &mut coord);
            tx.allocate(3, CacheGroupType::FullAttention)
                .expect("allocate should succeed");
            tx.commit().expect("commit should succeed");
        }

        // 3 blocks should be removed from the pool
        assert_eq!(coord.pool.free_queue.len(), initial_free - 3);
        // req should be tracked
        let fa = coord.groups.get(&CacheGroupType::FullAttention).unwrap();
        assert!(fa.req_to_blocks.contains_key("req-keep"));
        assert_eq!(fa.req_to_blocks["req-keep"].len(), 3);
    }

    #[test]
    fn test_kv_block_transaction_handles_allocate_failure() {
        let mut coord = KVCacheCoordinator::new(2);
        // Fill the pool
        let _slots = coord
            .allocate_slots("filler", 2, CacheGroupType::FullAttention)
            .expect("fill should succeed");

        // Attempt an allocation that will fail — no panic on drop
        let mut tx = KvBlockTransaction::new_for("req-fail", &mut coord);
        let result = tx.allocate(3, CacheGroupType::FullAttention);
        assert!(result.is_err(), "should fail on exhausted pool");
        // tx drops — no blocks were allocated for req-fail, so no-op cleanup

        // Filler blocks still intact
        let fa = coord.groups.get(&CacheGroupType::FullAttention).unwrap();
        assert!(fa.req_to_blocks.contains_key("filler"));
        assert_eq!(coord.pool.free_queue.len(), 0);
    }
}
