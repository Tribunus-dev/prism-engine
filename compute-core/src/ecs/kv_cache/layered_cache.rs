//! Layered, prefix-caching KV cache architecture (vLLM-inspired).
//!
//! Three-layer decomposition:
//!
//! 1. **BlockPool** — flat array of blocks with free-list LRU management and
//!    content-hash-based prefix cache lookup.
//! 2. **CacheGroupManager** — per-attention-type manager (FullAttention,
//!    SlidingWindow, MLA, Mamba, CrossAttention) owning per-request block
//!    tables and copy-on-write tracking for partial prefix hits.
//! 3. **KVCacheCoordinator** — top-level coordinator owning all group managers
//!    and the shared block pool.
//!
//! Blocks are content-addressed via a u64 hash (low 64 bits of SHA-256 of
//! token content) for prefix sharing across sequences.

use std::collections::HashMap;

use crate::ecs::scheduling::kv_transaction::KvBlockTransaction;
use crate::ecs::scheduling::SchedulerState;

// ---------------------------------------------------------------------------
// BlockPool
// ---------------------------------------------------------------------------

/// A single block entry in the block pool.
#[derive(Debug, Clone)]
pub struct KvBlock {
    /// Block ID (index in block pool).
    pub block_id: u64,
    /// Reference count (shared by multiple sequences).
    pub ref_cnt: usize,
    /// Content hash for prefix cache (low 64 bits of SHA-256 of token content).
    pub content_hash: Option<u64>,
    /// Whether this block is evictable.
    pub is_evictable: bool,
    /// Whether this block is free.
    pub is_free: bool,
}

/// Flat storage for KV cache blocks with free-list LRU management and
/// hash-based prefix cache lookup.
#[derive(Debug, Clone)]
pub struct BlockPool {
    /// Flat array of all blocks.
    pub blocks: Vec<KvBlock>,
    /// Free block indices in LRU order.
    pub free_queue: Vec<u64>,
    /// Content hash -> block ID lookup.
    pub hash_index: HashMap<u64, Vec<u64>>,
    /// Total block capacity.
    pub capacity: u64,
}

impl BlockPool {
    /// Create a new pool with the given total capacity.
    ///
    /// All blocks start free and evictable, with zero references and no
    /// content hash assigned.
    pub fn new(capacity: u64) -> Self {
        let mut free_queue = Vec::with_capacity(capacity as usize);
        let mut blocks = Vec::with_capacity(capacity as usize);
        for i in 0..capacity {
            blocks.push(KvBlock {
                block_id: i,
                ref_cnt: 0,
                content_hash: None,
                is_evictable: true,
                is_free: true,
            });
            free_queue.push(i);
        }
        BlockPool {
            blocks,
            free_queue,
            hash_index: HashMap::new(),
            capacity,
        }
    }

    /// Allocate one free block from the free queue (LRU order).
    ///
    /// Returns `None` when no blocks are available.
    pub fn allocate_block(&mut self) -> Option<u64> {
        let block_id = self.free_queue.pop()?;
        if let Some(block) = self.blocks.get_mut(block_id as usize) {
            block.is_free = false;
            block.ref_cnt = 1;
        }
        Some(block_id)
    }

    /// Free a block, returning it to the free queue.
    ///
    /// Marks the block as free and appends its ID to the end of the free
    /// queue (most recently freed position).
    pub fn free_block(&mut self, block_id: u64) {
        if let Some(block) = self.blocks.get_mut(block_id as usize) {
            block.is_free = true;
            block.ref_cnt = 0;
            block.content_hash = None;
        }
        self.free_queue.push(block_id);
    }

    /// Touch a block: increment its reference count and remove it from the
    /// free queue if it happens to be there (transient state during eviction).
    pub fn touch(&mut self, block_id: u64) {
        if let Some(block) = self.blocks.get_mut(block_id as usize) {
            block.ref_cnt += 1;
            block.is_free = false;
        }
        // Remove any stale entry in the free queue.
        self.free_queue.retain(|&id| id != block_id);
    }

    /// Look up cached blocks by content hash.
    ///
    /// Returns the block IDs that share this content hash, or `None` if the
    /// hash is not in the index.
    pub fn lookup_by_hash(&self, hash: u64) -> Option<&[u64]> {
        self.hash_index.get(&hash).map(|v| v.as_slice())
    }

    /// Insert a content hash -> block mapping.
    ///
    /// Records that `block_id` stores content identified by `hash`.
    pub fn insert_hash(&mut self, block_id: u64, hash: u64) {
        if let Some(block) = self.blocks.get_mut(block_id as usize) {
            block.content_hash = Some(hash);
        }
        self.hash_index.entry(hash).or_default().push(block_id);
    }
}

// ---------------------------------------------------------------------------
// CacheGroupType
// ---------------------------------------------------------------------------

/// The attention group a cache manager handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheGroupType {
    /// Standard full self-attention.
    FullAttention,
    /// Sliding-window attention.
    SlidingWindow,
    /// Multi-head latent attention (MLA).
    Mla,
    /// Mamba-style state-space model.
    Mamba,
    /// Cross-attention (encoder-decoder).
    CrossAttention,
}

// ---------------------------------------------------------------------------
// CacheGroupManager
// ---------------------------------------------------------------------------

/// Per-attention-type cache manager.
///
/// Owns the block tables (request_id -> Vec<block_id>) and tracks partial-hit
/// requests for copy-on-write handling.
#[derive(Debug, Clone)]
pub struct CacheGroupManager {
    /// Attention group this manager handles.
    pub group_type: CacheGroupType,
    /// Per-request block tables: request_id -> Vec<block_id>.
    pub req_to_blocks: HashMap<String, Vec<u64>>,
    /// Partial-hit requests (CoW tracking): request_id -> Vec<shared_block_id>.
    pub partial_hit_reqs: HashMap<String, Vec<u64>>,
}

impl CacheGroupManager {
    /// Create a new manager for the given group type.
    pub fn new(group_type: CacheGroupType) -> Self {
        CacheGroupManager {
            group_type,
            req_to_blocks: HashMap::new(),
            partial_hit_reqs: HashMap::new(),
        }
    }

    /// Register a prefix-cache hit block for a request (without allocating
    /// from the pool — the block already exists and is shared).
    pub fn add_local_block(&mut self, req_id: &str, block_id: u64) {
        self.req_to_blocks
            .entry(req_id.to_string())
            .or_default()
            .push(block_id);
    }

    /// Allocate `count` new blocks from the shared pool for the given request.
    ///
    /// Returns the list of newly allocated block IDs.  If the pool runs out of
    /// blocks, returns whatever was allocated so far (callers should check
    /// the length matches `count`).
    pub fn allocate_blocks(
        &mut self,
        req_id: &str,
        count: usize,
        pool: &mut BlockPool,
    ) -> Vec<u64> {
        let mut allocated = Vec::with_capacity(count);
        for _ in 0..count {
            match pool.allocate_block() {
                Some(block_id) => allocated.push(block_id),
                None => break,
            }
        }
        self.req_to_blocks
            .entry(req_id.to_string())
            .or_default()
            .extend_from_slice(&allocated);
        allocated
    }

    /// Free all blocks owned by a request and remove the request from tracking.
    ///
    /// Blocks are returned to the pool's free queue and the request entry is
    /// cleaned up from both `req_to_blocks` and `partial_hit_reqs`.
    pub fn free_request(&mut self, req_id: &str, pool: &mut BlockPool) {
        if let Some(blocks) = self.req_to_blocks.remove(req_id) {
            for block_id in blocks {
                pool.free_block(block_id);
            }
        }
        self.partial_hit_reqs.remove(req_id);
    }
}

// ---------------------------------------------------------------------------
// KVCacheCoordinator
// ---------------------------------------------------------------------------

/// Top-level coordinator across all per-attention-type cache managers.
///
/// Owns the shared block pool and dispatches allocation/free requests to the
/// appropriate group manager.
#[derive(Debug, Clone)]
pub struct KVCacheCoordinator {
    /// Managers by group type.
    pub groups: HashMap<CacheGroupType, CacheGroupManager>,
    /// Shared block pool.
    pub pool: BlockPool,
}

impl KVCacheCoordinator {
    /// Create a new coordinator with a shared block pool of the given capacity.
    pub fn new(pool_capacity: u64) -> Self {
        KVCacheCoordinator {
            groups: HashMap::new(),
            pool: BlockPool::new(pool_capacity),
        }
    }

    /// Get or create the cache group manager for a given attention type.
    pub fn get_or_create_group(&mut self, group_type: CacheGroupType) -> &mut CacheGroupManager {
        self.groups
            .entry(group_type)
            .or_insert_with(|| CacheGroupManager::new(group_type))
    }

    /// Allocate `num_tokens` of block slots for a request in the given
    /// attention group.
    ///
    /// Each block currently holds one token (1:1 mapping).  Returns the list
    /// of allocated block IDs, or an error if the pool is exhausted.
    pub fn allocate_slots(
        &mut self,
        req_id: &str,
        num_tokens: usize,
        group_type: CacheGroupType,
    ) -> Result<Vec<u64>, String> {
        // Phase 1: allocate from pool directly (no group borrow active).
        let mut allocated = Vec::with_capacity(num_tokens);
        for _ in 0..num_tokens {
            match self.pool.allocate_block() {
                Some(block_id) => allocated.push(block_id),
                None => break,
            }
        }

        let success = allocated.len() >= num_tokens;

        // Phase 2: register with the group manager.
        let manager: &mut CacheGroupManager = self
            .groups
            .entry(group_type)
            .or_insert_with(|| CacheGroupManager::new(group_type));

        if success {
            manager
                .req_to_blocks
                .entry(req_id.to_string())
                .or_default()
                .extend_from_slice(&allocated);
            Ok(allocated)
        } else {
            // Roll back: remove from manager first, then free blocks
            // (separate borrows via scope).
            {
                manager.req_to_blocks.remove(req_id);
            } // manager borrow ends here
            for &block_id in &allocated {
                self.pool.free_block(block_id);
            }
            Err(format!(
                "Block pool exhausted: requested {num_tokens} block(s), \
                 got {} free block(s) from capacity {}",
                allocated.len(),
                self.pool.capacity
            ))
        }
    }

    /// Free all blocks owned by a request across every group manager.
    ///
    /// Collects affected group types first to permit field-level splitting of
    /// `self.groups` and `self.pool` borrows during the inner free loop.
    pub fn free_request(&mut self, req_id: &str) {
        // Collect affected group types while only borrowing self.groups
        // immutably.
        let affected: Vec<CacheGroupType> = self
            .groups
            .iter()
            .filter(|(_, m)| m.req_to_blocks.contains_key(req_id))
            .map(|(&k, _)| k)
            .collect();
        let freed_any = !affected.is_empty();
        for group_type in &affected {
            if let Some(manager) = self.groups.get_mut(group_type) {
                manager.free_request(req_id, &mut self.pool);
            }
        }
        if !freed_any {
            for manager in self.groups.values_mut() {
                manager.partial_hit_reqs.remove(req_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integration layer — bridge between layered KVCacheCoordinator and
// the unified scheduler (vLLM v1 token-budget model).
// ---------------------------------------------------------------------------

/// Allocate KV cache blocks for a request via the layered KVCacheCoordinator.
///
/// Delegates to [`KVCacheCoordinator::allocate_slots`], reserving `num_tokens`
/// blocks in the specified attention group.  Returns the allocated block IDs.
pub fn allocate_with_layered_cache(
    coord: &mut KVCacheCoordinator,
    req_id: &str,
    num_tokens: usize,
    group_type: CacheGroupType,
) -> Result<Vec<u64>, String> {
    coord.allocate_slots(req_id, num_tokens, group_type)
}

/// Free all KV cache blocks owned by a request.
///
/// Delegates to [`KVCacheCoordinator::free_request`], cleaning up block
/// tables across every attention group.
pub fn free_with_layered_cache(coord: &mut KVCacheCoordinator, req_id: &str) {
    coord.free_request(req_id);
}

/// Pre-execution system: read scheduling requirements from [`SchedulerState`]
/// and pre-allocate layered KV cache blocks for every running request.
///
/// For each running request, computes the token deficit (remaining tokens to
/// compute) and allocates a block for each missing token.  Returns an error
/// if a running request is missing from the request map or block allocation
/// fails.
pub fn kv_cache_prepare_system(
    scheduler: &mut SchedulerState,
    cache_coord: &mut KVCacheCoordinator,
) -> Result<(), String> {
    for req_id in &scheduler.running.clone() {
        let data = scheduler
            .requests
            .get(req_id)
            .ok_or_else(|| format!("request {req_id} not found"))?;
        let needed = (data.num_tokens_with_spec - data.num_computed_tokens)
            .saturating_sub(1) // one token worth of blocks already allocated
            .max(1);
        // Wrap the allocation in a transaction so failure or early return
        // cannot leak blocks.  If allocation succeeds, commit makes the
        // blocks permanent.  If the Result early-returns (via ?), the
        // transaction drops and rolls back any partially allocated blocks.
        let mut tx = KvBlockTransaction::new_for(req_id, cache_coord);
        tx.allocate(needed, CacheGroupType::FullAttention)
            .map_err(|e| format!("KV allocation failed for {req_id}: {e}"))?;
        tx.commit()
            .map_err(|e| format!("KV commit failed for {req_id}: {e}"))?;
    }
    Ok(())
}

/// Post-completion system: free all KV cache blocks for a request that is
/// being removed from the scheduler (completed, cancelled, or preempted).
///
/// Call this after [`SchedulerState::remove_request`] to return blocks to
/// the pool for reuse.
pub fn kv_cache_cleanup_system(coord: &mut KVCacheCoordinator, req_id: &str) {
    coord.free_request(req_id);
}

// ---------------------------------------------------------------------------
// Tests — layered KV cache (BlockPool, CacheGroupManager, KVCacheCoordinator)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BlockPool tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_pool_new() {
        let pool = BlockPool::new(4);
        assert_eq!(pool.capacity, 4);
        assert_eq!(pool.blocks.len(), 4);
        assert_eq!(pool.free_queue.len(), 4);
        // All blocks start free and evictable.
        for block in &pool.blocks {
            assert!(block.is_free, "block {} should be free", block.block_id);
            assert!(
                block.is_evictable,
                "block {} should be evictable",
                block.block_id
            );
            assert_eq!(block.ref_cnt, 0);
            assert!(block.content_hash.is_none());
        }
        // Free queue is in order 0..3 (LRU order).
        assert_eq!(pool.free_queue, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_block_pool_allocate_and_free() {
        let mut pool = BlockPool::new(3);

        // Allocate all blocks.
        let b0 = pool.allocate_block().expect("b0");
        let b1 = pool.allocate_block().expect("b1");
        let b2 = pool.allocate_block().expect("b2");
        assert_eq!(b0, 2); // LIFO from Vec::pop
        assert_eq!(b1, 1);
        assert_eq!(b2, 0);

        // Pool should be exhausted.
        assert!(pool.allocate_block().is_none());

        // Free b1, then verify it's available again.
        pool.free_block(b1);
        assert!(pool.blocks[b1 as usize].is_free);
        assert_eq!(pool.free_queue.last(), Some(&b1));

        let re_alloc = pool.allocate_block().expect("re-alloc");
        assert_eq!(re_alloc, b1);
        assert!(!pool.blocks[re_alloc as usize].is_free);
    }

    #[test]
    fn test_touch_increments_refcnt_and_removes_from_free_queue() {
        let mut pool = BlockPool::new(5);
        let b0 = pool.allocate_block().expect("b0");
        pool.free_block(b0); // put it back (ref_cnt reset to 0)
        assert!(pool.free_queue.contains(&b0));

        pool.touch(b0);
        // free_block resets ref_cnt to 0, touch adds 1.
        assert_eq!(pool.blocks[b0 as usize].ref_cnt, 1);
        assert!(!pool.blocks[b0 as usize].is_free);
        assert!(!pool.free_queue.contains(&b0));
    }

    #[test]
    fn test_lookup_by_hash() {
        let mut pool = BlockPool::new(5);
        let b0 = pool.allocate_block().unwrap();
        let b1 = pool.allocate_block().unwrap();

        let hash_a = 0xDEAD_BEEF;
        let hash_b = 0xCAFE_F00D;

        pool.insert_hash(b0, hash_a);
        pool.insert_hash(b1, hash_b);

        // Exact match.
        let found = pool.lookup_by_hash(hash_a).expect("hash_a should exist");
        assert_eq!(found, &[b0]);

        // Multiple blocks with the same hash (hash collision).
        // Insert another block with hash_a.
        let b2 = pool.allocate_block().unwrap();
        pool.insert_hash(b2, hash_a);
        let found = pool
            .lookup_by_hash(hash_a)
            .expect("hash_a should have 2 blocks");
        assert_eq!(found.len(), 2);
        assert!(found.contains(&b0));
        assert!(found.contains(&b2));

        // Missing hash.
        assert!(pool.lookup_by_hash(0).is_none());
    }

    #[test]
    fn test_insert_hash_updates_block_content_hash() {
        let mut pool = BlockPool::new(3);
        let b = pool.allocate_block().unwrap();
        assert!(pool.blocks[b as usize].content_hash.is_none());

        pool.insert_hash(b, 42);
        assert_eq!(pool.blocks[b as usize].content_hash, Some(42));
    }

    // -----------------------------------------------------------------------
    // CacheGroupManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_group_manager_new() {
        let mgr = CacheGroupManager::new(CacheGroupType::FullAttention);
        assert_eq!(mgr.group_type, CacheGroupType::FullAttention);
        assert!(mgr.req_to_blocks.is_empty());
        assert!(mgr.partial_hit_reqs.is_empty());
    }

    #[test]
    fn test_add_local_block() {
        let mut mgr = CacheGroupManager::new(CacheGroupType::Mla);
        mgr.add_local_block("req-1", 10);
        mgr.add_local_block("req-1", 20);
        mgr.add_local_block("req-2", 30);

        assert_eq!(mgr.req_to_blocks.get("req-1").unwrap(), &vec![10, 20]);
        assert_eq!(mgr.req_to_blocks.get("req-2").unwrap(), &vec![30]);
    }

    #[test]
    fn test_allocate_blocks() {
        let mut pool = BlockPool::new(10);
        let mut mgr = CacheGroupManager::new(CacheGroupType::SlidingWindow);

        let allocated = mgr.allocate_blocks("req-1", 3, &mut pool);
        assert_eq!(allocated.len(), 3);
        assert_eq!(pool.free_queue.len(), 7);

        // Blocks are tracked in req_to_blocks.
        assert_eq!(mgr.req_to_blocks.get("req-1").unwrap().len(), 3);
    }

    #[test]
    fn test_allocate_blocks_exhaustion() {
        let mut pool = BlockPool::new(2);
        let mut mgr = CacheGroupManager::new(CacheGroupType::FullAttention);

        let allocated = mgr.allocate_blocks("req-1", 5, &mut pool);
        // Only 2 blocks available.
        assert_eq!(allocated.len(), 2);
        assert!(pool.allocate_block().is_none());
    }

    #[test]
    fn test_free_request() {
        let mut pool = BlockPool::new(10);
        let mut mgr = CacheGroupManager::new(CacheGroupType::FullAttention);

        mgr.allocate_blocks("req-1", 3, &mut pool);
        mgr.add_local_block("req-1", 42);
        assert!(mgr.req_to_blocks.contains_key("req-1"));

        mgr.free_request("req-1", &mut pool);
        assert!(!mgr.req_to_blocks.contains_key("req-1"));
        // allocate_blocks(3) allocates 3 blocks from pool of 10 → 7 free.
        // add_local_block adds block 42 which wasn't allocated from the pool,
        // so free_request frees all 4 entries (3 real + 1 orphan) → 7 + 4 = 11.
        assert_eq!(pool.free_queue.len(), 11);
    }

    // -----------------------------------------------------------------------
    // KVCacheCoordinator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_coordinator_new() {
        let coord = KVCacheCoordinator::new(16);
        assert_eq!(coord.pool.capacity, 16);
        assert!(coord.groups.is_empty());
    }

    #[test]
    fn test_get_or_create_group() {
        let mut coord = KVCacheCoordinator::new(16);
        let mgr = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert_eq!(mgr.group_type, CacheGroupType::FullAttention);

        // Getting the same type returns the existing manager.
        let mgr2 = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert_eq!(mgr2.group_type, CacheGroupType::FullAttention);

        // Different type creates a new one.
        let _ = coord.get_or_create_group(CacheGroupType::Mamba);
        assert_eq!(coord.groups.len(), 2);
    }

    #[test]
    fn test_allocate_slots_success() {
        let mut coord = KVCacheCoordinator::new(10);
        let slots = coord
            .allocate_slots("req-1", 3, CacheGroupType::FullAttention)
            .expect("allocation should succeed");
        assert_eq!(slots.len(), 3);

        // Verify the blocks are tracked in the right group.
        let mgr = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert_eq!(mgr.req_to_blocks.get("req-1").unwrap().len(), 3);
    }

    #[test]
    fn test_allocate_slots_exhaustion_rollback() {
        let mut coord = KVCacheCoordinator::new(2);
        // First allocation succeeds.
        let _slots = coord
            .allocate_slots("req-1", 2, CacheGroupType::FullAttention)
            .expect("first alloc should succeed");

        // Second allocation should fail and roll back.
        let err = coord
            .allocate_slots("req-2", 3, CacheGroupType::FullAttention)
            .expect_err("should fail on exhaustion");

        assert!(err.contains("exhausted"), "error: {err}");

        // req-2 should not have any partial tracking.
        let mgr = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert!(
            !mgr.req_to_blocks.contains_key("req-2"),
            "rolled-back request should not be tracked"
        );
    }

    #[test]
    fn test_coordinator_free_request() {
        let mut coord = KVCacheCoordinator::new(10);

        // Allocate to two groups.
        coord
            .allocate_slots("req-1", 2, CacheGroupType::FullAttention)
            .unwrap();
        coord
            .allocate_slots("req-1", 1, CacheGroupType::SlidingWindow)
            .unwrap();

        assert_eq!(coord.pool.free_queue.len(), 7);

        coord.free_request("req-1");

        // All blocks should be freed.
        assert_eq!(coord.pool.free_queue.len(), 10);
        let fa_mgr = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert!(!fa_mgr.req_to_blocks.contains_key("req-1"));
        let sw_mgr = coord.get_or_create_group(CacheGroupType::SlidingWindow);
        assert!(!sw_mgr.req_to_blocks.contains_key("req-1"));
    }

    #[test]
    fn test_coordinator_free_nonexistent_request() {
        let mut coord = KVCacheCoordinator::new(5);
        // Freeing a nonexistent request should not panic.
        coord.free_request("no-such-req");
        assert_eq!(coord.pool.free_queue.len(), 5);
    }

    #[test]
    fn test_allocate_slots_different_groups_independent() {
        let mut coord = KVCacheCoordinator::new(10);

        let fa_slots = coord
            .allocate_slots("req-a", 3, CacheGroupType::FullAttention)
            .unwrap();
        let sw_slots = coord
            .allocate_slots("req-a", 2, CacheGroupType::SlidingWindow)
            .unwrap();
        let mla_slots = coord
            .allocate_slots("req-b", 4, CacheGroupType::Mla)
            .unwrap();

        assert_eq!(fa_slots.len(), 3);
        assert_eq!(sw_slots.len(), 2);
        assert_eq!(mla_slots.len(), 4);

        // req-a has blocks in two groups.
        let fa_mgr = coord.get_or_create_group(CacheGroupType::FullAttention);
        assert_eq!(fa_mgr.req_to_blocks.get("req-a").unwrap().len(), 3);
        let sw_mgr = coord.get_or_create_group(CacheGroupType::SlidingWindow);
        assert_eq!(sw_mgr.req_to_blocks.get("req-a").unwrap().len(), 2);

        // req-b in MLA.
        let mla_mgr = coord.get_or_create_group(CacheGroupType::Mla);
        assert_eq!(mla_mgr.req_to_blocks.get("req-b").unwrap().len(), 4);

        // Freeing req-a frees its 5 blocks; req-b still holds 4.
        // Pool capacity 10 - 4 (req-b) = 6 free.
        coord.free_request("req-a");
        assert_eq!(coord.pool.free_queue.len(), 6);
    }

    #[test]
    fn test_kv_cache_prepare_system_allocates_for_running() {
        let mut scheduler = SchedulerState::new(128, 8);
        scheduler.add_request("req-1", 100, 0);
        scheduler.add_request("req-2", 50, 0);

        // Move both to running
        let _ = scheduler.schedule_once();

        let mut coord = KVCacheCoordinator::new(200);

        let result = kv_cache_prepare_system(&mut scheduler, &mut coord);
        assert!(result.is_ok(), "prepare should succeed: {:?}", result);

        // Each request should have blocks allocated
        let fa_mgr = coord.groups.get(&CacheGroupType::FullAttention).unwrap();
        assert!(fa_mgr.req_to_blocks.contains_key("req-1"));
        assert!(fa_mgr.req_to_blocks.contains_key("req-2"));
        assert!(!fa_mgr.req_to_blocks.get("req-1").unwrap().is_empty());
        assert!(!fa_mgr.req_to_blocks.get("req-2").unwrap().is_empty());
    }

    #[test]
    fn test_kv_cache_cleanup_system_frees_blocks() {
        let mut coord = KVCacheCoordinator::new(10);
        coord
            .allocate_slots("req-1", 3, CacheGroupType::FullAttention)
            .unwrap();

        assert!(coord
            .groups
            .get(&CacheGroupType::FullAttention)
            .unwrap()
            .req_to_blocks
            .contains_key("req-1"));

        kv_cache_cleanup_system(&mut coord, "req-1");

        assert!(!coord
            .groups
            .get(&CacheGroupType::FullAttention)
            .unwrap()
            .req_to_blocks
            .contains_key("req-1"));
        assert_eq!(coord.pool.free_queue.len(), 10);
    }

    #[test]
    fn test_allocate_with_layered_cache_delegates() {
        let mut coord = KVCacheCoordinator::new(5);
        let slots =
            allocate_with_layered_cache(&mut coord, "test-req", 3, CacheGroupType::FullAttention)
                .expect("allocation should work");
        assert_eq!(slots.len(), 3);
    }

    #[test]
    fn test_free_with_layered_cache_delegates() {
        let mut coord = KVCacheCoordinator::new(5);
        coord
            .allocate_slots("to-free", 2, CacheGroupType::FullAttention)
            .unwrap();

        free_with_layered_cache(&mut coord, "to-free");
        assert_eq!(coord.pool.free_queue.len(), 5);
    }
}
