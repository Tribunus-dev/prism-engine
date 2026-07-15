//! PagedAttention-style KV cache block allocator.
//!
//! Manages a fixed pool of fixed-size blocks for compressed KV cache
//! (NF4 tile format).  Each block holds 16 token-positions worth of
//! key+value data.  The allocator supports allocating contiguous runs of
//! blocks for a logical decode slot and recycling them back to a free
//! pool.


use std::collections::HashMap;

/// Per-position KV cache bytes in NF4 tile640 format (8 heads × 2 (K+V) × 360 bytes).
const KV_NF4_PER_POSITION: u64 = 5760;

/// Identifier for a physical block in the pool.
pub type BlockId = usize;

/// A single physical block entry.
#[derive(Debug, Clone)]
struct BlockEntry {
    /// Slot that owns this block, or `None` if free.
    slot_id: Option<u32>,
}

/// Tokens stored per block (each token occupies `KV_NF4_PER_POSITION` bytes).
const TOKENS_PER_BLOCK: usize = 16;

/// PagedAttention-style KV cache block allocator.
///
/// Maintains a fixed-size pool of physical blocks.  Blocks are allocated
/// in contiguous runs for a logical slot and recycled on free.
///
/// # Layout
///
/// Each block holds exactly `TOKENS_PER_BLOCK * KV_NF4_PER_POSITION` bytes
/// of compressed NF4 tile data for one layer's worth of key+value.
#[derive(Debug)]
pub struct BlockTable {
    /// Physical block pool indexed by block ID.
    blocks: Vec<BlockEntry>,
    /// Recycled block IDs available for reuse.
    free_list: Vec<BlockId>,
    /// Byte size of each block (`TOKENS_PER_BLOCK * KV_NF4_PER_POSITION`).
    block_size: usize,
    /// Total number of blocks in the pool.
    num_blocks: usize,
    /// Next auto-incrementing slot ID.
    next_slot_id: u32,
    /// Map from slot ID to its allocated block IDs.
    slots: HashMap<u32, Vec<BlockId>>,
}

impl BlockTable {
    /// Create a new block table with the given KV cache budget.
    ///
    /// `kv_budget` is the total byte budget for the KV cache.
    /// `block_size` is the byte size of each block (typically
    /// `TOKENS_PER_BLOCK * KV_NF4_PER_POSITION`).
    pub fn new(kv_budget: u64, block_size: u64) -> Self {
        let block_size = block_size as usize;
        let num_blocks = if block_size > 0 {
            (kv_budget as usize) / block_size
        } else {
            0
        };

        let blocks = vec![BlockEntry { slot_id: None }; num_blocks];
        // Recycled blocks accumulate here; initially empty since blocks
        // are allocated linearly from the free-start cursor.
        let free_list = Vec::new();

        Self {
            blocks,
            free_list,
            block_size,
            num_blocks,
            next_slot_id: 0,
            slots: HashMap::new(),
        }
    }

    /// Allocate a contiguous run of `count` blocks for a new slot.
    ///
    /// Returns the block IDs of the allocated blocks, or an empty `Vec`
    /// when insufficient contiguous free blocks are available.
    pub fn allocate_blocks(&mut self, count: usize) -> Vec<BlockId> {
        if count == 0 || count > self.num_blocks {
            return Vec::new();
        }

        // First, try to find a contiguous run from the recycled free_list.
        // Sort the free_list so we can detect runs.
        self.free_list.sort_unstable();
        if let Some(run) = find_contiguous_run(&self.free_list, count) {
            // Remove the blocks from free_list.
            let allocated = run.clone();
            for &bid in &allocated {
                if let Some(pos) = self.free_list.iter().position(|x| *x == bid) {
                    self.free_list.swap_remove(pos);
                }
            }

            let slot_id = self.next_slot_id;
            self.next_slot_id += 1;

            for &bid in &allocated {
                self.blocks[bid].slot_id = Some(slot_id);
            }
            self.slots.insert(slot_id, allocated.clone());
            return allocated;
        }

        // No contiguous run in free_list; scan the full block array for
        // a run of entirely free blocks.
        let mut start = 0;
        let mut found = false;
        for i in 0..=self.num_blocks.saturating_sub(count) {
            if self.blocks[i].slot_id.is_some() {
                continue;
            }
            let mut free_run = true;
            for j in 0..count {
                if self.blocks[i + j].slot_id.is_some() {
                    free_run = false;
                    break;
                }
            }
            if free_run {
                start = i;
                found = true;
                break;
            }
        }

        if !found {
            return Vec::new();
        }

        let slot_id = self.next_slot_id;
        self.next_slot_id += 1;

        let mut allocated = Vec::with_capacity(count);
        for i in 0..count {
            let bid = start + i;
            self.blocks[bid].slot_id = Some(slot_id);
            // Also remove from free_list if it somehow lingered.
            if let Some(pos) = self.free_list.iter().position(|x| *x == bid) {
                self.free_list.swap_remove(pos);
            }
            allocated.push(bid);
        }

        self.slots.insert(slot_id, allocated.clone());
        allocated
    }

    /// Free all blocks owned by `slot_id` back to the pool.
    pub fn free_blocks(&mut self, slot_id: u32) {
        if let Some(blocks) = self.slots.remove(&slot_id) {
            for &bid in &blocks {
                self.blocks[bid].slot_id = None;
                self.free_list.push(bid);
            }
        }
    }

    /// Map a `slot_id` + `token_position` to the physical byte offset in
    /// the block pool.
    ///
    /// The offset is computed as:
    /// ```text
    /// offset = block_id * block_size + (position % 16) * KV_NF4_PER_POSITION
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `slot_id` is unknown or `token_position` maps beyond the
    /// slot's allocated blocks.
    pub fn physical_offset(&self, slot_id: u32, token_position: u16) -> u64 {
        let blocks = self
            .slots
            .get(&slot_id)
            .unwrap_or_else(|| panic!("BlockTable::physical_offset: unknown slot_id {slot_id}"));

        let block_index = token_position as usize / TOKENS_PER_BLOCK;
        let block_id = blocks[block_index];
        let pos_in_block = token_position as usize % TOKENS_PER_BLOCK;

        (block_id * self.block_size + pos_in_block * KV_NF4_PER_POSITION as usize) as u64
    }
}

/// Find a contiguous run of `count` elements in a sorted slice.
/// Returns `None` if no such run exists.
fn find_contiguous_run(sorted: &[BlockId], count: usize) -> Option<Vec<BlockId>> {
    if count == 0 || sorted.len() < count {
        return None;
    }
    // Walk the sorted slice looking for `count` consecutive IDs.
    let mut run_len = 1;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] + 1 {
            run_len += 1;
            if run_len >= count {
                let start = i + 1 - count;
                return Some(sorted[start..start + count].to_vec());
            }
        } else {
            run_len = 1;
        }
    }

    // Run may extend to the end of the slice.
    if run_len >= count {
        let start = sorted.len() - run_len;
        return Some(sorted[start..start + count].to_vec());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK_SIZE: u64 = (TOKENS_PER_BLOCK as u64) * KV_NF4_PER_POSITION;

    #[test]
    fn test_allocate_sequential_ids() {
        let budget = 10 * BLOCK_SIZE + 100; // room for 10 blocks + slack
        let mut table = BlockTable::new(budget, BLOCK_SIZE);
        let blocks = table.allocate_blocks(10);
        assert_eq!(blocks.len(), 10, "expected 10 blocks");
        for i in 0..10 {
            assert_eq!(blocks[i], i, "block {i} should have ID {i}");
        }
    }

    #[test]
    fn test_allocate_free_reallocate_cycles_back() {
        let budget = 10 * BLOCK_SIZE;
        let mut table = BlockTable::new(budget, BLOCK_SIZE);

        let blocks1 = table.allocate_blocks(5);
        assert_eq!(blocks1.len(), 5);
        assert_eq!(blocks1, vec![0, 1, 2, 3, 4]);

        // Free slot 0 (the first allocation)
        table.free_blocks(0);

        // Re-allocate — should yield the same recycled blocks
        let blocks2 = table.allocate_blocks(5);
        assert_eq!(blocks2.len(), 5, "re-allocation should return 5 blocks");
        // They may be in a different order depending on pool reuse,
        // but they should be the same IDs cycled back.
        let mut sorted2 = blocks2.clone();
        sorted2.sort_unstable();
        assert_eq!(
            sorted2,
            vec![0, 1, 2, 3, 4],
            "recycled blocks should be IDs 0-4"
        );
    }

    #[test]
    fn test_physical_offset_mapping() {
        let budget = 8 * BLOCK_SIZE; // 8 blocks × 16 tokens
        let mut table = BlockTable::new(budget, BLOCK_SIZE);

        // Allocate 4 blocks for slot 0
        let _blocks = table.allocate_blocks(4);

        // token 0 → block[0], offset = block_id * block_size + 0
        let expected_0 = 0u64 * BLOCK_SIZE;
        assert_eq!(
            table.physical_offset(0, 0),
            expected_0,
            "token 0 should map to start of block 0"
        );

        // token 5 → block[0], offset = block[0] * block_size + 5 * KV_NF4_PER_POSITION
        let expected_5 = 0u64 * BLOCK_SIZE + 5 * KV_NF4_PER_POSITION;
        assert_eq!(
            table.physical_offset(0, 5),
            expected_5,
            "token 5 should map 5 positions into block 0"
        );

        // token 15 → block[0], last position in block 0
        let expected_15 = 0u64 * BLOCK_SIZE + 15 * KV_NF4_PER_POSITION;
        assert_eq!(
            table.physical_offset(0, 15),
            expected_15,
            "token 15 should map to last position of block 0"
        );

        // token 16 → block[1], offset = block[1] * block_size + 0
        let expected_16 = 1u64 * BLOCK_SIZE;
        assert_eq!(
            table.physical_offset(0, 16),
            expected_16,
            "token 16 should map to start of block 1"
        );

        // token 33 → block[2], offset = block[2] * block_size + 1 * KV_NF4_PER_POSITION
        // 33 / 16 = 2 (block index), 33 % 16 = 1 (position in block)
        let expected_33 = 2u64 * BLOCK_SIZE + 1 * KV_NF4_PER_POSITION;
        assert_eq!(
            table.physical_offset(0, 33),
            expected_33,
            "token 33 should map to block 2, position 1"
        );
    }

    #[test]
    fn test_allocate_insufficient_blocks() {
        let budget = 3 * BLOCK_SIZE; // only 3 blocks
        let mut table = BlockTable::new(budget, BLOCK_SIZE);
        let blocks = table.allocate_blocks(10);
        assert!(
            blocks.is_empty(),
            "should return empty when insufficient blocks"
        );
    }
}
