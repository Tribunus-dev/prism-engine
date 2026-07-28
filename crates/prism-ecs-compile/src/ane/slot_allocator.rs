//! LRU slot allocator for ANE SRAM row slots.
//!
//! Authority: pure LRU eviction policy for the ANE weight-row cache.
//!
//! Manages a fixed number of slots (one per cacheable weight row).
//! Each slot is identified by its index (0..max_slots). When all
//! slots are full, evicts the least-recently-used entry.
//!
//! This module is engine-neutral — no `Arena`, no `CoreAiModel`, no
//! `mlx_rs::Array` dependencies. The engine's `legacy_ane/` wraps
//! this allocator in an IOSurface-backed `Arena` for the actual SRAM
//! storage, but the eviction logic is canonical here.

/// LRU slot allocator for ANE SRAM row slots.
///
/// Tracks which token IDs occupy which slots, with strict LRU
/// eviction when a new token is admitted and the cache is full.
/// Allocation is O(N) over the current occupancy — fine for the
/// typical ANE SRAM budget (~256 rows).
#[derive(Debug, Clone)]
pub struct SlotAllocator {
    /// Maximum number of slots.
    pub max_slots: u32,
    /// Current occupancy: `token_id` → `slot_index` (or `None`).
    occupied: Vec<Option<u32>>,
    /// LRU tracking: `slot_index` → last access sequence number.
    lru_order: Vec<u64>,
    /// Monotonically increasing access counter.
    access_counter: u64,
}

impl SlotAllocator {
    /// Create a new slot allocator with the given capacity.
    pub fn new(max_slots: u32) -> Self {
        let count = max_slots as usize;
        Self {
            max_slots,
            occupied: vec![None; count],
            lru_order: vec![0; count],
            access_counter: 0,
        }
    }

    /// Allocate a slot for `token_id`, evicting LRU if full.
    ///
    /// Returns the slot index and whether the token was already
    /// present. When the token is already present, the LRU entry
    /// is touched but the slot is not relocated.
    pub fn allocate(&mut self, token_id: u32) -> (usize, bool) {
        // Check if already allocated
        for (idx, slot) in self.occupied.iter().enumerate() {
            if *slot == Some(token_id) {
                self.lru_order[idx] = self.access_counter;
                self.access_counter = self.access_counter.saturating_add(1);
                return (idx, true);
            }
        }

        // Find free slot or LRU victim
        let slot_idx = self.find_victim();
        self.occupied[slot_idx] = Some(token_id);
        self.lru_order[slot_idx] = self.access_counter;
        self.access_counter = self.access_counter.saturating_add(1);
        (slot_idx, false)
    }

    /// Find the slot to evict: a free slot if available, else the
    /// slot with the smallest `lru_order` value (oldest access).
    /// Ties go to the smaller slot index for determinism.
    fn find_victim(&self) -> usize {
        // Prefer a free slot first.
        for (i, slot) in self.occupied.iter().enumerate() {
            if slot.is_none() {
                return i;
            }
        }
        // All occupied: find the LRU slot.
        let mut min_idx = 0;
        let mut min_val = self.lru_order[0];
        for i in 1..self.lru_order.len() {
            if self.lru_order[i] < min_val {
                min_val = self.lru_order[i];
                min_idx = i;
            }
        }
        min_idx
    }

    /// Get the slot index for a `token_id`, or `None` if not cached.
    pub fn lookup(&self, token_id: u32) -> Option<usize> {
        self.occupied
            .iter()
            .position(|slot| *slot == Some(token_id))
    }

    /// Returns the token ID at a given slot index, or `None` if the
    /// slot is empty.
    pub fn token_at(&self, slot: usize) -> Option<u32> {
        self.occupied.get(slot).copied().flatten()
    }

    /// Number of occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.occupied.iter().filter(|s| s.is_some()).count()
    }

    /// Clear all slots, resetting the LRU counter.
    pub fn clear(&mut self) {
        for slot in &mut self.occupied {
            *slot = None;
        }
        for seq in &mut self.lru_order {
            *seq = 0;
        }
        self.access_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_returns_existing_slot() {
        let mut alloc = SlotAllocator::new(4);
        let (s0, was_cached0) = alloc.allocate(100);
        assert!(!was_cached0, "first allocation should be a fresh insert");
        assert_eq!(s0, 0);
        let (s1, was_cached1) = alloc.allocate(100);
        assert!(was_cached1, "second allocation should hit the cache");
        assert_eq!(s1, 0);
    }

    #[test]
    fn allocate_evicts_lru_when_full() {
        let mut alloc = SlotAllocator::new(2);
        alloc.allocate(1);
        alloc.allocate(2);
        assert_eq!(alloc.occupied_count(), 2);
        // Adding a third token evicts the oldest (1).
        let (slot, was_cached) = alloc.allocate(3);
        assert!(!was_cached, "third token is a fresh insert");
        assert_eq!(slot, 0, "should have evicted slot 0 (oldest)");
        assert_eq!(alloc.token_at(0), Some(3));
        assert_eq!(alloc.token_at(1), Some(2));
    }

    #[test]
    fn lookup_returns_some_for_present_token() {
        let mut alloc = SlotAllocator::new(3);
        alloc.allocate(7);
        assert_eq!(alloc.lookup(7), Some(0));
        assert_eq!(alloc.lookup(99), None);
    }

    #[test]
    fn clear_resets_state() {
        let mut alloc = SlotAllocator::new(2);
        alloc.allocate(1);
        alloc.allocate(2);
        assert_eq!(alloc.occupied_count(), 2);
        alloc.clear();
        assert_eq!(alloc.occupied_count(), 0);
        assert_eq!(alloc.lookup(1), None);
    }

    #[test]
    fn lru_touch_on_repeat_allocation() {
        let mut alloc = SlotAllocator::new(2);
        alloc.allocate(1);
        alloc.allocate(2);
        // Touch 1, so 2 is now the LRU.
        alloc.allocate(1);
        // Adding 3 should evict 2 (the LRU), not 1.
        let (slot, was_cached) = alloc.allocate(3);
        assert!(!was_cached);
        assert_eq!(slot, 1, "should have evicted slot 1 (LRU), not slot 0");
        assert_eq!(alloc.token_at(0), Some(1));
        assert_eq!(alloc.token_at(1), Some(3));
    }
}
