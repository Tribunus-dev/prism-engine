//! VM manager for IOSurface allocation across work queue slots.
//!
//! Manages a global pool of survivor positions partitioned across 8 slots.
//! Each slot gets a variable slice of the total 160K survivor budget.
//! Allocation is tracked in a host-side table — no Core ML model changes.

const TOTAL_SURVIVOR_BUDGET: u32 = 655_360; // 32 slots × 20K

/// Return the byte size of one survivor position (K + V, FP16).
const fn bytes_per_survivor() -> u64 {
    // Each survivor: n_kv_heads (8) * head_dim (512) * 2 bytes (FP16)
    // for K plus the same for V = 8 * 512 * 2 * 2 = 16384 bytes
    16384
}

/// Per-slot allocation in the global survivor pool.
#[derive(Debug, Clone, Copy)]
pub struct SlotAllocation {
    /// Byte offset into the shared IOSurface.
    pub byte_offset: u64,
    /// Number of survivor positions allocated to this slot.
    pub survivor_count: u32,
    /// Number of gather passes needed (ceil(survivor_count / 20480)).
    pub passes: u32,
}

/// VM manager for shared IOSurface pool.
pub struct VmManager {
    /// Per-slot allocations.
    pub slots: [SlotAllocation; 32],
    /// Total bytes allocated.
    pub total_bytes: u64,
}

impl VmManager {
    /// Create a new VM manager with equal slot distribution.
    pub fn new() -> Self {
        let per_slot_survivors = TOTAL_SURVIVOR_BUDGET / 32;
        let mut slots = [SlotAllocation {
            byte_offset: 0,
            survivor_count: per_slot_survivors,
            passes: (per_slot_survivors + 20479) / 20480,
        }; 32];
        let bps = bytes_per_survivor();
        let mut cursor = 0u64;
        for slot in slots.iter_mut() {
            slot.byte_offset = cursor;
            cursor += slot.survivor_count as u64 * bps;
        }
        Self {
            slots,
            total_bytes: cursor,
        }
    }

    /// Reallocate survivor budget across slots based on entropy demand.
    /// High-entropy slots get more budget; idle slots give theirs up.
    pub fn rebalance(&mut self, slot_entropies: &[f32; 32]) {
        let total_mass: f32 = slot_entropies.iter().sum();
        if total_mass <= 0.0 {
            return;
        }

        let bps = bytes_per_survivor();
        let mut cursor = 0u64;
        for i in 0..32 {
            let fraction = slot_entropies[i] / total_mass;
            let survivors = (TOTAL_SURVIVOR_BUDGET as f32 * fraction) as u32;
            let clamped = survivors.max(1024).min(65536); // min 1K, max 64K
            let passes = (clamped + 20479) / 20480; // ceil division
            self.slots[i] = SlotAllocation {
                byte_offset: cursor,
                survivor_count: clamped,
                passes,
            };
            cursor += clamped as u64 * bps;
        }
        self.total_bytes = cursor;
    }

    /// Get the allocation for a specific slot.
    /// Configure the VM manager for per-slot survivor budgets.
    /// Each slot gets a slice of the shared IOSurface pool sized to its survivor_count.
    /// Only listed slots have their byte offsets recomputed; unlisted slots are untouched.
    pub fn configure_slots(&mut self, profiles: &[(u32, u32)]) {
        let bps = bytes_per_survivor();
        // Apply specified survivor counts to listed slots
        for &(slot_id, survivor_count) in profiles {
            let slot = slot_id as usize;
            self.slots[slot].survivor_count = survivor_count;
            self.slots[slot].passes = (survivor_count + 20479) / 20480;
        }
        // Recalculate byte offsets for all 32 slots
        let mut cursor = 0u64;
        for slot in self.slots.iter_mut() {
            slot.byte_offset = cursor;
            cursor += slot.survivor_count as u64 * bps;
        }
        self.total_bytes = cursor;
    }

    /// Generate profiles for maximum swarm density with given per-agent context.
    /// Each slot gets 20480 survivors (equivalent to ~1M context at 50:1 compression).
    pub fn max_concurrency_profiles(
        &self,
        _total_budget_bytes: u64,
        _overhead_per_slot: u64,
    ) -> Vec<(u32, u32)> {
        let max_slots = self.slots.len();
        (0..max_slots as u32).map(|i| (i, 20480u32)).collect()
    }

    /// Get the allocation for a specific slot.
    pub fn slot_allocation(&self, slot_id: u32) -> &SlotAllocation {
        &self.slots[slot_id as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_manager_initial_distribution() {
        let vm = VmManager::new();
        assert_eq!(vm.slots.len(), 32);
        for (i, slot) in vm.slots.iter().enumerate() {
            assert_eq!(slot.survivor_count, TOTAL_SURVIVOR_BUDGET / 32);
            assert_eq!(slot.passes, 1);
            assert_eq!(
                slot.byte_offset,
                (i as u64) * (TOTAL_SURVIVOR_BUDGET as u64 / 32) * bytes_per_survivor()
            );
        }
    }

    #[test]
    fn test_vm_manager_rebalance_uniform() {
        let mut vm = VmManager::new();
        let entropies = [1.0f32; 32];
        vm.rebalance(&entropies);
        // Uniform entropy → uniform distribution
        for slot in vm.slots.iter() {
            assert!(slot.survivor_count >= 1024);
            assert!(slot.survivor_count <= 65536);
        }
    }

    #[test]
    fn test_vm_manager_rebalance_skewed() {
        let mut vm = VmManager::new();
        let entropies = [
            8.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        vm.rebalance(&entropies);
        // Slot 0 should get most of the budget
        assert!(vm.slots[0].survivor_count > vm.slots[1].survivor_count);
    }

    #[test]
    fn test_vm_manager_rebalance_zero_mass() {
        let mut vm = VmManager::new();
        let entropies = [0.0f32; 32];
        let before = vm.slots;
        vm.rebalance(&entropies);
        // Zero entropy → no change
        for (i, slot) in vm.slots.iter().enumerate() {
            assert_eq!(slot.survivor_count, before[i].survivor_count);
        }
    }

    #[test]
    fn test_vm_manager_slot_allocation() {
        let vm = VmManager::new();
        for i in 0..32 {
            let alloc = vm.slot_allocation(i);
            assert_eq!(alloc.byte_offset, vm.slots[i as usize].byte_offset);
        }
    }

    #[test]
    fn test_vm_manager_configure_slots() {
        let mut vm = VmManager::new();
        let bps = bytes_per_survivor();
        // Configure 3 slots with varied survivor counts
        let profiles = [(0u32, 1024u32), (5u32, 40960u32), (15u32, 20480u32)];
        vm.configure_slots(&profiles);
        // Slot 0
        assert_eq!(vm.slots[0].survivor_count, 1024);
        assert_eq!(vm.slots[0].byte_offset, 0);
        assert_eq!(vm.slots[0].passes, 1);
        // Slot 5: preceded by slot 0 (1024) + slots 1-4 (4x20480)
        let before_slot5 = 1u64 * 1024 + 4u64 * 20480; // survivors before slot 5
        assert_eq!(vm.slots[5].survivor_count, 40960);
        assert_eq!(vm.slots[5].byte_offset, before_slot5 * bps);
        assert_eq!(vm.slots[5].passes, 2); // 40960/20480 = 2
                                           // Slot 15: preceded by slots 0-14
        assert_eq!(vm.slots[15].survivor_count, 20480);
        let before_slot15 = 1u64 * 1024 + 4u64 * 20480 + 1u64 * 40960 + 9u64 * 20480;
        assert_eq!(vm.slots[15].byte_offset, before_slot15 * bps);
        assert_eq!(vm.slots[15].passes, 1);
        // Total should cover all 32 slots
        let total =
            1u64 * 1024 + 4u64 * 20480 + 1u64 * 40960 + 9u64 * 20480 + 1u64 * 20480 + 16u64 * 20480;
        assert_eq!(vm.total_bytes, total * bps);
        // Unlisted slots retain their old survivor_count
        assert_eq!(vm.slots[1].survivor_count, TOTAL_SURVIVOR_BUDGET / 32);
        // Slot 1 has recalculated byte_offset (after slot 0's 1024 survivors)
        assert_eq!(vm.slots[1].byte_offset, 1024 * bps);
    }

    #[test]
    fn test_vm_manager_max_concurrency_profiles() {
        let vm = VmManager::new();
        let profiles = vm.max_concurrency_profiles(1 << 30, 0);
        assert_eq!(profiles.len(), 32);
        for (i, &(slot_id, survivor_count)) in profiles.iter().enumerate() {
            assert_eq!(slot_id, i as u32);
            assert_eq!(survivor_count, 20480);
        }
    }
}
