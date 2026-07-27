//! Lease allocation system (constitutional home).
//!
//! Placeholder for the engine's `slot_lease_manager.rs` (778 LOC).
//! The full algorithm migrates in step 18. The engine file is the
//! legacy duplicate and is deleted in step 58.
//!
//! The lease-allocation system owns the per-slot lease state: which
//! request is leased to which slot, when leases expire, and the
//! fairness rules (preemption, anti-starvation).

use std::collections::BTreeMap;

/// Per-slot lease record (placeholder).
#[derive(Debug, Clone)]
pub struct SlotLease {
    pub slot_id: u64,
    pub request_id: Option<String>,
    pub epoch: u64,
}

#[derive(Debug, Clone, Default)]
pub struct SlotLeaseManager {
    leases: BTreeMap<u64, SlotLease>,
    next_slot_id: u64,
}

impl SlotLeaseManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a new slot for a request. Returns the slot id.
    pub fn allocate(&mut self, request_id: &str) -> u64 {
        let id = self.next_slot_id;
        self.next_slot_id += 1;
        self.leases.insert(
            id,
            SlotLease {
                slot_id: id,
                request_id: Some(request_id.to_string()),
                epoch: 0,
            },
        );
        id
    }

    /// Release a slot.
    pub fn release(&mut self, slot_id: u64) {
        self.leases.remove(&slot_id);
    }

    pub fn lease(&self, slot_id: u64) -> Option<&SlotLease> {
        self.leases.get(&slot_id)
    }

    pub fn len(&self) -> usize {
        self.leases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leases.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_manager_is_empty() {
        let m = SlotLeaseManager::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn allocate_increments_slot_id() {
        let mut m = SlotLeaseManager::new();
        let id1 = m.allocate("r1");
        let id2 = m.allocate("r2");
        assert_ne!(id1, id2);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn release_removes_lease() {
        let mut m = SlotLeaseManager::new();
        let id = m.allocate("r1");
        m.release(id);
        assert!(m.is_empty());
    }

    #[test]
    fn lease_lookup_returns_correct_record() {
        let mut m = SlotLeaseManager::new();
        let id = m.allocate("r1");
        let lease = m.lease(id).expect("lease present");
        assert_eq!(lease.request_id.as_deref(), Some("r1"));
    }
}
