//! KV transaction system (constitutional home).
//!
//! Placeholder for the engine's kv_transaction.rs. The engine file
//! is the legacy duplicate and is deleted in step 58. The full
//! implementation is added when the engine's KV arena types
//! migrate.

use crate::scheduling::state::lane_work::SlotLeaseId;

/// Placeholder KV transaction record. The full implementation
/// arrives with the kv_arena migration.
#[derive(Debug, Clone)]
pub struct KvTransaction {
    pub id: u64,
    pub slot: SlotLeaseId,
    pub span: u32,
}

impl KvTransaction {
    pub fn new(id: u64, slot: SlotLeaseId, span: u32) -> Self {
        Self { id, slot, span }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_transaction_carries_slot_and_span() {
        let tx = KvTransaction::new(1, SlotLeaseId(42), 1024);
        assert_eq!(tx.id, 1);
        assert_eq!(tx.slot, SlotLeaseId(42));
        assert_eq!(tx.span, 1024);
    }
}
