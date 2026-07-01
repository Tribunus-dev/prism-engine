//! Rolling transition ledger — bounded semantic receipt history.
//!
//! Stores the N most recent committed transition receipts in a VecDeque.
//! Capacity is immutable after construction.

use std::collections::VecDeque;
use std::num::NonZeroUsize;

use crate::runtime::ledger::entry::{
    TransitionReceipt, DEFAULT_TRANSITION_LEDGER_CAPACITY,
};

/// Bounded rolling window of semantic transition receipts.
///
/// Capacity is configured at construction and never changes.
/// The default capacity is 10 entries.  Zero capacity is rejected.
pub struct TransitionLedger {
    capacity: usize,
    history: VecDeque<TransitionReceipt>,
    next_receipt_sequence: u64,
}

impl TransitionLedger {
    /// Create a ledger with the given capacity.
    ///
    /// # Panics
    /// Panics if capacity is zero.
    pub fn new(capacity: NonZeroUsize) -> Self {
        let cap = capacity.get();
        Self {
            capacity: cap,
            history: VecDeque::with_capacity(cap),
            next_receipt_sequence: 0,
        }
    }

    /// Create a ledger with the default capacity (10).
    pub fn with_default_capacity() -> Self {
        Self::new(NonZeroUsize::new(DEFAULT_TRANSITION_LEDGER_CAPACITY).unwrap())
    }

    /// The next expected receipt sequence number.
    pub fn next_receipt_sequence(&self) -> u64 {
        self.next_receipt_sequence
    }

    /// The current number of receipts in the window.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Commit a receipt.  Drops the oldest if at capacity.
    ///
    /// # Panics
    /// Panics if `receipt.receipt_sequence` does not match
    /// the expected next sequence.  This is an internal integrity
    /// check — receipts must be committed in sequence order.
    pub fn commit(&mut self, receipt: TransitionReceipt) {
        assert_eq!(
            receipt.payload.receipt_sequence,
            self.next_receipt_sequence,
            "receipt sequence mismatch: expected {} got {}",
            self.next_receipt_sequence,
            receipt.payload.receipt_sequence,
        );
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(receipt);
        self.next_receipt_sequence += 1;
    }

    /// Iterate over receipts in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &TransitionReceipt> {
        self.history.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ledger::entry::DeterministicReceiptPayload;
    use crate::runtime::scheduling::metadata::Stage;

    fn dummy_receipt(seq: u64) -> TransitionReceipt {
        let payload = DeterministicReceiptPayload {
            schema_version: 1,
            receipt_sequence: seq,
            scheduler_epoch: 0,
            microcycle: 0,
            stage: Stage::Intake,
            command_count: 0,
            commands: vec![],
        };
        TransitionReceipt {
            payload,
            deterministic_digest: crate::runtime::ledger::digest::ReceiptDigest {
                algorithm: "blake3-256-jcs-rfc8785".to_string(),
                hex: "0".repeat(64),
            },
            observed_at_ns: None,
        }
    }

    #[test]
    fn transition_ledger_preserves_exact_capacity() {
        let mut ledger = TransitionLedger::new(NonZeroUsize::new(3).unwrap());
        for i in 0..3 {
            ledger.commit(dummy_receipt(i));
        }
        assert_eq!(ledger.len(), 3);
    }

    #[test]
    fn transition_ledger_discards_oldest_receipt_at_capacity() {
        let mut ledger = TransitionLedger::new(NonZeroUsize::new(3).unwrap());
        for i in 0..5 {
            ledger.commit(dummy_receipt(i));
        }
        assert_eq!(ledger.len(), 3);
        // The oldest (seq 0, 1) should be gone; seq 2, 3, 4 remain.
        let seqs: Vec<u64> = ledger.iter().map(|r| r.payload.receipt_sequence).collect();
        assert_eq!(seqs, vec![2, 3, 4]);
    }

    #[test]
    fn receipt_sequence_is_monotonic_across_eviction() {
        let mut ledger = TransitionLedger::new(NonZeroUsize::new(3).unwrap());
        for i in 0..10 {
            ledger.commit(dummy_receipt(i));
        }
        assert_eq!(ledger.next_receipt_sequence(), 10);
    }

    #[test]
    #[should_panic(expected = "receipt sequence mismatch")]
    fn transition_ledger_rejects_out_of_order_sequence() {
        let mut ledger = TransitionLedger::new(NonZeroUsize::new(5).unwrap());
        ledger.commit(dummy_receipt(42)); // expected 0, panics
    }

    #[test]
    fn empty_stage_does_not_create_receipt() {
        // The ledger itself doesn't enforce "no empty receipts" — the
        // scheduler decides when to commit.  This test verifies the
        // ledger handles zero-command receipts without panic.
        let mut ledger = TransitionLedger::with_default_capacity();
        ledger.commit(dummy_receipt(0));
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.iter().next().unwrap().payload.command_count, 0);
    }
}
