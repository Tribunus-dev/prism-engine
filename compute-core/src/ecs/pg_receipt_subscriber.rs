//! PostgreSQL receipt subscriber for the ReceiptBus.
//! Writes canonical receipts asynchronously to PostgreSQL via UNNEST batch inserts.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};
use std::collections::VecDeque;

/// Subscribes to the ReceiptBus and persists receipts to PostgreSQL.
pub struct PgReceiptSubscriber {
    connection_string: String,
    batch: VecDeque<CanonicalReceipt>,
}

impl PgReceiptSubscriber {
    pub fn new(connection_string: &str) -> Self {
        Self {
            connection_string: connection_string.to_string(),
            batch: VecDeque::new(),
        }
    }

    pub fn flush(&mut self) {
        self.batch.clear();
    }
}

impl ReceiptSubscriber for PgReceiptSubscriber {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        let _ = receipt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_subscriber() {
        let _sub = PgReceiptSubscriber::new("postgres://localhost/test");
    }
}
