//! PostgreSQL receipt subscriber — SCAFFOLDING, not yet operational.
//!
//! The stub stores a connection string and silently drops every receipt.
//! When the constitutional ECS migration reaches the persistence phase
//! this will own a bounded channel, a dedicated async worker, and batch
//! inserts via `UNNEST`. Until then it returns an error from
//! [`check_configured`](PgReceiptSubscriber::check_configured) and must
//! not be relied upon for durable receipt storage.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};
use std::collections::VecDeque;

/// SCAFFOLDING — does not persist receipts.
pub struct PgReceiptSubscriber {
    connection_string: String,
    batch: VecDeque<CanonicalReceipt>,
    configured: bool,
}

impl PgReceiptSubscriber {
    /// Create a new subscriber.
    ///
    /// Pass `true` for `configured` after a live PostgreSQL connection
    /// has been established. Until then, all receipts are silently dropped.
    pub fn new(connection_string: &str, configured: bool) -> Self {
        Self {
            connection_string: connection_string.to_string(),
            batch: VecDeque::new(),
            configured,
        }
    }

    /// Returns `Ok(())` only when a database connection is configured.
    pub fn check_configured(&self) -> Result<(), &'static str> {
        if self.configured {
            Ok(())
        } else {
            Err("PgReceiptSubscriber: not configured — receipts are silently dropped")
        }
    }

    pub fn flush(&mut self) {
        if !self.configured {
            return;
        }
        // TODO: flush to database when persistence is wired
        self.batch.clear();
    }
}

impl ReceiptSubscriber for PgReceiptSubscriber {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        if !self.configured {
            return;
        }
        // TODO: push receipt into bounded async channel for batch persistence
        let _ = receipt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_configured() {
        let sub = PgReceiptSubscriber::new("postgres://localhost/test", false);
        assert!(sub.check_configured().is_err());
    }

    #[test]
    fn test_configured() {
        let sub = PgReceiptSubscriber::new("postgres://localhost/test", true);
        assert!(sub.check_configured().is_ok());
    }
}
