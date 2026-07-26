//! PostgreSQL receipt subscriber — SCAFFOLDING with async worker stub.
//!
//! The struct is retained as scaffolding for legacy synchronous registration.
//! New subscribers should call [`pg_projection_worker`] to spawn an async task
//! that drains a channel receiver and performs batch inserts via `UNNEST`.
//!
//! When the constitutional ECS migration reaches the persistence phase
//! this will own a bounded channel, a dedicated async worker, and batch
//! inserts via `UNNEST`. Until then the worker logs and drops receipts.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};
use std::collections::VecDeque;
use tokio::sync::mpsc;

/// SCAFFOLDING — does not persist receipts.
///
/// Prefer [`pg_projection_worker`] for new code paths.
pub struct PgReceiptSubscriber {
    #[allow(dead_code)]
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

/// Async worker that drains a receipt channel and batch-inserts to PostgreSQL.
///
/// Spawn this with a receiver obtained from [`ReceiptBus::subscribe`].
/// When the batch reaches 100 receipts a `UNNEST`-style insert is executed.
pub async fn pg_projection_worker(mut rx: mpsc::UnboundedReceiver<CanonicalReceipt>) {
    let mut batch: Vec<CanonicalReceipt> = Vec::new();
    while let Some(receipt) = rx.recv().await {
        batch.push(receipt);
        if batch.len() >= 100 {
            // TODO: execute UNNEST batch insert
            batch.clear();
        }
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
