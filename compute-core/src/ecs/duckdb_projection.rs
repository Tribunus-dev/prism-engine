//! DuckDB projection subscriber — SCAFFOLDING, not yet operational.
//!
//! The stub stores a database path and silently drops every receipt.
//! When the constitutional ECS migration reaches the projection phase
//! this will own a bounded channel, a dedicated async worker, and batch
//! analytical inserts. Until then it returns an error from
//! [`check_configured`](DuckDbProjection::check_configured).

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};

/// SCAFFOLDING — does not project receipts into DuckDB.
pub struct DuckDbProjection {
    path: String,
    configured: bool,
}

impl DuckDbProjection {
    /// Create a new projection stub.
    ///
    /// Pass `true` for `configured` when a live DuckDB connection has
    /// been established.
    pub fn new(path: &str, configured: bool) -> Self {
        Self {
            path: path.to_string(),
            configured,
        }
    }

    /// Returns `Ok(())` only when a database connection is configured.
    pub fn check_configured(&self) -> Result<(), &'static str> {
        if self.configured {
            Ok(())
        } else {
            Err("DuckDbProjection: not configured — receipts are silently dropped")
        }
    }
}

impl ReceiptSubscriber for DuckDbProjection {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        if !self.configured {
            return;
        }
        // TODO: push receipt into bounded async channel for batch persistence
        let _ = receipt;
    }
}
