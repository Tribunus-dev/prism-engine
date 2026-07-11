//! Valkey (Redis-compatible) projection subscriber — SCAFFOLDING, not yet operational.
//!
//! The stub stores a connection string and silently drops every receipt.
//! When the constitutional ECS migration reaches the projection phase
//! this will own a bounded channel, a dedicated async worker, and cache
//! state for fast dashboard queries. Until then it returns an error from
//! [`check_configured`](ValkeyProjection::check_configured).

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};

/// SCAFFOLDING — does not cache receipt state in Valkey.
pub struct ValkeyProjection {
    connection_string: String,
    configured: bool,
}

impl ValkeyProjection {
    /// Create a new projection stub.
    ///
    /// Pass `true` for `configured` when a live Valkey connection has
    /// been established.
    pub fn new(connection_string: &str, configured: bool) -> Self {
        Self {
            connection_string: connection_string.to_string(),
            configured,
        }
    }

    /// Returns `Ok(())` only when a database connection is configured.
    pub fn check_configured(&self) -> Result<(), &'static str> {
        if self.configured {
            Ok(())
        } else {
            Err("ValkeyProjection: not configured — receipts are silently dropped")
        }
    }
}

impl ReceiptSubscriber for ValkeyProjection {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        if !self.configured {
            return;
        }
        // TODO: push receipt into bounded async channel for batch cache updates
        let _ = receipt;
    }
}
