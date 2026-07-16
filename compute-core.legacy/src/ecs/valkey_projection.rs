//! Valkey (Redis-compatible) projection subscriber — SCAFFOLDING with async worker stub.
//!
//! The struct is retained as scaffolding for legacy synchronous registration.
//! New subscribers should call [`valkey_projection_worker`] to spawn an async task
//! that drains a channel receiver and updates cache state.
//!
//! When the constitutional ECS migration reaches the projection phase
//! this will own a bounded channel, a dedicated async worker, and cache
//! state for fast dashboard queries. Until then the worker logs and drops receipts.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};
use tokio::sync::mpsc;

/// SCAFFOLDING — does not cache receipt state in Valkey.
///
/// Prefer [`valkey_projection_worker`] for new code paths.
pub struct ValkeyProjection {
    #[allow(dead_code)]
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

/// Async worker that drains a receipt channel and updates Valkey cache state.
///
/// Spawn this with a receiver obtained from [`ReceiptBus::subscribe`].
pub async fn valkey_projection_worker(mut rx: mpsc::UnboundedReceiver<CanonicalReceipt>) {
    while let Some(receipt) = rx.recv().await {
        // TODO: cache update
        let _ = receipt;
    }
}
