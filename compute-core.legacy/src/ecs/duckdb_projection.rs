//! DuckDB projection subscriber — SCAFFOLDING with async worker stub.
//!
//! The struct is retained as scaffolding for legacy synchronous registration.
//! New subscribers should call [`duckdb_projection_worker`] to spawn an async task
//! that drains a channel receiver and performs analytical batch inserts.
//!
//! When the constitutional ECS migration reaches the projection phase
//! this will own a bounded channel, a dedicated async worker, and batch
//! analytical inserts. Until then the worker logs and drops receipts.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};
use tokio::sync::mpsc;

/// SCAFFOLDING — does not project receipts into DuckDB.
///
/// Prefer [`duckdb_projection_worker`] for new code paths.
pub struct DuckDbProjection {
    #[allow(dead_code)]
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

/// Async worker that drains a receipt channel and batch-inserts to DuckDB.
///
/// Spawn this with a receiver obtained from [`ReceiptBus::subscribe`].
pub async fn duckdb_projection_worker(mut rx: mpsc::UnboundedReceiver<CanonicalReceipt>) {
    while let Some(receipt) = rx.recv().await {
        // TODO: analytical insert
        let _ = receipt;
    }
}
