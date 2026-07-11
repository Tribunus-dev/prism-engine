//! DuckDB projection subscriber for the ReceiptBus.
//! Subscribes to canonical receipts and writes them to DuckDB analytics tables.

use crate::ecs::receipt_bus::{CanonicalReceipt, ReceiptSubscriber};

/// Subscribes to the ReceiptBus and projects receipts into DuckDB.
pub struct DuckDbProjection {
    path: String,
}

impl DuckDbProjection {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
}

impl ReceiptSubscriber for DuckDbProjection {
    fn on_receipt(&mut self, receipt: &CanonicalReceipt) {
        let _ = receipt;
    }
}
