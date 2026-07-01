//! TransitionLedgerResource — scheduler-owned ledger exposed through the resource API.
//!
//! Only the scheduler commit path may mutate the ledger.  Systems may read
//! the ledger through the resource API but must not write to it.

use std::sync::Mutex;

use crate::runtime::ledger::ledger::TransitionLedger;
use crate::runtime::ledger::entry::TransitionReceipt;

/// Resource wrapper for the rolling transition ledger.
///
/// Systems access the ledger read-only through `get_resource::<TransitionLedgerResource>()`.
/// The scheduler commit path mutates it through the inner Mutex.
pub struct TransitionLedgerResource {
    inner: Mutex<TransitionLedger>,
}

impl TransitionLedgerResource {
    /// Create from an existing ledger.
    pub fn new(ledger: TransitionLedger) -> Self {
        Self {
            inner: Mutex::new(ledger),
        }
    }

    /// Create with default capacity (10).
    pub fn with_default_capacity() -> Self {
        Self::new(TransitionLedger::with_default_capacity())
    }

    /// Commit a receipt (scheduler-only).
    pub fn commit(&self, receipt: TransitionReceipt) {
        self.inner.lock().unwrap().commit(receipt);
    }

    /// The next expected receipt sequence number.
    pub fn next_receipt_sequence(&self) -> u64 {
        self.inner.lock().unwrap().next_receipt_sequence()
    }

    /// Current number of receipts in the window.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// Export the current window as JSONL.
    pub fn export_jsonl(&self) -> Result<String, crate::runtime::ledger::error::LedgerExportError> {
        use crate::runtime::ledger::error::LedgerExportError;
        let ledger = self.inner.lock().unwrap();
        let mut output = Vec::new();
        for receipt in ledger.iter() {
            serde_json::to_writer(&mut output, receipt)
                .map_err(LedgerExportError::Serialization)?;
            output.push(b'\n');
        }
        Ok(String::from_utf8(output).map_err(|e| LedgerExportError::Serialization(
            serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        ))?)
    }
}

unsafe impl Send for TransitionLedgerResource {}
unsafe impl Sync for TransitionLedgerResource {}
