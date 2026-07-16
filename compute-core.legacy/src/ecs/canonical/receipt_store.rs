//! Content-addressed receipt persistence.
//!
//! Every receipt produced during a lifecycle is serialized, SHA-256 hashed,
//! and stored by its digest. ReceiptId resolves to the persisted bytes,
//! and verify() confirms integrity by re-hashing on read.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::ecs::canonical::identity::ReceiptId;

/// Content-addressed store for immutable receipt records.
///
/// Every receipt is serialized, SHA-256 hashed, and stored by its digest.
/// `ReceiptId` is the hex digest string.
pub struct ReceiptStore {
    records: HashMap<String, Vec<u8>>,
}

impl ReceiptStore {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Store a serializable receipt and return its content-addressed ID.
    pub fn store<T: Serialize>(&mut self, receipt: &T) -> ReceiptId {
        // Try bincode first (compact), fall back to JSON.
        let bytes = bincode::serialize(receipt)
            .unwrap_or_else(|_| serde_json::to_vec(receipt).unwrap_or_default());
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        // hex crate is not available; format! is the canonical approach.
        let digest = format!("{:x}", hasher.finalize());
        self.records.insert(digest.clone(), bytes);
        ReceiptId(digest)
    }

    /// Resolve a receipt by ID.
    pub fn resolve(&self, id: &ReceiptId) -> Option<&[u8]> {
        self.records.get(&id.0).map(|v| v.as_slice())
    }

    /// Check if a receipt ID resolves and verify its digest matches.
    pub fn verify(&self, id: &ReceiptId) -> bool {
        self.records.get(&id.0).map_or(false, |bytes| {
            let mut hasher = Sha256::new();
            hasher.update(bytes);
            format!("{:x}", hasher.finalize()) == id.0
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Serializable receipt payload types for lifecycle stages that lack a
// dedicated receipt struct. These let us produce content-addressed receipts
// for *every* slot in LifecycleReceiptBundle.
// ---------------------------------------------------------------------------

/// Payload stored for the compiler-stage receipt.
#[derive(Serialize)]
pub struct CompilerReceiptData {
    pub precision_targets: Vec<String>,
    pub artifact_count: usize,
    pub timestamp: String,
}

/// Payload stored for the quality-stage receipt.
#[derive(Serialize)]
pub struct QualityReceiptData {
    pub numerical_passed: bool,
    pub timestamp: String,
}

/// Payload stored for the policy receipt.
#[derive(Serialize)]
pub struct PolicyReceiptData {
    pub max_runtime_seconds: u64,
    pub max_memory_bytes: u64,
    pub promotion_policy: String,
}

/// Payload stored for the promotion receipt.
#[derive(Serialize)]
pub struct PromotionReceiptData {
    pub generation_id: String,
    pub timestamp: String,
}
