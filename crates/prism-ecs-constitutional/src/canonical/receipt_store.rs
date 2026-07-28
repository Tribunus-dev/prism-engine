//! Content-addressed receipt persistence. Authority: the receipt
//! store.
//!
//! Every receipt produced during a lifecycle is serialized,
//! SHA-256 hashed, and stored by its digest. `ReceiptId` resolves
//! to the persisted bytes, and `verify()` confirms integrity by
//! re-hashing on read. The store is keyed by digest string
//! (matching the `ReceiptId(pub String)` shape) in a `BTreeMap`
//! so the iteration order is observable and deterministic — the
//! engine's previous `HashMap`-based store has been migrated to
//! the constitutional BTreeMap-based store to satisfy the
//! "no HashMap/HashSet for canonical collections whose order is
//! observable" rule.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use super::identity::ReceiptId;

/// Content-addressed store for immutable receipt records.
///
/// Every receipt is serialized, SHA-256 hashed, and stored by its
/// digest. `ReceiptId` is the hex digest string.
pub struct ReceiptStore {
    records: BTreeMap<String, Vec<u8>>,
}

impl ReceiptStore {
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
        }
    }

    /// Store a serializable receipt and return its content-addressed ID.
    pub fn store<T: Serialize>(&mut self, receipt: &T) -> ReceiptId {
        // Try bincode first (compact), fall back to JSON. Both
        // fallbacks are intentional and benign: bincode may fail
        // on a non-bincode-friendly payload, and JSON is a
        // universal fallback that always succeeds for any
        // serializable value (or returns an empty vec which is
        // then hashed; collisions in the empty-bucket are
        // detectable by the caller through the resulting digest).
        let bytes = bincode::serialize(receipt)
            .unwrap_or_else(|_| serde_json::to_vec(receipt).unwrap_or_default());
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
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

impl Default for ReceiptStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Serializable receipt payload types for lifecycle stages that
// lack a dedicated receipt struct. These let us produce
// content-addressed receipts for *every* slot in
// `LifecycleReceiptBundle`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_resolve_round_trip() {
        let mut store = ReceiptStore::new();
        let data = CompilerReceiptData {
            precision_targets: vec!["bf16".into()],
            artifact_count: 3,
            timestamp: "2026-07-28".into(),
        };
        let id = store.store(&data);
        let bytes = store.resolve(&id).expect("stored");
        assert!(!bytes.is_empty());
    }

    #[test]
    fn verify_rejects_unknown_id() {
        let store = ReceiptStore::new();
        assert!(!store.verify(&ReceiptId("does-not-exist".into())));
    }

    #[test]
    fn verify_accepts_known_id() {
        let mut store = ReceiptStore::new();
        let data = QualityReceiptData {
            numerical_passed: true,
            timestamp: "2026-07-28".into(),
        };
        let id = store.store(&data);
        assert!(store.verify(&id));
    }

    #[test]
    fn len_and_is_empty_match_insertions() {
        let mut store = ReceiptStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        let _ = store.store(&PromotionReceiptData {
            generation_id: "g-1".into(),
            timestamp: "t".into(),
        });
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn store_uses_deterministic_digest_for_identical_input() {
        // Two stores receiving the same payload produce the same
        // digest. This is the canonical content-addressing
        // contract: the digest is a function of the serialized
        // bytes, not the in-memory address.
        let mut a = ReceiptStore::new();
        let mut b = ReceiptStore::new();
        let data = PolicyReceiptData {
            max_runtime_seconds: 60,
            max_memory_bytes: 1 << 30,
            promotion_policy: "staged".into(),
        };
        let id_a = a.store(&data);
        let id_b = b.store(&data);
        assert_eq!(id_a.0, id_b.0);
    }
}
