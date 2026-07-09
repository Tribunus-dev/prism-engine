use serde::{Deserialize, Serialize};

/// Receipt produced by schema validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateStoreValidationReceipt {
    pub schema_name: String,
    pub store_count: u32,
    pub total_max_bytes: u64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub valid: bool,
}

/// Receipt produced by a KV token-append operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvAppendReceipt {
    pub store_id: String,
    pub span_id: String,
    pub epoch_id: u64,
    pub pages_allocated: u32,
    pub total_pages: u32,
    pub total_bytes_after: u64,
    pub memory_ok: bool,
}

/// Receipt produced by a KV read-window lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvReadReceipt {
    pub store_id: String,
    pub token_start: u32,
    pub token_count: u32,
    pub epoch_id: u64,
    pub pages_resolved: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub access_granted: bool,
}
