use crate::ecs::legacy_compute_image_core::kv_plan::KvCodec;
use serde::{Deserialize, Serialize};

/// Top-level state store schema — a collection of store declarations plus
/// access and eviction policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateStoreSchema {
    pub stores: Vec<StateStoreDecl>,
    pub access_policies: Vec<StateAccessPolicy>,
    pub eviction_policies: Vec<StateEvictionPolicy>,
}

/// A single store declaration within the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateStoreDecl {
    pub store_id: String,
    pub store_kind: String,
    pub owner_region: String,
    pub dtype: String,
    pub max_bytes: u64,
    pub persistence: String,
}

/// Access policy binding a set of regions to a kind of access on a store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAccessPolicy {
    pub policy_id: String,
    pub store_id: String,
    pub allowed_regions: Vec<String>,
    pub access: AccessKind,
}

/// Kind of access granted by a policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
    ReadWrite,
}

/// Eviction policy for a store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEvictionPolicy {
    pub store_id: String,
    pub eviction_kind: EvictionKind,
}

/// Eviction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionKind {
    NoEviction,
    Lru,
    PinProtected,
}

/// KV-cache-specific store declaration — dimensions, layout, precision, residency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheStoreDecl {
    pub store_id: String,
    pub model_partition_id: String,
    pub layer_count: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub head_dim: u32,
    pub max_sequence_len: u32,
    pub cache_layout: KvCacheLayout,
    pub codec_policy: KvCodecPolicy,
    pub residency_policy: KvResidencyPolicy,
}

/// Layout strategy for a KV cache store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KvCacheLayout {
    /// Paged layout: each page holds `page_tokens` tokens and byte-aligns
    /// to `alignment_bytes`.
    PagedLayerHead {
        page_tokens: u32,
        alignment_bytes: u32,
    },
}

/// KV codec — replaces string-typed precision policies.
/// The codec drives page sizing, runtime construction, and attention dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCodecPolicy {
    pub codec: KvCodec,
}

/// Residency policy — max active spans and pin support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvResidencyPolicy {
    pub max_active_spans: u32,
    pub span_pin_supported: bool,
}
