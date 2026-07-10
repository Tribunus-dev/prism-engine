use crate::ecs::{CompEntity, Component};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryDomain {
    DeviceLocal,
    HostVisible,
    HostCoherent,
    HostCached,
    Unified,
}
impl Component for MemoryDomain {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryPool {
    pub policy: PoolPolicy,
    pub pool_id: u32,
    pub total_bytes: u64,
    pub used_bytes: u64,
}
impl Component for MemoryPool {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolPolicy {
    Arena,
    TLSF,
    FixedBlock,
    Dedicated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferLifetime {
    pub alloc_epoch: u64,
    pub free_epoch: u64,
    pub causal_death_frontier: Option<(u64, u64)>, // (queue_axis, epoch)
}
impl Component for BufferLifetime {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferBinding {
    pub slot: u32,
    pub buffer: CompEntity,
    pub offset: u64,
    pub size: u64,
}
impl Component for BufferBinding {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub total_bytes: u64,
    pub scratch_bytes: u64,
    pub kv_cache_bytes: u64,
    pub weight_bytes: u64,
}
impl Component for MemoryBudget {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchConfig {
    pub per_dispatch_scratch: u64,
    pub persistent_scratch: u64,
    pub arena_policy: PoolPolicy,
}
impl Component for ScratchConfig {}
