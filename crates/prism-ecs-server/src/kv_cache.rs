//! KV Cache types — ported from compute-core.
//!
//! Provides KVCacheCoordinator, SchedulerRunner, and SchedulerConfig so
//! prism-ecs-server doesn't need compute-core for these type identities.

use serde::{Deserialize, Serialize};

/// Resource ID for KVCacheCoordinator.
pub const KV_CACHE_COORDINATOR_RESOURCE: u32 = 19;

// =============================================================================
// KVCacheCoordinator
// =============================================================================

/// ECS resource — sole state-store facade for all KV page memory.
pub struct KVCacheCoordinator {
    pub slots: u64,
}

impl KVCacheCoordinator {
    pub fn new(context_length: u64) -> Self {
        Self {
            slots: context_length,
        }
    }
}

// =============================================================================
// SchedulerRunner / SchedulerConfig
// =============================================================================

/// Configuration for the continuous batching scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_total_tokens: usize,
    pub max_prefill_batch: usize,
    pub prefill_many_ratio: f64,
    pub pause_threshold: usize,
    pub default_backend_id: u32,
    pub max_num_scheduled_tokens: usize,
    pub kv_cache_length: usize,
    pub kv_cache_pool_bytes: usize,
    pub kv_cache_pages_per_slot: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_total_tokens: 4096,
            max_prefill_batch: 8,
            prefill_many_ratio: 0.5,
            pause_threshold: 2048,
            default_backend_id: 0,
            max_num_scheduled_tokens: 256,
            kv_cache_length: 4096,
            kv_cache_pool_bytes: 256 * 1024 * 1024,
            kv_cache_pages_per_slot: 64,
        }
    }
}

/// Token-budget scheduler runner.
#[derive(Debug, Clone)]
pub struct SchedulerRunner {
    pub scheduled: usize,
    pub running: usize,
    pub waiting: usize,
}

impl SchedulerRunner {
    pub fn new(_config: &SchedulerConfig) -> Self {
        Self {
            scheduled: 0,
            running: 0,
            waiting: 0,
        }
    }
    pub fn step(&mut self) -> Result<(), String> {
        Ok(())
    }
}
