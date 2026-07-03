//! ANE platform-specific page migration policy.
//!
//! Provides [`AnePageMigrationPolicy`], which implements [`PageMigrationPolicy`]
//! by wrapping [`AneCompressor`] for compress/decompress operations during
//! promotion and demotion between tiers.
//!
//! Tier mapping (old ANE → platform-agnostic):
//! - L1AneSram  → [`KVCacheTier::L0Device`]
//! - L2Iosurface → [`KVCacheTier::L1Shared`]
//! - L3DramHeap  → [`KVCacheTier::L2System`]
//! - L4Disk      → [`KVCacheTier::L3Disk`]
//!
//! Data format per tier:
//! - L0Device: 3.5-bit packed (device-local)
//! - L1Shared: 3.5-bit packed (shared memory, e.g. IOSurface)
//! - L2System: 2-bit packed (host DRAM)
//! - L3Disk:   2-bit packed (disk, no resident data)

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::runtime::resources::kv_cache_coordinator::{
    KVCacheTier, PageBacking, PageMigrationPolicy, TiersPage,
};

use super::kv_decompress_program::AneCompressor;

/// ANE platform-specific page migration policy.
///
/// Uses [`AneCompressor`] to convert between 3.5-bit (L0Device/L1Shared) and
/// 2-bit (L2System) formats during promotion and demotion.
///
/// L2System→L3Disk eviction is handled by the generic
/// [`PageMigrationService::check_and_evict`], which persists the data to disk
/// before dropping it. This policy manages L0Device↔L1Shared↔L2System transitions
/// plus L3Disk→L2System promotion (prefetch).
pub struct AnePageMigrationPolicy {
    /// The ANE compressor/decompressor for tier transitions.
    pub compressor: Arc<AneCompressor>,
    /// Head dimension for KV cache pages.
    pub head_dim: u32,
    /// Number of KV heads per layer.
    pub n_kv_heads: u32,
    /// Duration of inactivity before a page is considered cold (demotion candidate).
    pub cold_threshold: Duration,
    /// Duration since last access for a page to be considered hot (promotion candidate).
    pub hot_threshold: Duration,
}

impl AnePageMigrationPolicy {
    pub fn new(
        compressor: Arc<AneCompressor>,
        head_dim: u32,
        n_kv_heads: u32,
        cold_threshold: Duration,
        hot_threshold: Duration,
    ) -> Self {
        Self { compressor, head_dim, n_kv_heads, cold_threshold, hot_threshold }
    }

    #[inline]
    fn page_age(&self, page: &TiersPage, now: Instant) -> Duration {
        now.duration_since(page.last_access)
    }

    /// Promote from L2System (2-bit) to L1Shared (3.5-bit).
    /// Pipeline: 2-bit → decompress (L3) → FP16 → compress (L2) → 3.5-bit.
    fn promote_to_shared(&self, page: &mut TiersPage) -> Result<(), String> {
        let packed = page.data.as_ref()
            .ok_or_else(|| "promote_to_shared: no data".to_string())?;
        let fp16 = self.compressor.decompress_from_l3(packed, self.head_dim, self.n_kv_heads)?;
        let compressed = self.compressor.compress_to_l2(&fp16, self.head_dim, self.n_kv_heads, 3)?;
        let bs = compressed.len() as u64;
        page.data = Some(compressed);
        page.backing = PageBacking::SharedBuffer { handle: 0, byte_size: bs, domain_tag: 0 };
        page.current_tier = KVCacheTier::L1Shared;
        Ok(())
    }

    /// Promote from L1Shared (3.5-bit) to L0Device (3.5-bit).
    /// Same format, just change backing.
    fn promote_to_device(&self, page: &mut TiersPage) -> Result<(), String> {
        let packed = page.data.as_ref()
            .ok_or_else(|| "promote_to_device: no data".to_string())?;
        let bs = packed.len() as u64;
        page.backing = PageBacking::DeviceBuffer { handle: 0, byte_size: bs };
        page.current_tier = KVCacheTier::L0Device;
        Ok(())
    }

    /// Demote from L0Device (3.5-bit) to L1Shared (3.5-bit).
    /// Same format, just change backing.
    fn demote_to_shared(&self, page: &mut TiersPage) -> Result<(), String> {
        let packed = page.data.as_ref()
            .ok_or_else(|| "demote_to_shared: no data".to_string())?;
        let bs = packed.len() as u64;
        page.backing = PageBacking::SharedBuffer { handle: 0, byte_size: bs, domain_tag: 0 };
        page.current_tier = KVCacheTier::L1Shared;
        Ok(())
    }

    /// Demote from L1Shared (3.5-bit) to L2System (2-bit).
    /// Pipeline: 3.5-bit → decompress (L2) → FP16 → compress (L3) → 2-bit.
    fn demote_to_system(&self, page: &mut TiersPage) -> Result<(), String> {
        let packed = page.data.as_ref()
            .ok_or_else(|| "demote_to_system: no data".to_string())?;
        let fp16 = self.compressor.decompress_from_l2(packed, self.head_dim, self.n_kv_heads, 3)?;
        let cold = self.compressor.compress_to_l3(&fp16, self.head_dim, self.n_kv_heads)?;
        let bs = cold.len() as u64;
        page.data = Some(cold);
        page.backing = PageBacking::SystemHeap { byte_size: bs };
        page.current_tier = KVCacheTier::L2System;
        Ok(())
    }

    /// Promote from L3Disk (2-bit, data loaded by prefetch_predicted) to L2System.
    fn promote_from_disk(&self, page: &mut TiersPage) -> Result<(), String> {
        if page.data.is_none() {
            return Err("promote_from_disk: page not loaded from disk".to_string());
        }
        let bs = page.data.as_ref().map_or(0, |d| d.len() as u64);
        page.backing = PageBacking::SystemHeap { byte_size: bs };
        page.current_tier = KVCacheTier::L2System;
        Ok(())
    }
}

impl PageMigrationPolicy for AnePageMigrationPolicy {
    fn evaluate_tick(&self, page: &mut TiersPage, now: Instant) -> Result<(), String> {
        let age = self.page_age(page, now);
        match page.current_tier {
            KVCacheTier::L0Device => {
                if age > self.cold_threshold {
                    self.demote_to_shared(page)?;
                }
            }
            KVCacheTier::L1Shared => {
                if age < self.hot_threshold {
                    self.promote_to_device(page)?;
                } else if age > self.cold_threshold {
                    self.demote_to_system(page)?;
                }
            }
            KVCacheTier::L2System => {
                if age < self.hot_threshold {
                    self.promote_to_shared(page)?;
                }
            }
            KVCacheTier::L3Disk => {
                if age < self.hot_threshold {
                    self.promote_from_disk(page)?;
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ane_page_migration_policy"
    }
}
