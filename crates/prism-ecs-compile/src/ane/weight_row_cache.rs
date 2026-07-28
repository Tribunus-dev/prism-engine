//! Weight-row cache — config for the ANE LM-head SRAM-resident
//! weight-row prefetch path.
//!
//! Authority: the canonical weight-row cache config + slot-allocator
//! re-export.
//!
//! The IOSurface-backed arena storage and the FP16 row read/write
//! paths are engine-coupled. The engine's `legacy_ane/weight_row_cache.rs`
//! wraps a `WeightRowCache` from this surface with a Core ML backend
//! and an `Arena`; the constitutional surface provides the config +
//! the LRU eviction policy ([`crate::ane::SlotAllocator`]).
//!
//! # Architecture
//!
//! A Core ML model serves as the SRAM container: its parameter
//! buffers hold the pre-loaded weight rows as FP16 values. The
//! `hybrid_lm_head` method computes the full matmul on GPU and
//! overwrites logits for cached tokens with fast-path values
//! computed from the cached rows.
//!
//! # Sizing
//!
//! ANE has ~2 MB of private SRAM. At FP16 (2 bytes per element) and
//! `hidden_size=3840`, each row is 7680 bytes. ~2 MB / 7680 bytes ≈
//! 272 rows. We default to 256 rows for a comfortable margin.

use crate::ane::slot_allocator::SlotAllocator;
use crate::ane::AneError;

/// Backend-agnostic config for [`WeightRowCache`].
///
/// `max_rows` is the maximum number of weight rows cached in ANE
/// SRAM (e.g. 256). `hidden_size` is the model's hidden-state
/// dimension. `bits` is the LM head quantisation width (e.g. 4 for
/// int4, 8 for int8).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightRowCacheConfig {
    /// Maximum number of cached rows.
    pub max_rows: u32,
    /// Hidden-state dimension.
    pub hidden_size: u32,
    /// LM head quantisation bit width.
    pub bits: u8,
}

impl WeightRowCacheConfig {
    /// Create a new config.
    pub fn new(max_rows: u32, hidden_size: u32, bits: u8) -> Self {
        Self {
            max_rows,
            hidden_size,
            bits,
        }
    }
}

/// Backend trait that performs the actual row extraction and
/// SRAM-backed storage for the weight-row cache.
///
/// The engine's `legacy_ane/` provides a `CoreMLWeightRowCacheBackend`
/// that owns an `Arena` of IOSurface-backed memory; other backends
/// (CPU-only simulators for tests) can implement this trait and
/// produce identical dot-product results.
pub trait WeightRowCacheBackend {
    /// Look up a cached logit value for `token_id`.
    ///
    /// Returns `None` if the token is not in the cache. The engine's
    /// backend reads the cached FP16 row from IOSurface and computes
    /// the dot product with `hidden_state`; a CPU-only test backend
    /// can use a `Vec<f32>` instead.
    fn read_logit(
        &self,
        token_id: u32,
        hidden_state: &[f32],
    ) -> Option<f32>;
}

/// ANE-resident LM-head weight-row cache.
///
/// Owns the LRU slot allocator and the backend handle. Construction
/// is backend-driven — the engine's `legacy_ane/weight_row_cache.rs`
/// loads a Core ML model parameter buffer; tests can use a CPU-only
/// simulator.
pub struct WeightRowCache {
    /// Public config (read-only after construction).
    pub config: WeightRowCacheConfig,
    /// LRU slot allocator — re-exported from `crate::ane::slot_allocator`.
    pub slot_allocator: SlotAllocator,
    /// Backend that performs the actual SRAM-backed read.
    backend: Box<dyn WeightRowCacheBackend>,
    /// Token IDs currently cached, indexed by slot.
    pub cached_token_ids: Vec<u32>,
}

impl WeightRowCache {
    /// Construct a new weight-row cache with the given config and backend.
    pub fn new(
        config: WeightRowCacheConfig,
        backend: Box<dyn WeightRowCacheBackend>,
    ) -> Result<Self, AneError> {
        let max_rows = config.max_rows;
        Ok(Self {
            slot_allocator: SlotAllocator::new(max_rows),
            cached_token_ids: vec![0u32; max_rows as usize],
            config,
            backend,
        })
    }

    /// Read a single cached logit value for `token_id`.
    ///
    /// Returns `None` if the token is not in the cache. The dot
    /// product with `hidden_state` is computed by the backend.
    pub fn read_logit(&self, token_id: u32, hidden_state: &[f32]) -> Option<f32> {
        if self.slot_allocator.lookup(token_id).is_none() {
            return None;
        }
        self.backend.read_logit(token_id, hidden_state)
    }

    /// Report the current occupancy as a `(used, max)` pair.
    pub fn occupancy(&self) -> (usize, u32) {
        (self.slot_allocator.occupied_count(), self.config.max_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Test backend backed by a `HashMap<token_id, row>`.
    struct MapBackend {
        rows: HashMap<u32, Vec<f32>>,
    }

    impl WeightRowCacheBackend for MapBackend {
        fn read_logit(&self, token_id: u32, hidden_state: &[f32]) -> Option<f32> {
            self.rows.get(&token_id).map(|row| {
                row.iter()
                    .zip(hidden_state.iter())
                    .map(|(w, h)| w * h)
                    .sum()
            })
        }
    }

    #[test]
    fn empty_cache_returns_none() {
        let config = WeightRowCacheConfig::new(4, 2, 4);
        let backend = Box::new(MapBackend {
            rows: HashMap::new(),
        });
        let cache = WeightRowCache::new(config, backend).unwrap();
        assert_eq!(cache.read_logit(42, &[1.0, 2.0]), None);
    }

    #[test]
    fn cache_reports_occupancy() {
        let config = WeightRowCacheConfig::new(4, 2, 4);
        let mut rows = HashMap::new();
        rows.insert(1u32, vec![0.5, 0.5]);
        let backend = Box::new(MapBackend { rows });
        let cache = WeightRowCache::new(config, backend).unwrap();
        let (used, max) = cache.occupancy();
        assert_eq!(used, 0);
        assert_eq!(max, 4);
    }
}
