//! Unified memory telemetry data types.
//!
//! This module owns the canonical authority for the engine-independent
//! `UnifiedMemoryTelemetry` / `CandleAllocatorStats` data types — the
//! point-in-time snapshot shape that aggregates memory subsystem
//! stats. The sampling function that populates the snapshot
//! (`sample_unified_memory`) stays engine-side at
//! `compute-core/src/ecs/memory_impl/telemetry_impl.rs` because it
//! depends on engine-internal `IosurfaceAllocator` and
//! `worker_memory` types.

/// Stats from the candle Metal allocator.
///
/// Candle manages Metal buffers through a bucket-allocator scheme.
/// These fields report the current active, cached, and cumulative
/// total allocations from that subsystem. All values are in bytes.
///
/// When candle has not been initialised or its allocator is not
/// wired into the unified pool, all fields will be 0.
#[derive(Debug, Clone, Default)]
pub struct CandleAllocatorStats {
    /// Bytes currently in use by live candle tensors.
    pub active_buffer_bytes: u64,
    /// Bytes held in the candle bucket cache (available for reuse).
    pub cached_buffer_bytes: u64,
    /// Cumulative bytes allocated over the lifetime of the candle
    /// allocator.
    pub total_allocation_bytes: u64,
}

/// A single point-in-time snapshot of every memory subsystem the
/// engine manages, plus process-level counters.
///
/// Use the engine-side `sample_unified_memory` to populate a
/// `UnifiedMachineProfile` + `UnifiedMemoryTelemetry` pair.
#[derive(Debug, Clone)]
pub struct UnifiedMemoryTelemetry {
    /// Total bytes allocated through the IOSurface allocator pool.
    pub iosurface_allocator_bytes: u64,
    /// Memory pressure of the IOSurface pool
    /// (`total_allocated / max_pool`).
    pub iosurface_pressure: f64,
    /// MLX Metal allocator snapshot (active, cache, peak).
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
    /// Candle bucket allocator snapshot.
    pub candle: CandleAllocatorStats,
    /// Cumulative page faults (pageins) for the current process.
    pub page_faults: u64,
}

impl Default for UnifiedMemoryTelemetry {
    fn default() -> Self {
        Self {
            iosurface_allocator_bytes: 0,
            iosurface_pressure: 0.0,
            mlx_active_bytes: 0,
            mlx_cache_bytes: 0,
            mlx_peak_bytes: 0,
            candle: CandleAllocatorStats::default(),
            page_faults: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_defaults_are_zero() {
        let stats = CandleAllocatorStats::default();
        assert_eq!(stats.active_buffer_bytes, 0);
        assert_eq!(stats.cached_buffer_bytes, 0);
        assert_eq!(stats.total_allocation_bytes, 0);
    }

    #[test]
    fn unified_telemetry_default_is_well_defined() {
        let t = UnifiedMemoryTelemetry::default();
        assert_eq!(t.iosurface_allocator_bytes, 0);
        assert!(t.iosurface_pressure.is_finite());
        assert_eq!(t.iosurface_pressure, 0.0);
        assert_eq!(t.mlx_active_bytes, 0);
        assert_eq!(t.mlx_cache_bytes, 0);
        assert_eq!(t.mlx_peak_bytes, 0);
        assert_eq!(t.page_faults, 0);
    }
}
