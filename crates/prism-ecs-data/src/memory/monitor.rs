//! Memory statistics and per-process pressure polling.
//!
//! This module owns the canonical authority for the `MemoryStats`
//! snapshot (RSS, total RAM, virtual memory, swap usage) and the
//! `MemoryMonitor` that polls the system and derives a
//! [`MemoryPressure`](super::MemoryPressure) level. It is the
//! engine-independent, platform-agnostic pressure oracle.

use std::time::{Duration, Instant};

use super::MemoryPressure;

/// Memory statistics snapshot.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub rss_bytes: u64,
    pub total_ram_bytes: u64,
    pub vm_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,
}

impl MemoryStats {
    /// Compute memory pressure level from current stats.
    pub fn pressure(&self) -> MemoryPressure {
        let ratio = self.rss_bytes as f64 / self.total_ram_bytes.max(1) as f64;
        if self.swap_used_bytes > 0
            && self.swap_total_bytes > 0
            && self.swap_used_bytes as f64 / self.swap_total_bytes as f64 > 0.5
        {
            MemoryPressure::Oom
        } else if ratio > 0.90 {
            MemoryPressure::Severe
        } else if ratio > 0.80 {
            MemoryPressure::Critical
        } else if ratio > 0.70 {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }
}

/// Real-time memory monitor.
///
/// Polls system memory stats at configurable intervals and reports a
/// [`MemoryPressure`](super::MemoryPressure) level. Platform-specific
/// sampling is intentionally not implemented here — the engine
/// wires the actual sampling in `memory_impl::telemetry_impl`.
#[allow(dead_code)]
pub struct MemoryMonitor {
    stats: MemoryStats,
    last_update: Instant,
    poll_interval: Duration,
    last_pressure: MemoryPressure,
}

impl MemoryMonitor {
    pub fn new(poll_interval: Duration) -> Self {
        Self {
            stats: MemoryStats {
                rss_bytes: 0,
                total_ram_bytes: 0,
                vm_bytes: 0,
                swap_used_bytes: 0,
                swap_total_bytes: 0,
            },
            last_update: Instant::now(),
            poll_interval,
            last_pressure: MemoryPressure::Normal,
        }
    }

    /// Poll current memory stats from the system.
    ///
    /// Returns the last-known snapshot and stamps `last_update`. The
    /// constitutional surface intentionally does not implement
    /// platform-specific sampling; the engine-side execution-plane
    /// equivalent lives at
    /// `compute-core/src/ecs/memory_impl/telemetry_impl.rs` and wires
    /// `mach_vm_info` / `host_statistics64` / `proc_info` for macOS.
    pub fn poll(&mut self) -> MemoryStats {
        self.last_update = Instant::now();
        self.stats.clone()
    }

    /// Borrow the last-known stats snapshot without stamping the
    /// poll timestamp. Useful for read-only diagnostics and for
    /// [`MemoryEnforcer`](super::enforcer::MemoryEnforcer) callers
    /// that want to expose the latest observed stats without
    /// advancing the monitor's state.
    pub fn last_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Get last known pressure level.
    pub fn pressure(&self) -> MemoryPressure {
        self.last_pressure
    }
}
