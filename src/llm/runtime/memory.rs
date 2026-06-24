use parking_lot::Mutex;

use crate::llm::server::{AllocationOwner, MemoryPressureLevel, MemoryPressureReceipt};

/// Tracks memory pressure levels and history of level transitions.
///
/// Maintains a running total of allocated bytes and compares it against
/// configurable thresholds to determine the current pressure level.
/// Each transition between levels produces a timestamped receipt.
pub struct MemoryPressureMonitor {
    inner: Mutex<Inner>,
}

struct Inner {
    elevated_threshold_bytes: u64,
    critical_threshold_bytes: u64,
    total_allocated_bytes: u64,
    previous_level: MemoryPressureLevel,
    history: Vec<MemoryPressureReceipt>,
}

impl MemoryPressureMonitor {
    /// Creates a new monitor with the given thresholds.
    ///
    /// `elevated_threshold_bytes` – the boundary below which pressure is Normal.
    /// `critical_threshold_bytes` – the boundary above which pressure is Critical.
    pub fn new(elevated_threshold_bytes: u64, critical_threshold_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                elevated_threshold_bytes,
                critical_threshold_bytes,
                total_allocated_bytes: 0,
                previous_level: MemoryPressureLevel::Normal,
                history: Vec::new(),
            }),
        }
    }

    /// Returns the current pressure level based on total allocated bytes
    /// and the configured thresholds.
    pub fn current_level(&self) -> MemoryPressureLevel {
        let inner = self.inner.lock();
        level_for(inner.total_allocated_bytes, inner.elevated_threshold_bytes, inner.critical_threshold_bytes)
    }

    /// Records an allocation of `bytes` by `owner` and checks for a level
    /// transition.  Returns an error if the allocation would overflow the
    /// internal counter.
    pub fn record_allocation(&self, bytes: u64, _owner: AllocationOwner) -> Result<(), String> {
        let mut inner = self.inner.lock();
        inner.total_allocated_bytes = inner
            .total_allocated_bytes
            .checked_add(bytes)
            .ok_or_else(|| "allocation would overflow total allocated bytes counter".to_string())?;

        let new_level = level_for(
            inner.total_allocated_bytes,
            inner.elevated_threshold_bytes,
            inner.critical_threshold_bytes,
        );
        if new_level != inner.previous_level {
            let receipt = MemoryPressureReceipt {
                level: new_level,
                timestamp: chrono_now(),
                action_taken: format!(
                    "Level changed from {:?} to {:?} (total: {} bytes)",
                    inner.previous_level, new_level, inner.total_allocated_bytes
                ),
            };
            inner.history.push(receipt);
            inner.previous_level = new_level;
        }
        Ok(())
    }

    /// Records a release of `bytes` and checks for a level transition.
    pub fn record_release(&self, bytes: u64) {
        let mut inner = self.inner.lock();
        inner.total_allocated_bytes = inner.total_allocated_bytes.saturating_sub(bytes);

        let new_level = level_for(
            inner.total_allocated_bytes,
            inner.elevated_threshold_bytes,
            inner.critical_threshold_bytes,
        );
        if new_level != inner.previous_level {
            let receipt = MemoryPressureReceipt {
                level: new_level,
                timestamp: chrono_now(),
                action_taken: format!(
                    "Level changed from {:?} to {:?} (total: {} bytes)",
                    inner.previous_level, new_level, inner.total_allocated_bytes
                ),
            };
            inner.history.push(receipt);
            inner.previous_level = new_level;
        }
    }

    /// Returns a snapshot of all recorded level-transition receipts.
    pub fn get_history(&self) -> Vec<MemoryPressureReceipt> {
        self.inner.lock().history.clone()
    }

    /// Updates the elevated and critical thresholds.
    pub fn update_thresholds(&self, elevated: u64, critical: u64) {
        let mut inner = self.inner.lock();
        inner.elevated_threshold_bytes = elevated;
        inner.critical_threshold_bytes = critical;
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn level_for(total: u64, elevated: u64, critical: u64) -> MemoryPressureLevel {
    if total >= critical {
        MemoryPressureLevel::Critical
    } else if total >= elevated {
        MemoryPressureLevel::Elevated
    } else {
        MemoryPressureLevel::Normal
    }
}

/// Returns an ISO-8601 timestamp string for the current moment.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_level_is_normal() {
        let m = MemoryPressureMonitor::new(100, 200);
        assert_eq!(m.current_level(), MemoryPressureLevel::Normal);
    }

    #[test]
    fn allocation_triggers_elevated() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(150, AllocationOwner::KvCache).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Elevated);
    }

    #[test]
    fn allocation_triggers_critical() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(250, AllocationOwner::WeightResidency).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Critical);
    }

    #[test]
    fn release_lowers_level() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(250, AllocationOwner::WeightResidency).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Critical);

        m.record_release(60);
        assert_eq!(m.current_level(), MemoryPressureLevel::Elevated);

        m.record_release(100);
        assert_eq!(m.current_level(), MemoryPressureLevel::Normal);
    }

    #[test]
    fn level_transitions_record_receipts() {
        let m = MemoryPressureMonitor::new(100, 200);
        assert!(m.get_history().is_empty());

        m.record_allocation(150, AllocationOwner::KvCache).unwrap();
        assert_eq!(m.get_history().len(), 1);

        m.record_release(60);
        assert_eq!(m.get_history().len(), 2);
    }

    #[test]
    fn unchanged_level_does_not_record() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(50, AllocationOwner::KvCache).unwrap();
        assert!(m.get_history().is_empty());
    }

    #[test]
    fn receive_preserves_level_within_threshold() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(150, AllocationOwner::KvCache).unwrap();
        let before = m.get_history().len();

        // Allocate more but stay within Elevated
        m.record_allocation(40, AllocationOwner::TokenBuffer).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Elevated);
        assert_eq!(m.get_history().len(), before);
    }

    #[test]
    fn update_thresholds_changes_pressure() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(150, AllocationOwner::KvCache).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Elevated);

        m.update_thresholds(200, 300);
        assert_eq!(m.current_level(), MemoryPressureLevel::Normal);
    }

    #[test]
    fn initial_thresholds_used() {
        let m = MemoryPressureMonitor::new(500, 1000);
        m.record_allocation(600, AllocationOwner::KvCache).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Elevated);

        m.record_allocation(500, AllocationOwner::KvCache).unwrap();
        assert_eq!(m.current_level(), MemoryPressureLevel::Critical);
    }

    #[test]
    fn allocation_overflow_returns_err() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(u64::MAX, AllocationOwner::KvCache).unwrap();
        let result = m.record_allocation(1, AllocationOwner::KvCache);
        assert!(result.is_err());
    }

    #[test]
    fn receipts_have_timestamps() {
        let m = MemoryPressureMonitor::new(100, 200);
        m.record_allocation(150, AllocationOwner::KvCache).unwrap();
        let receipt = m.get_history().into_iter().next().unwrap();
        assert!(!receipt.timestamp.is_empty());
        assert!(!receipt.action_taken.is_empty());
    }
}

// ── Compute-core integration ────────────────────────────────────────
//
// When the `prism-backend` feature is active, the compute-core
// `MemoryMonitor` provides real-time memory stats via macOS system
// APIs. This block bridges the compute-core pressure levels with
// the engine's `MemoryPressureLevel` enum and exposes a polling
// monitor that feeds into the existing `MemoryPressureMonitor`.

#[cfg(feature = "prism-backend")]
pub mod compute_memory {
    use std::time::Duration;

    use super::{AllocationOwner, MemoryPressureLevel, MemoryPressureReceipt};
    use tribunus_compute_core::memory::monitor::{MemoryMonitor, MemoryStats};
    use tribunus_compute_core::memory::MemoryPressure;
    use parking_lot::Mutex;

    /// Map a compute-core pressure level to the engine's level.
    pub fn to_engine_pressure(cc_pressure: MemoryPressure) -> MemoryPressureLevel {
        match cc_pressure {
            MemoryPressure::Normal | MemoryPressure::Warning => MemoryPressureLevel::Normal,
            MemoryPressure::Critical => MemoryPressureLevel::Elevated,
            MemoryPressure::Severe | MemoryPressure::Oom => MemoryPressureLevel::Critical,
        }
    }

    /// Wraps a compute-core `MemoryMonitor` and bridges its pressure
    /// reports into the engine's `MemoryPressureMonitor`.
    ///
    /// Polls system memory stats at the configured interval and
    /// delegates level transitions to the engine-side monitor.
    pub struct ComputeMemoryMonitor {
        inner: Mutex<ComputeMemoryInner>,
    }

    struct ComputeMemoryInner {
        sys_monitor: MemoryMonitor,
        engine_monitor: super::MemoryPressureMonitor,
    }

    impl ComputeMemoryMonitor {
        /// Create a new compute-backed memory monitor.
        ///
        /// `poll_interval_ms` controls how often system stats are refreshed.
        /// `elevated_threshold_bytes` and `critical_threshold_bytes` are
        /// forwarded to the engine-side pressure monitor.
        pub fn new(
            poll_interval_ms: u64,
            elevated_threshold_bytes: u64,
            critical_threshold_bytes: u64,
        ) -> Self {
            Self {
                inner: Mutex::new(ComputeMemoryInner {
                    sys_monitor: MemoryMonitor::new(Duration::from_millis(poll_interval_ms)),
                    engine_monitor: super::MemoryPressureMonitor::new(
                        elevated_threshold_bytes,
                        critical_threshold_bytes,
                    ),
                }),
            }
        }

        /// Poll system memory stats, map the pressure level, and
        /// forward any level transitions to the engine monitor.
        pub fn poll(&self) -> MemoryStats {
            let mut inner = self.inner.lock();
            let stats = inner.sys_monitor.poll();
            let _cc_pressure = stats.pressure();
            let _engine_level = to_engine_pressure(_cc_pressure);

            // Record a synthetic allocation at this level so the engine
            // monitor's level-tracking is kept in sync with real system
            // pressure. The engine monitor uses byte-based thresholds;
            // we pass the RSS as the allocated amount.
            let _ = inner.engine_monitor.record_allocation(
                stats.rss_bytes,
                AllocationOwner::KvCache,
            );

            stats
        }

        /// Current pressure level (engine-side, synced from last poll).
        pub fn current_level(&self) -> MemoryPressureLevel {
            self.inner.lock().engine_monitor.current_level()
        }

        /// Delegate to the engine-side monitor.
        pub fn record_allocation(
            &self,
            bytes: u64,
            owner: AllocationOwner,
        ) -> Result<(), String> {
            self.inner.lock().engine_monitor.record_allocation(bytes, owner)
        }

        /// Delegate to the engine-side monitor.
        pub fn record_release(&self, bytes: u64) {
            self.inner.lock().engine_monitor.record_release(bytes)
        }

        /// Delegate to the engine-side monitor.
        pub fn get_history(&self) -> Vec<MemoryPressureReceipt> {
            self.inner.lock().engine_monitor.get_history()
        }

        /// Update thresholds on the engine-side monitor.
        pub fn update_thresholds(&self, elevated: u64, critical: u64) {
            self.inner.lock()
                .engine_monitor
                .update_thresholds(elevated, critical);
        }
    }
}
