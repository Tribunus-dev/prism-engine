use super::LoadLevel;

/// Estimates system load from GPU utilisation and KV-cache pressure.
///
/// - GPU utilisation is estimated from the ratio of active slots to total slots.
/// - KV-cache pressure is the fraction of used KV pages.
/// - Queued count from the [`BatchScheduler`] queue depth is also factored in.
pub struct LoadMonitor {
    gpu_util: f64,
    kv_cache_pressure: f64,
    queued_count: usize,
    total_slots: u64,
    #[allow(dead_code)]
    kv_cache_max_pages: u64,
}

impl LoadMonitor {
    /// Create a new monitor with device parameters.
    pub fn new(num_slots: u64, kv_cache_max_pages: u64) -> Self {
        Self {
            gpu_util: 0.0,
            kv_cache_pressure: 0.0,
            queued_count: 0,
            total_slots: num_slots,
            kv_cache_max_pages,
        }
    }

    /// Sample current load and return a [`LoadLevel`].
    ///
    /// - Returns **Low** when both metrics are below 0.4.
    /// - Returns **High** when either metric exceeds 0.7.
    /// - Returns **Medium** otherwise.
    pub fn sample(&self) -> LoadLevel {
        let util = self.gpu_util.max(self.kv_cache_pressure);

        // Factor in queue depth: treat a deep queue as additional pressure.
        let queue_factor = if self.total_slots > 0 {
            (self.queued_count as f64) / (self.total_slots as f64)
        } else {
            0.0
        };
        let effective_load = util.max(queue_factor);

        if effective_load <= 0.4 {
            LoadLevel::Low
        } else if effective_load <= 0.7 {
            LoadLevel::Medium
        } else {
            LoadLevel::High
        }
    }

    /// Update state after dispatching a batch.
    ///
    /// `active_slots` is the number of slots currently in use,
    /// `total_slots` is the capacity.
    pub fn observe_dispatch(&mut self, batch_size: usize, active_slots: usize, total_slots: u64) {
        self.queued_count = batch_size.saturating_sub(active_slots);
        self.gpu_util = if total_slots > 0 {
            (active_slots as f64) / (total_slots as f64)
        } else {
            0.0
        };
        // KV pressure is estimated from active slots proportion.
        self.kv_cache_pressure = self.gpu_util;
    }

    // -- internal helpers ---------------------------

    #[allow(dead_code)]
    pub fn set_queued_count(&mut self, count: usize) {
        self.queued_count = count;
    }

    #[allow(dead_code)]
    pub(crate) fn kv_cache_pressure(&self) -> f64 {
        self.kv_cache_pressure
    }

    #[allow(dead_code)]
    pub(crate) fn gpu_util(&self) -> f64 {
        self.gpu_util
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_dispatches_returns_low() {
        let monitor = LoadMonitor::new(8, 4096);
        assert_eq!(monitor.sample(), LoadLevel::Low);
    }

    #[test]
    fn test_high_load_when_all_slots_active() {
        let mut monitor = LoadMonitor::new(4, 4096);
        // 4/4 active → gpu_util = 1.0 → should return High
        monitor.observe_dispatch(4, 4, 4);
        assert_eq!(monitor.sample(), LoadLevel::High);
    }

    #[test]
    fn test_medium_load() {
        let mut monitor = LoadMonitor::new(8, 4096);
        // 4/8 active → gpu_util = 0.5 → Medium
        monitor.observe_dispatch(4, 4, 8);
        assert_eq!(monitor.sample(), LoadLevel::Medium);
    }

    #[test]
    fn test_low_load_with_few_active() {
        let mut monitor = LoadMonitor::new(8, 4096);
        // 2/8 active → gpu_util = 0.25 → Low
        monitor.observe_dispatch(2, 2, 8);
        assert_eq!(monitor.sample(), LoadLevel::Low);
    }
}
