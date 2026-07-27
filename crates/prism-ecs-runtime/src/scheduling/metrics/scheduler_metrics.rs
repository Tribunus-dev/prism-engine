//! Scheduler metrics (constitutional home, advisory).
//!
//! Per the inventory v2.1 step 54, this replaces the engine's
//! `scheduler_metrics.rs` (443 LOC). Metrics here are ADVISORY
//! (M bucket), not evidence. They are rebuildable projections of
//! the canonical scheduling state.
//!
//! A metric becomes evidence only when a measurement is admitted
//! into an immutable receipt (see `evidence::scheduling_receipts`).

use std::collections::BTreeMap;

/// Per-scheduler metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    /// Per-phase metrics. BTreeMap for stable iteration order.
    per_phase: BTreeMap<String, PhaseMetrics>,
    /// Total dispatch attempts.
    pub total_dispatches: u64,
    /// Total successful completions.
    pub total_successes: u64,
    /// Total failed completions.
    pub total_failures: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PhaseMetrics {
    pub dispatches: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_latency_us: f64,
}

impl SchedulerMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_dispatch(&mut self, phase_id: &str) {
        self.total_dispatches += 1;
        let m = self.per_phase.entry(phase_id.to_string()).or_default();
        m.dispatches += 1;
    }

    pub fn record_success(&mut self, phase_id: &str, latency_us: u64) {
        self.total_successes += 1;
        let m = self.per_phase.entry(phase_id.to_string()).or_default();
        m.successes += 1;
        let n = m.successes as f64;
        m.avg_latency_us = m.avg_latency_us * (n - 1.0) / n + (latency_us as f64) / n;
    }

    pub fn record_failure(&mut self, phase_id: &str) {
        self.total_failures += 1;
        let m = self.per_phase.entry(phase_id.to_string()).or_default();
        m.failures += 1;
    }

    pub fn get(&self, phase_id: &str) -> Option<&PhaseMetrics> {
        self.per_phase.get(phase_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_metrics_have_zero_totals() {
        let m = SchedulerMetrics::new();
        assert_eq!(m.total_dispatches, 0);
        assert_eq!(m.total_successes, 0);
        assert_eq!(m.total_failures, 0);
    }

    #[test]
    fn record_dispatch_increments_total() {
        let mut m = SchedulerMetrics::new();
        m.record_dispatch("p1");
        m.record_dispatch("p1");
        m.record_dispatch("p2");
        assert_eq!(m.total_dispatches, 3);
        assert_eq!(m.get("p1").unwrap().dispatches, 2);
        assert_eq!(m.get("p2").unwrap().dispatches, 1);
    }

    #[test]
    fn record_success_updates_running_average_latency() {
        // Architectural invariant: the per-phase average latency
        // is a running mean. After N successes, it equals the
        // mean of the last N samples.
        let mut m = SchedulerMetrics::new();
        m.record_dispatch("p1");
        m.record_success("p1", 100);
        assert!((m.get("p1").unwrap().avg_latency_us - 100.0).abs() < 1e-9);
        m.record_success("p1", 200);
        assert!((m.get("p1").unwrap().avg_latency_us - 150.0).abs() < 1e-9);
    }
}
