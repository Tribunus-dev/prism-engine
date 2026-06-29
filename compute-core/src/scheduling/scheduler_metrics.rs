//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Scheduler diagnostics and telemetry.
//!
//! Production counters and histograms for the [`HeterogeneousExecutor`].
//! All counters use atomic operations with `Ordering::Relaxed` for lock-free,
//! lossy-but-observable metrics.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// LatencyHistogram
// ---------------------------------------------------------------------------

/// O(1) histogram bucket for latency tracking.
///
/// Bounds are specified at construction as an ordered list of upper bounds
/// (in nanoseconds). Each bucket counts samples <= its bound. Samples above
/// the last bound go into `overflow`.
pub struct LatencyHistogram {
    buckets: Vec<AtomicU64>,
    bounds: Vec<u64>,
    overflow: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new histogram with the given bucket upper bounds (in ns).
    ///
    /// `bounds` must be non-empty and strictly increasing.
    pub fn new(bounds: Vec<u64>) -> Self {
        assert!(
            !bounds.is_empty(),
            "LatencyHistogram requires at least one bucket"
        );
        let mut buckets = Vec::with_capacity(bounds.len());
        for _ in 0..bounds.len() {
            buckets.push(AtomicU64::new(0));
        }
        LatencyHistogram {
            buckets,
            bounds,
            overflow: AtomicU64::new(0),
        }
    }

    /// Record a single latency value (in nanoseconds).
    pub fn record(&self, value_ns: u64) {
        for (i, &bound) in self.bounds.iter().enumerate() {
            if value_ns <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        self.overflow.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot the current bucket counts.
    ///
    /// Returns `Vec<(bound, count)>` — one entry per bucket (including
    /// overflow as `(u64::MAX, count)`).
    pub fn snapshot(&self) -> Vec<(u64, u64)> {
        let mut out = Vec::with_capacity(self.buckets.len() + 1);
        for (i, &bound) in self.bounds.iter().enumerate() {
            out.push((bound, self.buckets[i].load(Ordering::Relaxed)));
        }
        out.push((u64::MAX, self.overflow.load(Ordering::Relaxed)));
        out
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        for b in &self.buckets {
            b.store(0, Ordering::Relaxed);
        }
        self.overflow.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// MetricsSnapshot
// ---------------------------------------------------------------------------

/// Atomic snapshot of all scheduler metrics.
///
/// Taken via [`SchedulerMetrics::snapshot`] and queried by the
/// autoscaler, dashboard, or test assertions.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // Active counts (snapshot-able)
    pub requests_active: i64,
    pub requests_queued: i64,
    pub sessions_active: i64,
    pub metal_in_flight: i64,
    pub ane_in_flight: i64,
    pub accelerate_in_flight: i64,
    pub metal_queue_depth: i64,
    pub ane_queue_depth: i64,
    pub accelerate_queue_depth: i64,
    pub slot_leases_active: i64,
    pub iosurface_pool_free: i64,
    pub artifact_warmed_count: i64,
    pub artifact_quarantined_count: i64,
    // Cumulative counters
    pub fallback_total: u64,
    pub cancellation_total: u64,
    pub work_timeout_total: u64,
    // Token throughput (placeholder: requires timer integration)
    pub tokens_per_second: f64,
    pub prefill_tokens_per_second: f64,
    // Overlap tracking
    pub overlap_ratio: f64,
}

// ---------------------------------------------------------------------------
// SchedulerMetrics
// ---------------------------------------------------------------------------

/// Thread-safe scheduler metrics.
///
/// All counters use relaxed atomic operations — individual updates are
/// non-blocking and safe across any number of concurrent readers/writers.
///
/// # Latency histograms
///
/// - `queue_wait_histogram` — time spent in the ready queue before dispatch.
/// - `execution_histogram` — backend execution duration.
/// - `total_latency_histogram` — end-to-end request latency.
pub struct SchedulerMetrics {
    // ── Active counts (snapshot-able) ──────────────────────────────────────
    pub requests_active: AtomicI64,
    pub requests_queued: AtomicI64,
    pub sessions_active: AtomicI64,
    pub metal_in_flight: AtomicI64,
    pub ane_in_flight: AtomicI64,
    pub accelerate_in_flight: AtomicI64,
    pub metal_queue_depth: AtomicI64,
    pub ane_queue_depth: AtomicI64,
    pub accelerate_queue_depth: AtomicI64,
    pub slot_leases_active: AtomicI64,
    pub iosurface_pool_free: AtomicI64,
    pub artifact_warmed_count: AtomicI64,
    pub artifact_quarantined_count: AtomicI64,

    // ── Cumulative counters ───────────────────────────────────────────────
    pub fallback_total: AtomicU64,
    pub cancellation_total: AtomicU64,
    pub work_timeout_total: AtomicU64,

    // ── Token throughput (delta counters reset on read) ───────────────────
    pub tokens_produced: AtomicU64,
    pub prefill_tokens_produced: AtomicU64,

    // ── Latency histograms ────────────────────────────────────────────────
    pub queue_wait_histogram: LatencyHistogram,
    pub execution_histogram: LatencyHistogram,
    pub total_latency_histogram: LatencyHistogram,

    // ── Overlap tracking ──────────────────────────────────────────────────
    pub overlap_ns_total: AtomicU64,
    pub overlap_event_count: AtomicU64,
}

impl SchedulerMetrics {
    /// Create a new metrics collector with default histogram bucket boundaries.
    ///
    /// **Execution histogram** (ns): 100, 500, 1K, 2K, 5K, 10K, 50K, 100K, 500K, 1M
    /// **Queue-wait histogram** (ns): 1K, 5K, 10K, 50K, 100K, 500K, 1M, 5M
    /// **Total-latency histogram** (ns): same as execution.
    pub fn new() -> Self {
        SchedulerMetrics {
            // Active counts
            requests_active: AtomicI64::new(0),
            requests_queued: AtomicI64::new(0),
            sessions_active: AtomicI64::new(0),
            metal_in_flight: AtomicI64::new(0),
            ane_in_flight: AtomicI64::new(0),
            accelerate_in_flight: AtomicI64::new(0),
            metal_queue_depth: AtomicI64::new(0),
            ane_queue_depth: AtomicI64::new(0),
            accelerate_queue_depth: AtomicI64::new(0),
            slot_leases_active: AtomicI64::new(0),
            iosurface_pool_free: AtomicI64::new(0),
            artifact_warmed_count: AtomicI64::new(0),
            artifact_quarantined_count: AtomicI64::new(0),

            // Cumulative counters
            fallback_total: AtomicU64::new(0),
            cancellation_total: AtomicU64::new(0),
            work_timeout_total: AtomicU64::new(0),

            // Token throughput
            tokens_produced: AtomicU64::new(0),
            prefill_tokens_produced: AtomicU64::new(0),

            // Latency histograms
            queue_wait_histogram: LatencyHistogram::new(vec![
                1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
            ]),
            execution_histogram: LatencyHistogram::new(vec![
                100, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
            ]),
            total_latency_histogram: LatencyHistogram::new(vec![
                100, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
            ]),

            // Overlap tracking
            overlap_ns_total: AtomicU64::new(0),
            overlap_event_count: AtomicU64::new(0),
        }
    }

    /// Atomically snapshot every metric.
    ///
    /// Each atomic is read individually with `Ordering::Relaxed` — the
    /// snapshot is deliberately racy (fine for observability, not for
    /// accounting).
    pub fn snapshot(&self) -> MetricsSnapshot {
        let overlap_total = self.overlap_ns_total.load(Ordering::Relaxed);
        let overlap_events = self.overlap_event_count.load(Ordering::Relaxed);
        let overlap_ratio = if overlap_events > 0 {
            overlap_total as f64 / overlap_events as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            // Active counts
            requests_active: self.requests_active.load(Ordering::Relaxed),
            requests_queued: self.requests_queued.load(Ordering::Relaxed),
            sessions_active: self.sessions_active.load(Ordering::Relaxed),
            metal_in_flight: self.metal_in_flight.load(Ordering::Relaxed),
            ane_in_flight: self.ane_in_flight.load(Ordering::Relaxed),
            accelerate_in_flight: self.accelerate_in_flight.load(Ordering::Relaxed),
            metal_queue_depth: self.metal_queue_depth.load(Ordering::Relaxed),
            ane_queue_depth: self.ane_queue_depth.load(Ordering::Relaxed),
            accelerate_queue_depth: self.accelerate_queue_depth.load(Ordering::Relaxed),
            slot_leases_active: self.slot_leases_active.load(Ordering::Relaxed),
            iosurface_pool_free: self.iosurface_pool_free.load(Ordering::Relaxed),
            artifact_warmed_count: self.artifact_warmed_count.load(Ordering::Relaxed),
            artifact_quarantined_count: self.artifact_quarantined_count.load(Ordering::Relaxed),

            // Cumulative counters
            fallback_total: self.fallback_total.load(Ordering::Relaxed),
            cancellation_total: self.cancellation_total.load(Ordering::Relaxed),
            work_timeout_total: self.work_timeout_total.load(Ordering::Relaxed),

            // Token throughput (placeholder: requires timer integration)
            tokens_per_second: 0.0,
            prefill_tokens_per_second: 0.0,

            // Overlap tracking
            overlap_ratio,
        }
    }
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TimingDomain
// ---------------------------------------------------------------------------

/// Marker for which timing domain a measurement comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimingDomain {
    /// Time spent in the scheduler itself (queuing, dispatch decisions).
    Scheduler,
    /// Time spent in the backend executor (Metal, ANE, Accelerate compute).
    Backend,
    /// Time spent waiting in a ready queue before dispatch.
    Queue,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_histogram_smoke() {
        let h = LatencyHistogram::new(vec![100, 500, 1000]);
        h.record(50);
        h.record(200);
        h.record(800);
        h.record(2000);
        let snap = h.snapshot();
        // Bucket 0 (<=100): 1
        assert_eq!(snap[0], (100, 1));
        // Bucket 1 (<=500): 1
        assert_eq!(snap[1], (500, 1));
        // Bucket 2 (<=1000): 1
        assert_eq!(snap[2], (1000, 1));
        // Overflow: 1
        assert_eq!(snap[3], (u64::MAX, 1));
    }

    #[test]
    fn test_latency_histogram_exact_boundary() {
        let h = LatencyHistogram::new(vec![100, 500]);
        h.record(100); // <= 100 → bucket 0
        h.record(500); // <= 500 → bucket 1
        let snap = h.snapshot();
        assert_eq!(snap[0], (100, 1));
        assert_eq!(snap[1], (500, 1));
        assert_eq!(snap[2], (u64::MAX, 0));
    }

    #[test]
    fn test_histogram_reset() {
        let h = LatencyHistogram::new(vec![100, 500]);
        h.record(50);
        h.record(600);
        h.reset();
        let snap = h.snapshot();
        assert_eq!(snap[0], (100, 0));
        assert_eq!(snap[1], (500, 0));
        assert_eq!(snap[2], (u64::MAX, 0));
    }

    #[test]
    fn test_metrics_new_and_default() {
        let m = SchedulerMetrics::new();
        let snap = m.snapshot();
        assert_eq!(snap.requests_active, 0);
        assert_eq!(snap.requests_queued, 0);
        assert_eq!(snap.fallback_total, 0);
        assert_eq!(snap.cancellation_total, 0);
        assert_eq!(snap.work_timeout_total, 0);
        assert_eq!(snap.metal_in_flight, 0);
        assert_eq!(snap.ane_in_flight, 0);
        assert_eq!(snap.accelerate_in_flight, 0);
        assert_eq!(snap.metal_queue_depth, 0);
        assert_eq!(snap.ane_queue_depth, 0);
        assert_eq!(snap.accelerate_queue_depth, 0);
        assert_eq!(snap.slot_leases_active, 0);
        assert_eq!(snap.iosurface_pool_free, 0);
        assert_eq!(snap.artifact_warmed_count, 0);
        assert_eq!(snap.artifact_quarantined_count, 0);
        assert_eq!(snap.tokens_per_second, 0.0);
        assert_eq!(snap.prefill_tokens_per_second, 0.0);
        assert_eq!(snap.overlap_ratio, 0.0);

        let m2 = SchedulerMetrics::default();
        assert_eq!(m2.snapshot().requests_active, 0);
    }

    #[test]
    fn test_metrics_atomic_read_write() {
        let m = SchedulerMetrics::new();

        m.requests_active.fetch_add(3, Ordering::Relaxed);
        m.requests_queued.fetch_add(5, Ordering::Relaxed);
        m.fallback_total.fetch_add(2, Ordering::Relaxed);
        m.cancellation_total.fetch_add(1, Ordering::Relaxed);
        m.work_timeout_total.fetch_add(7, Ordering::Relaxed);
        m.metal_in_flight.fetch_add(4, Ordering::Relaxed);
        m.ane_in_flight.fetch_add(2, Ordering::Relaxed);
        m.accelerate_in_flight.fetch_add(1, Ordering::Relaxed);
        m.metal_queue_depth.store(8, Ordering::Relaxed);
        m.ane_queue_depth.store(6, Ordering::Relaxed);
        m.accelerate_queue_depth.store(3, Ordering::Relaxed);
        m.slot_leases_active.store(9, Ordering::Relaxed);
        m.iosurface_pool_free.store(15, Ordering::Relaxed);
        m.artifact_warmed_count.store(12, Ordering::Relaxed);
        m.artifact_quarantined_count.store(1, Ordering::Relaxed);

        let snap = m.snapshot();
        assert_eq!(snap.requests_active, 3);
        assert_eq!(snap.requests_queued, 5);
        assert_eq!(snap.fallback_total, 2);
        assert_eq!(snap.cancellation_total, 1);
        assert_eq!(snap.work_timeout_total, 7);
        assert_eq!(snap.metal_in_flight, 4);
        assert_eq!(snap.ane_in_flight, 2);
        assert_eq!(snap.accelerate_in_flight, 1);
        assert_eq!(snap.metal_queue_depth, 8);
        assert_eq!(snap.ane_queue_depth, 6);
        assert_eq!(snap.accelerate_queue_depth, 3);
        assert_eq!(snap.slot_leases_active, 9);
        assert_eq!(snap.iosurface_pool_free, 15);
        assert_eq!(snap.artifact_warmed_count, 12);
        assert_eq!(snap.artifact_quarantined_count, 1);
    }

    #[test]
    fn test_overlap_ratio() {
        let m = SchedulerMetrics::new();
        // No events → 0.0
        assert_eq!(m.snapshot().overlap_ratio, 0.0);

        m.overlap_ns_total.fetch_add(500_000, Ordering::Relaxed);
        m.overlap_event_count.fetch_add(2, Ordering::Relaxed);
        let snap = m.snapshot();
        assert!((snap.overlap_ratio - 250_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timing_domain_roundtrip() {
        let cases = vec![
            TimingDomain::Scheduler,
            TimingDomain::Backend,
            TimingDomain::Queue,
        ];
        for domain in cases {
            let json = serde_json::to_string(&domain).unwrap();
            let parsed: TimingDomain = serde_json::from_str(&json).unwrap();
            assert_eq!(domain, parsed);
        }
    }

    #[test]
    fn test_histogram_concurrent_recording() {
        // Verify the histogram is usable from multiple threads without
        // data races (best-effort smoke — ThreadSanitizer would catch
        // actual races).
        let h = std::sync::Arc::new(LatencyHistogram::new(vec![100, 500, 1000]));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let h_clone = std::sync::Arc::clone(&h);
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    h_clone.record(i * 10);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("thread panicked");
        }

        let snap = h.snapshot();
        // 4 threads × 100 records = 400 total
        let total: u64 = snap.iter().map(|&(_, count)| count).sum();
        assert_eq!(total, 400);
    }
}
