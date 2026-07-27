//! Inference telemetry (constitutional home, advisory).
//!
//! Per the inventory v2.1 row 22 (referenced as M-bucket
//! component for autoscaling), this replaces the engine's
//! `InferenceTelemetry` defined in `compute-core/src/ecs/scheduling/mod.rs`.
//! Advisory metrics, not evidence.
//!
//! # Authority
//!
//! InferenceTelemetry is a thread-safe advisory collector. The
//! runtime scheduling systems write to it; the EXO cluster
//! autoscaler reads from it. The collector is a snapshot
//! projection of queue depth and latencies — never a canonical
//! scheduling decision.

use std::sync::{Arc, LazyLock, Mutex};

/// Snapshot of inference telemetry at a point in time.
#[derive(Debug, Clone)]
pub struct InferenceTelemetrySnapshot {
    pub queue_depth: usize,
    pub avg_latency_us: f64,
}

/// Thread-safe inference telemetry collector.
#[derive(Clone)]
pub struct InferenceTelemetry {
    inner: Arc<Mutex<InferenceTelemetryInner>>,
}

#[derive(Debug, Clone)]
struct InferenceTelemetryInner {
    queue_depth: usize,
    latencies: Vec<f64>,
    max_latency_samples: usize,
}

static GLOBAL_INFERENCE_TELEMETRY: LazyLock<InferenceTelemetry> =
    LazyLock::new(InferenceTelemetry::new);

impl InferenceTelemetry {
    /// Return the global singleton telemetry collector.
    pub fn global() -> Self {
        GLOBAL_INFERENCE_TELEMETRY.clone()
    }

    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(InferenceTelemetryInner {
                queue_depth: 0,
                latencies: Vec::with_capacity(128),
                max_latency_samples: 128,
            })),
        }
    }

    /// Record the current queue depth.
    pub fn set_queue_depth(&self, depth: usize) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.queue_depth = depth;
        }
    }

    /// Record a single inference latency in microseconds.
    pub fn record_latency(&self, latency_us: f64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.latencies.push(latency_us);
            if inner.latencies.len() > inner.max_latency_samples {
                inner.latencies.remove(0);
            }
        }
    }

    /// Atomically snapshot the current telemetry values.
    pub fn snapshot(&self) -> InferenceTelemetrySnapshot {
        let inner = self.inner.lock().expect("infallible mutex");
        let avg = if inner.latencies.is_empty() {
            0.0
        } else {
            inner.latencies.iter().sum::<f64>() / inner.latencies.len() as f64
        };
        InferenceTelemetrySnapshot {
            queue_depth: inner.queue_depth,
            avg_latency_us: avg,
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_telemetry_has_zero_queue_depth() {
        let t = InferenceTelemetry::new();
        let s = t.snapshot();
        assert_eq!(s.queue_depth, 0);
        assert_eq!(s.avg_latency_us, 0.0);
    }

    #[test]
    fn set_queue_depth_is_visible_in_snapshot() {
        // Architectural invariant: a queue-depth write is visible
        // in the next snapshot. The collector is monotonic.
        let t = InferenceTelemetry::new();
        t.set_queue_depth(42);
        let s = t.snapshot();
        assert_eq!(s.queue_depth, 42);
    }

    #[test]
    fn record_latency_updates_running_average() {
        // Architectural invariant: the avg_latency is a mean
        // over the last max_latency_samples records. After
        // recording [100, 200], the average is 150.
        let t = InferenceTelemetry::new();
        t.record_latency(100.0);
        t.record_latency(200.0);
        let s = t.snapshot();
        assert_eq!(s.avg_latency_us, 150.0);
    }

    #[test]
    fn global_returns_same_singleton() {
        // Architectural invariant: the global singleton is
        // shared across all callers. A write to global() is
        // visible in the next global().snapshot().
        let g1 = InferenceTelemetry::global();
        g1.set_queue_depth(7);
        let g2 = InferenceTelemetry::global();
        assert_eq!(g2.snapshot().queue_depth, 7);
    }
}
