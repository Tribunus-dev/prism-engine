//! Outlier detector (constitutional home, advisory).
//!
//! Per the inventory v2.1 step 53, this replaces the engine's
//! `outlier_detector.rs` (234 LOC). Advisory metrics, not evidence.
//! Used by feedback-driven scheduling to detect anomalous latencies.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct OutlierDetector {
    /// Per-phase sample window. BTreeMap for stable iteration.
    samples: BTreeMap<String, Vec<f64>>,
    /// Number of standard deviations above the mean to flag as an outlier.
    pub threshold_sigma: f64,
}

impl OutlierDetector {
    pub fn new() -> Self {
        Self {
            samples: BTreeMap::new(),
            threshold_sigma: 3.0,
        }
    }

    /// Record a sample for a phase.
    pub fn record(&mut self, phase_id: &str, latency_us: f64) {
        self.samples
            .entry(phase_id.to_string())
            .or_default()
            .push(latency_us);
    }

    /// Returns true if the latest sample is an outlier (>threshold_sigma stddevs).
    pub fn is_outlier(&self, phase_id: &str) -> bool {
        let Some(samples) = self.samples.get(phase_id) else {
            return false;
        };
        if samples.len() < 2 {
            return false;
        }
        let n = samples.len() as f64;
        let mean: f64 = samples.iter().sum::<f64>() / n;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let stddev = variance.sqrt();
        if stddev == 0.0 {
            return false;
        }
        let last = samples.last().copied().unwrap_or(0.0);
        (last - mean).abs() > self.threshold_sigma * stddev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_detector_returns_false_for_unknown_phase() {
        let d = OutlierDetector::new();
        assert!(!d.is_outlier("p1"));
    }

    #[test]
    fn single_sample_is_not_an_outlier() {
        // Architectural invariant: a single sample cannot be
        // classified as an outlier (no statistical basis). The
        // detector needs at least 2 samples.
        let mut d = OutlierDetector::new();
        d.record("p1", 100.0);
        assert!(!d.is_outlier("p1"));
    }

    #[test]
    fn uniform_samples_are_not_outliers() {
        // Architectural invariant: when all samples are identical,
        // the standard deviation is zero, and the last sample
        // cannot be an outlier.
        let mut d = OutlierDetector::new();
        for _ in 0..10 {
            d.record("p1", 100.0);
        }
        assert!(!d.is_outlier("p1"));
    }

    #[test]
    fn clearly_anomalous_sample_is_an_outlier() {
        // Architectural invariant: a sample that is many standard
        // deviations from the mean is classified as an outlier.
        let mut d = OutlierDetector::new();
        for _ in 0..10 {
            d.record("p1", 100.0);
        }
        d.record("p1", 10000.0); // huge spike
        assert!(d.is_outlier("p1"));
    }
}
