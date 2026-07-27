//! Phase telemetry (constitutional home, advisory).
//!
//! Per the inventory v2.1 step 55, this replaces the engine's
//! `phase_telemetry.rs` (202 LOC). Advisory metrics, not evidence.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Default)]
pub struct PhaseTelemetry {
    /// Per-phase telemetry. BTreeMap for stable iteration order.
    per_phase: BTreeMap<String, PhaseTelemetryRecord>,
}

#[derive(Debug, Clone, Default)]
pub struct PhaseTelemetryRecord {
    pub epochs: u64,
    pub total_time_ns: u64,
    pub last_observed_epoch: u64,
}

impl PhaseTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, phase_id: &str, epoch: u64, time_ns: u64) {
        let r = self.per_phase.entry(phase_id.to_string()).or_default();
        r.epochs += 1;
        r.total_time_ns += time_ns;
        r.last_observed_epoch = epoch;
    }

    pub fn get(&self, phase_id: &str) -> Option<&PhaseTelemetryRecord> {
        self.per_phase.get(phase_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates() {
        let mut t = PhaseTelemetry::new();
        t.record("p1", 1, 100);
        t.record("p1", 2, 200);
        let r = t.get("p1").unwrap();
        assert_eq!(r.epochs, 2);
        assert_eq!(r.total_time_ns, 300);
        assert_eq!(r.last_observed_epoch, 2);
    }

    #[test]
    fn get_unknown_phase_returns_none() {
        let t = PhaseTelemetry::new();
        assert!(t.get("nonexistent").is_none());
    }
}
