//! ANE-TRI-LANE-CALIBRATION-0001: Calibration evidence store for tri-lane
//! execution plans on Apple Silicon.
//!
//! The `CalibrationStore` accumulates runtime measurement records and
//! provides evidence-based decisions for ANE placement.  Decisions are
//! grounded in real measured latencies, not compile-time heuristics.

use crate::compilation::tri_lane::{AppleTriLaneCalibrationRecord, ShapeClass};

/// Calibration evidence store — keyed by (device_fingerprint, artifact_digest).
pub struct CalibrationStore {
    pub records: Vec<AppleTriLaneCalibrationRecord>,
}

impl CalibrationStore {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Add a single calibration record to the store.
    pub fn add(&mut self, record: AppleTriLaneCalibrationRecord) {
        self.records.push(record);
    }

    /// Find all calibration records matching a hardware signature.
    pub fn for_hardware(&self, hardware: &str) -> Vec<&AppleTriLaneCalibrationRecord> {
        self.records
            .iter()
            .filter(|r| r.hardware_signature == hardware)
            .collect()
    }

    /// Check whether ANE placement is justified for a given region.
    ///
    /// Returns `true` if the average measured ANE time is <= (1 - threshold)
    /// times the average measured Metal time across all matching records.
    /// Returns `false` when no evidence exists for the given hardware and
    /// region pair — no evidence means no assignment.
    pub fn ane_assignment_justified(
        &self,
        hardware: &str,
        region_fingerprint: &str,
        threshold: f64,
    ) -> bool {
        let records: Vec<_> = self
            .records
            .iter()
            .filter(|r| {
                r.hardware_signature == hardware
                    && r.region_fingerprint == region_fingerprint
            })
            .collect();

        if records.is_empty() {
            return false; // No evidence = no assignment
        }

        let count = records.len() as u64;
        let avg_ane_ns: u64 =
            records.iter().map(|r| r.measured_ane_ns).sum::<u64>() / count;
        let avg_metal_ns: u64 =
            records.iter().map(|r| r.measured_metal_ns).sum::<u64>() / count;

        (avg_ane_ns as f64) <= (1.0 - threshold) * (avg_metal_ns as f64)
    }

    /// Average epoch wall time for a given hardware and region fingerprint.
    ///
    /// Returns `None` when no matching records exist.
    pub fn avg_epoch_wall_ns(
        &self,
        hardware: &str,
        region_fingerprint: &str,
    ) -> Option<u64> {
        let records: Vec<_> = self
            .records
            .iter()
            .filter(|r| {
                r.hardware_signature == hardware
                    && r.region_fingerprint == region_fingerprint
            })
            .collect();

        if records.is_empty() {
            return None;
        }

        Some(
            records.iter().map(|r| r.measured_epoch_wall_ns).sum::<u64>()
                / records.len() as u64,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::tri_lane::ShapeClass;

    fn sample_shape() -> ShapeClass {
        ShapeClass {
            batch: 1,
            sequence: 1,
            hidden: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            sliding_window: 0,
            max_context: 8192,
        }
    }

    fn sample_record(
        hardware: &str,
        region: &str,
        ane_ns: u64,
        metal_ns: u64,
        epoch_wall_ns: u64,
    ) -> AppleTriLaneCalibrationRecord {
        AppleTriLaneCalibrationRecord {
            hardware_signature: hardware.to_string(),
            os_build: "macOS-14.5".into(),
            coreml_runtime_identity: "CoreML-8.0".into(),
            region_fingerprint: region.to_string(),
            artifact_digest: "sha256-deadbeef".into(),
            shape_class: sample_shape(),
            ring_depth: 3,
            measured_ane_ns: ane_ns,
            measured_metal_ns: metal_ns,
            measured_cpu_ns: ane_ns + metal_ns, // dummy value
            measured_epoch_wall_ns: epoch_wall_ns,
            measured_overlap_ns: 10,
            slot_wait_ns: 5,
            fallback_metal_ns: metal_ns,
            numerical_error: 0.001,
        }
    }

    #[test]
    fn test_calibration_store_add() {
        let mut store = CalibrationStore::new();
        assert!(store.records.is_empty());

        let rec = sample_record("M1-Max", "attn-qkv", 100, 200, 250);
        store.add(rec);
        assert_eq!(store.records.len(), 1);
    }

    #[test]
    fn test_calibration_store_for_hardware() {
        let mut store = CalibrationStore::new();
        store.add(sample_record("M1-Max", "attn-qkv", 100, 200, 250));
        store.add(sample_record("M1-Max", "ffn-gate", 80, 160, 200));
        store.add(sample_record("M2-Ultra", "attn-qkv", 90, 180, 220));

        let m1_results = store.for_hardware("M1-Max");
        assert_eq!(m1_results.len(), 2);

        let m2_results = store.for_hardware("M2-Ultra");
        assert_eq!(m2_results.len(), 1);

        let unknown = store.for_hardware("M3-Pro");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_ane_assignment_justified_below_threshold() {
        // ANE=80ns vs Metal=200ns → ANE is 60% faster.
        // With threshold=0.10 (10%): 80 <= 180 → passes.
        let mut store = CalibrationStore::new();
        store.add(sample_record("M1-Max", "attn-qkv", 80, 200, 250));

        assert!(store.ane_assignment_justified("M1-Max", "attn-qkv", 0.10));
    }

    #[test]
    fn test_ane_assignment_justified_above_threshold() {
        // ANE=195ns vs Metal=200ns → ANE is only 2.5% faster.
        // With threshold=0.10 (10%): 195 <= 180? No → fails.
        let mut store = CalibrationStore::new();
        store.add(sample_record("M1-Max", "attn-qkv", 195, 200, 250));

        assert!(!store.ane_assignment_justified("M1-Max", "attn-qkv", 0.10));
    }

    #[test]
    fn test_ane_assignment_no_evidence_returns_false() {
        let store = CalibrationStore::new();
        assert!(!store.ane_assignment_justified("M1-Max", "nonexistent", 0.10));
    }

    #[test]
    fn test_avg_epoch_wall_ns() {
        let mut store = CalibrationStore::new();
        store.add(sample_record("M1-Max", "attn-qkv", 100, 200, 250));
        assert_eq!(store.avg_epoch_wall_ns("M1-Max", "attn-qkv"), Some(250));

        assert!(store.avg_epoch_wall_ns("M1-Max", "unknown").is_none());
    }

    #[test]
    fn test_ane_assignment_multiple_records() {
        let mut store = CalibrationStore::new();
        // Average ANE = (80 + 90) / 2 = 85, Average Metal = (200 + 180) / 2 = 190
        // threshold 0.10: 85 <= 171 → passes
        store.add(sample_record("M1-Max", "attn-qkv", 80, 200, 250));
        store.add(sample_record("M1-Max", "attn-qkv", 90, 180, 230));

        assert!(store.ane_assignment_justified("M1-Max", "attn-qkv", 0.10));
    }

    #[test]
    fn test_ane_assignment_threshold_boundary() {
        let mut store = CalibrationStore::new();
        // ANE=180 vs Metal=200: 180 <= 200 * (1 - 0.10) = 180 → boundary passes.
        store.add(sample_record("M1-Max", "ffn-gate", 180, 200, 300));
        assert!(store.ane_assignment_justified("M1-Max", "ffn-gate", 0.10));

        // ANE=181 vs Metal=200: 181 <= 180? No → fails.
        let mut store2 = CalibrationStore::new();
        store2.add(sample_record("M1-Max", "ffn-gate", 181, 200, 300));
        assert!(!store2.ane_assignment_justified("M1-Max", "ffn-gate", 0.10));
    }
}
