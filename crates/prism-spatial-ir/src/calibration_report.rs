//! M1 calibration report — per-machine runtime cost measurements.
//!
//! The calibration report captures the execution profile of a specific
//! hardware target (Apple M1 Mac, in this case) so the cost model can
//! produce accurate estimates.  Each report is tied to a specific machine,
//! OS build, power/thermal state, and includes per-operation latency
//! percentiles and domain-transition costs.

use serde::{Deserialize, Serialize};

/// Power state of the machine during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PowerState {
    /// Running on battery power.
    OnBattery,
    /// Plugged into AC power.
    Plugged,
}

/// Thermal state of the machine during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThermalState {
    /// Normal operating temperature.
    Nominal,
    /// Slightly elevated temperature; minor throttling possible.
    Fair,
    /// Significant throttling in effect.
    Serious,
    /// Critical temperature; maximum throttling.
    Critical,
}

/// Memory pressure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Ample free memory.
    Low,
    /// Moderate usage; some swapping may occur.
    Moderate,
    /// Significant swapping; performance impact expected.
    High,
}

// ---------------------------------------------------------------------------
// Digest helpers
// ---------------------------------------------------------------------------

/// Deterministic FNV-1a 64-bit hash.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// M1CalibrationReport
// ---------------------------------------------------------------------------

/// Calibration measurements for a specific Apple M1 machine.
///
/// Contains per-operation latency percentiles and domain-transition costs
/// measured at calibration time.  The cost model interpolates from these
/// values when estimating spatial-graph schedules.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct M1CalibrationReport {
    /// Hardware model identifier, e.g. "MacBookPro17,1".
    pub hardware_id: String,
    /// macOS build number, e.g. "24F82".
    pub os_build: String,
    /// Hardware model name, e.g. "Mac14,2".
    pub hardware_model: String,
    /// Metal-capable GPU device name, e.g. "Apple M1 Pro".
    pub metal_device: String,
    /// Metal compiler version string.
    pub compiler_version: String,
    /// macOS kernel version string.
    pub kernel_version: String,
    /// Power state at calibration time.
    pub power_state: PowerState,
    /// Thermal state at calibration time.
    pub thermal_state: ThermalState,
    /// Memory pressure at calibration time.
    pub memory_pressure: MemoryPressure,

    // ── Latency percentiles (microseconds) ───────────────────────────────
    /// Median (p50) end-to-end latency in microseconds.
    pub latency_p50_us: f64,
    /// 95th percentile latency in microseconds.
    pub latency_p95_us: f64,
    /// 99th percentile latency in microseconds.
    pub latency_p99_us: f64,
    /// Variance of the latency distribution.
    pub latency_variance: f64,

    // ── Confidence and contention ────────────────────────────────────────
    /// Confidence level of the calibration (0.0 = unreliable, 1.0 = certain).
    pub confidence: f64,
    /// Measured resource contention factor (0.0 = no contention, 1.0 = saturated).
    pub contention: f64,

    // ── Domain-transition costs (microseconds) ───────────────────────────
    /// Cost of submitting a Metal GPU kernel (launch overhead).
    pub metal_submission_cost_us: f64,
    /// Cost of a CPU-side kernel dispatch.
    pub cpu_cost_us: f64,
    /// Cost of materializing data across a domain boundary.
    pub materialization_cost_us: f64,
    /// Cost of staging model weights into GPU-accessible memory.
    pub weight_staging_cost_us: f64,
    /// Cost of KV cache operations (read / write per token).
    pub kv_cache_cost_us: f64,
}

impl M1CalibrationReport {
    /// Create a new calibration report with the given latency measurements.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hardware_id: impl Into<String>,
        os_build: impl Into<String>,
        hardware_model: impl Into<String>,
        metal_device: impl Into<String>,
        compiler_version: impl Into<String>,
        kernel_version: impl Into<String>,
        power_state: PowerState,
        thermal_state: ThermalState,
        memory_pressure: MemoryPressure,
        latency_p50_us: f64,
        latency_p95_us: f64,
        latency_p99_us: f64,
        latency_variance: f64,
        confidence: f64,
        contention: f64,
        metal_submission_cost_us: f64,
        cpu_cost_us: f64,
        materialization_cost_us: f64,
        weight_staging_cost_us: f64,
        kv_cache_cost_us: f64,
    ) -> Self {
        Self {
            hardware_id: hardware_id.into(),
            os_build: os_build.into(),
            hardware_model: hardware_model.into(),
            metal_device: metal_device.into(),
            compiler_version: compiler_version.into(),
            kernel_version: kernel_version.into(),
            power_state,
            thermal_state,
            memory_pressure,
            latency_p50_us,
            latency_p95_us,
            latency_p99_us,
            latency_variance,
            confidence,
            contention,
            metal_submission_cost_us,
            cpu_cost_us,
            materialization_cost_us,
            weight_staging_cost_us,
            kv_cache_cost_us,
        }
    }

    /// Create a calibration report with plausible default values for a stock
    /// Apple M1 MacBook Pro (macOS 15.x, on AC power, nominal temperature).
    ///
    /// These defaults are rough estimates suitable for development; production
    /// use should run actual calibration measurements.
    pub fn plausible_defaults() -> Self {
        Self {
            hardware_id: "MacBookPro17,1".into(),
            os_build: "24F82".into(),
            hardware_model: "Mac14,2".into(),
            metal_device: "Apple M1".into(),
            compiler_version: "Apple Metal 3.1".into(),
            kernel_version: "macOS 15.0 (24F82)".into(),
            power_state: PowerState::Plugged,
            thermal_state: ThermalState::Nominal,
            memory_pressure: MemoryPressure::Low,
            latency_p50_us: 150.0,
            latency_p95_us: 320.0,
            latency_p99_us: 500.0,
            latency_variance: 12_000.0,
            confidence: 0.85,
            contention: 0.05,
            metal_submission_cost_us: 25.0,
            cpu_cost_us: 5.0,
            materialization_cost_us: 40.0,
            weight_staging_cost_us: 200.0,
            kv_cache_cost_us: 3.0,
        }
    }

    /// Returns a deterministic digest (hex-encoded) of the identifying and
    /// measurement fields in this report.  Two reports with the same digest
    /// are functionally identical for cost-model purposes.
    pub fn digest(&self) -> String {
        // Collect identifying + measurement fields into a canonical byte
        // sequence and hash with deterministic FNV-1a.
        let mut buf = Vec::new();
        buf.extend_from_slice(self.hardware_id.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.hardware_model.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.os_build.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.metal_device.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.compiler_version.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.kernel_version.as_bytes());
        buf.push(0);
        // Encode power/thermal/memory as canonical text
        buf.extend_from_slice(format!("{:?}", self.power_state).as_bytes());
        buf.push(0);
        buf.extend_from_slice(format!("{:?}", self.thermal_state).as_bytes());
        buf.push(0);
        buf.extend_from_slice(format!("{:?}", self.memory_pressure).as_bytes());
        buf.push(0);
        // Latency measurements
        buf.extend_from_slice(&self.latency_p50_us.to_le_bytes());
        buf.extend_from_slice(&self.latency_p95_us.to_le_bytes());
        buf.extend_from_slice(&self.latency_p99_us.to_le_bytes());
        buf.extend_from_slice(&self.latency_variance.to_le_bytes());
        buf.extend_from_slice(&self.confidence.to_le_bytes());
        buf.extend_from_slice(&self.contention.to_le_bytes());
        // Domain-transition costs
        buf.extend_from_slice(&self.metal_submission_cost_us.to_le_bytes());
        buf.extend_from_slice(&self.cpu_cost_us.to_le_bytes());
        buf.extend_from_slice(&self.materialization_cost_us.to_le_bytes());
        buf.extend_from_slice(&self.weight_staging_cost_us.to_le_bytes());
        buf.extend_from_slice(&self.kv_cache_cost_us.to_le_bytes());

        format!("{:016x}", fnv1a_hash(&buf))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_report_roundtrips_serde() {
        let report = M1CalibrationReport::plausible_defaults();
        let json = serde_json::to_string_pretty(&report).expect("serialize");
        let restored: M1CalibrationReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(report, restored);
    }

    #[test]
    fn calibration_report_has_plausible_values() {
        let r = M1CalibrationReport::plausible_defaults();
        assert_eq!(r.hardware_id, "MacBookPro17,1");
        assert_eq!(r.os_build, "24F82");
        assert_eq!(r.hardware_model, "Mac14,2");
        assert_eq!(r.metal_device, "Apple M1");
        assert_eq!(r.compiler_version, "Apple Metal 3.1");
        assert_eq!(r.kernel_version, "macOS 15.0 (24F82)");
        assert_eq!(r.power_state, PowerState::Plugged);
        assert_eq!(r.thermal_state, ThermalState::Nominal);
        assert_eq!(r.memory_pressure, MemoryPressure::Low);

        // Latencies should be positive and p50 <= p95 <= p99
        assert!(r.latency_p50_us > 0.0);
        assert!(r.latency_p95_us >= r.latency_p50_us);
        assert!(r.latency_p99_us >= r.latency_p95_us);

        // Confidence in [0, 1], contention in [0, 1]
        assert!((0.0..=1.0).contains(&r.confidence));
        assert!((0.0..=1.0).contains(&r.contention));

        // All domain-transition costs are positive
        assert!(r.metal_submission_cost_us > 0.0);
        assert!(r.cpu_cost_us > 0.0);
        assert!(r.materialization_cost_us > 0.0);
        assert!(r.weight_staging_cost_us > 0.0);
        assert!(r.kv_cache_cost_us > 0.0);
    }

    #[test]
    fn calibration_report_constructor() {
        let r = M1CalibrationReport::new(
            "MacBookPro17,1",
            "24F82",
            "Mac14,2",
            "Apple M1 Pro",
            "Apple Metal 3.2",
            "macOS 15.1 (24G42)",
            PowerState::OnBattery,
            ThermalState::Fair,
            MemoryPressure::Moderate,
            200.0,
            450.0,
            800.0,
            25_000.0,
            0.7,
            0.3,
            30.0,
            8.0,
            60.0,
            250.0,
            5.0,
        );
        assert_eq!(r.power_state, PowerState::OnBattery);
        assert_eq!(r.hardware_model, "Mac14,2");
        assert_eq!(r.metal_device, "Apple M1 Pro");
        assert_eq!(r.thermal_state, ThermalState::Fair);
        assert_eq!(r.latency_p50_us, 200.0);
        assert_eq!(r.metal_submission_cost_us, 30.0);
    }

    #[test]
    fn calibration_report_digest_is_deterministic() {
        let r1 = M1CalibrationReport::plausible_defaults();
        let r2 = M1CalibrationReport::plausible_defaults();
        assert_eq!(r1.digest(), r2.digest(), "same report → same digest");

        let r3 = M1CalibrationReport::new(
            "MacBookPro17,1",
            "24F82",
            "Mac14,2",
            "Apple M1 Pro",
            "Apple Metal 3.2",
            "macOS 15.1 (24G42)",
            PowerState::OnBattery,
            ThermalState::Fair,
            MemoryPressure::Moderate,
            200.0,
            450.0,
            800.0,
            25_000.0,
            0.7,
            0.3,
            30.0,
            8.0,
            60.0,
            250.0,
            5.0,
        );
        assert_ne!(
            r1.digest(),
            r3.digest(),
            "different fields → different digest"
        );

        // hex-formatted, 16 hex digits
        let d = r1.digest();
        assert_eq!(d.len(), 16);
        assert!(d.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
