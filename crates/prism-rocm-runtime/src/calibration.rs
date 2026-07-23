//! Native MI300X calibration for Prism's evolutionary compiler search.

use prism_spatial_ir::cost::Mi300xCostModel;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocmCalibrationSample {
    pub operation: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub datatype: String,
    pub latency_us: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mi300xCalibration {
    pub gpu: String,
    pub rocm_version: Option<String>,
    pub measured_at_unix: u64,
    pub samples: Vec<RocmCalibrationSample>,
}

impl Mi300xCalibration {
    pub fn new(samples: Vec<RocmCalibrationSample>) -> Self {
        Self {
            gpu: "gfx942".into(),
            rocm_version: None,
            measured_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            samples,
        }
    }

    /// Run one representative FP16 GEMM through rocblas-bench.
    ///
    /// The command is intentionally externalized at this boundary so the
    /// compiler remains portable; the search consumes only the resulting
    /// Prism-owned calibration record.
    pub fn measure_gemm(m: u32, n: u32, k: u32) -> Result<RocmCalibrationSample, String> {
        let output = Command::new("rocblas-bench")
            .args([
                "-f",
                "gemm_ex",
                "-r",
                "f16_r",
                "--transposeA",
                "N",
                "--transposeB",
                "N",
                "-m",
                &m.to_string(),
                "-n",
                &n.to_string(),
                "-k",
                &k.to_string(),
                "-i",
                "5",
            ])
            .output()
            .map_err(|error| format!("failed to launch rocblas-bench: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "rocblas-bench failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let latency_us = parse_gpu_time_us(&stdout)
            .ok_or_else(|| "rocblas-bench output did not contain a GPU time".to_string())?;
        Ok(RocmCalibrationSample {
            operation: "gemm".into(),
            m,
            n,
            k,
            datatype: "f16".into(),
            latency_us,
            source: "rocblas-bench".into(),
        })
    }

    /// Convert measured GEMM latency into a conservative multiplier for the
    /// analytical MI300X model. Unmeasured workloads retain the default model.
    pub fn cost_model(&self) -> Mi300xCostModel {
        let Some(sample) = self
            .samples
            .iter()
            .find(|sample| sample.operation == "gemm")
        else {
            return Mi300xCostModel::default();
        };
        let flops = 2.0 * sample.m as f64 * sample.n as f64 * sample.k as f64;
        let ideal_us = flops / (Mi300xCostModel::default().matrix_tflops * 1.0e6);
        Mi300xCostModel::default().with_latency_multiplier(sample.latency_us / ideal_us.max(0.001))
    }
}

fn parse_gpu_time_us(output: &str) -> Option<f64> {
    for line in output.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("gpu_time") || lower.contains("gpu time") {
            for token in line.split(|c: char| !c.is_ascii_digit() && c != '.') {
                if let Ok(value) = token.parse::<f64>() {
                    if value.is_finite() && value > 0.0 {
                        return Some(value);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_spatial_ir::cost::CostModel;

    #[test]
    fn parses_rocblas_gpu_time() {
        assert_eq!(parse_gpu_time_us("gpu_time_us: 12.5"), Some(12.5));
    }

    #[test]
    fn calibration_without_samples_keeps_default_model() {
        let calibration = Mi300xCalibration::new(Vec::new());
        assert_eq!(calibration.cost_model().name(), "mi300x_gfx942");
    }
}
