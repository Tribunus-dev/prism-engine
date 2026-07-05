//! Apple Neural Engine (ANE) device probe.
//!
//! On macOS, probes the ANE via IOKit (`ioreg -rc AppleNeuralEngine`) to
//! detect the number of ANE cores. Optionally verifies ANE is functional
//! using the private ANE runtime when `mlx-backend` or `prism-backend`
//! features are enabled.
//!
//! On non-macOS platforms, returns an empty device list.

use super::DeviceProbe;
use crate::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo};

/// Probes the Apple Neural Engine via IOKit ioreg.
pub struct AneProbe;

impl DeviceProbe for AneProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        #[cfg(target_os = "macos")]
        {
            return probe_macos_ane();
        }
        #[cfg(not(target_os = "macos"))]
        {
            Vec::new()
        }
    }

    fn name(&self) -> &'static str {
        "ane"
    }
}

/// macOS ANE probe implementation.
#[cfg(target_os = "macos")]
fn probe_macos_ane() -> Vec<DeviceInfo> {
    let ane_cores = detect_ane_cores();

    // Optionally verify ANE is functional via AneProgram::init().
    let functional = {
        #[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
        {
            crate::ane_bridge::AneProgram::init().is_ok()
        }
        #[cfg(not(any(feature = "mlx-backend", feature = "prism-backend")))]
        {
            false
        }
    };

    if !functional || ane_cores == 0 {
        return Vec::new();
    }

    vec![DeviceInfo {
        id: crate::device::DeviceId(0),
        kind: DeviceKind::Npu,
        backend: BackendKind::Ane,
        name: "Apple Neural Engine".into(),
        vendor: "Apple".into(),
        driver_version: String::new(),
        memory: DeviceMemoryInfo {
            total_bytes: 0,
            free_bytes: 0,
            bandwidth_gb_per_sec: 0.0,
            unified_with_cpu: true,
        },
        compute_units: ane_cores,
        clock_mhz: 0,
        ane_cores,
        supports_f16: true,
        supports_bf16: false,
        supports_int8: true,
        supports_ternary: false,
        pcie_link: None,
    }]
}

/// Detect the number of ANE (Apple Neural Engine) cores via ioreg.
#[cfg(target_os = "macos")]
fn detect_ane_cores() -> u32 {
    if let Ok(output) = std::process::Command::new("ioreg")
        .args(["-rc", "AppleNeuralEngine"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("ANE.CoreCount") || line.contains("CoreCount") {
                    if let Some(val) = line.split('=').nth(1) {
                        if let Ok(n) = val.trim().parse::<u32>() {
                            return n;
                        }
                    }
                }
            }
            if stdout.contains("AppleNeuralEngine") {
                if let Ok(chip) = std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                {
                    let chip_str = String::from_utf8_lossy(&chip.stdout);
                    if chip_str.to_lowercase().contains("ultra") {
                        return 32;
                    }
                }
                return 16;
            }
        }
    }
    16
}
