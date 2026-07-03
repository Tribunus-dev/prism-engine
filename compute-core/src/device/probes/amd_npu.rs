//! AMD XDNA NPU device probe — discovers AMD NPUs via sysfs on Linux.
//!
//! AMD NPUs (e.g. Ryzen AI XDNA) appear under `/sys/class/amdxdna/` or
//! as `/dev/dri/renderD*` devices bound to the `amdxdna` kernel driver.
//! This probe discovers them and reports basic properties.

#[cfg(target_os = "linux")]
use crate::device::{BackendKind, DeviceKind, DeviceMemoryInfo};
use crate::device::DeviceInfo;
use super::DeviceProbe;

/// Probes AMD XDNA NPU devices via sysfs on Linux.
pub struct AmdNpuProbe;

impl DeviceProbe for AmdNpuProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        probe_amd_npu()
    }

    fn name(&self) -> &'static str {
        "amd_npu"
    }
}

#[cfg(target_os = "linux")]
fn probe_amd_npu() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    // Primary path: enumerate /sys/class/amdxdna/
    if let Ok(entries) = std::fs::read_dir("/sys/class/amdxdna/") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            // Each entry is a symlink to the PCI device directory
            let device_path = entry.path();
            let npu_name_path = device_path.join("name");
            let npu_name = std::fs::read_to_string(&npu_name_path)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| format!("AMD XDNA NPU ({})", name_str));

            devices.push(DeviceInfo {
                id: crate::device::DeviceId(0), // registry reassigns real ids
                kind: DeviceKind::Npu,
                backend: BackendKind::Rocm,
                name: npu_name,
                vendor: "AMD".to_string(),
                driver_version: String::new(),
                memory: DeviceMemoryInfo {
                    total_bytes: 0, // NPU shares system RAM; no dedicated VRAM
                    free_bytes: 0,
                    bandwidth_gb_per_sec: 0.0,
                    unified_with_cpu: true,
                },
                compute_units: 0,
                clock_mhz: 0,
                ane_cores: 0,
                supports_f16: false,
                supports_bf16: false,
                supports_int8: false,
                supports_ternary: false,
                pcie_link: None,
            });
        }
    }

    // Fallback: enumerate /dev/dri/renderD* and check for amdxdna driver
    if devices.is_empty() {
        if let Ok(entries) = std::fs::read_dir("/dev/dri/") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with("renderD") {
                    continue;
                }

                // Check the sysfs uevent for DRIVER=amdxdna
                let uevent_path = format!("/sys/class/drm/{}/device/uevent", name_str);
                if let Ok(uevent) = std::fs::read_to_string(&uevent_path) {
                    if !uevent.lines().any(|l| l.trim() == "DRIVER=amdxdna") {
                        continue;
                    }

                    let npu_name_path = format!("/sys/class/drm/{}/device/name", name_str);
                    let npu_name = std::fs::read_to_string(&npu_name_path)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_else(|_| "AMD XDNA NPU".to_string());

                    devices.push(DeviceInfo {
                        id: crate::device::DeviceId(0),
                        kind: DeviceKind::Npu,
                        backend: BackendKind::Rocm,
                        name: npu_name,
                        vendor: "AMD".to_string(),
                        driver_version: String::new(),
                        memory: DeviceMemoryInfo {
                            total_bytes: 0,
                            free_bytes: 0,
                            bandwidth_gb_per_sec: 0.0,
                            unified_with_cpu: true,
                        },
                        compute_units: 0,
                        clock_mhz: 0,
                        ane_cores: 0,
                        supports_f16: false,
                        supports_bf16: false,
                        supports_int8: false,
                        supports_ternary: false,
                        pcie_link: None,
                    });
                }
            }
        }
    }

    devices
}

#[cfg(not(target_os = "linux"))]
fn probe_amd_npu() -> Vec<DeviceInfo> {
    Vec::new()
}
