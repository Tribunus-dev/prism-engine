//! Intel NPU device probe — enumerates Intel NPU accelerators on Linux.
//!
//! Intel NPUs (Meteor Lake, Arrow Lake, Lunar Lake) expose a compute
//! accelerator interface via the `intel_vpu` / `intel_npu` kernel driver.
//! Devices appear as `/dev/accel/accel*` character devices; sysfs provides
//! model name and hardware properties at `/sys/class/accel/accel*/`.
//!
//! NPU-local SRAM is managed entirely by the kernel driver — there is no
//! user-space allocatable VRAM, so `total_bytes` is set to `0`.
//!
//! Platform-gated: actual probe only on Linux; non-Linux returns empty Vec.

use super::DeviceProbe;
use crate::device::DeviceInfo;

/// Probes Intel NPU devices via `/dev/accel/accel*` device nodes.
pub struct IntelNpuProbe;

#[cfg(target_os = "linux")]
mod platform {
    use std::fs;
    use std::path::{Path, PathBuf};

    use crate::device::{BackendKind, DeviceId, DeviceInfo, DeviceKind, DeviceMemoryInfo};

    /// Maximum number of accelerators to check.
    const MAX_ACCEL_DEVICES: u32 = 64;

    /// Check whether the Intel NPU driver is present.
    fn driver_present() -> bool {
        Path::new("/dev/accel").exists()
    }

    /// Read the NPU model name from sysfs.
    fn read_device_name(index: u32) -> String {
        let device_dir = format!("/sys/class/accel/accel{index}/device");

        // Try device/name first.
        let name_path = PathBuf::from(format!("{device_dir}/name"));
        if let Ok(content) = fs::read_to_string(&name_path) {
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }

        // Fallback: check DRIVER from uevent.
        let uevent_path = PathBuf::from(format!("{device_dir}/uevent"));
        if let Ok(content) = fs::read_to_string(&uevent_path) {
            for line in content.lines() {
                if let Some(driver) = line.strip_prefix("DRIVER=") {
                    return format!("Intel NPU ({driver})");
                }
            }
        }

        format!("Intel NPU accel{index}")
    }

    /// Probe a single accelerator device by index.
    fn probe_accel(index: u32) -> Option<DeviceInfo> {
        let dev_path = format!("/dev/accel/accel{index}");
        if !Path::new(&dev_path).exists() {
            return None;
        }

        let name = read_device_name(index);

        Some(DeviceInfo {
            id: DeviceId(0), // registry reassigns
            kind: DeviceKind::Npu,
            backend: BackendKind::LevelZero,
            name,
            vendor: "Intel".to_string(),
            driver_version: String::new(),
            memory: DeviceMemoryInfo {
                total_bytes: 0, // NPU-local SRAM is driver-managed
                free_bytes: 0,
                bandwidth_gb_per_sec: 0.0, // unknown / driver-managed
                unified_with_cpu: false,
            },
            compute_units: 0,
            clock_mhz: 0,
            ane_cores: 1, // single neural engine
            supports_f16: false,
            supports_bf16: false,
            supports_int8: false,
            supports_ternary: false,
            pcie_link: None,
        })
    }

    /// Enumerate all Intel NPU devices on the system.
    pub(super) fn discover_devices() -> Vec<DeviceInfo> {
        if !driver_present() {
            return Vec::new();
        }

        let mut devices = Vec::new();
        for i in 0..MAX_ACCEL_DEVICES {
            if let Some(device) = probe_accel(i) {
                devices.push(device);
            } else {
                // Stop at the first gap — accel indices are contiguous.
                break;
            }
        }
        devices
    }
}

impl DeviceProbe for IntelNpuProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        #[cfg(target_os = "linux")]
        {
            return platform::discover_devices();
        }

        #[cfg(not(target_os = "linux"))]
        Vec::new()
    }

    fn name(&self) -> &'static str {
        "intel_npu"
    }
}
