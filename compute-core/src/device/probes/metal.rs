//! Metal GPU device probe — enumerates all Metal-capable GPUs on macOS.
//!
//! Uses the `metal` crate (v0.29) to call `MTLCopyAllDevices()` and query
//! each device's properties: name, unified memory, recommended working set
//! size, threadgroup limits, and feature set support.
//!
//! Feature-gated behind `metal-dispatch` or `prism-backend`.

use super::DeviceProbe;
use crate::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo, PcieLinkInfo};
use metal::Device;

/// Probes all Metal devices via `MTLCopyAllDevices()`.
pub struct MetalProbe;

impl DeviceProbe for MetalProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        let devices = Device::all();
        if devices.is_empty() {
            return Vec::new();
        }
        devices.into_iter().map(probe_metal_device).collect()
    }

    fn name(&self) -> &'static str {
        "metal"
    }
}

#[allow(deprecated)]
fn probe_metal_device(device: Device) -> DeviceInfo {
    let name = device.name().to_string();
    let headless = device.is_headless();
    let low_power = device.is_low_power();
    let has_unified = device.has_unified_memory();
    let max_working_set = device.recommended_max_working_set_size();
    let _max_threads = device.max_threads_per_threadgroup();
    let _max_buffer = device.max_buffer_length();

    // Determine device kind.
    let kind = if headless || !has_unified {
        // Headless = discrete GPU with dedicated VRAM (e.g. AMD dGPU,
        // or an eGPU enclosure).
        DeviceKind::GpuDiscrete
    } else if low_power {
        // Low-power integrated GPU (Intel UHD, etc.).
        DeviceKind::GpuIntegrated
    } else {
        // Apple Silicon unified GPU.
        DeviceKind::GpuUnified
    };

    // Vendor string from name heuristics.
    let vendor = if name.contains("Apple") || name.contains("M") {
        "Apple"
    } else if name.contains("AMD") || name.contains("Radeon") {
        "AMD"
    } else if name.contains("Intel") || name.contains("HD Graphics") || name.contains("UHD") {
        "Intel"
    } else if name.contains("NVIDIA") || name.contains("GeForce") {
        "NVIDIA"
    } else {
        "unknown"
    };

    // Estimate GPU cores / compute units.
    // On Apple Silicon, each GPU core cluster has ~16 cores.
    // For non-Apple devices, use max_threads as a proxy.
    let compute_units = estimate_compute_units(&device, &name);

    // Check feature set support.
    let supports_f16 = device.supports_family(metal::MTLGPUFamily::Apple7)
        || device.supports_feature_set(metal::MTLFeatureSet::iOS_GPUFamily2_v1);
    let supports_bf16 = false; // Metal does not natively support BF16
    let supports_int8 = device.supports_family(metal::MTLGPUFamily::Apple7)
        || device.supports_feature_set(metal::MTLFeatureSet::macOS_GPUFamily2_v1);

    // Estimate memory bandwidth.
    let bandwidth = estimate_bandwidth(&kind, &name);

    // PCIe link info for discrete GPUs.
    let pcie_link = if matches!(kind, DeviceKind::GpuDiscrete) {
        Some(PcieLinkInfo {
            generation: 4, // default — could probe via IORegistry
            lanes: 16,
            max_speed_gb_per_sec: estimate_pcie_bandwidth(&name),
        })
    } else {
        None
    };

    let driver_version = get_driver_version();

    DeviceInfo {
        id: crate::device::DeviceId(0), // placeholder — registry assigns real ids
        kind,
        backend: BackendKind::Metal,
        name,
        vendor: vendor.to_string(),
        driver_version,
        memory: DeviceMemoryInfo {
            total_bytes: max_working_set,
            free_bytes: 0, // Metal doesn't expose free GPU memory directly
            bandwidth_gb_per_sec: bandwidth,
            unified_with_cpu: has_unified,
        },
        compute_units,
        clock_mhz: 0, // Metal doesn't expose clock speed
        ane_cores: 0,
        supports_f16,
        supports_bf16,
        supports_int8,
        supports_ternary: true, // Prism Engine supports ternary via Metal
        pcie_link,
    }
}

/// Estimate GPU compute units from Metal device properties.
fn estimate_compute_units(device: &Device, name: &str) -> u32 {
    // Apple Silicon: extract core count from name if possible.
    if name.contains("Apple") || name.contains("M") {
        // Name format like "Apple M3 Max" or "Apple M2 Pro"
        // Try to extract known GPU core counts from the chip name.
        let lower = name.to_lowercase();
        if lower.contains("m3 ultra") {
            return 80;
        }
        if lower.contains("m3 max") {
            return 40;
        }
        if lower.contains("m3 pro") {
            return 18;
        }
        if lower.contains("m3") {
            return 10;
        }
        if lower.contains("m2 ultra") {
            return 76;
        }
        if lower.contains("m2 max") {
            return 38;
        }
        if lower.contains("m2 pro") {
            return 16;
        }
        if lower.contains("m2") {
            return 10;
        }
        if lower.contains("m1 ultra") {
            return 64;
        }
        if lower.contains("m1 max") {
            return 32;
        }
        if lower.contains("m1 pro") {
            return 14;
        }
        if lower.contains("m1") {
            return 7;
        }
        // Fallback: use max_threadgroup width as proxy.
        let max_tg = device.max_threads_per_threadgroup();
        return ((max_tg.width / 32) as u32).max(8);
    }

    // Non-Apple: use max threads as rough proxy.
    let max_tg = device.max_threads_per_threadgroup();
    ((max_tg.width / 64) as u32).max(4)
}

/// Estimate memory bandwidth based on device kind and name.
fn estimate_bandwidth(kind: &DeviceKind, name: &str) -> f64 {
    let lower = name.to_lowercase();
    match kind {
        DeviceKind::GpuUnified | DeviceKind::GpuIntegrated => {
            if lower.contains("m3") {
                150.0
            } else if lower.contains("m2") {
                100.0
            } else if lower.contains("m1") {
                70.0
            } else {
                50.0
            }
        }
        DeviceKind::GpuDiscrete => {
            // PCIe 4.0 x16 = ~32 GB/s bidirectional.
            // VRAM bandwidth depends on memory type (GDDR6, GDDR6X, etc.)
            if lower.contains("radeon") || lower.contains("amd") {
                500.0 // GDDR6 typical
            } else if lower.contains("nvidia") || lower.contains("geforce") {
                600.0 // GDDR6X typical
            } else {
                256.0 // conservative
            }
        }
        _ => 10.0,
    }
}

/// Estimate PCIe bandwidth for discrete GPUs.
fn estimate_pcie_bandwidth(name: &str) -> f64 {
    let lower = name.to_lowercase();
    // PCIe 4.0 x16 = 31.5 GB/s, PCIe 3.0 x16 = 15.8 GB/s
    if lower.contains("radeon") || lower.contains("nvidia") {
        // Modern GPUs use PCIe 4.0
        32.0
    } else {
        16.0 // PCIe 3.0 fallback
    }
}

/// Get the Metal driver version string.
fn get_driver_version() -> String {
    #[cfg(target_os = "macos")]
    {
        // Read macOS version as proxy for Metal driver version.
        use std::process::Command;
        if let Ok(out) = Command::new("sw_vers").args(["-productVersion"]).output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                return format!("Metal (macOS {})", s.trim());
            }
        }
    }
    "Metal (unknown)".to_string()
}
