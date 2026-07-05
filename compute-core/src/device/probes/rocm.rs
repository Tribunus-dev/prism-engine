//! AMD ROCm GPU device probe — discovers AMD GPUs via the HIP runtime API.
//!
//! Requires the `rocm-probe` feature (default OFF): the FFI block links
//! `libamdhip64`, which only exists on Linux boxes with the ROCm toolkit.
//! An unconditional `#[link]` here made EVERY Linux build of the crate fail
//! at link time on machines without ROCm (CI included) — the probe must be
//! opt-in. Without the feature, `probe()` compiles to an empty result.

use super::DeviceProbe;
#[cfg_attr(
    not(all(target_os = "linux", feature = "rocm-probe")),
    allow(unused_imports)
)]
use crate::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo, PcieLinkInfo};

/// Probes AMD GPUs accessible through the ROCm (HIP) runtime.
pub struct RocmProbe;

impl DeviceProbe for RocmProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        probe_rocm_devices()
    }

    fn name(&self) -> &'static str {
        "rocm"
    }
}

#[cfg(all(target_os = "linux", feature = "rocm-probe"))]
mod platform {
    use super::*;
    use std::ffi::CStr;
    use std::mem;

    const HIP_SUCCESS: i32 = 0;

    #[repr(C)]
    struct hipDeviceProp_t {
        name: [i8; 256],
        totalGlobalMem: usize,
        _pad1: [u8; 48], // offset 264→312: sharedMemPerBlock..maxGridSize
        clockRate: i32,
        _pad2: [u8; 28], // offset 316→344: memoryClockRate..minor
        multiProcessorCount: i32,
        _pad3: [u8; 1024], // tail — skip everything after
    }

    #[link(name = "amdhip64")]
    extern "C" {
        fn hipGetDeviceCount(count: *mut i32) -> i32;
        fn hipGetDeviceProperties(props: *mut hipDeviceProp_t, device: i32) -> i32;
    }

    pub fn probe_rocm_devices() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        let mut count: i32 = 0;
        let err = unsafe { hipGetDeviceCount(&mut count) };
        if err != HIP_SUCCESS || count <= 0 {
            return devices;
        }

        for _i in 0..count {
            let mut props: hipDeviceProp_t = unsafe { mem::zeroed() };
            let err = unsafe { hipGetDeviceProperties(&mut props, _i) };
            if err != HIP_SUCCESS {
                continue;
            }

            let name = unsafe { CStr::from_ptr(props.name.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            let memory = DeviceMemoryInfo {
                total_bytes: props.totalGlobalMem as u64,
                free_bytes: 0,
                bandwidth_gb_per_sec: estimate_bandwidth(&name, props.clockRate),
                unified_with_cpu: false,
            };

            // Compute before the struct literal: `name` moves into the `name:`
            // field (fields evaluate in source order), so borrowing it in the
            // later `pcie_link` initializer was a borrow-after-move.
            let pcie_bandwidth = estimate_pcie_bandwidth(&name);
            devices.push(DeviceInfo {
                id: crate::device::DeviceId(0), // placeholder — registry reassigns
                kind: DeviceKind::GpuDiscrete,
                backend: BackendKind::Rocm,
                name,
                vendor: "AMD".into(),
                driver_version: String::new(),
                memory,
                compute_units: props.multiProcessorCount as u32,
                clock_mhz: (props.clockRate / 1000) as u32,
                ane_cores: 0,
                supports_f16: true,
                supports_bf16: false, // only CDNA2+/RDNA3+ native BF16
                supports_int8: true,
                supports_ternary: false,
                pcie_link: Some(PcieLinkInfo {
                    generation: 4,
                    lanes: 16,
                    max_speed_gb_per_sec: pcie_bandwidth,
                }),
            });
        }

        devices
    }

    fn estimate_pcie_bandwidth(name: &str) -> f64 {
        if name.contains("MI300") || name.contains("MI350") {
            64.0
        } else if name.contains("MI250") {
            32.0
        } else {
            16.0
        }
    }

    fn estimate_bandwidth(name: &str, clock_khz: i32) -> f64 {
        if name.contains("MI300") || name.contains("MI350") {
            5300.0
        } else if name.contains("MI250") {
            3200.0
        } else if name.contains("MI210") {
            1600.0
        } else if name.contains("RX 7900") || name.contains("Radeon RX 7900") {
            960.0
        } else if name.contains("RX 7800") || name.contains("Radeon RX 7800") {
            620.0
        } else if name.contains("RX 7700") || name.contains("Radeon RX 7700") {
            520.0
        } else if name.contains("RX 7600") || name.contains("Radeon RX 7600") {
            480.0
        } else if name.contains("Instinct") || name.contains("MI") {
            2000.0
        } else if name.contains("Pro") || name.contains("WX") {
            600.0
        } else if name.contains("Radeon") {
            500.0
        } else {
            (clock_khz as f64) / 500_000.0 * 200.0
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "rocm-probe")))]
mod platform {
    use super::*;

    pub fn probe_rocm_devices() -> Vec<DeviceInfo> {
        Vec::new()
    }
}

fn probe_rocm_devices() -> Vec<DeviceInfo> {
    platform::probe_rocm_devices()
}
