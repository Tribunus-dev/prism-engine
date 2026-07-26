//! Intel Level Zero GPU device probe — enumerates Intel GPUs via the oneAPI
//! Level Zero API on Linux.
//!
//! Calls `zeInit` → `zeDriverGet` → `zeDriverGetProperties` → `zeDeviceGet`
//! → `zeDeviceGetProperties` to discover Intel integrated and discrete GPUs.
//!
//! Requires the `level-zero-probe` feature (default OFF): the FFI block links
//! `libze_loader`, which only exists where the Intel graphics/oneAPI stack is
//! installed. An unconditional `#[link]` here made EVERY Linux build of the
//! crate fail at link time on machines without it (CI included) — the probe
//! must be opt-in. Without the feature (or off Linux), `probe()` returns an
//! empty vec.

use super::DeviceProbe;
#[cfg_attr(
    not(all(target_os = "linux", feature = "level-zero-probe")),
    allow(unused_imports)
)]
use crate::ecs::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo, PcieLinkInfo};

/// Probes Intel GPUs via the Level Zero driver API.
pub struct LevelZeroProbe;

impl DeviceProbe for LevelZeroProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        #[cfg(all(target_os = "linux", feature = "level-zero-probe"))]
        {
            probe_level_zero_devices()
        }
        #[cfg(not(all(target_os = "linux", feature = "level-zero-probe")))]
        {
            Vec::new()
        }
    }

    fn name(&self) -> &'static str {
        "level_zero"
    }
}

// ── Linux FFI + probe logic ───────────────────────────────────────────────

#[cfg(all(target_os = "linux", feature = "level-zero-probe"))]
mod ffi {
    #![allow(non_camel_case_types, dead_code)]

    pub type ze_result_t = i32;
    pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

    // ze_structure_type_t enumerants (Level Zero v1.0+)
    pub const ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 1;
    pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 3;

    // ze_device_type_t
    pub const ZE_DEVICE_TYPE_GPU: u32 = 1;

    // ze_device_property_flags_t
    pub const ZE_DEVICE_PROPERTY_FLAG_INTEGRATED: u32 = 1 << 0;
    pub const ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE: u32 = 1 << 1;
    pub const ZE_DEVICE_PROPERTY_FLAG_ECC: u32 = 1 << 2;
    pub const ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING: u32 = 1 << 3;

    #[repr(C)]
    pub struct ze_driver_properties_t {
        pub stype: u32,
        pub pNext: *const std::ffi::c_void,
        pub uuid: [u8; 16],
        pub driverVersion: u32,
    }

    /// Level Zero v1.17 `ze_device_properties_t`.
    ///
    /// Fields match the spec layout exactly (verified against ze_api.h).
    /// WARNING: every field offset matters — a misaligned layout reads garbage.
    #[repr(C)]
    pub struct ze_device_properties_t {
        pub stype: u32,                        // 0
        pub pNext: *const std::ffi::c_void,    // 8
        pub deviceType: u32,                   // 16
        pub vendorId: u32,                     // 20
        pub deviceId: u32,                     // 24
        pub flags: u32,                        // 28 — must be before subdeviceId
        pub subdeviceId: u32,                  // 32
        pub coreClockRate: u32,                // 36
        pub maxMemAllocSize: u64,              // 40
        pub maxHardwareContexts: u32,          // 48
        pub maxCommandQueuePriority: u32,      // 52
        pub numThreadsPerEU: u32,              // 56
        pub physicalEUSimdWidth: u32,          // 60
        pub numEUsPerSubslice: u32,            // 64
        pub numSubslicesPerSlice: u32,         // 68
        pub numSlices: u32,                    // 72
        pub timerResolution: u64,              // 76
        pub timestampValidBits: u32,           // 84
        pub kernelTimestampValidBits: u32,     // 88
        pub uuid: [u8; 16],                    // 92
        pub name: [std::os::raw::c_char; 256], // 108
    }

    #[link(name = "ze_loader")]
    extern "C" {
        pub fn zeInit(flags: u32) -> ze_result_t;
        pub fn zeDriverGet(pCount: *mut u32, phDrivers: *mut *mut std::ffi::c_void) -> ze_result_t;
        pub fn zeDriverGetProperties(
            hDriver: *mut std::ffi::c_void,
            pProperties: *mut ze_driver_properties_t,
        ) -> ze_result_t;
        pub fn zeDriverGetApiVersion(
            hDriver: *mut std::ffi::c_void,
            version: *mut u32,
        ) -> ze_result_t;
        pub fn zeDeviceGet(
            hDriver: *mut std::ffi::c_void,
            pCount: *mut u32,
            phDevices: *mut *mut std::ffi::c_void,
        ) -> ze_result_t;
        pub fn zeDeviceGetProperties(
            hDevice: *mut std::ffi::c_void,
            pProperties: *mut ze_device_properties_t,
        ) -> ze_result_t;
    }
}

#[cfg(all(target_os = "linux", feature = "level-zero-probe"))]
fn probe_level_zero_devices() -> Vec<DeviceInfo> {
    use ffi::*;
    use std::ffi::CStr;

    // Initialize Level Zero driver.
    let res = unsafe { zeInit(0) };
    if res != ZE_RESULT_SUCCESS {
        return Vec::new();
    }

    // Query driver count.
    let mut driver_count: u32 = 0;
    let res = unsafe { zeDriverGet(&mut driver_count, std::ptr::null_mut()) };
    if res != ZE_RESULT_SUCCESS || driver_count == 0 {
        return Vec::new();
    }

    let mut drivers: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); driver_count as usize];
    let res = unsafe { zeDriverGet(&mut driver_count, drivers.as_mut_ptr()) };
    if res != ZE_RESULT_SUCCESS {
        return Vec::new();
    }

    let mut devices = Vec::new();

    for &driver in &drivers {
        if driver.is_null() {
            continue;
        }

        // Driver version.
        let mut driver_props = ze_driver_properties_t {
            stype: ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES,
            pNext: std::ptr::null(),
            uuid: [0u8; 16],
            driverVersion: 0,
        };
        unsafe {
            zeDriverGetProperties(driver, &mut driver_props);
        }

        // API version.
        let mut api_version: u32 = 0;
        unsafe {
            zeDriverGetApiVersion(driver, &mut api_version);
        }

        // Enumerate devices on this driver.
        let mut device_count: u32 = 0;
        let res = unsafe { zeDeviceGet(driver, &mut device_count, std::ptr::null_mut()) };
        if res != ZE_RESULT_SUCCESS || device_count == 0 {
            continue;
        }

        let mut raw_devices: Vec<*mut std::ffi::c_void> =
            vec![std::ptr::null_mut(); device_count as usize];
        let res = unsafe { zeDeviceGet(driver, &mut device_count, raw_devices.as_mut_ptr()) };
        if res != ZE_RESULT_SUCCESS {
            continue;
        }

        for &raw_device in &raw_devices {
            if raw_device.is_null() {
                continue;
            }

            let mut props = ze_device_properties_t {
                stype: ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                pNext: std::ptr::null(),
                deviceType: 0,
                vendorId: 0,
                deviceId: 0,
                flags: 0,
                subdeviceId: 0,
                coreClockRate: 0,
                maxMemAllocSize: 0,
                maxHardwareContexts: 0,
                maxCommandQueuePriority: 0,
                numThreadsPerEU: 0,
                physicalEUSimdWidth: 0,
                numEUsPerSubslice: 0,
                numSubslicesPerSlice: 0,
                numSlices: 0,
                timerResolution: 0,
                timestampValidBits: 0,
                kernelTimestampValidBits: 0,
                uuid: [0u8; 16],
                name: [0i8; 256],
            };

            let res = unsafe { zeDeviceGetProperties(raw_device, &mut props) };
            if res != ZE_RESULT_SUCCESS {
                continue;
            }

            // Only GPU devices.
            if props.deviceType != ZE_DEVICE_TYPE_GPU {
                continue;
            }

            let is_integrated = (props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
            let kind = if is_integrated {
                DeviceKind::GpuIntegrated
            } else {
                DeviceKind::GpuDiscrete
            };

            // Device name from C-string.
            let name = if props.name[0] != 0 {
                unsafe { CStr::from_ptr(props.name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Intel GPU".to_string()
            };

            let vendor = match props.vendorId {
                0x8086 => "Intel".to_string(),
                _ => format!("0x{:04x}", props.vendorId),
            };

            // Compute units = execution units: slices × subslices × EUs per subslice.
            let compute_units =
                props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;

            // Memory estimation.
            // On integrated GPUs maxMemAllocSize is not VRAM but max per-allocation
            // (~50% of system RAM). Discrete Arc GPUs report actual VRAM.
            let total_bytes = if is_integrated {
                props.maxMemAllocSize / 2
            } else {
                props.maxMemAllocSize
            };

            // Bandwidth estimate in GB/s.
            let bandwidth_gb_per_sec: f64 = if is_integrated {
                80.0
            } else if name.contains("A770") || name.contains("A7") {
                560.0
            } else if name.contains("A750") || name.contains("A580") {
                512.0
            } else if name.contains("A310") || name.contains("A380") {
                186.0
            } else {
                400.0
            };

            // Driver version string from packed u32.
            let dv = driver_props.driverVersion;
            let api_major = (api_version >> 24) & 0xff;
            let api_minor = (api_version >> 16) & 0xff;
            let driver_str = format!(
                "{}.{}.{} (API {}.{})",
                (dv >> 24) & 0xff,
                (dv >> 16) & 0xff,
                dv & 0xffff,
                api_major,
                api_minor,
            );

            // Conservative ISA feature reporting for Intel Gen12+ / Arc.
            let supports_f16 = true;
            let supports_bf16 = props.numSlices > 0; // Arc and newer
            let supports_int8 = true; // DP4a on Gen12+

            // PCIe info for discrete devices.
            let pcie_link = if is_integrated {
                None
            } else {
                // Assume PCIe 4.0 x16 for Arc discrete; best-effort guess.
                Some(PcieLinkInfo {
                    generation: 4,
                    lanes: 16,
                    max_speed_gb_per_sec: bandwidth_gb_per_sec * 0.85, // PCIe is typically ~85% of memory BW
                })
            };

            devices.push(DeviceInfo {
                id: crate::ecs::device::DeviceId(devices.len() as u32),
                kind,
                backend: BackendKind::LevelZero,
                name,
                vendor,
                driver_version: driver_str,
                memory: DeviceMemoryInfo {
                    total_bytes,
                    free_bytes: 0, // LV2 does not expose free memory via device properties
                    bandwidth_gb_per_sec,
                    unified_with_cpu: is_integrated,
                },
                compute_units,
                clock_mhz: props.coreClockRate,
                ane_cores: 0,
                supports_f16,
                supports_bf16,
                supports_int8,
                supports_ternary: false,
                pcie_link,
            });
        }
    }

    devices
}
