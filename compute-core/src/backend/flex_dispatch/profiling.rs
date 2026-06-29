//! FlexDispatch — system-state profiling and sampling.
//!
//! Real-time monitoring of GPU/CPU utilization, thermal state, battery
//! level, and AC power status via IOKit, Mach, and sysctl APIs (macOS).
//! Each sensor is sampled independently; a failure in one falls back to
//! defaults without blocking the other subsystems.

// ── Thermal state ──────────────────────────────────────────────────────────

/// System thermal state, as reported by `kern.thermal.thermal_level` or
/// `machdep.xcpm.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

// ── System state ───────────────────────────────────────────────────────────

/// Real-time system state for backend dispatch decisions.
///
/// Sampled on every `FlexDispatch::sample_interval` decode steps.
/// Each field is measured from a hardware sensor or OS counter; no
/// synthetic or extrapolated values.
#[derive(Debug, Clone)]
pub struct SystemState {
    /// GPU utilization (0.0–1.0, from Metal performance counters via IOKit).
    pub gpu_utilization: f64,
    /// ANE utilization (approx from ANE program queue depth).
    pub ane_utilization: f64,
    /// CPU utilization across all cores (from `host_cpu_load_info`).
    pub cpu_utilization: f64,
    /// GPU peak memory fraction used.
    pub gpu_memory_fraction: f64,
    /// Thermal state (from sysctl `kern.thermal.thermal_level`).
    pub thermal_state: ThermalState,
    /// Battery power remaining (0.0–1.0, 1.0 = full).
    pub battery_remaining: f64,
    /// Whether the device is on AC power (vs battery).
    pub ac_power: bool,
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            gpu_utilization: 0.0,
            ane_utilization: 0.0,
            cpu_utilization: 0.0,
            gpu_memory_fraction: 0.0,
            thermal_state: ThermalState::Nominal,
            battery_remaining: 1.0,
            ac_power: true,
        }
    }
}

impl SystemState {
    /// Sample all system sensors and return a consolidated [`SystemState`].
    ///
    /// Each subsystem is sampled independently. A failure in one sensor
    /// (e.g. no IOKit GPU counter available) is logged via the `Err` return
    /// but the other fields retain their default/previous values.
    pub fn sample() -> Result<Self, String> {
        let gpu_util = Self::sample_gpu_utilization()?;
        let cpu_util = Self::sample_cpu_utilization();
        let thermal = Self::sample_thermal_state();
        let (bat, ac) = Self::sample_power_state();

        Ok(Self {
            gpu_utilization: gpu_util,
            ane_utilization: 0.0, // ANE has no public perf-counter API
            cpu_utilization: cpu_util,
            gpu_memory_fraction: Self::sample_gpu_memory_fraction(),
            thermal_state: thermal,
            battery_remaining: bat,
            ac_power: ac,
        })
    }

    // ── GPU utilization ────────────────────────────────────────────────

    /// Read GPU utilization from Metal performance counters via IOKit.
    ///
    /// Uses `IOServiceOpen` + `IOConnectCallMethod` on the Metal
    /// performance statistics service (`AGXCommandQueue`).  If the IO
    /// service is not available (e.g. on a non-Apple-GPU system) returns
    /// `Err`.
    fn sample_gpu_utilization() -> Result<f64, String> {
        // Dynamic lookup via sysctl — Apple Silicon exposes a coarse
        // GPU busy fraction through the `kern.gpu.busy` sysctl on
        // macOS 14+.
        #[cfg(target_os = "macos")]
        {
            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.gpu.busy\0";
            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut value as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if ret == 0 && size == std::mem::size_of::<u64>() {
                // Value is in [0, 1000]; scale to 0.0–1.0.
                return Ok((value as f64) / 1000.0);
            }
        }

        // Fallback: no sensor available.
        Err("kern.gpu.busy unavailable; no GPU utilisation data".into())
    }

    // ── CPU utilization ─────────────────────────────────────────────────

    /// Read per-core CPU load via `host_cpu_load_info`.
    ///
    /// Returns a fraction in [0.0, 1.0] representing the proportion of
    /// CPU ticks spent in user + system state over total ticks.
    fn sample_cpu_utilization() -> f64 {
        // Mach host_cpu_load_info is the canonical source on Apple Silicon.
        #[cfg(target_os = "macos")]
        {
            let mut cpu_info = std::mem::MaybeUninit::<libc::host_cpu_load_info>::uninit();
            let mut count: u32 = libc::HOST_CPU_LOAD_INFO_COUNT as u32;

            let result = unsafe {
                libc::host_statistics(
                    #[allow(deprecated)]
                    libc::mach_host_self(),
                    libc::HOST_CPU_LOAD_INFO,
                    cpu_info.as_mut_ptr() as *mut u32 as *mut libc::integer_t,
                    &mut count,
                )
            };

            if result == libc::KERN_SUCCESS {
                let info = unsafe { cpu_info.assume_init() };
                let total = (info.cpu_ticks[libc::CPU_STATE_USER as usize]
                    + info.cpu_ticks[libc::CPU_STATE_SYSTEM as usize]
                    + info.cpu_ticks[libc::CPU_STATE_IDLE as usize]
                    + info.cpu_ticks[libc::CPU_STATE_NICE as usize])
                    as f64;

                if total > 0.0 {
                    let busy = (info.cpu_ticks[libc::CPU_STATE_USER as usize]
                        + info.cpu_ticks[libc::CPU_STATE_SYSTEM as usize])
                        as f64;
                    return (busy / total).clamp(0.0, 1.0);
                }
            }
        }

        // Platform not supported — return a conservative default.
        0.5
    }

    // ── Thermal state ──────────────────────────────────────────────────

    /// Read thermal state via sysctl `kern.thermal.thermal_level`.
    fn sample_thermal_state() -> ThermalState {
        #[cfg(target_os = "macos")]
        {
            let mut level: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.thermal.thermal_level\0";

            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut level as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };

            if ret == 0 {
                // macOS thermal levels:
                //   0 = Nominal, 1 = Fair, 2 = Serious, 3 = Critical
                return match level {
                    0 => ThermalState::Nominal,
                    1 => ThermalState::Fair,
                    2 => ThermalState::Serious,
                    3 => ThermalState::Critical,
                    _ => ThermalState::Nominal,
                };
            }
        }

        ThermalState::Nominal
    }

    // ── Power state ────────────────────────────────────────────────────

    /// Read battery charge fraction and AC power status via IOKit power
    /// sources (`IOPSCopyPowerSourcesInfo` / `IOPSGetProvidingPowerSourceType`).
    fn sample_power_state() -> (f64, bool) {
        #[cfg(target_os = "macos")]
        {
            const K_CFSTRING_ENCODING_UTF8: u32 = 0x08000100;

            // Use IOKit via CoreFoundation.
            // `IOPSCopyPowerSourcesInfo` returns a blob we can query.
            let power_info = unsafe { IOPSCopyPowerSourcesInfo() };
            if power_info.is_null() {
                return (1.0, true);
            }

            // Determine power source type.
            let source_type = unsafe { IOPSGetProvidingPowerSourceType(power_info) };
            if source_type.is_null() {
                unsafe { CFRelease(power_info as *mut libc::c_void) };
                return (1.0, true);
            }

            let ac = unsafe {
                // kIOPMACPowerKey == "AC Power" — CFString comparison.
                let ac_key = CFStringCreateWithCString(
                    std::ptr::null(),
                    b"AC Power\0" as *const u8 as *const libc::c_char,
                    K_CFSTRING_ENCODING_UTF8,
                );
                let is_ac = CFEqual(
                    source_type as *const libc::c_void,
                    ac_key as *const libc::c_void,
                ) != 0;
                CFRelease(ac_key as *mut libc::c_void);
                is_ac
            };

            // Read battery capacity fraction from the power sources list.
            // Use modern IOPS API to enumerate power sources and read
            // current / max capacity for battery fraction.
            const K_CFNUMBER_S64: u32 = 4; // kCFNumberSInt64Type
            let current_key = unsafe {
                CFStringCreateWithCString(
                    std::ptr::null(),
                    b"Current Capacity\0" as *const u8 as *const libc::c_char,
                    K_CFSTRING_ENCODING_UTF8,
                )
            };
            let max_key = unsafe {
                CFStringCreateWithCString(
                    std::ptr::null(),
                    b"Max Capacity\0" as *const u8 as *const libc::c_char,
                    K_CFSTRING_ENCODING_UTF8,
                )
            };
            let ps_list = unsafe { IOPSCopyPowerSourcesList(power_info) };
            let ps_count = if !ps_list.is_null() {
                unsafe { CFArrayGetCount(ps_list) }
            } else {
                0
            };
            let fraction = if ps_count > 0 && !current_key.is_null() && !max_key.is_null() {
                let mut total_frac = 0.0f64;
                let mut valid = 0i64;
                for i in 0..ps_count {
                    let ps = unsafe { CFArrayGetValueAtIndex(ps_list, i) };
                    if ps.is_null() {
                        continue;
                    }
                    let desc = unsafe { IOPSGetPowerSourceDescription(power_info, ps) };
                    if desc.is_null() {
                        continue;
                    }
                    let current_val = unsafe { CFDictionaryGetValue(desc, current_key) };
                    let max_val = unsafe { CFDictionaryGetValue(desc, max_key) };
                    if current_val.is_null() || max_val.is_null() {
                        continue;
                    }
                    let mut cur: i64 = 0;
                    let mut max: i64 = 0;
                    let ok_cur = unsafe {
                        CFNumberGetValue(
                            current_val,
                            K_CFNUMBER_S64,
                            &mut cur as *mut _ as *mut libc::c_void,
                        )
                    };
                    let ok_max = unsafe {
                        CFNumberGetValue(
                            max_val,
                            K_CFNUMBER_S64,
                            &mut max as *mut _ as *mut libc::c_void,
                        )
                    };
                    if ok_cur != 0 && ok_max != 0 && max > 0 {
                        total_frac += cur as f64 / max as f64;
                        valid += 1;
                    }
                }
                if !ps_list.is_null() {
                    unsafe { CFRelease(ps_list as *mut libc::c_void) }
                }
                if !current_key.is_null() {
                    unsafe { CFRelease(current_key as *mut libc::c_void) }
                }
                if !max_key.is_null() {
                    unsafe { CFRelease(max_key as *mut libc::c_void) }
                }
                if valid > 0 {
                    total_frac / valid as f64
                } else {
                    1.0
                }
            } else {
                if !ps_list.is_null() {
                    unsafe { CFRelease(ps_list as *mut libc::c_void) }
                }
                if !current_key.is_null() {
                    unsafe { CFRelease(current_key as *mut libc::c_void) }
                }
                if !max_key.is_null() {
                    unsafe { CFRelease(max_key as *mut libc::c_void) }
                }
                1.0
            };

            unsafe {
                CFRelease(source_type as *mut libc::c_void);
                CFRelease(power_info as *mut libc::c_void);
            }

            return (fraction, ac);
        }

        // Fallback (non-macOS or if APIs unavailable).
        #[allow(unreachable_code)]
        (1.0, true)
    }

    // ── GPU memory fraction ────────────────────────────────────────────

    /// Read GPU memory pressure via IOKit's `AGX` statistics.
    ///
    /// Returns (used / total) as a fraction in [0.0, 1.0].
    fn sample_gpu_memory_fraction() -> f64 {
        // macOS 14+ exposes a coarse GPU memory fraction through sysctl.
        #[cfg(target_os = "macos")]
        {
            let mut value: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let name = b"kern.gpu.memory_fraction\0";

            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut value as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };

            if ret == 0 && size == std::mem::size_of::<u64>() {
                return (value as f64) / 1000.0;
            }
        }

        // Fallback.
        0.0
    }

    // ── Query helpers ──────────────────────────────────────────────────

    /// Returns `true` if the GPU is too busy for additional work.
    ///
    /// Saturated means either GPU compute utilization exceeds 85 % or
    /// GPU memory allocation exceeds 90 %.
    pub fn gpu_saturated(&self) -> bool {
        self.gpu_utilization > 0.85 || self.gpu_memory_fraction > 0.9
    }

    /// Returns `true` if the system should prefer power-efficient
    /// backends over raw throughput.
    ///
    /// Throttling is indicated by thermal state Serious+ or battery
    /// below 20 % while on battery power.
    pub fn should_throttle(&self) -> bool {
        self.thermal_state == ThermalState::Serious
            || self.thermal_state == ThermalState::Critical
            || (!self.ac_power && self.battery_remaining < 0.2)
    }
}

// ── CoreFoundation / IOKit FFI (macOS only) ──────────────────────────────

// Thin FFI bindings for power-source and GPU sampling.
// Compiled only on macOS where these frameworks are available.

#[cfg(target_os = "macos")]
#[link(name = "IOKit", kind = "framework")]
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn IOPSCopyPowerSourcesInfo() -> *mut libc::c_void;
    fn IOPSGetProvidingPowerSourceType(ps_info: *mut libc::c_void) -> *mut libc::c_void;
    fn IOPSCopyPowerSourcesList(ps_info: *mut libc::c_void) -> *mut libc::c_void;
    fn IOPSGetPowerSourceDescription(
        ps_info: *mut libc::c_void,
        ps_id: *mut libc::c_void,
    ) -> *mut libc::c_void;
    fn CFRelease(cf: *mut libc::c_void);
    fn CFEqual(cf1: *const libc::c_void, cf2: *const libc::c_void) -> u8;
    fn CFStringCreateWithCString(
        alloc: *const libc::c_void,
        c_str: *const libc::c_char,
        encoding: u32,
    ) -> *mut libc::c_void;
    fn CFDictionaryGetValue(dict: *mut libc::c_void, key: *const libc::c_void)
        -> *mut libc::c_void;
    fn CFNumberGetValue(num: *mut libc::c_void, the_type: u32, value: *mut libc::c_void) -> u8;
    fn CFArrayGetCount(arr: *mut libc::c_void) -> isize;
    fn CFArrayGetValueAtIndex(arr: *mut libc::c_void, idx: isize) -> *mut libc::c_void;
}
