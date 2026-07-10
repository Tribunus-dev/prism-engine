//! CPU device probe — discovers CPU cores, cache, and ISA features.

use super::DeviceProbe;
use crate::ecs::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo};

/// Probes CPU capabilities using `std::thread` and target-feature cfg checks.
pub struct CpuProbe;

impl DeviceProbe for CpuProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        vec![probe_cpu()]
    }

    fn name(&self) -> &'static str {
        "cpu"
    }
}

fn probe_cpu() -> DeviceInfo {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);

    // Estimate physical cores (assume hyperthreading: 2x logical per physical).
    let physical_cores = (logical_cores + 1) / 2;

    // Detect ARM SVE/SME vs x86 AVX via cfg.
    let (supports_f16, supports_bf16, supports_int8) = detect_isa_features();

    // Estimate total system RAM from OS.
    let total_ram_bytes = estimate_total_ram();

    let vendor = if cfg!(target_arch = "aarch64") {
        if cfg!(target_os = "macos") {
            "Apple"
        } else {
            "ARM"
        }
    } else if cfg!(target_arch = "x86_64") {
        "Intel/AMD"
    } else {
        "generic"
    };

    let arch_name = if cfg!(target_arch = "aarch64") {
        if cfg!(target_os = "macos") {
            "Apple Silicon"
        } else {
            "ARM64"
        }
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "unknown"
    };

    DeviceInfo {
        id: crate::ecs::device::DeviceId(0), // placeholder — registry assigns real ids
        kind: DeviceKind::Cpu,
        backend: BackendKind::Cpu,
        name: format!("{} CPU ({} cores)", arch_name, logical_cores),
        vendor: vendor.to_string(),
        driver_version: String::new(),
        memory: DeviceMemoryInfo {
            total_bytes: total_ram_bytes,
            free_bytes: 0, // populated by OS-specific probe if needed
            bandwidth_gb_per_sec: estimate_memory_bandwidth(),
            unified_with_cpu: true,
        },
        compute_units: physical_cores,
        clock_mhz: 0,
        ane_cores: 0,
        supports_f16,
        supports_bf16,
        supports_int8,
        supports_ternary: false,
        pcie_link: None,
    }
}

/// Detect ISA features via cfg attributes.
fn detect_isa_features() -> (bool, bool, bool) {
    // ARM NEON supports FP16, no BF16.
    // x86 AVX-512 supports FP16 (AVX-512_FP16), no BF16 without AMX.
    // Both support INT8.

    let f16 = cfg!(any(
        target_feature = "fp16c",                          // ARM NEON FP16
        target_feature = "avx512fp16",                     // x86 AVX-512 FP16
        all(target_arch = "aarch64", target_os = "macos"), // Apple Silicon NEON
    ));

    let bf16 = cfg!(any(
        target_feature = "bf16",     // ARM BF16
        target_feature = "amx-bf16", // x86 AMX BF16
    ));

    let int8 = true; // All modern CPUs support INT8

    (f16, bf16, int8)
}

/// Estimate total physical RAM. Uses OS-specific sysctl on macOS,
/// /proc/meminfo on Linux, and a fallback otherwise.
fn estimate_total_ram() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes;
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Read /proc/meminfo for MemTotal.
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let cleaned: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
                    if let Ok(kb) = cleaned.parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }

    // Fallback: use libc sysconf if available.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        // _SC_PHYS_PAGES * _SC_PAGESIZE gives total RAM.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            return (pages as u64) * (page_size as u64);
        }
    }

    8_000_000_000 // 8 GB fallback
}

/// Estimate memory bandwidth based on architecture.
fn estimate_memory_bandwidth() -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if cfg!(target_os = "macos") {
            // Apple Silicon: ~120 GB/s for M3 Max
            120.0
        } else {
            // Generic ARM: ~30 GB/s
            30.0
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // DDR5 typical: ~50 GB/s
        50.0
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        10.0
    }
}
