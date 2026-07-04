//! Pluggable device probes — each probe discovers devices of a specific type.
//!
//! Probes implement [`DeviceProbe`] and are registered in [`all_probes()`].
//! The [`DeviceRegistry::discover()`] method runs all probes and collects
//! their results.

pub mod amd_npu;
pub mod cpu;

#[cfg(any(feature = "metal-dispatch", feature = "prism-backend"))]
pub mod metal;

pub mod ane;
pub mod cuda;
pub mod intel_npu;
/// Level Zero GPU probe — available on Linux with Intel Graphics driver.
#[cfg(target_os = "linux")]
pub mod level_zero;
pub mod rocm;

use crate::device::DeviceInfo;

/// A probe discovers compute devices of a specific backend type.
pub trait DeviceProbe: Send + Sync {
    /// Run the probe and return discovered devices.
    fn probe(&self) -> Vec<DeviceInfo>;

    /// Human-readable name for this probe (for diagnostics).
    fn name(&self) -> &'static str;
}

/// Register all available probes for the current platform.
///
/// Returns a list of probe instances. The registry calls `probe()` on each.
pub fn all_probes() -> Vec<Vec<DeviceInfo>> {
    let mut results: Vec<Vec<DeviceInfo>> = Vec::new();

    // CPU probe — always available.
    results.push(cpu::CpuProbe.probe());

    // Metal GPU probe — available on macOS with the metal-dispatch feature.
    #[cfg(any(feature = "metal-dispatch", feature = "prism-backend"))]
    {
        results.push(metal::MetalProbe.probe());
    }

    // Intel NPU probe.
    results.push(intel_npu::IntelNpuProbe.probe());

    // AMD XDNA NPU probe — discovers AMD NPUs on Linux; empty stub otherwise.
    results.push(amd_npu::AmdNpuProbe.probe());

    // ROCm GPU probe — available on Linux with HIP runtime.
    #[cfg(target_os = "linux")]
    {
        results.push(cuda::CudaProbe.probe());
        results.push(rocm::RocmProbe.probe());
    }

    // Level Zero GPU probe — available on Linux with Intel Graphics driver.
    #[cfg(target_os = "linux")]
    {
        results.push(level_zero::LevelZeroProbe.probe());
    }

    // ANE probe — available on macOS.
    #[cfg(target_os = "macos")]
    {
        results.push(ane::AneProbe.probe());
    }

    results
}
