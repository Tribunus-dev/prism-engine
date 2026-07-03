//! NVIDIA CUDA GPU device probe — discovers CUDA-capable GPUs on Linux.
//!
//! Uses the CUDA Runtime API (`libcudart`) to call `cudaGetDeviceCount()`
//! and `cudaGetDeviceProperties()` for each device. The probe compiles on
//! any Linux target; if no CUDA runtime is linked at runtime (or no NVIDIA
//! driver is installed), `cudaGetDeviceCount` returns an error and the probe
//! returns an empty vec.
//!
//! Platform-gated: actual probe only on Linux; non-Linux returns empty Vec.

#[cfg_attr(not(target_os = "linux"), allow(unused_imports))]
use crate::device::{BackendKind, DeviceInfo, DeviceKind, DeviceMemoryInfo};
use super::DeviceProbe;

/// Probes CUDA-capable GPUs via the CUDA Runtime API.
pub struct CudaProbe;

impl DeviceProbe for CudaProbe {
    fn probe(&self) -> Vec<DeviceInfo> {
        probe_cuda_devices()
    }

    fn name(&self) -> &'static str {
        "cuda"
    }
}

// ── Linux FFI + probe logic ───────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::CStr;
    use std::mem;
    use super::*;

    const CUDA_SUCCESS: i32 = 0;

    /// Minimal `cudaDeviceProp` struct — only the fields we read, plus padding
    /// to match the CUDA 12.x layout (total ~1040 bytes).
    #[repr(C)]
    struct cudaDeviceProp {
        // Offset 0 — all fields in CUDA 12.x declaration order
        name: [i8; 256],
        _uuid: [u8; 16],
        _luid: [i8; 8],
        _luidDeviceNodeMask: u32,
        _pad0: [u8; 4],
        totalGlobalMem: usize,          // 288
        sharedMemPerBlock: usize,       // 296
        regsPerBlock: i32,              // 304
        warpSize: i32,                  // 308
        _memPitch: usize,               // 312
        maxThreadsPerBlock: i32,        // 320
        maxThreadsDim: [i32; 3],        // 324
        maxGridSize: [i32; 3],          // 336
        clockRate: i32,                 // 348
        _totalConstMem: usize,          // 352
        major: i32,                     // 360
        minor: i32,                     // 364
        _textureAlignment: usize,       // 368
        _texturePitchAlignment: usize,  // 376
        _deviceOverlap: i32,            // 384
        multiProcessorCount: i32,       // 388
        _kernelExecTimeoutEnabled: i32, // 392
        integrated: i32,                // 396
        _canMapHostMemory: i32,         // 400
        computeMode: i32,               // 404
        _maxTexture1D: i32,
        _maxTexture1DLinear: i32,
        _maxTexture1DMipmap: i32,
        _maxTexture2D: [i32; 2],
        _maxTexture2DLinear: [i32; 3],
        _maxTexture2DMipmap: [i32; 2],
        _maxTexture2DGather: [i32; 2],
        _maxTexture3D: [i32; 3],
        _maxTexture3DAlt: [i32; 3],
        _maxTexture1DLayered: [i32; 2],
        _maxTexture2DLayered: [i32; 3],
        _maxSurface1D: i32,
        _maxSurface2D: [i32; 2],
        _maxSurface3D: [i32; 3],
        _maxSurface1DLayered: [i32; 2],
        _maxSurface2DLayered: [i32; 3],
        _maxSurfaceCubemap: i32,
        _maxSurfaceCubemapLayered: [i32; 2],
        _surfaceAlignment: usize,       // ~584
        _concurrentKernels: i32,        // 592
        _ECCEnabled: i32,               // 596
        pciBusID: i32,                  // 600
        pciDeviceID: i32,               // 604
        pciDomainID: i32,               // 608
        _tccDriver: i32,                // 612
        _asyncEngineCount: i32,         // 616
        _unifiedAddressing: i32,        // 620
        memoryClockRate: i32,           // 624
        memoryBusWidth: i32,            // 628
        l2CacheSize: i32,               // 632
        _maxThreadsPerMultiProcessor: i32,
        _maxBlocksPerMultiProcessor: i32,
        _concurrentManagedAccess: i32,
        _computePreemptionSupported: i32,
        _canUseHostPointerForRegisteredMem: i32,
        _cooperativeLaunch: i32,
        _cooperativeMultiDeviceLaunch: i32,
        _pageableMemoryAccess: i32,
        _pageableMemoryAccessUsesHostPageTables: i32,
        _directManagedMemAccessFromHost: i32,
        _managedMemory: i32,
        _globalL1CacheSupported: i32,
        _localL1CacheSupported: i32,
        _isMultiGpuBoard: i32,
        _multiGpuBoardGroupID: i32,
        _hostNativeAtomicSupported: i32,
        _singleToDoublePrecisionPerfRatio: i32,
        _accessPolicyMaxWindowSize: i32,
        _deferredMappingCudaArraySupported: i32,
        _ipcEventSupported: i32,
        _clusterLaunch: i32,
        _unifiedMemopSupported: i32,
        _mpsEnabled: i32,
        _gpuDirectRDMASupported: i32,
        _gpuDirectRDMAFlushWritesOptions: u32,
        _gpuDirectRDMAWritesOrdering: i32,
        _memoryPoolSupportedHandleTypes: u32,
        _memoryPoolsSupported: i32,
        _hostRegisterSupported: i32,
        _hostRegisterReadOnlySupported: i32,
        _sparseCudaArraySupported: i32,
        _hostNumaId: i32,
        _deviceNumaId: i32,
        _deviceNumaConfig: i32,
        _hostNumaMultinodeIpcSupported: i32,
        _reserved: [u8; 220],          // forward-compat padding
    }

    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaGetDeviceProperties(props: *mut cudaDeviceProp, device: i32) -> i32;
    }

    pub fn probe_cuda_devices() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        let mut count: i32 = 0;
        let err = unsafe { cudaGetDeviceCount(&mut count) };
        if err != CUDA_SUCCESS || count <= 0 {
            return devices;
        }

        for i in 0..count {
            let mut props: cudaDeviceProp = unsafe { mem::zeroed() };
            let err = unsafe { cudaGetDeviceProperties(&mut props, i) };
            if err != CUDA_SUCCESS {
                continue;
            }

            let name = unsafe { CStr::from_ptr(props.name.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            let compute_capability = props.major * 10 + props.minor;

            let memory = DeviceMemoryInfo {
                total_bytes: props.totalGlobalMem as u64,
                free_bytes: 0,
                bandwidth_gb_per_sec: estimate_bandwidth(
                    props.memoryClockRate,
                    props.memoryBusWidth,
                ),
                unified_with_cpu: props.integrated != 0,
            };

            let kind = if props.integrated != 0 {
                DeviceKind::GpuIntegrated
            } else {
                DeviceKind::GpuDiscrete
            };

            devices.push(DeviceInfo {
                id: crate::device::DeviceId(i as u32),
                kind,
                backend: BackendKind::Cuda,
                name: format!("CUDA GPU {}: {}", i, name),
                vendor: "NVIDIA".into(),
                driver_version: String::new(),
                memory,
                compute_units: props.multiProcessorCount as u32,
                clock_mhz: (props.clockRate / 1000) as u32,
                ane_cores: 0,
                // Feature support derived from compute capability major.minor
                supports_f16: compute_capability >= 53,
                supports_bf16: compute_capability >= 80,
                supports_int8: compute_capability >= 61,
                supports_ternary: false,
                pcie_link: Some(crate::device::PcieLinkInfo {
                    generation: pcie_generation(&name, props.major, props.minor),
                    lanes: 16,
                    max_speed_gb_per_sec: 0.0,
                }),
            });
        }

        devices
    }

    /// Estimate memory bandwidth in GB/s from the memory clock and bus width.
    /// `clock_khz` is the memory clock in kHz; `bus_width_bits` is the memory
    /// bus width in bits. DDR memory transfers twice per clock cycle.
    fn estimate_bandwidth(clock_khz: i32, bus_width_bits: i32) -> f64 {
        if clock_khz <= 0 || bus_width_bits <= 0 {
            return 0.0;
        }
        // DDR: effective data rate = 2x the clock rate
        let clock_hz = clock_khz as f64 * 1_000.0;
        let bytes_per_transfer = bus_width_bits as f64 / 8.0;
        (clock_hz * bytes_per_transfer * 2.0) / 1_000_000_000.0
    }

    /// Heuristic PCIe generation from device generation and name.
    fn pcie_generation(name: &str, major: i32, minor: i32) -> u32 {
        let cc = major * 10 + minor;
        if name.contains("RTX 40") || name.contains("RTX 50") || name.contains("H100") || name.contains("B100") {
            5
        } else if name.contains("RTX 30") || name.contains("A100") || cc >= 80 {
            4
        } else if name.contains("RTX 20") || name.contains("GTX 16") || cc >= 70 {
            3
        } else {
            3
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::*;

    pub fn probe_cuda_devices() -> Vec<DeviceInfo> {
        Vec::new()
    }
}

fn probe_cuda_devices() -> Vec<DeviceInfo> {
    platform::probe_cuda_devices()
}
