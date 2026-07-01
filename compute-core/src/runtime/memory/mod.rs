//! Unified memory allocation across GPU backends.
//!
//! Provides a single allocation interface for CPU+GPU accessible memory
//! across CUDA (NVIDIA), ROCm (AMD), and Level Zero (Intel) backends.
//!
//! # Architecture
//!
//! All three backends support allocating memory that is natively accessible
//! from both the CPU and GPU without explicit data transfers:
//!
//! - **CUDA**: `cudaMallocManaged` — driver-managed page migration.
//! - **ROCm**: `hipMallocManaged` — HMM-based page migration (`HSA_XNACK=1`).
//! - **Level Zero**: `zeMemAllocShared` — shared physical pages on integrated GPUs.
//!
//! # Constraint
//!
//! The `register_host_memory` pattern (wrapping an existing userspace pointer)
//! is intentionally omitted.  Two of three backends (CUDA, ROCm) support it,
//! but Level Zero has no zero-copy equivalent without a kernel driver.  The
//! engine always owns its allocations through `allocate_shared`, guaranteeing
//! a zero-copy pipeline on all platforms.

/// Identifies the active GPU memory backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Rocm,
    LevelZero,
    /// No GPU backend — uses regular `malloc` (for testing / macOS).
    Dummy,
}

/// A contiguous memory buffer accessible from both CPU and GPU.
///
/// The single `ptr` field is valid in both address spaces.
/// On integrated architectures (Apple Silicon, Intel integrated, some AMD APUs)
/// this is truly zero-copy; on discrete GPUs the driver migrates pages on
/// access.
#[derive(Debug)]
pub struct UnifiedBuffer {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

/// Detect the available GPU backend at runtime.
///
/// Order: Level Zero → ROCm → CUDA → Dummy.
/// On non-Linux platforms (macOS), this always returns Dummy.
pub fn detect_backend() -> Backend {
    // L0: check for zeInit success
    // ROCm: check for hipGetDeviceCount
    // CUDA: check for cudaGetDeviceCount
    #[cfg(target_os = "linux")]
    {
        if level_zero_available() {
            return Backend::LevelZero;
        }
        if rocm_available() {
            return Backend::Rocm;
        }
        if cuda_available() {
            return Backend::Cuda;
        }
    }
    Backend::Dummy
}

/// Allocate memory accessible from both CPU and GPU.
///
/// # Panics
///
/// Panics if the underlying backend allocation fails.
pub fn allocate_shared(backend: Backend, size: usize) -> UnifiedBuffer {
    if size == 0 {
        return UnifiedBuffer {
            ptr: std::ptr::null_mut(),
            size: 0,
        };
    }

    let ptr = match backend {
        Backend::Cuda => cuda_alloc_managed(size),
        Backend::Rocm => rocm_alloc_managed(size),
        Backend::LevelZero => level_zero_alloc_shared(size),
        Backend::Dummy => dummy_alloc(size),
    };

    UnifiedBuffer { ptr, size }
}

/// Release a unified allocation.
pub fn free_shared(backend: Backend, buffer: &mut UnifiedBuffer) {
    if buffer.ptr.is_null() {
        return;
    }
    match backend {
        Backend::Cuda => cuda_free(buffer.ptr),
        Backend::Rocm => rocm_free(buffer.ptr),
        Backend::LevelZero => level_zero_free(buffer.ptr),
        Backend::Dummy => dummy_free(buffer.ptr),
    }
    buffer.ptr = std::ptr::null_mut();
    buffer.size = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA FFI
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
mod cuda_ffi {
    #![allow(non_camel_case_types)]

    pub type cudaError_t = i32;
    pub const cudaSuccess: cudaError_t = 0;
    pub const cudaErrorNoDevice: cudaError_t = 100;

    extern "C" {
        pub fn cudaMallocManaged(
            devPtr: *mut *mut std::ffi::c_void,
            size: usize,
            flags: u32,
        ) -> cudaError_t;
        pub fn cudaFree(ptr: *mut std::ffi::c_void) -> cudaError_t;
        pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    }

    pub fn available() -> bool {
        unsafe {
            let mut count: i32 = 0;
            let err = cudaGetDeviceCount(&mut count);
            err == cudaSuccess && count > 0
        }
    }

    pub fn alloc_managed(size: usize) -> *mut u8 {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = cudaMallocManaged(&mut ptr, size, 1); // cudaMemAttachGlobal
            if err != cudaSuccess || ptr.is_null() {
                panic!("cudaMallocManaged({}) failed with error {}", size, err);
            }
            ptr as *mut u8
        }
    }

    pub unsafe fn free(ptr: *mut u8) {
        let err = cudaFree(ptr as *mut std::ffi::c_void);
        if err != cudaSuccess {
            panic!("cudaFree failed with error {}", err);
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
mod cuda_ffi {
    pub fn available() -> bool {
        false
    }
    pub fn alloc_managed(_size: usize) -> *mut u8 {
        panic!("CUDA not available on this platform");
    }
    pub unsafe fn free(_ptr: *mut u8) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// ROCm FFI
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
mod rocm_ffi {
    #![allow(non_camel_case_types)]

    pub type hipError_t = i32;
    pub const hipSuccess: hipError_t = 0;
    pub const hipErrorNoDevice: hipError_t = 100;

    extern "C" {
        pub fn hipMallocManaged(
            devPtr: *mut *mut std::ffi::c_void,
            size: usize,
            flags: u32,
        ) -> hipError_t;
        pub fn hipFree(ptr: *mut std::ffi::c_void) -> hipError_t;
        pub fn hipGetDeviceCount(count: *mut i32) -> hipError_t;
    }

    pub fn available() -> bool {
        unsafe {
            let mut count: i32 = 0;
            let err = hipGetDeviceCount(&mut count);
            err == hipSuccess && count > 0
        }
    }

    pub fn alloc_managed(size: usize) -> *mut u8 {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // NOTE: Requires HSA_XNACK=1 environment variable for page faults
            let err = hipMallocManaged(&mut ptr, size, 0);
            if err != hipSuccess || ptr.is_null() {
                panic!("hipMallocManaged({}) failed with error {}", size, err);
            }
            ptr as *mut u8
        }
    }

    pub unsafe fn free(ptr: *mut u8) {
        let err = hipFree(ptr as *mut std::ffi::c_void);
        if err != hipSuccess {
            panic!("hipFree failed with error {}", err);
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
mod rocm_ffi {
    pub fn available() -> bool {
        false
    }
    pub fn alloc_managed(_size: usize) -> *mut u8 {
        panic!("ROCm not available on this platform");
    }
    pub unsafe fn free(_ptr: *mut u8) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Level Zero FFI
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
mod level_zero_ffi {
    #![allow(non_camel_case_types)]

    pub type ze_result_t = i32;
    pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

    pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 5;
    pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 6;

    #[repr(C)]
    pub struct ze_device_mem_alloc_desc_t {
        pub stype: u32,
        pub pNext: *const std::ffi::c_void,
        pub flags: u32,
        pub ordinal: u32,
    }

    #[repr(C)]
    pub struct ze_host_mem_alloc_desc_t {
        pub stype: u32,
        pub pNext: *const std::ffi::c_void,
        pub flags: u32,
    }

    extern "C" {
        pub fn zeInit(flags: u32) -> ze_result_t;
        pub fn zeDriverGet(pCount: *mut u32, phDrivers: *mut *mut std::ffi::c_void) -> ze_result_t;
        pub fn zeDeviceGet(
            hDriver: *mut std::ffi::c_void,
            pCount: *mut u32,
            phDevices: *mut *mut std::ffi::c_void,
        ) -> ze_result_t;
        pub fn zeContextCreate(
            hDriver: *mut std::ffi::c_void,
            desc: *const std::ffi::c_void,
            phContext: *mut *mut std::ffi::c_void,
        ) -> ze_result_t;
        pub fn zeMemAllocShared(
            hContext: *mut std::ffi::c_void,
            device_desc: *const ze_device_mem_alloc_desc_t,
            host_desc: *const ze_host_mem_alloc_desc_t,
            size: usize,
            alignment: usize,
            hDevice: *mut std::ffi::c_void,
            pptr: *mut *mut std::ffi::c_void,
        ) -> ze_result_t;
        pub fn zeMemFree(
            hContext: *mut std::ffi::c_void,
            ptr: *mut std::ffi::c_void,
        ) -> ze_result_t;
    }

    // Cached Level Zero handles (initialized once)
    struct L0Context {
        driver: *mut std::ffi::c_void,
        device: *mut std::ffi::c_void,
        context: *mut std::ffi::c_void,
    }

    static mut L0: Option<L0Context> = None;

    pub fn ensure_init() -> &'static L0Context {
        unsafe {
            if L0.is_none() {
                let mut driver: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut device: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut context: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut driver_count: u32 = 1;
                let mut device_count: u32 = 1;

                let res = zeInit(0);
                assert_eq!(res, ZE_RESULT_SUCCESS, "zeInit failed");

                let res = zeDriverGet(&mut driver_count, &mut driver);
                assert_eq!(res, ZE_RESULT_SUCCESS, "zeDriverGet failed");

                let res = zeDeviceGet(driver, &mut device_count, &mut device);
                assert_eq!(res, ZE_RESULT_SUCCESS, "zeDeviceGet failed");

                let res = zeContextCreate(driver, std::ptr::null(), &mut context);
                assert_eq!(res, ZE_RESULT_SUCCESS, "zeContextCreate failed");

                L0 = Some(L0Context {
                    driver,
                    device,
                    context,
                });
            }
            L0.as_ref().unwrap()
        }
    }

    pub fn available() -> bool {
        unsafe {
            let res = zeInit(0);
            res == ZE_RESULT_SUCCESS
        }
    }

    pub fn alloc_shared(size: usize) -> *mut u8 {
        let ctx = ensure_init();
        let device_desc = ze_device_mem_alloc_desc_t {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            pNext: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let host_desc = ze_host_mem_alloc_desc_t {
            stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            pNext: std::ptr::null(),
            flags: 0,
        };
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let res = zeMemAllocShared(
                ctx.context,
                &device_desc,
                &host_desc,
                size,
                64,
                ctx.device,
                &mut ptr,
            );
            if res != ZE_RESULT_SUCCESS || ptr.is_null() {
                panic!("zeMemAllocShared({}) failed with error {}", size, res);
            }
            ptr as *mut u8
        }
    }

    pub unsafe fn free(ptr: *mut u8) {
        let ctx = ensure_init();
        let res = zeMemFree(ctx.context, ptr as *mut std::ffi::c_void);
        if res != ZE_RESULT_SUCCESS {
            panic!("zeMemFree failed with error {}", res);
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
mod level_zero_ffi {
    pub fn available() -> bool {
        false
    }
    pub fn alloc_shared(_size: usize) -> *mut u8 {
        panic!("Level Zero not available on this platform");
    }
    pub unsafe fn free(_ptr: *mut u8) {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Backend detection helpers
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn level_zero_available() -> bool {
    level_zero_ffi::available()
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn level_zero_available() -> bool {
    false
}

#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn rocm_available() -> bool {
    rocm_ffi::available()
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn rocm_available() -> bool {
    false
}

#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn cuda_available() -> bool {
    cuda_ffi::available()
}

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
fn cuda_available() -> bool {
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// Backend-specific allocators
// ═══════════════════════════════════════════════════════════════════════════

fn cuda_alloc_managed(size: usize) -> *mut u8 {
    cuda_ffi::alloc_managed(size)
}

fn cuda_free(ptr: *mut u8) {
    unsafe { cuda_ffi::free(ptr) }
}

fn rocm_alloc_managed(size: usize) -> *mut u8 {
    rocm_ffi::alloc_managed(size)
}

fn rocm_free(ptr: *mut u8) {
    unsafe { rocm_ffi::free(ptr) }
}

fn level_zero_alloc_shared(size: usize) -> *mut u8 {
    level_zero_ffi::alloc_shared(size)
}

fn level_zero_free(ptr: *mut u8) {
    unsafe { level_zero_ffi::free(ptr) }
}

fn dummy_alloc(size: usize) -> *mut u8 {
    let mut vec = vec![0u8; size];
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec); // Leak to get a stable pointer
    ptr
}

fn dummy_free(ptr: *mut u8) {
    // No-op: the leaked allocation persists for process lifetime.
    // A production dummy would need a global allocator registry.
    let _ = ptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_alloc_free() {
        let mut buf = allocate_shared(Backend::Dummy, 1024);
        assert!(!buf.ptr.is_null());
        assert_eq!(buf.size, 1024);
        // Write and read back
        unsafe {
            std::ptr::write_bytes(buf.ptr, 0xAB, 1024);
            assert_eq!(*buf.ptr, 0xAB);
        }
        free_shared(Backend::Dummy, &mut buf);
        assert!(buf.ptr.is_null());
        assert_eq!(buf.size, 0);
    }

    #[test]
    fn test_zero_size_allocation() {
        let mut buf = allocate_shared(Backend::Dummy, 0);
        assert!(buf.ptr.is_null());
        assert_eq!(buf.size, 0);
        free_shared(Backend::Dummy, &mut buf);
        assert!(buf.ptr.is_null());
    }

    #[test]
    fn test_detect_backend_defaults_to_dummy() {
        // On macOS or without GPU drivers, this should return Dummy
        let backend = detect_backend();
        // We don't know what's installed, just verify it doesn't panic
        let _ = allocate_shared(backend, 64);
    }
}
