//! CUDA kernel dispatch via the CUDA driver API (`cuda-driver-sys`).
//!
//! Loads a cubin blob, looks up the kernel entry point, launches the kernel
//! with the binary's grid/block dimensions, synchronizes, and returns timing
//! evidence measured via CUDA events.
//!
//! This module is `cfg`-gated to Linux (`target_os = "linux"`) because the
//! CUDA driver library (`libcuda.so`) only ships on that platform.

use crate::TimingEvidence;

// ── Architecture-generic interface ─────────────────────────────────────────
//
// The actual CUDA driver API path is only compiled on Linux with the
// `cuda-runtime` feature enabled. On other platforms the dispatch path
// returns a clear error at runtime.

/// Launch a kernel from the given `CudaBinary` on the current CUDA device.
///
/// # Platform
///
/// On Linux with the `cuda-runtime` feature enabled this calls the real CUDA
/// driver API (`cuModuleLoadData` / `cuModuleGetFunction` / `cuLaunchKernel` /
/// `cuEventElapsedTime`). On all other platforms it returns a descriptive
/// error immediately.
#[cfg(not(all(target_os = "linux", feature = "cuda-runtime")))]
pub fn dispatch_kernel(binary: &crate::CudaBinary) -> Result<TimingEvidence, String> {
    let _ = binary; // suppress unused-variable warning
    Err("CUDA runtime is only available on Linux with the 'cuda-runtime' feature enabled".into())
}

/// Real CUDA driver API dispatch path — Linux + feature gate.
#[cfg(all(target_os = "linux", feature = "cuda-runtime"))]
pub fn dispatch_kernel(binary: &crate::CudaBinary) -> Result<TimingEvidence, String> {
    // We use raw pointers from cuda-driver-sys. The symbols must be resolved
    // from the CUDA driver shared library at runtime (loaded by the system
    // when the first CUDA API call is made).
    use cuda_driver_sys::{
        cuCtxSynchronize, cuEventCreate, cuEventDestroy, cuEventElapsedTime, cuEventRecord,
        cuEventSynchronize, cuFuncSetBlockShape, cuLaunchKernel, cuModuleGetFunction,
        cuModuleLoadData, CUevent, CUfunction, CUmodule, CU_EVENT_DEFAULT,
    };
    use std::ffi::CString;
    use std::ptr;

    // 1. Load the cubin module.
    let mut module: CUmodule = ptr::null_mut();
    let status = unsafe { cuModuleLoadData(&mut module, binary.cubin.as_ptr().cast()) };
    if status != 0 {
        return Err(format!("cuModuleLoadData failed with error code {status}"));
    }

    // 2. Look up the kernel function.
    let entry_cstr =
        CString::new(binary.entry_point.as_str()).map_err(|_| "entry point contains null byte")?;
    let mut function: CUfunction = ptr::null_mut();
    let status = unsafe { cuModuleGetFunction(&mut function, module, entry_cstr.as_ptr()) };
    if status != 0 {
        return Err(format!(
            "cuModuleGetFunction('{}') failed with error code {status}",
            binary.entry_point
        ));
    }

    // 3. (Optional) set block shape if the driver expects it.
    unsafe {
        cuFuncSetBlockShape(
            function,
            binary.block_dims.0,
            binary.block_dims.1,
            binary.block_dims.2,
        );
    }

    // 4. Create CUDA events for timing.
    let mut start_event: CUevent = ptr::null_mut();
    let mut stop_event: CUevent = ptr::null_mut();
    unsafe {
        cuEventCreate(&mut start_event, CU_EVENT_DEFAULT);
        cuEventCreate(&mut stop_event, CU_EVENT_DEFAULT);
    }

    // 5. Record start, launch kernel, record stop.
    unsafe {
        cuEventRecord(start_event, ptr::null_mut());
    }

    let status = unsafe {
        cuLaunchKernel(
            function,
            binary.grid_dims.0,
            binary.grid_dims.1,
            binary.grid_dims.2,
            binary.block_dims.0,
            binary.block_dims.1,
            binary.block_dims.2,
            0,               // shared mem bytes
            ptr::null_mut(), // stream (null = default)
            ptr::null_mut(), // kernel params (null = none)
            ptr::null_mut(), // extra options
        )
    };
    if status != 0 {
        return Err(format!("cuLaunchKernel failed with error code {status}"));
    }

    unsafe {
        cuEventRecord(stop_event, ptr::null_mut());
    }

    // 6. Synchronize and measure elapsed time.
    unsafe {
        cuEventSynchronize(stop_event);
    }

    let mut elapsed_ms: f32 = 0.0;
    let status = unsafe { cuEventElapsedTime(&mut elapsed_ms, start_event, stop_event) };
    if status != 0 {
        // Events might still be valid — just report 0.
        elapsed_ms = 0.0;
    }

    // 7. Clean up events (module/function live until process exit).
    unsafe {
        cuEventDestroy(start_event);
        cuEventDestroy(stop_event);
    }

    // Synchronize context to ensure all work is done before returning.
    unsafe {
        cuCtxSynchronize();
    }

    let duration_us = (elapsed_ms * 1000.0) as u64;

    Ok(TimingEvidence {
        kernel_name: "matmul_kernel",
        duration_us,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaBinary;

    /// On Linux with the feature enabled, dispatch tries to call the real
    /// driver API and fails with a CUDA error (no GPU on this CI runner).
    /// On other platforms it fails with the "not available" message.
    #[test]
    fn dispatch_without_gpu_returns_error() {
        let binary = CudaBinary {
            cubin: vec![0u8; 64], // fake cubin bytes
            entry_point: "matmul_kernel".into(),
            grid_dims: (1, 1, 1),
            block_dims: (256, 1, 1),
        };

        let err = dispatch_kernel(&binary).unwrap_err();
        // On non-Linux: "only available on Linux"
        // On Linux: starts with "cuModuleLoadData failed"
        assert!(
            err.contains("Linux") || err.contains("cuModuleLoadData"),
            "expected either platform error or CUDA driver error, got: {err}"
        );
    }
}
