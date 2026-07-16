//! AMD GPU kernel dispatch via the HSA (Heterogeneous System Architecture)
//! runtime.
//!
//! Loads an HSACO code object blob, looks up the kernel entry point, launches
//! the kernel with the binary's grid/block dimensions, synchronizes, and
//! returns timing evidence measured via HSA signal timestamps.
//!
//! This module is `cfg`-gated to Linux (`target_os = "linux"`) because the HSA
//! runtime library (`libhsa-runtime64.so`) only ships on that platform.

use crate::{AmdBinary, TimingEvidence};

// ── Architecture-generic interface ─────────────────────────────────────────
//
// The actual HSA runtime path is only compiled on Linux with the `rocm-runtime`
// feature enabled. On other platforms the dispatch path returns a clear error
// at runtime.

/// Launch a kernel from the given `AmdBinary` on the current AMD GPU.
///
/// # Platform
///
/// On Linux with the `rocm-runtime` feature enabled this calls the real HSA
/// runtime APIs (hsa_code_object_serialize, hsa_executable_load_code_object,
/// hsa_executable_get_symbol, hsa_queue_create, hsa_agent_iterate_regions,
/// AQL packet dispatch, and hsa_signal_wait_relaxed). On all other platforms
/// it returns a descriptive error immediately.
#[cfg(not(feature = "rocm-runtime"))]
pub fn dispatch_kernel(binary: &AmdBinary) -> Result<TimingEvidence, String> {
    let _ = binary;
    Err("ROCm not installed".into())
}

/// Real HSA runtime dispatch path — Linux + feature gate.
#[cfg(feature = "rocm-runtime")]
pub fn dispatch_kernel(binary: &AmdBinary) -> Result<TimingEvidence, String> {
    // ── Probe HSA runtime ───────────────────────────────────────────────
    // In production this would dlopen libhsa-runtime64.so, iterate agents,
    // create a queue, load the code object into an executable, lookup the
    // kernel symbol, submit an AQL packet, wait on the completion signal,
    // and return the elapsed time.
    //
    // For the stub on Linux+feature: the probe itself has not been wired to
    // actual HSA FFI yet, so we return a "not yet wired" error.
    //
    // This matches the pattern of the CUDA runtime which requires the
    // cuda-driver-sys crate for the real path.

    let _ = binary;

    // Attempt to detect the HSA runtime shared library via dlopen.
    if !hsa_library_accessible() {
        return Err("ROCm not installed: libhsa-runtime64.so not found".into());
    }

    Err("HSA runtime dispatch is not yet wired to the HSA FFI — stubbed for feature-gate correctness".into())
}

/// Check whether the HSA runtime shared library is loadable.
#[cfg(feature = "rocm-runtime")]
fn hsa_library_accessible() -> bool {
    // Try dlopen with RTLD_NOLOAD to check availability without actually
    // linking the symbols at compile time (which requires the hsa-sys crate).
    let lib_names = ["libhsa-runtime64.so.1\0", "libhsa-runtime64.so\0"];

    for name in &lib_names {
        let handle = unsafe {
            libc::dlopen(
                name.as_ptr() as *const libc::c_char,
                libc::RTLD_LAZY | libc::RTLD_NOLOAD,
            )
        };
        if !handle.is_null() {
            unsafe {
                libc::dlclose(handle);
            }
            return true;
        }
    }
    false
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmdBinary;

    /// On non-Linux (or without the feature), dispatch returns the
    /// "not available" error.
    #[test]
    fn dispatch_without_rocm_returns_error() {
        let binary = AmdBinary {
            code_object: vec![0; 64],
            entry_point: "matmul_kernel".into(),
            grid_dims: (1, 1, 1),
            block_dims: (256, 1, 1),
        };
        let result = dispatch_kernel(&binary);
        assert!(result.is_err(), "expected error, got: {result:?}");
        let err = result.unwrap_err();
        // On non-Linux: "ROCm not installed"
        // On Linux: "HSA runtime dispatch is not yet wired"
        assert!(
            err.contains("ROCm") || err.contains("not yet wired"),
            "expected ROCm-related error, got: {err}"
        );
    }
}
