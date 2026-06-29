//! C-compatible FFI bridge for the Prism Engine Swift menu bar app.
//!
//! Exports `extern "C"` functions that the Swift side links against.
//! When compiled with the `compute-core` feature, delegates to the real
//! implementation in `tribunus_compute_core::ffi`. Otherwise returns
//! error codes / null pointers as stubs.

use std::os::raw::{c_char, c_int};

/// Opaque handle to the runtime multiplexer state.
/// Swift holds this as `OpaquePointer?`.
#[repr(C)]
pub struct OpaqueMultiplexer {
    _private: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Compile a .cimage from downloaded safetensors + bundled resources.
/// Returns 0 on success, negative on error (or -2 when compute-core is not linked).
#[no_mangle]
pub unsafe extern "C" fn prism_compile_and_pack(
    safetensors_dir: *const c_char,
    output_cimage_path: *const c_char,
    resource_dir: *const c_char,
) -> c_int {
    #[cfg(feature = "compute-core")]
    {
        tribunus_compute_core::ffi::prism_compile_and_pack(
            safetensors_dir,
            output_cimage_path,
            resource_dir,
        )
    }
    #[cfg(not(feature = "compute-core"))]
    {
        let _ = (safetensors_dir, output_cimage_path, resource_dir);
        -2
    }
}

/// Initialize the runtime multiplexer from a compiled .cimage.
/// Returns a pointer to an OpaqueMultiplexer, or null on failure.
#[no_mangle]
pub unsafe extern "C" fn prism_runtime_init(
    cimage_path: *const c_char,
) -> *mut OpaqueMultiplexer {
    #[cfg(feature = "compute-core")]
    {
        tribunus_compute_core::ffi::prism_runtime_init(cimage_path)
            as *mut OpaqueMultiplexer
    }
    #[cfg(not(feature = "compute-core"))]
    {
        let _ = cimage_path;
        std::ptr::null_mut()
    }
}

/// Free a previously initialized OpaqueMultiplexer.
#[no_mangle]
pub unsafe extern "C" fn prism_runtime_free(multiplexer: *mut OpaqueMultiplexer) {
    #[cfg(feature = "compute-core")]
    {
        tribunus_compute_core::ffi::prism_runtime_free(
            multiplexer as *mut tribunus_compute_core::ffi::OpaqueMultiplexer,
        );
    }
    #[cfg(not(feature = "compute-core"))]
    {
        let _ = multiplexer;
    }
}
