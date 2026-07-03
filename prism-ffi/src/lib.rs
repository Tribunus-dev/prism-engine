//! C-compatible FFI bridge for the Prism Engine Swift menu bar app.
//!
//! Exports `extern "C"` functions that the Swift side links against.
//! When compiled with the `compute-core` feature, delegates to the real
//! implementation in `tribunus_compute_core::ffi`. Otherwise returns
//! error codes / null pointers as stubs.

use std::os::raw::{c_char, c_int, c_void};
/// C-compatible multimodal input payload.
#[cfg(not(feature = "compute-core"))]
#[cfg(not(feature = "compute-core"))]
#[repr(C)]
pub struct MultimodalPayload {
    pub text_prompt: *const c_char,
    pub image_surface_id: u32,
    pub audio_surface_id: u32,
}

/// C-compatible multimodal input payload (alias to compute-core type).
#[cfg(feature = "compute-core")]
pub type MultimodalPayload = tribunus_compute_core::ffi::MultimodalPayload;

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

/// Extended multimodal execution with priority and lane pinning.
#[no_mangle]
pub unsafe extern "C" fn prism_execute_multimodal_ex(
    multiplexer: *mut OpaqueMultiplexer,
    agent_id: u32,
    payload: MultimodalPayload,
    priority: u32,
    lane_hint: u32,
) {
    #[cfg(feature = "compute-core")]
    {
        tribunus_compute_core::ffi::prism_execute_multimodal_ex(
            multiplexer as *mut tribunus_compute_core::ffi::OpaqueMultiplexer,
            agent_id,
            payload,
            priority,
            lane_hint,
        );
    }
    #[cfg(not(feature = "compute-core"))]
    {
        let _ = (multiplexer, agent_id, payload, priority, lane_hint);
    }
}

/// Return the number of discovered compute devices.
#[no_mangle]
pub extern "C" fn prism_device_count() -> u32 {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_device_count() }
    #[cfg(not(feature = "compute-core"))]
    { 0 }
}

/// Fill a PrismDeviceInfo struct for device at `index`.
/// Returns 0 on success, -1 if index out of range.
#[no_mangle]
pub unsafe extern "C" fn prism_device_info(index: u32, info: *mut c_void) -> c_int {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_device_info(index, info as *mut tribunus_compute_core::ffi::PrismDeviceInfo) }
    #[cfg(not(feature = "compute-core"))]
    { let _ = (index, info); -1 }
}

#[no_mangle]
pub unsafe extern "C" fn prism_device_info_free_name(name: *mut c_char) {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_device_info_free_name(name) }
    #[cfg(not(feature = "compute-core"))]
    { let _ = name; }
}

#[no_mangle]
pub unsafe extern "C" fn prism_device_info_free_vendor(vendor: *mut c_char) {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_device_info_free_vendor(vendor) }
    #[cfg(not(feature = "compute-core"))]
    { let _ = vendor; }
}

#[no_mangle]
pub extern "C" fn prism_device_list_json() -> *mut c_char {
    #[cfg(feature = "compute-core")]
    { unsafe { tribunus_compute_core::ffi::prism_device_list_json() } }
    #[cfg(not(feature = "compute-core"))]
    { std::ptr::null_mut() }
}

#[no_mangle]
pub unsafe extern "C" fn prism_free_json_string(s: *mut c_char) {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_free_json_string(s) }
    #[cfg(not(feature = "compute-core"))]
    { let _ = s; }
}

#[no_mangle]
pub extern "C" fn prism_load_config(config: *mut c_void) {
    #[cfg(feature = "compute-core")]
    { tribunus_compute_core::ffi::prism_load_config(config as *mut tribunus_compute_core::ffi::PrismServerConfig) }
    #[cfg(not(feature = "compute-core"))]
    { let _ = config; }
}
