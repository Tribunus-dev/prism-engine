//! C-compatible FFI bridge for the Prism Engine Swift menu bar app.
//!
//! Exports `extern "C"` functions that the Swift side links against.
//! All functions are stubs that return error codes or null pointers.
//! The real implementation lives in the Swift app layer.

use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};

/// C-compatible multimodal input payload.
#[repr(C)]
pub struct MultimodalPayload {
    pub text_prompt: *const c_char,
    pub image_surface_id: u32,
    pub audio_surface_id: u32,
}

/// Opaque handle to the runtime multiplexer state.
/// Swift holds this as `OpaquePointer?`.
#[repr(C)]
pub struct OpaqueMultiplexer {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, std::marker::PhantomPinned)>,
}

/// Compile a .cimage from downloaded safetensors + bundled resources.
/// Returns -2 when compute-core is not linked.
#[no_mangle]
pub unsafe extern "C" fn prism_compile_and_pack(
    _safetensors_dir: *const c_char,
    _output_cimage_path: *const c_char,
    _resource_dir: *const c_char,
) -> c_int {
    -2
}

/// Initialize the runtime multiplexer from a compiled .cimage.
/// Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn prism_engine_init(_cimage_path: *const c_char) -> *mut OpaqueMultiplexer {
    std::ptr::null_mut()
}

/// Free a previously initialized OpaqueMultiplexer.
#[no_mangle]
pub unsafe extern "C" fn prism_engine_free(_multiplexer: *mut OpaqueMultiplexer) {}

/// Extended multimodal execution with priority and lane pinning.
#[no_mangle]
pub unsafe extern "C" fn prism_execute_multimodal_ex(
    _multiplexer: *mut OpaqueMultiplexer,
    _agent_id: u32,
    _payload: MultimodalPayload,
    _priority: u32,
    _lane_hint: u32,
) {
}

/// Return the number of discovered compute devices.
#[no_mangle]
pub extern "C" fn prism_device_count() -> u32 {
    0
}

/// Fill a PrismDeviceInfo struct for device at `index`.
/// Returns -1 if index out of range.
#[no_mangle]
pub unsafe extern "C" fn prism_device_info(_index: u32, _info: *mut c_void) -> c_int {
    -1
}

#[no_mangle]
pub unsafe extern "C" fn prism_device_info_free_name(_name: *mut c_char) {}

#[no_mangle]
pub unsafe extern "C" fn prism_device_info_free_vendor(_vendor: *mut c_char) {}

#[no_mangle]
pub extern "C" fn prism_device_list_json() -> *mut c_char {
    std::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn prism_free_json_string(_s: *mut c_char) {}

#[no_mangle]
pub extern "C" fn prism_load_config(_config: *mut c_void) {}
