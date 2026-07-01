//! Raw FFI bindings to the NPU C API implemented in npu_dispatch.cpp
//!
//! Each function dispatches to the correct vendor backend (Apple ANE,
//! Intel VPU, AMD XDNA) based on the TargetNpu enum.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TargetNpu {
    AppleAne = 0,
    IntelVpu = 1,
    AmdXdna = 2,
}

#[repr(C)]
pub struct NpuBuffer {
    pub ptr: *mut c_void,
    pub size: usize,
    pub vendor_handle: *mut c_void,
}

extern "C" {
    /// Load a compiled graph onto the target NPU.
    /// - Apple ANE: path to .mlmodelc directory
    /// - Intel VPU: path to OpenVINO IR .xml file
    /// - AMD XDNA: "path/to/xclbin.xclbin:kernel_name"
    pub fn npu_load_graph(target: TargetNpu, blob_path: *const c_char) -> *mut c_void;

    /// Submit an NPU execution with single input/output buffers.
    /// Returns a monotonically increasing submission ID, or 0 on failure.
    pub fn npu_submit_execution(
        target: TargetNpu,
        session: *mut c_void,
        input_buf: *mut c_void,
        output_buf: *mut c_void,
        input_bytes: usize,
        output_bytes: usize,
    ) -> u64;

    /// Non-blocking poll. Returns 1 if complete, 0 if still running.
    pub fn npu_poll_completion(
        target: TargetNpu,
        session: *mut c_void,
        submission_id: u64,
    ) -> i32;

    /// Release a loaded graph session.
    pub fn npu_destroy_session(target: TargetNpu, session: *mut c_void);
}

/// Safe wrapper: load a compiled graph onto the NPU.
pub fn load_graph_safe(target: TargetNpu, blob_path: &str) -> Result<*mut c_void, String> {
    let cpath = CString::new(blob_path).map_err(|e| format!("invalid path: {e}"))?;
    let session = unsafe { npu_load_graph(target, cpath.as_ptr()) };
    if session.is_null() {
        return Err(format!(
            "npu_load_graph({:?}, {}) returned null",
            target, blob_path
        ));
    }
    Ok(session)
}
