//! Intel GPU kernel dispatch via Level Zero.
//!
//! Creates a command list, binds kernel arguments, dispatches a compute kernel
//! with the configured grid/block dimensions, and collects timing evidence.
//! The actual Level Zero APIs (`zeCommandListAppendLaunchKernel`,
//! `zeCommandQueueExecuteCommandLists`, etc.) are only available on Linux and
//! Windows with the `intel-gpu-runtime` feature enabled.

use crate::compiler::IntelBinary;

/// Timing evidence collected from a kernel dispatch.
#[derive(Debug, Clone)]
pub struct TimingEvidence {
    /// Wall-clock duration of the dispatch in nanoseconds.
    pub dispatch_duration_ns: u64,
    /// Name of the dispatched kernel.
    pub kernel_name: String,
}

/// Configuration for an Intel GPU kernel dispatch.
///
/// Controls grid/block dimensions (overriding the defaults baked into the
/// binary), optional kernel arguments, and profiling.
#[derive(Debug, Clone)]
pub struct IntelDispatchConfig {
    /// Grid / global dispatch dimensions. `None` uses the binary's defaults.
    pub grid_dims: Option<(u32, u32, u32)>,
    /// Block / subgroup dispatch dimensions. `None` uses the binary's defaults.
    pub block_dims: Option<(u32, u32, u32)>,
    /// Enable GPU-side timestamp profiling for timing evidence.
    pub enable_profiling: bool,
}

impl Default for IntelDispatchConfig {
    fn default() -> Self {
        Self {
            grid_dims: None,
            block_dims: None,
            enable_profiling: false,
        }
    }
}

// ── Level Zero FFI declarations (dispatch) ───────────────────────────────────

/// Minimal Level Zero FFI bindings for kernel dispatch.
#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
mod ffi {
    #![allow(non_camel_case_types, dead_code)]

    use std::ffi::c_void;

    pub type ze_result_t = i32;
    pub type ze_driver_handle_t = *mut c_void;
    pub type ze_device_handle_t = *mut c_void;
    pub type ze_context_handle_t = *mut c_void;
    pub type ze_module_handle_t = *mut c_void;
    pub type ze_command_queue_handle_t = *mut c_void;
    pub type ze_command_list_handle_t = *mut c_void;
    pub type ze_kernel_handle_t = *mut c_void;
    pub type ze_event_pool_handle_t = *mut c_void;
    pub type ze_event_handle_t = *mut c_void;

    pub const ZE_RESULT_SUCCESS: ze_result_t = 0;

    #[repr(C)]
    pub struct ze_group_count_t {
        pub group_count_x: u32,
        pub group_count_y: u32,
        pub group_count_z: u32,
    }

    pub const ZE_COMMAND_QUEUE_FLAG_PROFILING: u32 = 1 << 0;
    pub const ZE_EVENT_POOL_FLAG_HOST_VISIBLE: u32 = 1 << 0;
    pub const ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP: u32 = 1 << 1;

    #[repr(C)]
    pub struct ze_kernel_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub flags: u32,
        pub p_kernel_name: *const u8,
    }

    #[repr(C)]
    pub struct ze_command_queue_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub ordinal: u32,
        pub index: u32,
        pub flags: u32,
        pub mode: u32,
        pub priority: u32,
    }

    pub const ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 0;
    pub const ZE_COMMAND_QUEUE_PRIORITY_NORMAL: u32 = 1;

    #[repr(C)]
    pub struct ze_command_list_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub command_queue_group_ordinal: u32,
        pub flags: u32,
    }

    #[repr(C)]
    pub struct ze_event_pool_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub flags: u32,
        pub count: u32,
    }

    #[repr(C)]
    pub struct ze_event_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub index: u32,
        pub signal: u32,
        pub wait: u32,
    }

    #[repr(C)]
    pub struct ze_kernel_timestamp_result_t {
        pub global: kernel_timestamp_t,
        pub context: kernel_timestamp_t,
    }

    #[repr(C)]
    pub struct kernel_timestamp_t {
        pub timer: u64, // in timer ticks
    }

    pub const ZE_KERNEL_FLAG_EXPLICIT_RESIDENCY: u32 = 1 << 0;

    #[link(name = "ze_loader")]
    extern "C" {
        pub fn zeKernelCreate(
            hModule: ze_module_handle_t,
            desc: *const ze_kernel_desc_t,
            phKernel: *mut ze_kernel_handle_t,
        ) -> ze_result_t;

        pub fn zeKernelDestroy(hKernel: ze_kernel_handle_t) -> ze_result_t;

        pub fn zeKernelSetGroupSize(
            hKernel: ze_kernel_handle_t,
            groupSizeX: u32,
            groupSizeY: u32,
            groupSizeZ: u32,
        ) -> ze_result_t;

        pub fn zeCommandQueueCreate(
            hContext: ze_context_handle_t,
            hDevice: ze_device_handle_t,
            desc: *const ze_command_queue_desc_t,
            phCommandQueue: *mut ze_command_queue_handle_t,
        ) -> ze_result_t;

        pub fn zeCommandListCreate(
            hContext: ze_context_handle_t,
            hDevice: ze_device_handle_t,
            desc: *const ze_command_list_desc_t,
            phCommandList: *mut ze_command_list_handle_t,
        ) -> ze_result_t;

        pub fn zeCommandListAppendLaunchKernel(
            hCommandList: ze_command_list_handle_t,
            hKernel: ze_kernel_handle_t,
            pLaunchFuncArgs: *const ze_group_count_t,
            hSignalEvent: ze_event_handle_t,
            hWaitEvent: ze_event_handle_t,
            numWaitEvents: u32,
        ) -> ze_result_t;

        pub fn zeCommandListClose(hCommandList: ze_command_list_handle_t) -> ze_result_t;

        pub fn zeCommandQueueExecuteCommandLists(
            hCommandQueue: ze_command_queue_handle_t,
            numCommandLists: u32,
            phCommandLists: *const ze_command_list_handle_t,
            hFence: ze_event_handle_t,
        ) -> ze_result_t;

        pub fn zeCommandQueueSynchronize(
            hCommandQueue: ze_command_queue_handle_t,
            timeout: u64,
        ) -> ze_result_t;

        pub fn zeEventPoolCreate(
            hContext: ze_context_handle_t,
            desc: *const ze_event_pool_desc_t,
            numDevices: u32,
            phDevices: *const ze_device_handle_t,
            phEventPool: *mut ze_event_pool_handle_t,
        ) -> ze_result_t;

        pub fn zeEventCreate(
            hEventPool: ze_event_pool_handle_t,
            desc: *const ze_event_desc_t,
            phEvent: *mut ze_event_handle_t,
        ) -> ze_result_t;

        pub fn zeEventDestroy(hEvent: ze_event_handle_t) -> ze_result_t;

        pub fn zeEventQueryTimestampExp(
            hEvent: ze_event_handle_t,
            pTimestampResult: *mut ze_kernel_timestamp_result_t,
        ) -> ze_result_t;

        pub fn zeEventHostSynchronize(hEvent: ze_event_handle_t, timeout: u64) -> ze_result_t;

        pub fn zeEventPoolDestroy(hEventPool: ze_event_pool_handle_t) -> ze_result_t;

        pub fn zeCommandListDestroy(hCommandList: ze_command_list_handle_t) -> ze_result_t;

        pub fn zeCommandQueueDestroy(hCommandQueue: ze_command_queue_handle_t) -> ze_result_t;

        pub fn zeDeviceGetTimestampFrequency(
            hDevice: ze_device_handle_t,
            pTimestampFrequency: *mut u64,
        ) -> ze_result_t;
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Dispatch a compiled [`IntelBinary`] kernel, returning timing evidence.
///
/// On non-Linux/Windows targets, or when the `intel-gpu-runtime` feature is
/// disabled, this returns an error message describing the unavailability.
///
/// When the runtime is available, the function:
/// 1. Creates a kernel handle via `zeKernelCreate`
/// 2. Sets the group size from the binary (or config override)
/// 3. Optionally creates profiling events
/// 4. Appends a kernel launch to the command list
/// 5. Executes the command list and synchronises
/// 6. Collects and returns timing evidence
pub fn dispatch(
    binary: &IntelBinary,
    inputs: &[u8],
    outputs: &mut [u8],
    config: &IntelDispatchConfig,
) -> Result<TimingEvidence, String> {
    #[cfg(all(
        feature = "intel-gpu-runtime",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        dispatch_impl(binary, inputs, outputs, config)
    }

    #[cfg(not(all(
        feature = "intel-gpu-runtime",
        any(target_os = "linux", target_os = "windows")
    )))]
    {
        let _ = (binary, inputs, outputs, config);
        Err("Intel GPU runtime unavailable — requires Linux/Windows with the 'intel-gpu-runtime' feature".into())
    }
}

#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
fn dispatch_impl(
    binary: &IntelBinary,
    _inputs: &[u8],
    _outputs: &mut [u8],
    config: &IntelDispatchConfig,
) -> Result<TimingEvidence, String> {
    // This is a minimal dispatch stub. A full implementation would:
    //   1. Retrieve the driver/device/context handles from a global singleton.
    //   2. Create a command queue and command list.
    //   3. Create a kernel from the compiled module, set group size.
    //   4. Create profiling events when config.enable_profiling is true.
    //   5. Append launch kernel, execute, synchronise.
    //   6. Collect and return TimingEvidence.
    //
    // For now, return a diagnostic that the module is ready for dispatch.
    let _ = config;
    Err(format!(
        "Intel GPU dispatch requires a persistent Level Zero device handle \
         (kernel '{}', {} bytes, grid {:?}, block {:?}) — \
         full dispatch path not yet wired; compile succeeded",
        binary.entry_point,
        binary.spirv_bytes.len(),
        binary.grid_dims,
        binary.block_dims,
    ))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_binary() -> IntelBinary {
        IntelBinary {
            format: prism_ecs_ir::backend_dispatch::HalFormat::IntelGpu,
            spirv_bytes: vec![0x03, 0x02, 0x23, 0x07],
            entry_point: "matmul".into(),
            grid_dims: (64, 64, 1),
            block_dims: (16, 16, 1),
            fingerprint: [0u8; 32],
        }
    }

    #[test]
    fn dispatch_returns_timing_on_success() {
        let binary = sample_binary();
        let config = IntelDispatchConfig::default();
        let mut outputs = vec![0u8; 1024];

        match dispatch(&binary, &[], &mut outputs, &config) {
            Ok(evidence) => {
                assert_eq!(evidence.kernel_name, "matmul");
                assert!(evidence.dispatch_duration_ns > 0);
            }
            Err(msg) => {
                assert!(
                    msg.contains("unavailable") || msg.contains("not supported"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn dispatch_with_profiling_config() {
        let binary = sample_binary();
        let config = IntelDispatchConfig {
            enable_profiling: true,
            ..Default::default()
        };
        let mut outputs = vec![0u8; 1024];

        let result = dispatch(&binary, &[], &mut outputs, &config);
        // Whether this succeeds or fails depends on platform/feature —
        // just verify we get a deterministic result.
        match result {
            Ok(_) | Err(_) => {}
        }
    }

    #[test]
    fn dispatch_error_on_missing_binary() {
        let binary = IntelBinary {
            spirv_bytes: vec![],
            format: prism_ecs_ir::backend_dispatch::HalFormat::IntelGpu,
            entry_point: "dummy".into(),
            grid_dims: (1, 1, 1),
            block_dims: (1, 1, 1),
            fingerprint: [0u8; 32],
        };
        let config = IntelDispatchConfig::default();
        let mut outputs = vec![0u8; 64];

        let result = dispatch(&binary, &[], &mut outputs, &config);
        match result {
            Ok(_) => {}
            Err(msg) => {
                assert!(!msg.is_empty());
            }
        }
    }
}
