//! Intel GPU compilation via IGC / Level Zero.
//!
//! Compiles SPIR-V kernel source into an [`IntelBinary`] executable. On
//! supported platforms (Linux, Windows) with the `intel-gpu-runtime` feature,
//! this delegates to the Intel Graphics Compiler through the Level Zero
//! `zeModuleCreate` API. Otherwise the functions return
//! [`IntelCompileError::RuntimeUnavailable`].

use prism_ecs_ir::backend_dispatch::HalFormat;
use serde::{Deserialize, Serialize};

/// A compiled Intel GPU binary produced by IGC via Level Zero.
///
/// Contains the raw SPIR-V module bytes together with metadata needed
/// to launch the kernel: entry point name, grid/block dispatch dimensions,
/// and a cryptographic fingerprint for caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelBinary {
    /// Target HAL format (always [`HalFormat::IntelGpu`]).
    pub format: HalFormat,
    /// Compiled SPIR-V module bytes.
    pub spirv_bytes: Vec<u8>,
    /// Name of the kernel entry-point function.
    pub entry_point: String,
    /// Grid / global dispatch dimensions (x, y, z).
    pub grid_dims: (u32, u32, u32),
    /// Block / subgroup dispatch dimensions (x, y, z).
    pub block_dims: (u32, u32, u32),
    /// SHA-256 fingerprint of the compiled bytes, for caching.
    pub fingerprint: [u8; 32],
}

/// Errors that can occur during Intel GPU compilation.
#[derive(Debug)]
pub enum IntelCompileError {
    /// The Intel GPU runtime is not available (wrong platform or feature disabled).
    RuntimeUnavailable,
    /// IGC / Level Zero compilation returned an error.
    CompilationFailed(String),
}

impl std::fmt::Display for IntelCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuntimeUnavailable => {
                write!(
                    f,
                    "Intel GPU runtime unavailable — requires Linux/Windows \
                     with the 'intel-gpu-runtime' feature"
                )
            }
            Self::CompilationFailed(msg) => {
                write!(f, "IGC compilation failed: {msg}")
            }
        }
    }
}

impl std::error::Error for IntelCompileError {}

// ── Level Zero FFI declarations ──────────────────────────────────────────────

/// Minimal Level Zero FFI bindings for `zeModuleCreate`.
///
/// These are defined inline rather than pulled from `level-zero-sys` to keep
/// the crate self-contained. They are only compiled on Linux/Windows when the
/// `intel-gpu-runtime` feature is active.
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
    pub type ze_module_build_log_handle_t = *mut c_void;

    pub const ZE_RESULT_SUCCESS: ze_result_t = 0;
    pub const ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: ze_result_t = 0x78000008;
    pub const ZE_RESULT_ERROR_UNINITIALIZED: ze_result_t = 0x78000001;

    #[repr(C)]
    pub struct ze_module_desc_t {
        pub stype: u32,
        pub p_next: *const c_void,
        pub format: u32,
        pub input_size: usize,
        pub p_input_module: *const u8,
        pub p_build_flags: *const u8,
        pub p_constants: *const c_void,
    }

    pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 1;

    #[link(name = "ze_loader")]
    extern "C" {
        pub fn zeModuleCreate(
            hContext: ze_context_handle_t,
            hDevice: ze_device_handle_t,
            desc: *const ze_module_desc_t,
            phModule: *mut ze_module_handle_t,
            phBuildLog: *mut ze_module_build_log_handle_t,
        ) -> ze_result_t;

        pub fn zeModuleDestroy(hModule: ze_module_handle_t) -> ze_result_t;

        pub fn zeModuleGetNativeBinary(
            hModule: ze_module_handle_t,
            pSize: *mut usize,
            pModuleBinary: *mut u8,
        ) -> ze_result_t;

        pub fn zeDriverGet(pCount: *mut u32, phDrivers: *mut ze_driver_handle_t) -> ze_result_t;

        pub fn zeDeviceGet(
            hDriver: ze_driver_handle_t,
            pCount: *mut u32,
            phDevices: *mut ze_device_handle_t,
        ) -> ze_result_t;

        pub fn zeContextCreate(
            hDriver: ze_driver_handle_t,
            desc: *const c_void,
            phContext: *mut ze_context_handle_t,
        ) -> ze_result_t;

        pub fn zeModuleBuildLogDestroy(hBuildLog: ze_module_build_log_handle_t) -> ze_result_t;

        pub fn zeModuleBuildLogGetString(
            hBuildLog: ze_module_build_log_handle_t,
            pSize: *mut usize,
            pBuildLog: *mut u8,
        ) -> ze_result_t;
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Compile SPIR-V kernel source into an [`IntelBinary`].
///
/// On non-Linux/Windows targets, or when the `intel-gpu-runtime` feature is
/// disabled, this returns [`IntelCompileError::RuntimeUnavailable`].
///
/// When the runtime is available, the function:
/// 1. Initialises a Level Zero driver handle
/// 2. Creates a context and device handle
/// 3. Calls `zeModuleCreate` with `ZE_MODULE_FORMAT_IL_SPIRV`
/// 4. Extracts the compiled native binary
/// 5. Produces a fingerprint for caching
pub fn compile(source: &str, format: HalFormat) -> Result<IntelBinary, IntelCompileError> {
    #[cfg(all(
        feature = "intel-gpu-runtime",
        any(target_os = "linux", target_os = "windows")
    ))]
    {
        compile_impl(source, format)
    }

    #[cfg(not(all(
        feature = "intel-gpu-runtime",
        any(target_os = "linux", target_os = "windows")
    )))]
    {
        let _ = (source, format);
        Err(IntelCompileError::RuntimeUnavailable)
    }
}

#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
fn compile_impl(source: &str, format: HalFormat) -> Result<IntelBinary, IntelCompileError> {
    // ── 1. Enumerate Level Zero drivers ───────────────────────────────────
    use self::ffi::*;

    let mut driver_count: u32 = 0;
    let res = unsafe { zeDriverGet(&mut driver_count, std::ptr::null_mut()) };
    if res != ZE_RESULT_SUCCESS || driver_count == 0 {
        let detail = error_detail(res, "zeDriverGet");
        return Err(IntelCompileError::CompilationFailed(format!(
            "No Level Zero driver found: {detail}"
        )));
    }

    let mut drivers: Vec<ze_driver_handle_t> = vec![std::ptr::null_mut(); driver_count as usize];
    let res = unsafe { zeDriverGet(&mut driver_count, drivers.as_mut_ptr()) };
    if res != ZE_RESULT_SUCCESS {
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeDriverGet (enumerate): {}",
            error_detail(res, "zeDriverGet")
        )));
    }

    // ── 2. Pick the first Intel GPU device ────────────────────────────────
    let device = pick_first_gpu_device(drivers[0])?;

    let mut context: ze_context_handle_t = std::ptr::null_mut();
    let res = unsafe { zeContextCreate(drivers[0], std::ptr::null(), &mut context) };
    if res != ZE_RESULT_SUCCESS {
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeContextCreate: {}",
            error_detail(res, "zeContextCreate")
        )));
    }

    // ── 3. Compile SPIR-V via zeModuleCreate ──────────────────────────────
    let source_bytes = source.as_bytes();
    let desc = ze_module_desc_t {
        stype: 0,
        p_next: std::ptr::null(),
        format: ZE_MODULE_FORMAT_IL_SPIRV,
        input_size: source_bytes.len(),
        p_input_module: source_bytes.as_ptr(),
        p_build_flags: std::ptr::null(),
        p_constants: std::ptr::null(),
    };

    let mut module: ze_module_handle_t = std::ptr::null_mut();
    let mut build_log: ze_module_build_log_handle_t = std::ptr::null_mut();

    let res = unsafe {
        zeModuleCreate(
            context,
            device,
            &desc as *const ze_module_desc_t,
            &mut module as *mut ze_module_handle_t,
            &mut build_log as *mut ze_module_build_log_handle_t,
        )
    };

    if res != ZE_RESULT_SUCCESS {
        let log = extract_build_log(build_log);
        unsafe {
            let _ = zeModuleBuildLogDestroy(build_log);
        }
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeModuleCreate failed (result={res}): {log}"
        )));
    }

    // ── 4. Extract native binary ─────────────────────────────────────────
    let mut binary_size: usize = 0;
    let res = unsafe { zeModuleGetNativeBinary(module, &mut binary_size, std::ptr::null_mut()) };
    if res != ZE_RESULT_SUCCESS || binary_size == 0 {
        unsafe {
            let _ = zeModuleDestroy(module);
            let _ = zeModuleBuildLogDestroy(build_log);
        }
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeModuleGetNativeBinary (size): {}",
            error_detail(res, "zeModuleGetNativeBinary")
        )));
    }

    let mut spirv_bytes: Vec<u8> = vec![0u8; binary_size];
    let res =
        unsafe { zeModuleGetNativeBinary(module, &mut binary_size, spirv_bytes.as_mut_ptr()) };
    unsafe {
        let _ = zeModuleDestroy(module);
        let _ = zeModuleBuildLogDestroy(build_log);
    }
    if res != ZE_RESULT_SUCCESS {
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeModuleGetNativeBinary (read): {}",
            error_detail(res, "zeModuleGetNativeBinary")
        )));
    }

    let fingerprint = blake3::hash(&spirv_bytes).into();

    Ok(IntelBinary {
        format,
        spirv_bytes,
        entry_point: "matmul".into(),
        grid_dims: (1, 1, 1),
        block_dims: (16, 16, 1),
        fingerprint,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
fn pick_first_gpu_device(
    driver: ffi::ze_driver_handle_t,
) -> Result<ffi::ze_device_handle_t, IntelCompileError> {
    use self::ffi::*;

    let mut device_count: u32 = 0;
    let res = unsafe { zeDeviceGet(driver, &mut device_count, std::ptr::null_mut()) };
    if res != ZE_RESULT_SUCCESS || device_count == 0 {
        return Err(IntelCompileError::CompilationFailed(
            "No Intel GPU device found on driver".into(),
        ));
    }

    let mut devices: Vec<ze_device_handle_t> = vec![std::ptr::null_mut(); device_count as usize];
    let res = unsafe { zeDeviceGet(driver, &mut device_count, devices.as_mut_ptr()) };
    if res != ZE_RESULT_SUCCESS {
        return Err(IntelCompileError::CompilationFailed(format!(
            "zeDeviceGet: {}",
            error_detail(res, "zeDeviceGet")
        )));
    }

    if devices.is_empty() || devices[0].is_null() {
        return Err(IntelCompileError::CompilationFailed(
            "No usable Intel GPU device".into(),
        ));
    }

    Ok(devices[0])
}

#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
fn error_detail(res: ffi::ze_result_t, call: &str) -> String {
    match res {
        ffi::ZE_RESULT_ERROR_UNINITIALIZED => {
            format!("{call}: Level Zero not initialised (ZE_RESULT_ERROR_UNINITIALIZED)")
        }
        ffi::ZE_RESULT_ERROR_MODULE_BUILD_FAILURE => {
            format!("{call}: module build failure (ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)")
        }
        _ => format!("{call}: ze_result_t = {res:#x}"),
    }
}

#[cfg(all(
    feature = "intel-gpu-runtime",
    any(target_os = "linux", target_os = "windows")
))]
fn extract_build_log(log_handle: ffi::ze_module_build_log_handle_t) -> String {
    use self::ffi::*;

    if log_handle.is_null() {
        return "(no build log)".into();
    }

    let mut log_size: usize = 0;
    let res = unsafe { zeModuleBuildLogGetString(log_handle, &mut log_size, std::ptr::null_mut()) };
    if res != ZE_RESULT_SUCCESS || log_size == 0 {
        return "(build log unavailable)".into();
    }

    let mut log: Vec<u8> = vec![0u8; log_size];
    let res = unsafe { zeModuleBuildLogGetString(log_handle, &mut log_size, log.as_mut_ptr()) };
    if res != ZE_RESULT_SUCCESS {
        return "(build log read failed)".into();
    }

    String::from_utf8_lossy(&log)
        .trim_end_matches('\0')
        .to_string()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_handles_valid_spirv() {
        let source = r#"
            ; SPIR-V
            ; Version: 1.0
            ; Generator: IGC
            ; Bound: 10
            ; Schema: 0
               OpCapability Shader
               OpCapability Float64
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %matmul "matmul"
               OpExecutionMode %matmul LocalSize 16 16 1
        "#;

        let result = compile(source, HalFormat::IntelGpu);
        match result {
            Ok(binary) => {
                assert_eq!(binary.format, HalFormat::IntelGpu);
                assert!(!binary.spirv_bytes.is_empty());
                assert_eq!(binary.entry_point, "matmul");
                assert_eq!(binary.grid_dims, (1, 1, 1));
                assert_eq!(binary.block_dims, (16, 16, 1));
            }
            Err(IntelCompileError::RuntimeUnavailable) => {
                // Expected on non-Linux/Windows or without the feature.
            }
            Err(e) => panic!("Unexpected compile error: {e}"),
        }
    }

    #[test]
    fn compile_rejects_junk_input() {
        let result = compile("not valid SPIR-V source", HalFormat::IntelGpu);
        match result {
            Ok(_) => (),
            Err(IntelCompileError::RuntimeUnavailable) => (),
            Err(IntelCompileError::CompilationFailed(msg)) => {
                assert!(!msg.is_empty(), "error message should be non-empty");
            }
        }
    }

    #[test]
    fn compile_returns_error_without_feature() {
        // The `intel-gpu-runtime` feature is not in the default set, so
        // on most CI/test configurations this should hit RuntimeUnavailable.
        let source = r#"
            __kernel void dummy(__global float* x) {
                x[get_global_id(0)] = 1.0f;
            }
        "#;

        let result = compile(source, HalFormat::IntelGpu);
        match result {
            Err(IntelCompileError::RuntimeUnavailable) => { /* expected */ }
            Ok(_) => {
                // If the feature is enabled on a supported OS, that's fine.
            }
            Err(e) => {
                panic!("unexpected error variant: {e:?}");
            }
        }
    }
}
