//! Level Zero backend utilities — JIT source capture for AOT compilation.
//!
//! Provides access to MLX's Intel Level Zero / ocloc JIT compilation hooks
//! for capturing generated OpenCL C source and compiled SPIR-V for AOT use.

use std::ffi::CString;
use std::path::Path;

/// Set a directory for capturing JIT-generated OpenCL C source and SPIR-V.
///
/// When set, MLX's Level Zero backend writes each JIT-compiled OpenCL C
/// source string to `<path>/generated.cl` and the compiled SPIR-V binary
/// to `<path>/model.spv`.
///
/// The captured `.spv` file can then be loaded via `zeModuleCreate` for AOT
/// reuse, eliminating the ocloc compilation step at runtime.
///
/// Pass an empty path to disable capture.
pub fn set_capture_dir(path: &Path) {
    let cpath = CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        mlx_sys::mlx_level_zero_set_capture_dir(cpath.as_ptr());
    }
}
