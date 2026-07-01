//! Metal backend utilities — JIT source capture for AOT compilation.
//!
//! Provides access to MLX's Metal JIT compilation hooks for capturing
//! generated shader source and compiling it into a .metallib for AOT use.

use std::ffi::CString;
use std::path::Path;

/// Set a directory for capturing JIT-generated Metal shader source.
///
/// When set, MLX's Metal backend writes each JIT-compiled kernel source
/// string to `<path>/generated.metal` before compiling via `newLibrary`.
///
/// The captured `.metal` file can then be compiled to a `.metallib` via
/// `xcrun metal + metallib` for AOT reuse, eliminating runtime JIT
/// compilation overhead.
///
/// Pass an empty path to disable capture.
pub fn set_capture_dir(path: &Path) {
    let cpath = CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        mlx_sys::mlx_metal_set_capture_dir(cpath.as_ptr());
    }
}
