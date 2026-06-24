//! Manual FFI bindings for Tribunus-extended MLX C API.
//!
//! These functions are declared in `mlx/c/array.h` but may not be picked
//! up by bindgen unless a full rebuild regenerates the bindings.
//! Providing them here ensures the Rust code can always link against them.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::c_void;

extern "C" {
    /// Evaluate the array and write the result into externally-owned memory.
    /// `ptr` must point to at least `array.nbytes()` bytes.
    /// Returns 0 on success, non-zero on error.
    pub fn mlx_array_evaluate_into(
        array: super::mlx_array,
        ptr: *mut c_void,
        byte_size: usize,
    ) -> i32;

    /// Expose the raw data pointer of an evaluated MLX array.
    /// Returns a pointer to the start of the data, or NULL if not evaluated.
    pub fn mlx_array_data_ptr(array: super::mlx_array) -> *mut c_void;

    /// Set an external memory allocator for MLX Metal backend.
    /// Pass NULL for both to restore the default allocator.
    pub fn mlx_set_metal_allocator(
        alloc_fn: Option<
            unsafe extern "C" fn(usize) -> *mut c_void,
        >,
        free_fn: Option<
            unsafe extern "C" fn(*mut c_void, usize),
        >,
    ) -> i32;
}
