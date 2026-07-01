// Level Zero stub symbols.
//
// When MLX is built with MLX_BUILD_LEVEL_ZERO=OFF, the level_zero.cpp
// C++ source file is not compiled and its symbols are absent from the
// linked library.  We provide no-op Rust implementations here so that
// mlx-rs::level_zero can reference them without a linker error.
// These functions are never called in practice; if they are called they
// return 0 (success with no effect).

#[no_mangle]
pub unsafe extern "C" fn mlx_level_zero_set_capture_dir(
    _path: *const std::os::raw::c_char,
) -> i32 {
    0
}
