// ── Accelerate vDSP helpers for nf4tile640 packing ──────────────────────────
//
// Apple Silicon vector-optimised max-abs and scalar-divide operations used in
// the tile packer inner loop.  Falls back to pure-Rust on other hosts.
//
// Each operates on pinned 128-element arrays (GROUP_SIZE), so the FFI call
// overhead is amortised over the vector width of the M-series NEU.

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// vDSP_maxmgv — maximum absolute value across vector.
    fn vDSP_maxmgv(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    /// vDSP_vsdiv — element-wise division by scalar: C[i] = A[i] / B
    fn vDSP_vsdiv(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );
}

// ── Accelerate vDSP (macOS) ─────────────────────────────────────────────────

/// Maximum absolute value (|max|) of a 128-element f32 vector.
#[cfg(target_os = "macos")]
pub fn max_abs(data: &[f32; 128]) -> f32 {
    let mut result = 0.0f32;
    unsafe {
        vDSP_maxmgv(data.as_ptr(), 1, &mut result, 128);
    }
    result
}

/// Vector-scalar divide: `result[i] = data[i] / scalar` for 128 elements.
#[cfg(target_os = "macos")]
pub fn vsdiv(data: &[f32; 128], scalar: f32) -> [f32; 128] {
    let mut result = [0.0f32; 128];
    unsafe {
        vDSP_vsdiv(data.as_ptr(), 1, &scalar, result.as_mut_ptr(), 1, 128);
    }
    result
}

// ── Pure-Rust fallbacks ────────────────────────────────────────────────────

/// Maximum absolute value (|max|) of a 128-element f32 vector.
#[cfg(not(target_os = "macos"))]
pub fn max_abs(data: &[f32; 128]) -> f32 {
    data.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

/// Vector-scalar divide.
#[cfg(not(target_os = "macos"))]
pub fn vsdiv(data: &[f32; 128], scalar: f32) -> [f32; 128] {
    let mut result = [0.0f32; 128];
    for (r, &v) in result.iter_mut().zip(data.iter()) {
        *r = v / scalar;
    }
    result
}
