// ── Accelerate vDSP helpers for nf4tile640 packing and sweep validation ──
//
// Apple Silicon vector-optimised max-abs, element-wise ops, and reductions
// used in tile packing and weight-space validation.
// Falls back to pure-Rust on other hosts or when Accelerate is unavailable.

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn vDSP_maxmgv(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    fn vDSP_maxv(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    fn vDSP_sve(a: *const f32, a_stride: i32, result: *mut f32, n: i32);
    fn vDSP_vsq(a: *const f32, a_stride: i32, b: *mut f32, b_stride: i32, n: i32);
    fn vDSP_vsdiv(a: *const f32, a_stride: i32, b: *const f32, c: *mut f32, c_stride: i32, n: i32);
    fn vDSP_vsmsa(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        c: *const f32,
        d: *mut f32,
        d_stride: i32,
        n: i32,
    );
    fn vDSP_vsub(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );
    fn vDSP_vmul(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );
    fn vDSP_distancesq(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        result: *mut f32,
        n: i32,
    );
}

// ═════════════════════════════════════════════════════════════════════════
// macOS Accelerate implementations
// ═════════════════════════════════════════════════════════════════════════

/// Maximum absolute value of an f32 slice of any length.
#[cfg(target_os = "macos")]
pub fn max_abs_slice(data: &[f32]) -> f32 {
    let n = data.len() as i32;
    if n <= 0 {
        return 0.0;
    }
    let mut result = 0.0f32;
    unsafe {
        vDSP_maxmgv(data.as_ptr(), 1, &mut result, n);
    }
    result
}

/// Maximum value (signed) of an f32 slice of any length.
#[cfg(target_os = "macos")]
pub fn max_slice(data: &[f32]) -> f32 {
    let n = data.len() as i32;
    if n <= 0 {
        return 0.0;
    }
    let mut result = 0.0f32;
    unsafe {
        vDSP_maxv(data.as_ptr(), 1, &mut result, n);
    }
    result
}

/// Sum of all elements in an f32 slice.
#[cfg(target_os = "macos")]
pub fn sum_slice(data: &[f32]) -> f32 {
    let n = data.len() as i32;
    if n <= 0 {
        return 0.0;
    }
    let mut result = 0.0f32;
    unsafe {
        vDSP_sve(data.as_ptr(), 1, &mut result, n);
    }
    result
}

/// Element-wise square: `result[i] = data[i] * data[i]`.
#[cfg(target_os = "macos")]
pub fn vsq(data: &[f32], result: &mut [f32]) {
    let n = data.len() as i32;
    if n <= 0 {
        return;
    }
    unsafe {
        vDSP_vsq(data.as_ptr(), 1, result.as_mut_ptr(), 1, n);
    }
}

/// Element-wise multiply-add: `result[i] = a[i] * scale + bias`.
/// Used for batched NF4 dequantization: `recon[i] = codebook[code[i]] * scale + bias`.
#[cfg(target_os = "macos")]
pub fn vsmsa(a: &[f32], scale: f32, bias: f32, result: &mut [f32]) {
    let n = a.len() as i32;
    if n <= 0 {
        return;
    }
    unsafe {
        vDSP_vsmsa(a.as_ptr(), 1, &scale, &bias, result.as_mut_ptr(), 1, n);
    }
}

/// Element-wise subtraction: `result[i] = a[i] - b[i]`.
#[cfg(target_os = "macos")]
pub fn vsub(a: &[f32], b: &[f32], result: &mut [f32]) {
    let n = a.len() as i32;
    if n <= 0 {
        return;
    }
    unsafe {
        vDSP_vsub(a.as_ptr(), 1, b.as_ptr(), 1, result.as_mut_ptr(), 1, n);
    }
}

/// Element-wise multiplication: `result[i] = a[i] * b[i]`.
#[cfg(target_os = "macos")]
pub fn vmul(a: &[f32], b: &[f32], result: &mut [f32]) {
    let n = a.len() as i32;
    if n <= 0 {
        return;
    }
    unsafe {
        vDSP_vmul(a.as_ptr(), 1, b.as_ptr(), 1, result.as_mut_ptr(), 1, n);
    }
}

/// Squared Euclidean distance between two vectors.
/// Returns Σ(a[i] - b[i])².
#[cfg(target_os = "macos")]
pub fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as i32;
    if n <= 0 {
        return 0.0;
    }
    let mut result = 0.0f32;
    unsafe {
        vDSP_distancesq(a.as_ptr(), 1, b.as_ptr(), 1, &mut result, n);
    }
    result
}

/// RMSE (root mean squared error) between two vectors.
#[cfg(target_os = "macos")]
pub fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sq = distance_sq(a, b) as f64;
    (sq / n as f64).sqrt()
}

/// Maximum absolute error between two vectors.
#[cfg(target_os = "macos")]
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    // Allocate diff buffer and compute element-wise difference
    let mut diff = vec![0.0f32; n];
    unsafe {
        vDSP_vsub(a.as_ptr(), 1, b.as_ptr(), 1, diff.as_mut_ptr(), 1, n as i32);
    }
    let mut result = 0.0f32;
    unsafe {
        vDSP_maxmgv(diff.as_ptr(), 1, &mut result, n as i32);
    }
    result as f64
}

/// Legacy: maximum absolute value of a 128-element f32 vector.
#[cfg(target_os = "macos")]
pub fn max_abs(data: &[f32; 128]) -> f32 {
    max_abs_slice(data)
}

/// Legacy: vector-scalar divide for 128 elements.
#[cfg(target_os = "macos")]
pub fn vsdiv(data: &[f32; 128], scalar: f32) -> [f32; 128] {
    let mut result = [0.0f32; 128];
    unsafe {
        vDSP_vsdiv(data.as_ptr(), 1, &scalar, result.as_mut_ptr(), 1, 128);
    }
    result
}

// ═════════════════════════════════════════════════════════════════════════
// Pure-Rust fallbacks (non-macOS or when Accelerate is unavailable)
// ═════════════════════════════════════════════════════════════════════════

#[cfg(not(target_os = "macos"))]
pub fn max_abs_slice(data: &[f32]) -> f32 {
    data.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

#[cfg(not(target_os = "macos"))]
pub fn max_slice(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |a, &b| a.max(b))
}

#[cfg(not(target_os = "macos"))]
pub fn sum_slice(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[cfg(not(target_os = "macos"))]
pub fn vsq(data: &[f32], result: &mut [f32]) {
    for (r, &v) in result.iter_mut().zip(data.iter()) {
        *r = v * v;
    }
}

#[cfg(not(target_os = "macos"))]
pub fn vsmsa(a: &[f32], scale: f32, bias: f32, result: &mut [f32]) {
    for (r, &v) in result.iter_mut().zip(a.iter()) {
        *r = v * scale + bias;
    }
}

#[cfg(not(target_os = "macos"))]
pub fn vsub(a: &[f32], b: &[f32], result: &mut [f32]) {
    for ((r, &av), &bv) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *r = av - bv;
    }
}

#[cfg(not(target_os = "macos"))]
pub fn vmul(a: &[f32], b: &[f32], result: &mut [f32]) {
    for ((r, &av), &bv) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *r = av * bv;
    }
}

#[cfg(not(target_os = "macos"))]
pub fn distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| (av - bv).powi(2))
        .sum()
}

#[cfg(not(target_os = "macos"))]
pub fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let d = *av as f64 - *bv as f64;
            d * d
        })
        .sum();
    (sq / n as f64).sqrt()
}

#[cfg(not(target_os = "macos"))]
pub fn max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| (*av as f64 - *bv as f64).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// Legacy: maximum absolute value of a 128-element f32 vector (fallback).
#[cfg(not(target_os = "macos"))]
pub fn max_abs(data: &[f32; 128]) -> f32 {
    max_abs_slice(data)
}

/// Legacy: vector-scalar divide for 128 elements (fallback).
#[cfg(not(target_os = "macos"))]
pub fn vsdiv(data: &[f32; 128], scalar: f32) -> [f32; 128] {
    let mut result = [0.0f32; 128];
    for (r, &v) in result.iter_mut().zip(data.iter()) {
        *r = v / scalar;
    }
    result
}
