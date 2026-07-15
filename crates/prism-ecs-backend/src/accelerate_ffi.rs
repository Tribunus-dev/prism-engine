//! Direct FFI bindings to Accelerate BLAS (cblas_sgemm) and vForce / vDSP vector math.
//!
//! Accelerate is a system framework on all Apple platforms.
//! #[link] attribute is sufficient — no third-party crate required.

// ── vDSP FFT types ───────────────────────────────────────────────────────────

/// Split-complex data structure for vDSP FFT operations.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DSPSplitComplex {
    pub realp: *mut f32,
    pub imagp: *mut f32,
}

/// Opaque FFT setup object created by `vDSP_create_fftsetup`.
pub type FFTSetup = *mut std::ffi::c_void;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// Single-precision general matrix multiply: C = alpha * op(A) * op(B) + beta * C.
    /// cblas_sgemm uses column-major storage by default.
    ///
    /// Parameters:
    ///   Order: CblasRowMajor (101) or CblasColMajor (102)
    ///   TransA, TransB: CblasNoTrans (111) or CblasTrans (112)
    ///   M: rows of op(A) and C
    ///   N: cols of op(B) and C
    ///   K: cols of op(A) / rows of op(B)
    ///   alpha: scalar multiplier
    ///   A: matrix A
    ///   lda: leading dimension of A
    ///   B: matrix B
    ///   ldb: leading dimension of B
    ///   beta: scalar multiplier for C
    ///   C: result matrix
    ///   ldc: leading dimension of C
    pub fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    /// Vector reciprocal square root: y[i] = 1/sqrt(x[i]).
    pub fn vvrsqrtf(y: *mut f32, x: *const f32, n: *const i32);

    /// Vector exponential: y[i] = exp(x[i]).
    pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);

    /// Vector natural log: y[i] = log(x[i]).
    pub fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);

    /// Element-wise addition: C[i] = A[i] + B[i]
    pub fn vDSP_vadd(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// Element-wise multiplication: C[i] = A[i] * B[i]
    pub fn vDSP_vmul(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// Vector-scalar add: C[i] = A[i] + B
    pub fn vDSP_vsadd(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// Element-wise maximum: C[i] = A[i] if A[i] > B[i] else B[i]
    pub fn vDSP_vmax(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// Element-wise division: C[i] = B[i] / A[i]
    pub fn vDSP_vdiv(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// Vector sum: result = sum(A[i]) over i in [0, N)
    pub fn vDSP_sve(a: *const f32, a_stride: i32, result: *mut f32, n: i32);

    /// Vector multiply with scalar add: D[i] = A[i] * B[i] + C[i]
    pub fn vDSP_vsma(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        b_stride: i32,
        c: *const f32,
        c_stride: i32,
        d: *mut f32,
        d_stride: i32,
        n: i32,
    );

    /// Vector-scalar multiply: C[i] = A[i] * B (B is pointer to a single float)
    pub fn vDSP_vsmul(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );
    /// Vector-scalar divide: C[i] = A[i] / B
    pub fn vDSP_vsdiv(
        a: *const f32,
        a_stride: i32,
        b: *const f32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    /// In-place matrix transpose: C = A^T (M rows, N cols -> N rows, M cols)
    pub fn vDSP_mtrans(a: *const f32, a_stride: i32, c: *mut f32, c_stride: i32, m: i32, n: i32);

    /// Vector gather: C[i] = A[B[i]] for i in [0, N)
    pub fn vDSP_vgathr(
        a: *const f32,
        b: *const i32,
        b_stride: i32,
        c: *mut f32,
        c_stride: i32,
        n: i32,
    );

    // ── vDSP FFT ────────────────────────────────────────────────────────────

    /// Create an FFT setup object. log2n = log2 of FFT size.
    /// Returns null on failure.
    pub fn vDSP_create_fftsetup(log2n: u32) -> FFTSetup;

    /// Destroy an FFT setup object.
    pub fn vDSP_destroy_fftsetup(setup: FFTSetup);

    /// In-place real FFT.
    pub fn vDSP_fft_zrip(
        setup: FFTSetup,
        io_data: *mut DSPSplitComplex,
        stride: i32,
        log2n: u32,
        direction: i32,
    );

    /// Magnitude squared for split-complex data.
    pub fn vDSP_zvmags(a: *const DSPSplitComplex, stride: i32, c: *mut f32, c_stride: i32, n: u32);

    /// Generate a Hann window.
    pub fn vDSP_hann_window(c: *mut f32, n: u32, flag: i32);
}

// BLAS constants
pub const CBLAS_ROW_MAJOR: i32 = 101;
pub const CBLAS_NO_TRANS: i32 = 111;
pub const CBLAS_TRANS: i32 = 112;

// ── vDSP FFT constants ────────────────────────────────────────────────────

/// Forward FFT direction.
pub const FFT_FORWARD: i32 = 1;
/// Inverse FFT direction.
pub const FFT_INVERSE: i32 = -1;
