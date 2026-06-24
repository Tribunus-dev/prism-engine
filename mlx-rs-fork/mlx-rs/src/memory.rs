//! Tribunus — memory management extensions for MLX.
//!
//! Provides the [`OutputBufferHint`] trait for zero-copy materialization of
//! evaluation results into pre-allocated buffers (e.g. IOSurface arenas).

use std::ffi::c_void;

/// A pre-allocated buffer that can serve as the output target for MLX
/// evaluation.  Implementors provide a stable pointer and byte size that
/// the Metal allocator wraps as an `MTLBuffer` instead of allocating fresh
/// Metal heap memory.
///
/// # Safety
///
/// The pointer returned by [`buffer_ptr`](OutputBufferHint::buffer_ptr) must
/// be valid, non-null, and remain valid for the duration of the MLX
/// `evaluate_into` call and any subsequent GPU reads that depend on the
/// result.  The memory must be aligned to the backend's requirements
/// (typically page-aligned for Metal shared storage on Apple Silicon).
pub trait OutputBufferHint {
    /// Raw pointer to the start of the pre-allocated buffer.
    fn buffer_ptr(&self) -> *const c_void;

    /// Total usable size of the buffer in bytes.
    fn buffer_size(&self) -> usize;
}
