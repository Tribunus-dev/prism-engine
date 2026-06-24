//! External storage abstractions for zero-copy array data access.
//!
//! Provides the [`ExternalStorage`] trait for types that expose a raw
//! data pointer, enabling cross-backend buffer sharing without copies.

use std::ffi::c_void;

/// A type that owns or wraps an externally-allocated memory buffer
/// whose raw pointer can be exposed for zero-copy access.
///
/// # Safety
///
/// The pointer returned by [`data_ptr`](ExternalStorage::data_ptr) must be
/// valid, non-null, and remain valid for the lifetime of the implementing
/// object.  The implementor is responsible for ensuring the memory is
/// properly aligned and sized.
pub trait ExternalStorage {
    /// Return a raw pointer to the start of the buffer.
    fn data_ptr(&self) -> *const c_void;
}

/// Expose the raw pointer from an [`ExternalStorage`] implementor.
pub fn external_storage_ptr(storage: &dyn ExternalStorage) -> *const u8 {
    storage.data_ptr() as *const u8
}
