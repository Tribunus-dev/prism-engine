//! Canonical JSON serialization helpers for deterministic cimage output.
//!
//! Wraps `serde_json_canonicalizer` for V0 proof format stability.

use serde::Serialize;

use crate::cimage_v0::error::{CImageError, CImageResult};

/// Serialize `value` to canonical (deterministic) JSON bytes.
///
/// The output is:
/// - Deterministic: same input always produces same bytes
/// - Unicode-escaped for portability
/// - Supports any `Serialize` type
pub fn canonical_json_bytes<T: Serialize>(value: &T) -> CImageResult<Vec<u8>> {
    serde_json_canonicalizer::to_vec(value).map_err(|e| CImageError::JsonSerialize(e.to_string()))
}

/// Serialize `value` to a canonical JSON string.
pub fn canonical_json_string<T: Serialize>(value: &T) -> CImageResult<String> {
    let bytes = canonical_json_bytes(value)?;
    String::from_utf8(bytes)
        .map_err(|e| CImageError::Other(format!("canonical json is not valid utf-8: {e}")))
}
