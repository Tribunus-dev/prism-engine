//! CImage error types.
//!
//! Defines the structured error enum for all cimage operations: writing,
//! loading, validating, and numerical comparison.

use thiserror::Error;

/// Errors that can occur during cimage operations.
#[derive(Debug, Clone, Error)]
pub enum CImageError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("invalid magic: expected {expected:?} got {got:?}")]
    InvalidMagic { expected: [u8; 8], got: Vec<u8> },

    #[error("unsupported format version: {0}")]
    UnsupportedFormatVersion(u32),

    #[error("offset {offset} + len {len} exceeds file size {file_size}")]
    RangeOutOfBounds {
        offset: u64,
        len: u64,
        file_size: u64,
    },

    #[error("digest mismatch for {section}: expected {expected} got {actual}")]
    DigestMismatch {
        section: String,
        expected: String,
        actual: String,
    },

    #[error("unresolved payload ref: {0}")]
    UnresolvedPayloadRef(String),

    #[error("unresolved tensor ref: {0}")]
    UnresolvedTensorRef(String),

    #[error("unresolved receipt ref: {0}")]
    UnresolvedReceiptRef(String),

    #[error("codec mismatch: tensor {tensor} uses {codec} but payload directory declares {payload_codec}")]
    CodecMismatch {
        tensor: String,
        codec: String,
        payload_codec: String,
    },

    #[error("mixed codec without precision plan for tensor {0}")]
    MixedCodecWithoutPrecisionPlan(String),

    #[error("non-mixed tensor {0} carries mixed payload ref")]
    NonMixedWithMixedPayloadRef(String),

    #[error("json serialization error: {0}")]
    JsonSerialize(String),

    #[error("json deserialization error: {0}")]
    JsonDeserialize(String),

    #[error("sha256 error: {0}")]
    Sha256(String),

    #[error("physical layout invalid for codec {codec}: {detail}")]
    InvalidLayout { codec: String, detail: String },

    #[error("shape mismatch: {detail}")]
    ShapeMismatch { detail: String },

    #[error("alignment error: {detail}")]
    Alignment { detail: String },

    #[error("numerical validation failed: {detail}")]
    NumericalValidation { detail: String },

    #[error("unknown codec family: {0}")]
    UnknownCodecFamily(String),

    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for CImageError {
    fn from(e: std::io::Error) -> Self {
        CImageError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for CImageError {
    fn from(e: serde_json::Error) -> Self {
        CImageError::JsonSerialize(e.to_string())
    }
}

/// Convenience result alias for cimage operations.
pub type CImageResult<T> = Result<T, CImageError>;
