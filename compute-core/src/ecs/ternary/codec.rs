//! Ternary codec data types — packed tensor metadata and error types.
//!
//! Ternary codes use 2 bits per weight:
//! - `00` = -1
//! - `01` = 0
//! - `10` = +1
//! - `11` = reserved / invalid

use half::f16;
use thiserror::Error;

/// Errors that can occur during ternary codec operations.
#[derive(Debug, Clone, Error)]
pub enum TernaryCodecError {
    #[error("invalid ternary weight value: {0} (expected -1, 0, or +1)")]
    InvalidWeight(i8),
    #[error("reserved code 11 encountered in packed ternary data")]
    ReservedCode11,
    #[error("length mismatch: expected {expected} values, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("packing error: {0}")]
    PackingError(String),
}

/// A tensor packed in the Ternary1_58 codec.
///
/// Weights are packed 4 values per byte (2 bits each: 00=-1, 01=0, 10=+1).
/// Scales are stored as f16, one per group of `group_size` weights.
#[derive(Debug, Clone)]
pub struct TernaryPackedTensor {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub groups_per_row: usize,
    pub bytes_per_group: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f16>,
}
