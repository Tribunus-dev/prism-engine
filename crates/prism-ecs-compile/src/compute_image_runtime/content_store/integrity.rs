//! Integrity verifier — pure data types for content integrity.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;

/// Integrity verifier that compares expected and actual content hashes.
#[derive(Debug, Clone, Default)]
pub struct IntegrityVerifier;

impl IntegrityVerifier {
    /// Create a new integrity verifier.
    pub fn new() -> Self {
        Self
    }

    /// Verify that the actual hash matches the expected hash.
    pub fn verify(&self, expected: ContentHash, actual: ContentHash) -> Result<(), String> {
        if expected == actual {
            Ok(())
        } else {
            Err(format!(
                "integrity mismatch: expected {} got {}",
                expected, actual
            ))
        }
    }
}

/// An integrity check record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheck {
    /// Object id checked.
    pub object_id: String,
    /// Expected hash.
    pub expected_hash: ContentHash,
    /// Actual hash observed.
    pub actual_hash: ContentHash,
    /// Whether the check passed.
    pub passed: bool,
}
